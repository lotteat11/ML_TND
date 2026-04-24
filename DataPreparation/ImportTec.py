# %%
import os
import re
import requests
from tqdm import trange # Import tqdm for a nice progress bar
# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust IONEX downloader + parser for CODE GIM (and other IGS centers).

Features:
- Earthdata-authenticated HTTPS downloads from NASA CDDIS with response validation.
- Auto-detect compression (LZW .Z vs. GZIP .gz served as .Z) by magic bytes.
- Parse IONEX (.i) files into a tidy DataFrame: epoch, center, latitude, longitude, tec_value.
- Handles both .Z and .i in the input folder; filters by requested years; saves CSV/Parquet.
- Prints coverage sanity (years present, counts by year, newest timestamps).

References:
- CDDIS Archive Access (Earthdata Login, HTTPS, .netrc examples): https://www.earthdata.nasa.gov/centers/cddis-daac/archive-access
- Magic bytes: gzip (1F 8B), UNIX compress .Z (1F 9D): http://fileformats.archiveteam.org/wiki/GZIP
"""

import os
import re
import time
import gzip
import netrc
import requests
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# tqdm is optional
try:
    from tqdm import trange
except Exception:
    def trange(a, b=None, **kwargs):
        if b is None:
            b = a
            a = 0
        return range(a, b)

# LZW decompressor
try:
    from unlzw3 import unlzw
except ImportError:
    from unlzw import unlzw  # fallback if environment exposes it as 'unlzw'

# ----------------------------
# ======= Configuration =======
# ----------------------------
START_YEAR = 2009
END_YEAR   = 2017
START_DOY  = 1
END_DOY    = 365

CENTER   = "codg"  # other common: jplg, upcg, esag
BASE_URL = "https://cddis.nasa.gov/archive/gnss/products/ionex"
OUT_DIR  = "ionex_files_0917_v4"   # download folder (also read from here)

# Output files (dynamic names keep runs clear)
SAVE_CSV     = f"tec_{CENTER}_{START_YEAR}-{END_YEAR}_doy{START_DOY}-{END_DOY}_v2.csv"
SAVE_PARQUET = f"tec_{CENTER}_{START_YEAR}-{END_YEAR}_doy{START_DOY}-{END_DOY}_v2.parquet"

# Download retry/backoff
MAX_RETRIES = 3
BACKOFF_S   = 1.5

# ----------------------------
# ======= HTTP Session ========
# ----------------------------
URS = "urs.earthdata.nasa.gov"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AAU-IONEX-Downloader/1.0 (+research use)"})
SESSION.max_redirects = 10

def attach_netrc_auth(session: requests.Session) -> bool:
    """Attach Earthdata credentials from ~/.netrc if present."""
    try:
        nrc = netrc.netrc()
        auth = nrc.authenticators(URS)
        if auth:
            login, _, password = auth
            session.auth = (login, password)
            return True
    except FileNotFoundError:
        pass
    return False

def is_html(content_type: str, blob: bytes) -> bool:
    """Heuristic: detect HTML/redirect page content."""
    if content_type and "text/html" in content_type.lower():
        return True
    head = blob[:64].lstrip()
    return head.startswith(b"<!") or head.lower().startswith(b"<html")

def magic_kind(blob: bytes) -> str:
    """Return 'lzw' for .Z (1F 9D), 'gz' for .gz (1F 8B), else 'other'."""
    if len(blob) >= 2:
        if blob[0:2] == b"\x1f\x9d":
            return "lzw"
        if blob[0:2] == b"\x1f\x8b":
            return "gz"
    return "other"

# ---------------------------------
# ======= Download one file ========
# ---------------------------------
def download_ionex_file(year: int, doy: int, center: str, out_dir: str) -> str | None:
    """
    Download (with auth & validation) the .i.Z file for (year, doy, center) into out_dir.
    Returns the full path to the saved file, or None if not saved.
    """
    yy   = str(year)[2:]  # e.g. '17'
    name = f"{center}{doy:03d}0.{yy}i.Z"
    url  = f"{BASE_URL}/{year}/{doy:03d}/{name}"
    out_path = os.path.join(out_dir, name)

    # Skip if already present and non-zero
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    # Ensure session has URS creds if available
    if not hasattr(SESSION, "_auth_ready"):
        SESSION._auth_ready = attach_netrc_auth(SESSION)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, timeout=60, allow_redirects=True)
            ct   = resp.headers.get("Content-Type", "")
            if resp.status_code != 200:
                print(f"Missing/HTTP {resp.status_code}: {name}")
            else:
                blob = resp.content
                # Reject HTML (auth page) & tiny payloads
                if is_html(ct, blob) or len(blob) < 2000:
                    print(f"AUTH/HTML or too small: {name} (CT={ct}, size={len(blob)})")
                else:
                    k = magic_kind(blob)
                    if k in ("lzw", "gz") or blob.startswith(b"     1.0"):  # rare uncompressed IONEX header
                        with open(out_path, "wb") as f:
                            f.write(blob)
                        return out_path
                    else:
                        print(f"Bad magic bytes for {name}: {blob[:8].hex()} (CT={ct}, size={len(blob)})")
        except Exception as e:
            print(f"Error {name} (attempt {attempt}/{MAX_RETRIES}): {e}")
        time.sleep(BACKOFF_S * attempt)

    print(f"FAIL (not saved): {name}")
    return None

# -------------------------------------------------
# ======= Decompress: LZW .Z or GZIP-as-.Z ========
# -------------------------------------------------
def decompress_to_i_auto(path_Z: str) -> str:
    """
    Decompress an IONEX file whose name ends with .Z but might actually be:
      - LZW (.Z)  -> use unlzw
      - GZIP (.gz)-> use gzip
      - HTML/error-> raise with helpful diagnostics
    Returns the path to the .i file.
    """
    out_path = path_Z[:-2]  # strip ".Z" -> expected ".i"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    with open(path_Z, 'rb') as fin:
        head = fin.read(4)
        fin.seek(0)
        if head.startswith(b'<!') or head.lower().startswith(b'<htm'):
            txt = fin.read(200).decode('utf-8', errors='ignore')
            raise RuntimeError(f"{os.path.basename(path_Z)} looks like HTML (auth/redirect). Snippet: {txt[:80]}...")
        if head[:2] == b'\x1f\x8b':  # gzip
            with gzip.open(fin, 'rb') as gz, open(out_path, 'wb') as fout:
                fout.write(gz.read())
            return out_path
        if head[:2] == b'\x1f\x9d':  # UNIX compress (LZW)
            raw = unlzw(fin.read())
            with open(out_path, 'wb') as fout:
                fout.write(raw)
            return out_path
        raise RuntimeError(f"Unrecognized magic bytes {head.hex()} in {os.path.basename(path_Z)}")

# ------------------------------------------------
# ======= IONEX (.i) parser to DataFrame =========
# ------------------------------------------------
# Regex for merged "LAT/LON1" tokens occasionally seen in headers/rows
MERGED_LAT_LON1_RE = re.compile(r'([\-]?\d+\.?\d*)\s*([\-]\d+\.?\d*)')
epoch_re = re.compile(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.Ee+-]+)')

def parse_ionex_file(path_i: str, center_hint: str | None = None) -> pd.DataFrame:
    """
    Parse a decompressed IONEX (.i) file into a DataFrame with columns:
    epoch (UTC), center, latitude, longitude, tec_value (TECU).
    """
    with open(path_i, 'r', errors='ignore') as f:
        lines = f.readlines()

    # ---- Header: grid + exponent ----
    lat1 = lat2 = dlat = None
    lon1 = lon2 = dlon = None
    tec_exp = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'LAT1 / LAT2 / DLAT' in line:
            lat1, lat2, dlat = map(float, line[:60].split()[:3])
        elif 'LON1 / LON2 / DLON' in line:
            lon1, lon2, dlon = map(float, line[:60].split()[:3])
        elif 'EXPONENT' in line:
            tec_exp = int(line[:6])
        elif 'END OF HEADER' in line:
            i += 1
            break
        i += 1

    if None in (lat1, lat2, dlat, lon1, lon2, dlon):
        raise RuntimeError(f"Header parse failed in {path_i}")

    lat_vals = np.arange(lat1, lat2 + 0.5 * dlat, dlat)
    lon_vals = np.arange(lon1, lon2 + 0.5 * dlon, dlon)

    rows = []

    # ---- TEC maps ----
    while i < len(lines):
        if 'START OF TEC MAP' in lines[i]:
            # Find epoch
            j = i
            epoch = None
            while j < len(lines):
                if 'EPOCH OF CURRENT MAP' in lines[j]:
                    m = epoch_re.match(lines[j][:60])
                    if not m:
                        raise RuntimeError(f"Malformed EPOCH line in {path_i}")
                    yyyy, mm, dd, hh, minute, sec = m.groups()
                    epoch = datetime(int(yyyy), int(mm), int(dd), int(hh), int(minute), int(float(sec)))
                    break
                j += 1
            if epoch is None:
                raise RuntimeError(f"Epoch not found in {path_i}")

            tec = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype=float)

            # Walk latitude slices
            k = j + 1
            while k < len(lines):
                if 'LAT/LON1/LON2/DLON' in lines[k]:
                    # Tokenize grid header
                    data_fields_raw = lines[k][:40].split()
                    data_fields = data_fields_raw

                    # Fix merged first token if present (LAT and LON1 stuck together)
                    if len(data_fields_raw) == 4 and data_fields_raw[0].count('-') >= 1:
                        m = MERGED_LAT_LON1_RE.match(data_fields_raw[0])
                        if m:
                            lat_str, lon1_str = m.groups()
                            data_fields = [lat_str, lon1_str] + data_fields_raw[1:]

                    data_fields = [f.strip() for f in data_fields if f.strip()]
                    if len(data_fields) < 4:
                        # Skip malformed row defensively
                        k += 1
                        continue

                    lat_row   = float(data_fields[0])
                    lon_start = float(data_fields[1])
                    lon_stop  = float(data_fields[2])
                    dlon_row  = float(data_fields[3])

                    nvals = int(round((lon_stop - lon_start) / dlon_row)) + 1

                    # Collect 5-char TEC integers across wrapped lines
                    digits = ''
                    kk = k + 1
                    while len(digits) < 5 * nvals and kk < len(lines):
                        if ('LAT/LON1/LON2/DLON' in lines[kk]) or ('END OF TEC MAP' in lines[kk]):
                            break
                        digits += lines[kk][:80].rstrip()
                        kk += 1
                    while len(digits) < 5 * nvals and kk < len(lines):
                        if ('LAT/LON1/LON2/DLON' in lines[kk]) or ('END OF TEC MAP' in lines[kk]):
                            break
                        digits += lines[kk][:80].rstrip()
                        kk += 1
                    if len(digits) < 5 * nvals:
                        raise RuntimeError(f"Not enough TEC digits for a latitude row in {path_i}")

                    digits = digits[:5 * nvals]
                    raw_vals = [int(digits[m*5:(m+1)*5]) for m in range(nvals)]
                    vals = np.array(raw_vals, dtype=float)

                    # Map IONEX missing markers to NaN
                    vals[(vals == 9999) | (vals == -1)] = np.nan
                    vals = vals * (10.0 ** tec_exp)

                    lat_idx = int(round((lat_row - lat1) / dlat))
                    lon_idx_start = int(round((lon_start - lon1) / dlon))
                    tec[lat_idx, lon_idx_start:lon_idx_start+nvals] = vals

                    k = kk

                elif 'END OF TEC MAP' in lines[k]:
                    # Emit rows for this epoch
                    LAT, LON = np.meshgrid(lat_vals, lon_vals, indexing='ij')
                    df_map = pd.DataFrame({
                        'epoch': epoch,
                        'center': center_hint if center_hint else os.path.basename(path_i)[:4].lower(),
                        'latitude': LAT.ravel(),
                        'longitude': LON.ravel(),
                        'tec_value': tec.ravel()
                    })
                    rows.append(df_map)
                    i = k + 1
                    break
                else:
                    k += 1
        else:
            i += 1

    if not rows:
        return pd.DataFrame(columns=['epoch','center','latitude','longitude','tec_value'])
    return pd.concat(rows, ignore_index=True)

# -------------------------------------------------------
# ======= Drive: download, decompress, parse, save ======
# -------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Years {START_YEAR}–{END_YEAR}, DOY {START_DOY}–{END_DOY}, center {CENTER}")
    print(f"Download base: {BASE_URL}")
    print(f"Output folder: {OUT_DIR}")

    # 1) Download loop with auth + validation
    for year in range(START_YEAR, END_YEAR + 1):
        for doy in trange(START_DOY, END_DOY + 1, desc=f"Downloading {CENTER} {year}"):
            saved = download_ionex_file(year, doy, CENTER, OUT_DIR)
            # (saved can be None if not found/failed; continue)

    # 2) Parse both .Z and .i files
    pat_name = re.compile(rf"^{CENTER}\d{{3}}0\.\d{{2}}i(\.Z)?$", re.IGNORECASE)
    all_dfs = []

    for fname in sorted(os.listdir(OUT_DIR)):
        if not pat_name.match(fname):
            continue
        fpath = os.path.join(OUT_DIR, fname)
        fl = fname.lower()

        try:
            if fl.endswith(".z"):
                path_i = decompress_to_i_auto(fpath)
            elif fl.endswith(".i"):
                path_i = fpath
            else:
                continue

            center_hint = fname[:4].lower()
            df_one = parse_ionex_file(path_i, center_hint=center_hint)
            if not df_one.empty:
                all_dfs.append(df_one)
                print(f"OK: {os.path.basename(path_i)} -> {len(df_one):,} rows")
            else:
                print(f"EMPTY: {os.path.basename(path_i)}")

        except Exception as e:
            print(f"FAIL: {fname} -> {e}")

    if not all_dfs:
        print("No data parsed.")
        return

    # 3) Concatenate and filter to requested years
    df = pd.concat(all_dfs, ignore_index=True)
    df['epoch'] = pd.to_datetime(df['epoch'], utc=True, errors='coerce')

    # Keep only requested span as safety net
    df = df[(df['epoch'].dt.year >= START_YEAR) & (df['epoch'].dt.year <= END_YEAR)]

    # 4) Sanity checks
    years = sorted(df['epoch'].dt.year.unique())
    print("Years present in df:", years)
    print("Row counts by year:\n", df.groupby(df['epoch'].dt.year).size())

    print("Newest timestamps:")
    print(df.sort_values('epoch').tail(5)[['epoch','center','latitude','longitude','tec_value']])

    # 5) Save outputs
    if SAVE_CSV:
        df.to_csv(SAVE_CSV, index=False)
        print(f"Saved CSV -> {SAVE_CSV}")
    if SAVE_PARQUET:
        df.to_parquet(SAVE_PARQUET, index=False)
        print(f"Saved Parquet -> {SAVE_PARQUET}")

if __name__ == "__main__":
    main()


# %%
d
