# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
GettingData.py
- Downloads GRACE or Swarm DNS files from the TU Delft FTP server.
- Parses the ASCII text format and stacks all years into one dataframe.
- Saves the result as a single parquet file.
"""

import re
import zipfile
from ftplib import FTP
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

FTP_HOST   = "thermosphere.tudelft.nl"
FTP_PATH   = "/version_02/Swarm_data"   # change to /version_02/GRACE_data for GRACE

MISSION    = "Swarm"                    # "Swarm" | "GRACE"
YEARS      = (2015, 2016)               # inclusive range
OUTDIR     = Path("SWARM_201516_v02")
PARQUET_OUT = "swarm_dns_2015_2016.parquet"

# ---------------------------------------------------------------------------
# READERS
# ---------------------------------------------------------------------------

def _find_first_data_line(lines: list[str]) -> int:
    """Return the index of the first non-comment data line (YYYY-MM-DD ...)."""
    pat = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")
    for i, line in enumerate(lines):
        s = line.strip()
        if s and not s.startswith("#") and pat.match(s):
            return i
    raise ValueError("Could not locate first data line.")


def read_swarm_dns_txt(path: Path) -> pd.DataFrame:
    """Read a TU Delft Swarm DNS text file into a tidy DataFrame."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    first = _find_first_data_line(lines)

    names = [
        "date", "time", "time_sys", "alt_m", "lon_deg", "lat_deg",
        "lst_h", "arglat_deg", "dens_kg_m3",
    ]
    df = pd.read_csv(
        path, sep=r"\s+", engine="python",
        skiprows=first, names=names, comment="#",
    )

    return df.assign(
        time    = pd.to_datetime(df["date"] + " " + df["time"], utc=True),
        lat     = df["lat_deg"].astype(float),
        lon     = ((df["lon_deg"].astype(float) + 180) % 360) - 180,
        alt_km  = df["alt_m"].astype(float) / 1_000.0,
        rho_obs = df["dens_kg_m3"].astype(float),
    ).drop(columns=["date", "lat_deg", "lon_deg", "alt_m", "dens_kg_m3"])


def read_grace_dns_txt(
    path: Path,
    *,
    filter_nominal_only: bool = True,
) -> pd.DataFrame:
    """Read a TU Delft GRACE DNS text file into a tidy DataFrame.

    Parameters
    ----------
    filter_nominal_only:
        When True (default), keep only rows where ``flag_dens == 0``.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    first = _find_first_data_line(lines)

    names = [
        "date", "time_txt", "time_sys", "alt_m", "lon_deg", "lat_deg",
        "lst_h", "arglat_deg", "dens_kg_m3", "dens_mean_kg_m3",
        "flag_dens", "flag_dens_mean",
    ]
    df = pd.read_csv(
        path, sep=r"\s+", engine="python",
        skiprows=first, names=names,
    )

    out = pd.DataFrame({
        "time"          : pd.to_datetime(df["date"] + " " + df["time_txt"], utc=True),
        "lat"           : df["lat_deg"].astype(float),
        "lon"           : ((df["lon_deg"].astype(float) + 180.0) % 360.0) - 180.0,
        "alt_km"        : df["alt_m"].astype(float) / 1_000.0,
        "rho_obs"       : df["dens_kg_m3"].astype(float),
        "rho_mean"      : df["dens_mean_kg_m3"].astype(float),
        "flag_dens"     : df["flag_dens"].astype(float),
        "flag_dens_mean": df["flag_dens_mean"].astype(float),
        "lst_h"         : df["lst_h"].astype(float),
        "arglat_deg"    : df["arglat_deg"].astype(float),
        "time_sys"      : df["time_sys"].astype(str),
    })

    if filter_nominal_only:
        out = out[out["flag_dens"] == 0].copy()

    return out.sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# READER DISPATCH
# ---------------------------------------------------------------------------

def read_dns_txt(path: Path, mission: str) -> pd.DataFrame:
    if mission.upper() == "GRACE":
        return read_grace_dns_txt(path)
    return read_swarm_dns_txt(path)


# ---------------------------------------------------------------------------
# FTP DOWNLOAD
# ---------------------------------------------------------------------------

def download_dns_zips(
    host: str,
    ftp_path: str,
    outdir: Path,
    years: tuple[int, int],
) -> list[Path]:
    """Download DNS zip files for the requested years. Returns local paths."""
    year_pat = "|".join(str(y) for y in range(years[0], years[1] + 1))
    pattern  = re.compile(rf"_({year_pat})_\d{{2}}_v02\.zip$")

    outdir.mkdir(parents=True, exist_ok=True)

    ftp = FTP(host, timeout=30)
    ftp.login("anonymous", "")
    ftp.cwd(ftp_path)

    all_files = ftp.nlst()
    targets   = [f for f in all_files if pattern.search(f) and "DNS" in f]
    print(f"Found {len(targets)} DNS zip(s) to download.")

    local_paths = []
    for fname in targets:
        local = outdir / fname
        if local.exists():
            print(f"  Already downloaded: {fname}")
        else:
            print(f"  Downloading: {fname}")
            with open(local, "wb") as fh:
                ftp.retrbinary(f"RETR {fname}", fh.write)
        local_paths.append(local)

    ftp.quit()
    return sorted(local_paths)


# ---------------------------------------------------------------------------
# PROCESSING
# ---------------------------------------------------------------------------

def _parse_stem(stem: str) -> dict:
    """Extract mission, product, year, month from a zip stem like GA_DNS_ACC_2009_01_v02."""
    parts = stem.split("_")
    return {
        "mission": parts[0],
        "product": parts[1],
        "year"   : int(parts[3]),
        "month"  : int(parts[4]),
    }


def build_dataframe(zip_paths: list[Path], years: tuple[int, int]) -> pd.DataFrame:
    frames = []

    for zpath in tqdm(zip_paths, desc="Processing months"):
        meta = _parse_stem(zpath.stem)

        if not (years[0] <= meta["year"] <= years[1]):
            continue
        if meta["product"] != "DNS":
            continue

        folder = zpath.parent / zpath.stem
        if not folder.exists():
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(folder)

        # Prefer <stem>.txt; fall back to first .txt in folder
        txt_path = folder / f"{zpath.stem}.txt"
        if not txt_path.exists():
            txts = sorted(folder.glob("*.txt"))
            if not txts:
                print(f"  No .txt found in {folder}")
                continue
            txt_path = txts[0]

        try:
            df_month = read_dns_txt(txt_path, meta["mission"])
        except Exception as exc:
            print(f"  Failed to read {txt_path.name}: {exc}")
            continue

        df_month["mission"] = meta["mission"]
        df_month["year"]    = meta["year"]
        df_month["month"]   = meta["month"]
        df_month["source"]  = zpath.stem
        frames.append(df_month)

    if not frames:
        raise RuntimeError("No monthly files were read. Check paths / filters.")

    return pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_density(df: pd.DataFrame, title: str = "Thermospheric neutral density") -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["time"].iloc[::100], df["rho_obs"].iloc[::100], linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Neutral density [kg/m³]")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Download
    zip_paths = download_dns_zips(FTP_HOST, FTP_PATH, OUTDIR, YEARS)

    # 2. Build combined DataFrame
    df = build_dataframe(zip_paths, YEARS)
    print(f"Total rows: {df.shape[0]:,}")
    print(df[["time", "rho_obs"]].head())

    # 3. Save
    df.assign(time=df["time"].dt.tz_localize(None)).to_parquet(
        PARQUET_OUT, engine="pyarrow", index=False
    )
    print(f"Saved → {PARQUET_OUT}")

    # 4. Quick sanity plot
    plot_density(df)
