# %%


import re
from ftplib import FTP
from pathlib import Path
import re
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  


# %% 

def read_swarm_dns_txt(path: Path) -> pd.DataFrame:
    import pandas as pd
    import re

    # Read lines
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Find first data line
    date_line_pat = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s+(UTC|GPS)\b")
    first_data_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#") or not s:
            continue
        if date_line_pat.match(s):
            first_data_idx = i
            break

    if first_data_idx is None:
        raise ValueError("Could not locate first data line in file: " + str(path))

    # Column names for Swarm
    names = [
        "date", "time", "time_sys", "alt_m", "lon_deg", "lat_deg",
        "lst_h", "arglat_deg", "dens_kg_m3"
    ]

    # Read data
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        skiprows=first_data_idx,
        names=names,
        comment="#",
    )

    # Build datetime
    dt = pd.to_datetime(df["date"] + " " + df["time"], utc=True, errors="raise")

    # Final dataframe
    df = df.assign(
        time=dt,
        lat=df["lat_deg"].astype(float),
        lon=((df["lon_deg"].astype(float) + 180) % 360) - 180,
        alt_km=df["alt_m"].astype(float) / 1000.0,
        rho_obs=df["dens_kg_m3"].astype(float),
    ).drop(columns=["date", "lat_deg", "lon_deg", "alt_m", "dens_kg_m3"])

    return df


# %% 
def read_grace_dns_txt(
    path: Path,
    *,
    keep_raw: bool = False,
    convert_gps_to_utc: bool = False,
    filter_nominal_only: bool = False  # keep only rows with flag_dens == 0
) -> pd.DataFrame:
    """
    Read a Delft GRACE accelerometer-derived density (DNS) text file into a tidy DataFrame.

    Returns columns (always):
        time [UTC pandas.Timestamp], lat, lon, alt_km, rho_obs, rho_mean,
        flag_dens, flag_dens_mean, lst_h, arglat_deg, time_sys

    Options:
        keep_raw=True      -> keep original raw columns too (date, time_txt, alt_m, lon_deg, lat_deg, ...)
        convert_gps_to_utc -> if time_sys == 'GPS', subtract GPS-UTC offset to get true UTC
        filter_nominal_only-> drop rows where flag_dens != 0
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # First real data line looks like: YYYY-MM-DD HH:MM:SS[.mmm] UTC|GPS ...
    date_line_pat = re.compile(
        r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+(?:UTC|GPS)\b"
    )
    first_data_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if date_line_pat.match(s):
            first_data_idx = i
            break
    if first_data_idx is None:
        raise ValueError(f"Couldn't find first data line in {path}")

    names = [
        "date", "time_txt", "time_sys", "alt_m", "lon_deg", "lat_deg",
        "lst_h", "arglat_deg", "dens_kg_m3", "dens_mean_kg_m3",
        "flag_dens", "flag_dens_mean"
    ]

    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        skiprows=first_data_idx,
        names=names,
    )

    # Build datetime from date + time string
    t = pd.to_datetime(df["date"] + " " + df["time_txt"], utc=True, errors="raise")


    # Normalize fields
    df_norm = pd.DataFrame({
        "time": t,
        "lat": df["lat_deg"].astype(float),
        "lon": ((df["lon_deg"].astype(float) + 180.0) % 360.0) - 180.0,
        "alt_km": df["alt_m"].astype(float) / 1000.0,
        "rho_obs": df["dens_kg_m3"].astype(float),
        "rho_mean": df["dens_mean_kg_m3"].astype(float),
        "flag_dens": df["flag_dens"].astype(float),
        "flag_dens_mean": df["flag_dens_mean"].astype(float),
        "lst_h": df["lst_h"].astype(float),
        "arglat_deg": df["arglat_deg"].astype(float),
        "time_sys": df["time_sys"].astype(str),
    })

    if filter_nominal_only:
        df_norm = df_norm[df_norm["flag_dens"] == 0].copy()

    # Optional: include raw fields
    if keep_raw:
        raw = df[["date","time_txt","alt_m","lon_deg","lat_deg","dens_kg_m3","dens_mean_kg_m3"]].copy()
        out = pd.concat([df_norm, raw], axis=1)
    else:
        out = df_norm

    return out.sort_values("time").reset_index(drop=True)
# %%

def read_grace_dns_txt(path: Path) -> pd.DataFrame:
    # Find the first data line (starts with YYYY-MM-DD)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    date_line_pat = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s+(UTC|GPS)\b")
    first_data_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#") or not s:
            continue
        if date_line_pat.match(s):
            first_data_idx = i
            break

    if first_data_idx is None:
        raise ValueError("Could not locate first data line in file: " + str(path))

    # Column names per the header you pasted
    names = [
        "date", "time", "time_sys", "alt_m", "lon_deg", "lat_deg",
        "lst_h", "arglat_deg", "dens_kg_m3", "dens_mean_kg_m3",
        "flag_dens", "flag_dens_mean"
    ]

    # Read from the first data line
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        skiprows=first_data_idx,
        names=names,
        comment="#",
    )

    # Build datetime (treat GPS as UTC unless you need sub-minute precision)
    dt = pd.to_datetime(df["date"] + " " + df["time"], utc=True, errors="raise")
    # If you *really* want UTC from GPS, subtract the GPS-UTC offset (varies by year).
    # For drag modeling, the ~tens-of-seconds difference is negligible; most users skip it.

    df = df.assign(
        time=dt,
        lat=df["lat_deg"].astype(float),
        lon=((df["lon_deg"].astype(float) + 180) % 360) - 180,
        alt_km=df["alt_m"].astype(float) / 1000.0,
        rho_obs=df["dens_kg_m3"].astype(float),
    ).drop(columns=["date", "lat_deg", "lon_deg", "alt_m", "dens_kg_m3"])

    return df



# %%

from ftplib import FTP
import socket

HOST = "thermosphere.tudelft.nl"
PATH = "/version_02/Swarm_data"

for passive in (True, False):
    try:
        print(f"Trying passive={passive}")
        ftp = FTP(HOST, timeout=20)
        ftp.set_pasv(passive)  # True = PASV, False = active
        ftp.login("anonymous", "")
        ftp.cwd(PATH)
        print(ftp.nlst()[:10])
        ftp.quit()
        break
    except Exception as e:
        print(f"Mode passive={passive} failed: {e}")

# %%

from ftplib import FTP

# Connect to FTP
ftp = FTP("thermosphere.tudelft.nl")

ftp.login("anonymous", "")

#ftp.cwd("/version_02/Swarm_data")
ftp.cwd("/version_02/GRACE_data")

# List all files
files = ftp.nlst()
print(files)
# %%


# Filter: only DNS files 2009–2010 (change to WND if you want wind data)
#pattern = re.compile(r"_(2009|2010|2011|2012)_\d{2}_v02\.zip$")
pattern = re.compile(r"_(2015|2016)_\d{2}_v02\.zip$")

filtered = [f for f in files if pattern.search(f) and "DNS" in f]

print("Number of files:", len(filtered))
print(filtered[:10])  # preview


# Download the filtered files

#outdir = Path("GRACE_2009_2012_v02")
outdir = Path("SWARM_201516_v02")
outdir.mkdir(exist_ok=True)

for fname in filtered:
    local = outdir / fname
    if local.exists():
        print("Already downloaded:", fname)
        continue
    print("Downloading:", fname)
    with open(local, "wb") as f:
        ftp.retrbinary(f"RETR {fname}", f.write)

ftp.quit()



# Extract all zip files
for zpath in outdir.glob("*.zip"):
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(outdir / zpath.stem)  # extract into folder per month
        print("Extracted:", zpath.name)



# %%

from pathlib import Path

print("OUTDIR:", outdir.resolve())
all_zip_paths = sorted(outdir.glob("*.zip"))
print("ZIP COUNT:", len(all_zip_paths))
for p in all_zip_paths[:5]:
    print("  ", p.name)
# %%
outdir = Path("SWARM_201516_v02")

#outdir = Path("GRACE_2009_2012_v02")  # folder where your .zip files live
all_zip_paths = sorted(outdir.glob("*.zip"))

frames = []

for zpath in tqdm(all_zip_paths, desc="Processing months"):
    stem = zpath.stem  # e.g., GA_DNS_ACC_2009_01_v02
    parts = stem.split("_")
    mission = parts[0]          # GA or GB
    product = parts[1]          # DNS or WND
    year    = int(parts[3])
    month   = int(parts[4])

    # Only use 2009–2013 (you can skip this if your zips are already filtered)
    if not (2014 <= year <= 2016):
        continue
    if product != "DNS":
        continue  # only density; change/remove if you want WND too

    folder = outdir / stem
    print(folder)
    if not folder.exists():
        # extract zip into subfolder named after the zip
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(folder)

    # main data file is usually <stem>.txt inside the subfolder
    sample_file = folder / f"{stem}.txt"
    if not sample_file.exists():
        # fallback: first .txt in the folder
        txts = sorted(folder.glob("*.txt"))
        if not txts:
            print(" No .txt in", folder)
            continue
        sample_file = txts[0]

    try:
       # df_month = read_grace_dns_txt(sample_file)
        df_month = read_swarm_dns_txt(sample_file)
    except Exception as e:
        print(f" Failed to read {sample_file.name}: {e}")
        continue

    # Add identifiers and filter out anomalous data (flag 0 = nominal)
    df_month["mission"] = mission
    df_month["year"] = year
    df_month["month"] = month
    df_month["source"] = stem

    print("PUT DQ BACK IN FOR GRACE")
   #df_month = df_month[df_month["flag_dens"] == 0].copy()

    frames.append(df_month)

# Concatenate
if not frames:
    raise RuntimeError("No monthly files read. Check paths/filters.")



big_0911 = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)

print(big_0911.shape, big_0911[["time","rho_obs"]].head())
# %%
big_0911.assign(time=big_0911["time"].dt.tz_localize(None)).to_parquet("swarm_dns_2015_2016_03092025.parquet", engine="pyarrow", index=False)

# %%
big_0911.assign(time=big_0911["time"].dt.tz_localize(None)).to_parquet("grace_dns_2009_2012.parquet", engine="pyarrow", index=False)

# %%
big_0911.to_csv(outdir / "grace_dns_2009_2010.csv", index=False)

# %%
print(big_0911.shape, big_0911[["time","rho_obs"]].head())

# %%
plt.figure(figsize=(10,4))
plt.plot(big_0911["time"].loc[::100], big_0911["rho_obs"].loc[::100], linewidth=0.8)
plt.yscale("log")  # densities span orders of magnitude
plt.xlabel("Time (UTC)")
plt.ylabel("Thermospheric neutral density [kg/m³]")
plt.title(sample_dir.name)
plt.tight_layout()
plt.show()
# %%
