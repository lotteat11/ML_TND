# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
run_pymsis.py
- Loads GRACE observations and filters to the 2009–2016 training window.
- Fetches F10.7 and Ap space weather indices via pymsis, then runs NRLMSISE-2.1.
- Saves the result with an msis_rho column added as a parquet file.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymsis

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent

INPUT_PARQUET  = str(ROOT / "grace_dns_2009_2016.parquet")
OUTPUT_PARQUET = str(ROOT / "grace_dns_with_tnd_y200916_v4_0809.parquet")

TIME_MIN = "2009-06-06"
TIME_MAX = "2016-01-01"

AP_COLS = [
    "ap_daily",
    "ap_0h",
    "ap_m3h",
    "ap_m6h",
    "ap_m9h",
    "ap_avg12_33h",
    "ap_avg36_57h",
]

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load
    print("Loading GRACE data...")
    df = pd.read_parquet(INPUT_PARQUET, engine="pyarrow")
    df["time"] = pd.to_datetime(df["time"])
    df = df[(df["time"] > TIME_MIN) & (df["time"] < TIME_MAX)].copy()
    df = df.sort_values("time")
    print(f"GRACE rows after filtering: {len(df):,}")

    # 2. Fetch space weather
    print("Fetching space weather indices...")
    f107, f107a, ap = pymsis.utils.get_f107_ap(df["time"])
    df["f107"]  = f107
    df["f107a"] = f107a
    df[AP_COLS] = ap.astype(float)

    # 3. Run NRLMSISE-2.1
    print("Running NRLMSISE-2.1...")
    out = pymsis.msis.calculate(
        pd.to_datetime(df["time"]),
        df["lon"].values,
        df["lat"].values,
        df["alt_km"].values,
    )
    df["msis_rho"] = out[:, 0]
    print(f"NaNs in msis_rho: {df['msis_rho'].isna().sum()}")

    # 4. Save
    df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", index=False)
    print(f"Saved → {OUTPUT_PARQUET}")

    # 5. Sanity plot — MSIS vs observed density over last day
    end   = df["time"].max()
    start = end - pd.Timedelta(days=1)
    sub   = df.loc[(df["time"] >= start) & (df["time"] <= end)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sub["time"], sub["msis_rho"], label="MSIS Density",    color="orange", alpha=0.7)
    ax.plot(sub["time"], sub["rho_obs"],  label="Observed Density", color="blue",   alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Density (kg/m³)")
    ax.set_title("MSIS vs Observed Atmospheric Density — Last Day")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()
