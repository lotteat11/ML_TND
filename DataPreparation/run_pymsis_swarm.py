# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
run_pymsis_swarm.py
- Loads Swarm DNS observations and adds NRLMSISE-2.1 density.
- Fetches F10.7 and Ap space weather indices and joins them hourly.
- Runs MSIS at the satellite altitude and at 400 km reference altitude.
- Saves the result with tnd_kg_m3, tnd_kg_m3_400, and rho_obs_400 columns
  as swarm_dns_with_tnd_y2001516_v1_0309.parquet.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import numpy as np

import pymsis_utils as func_pymis

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INPUT_PARQUET  = "swarm_dns_2015_2016_03092025.parquet"
OUTPUT_PARQUET = "swarm_dns_with_tnd_y2001516_v1_0309.parquet"

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load
    print("Loading Swarm data...")
    df = func_pymis.load_grace_csv(Path(INPUT_PARQUET))

    # 2. Fetch space weather
    print("Fetching space weather indices...")
    tmin = df["time"].min().compute().floor("D")
    tmax = df["time"].max().compute().floor("D")
    sw_hourly = func_pymis.fetch_spaceweather_hourly_robust(tmin, tmax)

    # 3. Join indices
    print("Joining space weather indices...")
    df_joined = func_pymis.join_indices2(df, sw_hourly)

    # 4. Prepare pymsis inputs
    print("Preparing MSIS inputs...")
    inputs = func_pymis.prepare_inputs_for_pymsis(df_joined)

    # 5. Run MSIS at satellite altitude and 400 km reference altitude
    print("Running NRLMSISE-2.1...")
    _, tnd, _, tnd_400 = func_pymis.run_msis_400(inputs, version=2.1)

    # 6. Assemble output DataFrame
    df_out = df_joined.compute()
    df_out["tnd_kg_m3"]     = tnd
    df_out["tnd_kg_m3_400"] = tnd_400
    df_out["rho_obs_400"]   = (df_out["tnd_kg_m3_400"] / df_out["tnd_kg_m3"]) * df_out["rho_obs"]

    print(f"\nSwarm input rows : {len(df_out):,}")
    print(f"NaNs in tnd      : {np.isnan(tnd).sum()}")

    # 7. Save
    df_out.to_parquet(OUTPUT_PARQUET, engine="pyarrow", index=False)
    print(f"Saved → {OUTPUT_PARQUET}")
