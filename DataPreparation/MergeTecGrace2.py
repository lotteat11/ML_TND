"""
GRACE × TEC Matching Pipeline
==============================
Loads GRACE density data and CODE TEC maps, then spatially matches each
GRACE observation to its nearest TEC grid point using a K-D tree.

Output
------
grace_data_merged_v3.parquet  — GRACE DataFrame with matched TEC columns appended.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import Feature_functions as ff  # noqa: F401  (kept for side-effects / downstream use)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

GRACE_PARQUET   = "grace_dns_with_tnd_y200916_v4_0809.parquet"
TEC_PARQUET     = "tec_codg_2009-2017_doy1-365.parquet"
OUTPUT_PARQUET  = "grace_data_merged_v3.parquet"

GRACE_TIME_MAX  = "2016-01-01"          # exclusive upper bound

TEC_TIME_WINDOW = timedelta(hours=3)    # ± window around each TEC epoch
MAX_CHORD_DIST  = 4.15                  # chord-distance QC threshold
MIN_TEC_POINTS  = 100                   # skip epoch if grid is too sparse

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def latlon_to_cartesian(coords_rad: np.ndarray) -> np.ndarray:
    """Convert (lat, lon) in radians to unit-sphere Cartesian (X, Y, Z)."""
    lat, lon = coords_rad[:, 0], coords_rad[:, 1]
    return np.vstack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ]).T


# ---------------------------------------------------------------------------
# LOAD & PREPROCESS
# ---------------------------------------------------------------------------

def load_grace(path: str) -> tuple[pd.DataFrame, pl.DataFrame]:
    """Load GRACE parquet, filter to before GRACE_TIME_MAX, return both pandas and polars."""
    df = pd.read_parquet(path, engine="pyarrow")
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"] < GRACE_TIME_MAX].sort_values("time").copy()
    return df


def load_tec(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


def to_polars_grace(df: pd.DataFrame) -> pl.DataFrame:
    return (
        pl.from_pandas(df)
        .with_columns([
            pl.col("time")
              .dt.cast_time_unit("us")
              .dt.replace_time_zone("UTC")
              .alias("grace_time"),
            pl.arange(0, pl.len()).alias("original_index"),
        ])
        .drop("time")
        .sort("grace_time")
    )


def to_polars_tec(df: pd.DataFrame) -> pl.DataFrame:
    return (
        pl.from_pandas(df)
        .with_columns([
            pl.col("epoch")
              .dt.cast_time_unit("us")
              .dt.replace_time_zone("UTC")
              .alias("epoch_tec"),
        ])
        .drop("epoch")
        .sort("epoch_tec")
    )


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_density_sample(df: pd.DataFrame, days: int = 1) -> None:
    """Quick sanity plot: MSIS vs observed density over the last `days` days."""
    end   = df["time"].max()
    start = end - pd.Timedelta(days=days)
    sub   = df.loc[(df["time"] >= start) & (df["time"] <= end)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sub["time"], sub["msis_rho"],  label="MSIS Density",     color="orange", alpha=0.7)
    ax.plot(sub["time"], sub["rho_obs"],   label="Observed Density",  color="blue",   alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Density (kg/m³)")
    ax.set_title(f"MSIS vs Observed Atmospheric Density — Last {days} Day(s)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# K-D TREE MATCHING
# ---------------------------------------------------------------------------

def match_tec_to_grace(
    grace_pl: pl.DataFrame,
    tec_pl:   pl.DataFrame,
) -> pl.DataFrame:
    """
    For every TEC epoch, spatially match each nearby GRACE point to its
    nearest TEC grid cell. Falls back to the last valid TEC value when no
    neighbour is found within MAX_CHORD_DIST.

    Returns a Polars DataFrame with columns:
        original_index, matched_tec_value, matched_tec_latitude,
        matched_tec_longitude, chord_distance, fallback_used
    """
    unique_epochs = (
        tec_pl.select("epoch_tec").unique().sort("epoch_tec")
        .to_series().to_list()
    )
    print(f"Starting K-D Tree matching for {len(unique_epochs)} epochs...")

    results        = []
    n_empty_grace  = 0

    for i, epoch in enumerate(unique_epochs):

        # --- A. TEC grid for this epoch ---
        tec_grid = tec_pl.filter(pl.col("epoch_tec") == epoch)
        tec_np   = tec_grid.select(["latitude", "longitude", "tec_value"]).to_numpy()

        if tec_np.shape[0] < MIN_TEC_POINTS:
            print(f"  Skipping epoch {epoch}: only {tec_np.shape[0]} TEC points.")
            continue

        # --- B. GRACE targets in this time window ---
        grace_targets = grace_pl.filter(
            (pl.col("grace_time") >= epoch - TEC_TIME_WINDOW) &
            (pl.col("grace_time") <  epoch + TEC_TIME_WINDOW)
        ).select(["original_index", "lat", "lon"])

        if grace_targets.height == 0:
            n_empty_grace += 1
            continue

        # --- C. K-D Tree spatial query ---
        tec_cart   = latlon_to_cartesian(np.radians(tec_np[:, :2]))
        grace_cart = latlon_to_cartesian(np.radians(grace_targets.select(["lat", "lon"]).to_numpy()))

        tree = cKDTree(tec_cart)
        distances, indices = tree.query(grace_cart, k=1, distance_upper_bound=MAX_CHORD_DIST)

        # --- D. Assign matches; fallback for out-of-range points ---
        n              = grace_targets.height
        matched        = np.full((n, 3), np.nan)   # [lat, lon, tec]
        quality_flag   = np.zeros(n, dtype=np.int8)
        invalid_mask   = (indices == tree.n)
        valid_mask     = ~invalid_mask

        matched[valid_mask] = tec_np[indices[valid_mask]]

        # Forward-fill last valid TEC value for invalid points
        last_valid = np.nan
        for idx in range(n):
            if invalid_mask[idx]:
                matched[idx, 2] = last_valid
                quality_flag[idx] = 1
            else:
                last_valid = matched[idx, 2]

        results.append(pl.DataFrame({
            "original_index":       grace_targets["original_index"].to_list(),
            "matched_tec_value":    matched[:, 2],
            "matched_tec_latitude": matched[:, 0],
            "matched_tec_longitude":matched[:, 1],
            "chord_distance":       distances,
            "fallback_used":        quality_flag,
        }))

        if i % 500 == 0 and i > 0:
            print(f"  Progress: {i}/{len(unique_epochs)} epochs processed.")

    print(f"  Epochs with no GRACE coverage: {n_empty_grace}")

    if not results:
        raise RuntimeError("No epochs matched. Check data alignment / filters.")

    # Keep best match per GRACE point (smallest chord distance)
    return (
        pl.concat(results)
        .sort(["original_index", "chord_distance"])
        .unique(subset=["original_index"], keep="first")
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load
    print("Loading GRACE data...")
    grace_pd = load_grace(GRACE_PARQUET)

    print("Loading TEC data...")
    tec_pd = load_tec(TEC_PARQUET)

    # 2. Sanity plot
    plot_density_sample(grace_pd, days=1)

    # 3. Convert to Polars
    grace_pl = to_polars_grace(grace_pd)
    tec_pl   = to_polars_tec(tec_pd)

    # 4. Match
    matched_pl = match_tec_to_grace(grace_pl, tec_pl)

    # 5. Join back onto GRACE
    grace_final_pl = grace_pl.join(matched_pl, on="original_index", how="left")

    print(f"\nGRACE input rows : {len(grace_pd):,}")
    print(f"TEC   input rows : {len(tec_pd):,}")
    print(f"Output rows      : {grace_final_pl.shape[0]:,}")
    print("NaNs for unmatched TEC points are preserved in matched_tec_value.")

    # 6. Save
    grace_final_pl.write_parquet(OUTPUT_PARQUET)
    print(f"Saved → {OUTPUT_PARQUET}")
