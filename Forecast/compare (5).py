


"""
compare.py
- Collocates Swarm observations (scaled to GRACE altitude) to the model prediction grid.
- Computes bias, MAE, RMSE, MAPE, R², and log-space metrics for prediction vs MSIS.
- Runs from the command line or imported as a module; saves collocated CSV and optional plots.

Usage:
    python "compare (5).py" --result_df result_df.csv --scaled_swarm scaled_swarm.csv --plot
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------
@dataclass
class GridSpec:
    lat_start: float = -87.5
    lat_step: float  =  0.5
    lon_start: float = -180.0
    lon_step: float  =   0.5


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


# ---------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------
def normalize_lon_deg(lon: pd.Series | np.ndarray) -> np.ndarray:
    """Normalize longitudes to [-180, 180)."""
    lon_np = np.asarray(lon, dtype=float)
    return ((lon_np + 180.0) % 360.0) - 180.0


def _grid_bin_center(values: np.ndarray, start: float, step: float) -> np.ndarray:
    """
    Map continuous coordinates to nearest-bin *center*
    given uniform bins with edges: start + n*step
    and centers: start + (n + 0.5)*step.
    """
    n = np.floor((values - start) / step).astype(int)
    return start + (n + 0.5) * step


# ---------------------------------------------------------------------
# Collocation
# ---------------------------------------------------------------------
def deduplicate_cells(grid_df: pd.DataFrame,
                      lat_col: str = "latitude",
                      lon_col: str = "longitude") -> pd.DataFrame:
    """
    Ensure there is exactly one row per (lat, lon) cell.
    If duplicates exist, keep the first. (You could also average here.)
    """
    before = len(grid_df)
    out = grid_df.drop_duplicates(subset=[lat_col, lon_col]).copy()
    after = len(out)
    if after < before:
        print(f"[deduplicate_cells] Reduced grid from {before} to {after} unique cells.")
    return out


def collocate_points_to_grid(points_df: pd.DataFrame,
                             grid_df: pd.DataFrame,
                             grid: GridSpec,
                             lat_col_p: str = "lat",
                             lon_col_p: str = "lon",
                             lat_col_g: str = "latitude",
                             lon_col_g: str = "longitude") -> pd.DataFrame:
    """
    Hard collocation (no interpolation): snap each point to the nearest model grid cell center
    and join that cell's fields (rho_pred, msis_rho, ...).
    """
    if lat_col_p not in points_df.columns or lon_col_p not in points_df.columns:
        # Accept 'latitude'/'longitude' in Swarm too, if present
        if {"latitude", "longitude"}.issubset(points_df.columns):
            points_df = points_df.rename(columns={"latitude": lat_col_p, "longitude": lon_col_p}).copy()
        else:
            missing = {lat_col_p, lon_col_p} - set(points_df.columns)
            raise ValueError(f"Points DataFrame missing columns: {missing}")

    pts = points_df.copy()
    pts["_lon_norm"] = normalize_lon_deg(pts[lon_col_p].to_numpy())
    pts["_lat_cell"] = _grid_bin_center(pts[lat_col_p].to_numpy(dtype=float), grid.lat_start, grid.lat_step)
    pts["_lon_cell"] = _grid_bin_center(pts["_lon_norm"].to_numpy(dtype=float), grid.lon_start, grid.lon_step)

    # Prepare grid (ensure same lon normalization)
    grd = grid_df.copy()
    grd["_lon_norm"] = normalize_lon_deg(grd[lon_col_g].to_numpy())
    grd["_lat_cell"] = _grid_bin_center(grd[lat_col_g].to_numpy(dtype=float), grid.lat_start, grid.lat_step)
    grd["_lon_cell"] = _grid_bin_center(grd["_lon_norm"].to_numpy(dtype=float), grid.lon_start, grid.lon_step)
    # Reduce to unique cell rows
    grd_u = deduplicate_cells(grd, lat_col="_lat_cell", lon_col="_lon_cell").set_index(["_lat_cell", "_lon_cell"])

    # Join by cell
    joined = pts.set_index(["_lat_cell", "_lon_cell"]).join(grd_u, how="left", rsuffix="_grid").reset_index()
    joined = joined.rename(columns={"_lat_cell": "latitude_cell", "_lon_cell": "longitude_cell"})

    return joined


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Bias, MAE, RMSE, MAPE (obs>0), R², Pearson r,
    plus MSE in log-space (y>0), and Top5% MAE in linear and log spaces.

    Definitions/assumptions:
      - mse_log: mean((ln(y_pred) - ln(y_true))^2) over pairs with y_true>0 and y_pred>0.
      - top5: mean absolute error computed over the largest 5% absolute residuals in linear space.
      - top5_log: mean absolute error over the largest 5% absolute residuals in log-space,
                  computed on pairs with y_true>0 and y_pred>0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Finite mask
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    t = y_true[mask]
    p = y_pred[mask]

    if t.size == 0:
        return dict(
            count=0, bias=np.nan, mae=np.nan, rmse=np.nan, mape=np.nan, r2=np.nan, corr=np.nan,
            mse_log=np.nan, top5=np.nan, top5_log=np.nan
        )

    # Linear residuals
    diff = p - t
    abs_diff = np.abs(diff)

    # Core linear metrics
    count = int(t.size)
    bias = float(np.mean(diff))
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff**2)))

    # MAPE (obs>0)
    pos_obs = t > 0
    mape = float(np.mean(np.abs((p[pos_obs] - t[pos_obs]) / t[pos_obs]))) if np.any(pos_obs) else np.nan

    # R^2 and Pearson r
    if np.std(t) == 0:
        r2 = np.nan
        corr = np.nan
    else:
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot)
        corr = float(np.corrcoef(t, p)[0, 1])

    # --- New metrics ---

    # Log-space subset (strictly positive pairs)
    log_mask = (t > 0) & (p > 0)
    if np.any(log_mask):
        lt = np.log(t[log_mask])
        lp = np.log(p[log_mask])
        ldiff = lp - lt
        abs_ldiff = np.abs(ldiff)
        mse_log = float(np.mean(ldiff ** 2))

        # Top 5% by absolute error in log-space
        k_log = max(1, int(np.ceil(0.05 * abs_ldiff.size)))
        # argsort descending on absolute log residuals
        idx_log = np.argpartition(-abs_ldiff, kth=k_log - 1)[:k_log]
        top5_log = float(np.mean(abs_ldiff[idx_log]))
    else:
        mse_log = np.nan
        top5_log = np.nan

    # Top 5% by absolute error in linear space
    if abs_diff.size > 0:
        k = max(1, int(np.ceil(0.05 * abs_diff.size)))
        idx = np.argpartition(-abs_diff, kth=k - 1)[:k]
        top5 = float(np.mean(abs_diff[idx]))
    else:
        top5 = np.nan

    return dict(
        count=count, bias=bias, mae=mae, rmse=rmse, mape=mape, r2=r2, corr=corr,
        mse_log=mse_log, top5=top5, top5_log=top5_log
    )


# ---------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------
def collocate_and_compare(result_df_path: str,
                          scaled_swarm_path: str,
                          out_csv: str = "swarm_vs_model_collocated.csv",
                          grid: GridSpec = GridSpec(),
                          drop_nonpositive_obs: bool = False,
                          verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Load result_df and scaled_swarm, collocate Swarm to grid, compute residuals and metrics,
    and save the collocated table.
    """
    # 1) Read
    result_df = _read_any(result_df_path)
    scaled_swarm = _read_any(scaled_swarm_path)

    # 2) Sanity checks for required columns
    required_grid_cols = {"latitude", "longitude", "rho_pred", "msis_rho"}
    missing_grid = required_grid_cols - set(result_df.columns)
    if missing_grid:
        raise ValueError(f"result_df is missing columns: {missing_grid}")

    if "rho_obs_scaled_to_tgt" not in scaled_swarm.columns:
        raise ValueError("scaled_swarm is missing 'rho_obs_scaled_to_tgt'.")

    # 3) Optional: drop nonpositive obs (if any numerical noise)
    if drop_nonpositive_obs:
        before = len(scaled_swarm)
        scaled_swarm = scaled_swarm.loc[scaled_swarm["rho_obs_scaled_to_tgt"] > 0].copy()
        if verbose:
            print(f"[filter] Dropped {before - len(scaled_swarm)} rows with nonpositive 'rho_obs_scaled_to_tgt'.")

    # 4) Collocate
    joined = collocate_points_to_grid(
        points_df=scaled_swarm,
        grid_df=result_df,
        grid=grid,
        lat_col_p="lat", lon_col_p="lon",
        lat_col_g="latitude", lon_col_g="longitude"
    )

    # 5) Residuals (Swarm minus Model / MSIS)
    joined["diff_swarm_pred"] = joined["rho_obs_scaled_to_tgt"] - joined["rho_pred"]
    joined["diff_swarm_msis"] = joined["rho_obs_scaled_to_tgt"] - joined["msis_rho"]

    # 6) Metrics
    m_pred = regression_metrics(joined["rho_obs_scaled_to_tgt"].to_numpy(), joined["rho_pred"].to_numpy())
    m_msis = regression_metrics(joined["rho_obs_scaled_to_tgt"].to_numpy(), joined["msis_rho"].to_numpy())

    if verbose:
        print("\n=== Metrics: Prediction vs Swarm (scaled) ===")
        for k, v in m_pred.items():
            print(f"{k:>6}: {v:.6e}" if isinstance(v, float) else f"{k:>6}: {v}")

        print("\n=== Metrics: MSIS vs Swarm (scaled) ===")
        for k, v in m_msis.items():
            print(f"{k:>6}: {v:.6e}" if isinstance(v, float) else f"{k:>6}: {v}")

    # 7) Save tidy collocated CSV
    cols_keep = [
        "time" if "time" in joined.columns else None,
        "lat", "lon", "latitude_cell", "longitude_cell",
        "rho_obs_scaled_to_tgt",
        "rho_obs" if "rho_obs" in joined.columns else None,
        "rho_pred", "msis_rho",
        "diff_swarm_pred", "diff_swarm_msis"
    ]
    cols_keep = [c for c in cols_keep if c is not None]
    out = joined[cols_keep].copy()
    out.to_csv(out_csv, index=False)
    if verbose:
        print(f"\nSaved collocated comparisons to: {out_csv} (n={len(out)})")

    return out, m_pred, m_msis


# ---------------------------------------------------------------------
# Quick plots (optional)
# ---------------------------------------------------------------------
def quick_plots(colloc: pd.DataFrame):
    """
    Produce simple diagnostic plots:
    - Obs (scaled) vs Pred
    - Obs (scaled) vs MSIS
    - Residual histograms
    - Median residual by latitude band
    """
    # Scatter: obs vs model
    def scatter_obs_vs(model_col: str, title: str):
        x = colloc["rho_obs_scaled_to_tgt"].to_numpy(dtype=float)
        y = colloc[model_col].to_numpy(dtype=float)
        plt.figure(figsize=(5, 5))
        plt.scatter(x, y, s=10, alpha=0.4)
        lo = min(np.nanmin(x), np.nanmin(y))
        hi = max(np.nanmax(x), np.nanmax(y))
        plt.plot([lo, hi], [lo, hi], "k--", lw=1)
        plt.xlabel("Swarm (scaled) ρ [kg/m³]")
        plt.ylabel(f"{title} ρ [kg/m³]")
        plt.title(f"Obs vs {title}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = f"scatter_{model_col}.png"
        plt.savefig(filename, dpi=300)
        plt.show()

    scatter_obs_vs("rho_pred", "Prediction")
    scatter_obs_vs("msis_rho", "MSIS")

    # Residual histograms
    plt.figure(figsize=(6, 4))
    plt.hist(colloc["diff_swarm_pred"], bins=60, alpha=0.6, label="Swarm − Pred")
    plt.hist(colloc["diff_swarm_msis"], bins=60, alpha=0.6, label="Swarm − MSIS")
    plt.xlabel("Residual [kg/m³]")
    plt.ylabel("Count")
    plt.title("Residual distributions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("residuals.png", dpi=300)
    plt.show()

    # Latitudinal bands (10°)
    bins = np.arange(-90, 95, 10)
    labels = 0.5 * (bins[1:] + bins[:-1])
    lat_series = colloc["lat"] if "lat" in colloc.columns else colloc["latitude"]
    colloc = colloc.copy()
    colloc["lat_bin"] = pd.cut(lat_series, bins=bins, labels=labels, include_lowest=True)
    lat_stats = colloc.groupby("lat_bin")[["diff_swarm_pred", "diff_swarm_msis"]].median()
    lat_stats.plot(kind="bar", figsize=(10, 4))
    plt.ylabel("Median residual [kg/m³]")
    plt.title("Median (Swarm − Model) by latitude band")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("lat_bands.png", dpi=300)
    plt.show()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare model/MSIS to Swarm (scaled) via grid collocation.")
    ap.add_argument("--result_df", required=True, help="Path to result_df (CSV or Parquet).")
    ap.add_argument("--scaled_swarm", required=True, help="Path to scaled_swarm (CSV or Parquet).")
    ap.add_argument("--out_csv", default="swarm_vs_model_collocated.csv", help="Output CSV for collocated rows.")
    ap.add_argument("--lat_start", type=float, default=GridSpec.lat_start, help="Grid latitude start.")
    ap.add_argument("--lat_step",  type=float, default=GridSpec.lat_step,  help="Grid latitude step.")
    ap.add_argument("--lon_start", type=float, default=GridSpec.lon_start, help="Grid longitude start.")
    ap.add_argument("--lon_step",  type=float, default=GridSpec.lon_step,  help="Grid longitude step.")
    ap.add_argument("--drop_nonpositive_obs", action="store_true", help="Drop obs<=0 before comparing.")
    ap.add_argument("--plot", action="store_true", help="Make quick diagnostic plots.")
    return ap.parse_args()


def main():
    args = parse_args()
    grid = GridSpec(
        lat_start=args.lat_start,
        lat_step=args.lat_step,
        lon_start=args.lon_start,
        lon_step=args.lon_step,
    )
    colloc, m_pred, m_msis = collocate_and_compare(
        result_df_path=args.result_df,
        scaled_swarm_path=args.scaled_swarm,
        out_csv=args.out_csv,
        grid=grid,
        drop_nonpositive_obs=args.drop_nonpositive_obs,
        verbose=True
    )
    if args.plot:
        quick_plots(colloc)

if __name__ == "__main__":
    main()