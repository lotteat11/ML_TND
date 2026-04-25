"""
off_track.py
- Builds a global lat/lon grid for a chosen UTC snapshot and runs NRLMSISE-2.1 on it.
- Loads a warm-start model snapshot and predicts thermospheric density across the grid.
- Optionally overlays Swarm satellite observations scaled to the GRACE altitude for validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timezone

# External deps used in specific stages
import joblib
import xgboost as xgb

import pymsis
from pymsis import msis
import pymsis.utils


# ------------------------------
# Configuration
# ------------------------------
@dataclass
class Config:
    # Data inputs
    grace_parquet: str = "grace_dns_with_tnd_y200916_v4_0809.parquet"
    tec_parquet: str = "tec_codg_2009-2017_doy1-365.parquet"

    # Model artefacts
    model_file: str = "xgb_model_saved_for_start_date_2016-02-18.json"
    model_file: str = "xgb_model_saved_dr1_post2016_h3_start_2016-02-18.json"
    model_file_core: str = "xgb_model_updated.json"
    scaler_x_file: str = "scaler_xgboost_X_v3.joblib"
    scaler_y_file: str = "scaler_xgboost_y_v3.joblib"

    # Time selection
   # selected_time_utc: datetime = datetime(2016, 2, 16, 8, 0, 0, tzinfo=timezone.utc)
    selected_time_utc: datetime = datetime(2016, 2, 18, 6, 0, 0, tzinfo=timezone.utc)



    # Grid resolution / ranges
    lat_start: float = -87.5
    lat_stop: float = 90.0
    lat_step: float = 0.2
    lon_start: float = -180.0
    lon_stop: float = 185.0
    lon_step: float = 0.09

    # Plotting
    plot_results: bool = True


    ALT_FEATURE_ORDER = [
        "f107a", "lat", "matched_tec_value",
        "lon_cos", "lon_sin", "lst_sin",
        "doy_sin", "doy_cos", "f107", "alt_km",
        "ap_m3h", "ap_m6h",
        "vtec_matched_lag", "vtec_matched_lag2",
        "lst_lat_sin"
    ]

# ------------------------------
# Data loading & preprocessing
# ------------------------------

def load_grace_hourly(grace_path: str) -> pd.DataFrame:
    """Load GRACE parquet and aggregate to hourly means for numeric columns."""
    df = pd.read_parquet(grace_path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['hour'] = df['time'].dt.floor('H')
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    hourly_df = df.groupby('hour')[numeric_cols].mean().reset_index()
    return hourly_df


def get_altitude_from_hour(hourly_df: pd.DataFrame, when: datetime) -> float:
    """Pick the altitude (alt_km) for the given UTC hour from the aggregated GRACE data."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    row = hourly_df[hourly_df['hour'] == pd.Timestamp(when)]
    if row.empty:
        raise ValueError(f"Selected time {when} not found in GRACE hourly data")
    return float(row["alt_km"].iloc[0])


# ------------------------------
# Feature grid construction
# ------------------------------

def build_global_feature_grid(selected_time: datetime,
                              alt_km: float,
                              lat_range: np.ndarray,
                              lon_range: np.ndarray) -> pd.DataFrame:
    """Create a global grid with derived features (lon/lst/doy trigs, etc.)."""
    DOY = selected_time.timetuple().tm_yday
    UTC_hour = selected_time.hour

    rows = []
    for lat in lat_range:
        for lon in lon_range:
            lst = (UTC_hour + lon / 15.0) % 24
            lst_sin = np.sin(2 * np.pi * lst / 24)
            lst_cos = np.cos(2 * np.pi * lst / 24)

            lon_sin = np.sin(np.radians(lon))
            lon_cos = np.cos(np.radians(lon))
            doy_sin = np.sin(2 * np.pi * DOY / 365)
            doy_cos = np.cos(2 * np.pi * DOY / 365)

            rows.append({
                'latitude': lat,
                'longitude': lon,
                'alt_km': alt_km,
                'lon_sin': lon_sin,
                'lon_cos': lon_cos,
                'lst_sin': lst_sin,
                'lst_cos': lst_cos,
                'doy_sin': doy_sin,
                'doy_cos': doy_cos,
                'lst_lat_sin': lst_sin * lat,
                'lst_lat_cos': lst_cos * lat,
            })

    grid_df = pd.DataFrame(rows)
    grid_df["lat"] = grid_df["latitude"]
    grid_df["lon"] = grid_df["longitude"]
    return grid_df

from scipy.interpolate import griddata

def interpolate_tec_to_grid(df_tec: pd.DataFrame, lat_range: np.ndarray, lon_range: np.ndarray) -> pd.DataFrame:
    """
    Interpolates TEC values from the original coarse grid to the finer prediction grid using bilinear interpolation.
    
    Args:
        df_tec (pd.DataFrame): TEC data with columns ['latitude', 'longitude', 'tec_value'].
        lat_range (np.ndarray): Array of latitudes for the finer grid.
        lon_range (np.ndarray): Array of longitudes for the finer grid.
    
    Returns:
        pd.DataFrame: DataFrame with interpolated TEC values for the finer grid.
    """
    # Prepare original points and values
    points = df_tec[['latitude', 'longitude']].values
    values = df_tec['tec_value'].values

    # Create mesh for finer grid
    grid_lat, grid_lon = np.meshgrid(lat_range, lon_range)
    grid_points = np.column_stack((grid_lat.ravel(), grid_lon.ravel()))

    # Bilinear interpolation
    interpolated_values = griddata(points, values, grid_points, method='linear')

    # Build DataFrame for finer grid
    interp_df = pd.DataFrame({
        'latitude': grid_points[:, 0],
        'longitude': grid_points[:, 1],
        'tec_value': interpolated_values
    })

    return interp_df

# ------------------------------
# TEC loading & merge
# ------------------------------

def load_tec_epoch_and_merge(grid_df: pd.DataFrame,
                             tec_parquet: str,
                             selected_time: datetime) -> pd.DataFrame:
    """Load TEC parquet, add lag features (hard-coded), filter to selected epoch, and merge to grid."""
    df_tec = pd.read_parquet(tec_parquet)
    df_tec['epoch'] = pd.to_datetime(df_tec['epoch'], utc=True)

    # Hard-coded lag features as in the original script
    df_tec['matched_tec_value'] = df_tec['tec_value']

    df_tec['latitude'] = df_tec['latitude'].astype(float)
    df_tec['longitude'] = df_tec['longitude'].astype(float)

    df_tec = df_tec.sort_values(['latitude', 'longitude', 'epoch'])
    df_tec['matched_tec_value'] = df_tec['tec_value']
    df_tec['vtec_matched_lag']  = df_tec.groupby(['latitude', 'longitude'])['tec_value'].shift(1)
    df_tec['vtec_matched_lag2'] = df_tec.groupby(['latitude', 'longitude'])['tec_value'].shift(24)

    print("HARD CORE LAG _ REVIEW IF CORRECT")
    print("HARD CORE LAG _ REVIEW IF CORRECT")

    print("HARD CORE LAG _ REVIEW IF CORRECT")


    tec_epoch = df_tec[df_tec['epoch'] == pd.Timestamp(selected_time)]

    merged = grid_df.merge(
        tec_epoch[['latitude', 'longitude', 'tec_value', 'matched_tec_value',
                   'vtec_matched_lag', 'vtec_matched_lag2']],
        on=['latitude', 'longitude'], how='left'
    )
    return merged


# ------------------------------
# Space weather inputs & MSIS
# ------------------------------
AP_COLS = [
    "ap_daily",     # (0) Daily Ap
    "ap_0h",        # (1) 3-hr ap for current time
    "ap_m3h",       # (2) 3-hr ap for 3 hrs before
    "ap_m6h",       # (3) 3-hr ap for 6 hrs before
    "ap_m9h",       # (4) 3-hr ap for 9 hrs before
    "ap_avg12_33h", # (5) avg of 8×3-hr ap, 12–33 hrs prior
    "ap_avg36_57h", # (6) avg of 8×3-hr ap, 36–57 hrs prior
]


def add_space_weather_and_msis(df: pd.DataFrame,
                               selected_time: datetime) -> pd.DataFrame:
    """Fetch f107/f107a/Ap for selected time, duplicate across grid, and run MSIS to get rho."""
    n = len(df)

    f107_arr, f107a_arr, ap_vec = pymsis.utils.get_f107_ap([selected_time])
    f107_scalar = float(f107_arr.item())
    f107a_scalar = float(f107a_arr.item())

    df = df.copy()
    df['f107'] = f107_scalar
    df['f107a'] = f107a_scalar

    ap_vec = np.array(ap_vec).flatten()
    if len(ap_vec) != len(AP_COLS):
        raise RuntimeError(f"Expected {len(AP_COLS)} Ap values, got {len(ap_vec)}")
    ap_repeated = np.tile(ap_vec, (n, 1))
    for i, col in enumerate(AP_COLS):
        df[col] = ap_repeated[:, i]

    # Run MSIS at the grid points
    dates = [selected_time] * n
    lons = df['longitude'].to_numpy()
    lats = df['latitude'].to_numpy()
    alts = df['alt_km'].to_numpy()

    out = msis.calculate(dates, lons, lats, alts)
    df['msis_rho'] = out[:, 0]  # total neutral density
    return df

import numpy as np
import pandas as pd
from pymsis import msis

def scale_swarm_hour_to_alt_many(
    df_hour: pd.DataFrame,
    selected_hour: pd.Timestamp,
    target_alt_km: float,
    lat_col='lat', lon_col='lon', alt_col='alt_km', rho_obs_col='rho_obs',
    batch_size=20000
) -> pd.DataFrame:
    """
    Scale *all* Swarm samples in the selected hour to target_alt_km using MSIS transfer factors.
    Returns a copy of df_hour with added columns:
      - msis_rho_src, msis_rho_tgt, scale_to_tgt, rho_obs_scaled_to_tgt, target_alt_km
    """
    if df_hour.empty:
        return df_hour.copy()

    # Ensure timezone-aware timestamp
    selected_hour = pd.to_datetime(selected_hour, utc=True)

    out = df_hour.copy()
    # Prepare arrays
    lats = out[lat_col].astype(float).to_numpy()
    lons = out[lon_col].astype(float).to_numpy()
    alts_src = out[alt_col].astype(float).to_numpy()
    n = len(out)

    # Prepare outputs
    rho_src = np.empty(n); rho_tgt = np.empty(n)

    # Batch to avoid huge memory spikes
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = slice(start, end)
        # Dates array: same timestamp for all samples (we scale to same hour)
        dates = [selected_hour] * (end - start)

        # MSIS at source altitudes (per-sample alt)
        out_src = msis.calculate(dates, lons[idx], lats[idx], alts_src[idx])
        rho_src[idx] = out_src[:, 0]

        # MSIS at target altitude (same for each sample but location-dependent)
        alts_tgt = np.full(end - start, float(target_alt_km))
        out_tgt = msis.calculate(dates, lons[idx], lats[idx], alts_tgt)
        rho_tgt[idx] = out_tgt[:, 0]

    # Transfer factor and scaled obs
    scale = rho_tgt / np.maximum(rho_src, 1e-30)
    out['msis_rho_src'] = rho_src
    out['msis_rho_tgt'] = rho_tgt
    out['scale_to_tgt'] = scale
    out['rho_obs_scaled_to_tgt'] = out[rho_obs_col].astype(float).to_numpy() * scale
    out['target_alt_km'] = float(target_alt_km)
    return out


def scale_swarm_hour_to_alt(row: pd.DataFrame,
                            selected_hour: pd.Timestamp,
                            target_alt_km: float,
                            lat_col='lat', lon_col='lon', alt_col='alt_km',
                            rho_obs_col='rho_obs') -> pd.DataFrame:
    """
    Scale the hourly-mean Swarm observed density from its mean altitude to 'target_alt_km'
    using an MSIS-based transfer factor computed at the Swarm hourly-mean location/time.

    Parameters
    ----------
    row : pd.DataFrame
        A DataFrame with exactly one row (the hourly mean), containing columns:
        lat_col, lon_col, alt_col, rho_obs_col.
    selected_hour : pd.Timestamp (UTC)
        The hour corresponding to 'row["hour"]'.
    target_alt_km : float
        The target altitude to scale to (e.g., GRACE hourly mean altitude at the same hour).
    lat_col, lon_col, alt_col, rho_obs_col : str
        Column names for latitude, longitude, altitude (km), and observed density.

    Returns
    -------
    pd.DataFrame
        The same row with added columns:
          - msis_rho_src: MSIS rho at Swarm mean altitude
          - msis_rho_tgt: MSIS rho at target altitude
          - scale_to_tgt: msis_rho_tgt / msis_rho_src
          - rho_obs_scaled_to_tgt: rho_obs * scale_to_tgt
    """
    import numpy as np
    from pymsis import msis

    if row.shape[0] != 1:
        raise ValueError("scale_swarm_hour_to_alt expects a single-row DataFrame (one hourly mean).")

    # Extract hourly-mean location and alt
    lat = float(row[lat_col].iloc[0])
    lon = float(row[lon_col].iloc[0])
    alt = float(row[alt_col].iloc[0])
    rho_obs = float(row[rho_obs_col].iloc[0])

    # MSIS at source (hourly Swarm mean altitude)
    out_src = msis.calculate([selected_hour], [lon], [lat], [alt])
    msis_rho_src = out_src[:, 0][0]

    # MSIS at target altitude (e.g., GRACE)
    out_tgt = msis.calculate([selected_hour], [lon], [lat], [float(target_alt_km)])
    msis_rho_tgt = out_tgt[:, 0][0]

    # Transfer factor and scaled observed density
    scale_to_tgt = msis_rho_tgt / max(msis_rho_src, 1e-30)
    rho_obs_scaled_to_tgt = rho_obs * scale_to_tgt

    # Attach results
    out = row.copy()
    out.loc[:, 'msis_rho_src'] = msis_rho_src
    out.loc[:, 'msis_rho_tgt'] = msis_rho_tgt
    out.loc[:, 'scale_to_tgt'] = scale_to_tgt
    out.loc[:, 'rho_obs_scaled_to_tgt'] = rho_obs_scaled_to_tgt
    out.loc[:, 'target_alt_km'] = float(target_alt_km)

    return out


# ------------------------------
# Model inference
# ------------------------------
COLS_TO_SCALE = [
    "f107", "ap_m6h", "lat", "f107a", "alt_km",
    "matched_tec_value", "ap_m3h", "vtec_matched_lag", "vtec_matched_lag2"
]

FEATURE_ORDER = [
    "f107a", "lat",
    "matched_tec_value",
    "lon_cos",
    "lon_sin", "lst_sin", "ap_m3h",
    "doy_sin", "doy_cos", "f107", "alt_km",
    "ap_m6h",
    "vtec_matched_lag", "vtec_matched_lag2",
    'lst_lat_sin'
]


def load_model_and_scalers(model_file: str, scaler_x_file: str, scaler_y_file: str):
    model = xgb.XGBRegressor()
    model.load_model(model_file)
    scaler_X = joblib.load(scaler_x_file)
    scaler_y = joblib.load(scaler_y_file)
    return model, scaler_X, scaler_y


def prepare_feature_matrix(df: pd.DataFrame, scaler_X) -> pd.DataFrame:
    X_to_scale = df[COLS_TO_SCALE]
    X_scaled = pd.DataFrame(scaler_X.transform(X_to_scale),
                            columns=COLS_TO_SCALE, index=X_to_scale.index)
    X_unscaled = df[[col for col in FEATURE_ORDER if col not in COLS_TO_SCALE]]
    X_final = pd.concat([X_scaled, X_unscaled], axis=1)[FEATURE_ORDER]
    assert list(X_final.columns) == FEATURE_ORDER, "Feature order mismatch!"
    return X_final


def predict_density(df: pd.DataFrame, model, scaler_y, X: pd.DataFrame) -> pd.DataFrame:
    y_pred_scaled = model.predict(X)
    y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    out_df = df.copy()
    out_df["y_pred_log"] = y_pred_unscaled
    out_df["rho_pred"] = out_df["msis_rho"] * np.exp(out_df["y_pred_log"])
    return out_df



def scale_swarm_hour_to_alt(row: pd.DataFrame,
                            selected_hour: pd.Timestamp,
                            target_alt_km: float,
                            lat_col='lat', lon_col='lon', alt_col='alt_km',
                            rho_obs_col='rho_obs') -> pd.DataFrame:
    """
    Scale the hourly-mean Swarm observed density from its mean altitude to 'target_alt_km'
    using an MSIS-based transfer factor computed at the Swarm hourly-mean location/time.

    Parameters
    ----------
    row : pd.DataFrame
        A DataFrame with exactly one row (the hourly mean), containing columns:
        lat_col, lon_col, alt_col, rho_obs_col.
    selected_hour : pd.Timestamp (UTC)
        The hour corresponding to 'row["hour"]'.
    target_alt_km : float
        The target altitude to scale to (e.g., GRACE hourly mean altitude at the same hour).
    lat_col, lon_col, alt_col, rho_obs_col : str
        Column names for latitude, longitude, altitude (km), and observed density.

    Returns
    -------
    pd.DataFrame
        The same row with added columns:
          - msis_rho_src: MSIS rho at Swarm mean altitude
          - msis_rho_tgt: MSIS rho at target altitude
          - scale_to_tgt: msis_rho_tgt / msis_rho_src
          - rho_obs_scaled_to_tgt: rho_obs * scale_to_tgt
    """
    import numpy as np
    from pymsis import msis

    if row.shape[0] != 1:
        raise ValueError("scale_swarm_hour_to_alt expects a single-row DataFrame (one hourly mean).")

    # Extract hourly-mean location and alt
    lat = float(row[lat_col].iloc[0])
    lon = float(row[lon_col].iloc[0])
    alt = float(row[alt_col].iloc[0])
    rho_obs = float(row[rho_obs_col].iloc[0])

    # MSIS at source (hourly Swarm mean altitude)
    out_src = msis.calculate([selected_hour], [lon], [lat], [alt])
    msis_rho_src = out_src[:, 0][0]

    # MSIS at target altitude (e.g., GRACE)
    out_tgt = msis.calculate([selected_hour], [lon], [lat], [float(target_alt_km)])
    msis_rho_tgt = out_tgt[:, 0][0]

    # Transfer factor and scaled observed density
    scale_to_tgt = msis_rho_tgt / max(msis_rho_src, 1e-30)
    rho_obs_scaled_to_tgt = rho_obs * scale_to_tgt

    # Attach results
    out = row.copy()
    out.loc[:, 'msis_rho_src'] = msis_rho_src
    out.loc[:, 'msis_rho_tgt'] = msis_rho_tgt
    out.loc[:, 'scale_to_tgt'] = scale_to_tgt
    out.loc[:, 'rho_obs_scaled_to_tgt'] = rho_obs_scaled_to_tgt
    out.loc[:, 'target_alt_km'] = float(target_alt_km)

    return out

# ------------------------------
# Plotting (kept as functions and only called if enabled)
# ------------------------------
def plot_msis_global(df: pd.DataFrame, lon_col="longitude", lat_col="latitude", value_col="msis_rho",
                      res_deg=5, cmap="turbo", title=None, save_path = "plot_gloval",
                      vmin=None, vmax=None): # <-- New Parameters Added Here

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np # <-- Added import for clarity, though it's likely available

    lon = ((df[lon_col] + 180) % 360) - 180
    lon = df[lon_col]
    lat = df[lat_col]
    values = df[value_col]

    # --- Data Gridding (Same as before) ---
    lon_edges = np.arange(-180, 180 + res_deg, res_deg)
    lat_edges = np.arange(-90, 90 + res_deg, res_deg)
    lon_idx = np.digitize(lon, lon_edges) - 1
    lat_idx = np.digitize(lat, lat_edges) - 1
    grid = np.full((len(lat_edges)-1, len(lon_edges)-1), np.nan)
    count = np.zeros_like(grid)

    for i, j, v in zip(lat_idx, lon_idx, values):
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            if np.isnan(grid[i, j]):
                grid[i, j] = v
            else:
                grid[i, j] += v
            count[i, j] += 1

    mask = count > 0
    grid[mask] /= count[mask]

    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    Lon, Lat = np.meshgrid(lon_c, lat_c)
    # --- End Data Gridding ---

    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.35)

    # --- Key Change: Pass vmin and vmax to pcolormesh ---
    h = ax.pcolormesh(
        Lon, Lat, grid,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,  # <--- Sets the minimum for the color scale
        vmax=vmax,
       # shading="gouraud",   #
    )
    # --------------------------------------------------

    plt.colorbar(h, ax=ax, shrink=0.7, pad=0.03, label="MSIS Neutral Density (kg/m³)")
    ax.set_title(title or "MSIS Neutral Density Global Map", pad=8)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.show()


def plot_difference_global(df: pd.DataFrame, lon_col="longitude", lat_col="latitude",
                           pred_col="rho_pred", msis_col="msis_rho",
                           res_deg=6, cmap="coolwarm", title=None, save_path="plot_diff"):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    diff_values = df[pred_col] - df[msis_col]
    lon = ((df[lon_col] + 180) % 360) - 180
    lat = df[lat_col]

    lon_edges = np.arange(-180, 180 + res_deg, res_deg)
    lat_edges = np.arange(-90, 90 + res_deg, res_deg)
    lon_idx = np.digitize(lon, lon_edges) - 1
    lat_idx = np.digitize(lat, lat_edges) - 1
    grid = np.full((len(lat_edges)-1, len(lon_edges)-1), np.nan)
    count = np.zeros_like(grid)

    for i, j, v in zip(lat_idx, lon_idx, diff_values):
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            if np.isnan(grid[i, j]):
                grid[i, j] = v
            else:
                grid[i, j] += v
            count[i, j] += 1

    mask = count > 0
    grid[mask] /= count[mask]

    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    Lon, Lat = np.meshgrid(lon_c, lat_c)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.35)
    h = ax.pcolormesh(Lon, Lat, grid, transform=ccrs.PlateCarree(), cmap=cmap)
    plt.colorbar(h, ax=ax, shrink=0.7, pad=0.03, label="Difference (rho_pred - msis_rho) [kg/m³]")
    ax.set_title(title or "Global Difference: Predicted vs MSIS", pad=8)
    plt.show() 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def compute_swarm_diffs_hourly_mean(result_df, scaled_swarm_row, cfg: Config):
    """
    Compute Swarm−Prediction and Swarm−MSIS at the hourly-mean Swarm location,
    strictly collocated to the nearest model grid cell (no interpolation).

    Returns a 1-row DataFrame with the differences and collocation info.
    """
    if scaled_swarm_row is None or scaled_swarm_row.shape[0] != 1:
        raise ValueError("scaled_swarm_row must be a single-row DataFrame.")

    pts = scaled_swarm_row[['lat','lon','rho_obs','rho_obs_scaled_to_tgt']].copy()
    pts.rename(columns={'lat':'lat','lon':'lon'}, inplace=True)

    coll = collocate_points_to_grid(
        points_df=pts,
        grid_df=result_df,
        lat_start=cfg.lat_start,
        lat_step=cfg.lat_step,
        lon_start=cfg.lon_start,
        lon_step=cfg.lon_step,
        lat_col='lat', lon_col='lon'
    )

    # Compute differences
    coll['diff_swarm_pred'] = coll['rho_obs_scaled_to_tgt'] - coll['rho_pred']
    coll['diff_swarm_msis'] = coll['rho_obs_scaled_to_tgt'] - coll['msis_rho']

    # Keep tidy columns
    out = coll.assign(
        swarm_lat=coll['lat'],
        swarm_lon=coll['lon'],
        grid_lat=coll['latitude_cell'],
        grid_lon=coll['longitude_cell']
    )[['swarm_lat','swarm_lon','grid_lat','grid_lon',
       'rho_obs','rho_obs_scaled_to_tgt','rho_pred','msis_rho',
       'diff_swarm_pred','diff_swarm_msis','alt_km']]

    return out

def compute_swarm_diffs_all_points(result_df, swarm_df_hour, target_alt_km, selected_hour, cfg: Config):
    """
    For every Swarm sample inside the selected hour:
    - Scale rho_obs to target_alt_km using your existing scale_swarm_hour_to_alt logic,
      but done per-sample (here we call MSIS for each row).
    - Hard-collocate to model grid cell without interpolation.
    - Compute differences: Swarm−Prediction and Swarm−MSIS.

    Returns a DataFrame with one row per Swarm sample in that hour.
    """
    from pymsis import msis

    if swarm_df_hour.empty:
        return pd.DataFrame()

    df = swarm_df_hour.copy()

    # Ensure time (for clarity)
    df['hour'] = pd.to_datetime(df['time']).dt.floor('H').dt.tz_localize('UTC')

    # MSIS at source (Swarm alt) and at target alt, per-point
    out_src = msis.calculate(df['hour'].tolist(),
                             df['lon'].astype(float).tolist(),
                             df['lat'].astype(float).tolist(),
                             df['alt_km'].astype(float).tolist())
    out_tgt = msis.calculate(df['hour'].tolist(),
                             df['lon'].astype(float).tolist(),
                             df['lat'].astype(float).tolist(),
                             [float(target_alt_km)]*len(df))

    df['msis_rho_src'] = out_src[:,0]
    df['msis_rho_tgt'] = out_tgt[:,0]
    df['scale_to_tgt'] = df['msis_rho_tgt'] / df['msis_rho_src'].clip(lower=1e-30)
    df['rho_obs_scaled_to_tgt'] = df['rho_obs'] * df['scale_to_tgt']

    # Collocate each point to nearest model grid cell (strict binning)
    cols = collocate_points_to_grid(
        points_df=df[['lat','lon','rho_obs','rho_obs_scaled_to_tgt']].copy(),
        grid_df=result_df,
        lat_start=cfg.lat_start, lat_step=cfg.lat_step,
        lon_start=cfg.lon_start, lon_step=cfg.lon_step,
        lat_col='lat', lon_col='lon'
    )

    # Add differences
    cols['diff_swarm_pred'] = cols['rho_obs_scaled_to_tgt'] - cols['rho_pred']
    cols['diff_swarm_msis'] = cols['rho_obs_scaled_to_tgt'] - cols['msis_rho']

    # Keep useful columns
    cols = cols.assign(
        swarm_lat=cols['lat'],
        swarm_lon=cols['lon'],
        grid_lat=cols['latitude_cell'],
        grid_lon=cols['longitude_cell'],
        target_alt_km=float(target_alt_km)
    )[['time','swarm_lat','swarm_lon','grid_lat','grid_lon',
       'rho_obs','rho_obs_scaled_to_tgt','rho_pred','msis_rho',
       'diff_swarm_pred','diff_swarm_msis','target_alt_km']]

    return cols

def plot_swarm_point_diffs_on_map(result_df, diffs_df, which='pred', title=None, save_path=None):
    """
    Draw the global background (msis_rho or rho_pred for context) and overlay
    Swarm point(s) colored by the chosen difference (no interpolation; points only).
    which: 'pred' -> (Swarm−Prediction), 'msis' -> (Swarm−MSIS)
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np

    # Background (reuse your cell-averaging approach)
    bg_field = 'rho_pred' if which == 'pred' else 'msis_rho'
    lon = (((result_df['longitude'] + 180) % 360) - 180).to_numpy()
    lon =  result_df['longitude']
    lon = result_df["longitude"] 
    lat = result_df['latitude'].to_numpy()
    val = result_df[bg_field].to_numpy()

    res_deg = 5
    lon_edges = np.arange(-180, 180 + res_deg, res_deg)
    lat_edges = np.arange(-90,   90 + res_deg, res_deg)
    lon_idx = np.digitize(lon, lon_edges) - 1
    lat_idx = np.digitize(lat, lat_edges) - 1

    grid = np.full((len(lat_edges)-1, len(lon_edges)-1), np.nan)
    count = np.zeros_like(grid)
    for i, j, v in zip(lat_idx, lon_idx, val):
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            if np.isnan(grid[i, j]):
                grid[i, j] = v
            else:
                grid[i, j] += v
            count[i, j] += 1
    mask = count > 0
    grid[mask] /= count[mask]

    Lon, Lat = np.meshgrid((lon_edges[:-1] + lon_edges[1:]) / 2,
                           (lat_edges[:-1] + lat_edges[1:]) / 2)

    # Choose difference values
    if which == 'pred':
        diff_vals = diffs_df['diff_swarm_pred'].to_numpy()
        label = "Swarm − Prediction [kg/m³]"
        default_title = "Swarm vs Prediction (no interpolation)"
    else:
        diff_vals = diffs_df['diff_swarm_msis'].to_numpy()
        label = "Swarm − MSIS [kg/m³]"
        default_title = "Swarm vs MSIS (no interpolation)"

    swarm_lon = diffs_df['swarm_lon'].to_numpy()
    swarm_lat = diffs_df['swarm_lat'].to_numpy()

    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot(1,1,1, projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.35)

    # Background
    h = ax.pcolormesh(Lon, Lat, grid, transform=ccrs.PlateCarree(), cmap='turbo')
    plt.colorbar(h, ax=ax, shrink=0.7, pad=0.03, label=f"{bg_field} [kg/m³]")

    # Points (colored by diff)
    sc = ax.scatter(swarm_lon, swarm_lat, c=diff_vals, s=60, cmap='coolwarm',
                    edgecolors='k', linewidths=0.6, transform=ccrs.PlateCarree())
    plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.06, label=label)

    ax.set_title(title or default_title, pad=8)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  



def plot_swarm_track_with_line(result_df, swarm_df, value_col="rho_obs_scaled_to_tgt", val="rho_pred",
                               title=None, save_path=None, s=20):
  

    """
    Plot global predicted density and Swarm points with a unified color scale.
    Removes black orbit lines for clarity.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np

    # Sort Swarm points by time for consistency
    swarm_df = swarm_df.sort_values("time")

    # Extract Swarm coordinates and values
    swarm_lon = swarm_df["lon"].to_numpy()
    print(swarm_lon)
    swarm_lon = ((swarm_df["lon"].to_numpy() + 180) % 360) - 180 
    swarm_lat = swarm_df["lat"].to_numpy()
    values = swarm_df[value_col].to_numpy()

    # Background grid from result_df
    lon = ((result_df["longitude"] + 180) % 360) - 180
   # lon = result_df["longitude"] 
    lat = result_df["latitude"]
    val = result_df[val]

    # Bin global grid for plotting
    res_deg = 5
    lon_edges = np.arange(-180, 180 + res_deg, res_deg)
    lat_edges = np.arange(-90, 90 + res_deg, res_deg)
    lon_idx = np.digitize(lon, lon_edges) - 1
    lat_idx = np.digitize(lat, lat_edges) - 1
    grid = np.full((len(lat_edges)-1, len(lon_edges)-1), np.nan)
    count = np.zeros_like(grid)
    for i, j, v in zip(lat_idx, lon_idx, val):
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            if np.isnan(grid[i, j]):
                grid[i, j] = v
            else:
                grid[i, j] += v
            count[i, j] += 1
    mask = count > 0
    grid[mask] /= count[mask]
    Lon, Lat = np.meshgrid((lon_edges[:-1] + lon_edges[1:]) / 2,
                            (lat_edges[:-1] + lat_edges[1:]) / 2)

    # Unified color scale across both datasets
    
    vmin = min(val.min(), values.min()) 
    vmax = max(val.max(), values.max()) 

    vmin = 1e-12
    vmax = 6e-12


    #vmin = 1E-12# min(val.min(), values.min()) 
    #vmax = 6E-12 #max(val.max(), values.max()) 
    # Plot
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.1, alpha=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.1, alpha=0.35)

    # Background (model prediction)
    h = ax.pcolormesh(Lon, Lat, grid, transform=ccrs.PlateCarree(), shading="gouraud",
                      cmap="turbo", vmin=vmin, vmax=vmax)

    # Swarm points (same color scale)
    sc = ax.scatter(swarm_lon, swarm_lat, c=values, s=s,
                    cmap="turbo", vmin=vmin, vmax=vmax,
                    edgecolors="k", linewidths=0.4,
                    transform=ccrs.PlateCarree()
                    )

    # Single colorbar for both layers
    plt.colorbar(h, ax=ax, shrink=0.7, pad=0.03,
                 label="Neutral Density [kg/m³]")
  #  plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.03,
   #              label="Neutral Density [kg/m³]")

    ax.set_title(title or "Swarm Track with Unified Colorbar", pad=8)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def extract_grace_last_days(
    grace_parquet: str,
    selected_time: datetime,
    days_back: int = 5,
    lat_col: str = "lat",
    lon_col: str = "lon",
    alt_col: str = "alt_km",
    rho_obs_col: str = "rho_obs",
) -> pd.DataFrame:
    """
    Return raw GRACE samples in [selected_time - days_back, selected_time)
    with standardized columns for overlay plotting:
      time, lat, lon, alt_km, rho_obs

    No scaling is performed.
    """
    df = pd.read_parquet(grace_parquet)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    t_end = pd.Timestamp(selected_time)
    t_start = t_end - pd.Timedelta(days=days_back)
    out = df[(df["time"] >= t_start) & (df["time"] < t_end)].copy()

    # Basic checks (optional)
    required = {lat_col, lon_col, alt_col, rho_obs_col, "time"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"Missing columns in GRACE parquet: {missing}")

    # Standardize names
    out.rename(
        columns={lat_col:"lat", lon_col:"lon", alt_col:"alt_km", rho_obs_col:"rho_obs"},
        inplace=True,
    )
    return out[["time", "lat", "lon", "alt_km", "rho_obs"]].sort_values("time")


def predict_on_grid(
    df_with_msis: pd.DataFrame,
    X: pd.DataFrame,
    model,
    y_scaler=None,
    target_kind: str = "rho_pred"  # "log_residual" | "log_density" | "density"
) -> pd.DataFrame:
    """
    Model-agnostic inference on the grid; composes rho_pred depending on target_kind.
    Use 'log_residual' for the bulk-trained core model (log(ρ/ρ_MSIS)).
    """
    import numpy as np
    y_pred = model.predict(X) if hasattr(model, "predict") else model(X)

    if y_scaler is not None:
        y_pred = y_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).reshape(-1)

    out = df_with_msis.copy()
    out["y_pred"] = np.array(y_pred).reshape(-1)

    if target_kind == "log_residual":
        out["rho_pred"] = out["msis_rho"] * np.exp(out["y_pred"])
    elif target_kind == "log_density":
        out["rho_pred"] = np.exp(out["y_pred"])
    elif target_kind == "density":
        out["rho_pred"] = out["msis_rho"] * np.exp(out["y_pred"])
    else:
        raise ValueError(f"Unknown target_kind: {target_kind}")

    return out



def plot_swarm_track_with_line2(result_df, swarm_df, value_col="rho_obs_scaled_to_tgt", val="rho_pred",
                               title=None, save_path=None):
    
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    from matplotlib.colors import Normalize

    # Sort Swarm points by time for consistency
    swarm_df = swarm_df.sort_values("time")
   
    # --- CRITICAL FIX: USE LONGITUDE DATA DIRECTLY (Input is [-180, 180]) ---
    swarm_lon = swarm_df["lon"].to_numpy()
    swarm_lat = swarm_df["lat"].to_numpy()
    values = swarm_df[value_col].to_numpy()

    # Background grid from result_df (assuming [-180, 180])
    lon = result_df["longitude"] 
    lat = result_df["latitude"]
    val = result_df[val]

    # --- 1. Bin global grid for plotting (Background Model Data) ---
    # (Gridding logic remains identical for the background)
    res_deg = 5
    lon_edges = np.arange(-180, 180 + res_deg, res_deg)
    lat_edges = np.arange(-90, 90 + res_deg, res_deg)
    lon_idx = np.digitize(lon, lon_edges) - 1
    lat_idx = np.digitize(lat, lat_edges) - 1
    grid = np.full((len(lat_edges)-1, len(lon_edges)-1), np.nan)
    count = np.zeros_like(grid)
    for i, j, v in zip(lat_idx, lon_idx, val):
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            if np.isnan(grid[i, j]):
                grid[i, j] = v
            else:
                grid[i, j] += v
            count[i, j] += 1
    mask = count > 0
    grid[mask] /= count[mask]
    Lon, Lat = np.meshgrid((lon_edges[:-1] + lon_edges[1:]) / 2,
                            (lat_edges[:-1] + lat_edges[1:]) / 2)

    # --- 2. Unified color scale across both datasets ---
    vmin = min(val.min(), values.min()) 
    vmax = max(val.max(), values.max()) 
    vmin = 1e-12
    vmax = 6e-12

    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=vmin, vmax=vmax)

    # --- 3. Swarm Track Segmentation ---
    # Finds breaks where the track crosses the IDL (e.g., from 179 to -179)
    # The jump should be around 358 degrees.
    lon_diff = np.abs(np.diff(swarm_lon))
    break_indices = np.where(lon_diff > 300)[0] + 1 
    
    segments = []
    start_index = 0
    for break_index in break_indices:
        segments.append((start_index, break_index))
        start_index = break_index
    segments.append((start_index, len(swarm_lon)))

    # --- 4. Plotting Setup ---
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.1, alpha=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.1, alpha=0.35)

    # --- 5. Plot Background (Model prediction) ---
    h = ax.pcolormesh(Lon, Lat, grid, transform=ccrs.PlateCarree(),
                      cmap=cmap, norm=norm)

    # --- 6. Plot Swarm Track (Observations) using plot and markers ---
    sc = None 
    for start, end in segments:
        segment_lon = swarm_lon[start:end]
        segment_lat = swarm_lat[start:end]
        segment_values = values[start:end]
        
        # Plot the segment points one by one to ensure accurate coloring
        for slon, slat, sval in zip(segment_lon, segment_lat, segment_values):
             sc = ax.scatter(slon, slat, c=[sval], s=40, # Retained original size for points
                             cmap=cmap, norm=norm,
                             edgecolors="k", linewidths=0.7,
                             transform=ccrs.PlateCarree(), zorder=10)
                        
    # --- 7. Single Colorbar for both layers ---
    # We reference 'h' (pcolormesh) for the colorbar, which uses the same norm/cmap
    plt.colorbar(h, ax=ax, shrink=0.7, pad=0.03,
                 label="Neutral Density [kg/m³]")

    ax.set_title(title or "Swarm Track with Unified Colorbar (Final Fix)", pad=8)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_swarm_track_with_line2(
    result_df,
    swarm_df,
    value_col="rho_obs_scaled_to_tgt",     # Swarm quantity to color
    val="rho_pred",                        # background column in result_df: 'rho_pred' or 'msis_rho'
    title=None,
    save_path=None,
    res_deg=5,
    fixed_limits=True,                     # e.g., (1e-12, 6e-12) or None for auto
    draw_line=True,                       # draw a thin orbit line in segments
    verbose=True                           # print diagnostics
):
    """
    Plot global background field and overlay Swarm points using ONE shared color scale.
    Includes thorough diagnostics and consistency checks.

    Parameters
    ----------
    result_df : pd.DataFrame
        Must include columns: 'longitude', 'latitude', and the background field named by `val`.
    swarm_df : pd.DataFrame
        Must include columns: 'time', 'lon', 'lat', and the Swarm value column `value_col`.
    value_col : str
        Column in swarm_df used for coloring the points (e.g., 'rho_obs_scaled_to_tgt').
    val : str
        Background field column in result_df ('rho_pred' or 'msis_rho').
    fixed_limits : tuple(float, float) or None
        If provided, use (vmin, vmax) for the color scale; otherwise auto from data.
    draw_line : bool
        If True, plots a thin orbit line, segmented at dateline crossings to prevent bridging.
    verbose : bool
        If True, prints diagnostics and checks.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import Normalize
    import warnings

    def normalize_lon(x):
        """Return longitudes in [-180, 180)."""
        x = np.asarray(x, dtype=float)
        return ((x + 180.0) % 360.0) - 180.0

    # ----------------------------
    # 0) Basic column existence checks
    # ----------------------------
    required_bg = {"longitude", "latitude", val}
    missing_bg = [c for c in required_bg if c not in result_df.columns]
    if missing_bg:
        raise KeyError(f"result_df missing columns: {missing_bg}")

    required_swarm = {"time", "lon", "lat", value_col}
    missing_swarm = [c for c in required_swarm if c not in swarm_df.columns]
    if missing_swarm:
        raise KeyError(f"swarm_df missing columns: {missing_swarm}")

    # ----------------------------
    # 1) Sort and extract arrays
    # ----------------------------
    swarm_df = swarm_df.sort_values("time").copy()

    # Wrap BOTH layers to [-180, 180)
    swarm_lon = normalize_lon(swarm_df["lon"].to_numpy())
    swarm_lat = np.asarray(swarm_df["lat"].to_numpy(), dtype=float)
    swarm_vals = np.asarray(swarm_df[value_col].to_numpy(), dtype=float)

    lon_bg = normalize_lon(np.asarray(result_df["longitude"].to_numpy(), dtype=float))
    lat_bg = np.asarray(result_df["latitude"].to_numpy(), dtype=float)
    bg_vals = np.asarray(result_df[val].to_numpy(), dtype=float)

    # ----------------------------
    # 2) Diagnostics: ranges, NaNs, sizes
    # ----------------------------
    if verbose:
        print("\n--- Diagnostics: Input ranges and NaNs ---")
        def _summ(name, arr):
            return (np.nanmin(arr), np.nanmax(arr), np.isnan(arr).sum(), arr.size)
        s_lon_min, s_lon_max, s_lon_nnan, s_lon_n = _summ("swarm lon", swarm_lon)
        s_lat_min, s_lat_max, s_lat_nnan, s_lat_n = _summ("swarm lat", swarm_lat)
        s_val_min, s_val_max, s_val_nnan, s_val_n = _summ("swarm vals", swarm_vals)

        b_lon_min, b_lon_max, b_lon_nnan, b_lon_n = _summ("bg lon", lon_bg)
        b_lat_min, b_lat_max, b_lat_nnan, b_lat_n = _summ("bg lat", lat_bg)
        b_val_min, b_val_max, b_val_nnan, b_val_n = _summ("bg vals", bg_vals)

        print(f"Swarm lon  min/max: {s_lon_min:.3f} .. {s_lon_max:.3f} | NaNs: {s_lon_nnan} / {s_lon_n}")
        print(f"Swarm lat  min/max: {s_lat_min:.3f} .. {s_lat_max:.3f} | NaNs: {s_lat_nnan} / {s_lat_n}")
        print(f"Swarm vals min/max: {s_val_min:.3e} .. {s_val_max:.3e} | NaNs: {s_val_nnan} / {s_val_n}")

        print(f"Backgr lon min/max: {b_lon_min:.3f} .. {b_lon_max:.3f} | NaNs: {b_lon_nnan} / {b_lon_n}")
        print(f"Backgr lat min/max: {b_lat_min:.3f} .. {b_lat_max:.3f} | NaNs: {b_lat_nnan} / {b_lat_n}")
        print(f"Backgr vals min/max: {b_val_min:.3e} .. {b_val_max:.3e} | NaNs: {b_val_nnan} / {b_val_n}")

    # Checks: longitude range consistency
    try:
        assert np.nanmin(swarm_lon) >= -180.0 and np.nanmax(swarm_lon) < 180.0, "Swarm longitudes not in [-180, 180)"
        assert np.nanmin(lon_bg)    >= -180.0 and np.nanmax(lon_bg)    < 180.0, "Background longitudes not in [-180, 180)"
    except AssertionError as e:
        raise AssertionError(f"[Longitude wrap] {e}. Ensure normalize_lon applied to BOTH layers.")

    # Warn on NaNs
    if np.isnan(swarm_vals).any():
        warnings.warn(f"[Swarm] {np.isnan(swarm_vals).sum()} NaNs in '{value_col}' — these points will be skipped.", RuntimeWarning)
    if np.isnan(bg_vals).any():
        warnings.warn(f"[Background] {np.isnan(bg_vals).sum()} NaNs in '{val}' — some grid cells may be empty.", RuntimeWarning)

    # ----------------------------
    # 3) Bin background to coarse grid
    # ----------------------------
    lon_edges = np.arange(-180, 180 + res_deg, res_deg)
    lat_edges = np.arange(-90,   90 + res_deg, res_deg)

    lon_idx = np.digitize(lon_bg, lon_edges) - 1
    lat_idx = np.digitize(lat_bg, lat_edges) - 1

    # Clip indices at boundaries to avoid dropping cells
    lon_idx = np.clip(lon_idx, 0, len(lon_edges) - 2)
    lat_idx = np.clip(lat_idx, 0, len(lat_edges) - 2)

    grid = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan)
    count = np.zeros_like(grid)

    # Fill grid
    for i, j, v in zip(lat_idx, lon_idx, bg_vals):
        if np.isnan(v):
            continue
        if np.isnan(grid[i, j]):
            grid[i, j] = v
        else:
            grid[i, j] += v
        count[i, j] += 1

    mask = count > 0
    grid[mask] /= count[mask]

    if verbose:
        filled = int(mask.sum())
        total  = int(mask.size)
        print(f"\n--- Binning summary ---\nFilled cells: {filled}/{total} ({filled/total:0.1%})")

    Lon, Lat = np.meshgrid(
        (lon_edges[:-1] + lon_edges[1:]) / 2.0,
        (lat_edges[:-1] + lat_edges[1:]) / 2.0
    )

    # ----------------------------
    # 4) Shared color scale
    # ----------------------------
    if fixed_limits is not None and len(fixed_limits) == 2:
        vmin, vmax = map(float, fixed_limits)
        if verbose:
            print(f"\n--- Color scale (fixed) ---\nvmin={vmin:.3e}, vmax={vmax:.3e}")
    else:
        # Auto from data (robust to NaNs)
        vmin = float(np.nanmin([np.nanmin(bg_vals), np.nanmin(swarm_vals)]))
        vmax = float(np.nanmax([np.nanmax(bg_vals), np.nanmax(swarm_vals)]))
        if verbose:
            print(f"\n--- Color scale (auto) ---\nvmin={vmin:.3e}, vmax={vmax:.3e}")
        # Sanity
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            warnings.warn("Invalid color limits detected; falling back to (1e-12, 6e-12).", RuntimeWarning)
            vmin, vmax = 1e-12, 6e-12

    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=vmin, vmax=vmax)

    # ----------------------------
    # 5) Segment the orbit line (optional)
    # ----------------------------
    segments = None
    if draw_line:
        lon_diff = np.abs(np.diff(swarm_lon))
        # Split anywhere a large jump occurs (dateline or data gap in lon)
        break_idx = np.where(lon_diff > 180.0)[0] + 1
        segments = []
        start_idx = 0
        for b in break_idx:
            segments.append((start_idx, b))
            start_idx = b
        segments.append((start_idx, len(swarm_lon)))
        if verbose:
            print(f"\n--- Track segmentation ---\nSegments: {len(segments)} | breaks at indices: {break_idx.tolist()}")

    # ----------------------------
    # 6) Plot
    # ----------------------------
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.1, alpha=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.1, alpha=0.35)

    # Background
    h = ax.pcolormesh(
        Lon, Lat, grid,
        transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm
    )

    # Swarm points (skip NaN values)
    ok = np.isfinite(swarm_vals)
    sc = ax.scatter(
        swarm_lon[ok], swarm_lat[ok],
        c=swarm_vals[ok], s=40, cmap=cmap, norm=norm,
        edgecolors="k", linewidths=0.7,
        transform=ccrs.PlateCarree(), zorder=10
    )

    # Optional orbit line (per segment)
    if draw_line and segments is not None:
        for s, e in segments:
            if e - s >= 2:
                ax.plot(
                    swarm_lon[s:e], swarm_lat[s:e],
                    color="k", linewidth=0.6, alpha=0.6,
                    transform=ccrs.PlateCarree(), zorder=9
                )

    # Single colorbar
    plt.colorbar(h, ax=ax, shrink=0.7, pad=0.03, label="Neutral Density [kg/m³]")

    ax.set_title(title or "Swarm Track with Unified Colorbar", pad=8)

    # Save OR show, then close
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"\nSaved figure to: {save_path}")
    else:
        plt.show()
        plt.close(fig)


def plot_swarm_track_with_line2(
    result_df,
    swarm_df,
    value_col="rho_obs_scaled_to_tgt",     # Swarm quantity to color
    val="rho_pred",                        # background column in result_df: 'rho_pred' or 'msis_rho'
    title=None,
    save_path=None,
    res_deg=5,
    fixed_limits=None,                     # e.g., (1e-12, 6e-12) or None for auto; do NOT pass True/False
    draw_line=True,                       # draw a thin orbit line in segments
    verbose=True                           # print diagnostics
):
    """
    Plot global background field and overlay Swarm points using ONE shared color scale.
    Includes diagnostics and consistency checks. Hardened to avoid 'len(bool)' errors.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import Normalize
    import warnings

    def normalize_lon(x):
        """Return longitudes in [-180, 180)."""
        x = np.asarray(x, dtype=float)
        return ((x + 180.0) % 360.0) - 180.0

    # ----------------------------
    # 0) Basic column existence checks
    # ----------------------------
    required_bg = {"longitude", "latitude", val}
    missing_bg = [c for c in required_bg if c not in result_df.columns]
    if missing_bg:
        raise KeyError(f"result_df missing columns: {missing_bg}")

    required_swarm = {"time", "lon", "lat", value_col}
    missing_swarm = [c for c in required_swarm if c not in swarm_df.columns]
    if missing_swarm:
        raise KeyError(f"swarm_df missing columns: {missing_swarm}")

    # ----------------------------
    # 1) Sort and extract arrays
    # ----------------------------
    swarm_df = swarm_df.sort_values("time").copy()

    # Wrap BOTH layers to [-180, 180)
    swarm_lon = normalize_lon(swarm_df["lon"].to_numpy())
    swarm_lat = np.asarray(swarm_df["lat"].to_numpy(), dtype=float)
    swarm_vals = np.asarray(swarm_df[value_col].to_numpy(), dtype=float)

    lon_bg = normalize_lon(np.asarray(result_df["longitude"].to_numpy(), dtype=float))
    lat_bg = np.asarray(result_df["latitude"].to_numpy(), dtype=float)
    bg_vals = np.asarray(result_df[val].to_numpy(), dtype=float)

    # ----------------------------
    # 2) Diagnostics: ranges, NaNs, sizes
    # ----------------------------
    if verbose:
        print("\n--- Diagnostics: Input ranges and NaNs ---")
        def _summ(arr):
            return (np.nanmin(arr), np.nanmax(arr), int(np.isnan(arr).sum()), int(arr.size))
        s_lon_min, s_lon_max, s_lon_nnan, s_lon_n = _summ(swarm_lon)
        s_lat_min, s_lat_max, s_lat_nnan, s_lat_n = _summ(swarm_lat)
        s_val_min, s_val_max, s_val_nnan, s_val_n = _summ(swarm_vals)

        b_lon_min, b_lon_max, b_lon_nnan, b_lon_n = _summ(lon_bg)
        b_lat_min, b_lat_max, b_lat_nnan, b_lat_n = _summ(lat_bg)
        b_val_min, b_val_max, b_val_nnan, b_val_n = _summ(bg_vals)

        print(f"Swarm lon  min/max: {s_lon_min:.3f} .. {s_lon_max:.3f} | NaNs: {s_lon_nnan} / {s_lon_n}")
        print(f"Swarm lat  min/max: {s_lat_min:.3f} .. {s_lat_max:.3f} | NaNs: {s_lat_nnan} / {s_lat_n}")
        print(f"Swarm vals min/max: {s_val_min:.3e} .. {s_val_max:.3e} | NaNs: {s_val_nnan} / {s_val_n}")

        print(f"Backgr lon min/max: {b_lon_min:.3f} .. {b_lon_max:.3f} | NaNs: {b_lon_nnan} / {b_lon_n}")
        print(f"Backgr lat min/max: {b_lat_min:.3f} .. {b_lat_max:.3f} | NaNs: {b_lat_nnan} / {b_lat_n}")
        print(f"Backgr vals min/max: {b_val_min:.3e} .. {b_val_max:.3e} | NaNs: {b_val_nnan} / {b_val_n}")

    # Checks: longitude range consistency
    if not (np.nanmin(swarm_lon) >= -180.0 and np.nanmax(swarm_lon) < 180.0):
        raise AssertionError("[Longitude wrap] Swarm longitudes not in [-180, 180).")
    if not (np.nanmin(lon_bg) >= -180.0 and np.nanmax(lon_bg) < 180.0):
        raise AssertionError("[Longitude wrap] Background longitudes not in [-180, 180).")

    # Warn on NaNs
    if np.isnan(swarm_vals).any():
        warnings.warn(f"[Swarm] {int(np.isnan(swarm_vals).sum())} NaNs in '{value_col}' — these points will be skipped.", RuntimeWarning)
    if np.isnan(bg_vals).any():
        warnings.warn(f"[Background] {int(np.isnan(bg_vals).sum())} NaNs in '{val}' — some grid cells may be empty.", RuntimeWarning)

    # ----------------------------
    # 3) Bin background to coarse grid
    # ----------------------------
    lon_edges = np.arange(-180, 180 + res_deg, res_deg)
    lat_edges = np.arange(-90,   90 + res_deg, res_deg)

    lon_idx = np.digitize(lon_bg, lon_edges) - 1
    lat_idx = np.digitize(lat_bg, lat_edges) - 1

    # Clip indices at boundaries to avoid dropping cells
    lon_idx = np.clip(lon_idx, 0, len(lon_edges) - 2)
    lat_idx = np.clip(lat_idx, 0, len(lat_edges) - 2)

    grid = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan)
    count = np.zeros_like(grid)

    # Fill grid
    for i, j, v in zip(lat_idx, lon_idx, bg_vals):
        if np.isnan(v):
            continue
        if np.isnan(grid[i, j]):
            grid[i, j] = v
        else:
            grid[i, j] += v
        count[i, j] += 1

    mask = count > 0
    grid[mask] /= count[mask]

    if verbose:
        filled = int(mask.sum())
        total  = int(mask.size)
        print(f"\n--- Binning summary ---\nFilled cells: {filled}/{total} ({filled/total:0.1%})")

    Lon, Lat = np.meshgrid(
        (lon_edges[:-1] + lon_edges[1:]) / 2.0,
        (lat_edges[:-1] + lat_edges[1:]) / 2.0
    )

    # ----------------------------
    # 4) Shared color scale (hardened)
    # ----------------------------
    # Decide color limits
    vmin = vmax = None
    use_fixed = False
    if fixed_limits is not None:
        # Accept tuple/list/array of length 2; reject bools or other scalars
        if isinstance(fixed_limits, (tuple, list, np.ndarray)) and len(fixed_limits) == 2:
            vmin, vmax = map(float, fixed_limits)
            use_fixed = True
        else:
            warnings.warn(
                f"[Color scale] 'fixed_limits' must be a 2-element tuple/list/array, "
                f"got type={type(fixed_limits).__name__} value={fixed_limits}. Falling back to auto.",
                RuntimeWarning
            )

    if not use_fixed:
        # Auto from data (robust to NaNs)
        vmin = float(np.nanmin([np.nanmin(bg_vals), np.nanmin(swarm_vals)]))
        vmax = float(np.nanmax([np.nanmax(bg_vals), np.nanmax(swarm_vals)]))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            warnings.warn("[Color scale] Invalid auto limits; using fallback (1e-12, 6e-12).", RuntimeWarning)
            vmin, vmax = 1e-12, 6e-12

    if verbose:
        mode = "fixed" if use_fixed else "auto"
        print(f"\n--- Color scale ({mode}) ---\nvmin={vmin:.3e}, vmax={vmax:.3e}")

    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=vmin, vmax=vmax)

    # ----------------------------
    # 5) Segment the orbit line (optional)
    # ----------------------------
    segments = None
    if draw_line:
        lon_diff = np.abs(np.diff(swarm_lon))
        # Split anywhere a large jump occurs (dateline or data gap in lon)
        break_idx = np.where(lon_diff > 180.0)[0] + 1
        segments = []
        start_idx = 0
        for b in break_idx:
            segments.append((start_idx, b))
            start_idx = b
        segments.append((start_idx, len(swarm_lon)))
        if verbose:
            print(f"\n--- Track segmentation ---\nSegments: {len(segments)} | breaks at indices: {break_idx.tolist()}")

    # ----------------------------
    # 6) Plot
    # ----------------------------
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.1, alpha=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.1, alpha=0.35)

    # Background
    h = ax.pcolormesh(
        Lon, Lat, grid,
        transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm
    )

    # Swarm points (skip NaN values)
    ok = np.isfinite(swarm_vals) & np.isfinite(swarm_lon) & np.isfinite(swarm_lat)
    sc = ax.scatter(
        swarm_lon[ok], swarm_lat[ok],
        c=swarm_vals[ok], s=40, cmap=cmap, norm=norm,
        edgecolors="k", linewidths=0.7,
        transform=ccrs.PlateCarree(), zorder=10
    )

    # Optional orbit line (per segment)
    if draw_line and segments is not None:
        for s, e in segments:
            if e - s >= 2:
                ax.plot(
                    swarm_lon[s:e], swarm_lat[s:e],
                    color="k", linewidth=0.6, alpha=0.6,
                    transform=ccrs.PlateCarree(), zorder=9
                )

    # Single colorbar
    plt.colorbar(h, ax=ax, shrink=0.7, pad=0.03, label="Neutral Density [kg/m³]")

    ax.set_title(title or "Swarm Track with Unified Colorbar", pad=8)

    # Save OR show, then close
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"\nSaved figure to: {save_path}")
    else:
        plt.show()
        plt.close(fig)

    # ------------------------------
# Orchestration
# ------------------------------

def main(cfg: Config = Config()):
    # 1) Load & aggregate GRACE
    print("Loading GRACE hourly data ...")
    hourly_df = load_grace_hourly(cfg.grace_parquet)
    print(f"Total hourly records: {len(hourly_df)}")

    # 2) Select altitude for the chosen timestamp
    alt_km = get_altitude_from_hour(hourly_df, cfg.selected_time_utc)
    print(f"Selected altitude (km): {alt_km}")

    # 3) Build global grid with features
    lat_range = np.arange(cfg.lat_start, cfg.lat_stop, cfg.lat_step)
    lon_range = np.arange(cfg.lon_start, cfg.lon_stop, cfg.lon_step)
    grid_df = build_global_feature_grid(cfg.selected_time_utc, alt_km, lat_range, lon_range)
    print(f"Global grid created with {len(grid_df)} points")

    # 4) Load TEC for epoch & merge to grid
    print("Merging TEC values for selected epoch ...")

# Load TEC epoch data
    df_tec = pd.read_parquet(cfg.tec_parquet)
    df_tec['epoch'] = pd.to_datetime(df_tec['epoch'], utc=True)
    tec_epoch = df_tec[df_tec['epoch'] == pd.Timestamp(cfg.selected_time_utc)]

# Interpolate TEC to finer grid
    interp_tec = interpolate_tec_to_grid(tec_epoch, lat_range, lon_range)

    interp_tec['vtec_matched_lag'] = interp_tec['tec_value']  # placeholder or compute lag
    interp_tec['vtec_matched_lag2'] = interp_tec['tec_value']  # placeholder or compute lag
    interp_tec['matched_tec_value'] = interp_tec['tec_value']

# Merge interpolated TEC with grid
    grid_with_tec = grid_df.merge(interp_tec, on=['latitude', 'longitude'], how='left')

    #grid_with_tec = load_tec_epoch_and_merge(grid_df, cfg.tec_parquet, cfg.selected_time_utc)
    print(f"Rows after TEC merge: {len(grid_with_tec)}")

    # 5) Add space weather params and MSIS density
    print("Fetching space weather and running MSIS ...")
    grid_with_msis = add_space_weather_and_msis(grid_with_tec, cfg.selected_time_utc)

    # 6) Load model & scalers
    print("Loading model and scalers ...")
    model, scaler_X, scaler_y = load_model_and_scalers(cfg.model_file, cfg.scaler_x_file, cfg.scaler_y_file)

    # 7) Prepare features and predict
    print("Preparing features and predicting ...")
    X_final = prepare_feature_matrix(grid_with_msis, scaler_X)
    result_df = predict_density(grid_with_msis, model, scaler_y, X_final)

    # 8) Plot (optional)
    if cfg.plot_results:
        try:
            plot_msis_global(result_df, title=f"Rho msis {alt_km:.0f} km", value_col="msis_rho", save_path = "plot_grace_globalmsis", vmin=1E-12, vmax=6E-12)
            plot_msis_global(result_df, title=f"Rho pred {alt_km:.0f} km", value_col="rho_pred", save_path = "plot_grace_pred", vmin=1E-12, vmax=6E-12)
           
            plot_difference_global(result_df, title="Predicted - MSIS Difference")
        except Exception as e:
            print(f"Plotting skipped due to error: {e}")

    # 9) Return / save


    print("Performing Swarm validation ...")
    try:
        swarm_path = "swarm_dns_with_tnd_y2001516_v1_0309.parquet"  # Adjust path if needed
        swarm_df = pd.read_parquet(swarm_path)
        swarm_df["longitude"]= swarm_df["lon"]
        swarm_df["latitude"]= swarm_df["lat"]
        swarm_df['hour'] = swarm_df['time'].dt.floor('H')

        swarm_df['hour'] = pd.to_datetime(swarm_df['hour'], utc=True)
        selected_hour = cfg.selected_time_utc
        hourly_data = swarm_df[swarm_df['hour'] == selected_hour]

        if hourly_data.empty:
            print(f"No Swarm data found for {selected_hour}")
            scaled_swarm = None
        else:
            # 2. Aggregate the hourly data into a single row (the "hourly mean")
            # We take the mean for all relevant numerical columns.
            mean_row = hourly_data.agg({
                'lat': 'mean',
                'lon': 'mean',
                'alt_km': 'mean',
                'rho_obs': 'mean'
            }).to_frame().T # .T makes it a single-row DataFrame

            # Ensure the required column 'hour' is present for the function (if it uses it)
            mean_row['hour'] = selected_hour 

            # 3. Pass the single-row DataFrame to the scaling function
            scaled_swarm = scale_swarm_hour_to_alt(
                mean_row, # Pass the single-row DataFrame
                selected_hour,
                target_alt_km=alt_km,
                lat_col='lat', lon_col='lon', alt_col='alt_km', rho_obs_col='rho_obs'
            )
            print("Swarm validation completed:")
            print(scaled_swarm[['rho_obs', 'rho_obs_scaled_to_tgt', 'scale_to_tgt', 'target_alt_km']])
    except Exception as e:
        print(f"Swarm validation failed: {e}")
        scaled_swarm = None

    





    print("Performing Swarm validation (no aggregation) ...")
    scaled_swarm = None
    try:
        swarm_path = "swarm_dns_with_tnd_y2001516_v1_0309.parquet"  # adjust if needed
        swarm_df = pd.read_parquet(swarm_path)
        swarm_df['time'] = pd.to_datetime(swarm_df['time'], utc=True)
        swarm_df['hour'] = swarm_df['time'].dt.floor('H')

        selected_hour = cfg.selected_time_utc

    # All rows whose time falls in the selected hour
        df_hour = swarm_df[swarm_df['hour'] == selected_hour]
        if df_hour.empty:
            print(f"No Swarm samples found in hour {selected_hour}")
        else:
            scaled_swarm = scale_swarm_hour_to_alt_many(
                df_hour,
                selected_hour=selected_hour,
                target_alt_km=alt_km,
                lat_col='lat', lon_col='lon', alt_col='alt_km', rho_obs_col='rho_obs',
                batch_size=20000  # tune if needed
            )
            print(f"Scaled {len(scaled_swarm)} Swarm samples to {alt_km:.1f} km.")
        # Quick preview
            print(scaled_swarm[['time','lat','lon','alt_km','rho_obs','rho_obs_scaled_to_tgt']].head())
    except Exception as e:
        print(f"Swarm validation failed: {e}")


    if cfg.plot_results:
        try:
            scaled_swarm["longitude"]= scaled_swarm["lon"]
            scaled_swarm["latitude"]= scaled_swarm["lat"]
            plot_msis_global(scaled_swarm, title=f"Plot_swarm_Rho obs {alt_km:.0f} km", value_col="rho_obs_scaled_to_tgt", save_path = "plot_swarm_global_scale", vmin=1E-12, vmax=6E-12)
            plot_msis_global(scaled_swarm, title=f"Plot_swarm_Rho obs normal km", value_col="rho_obs", save_path = "plot_swarm_global_unscale", vmin=1E-12, vmax=6E-12)

            plot_swarm_track_with_line(
                result_df=result_df,          # Background prediction grid
                swarm_df=scaled_swarm,        # Swarm scaled data
                value_col="rho_obs_scaled_to_tgt",  # Color by scaled density
                title=f"Swarm Track at {alt_km:.0f} km",
                save_path="plot_swarm_global_scale_line_pred", 
                val="rho_pred",
            )
            plot_swarm_track_with_line(
                result_df=result_df,          # Background prediction grid
                swarm_df=scaled_swarm,        # Swarm scaled data
                value_col="rho_obs_scaled_to_tgt",  # Color by scaled density
                title=f"Swarm Track at {alt_km:.0f} km",
                save_path="plot_swarm_global_scale_line_msis", 
                val="msis_rho",
            )
            #plot_difference_global(result_df, title="Predicted - MSIS Difference")
        except Exception as e:
            print(f"Plotting skipped due to error: {e}")    

    
# 6) Load the bulk-trained CORE model (pre-forecast baseline)
    print("Loading CORE (bulk-trained) model and scalers ...")
    core_model, core_scaler_X, core_scaler_y = load_model_and_scalers(
        model_file=cfg.model_file_core,         # path to your bulk-trained core model
        scaler_x_file=cfg.scaler_x_file,   # X-scaler used during core training
        scaler_y_file=cfg.scaler_y_file    # y-scaler used during core training
    )

# 7) Prepare features for the core model
# Option A: use your existing helper (keeps FEATURE_ORDER & scaling aligned)
    X_core = prepare_feature_matrix(grid_with_msis, core_scaler_X)

# Option B: if you need an alternative order: X_core = grid_with_msis[ALT_FEATURE_ORDER]

# 8) Predict using CORE (pre-forecast), composing rho_pred from msis_rho
    print("Predicting on grid using CORE model (pre-forecast baseline) ...")
    result_df2 = predict_on_grid(
        df_with_msis=grid_with_msis,
        X=X_core,
        model=core_model,
        y_scaler=core_scaler_y,
        target_kind="density"   # core trained on log(ρ/ρ_MSIS)    
    )

# --- GRACE last-5-days overlay (NO SCALING) to illustrate training data used by CORE ---
    try:
        print("Preparing GRACE last-5-days overlay (no scaling) ...")
        grace_last5 = extract_grace_last_days(
                grace_parquet=cfg.grace_parquet,
                selected_time=cfg.selected_time_utc,
                days_back=1,
                lat_col="lat", lon_col="lon", alt_col="alt_km", rho_obs_col="rho_obs"
        )
        print(grace_last5)
        grace_last5_ds = grace_last5.iloc[::20].copy()
        if cfg.plot_results:
        # Overlay raw GRACE points on CORE prediction background
                plot_swarm_track_with_line(
                    result_df=result_df2,                 # background: CORE grid
                    swarm_df=grace_last5,               # using GRACE points (no scaling)
                    value_col="rho_obs",                # color by raw GRACE density
                    title=f"GRACE Track (last 1 days) over CORE Prediction ({alt_km:.0f} km)",
                    save_path="plot_grace_last1_on_core",
                    val="rho_pred",
                    s=4                      # background field from CORE
              )
    except Exception as e:
        print(f"GRACE overlay (no scaling) skipped due to error: {e}")

# Save outputs
    result_df2.to_csv('result_df_core.csv', index=False)
        

# Return both prediction results and Swarm validation
    result_df.to_csv('result_df.csv')
    scaled_swarm.to_csv('scaled_swarm.csv')
    return result_df, scaled_swarm

    # Example: scale Swarm hourly mean to GRACE altitude at a single time
   


if __name__ == "__main__":
    _ = main()
