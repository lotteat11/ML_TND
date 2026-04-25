"""
pymsis_utils.py
- Dask-based helpers for loading GRACE data and joining space weather indices.
- Fetches and interpolates F10.7 and Ap to hourly cadence from the spaceweather package.
- Wraps pymsis.calculate for fly-through and fixed-altitude MSIS runs.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import dask.dataframe as dd   # switched from pandas
import spaceweather as sw
import pymsis
import pandas as pd
import requests
import io



def load_grace_csv(infile: Path) -> dd.DataFrame:
    """
    1) Read and validate the GRACE density Parquet/CSV into Dask DataFrame.
    Ensures required columns exist and 'time' is tz-aware UTC.
    """
    print("Working directory:", os.getcwd())
    print("Files here:", os.listdir('.'))

    print("Read")
    df = dd.read_parquet(infile, engine="pyarrow")
    # df = dd.read_csv(infile, parse_dates=["time"])

    needed = {"time", "lat", "lon", "alt_km"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # make tz-aware UTC (do on-the-fly; will stay lazy until compute)
    df["time"] = dd.to_datetime(df["time"], utc=True)
    return df


def fetch_spaceweather_indices(tmin, tmax):
    sw_daily = sw.sw_daily(update=True)
    
    sw_daily.index = pd.to_datetime(sw_daily.index)
    if sw_daily.index.tz is None:
        sw_daily.index = sw_daily.index.tz_localize("UTC")
    else:
        sw_daily.index = sw_daily.index.tz_convert("UTC")

    sw_daily = sw_daily.loc[tmin:tmax].copy()
    if sw_daily.empty:
        raise RuntimeError("No space weather data returned.")

    colmap = {"f107_obs": "f107", "f107_81ctr_adj": "f107a", "Apavg": "ap"}
    sw_daily = sw_daily.rename(columns=colmap)[["f107", "f107a", "ap"]]
    return sw_daily




def prepare_inputs_for_pymsis(df: dd.DataFrame) -> Dict[str, Any]:
    """
    4) Compute dask → numpy arrays for pymsis.
    """
    # materialize here
    df = df.compute()

    dates_np = df["time"].values.astype("datetime64[ns]")
    lons_np = df["lon"].to_numpy(dtype=float)
    lats_np = df["lat"].to_numpy(dtype=float)
    alts_np = df["alt_km"].to_numpy(dtype=float)

    f107s = df["f107"].to_numpy(dtype=float).tolist()
    f107as = df["f107a"].to_numpy(dtype=float).tolist()
    aps = [[float(a)] * 7 for a in df["Ap"].to_numpy(dtype=float)]

    return {
        "dates_np": dates_np,
        "lons_np": lons_np,
        "lats_np": lats_np,
        "alts_np": alts_np,
        "f107s": f107s,
        "f107as": f107as,
        "aps": aps,
    }


def run_msis(inputs: Dict[str, Any], version: float = 2.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    5) Run MSIS and return (full_output, total_neutral_density).
    """
    data = pymsis.calculate(
        inputs["dates_np"],
        inputs["lons_np"],
        inputs["lats_np"],
        inputs["alts_np"],
     #   inputs["f107s"],
      #  inputs["f107as"],
        geomagnetic_activity=-1,
        version=version,
    )

    data = np.asarray(data)
    tnd = data[:, 0]
    return data, tnd

def join_indices2(df: dd.DataFrame, sw_hourly: pd.DataFrame) -> dd.DataFrame:
    """
    Join hourly (or 3-hourly) space weather indices onto GRACE Dask DataFrame.
    
    Parameters
    ----------
    df : dd.DataFrame
        GRACE dataframe with 'time' column.
    sw_hourly : pd.DataFrame
        Pandas dataframe of space weather, indexed by hourly timestamps in UTC.
        Example columns: ['f107', 'f107a', 'ap'].
    
    Returns
    -------
    dd.DataFrame
        Original GRACE df with hourly indices joined and filled across partitions.
    """
    # Ensure UTC + datetime
    df = df.assign(time=dd.to_datetime(df["time"], utc=True))

    # Align SW index: force to UTC, floor to 1H
    sw_hourly = sw_hourly.copy()
    sw_hourly.index = pd.to_datetime(sw_hourly.index, utc=True).floor("H")

    # Build hour key in df
    df = df.assign(hour_utc=df["time"].dt.floor("H"))

    # Merge on hour
    df = df.merge(sw_hourly, left_on="hour_utc", right_index=True, how="left")

    # Partition-safe sort and fill
    df = (
        df.set_index("time", sorted=False, shuffle="tasks")
          .map_partitions(lambda pdf: pdf.sort_index())
    )
    df = df.ffill().bfill().reset_index()  # put 'time' back as column

    return df

def run_msis_400(inputs: Dict[str, Any], version: float = 2.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    5) Run MSIS and return (full_output, total_neutral_density).
    """
 
    n_points = len(inputs["dates_np"])

# Create an array of 400.0 km repeated for each input point
    alt_400 = np.full(n_points, 400.0)
    data_400 = pymsis.calculate(
        inputs["dates_np"],
        inputs["lons_np"],
        inputs["lats_np"],
        alt_400,
       # inputs["f107s"],
       # inputs["f107as"],
      #  inputs["aps"],
        geomagnetic_activity=-1, 
        version=version,
    )
    data_400 = np.asarray(data_400)
    tnd_400 = data_400[:, 0]

    data = pymsis.calculate(
        inputs["dates_np"],
        inputs["lons_np"],
        inputs["lats_np"],
        inputs["alts_np"],
        #inputs["f107s"],
        #inputs["f107as"],
        geomagnetic_activity=-1, 
        version=version,
    )

    data = np.asarray(data)
    tnd = data[:, 0]

    return data, tnd, data_400, tnd_400

def join_indices(df: dd.DataFrame, sw_daily) -> dd.DataFrame:
    """
    3) Join daily indices to each row by UTC calendar date.
    """
    df = df.assign(date_utc=df["time"].dt.normalize())
    df = df.merge(sw_daily, left_on="date_utc", right_index=True, how="left")

    # fillna not always lazy → enforce later at compute-time
    df = df.ffill().bfill()
    return df

def fetch_spaceweather_hourly(tmin: pd.Timestamp,
                              tmax: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch F10.7 (daily), F10.7a (81-day avg), and Ap/Kp (3-hourly) indices,
    combine into one hourly dataframe over [tmin, tmax].
    
    Returns
    -------
    sw_hourly : pd.DataFrame
        Columns: f107, f107a, ap, kp
        Indexed by hourly UTC time.
    """

    # --- 1) Daily f107, f107a ---
    sw_daily = sw.sw_daily(update=True)
    sw_daily.index = pd.to_datetime(sw_daily.index).tz_localize("UTC")
    sw_daily = sw_daily.loc[tmin.floor("D"):tmax.ceil("D")].copy()
    sw_daily = sw_daily.rename(columns={
        "f107_obs": "f107",
        "f107_81ctr_adj": "f107a",
        "Apavg": "ap_daily"
    })[["f107", "f107a", "ap_daily"]]

    # Broadcast to hourly
    hourly_index = pd.date_range(
        sw_daily.index.min(),
        sw_daily.index.max(),
        freq="1H", tz="UTC"
    )
    sw_daily_hourly = sw_daily.reindex(hourly_index, method="ffill")

    # --- 2) 3-hourly Ap/Kp ---
    sw_apkp = sw.ap_kp_3h(update=True)
    sw_apkp.index = pd.to_datetime(sw_apkp.index, utc=True)

    # Resample 3-hourly → hourly with ffill
    ap_kp_hourly = sw_apkp.resample("1H").ffill()

    # --- 3) Merge both (align on index) ---
    sw_hourly = pd.concat([sw_daily_hourly, ap_kp_hourly], axis=1)

    # --- 4) Final cut to [tmin, tmax] ---
    sw_hourly = sw_hourly.loc[tmin:tmax]

    return sw_hourly
def fetch_spaceweather_indices_hourly(tmin: pd.Timestamp,
                                      tmax: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch daily space weather indices from spaceweather.sw_daily
    and expand to hourly cadence (broadcast).
    """
    import spaceweather as sw

    # Daily
    sw_daily = sw.sw_daily(update=True)
    sw_daily.index = pd.to_datetime(sw_daily.index).tz_localize("UTC")

    # Slice range
    sw_daily = sw_daily.loc[tmin.floor("D"):tmax.ceil("D")].copy()
    if sw_daily.empty:
        raise RuntimeError("No daily space weather data for requested range.")

    # Rename/keep expected cols
    colmap = {"f107_obs": "f107", "f107_81ctr_adj": "f107a", "Apavg": "ap"}
    sw_daily = sw_daily.rename(columns=colmap)[["f107", "f107a", "ap"]]

    # Expand to hourly → reindex on hourly range and forward-fill
    hourly_index = pd.date_range(sw_daily.index.min(),
                                 sw_daily.index.max() + pd.Timedelta(days=1),
                                 freq="1H", tz="UTC")

    sw_hourly = sw_daily.reindex(hourly_index, method="ffill")
    sw_hourly.index.name = "time"

    # Cut back to requested window
    sw_hourly = sw_hourly.loc[tmin:tmax]

    return sw_hourly


def fetch_spaceweather_hourly_robust(tmin: pd.Timestamp,
                              tmax: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch F10.7 (daily), F10.7a (81-day avg), and Ap/Kp (3-hourly) indices,
    and combine into one hourly dataframe over [tmin, tmax].
    
    This version uses both ffill and bfill to handle data gaps more robustly.
    
    Returns
    -------
    sw_hourly : pd.DataFrame
        Columns: f107, f107a, ap, kp
        Indexed by hourly UTC time.
    """

    # --- 1) Daily f107, f107a ---
    sw_daily = sw.sw_daily(update=True)
    sw_daily.index = pd.to_datetime(sw_daily.index).tz_localize("UTC")
    sw_daily = sw_daily.loc[tmin.floor("D"):tmax.ceil("D")].copy()
    sw_daily = sw_daily.rename(columns={
        "f107_obs": "f107",
        "f107_81ctr_adj": "f107a",
        "Apavg": "ap_daily"
    })[["f107", "f107a", "ap_daily"]]

    # Broadcast to hourly using ffill and bfill
    hourly_index = pd.date_range(
        sw_daily.index.min(),
        sw_daily.index.max(),
        freq="1H", tz="UTC"
    )
    sw_daily_hourly = sw_daily.reindex(hourly_index).ffill().bfill()

    # --- 2) 3-hourly Ap/Kp ---
    sw_apkp = sw.ap_kp_3h(update=True)
    sw_apkp.index = pd.to_datetime(sw_apkp.index, utc=True)

    # Resample 3-hourly → hourly with ffill and bfill
    ap_kp_hourly = sw_apkp.resample("1H").ffill().bfill()

    # --- 3) Merge both (align on index) ---
    sw_hourly = pd.concat([sw_daily_hourly, ap_kp_hourly], axis=1)

    # --- 4) Final cut to [tmin, tmax] ---
    sw_hourly = sw_hourly.loc[tmin:tmax]
    
    return sw_hourly

def run_msis_v2(inputs: Dict[str, Any], version: float = 2.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run MSIS and return:
      data : full output (shape (N,11) in fly-through OR (nd,nlon,nlat,nalt,11) in grid)
      rho  : total mass density (kg/m^3), i.e. data[..., 0]
    """
    dates = inputs["dates_np"]
    lons  = inputs["lons_np"]
    lats  = inputs["lats_np"]
    alts  = inputs["alts_np"]

    kwargs: Dict[str, Any] = {"version": version}
    if inputs.get("f107s")  is not None: kwargs["f107s"]  = inputs["f107s"]
    if inputs.get("f107as") is not None: kwargs["f107as"] = inputs["f107as"]
    if inputs.get("aps")    is not None:
        kwargs["aps"] = inputs["aps"]
        # daily Ap -> geomagnetic_activity=1; 7 components -> storm-time
        if np.shape(inputs["aps"])[-1] == 7:
            kwargs["geomagnetic_activity"] = -1
        else:
            kwargs["geomagnetic_activity"] = 1

    data = pymsis.calculate(dates=dates, lons=lons, lats=lats, alts=alts, **kwargs)
    data = np.asarray(data)
    rho  = data[..., 0]  # MASS_DENSITY

    return data, rho