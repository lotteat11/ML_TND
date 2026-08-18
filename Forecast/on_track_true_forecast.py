# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
on_track_true_forecast.py — FULLY OPERATIONAL variant of on_track_persisted.py.

Motivation:
on_track_persisted.py replaces the ML feature drivers in the forecast window
with issue-time (persisted) values, but the final density prediction is still
rho_pred = msis_rho * exp(correction), where msis_rho was generated with
OBSERVED drivers on the forecast days. Future driver knowledge therefore still
leaks in through the MSIS scaffold, and the MSIS baseline column remains a
perfect-driver upper bound. Neither column is a true issue-time product.

This script closes that gap. For every forecast window it ALSO re-runs
NRLMSISE-2.1 with persisted drivers:

  - f107, f107a : last value available in the day t-1 data, held constant
                  over the window (same information set as the ML features).
  - ap          : last observed 3-hourly ap at issue time, used for all seven
                  ap components (daily Ap + 3-hourly history). The stored
                  msis_rho was generated in pymsis' default daily-Ap mode, so
                  only the daily-Ap slot is active — passing all seven keeps
                  the call mode-independent.
  - geometry    : the forecast points' own (time, lat, lon, alt) — satellite
                  position is predictable, so these are known at issue time.

The prediction becomes  rho_pred = msis_persist * exp(correction_persisted),
compared against msis_persist as the OPERATIONAL baseline — both columns are
now genuine issue-time products. The stored perfect-driver msis_rho and the
observed-scaffold prediction are kept as reference columns.

CAVEATS (documented, not fixed here):
  - f107a is an 81-day CENTRED average, so even the "persisted" value contains
    ~40 future days. It varies slowly and this is standard practice, but a
    trailing average would be the fully strict alternative.
  - The correction model was trained on residuals w.r.t. observed-driver MSIS;
    applying it to persisted-driver MSIS is a (small) train/deploy mismatch.

A startup sanity check verifies that pymsis with the dataframe's own observed
driver columns reproduces the stored msis_rho, so the persisted run differs
from the original generation ONLY in the driver values.

Run from the repo root:   python Forecast/on_track_true_forecast.py
Quick validation:         python Forecast/on_track_true_forecast.py --smoke
Outputs go to runs_true_forecast/ (runs/ and runs_persisted/ untouched).
"""

# %% --------------------------------- IMPORTS ---------------------------------
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import argparse
import numpy as np
import pandas as pd
import joblib
import pymsis
import xgboost as xgb
from scipy.spatial import cKDTree

import feature_functions as ff

# %% ----------------------------- CONFIGURATION --------------------------------
TARGET_COL = "log_ratio"

MODEL_FILE = "xgb_model_v3.json"
SCALER_X_FILE = "scaler_xgboost_X_v3.joblib"
SCALER_Y_FILE = "scaler_xgboost_y_v3.joblib"
DATA_FILE = "grace_data_merged2.parquet"

OUTPUT_ROOT = "runs_true_forecast"

# Weight of the LST coordinate (in "degrees of latitude" per radian of LST
# angle) in the nearest-geometry lookup. The two LST planes are ~12 h apart,
# so any moderate weight cleanly separates ascending/descending passes.
LST_KDTREE_WEIGHT = 20.0

TEC_COLS = ["matched_tec_value", "vtec_matched_lag", "vtec_matched_lag2"]

# ap columns in pymsis input order: daily Ap, ap now, -3h, -6h, -9h,
# avg 12-33h, avg 36-57h (same order as run_pymsis.py wrote them).
AP_COLS_MSIS = ["ap_daily", "ap_0h", "ap_m3h", "ap_m6h", "ap_m9h",
                "ap_avg12_33h", "ap_avg36_57h"]

MSIS_VERSION = 2.1

# %% ----------------------------- LOAD THE DATA --------------------------------
df = pd.read_parquet(DATA_FILE)

# %% ----------------------- FEATURE LISTS (SAME AS on_track.py) ----------------
cols_to_scale = [
    "f107", "ap_m6h", "lat", "f107a", "alt_km",
    "matched_tec_value", "ap_m3h", "vtec_matched_lag", "vtec_matched_lag2"
]

columns_to_keep = [
    "f107a", "lat",
    "matched_tec_value",
    "lon_cos",
    "lon_sin", "lst_sin", "ap_m3h",
    "doy_sin", "doy_cos", "f107", "alt_km",
    "ap_m6h",
    "vtec_matched_lag", "vtec_matched_lag2",
    "lst_lat_sin"
]

# %% ----------------------- LR SCHEDULER (SAME AS on_track.py) -----------------
def lr_scheduler(current_round: int):
    initial_lr = 0.03
    if current_round < 4:
        initial_lr = 1e-7
    decay_factor = 0.9
    step_size = 12
    calculated_lr = initial_lr * (decay_factor ** (current_round // step_size))
    if current_round % 100 == 0:
        print(f"Round {current_round}: LR = {calculated_lr:.8f}")
    return calculated_lr

# %% ------------- WARM-START UPDATE (SAME AS on_track.py) ----------------------
def update_xgb_model_aggressive_with_callbacks(
    existing_model,
    new_data: pd.DataFrame,
    target_col: str,
    scaler_X,
    scaler_y,
    columns_to_keep,
    cols_to_scale,
    extra_rounds: int = 2000,
    patience_rounds: int = 300,
    lr_scheduler=lr_scheduler
):
    new_data = new_data.sort_values(by=['date', 'time']).reset_index(drop=True)

    K = 6
    block_size = len(new_data) // K if len(new_data) >= K else 1
    np.random.seed(42)
    block_indices = np.arange(K)
    num_train_blocks = int(K * (2/3))
    train_block_indices = np.random.choice(block_indices, size=num_train_blocks, replace=False)

    train_chunks, val_chunks = [], []
    for i in range(K):
        start_idx = i * block_size
        end_idx = len(new_data) if i == K - 1 else start_idx + block_size
        block = new_data.iloc[start_idx:end_idx]
        (train_chunks if i in train_block_indices else val_chunks).append(block)

    train_chunk = pd.concat(train_chunks).sort_values(by=['date', 'time'])
    val_chunk   = pd.concat(val_chunks).sort_values(by=['date', 'time'])

    def prepare_features(chunk: pd.DataFrame):
        X_scaled = pd.DataFrame(
            scaler_X.transform(chunk[cols_to_scale]),
            columns=cols_to_scale,
            index=chunk.index
        )
        X_unscaled = chunk[[c for c in columns_to_keep if c not in cols_to_scale]]
        return pd.concat([X_scaled, X_unscaled], axis=1)[columns_to_keep]

    X_train_final = prepare_features(train_chunk)
    X_val_final   = prepare_features(val_chunk)

    y_train_scaled = scaler_y.transform(train_chunk[target_col].values.reshape(-1, 1)).ravel()
    y_val_scaled   = scaler_y.transform(val_chunk[target_col].values.reshape(-1, 1)).ravel()

    booster = existing_model if isinstance(existing_model, xgb.Booster) else existing_model.get_booster()
    feat_order = booster.feature_names
    X_train_final = X_train_final[feat_order]
    X_val_final   = X_val_final[feat_order]

    dtrain = xgb.DMatrix(X_train_final, label=y_train_scaled)
    dval   = xgb.DMatrix(X_val_final,   label=y_val_scaled)

    params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
    evals_result = {}

    updated_booster = xgb.train(
        params,
        dtrain,
        num_boost_round=extra_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        xgb_model=booster,
        callbacks=[
            xgb.callback.EarlyStopping(rounds=patience_rounds, save_best=True),
            xgb.callback.LearningRateScheduler(lr_scheduler),
        ],
    )

    tmp_model_path = "model_aggressive_best_true_forecast.json"
    updated_booster.save_model(tmp_model_path)
    best_booster = xgb.Booster()
    best_booster.load_model(tmp_model_path)
    return best_booster

# %% ----------------- DRIVER PERSISTENCE (SAME AS on_track_persisted.py) -------
def persist_drivers(window_data: pd.DataFrame, source_day: pd.DataFrame) -> pd.DataFrame:
    """
    Replace driver values in the forecast window with issue-time knowledge.

    window_data : rows of the forecast window (days t .. t+h-1)
    source_day  : rows of the last known day (day t-1), post feature
                  engineering, i.e. TEC lag columns already present.
    """
    out = window_data.copy()
    src = source_day.sort_values("time")

    # --- TEC field persistence via nearest (lat, LST) geometry ---
    theta_src = 2.0 * np.pi * src["lst_h"].to_numpy() / 24.0
    tree = cKDTree(np.column_stack([
        src["lat"].to_numpy(),
        LST_KDTREE_WEIGHT * np.sin(theta_src),
        LST_KDTREE_WEIGHT * np.cos(theta_src),
    ]))
    theta_win = 2.0 * np.pi * out["lst_h"].to_numpy() / 24.0
    _, idx = tree.query(np.column_stack([
        out["lat"].to_numpy(),
        LST_KDTREE_WEIGHT * np.sin(theta_win),
        LST_KDTREE_WEIGHT * np.cos(theta_win),
    ]))
    for col in TEC_COLS:
        out[col] = src[col].to_numpy()[idx]

    # --- Scalar driver persistence (last known values, held constant) ---
    out["f107"]  = src["f107"].iloc[-1]
    out["f107a"] = src["f107a"].iloc[-1]

    if "ap_0h" in src.columns and src["ap_0h"].notna().any():
        last_ap = src["ap_0h"].dropna().iloc[-1]
    else:
        last_ap = src["ap_m3h"].dropna().iloc[-1]
    out["ap_m3h"] = last_ap
    out["ap_m6h"] = last_ap

    return out

# %% ----------------------- PERSISTED-DRIVER MSIS (NEW) ------------------------
def naive_utc_dates(series: pd.Series) -> np.ndarray:
    """pymsis wants tz-naive datetime64; the parquet stores tz-aware UTC."""
    s = pd.to_datetime(series)
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.to_numpy(dtype="datetime64[ns]")


def issue_time_drivers(source_day: pd.DataFrame):
    """Last driver values available in the day t-1 data (same set persist_drivers uses)."""
    src = source_day.sort_values("time")
    f107_iss  = float(src["f107"].dropna().iloc[-1])
    f107a_iss = float(src["f107a"].dropna().iloc[-1])
    if "ap_0h" in src.columns and src["ap_0h"].notna().any():
        ap_iss = float(src["ap_0h"].dropna().iloc[-1])
    else:
        ap_iss = float(src["ap_m3h"].dropna().iloc[-1])
    return f107_iss, f107a_iss, ap_iss


def msis_with_persisted_drivers(window_data: pd.DataFrame,
                                source_day: pd.DataFrame) -> np.ndarray:
    """
    Re-run NRLMSISE-2.1 for the forecast-window points with all drivers frozen
    at their issue-time values. Geometry/time come from the points themselves.
    """
    f107_iss, f107a_iss, ap_iss = issue_time_drivers(source_day)
    n = len(window_data)
    out = pymsis.calculate(
        naive_utc_dates(window_data["time"]),
        window_data["lon"].to_numpy(dtype=float),
        window_data["lat"].to_numpy(dtype=float),
        window_data["alt_km"].to_numpy(dtype=float),
        f107s=np.full(n, f107_iss),
        f107as=np.full(n, f107a_iss),
        aps=np.full((n, 7), ap_iss),
        version=MSIS_VERSION,
    )
    return np.asarray(out)[:, 0]


def msis_sanity_check(df_all: pd.DataFrame, n_sample: int = 20000, seed: int = 0):
    """
    Verify that pymsis called with the dataframe's own OBSERVED driver columns
    reproduces the stored msis_rho. If it does, the persisted run below differs
    from the original msis_rho generation only in the driver VALUES.
    """
    cols = ["grace_time", "lon", "lat", "alt_km", "msis_rho", "f107", "f107a"] + AP_COLS_MSIS
    sub = df_all.dropna(subset=cols).sample(min(n_sample, len(df_all)), random_state=seed)
    sub = sub.sort_values("grace_time")
    stored = sub["msis_rho"].to_numpy(dtype=float)

    explicit = pymsis.calculate(
        naive_utc_dates(sub["grace_time"]),
        sub["lon"].to_numpy(dtype=float),
        sub["lat"].to_numpy(dtype=float),
        sub["alt_km"].to_numpy(dtype=float),
        f107s=sub["f107"].to_numpy(dtype=float),
        f107as=sub["f107a"].to_numpy(dtype=float),
        aps=sub[AP_COLS_MSIS].to_numpy(dtype=float),
        version=MSIS_VERSION,
    )
    explicit = np.asarray(explicit)[:, 0]
    rel = np.abs(explicit - stored) / stored
    print(f"[sanity] explicit-driver MSIS vs stored msis_rho on {len(sub):,} points: "
          f"median |rel diff| = {np.median(rel):.2e}, max = {np.max(rel):.2e}")
    if np.median(rel) > 0.01:
        print("[sanity] WARNING: explicit-driver recompute does not reproduce stored "
              "msis_rho — persisted-MSIS numbers are not directly comparable to it!")

# %% ----------------------------- METRICS HELPER -------------------------------
def compute_metrics(df: pd.DataFrame,
                    pred_col: str = "rho_pred",
                    obs_col: str = "rho_obs",
                    prefix: str = "") -> dict:
    """Paper-comparable metrics: RMSE, Top-5% error, MAPE, R2, log-RMSE, log-Top5."""
    y = df[obs_col].values
    yhat = df[pred_col].values
    err = yhat - y
    abs_err = np.abs(err)

    rmse = float(np.sqrt(np.mean(err ** 2)))
    thr = np.quantile(abs_err, 0.95)
    top5 = float(np.sqrt(np.mean(err[abs_err >= thr] ** 2)))

    mask = y > 0
    mape = float(np.mean(abs_err[mask] / y[mask]) * 100) if np.any(mask) else np.nan
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # log-space metrics (only where both sides positive)
    lmask = (y > 0) & (yhat > 0)
    if np.any(lmask):
        lerr = np.log(yhat[lmask]) - np.log(y[lmask])
        rmse_log = float(np.sqrt(np.mean(lerr ** 2)))
        lthr = np.quantile(np.abs(lerr), 0.95)
        top5_log = float(np.sqrt(np.mean(lerr[np.abs(lerr) >= lthr] ** 2)))
    else:
        rmse_log, top5_log = np.nan, np.nan

    return {
        f"{prefix}n": len(df),
        f"{prefix}rmse": rmse,
        f"{prefix}top5": top5,
        f"{prefix}mape_pct": mape,
        f"{prefix}bias": bias,
        f"{prefix}r2": r2,
        f"{prefix}rmse_log": rmse_log,
        f"{prefix}top5_log": top5_log,
    }

# %% -------------------------- ONE RUN (PARAMETERIZED) -------------------------
def run_experiment(do_retrain: int,
                   date_filter: str,
                   window_size: int,
                   tag: str,
                   output_root: str = OUTPUT_ROOT,
                   max_steps: int = None):
    """
    Rolling fine-tune + forecast with PERSISTED drivers for BOTH the ML
    correction and the MSIS scaffold. Mirrors on_track_persisted.py; the only
    additions are the per-window persisted-MSIS recompute and the extra
    reference columns.
    """
    print(tag)
    run_dir = os.path.join(output_root, tag)
    os.makedirs(run_dir, exist_ok=True)

    # ---- 0) Select data subset for this run ----
    df_local = df.copy()
    if date_filter == "pre2009":
        df_local = df_local[(df_local['grace_time'] < '2009-06-06')]
    elif date_filter == "post2016":
        df_local = df_local[(df_local['grace_time'] > '2016-01-01')]
    else:
        raise ValueError("date_filter must be 'pre2009' or 'post2016'")

    # ---- 1) Feature engineering (identical to on_track.py) ----
    df_local["time"] = df_local['grace_time']
    df_feat_local = df_local.copy()
    df_feat_local = ff.add_lst_doy_features(df_feat_local)
    df_feat_local['lon_sin'] = np.sin(np.deg2rad(df_feat_local['lon']))
    df_feat_local['lon_cos'] = np.cos(np.deg2rad(df_feat_local['lon']))
    df_feat_local['lst_lat_cos'] = df_feat_local['lst_cos'] * df_feat_local['lat']
    df_feat_local['vtec_matched_lag']  = df_feat_local['matched_tec_value'].shift(500)
    df_feat_local['vtec_matched_lag2'] = df_feat_local['matched_tec_value'].shift(17280)
    df_feat_local['lst_lat_sin'] = df_feat_local['lst_sin'] * df_feat_local['lat']
    df_feat_local['ap_change'] = df_feat_local['ap_0h'] - df_feat_local['ap_m3h']
    df_feat_local[TARGET_COL] = np.log(df_feat_local["rho_obs"] / df_feat_local["msis_rho"])
    df_feat_local = df_feat_local.dropna(subset=[
        "f107", "ap_m6h", "lat", "f107a", "alt_km",
        "matched_tec_value", "ap_m3h", "vtec_matched_lag", "vtec_matched_lag2", "log_ratio"
    ])
    df_feat_predict_local = df_feat_local.copy()

    # ---- 2) Load original model + scalers ----
    original_model = xgb.XGBRegressor()
    original_model.load_model(MODEL_FILE)
    base_model = copy.deepcopy(original_model)

    scaler_X = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)

    # ---- 3) Rolling loop ----
    step_size = 1
    df_feat_predict_local['date'] = pd.to_datetime(df_feat_predict_local['time']).dt.date
    unique_dates = df_feat_predict_local['date'].drop_duplicates().sort_values().tolist()

    all_preds = []
    step = 10
    n_windows = 0

    for start_idx in range(6, len(unique_dates) - window_size + 1, step_size):
        if max_steps is not None and n_windows >= max_steps:
            break
        step += 1

        if step % 7 == 0:
            base_model = copy.deepcopy(original_model)

        # previous 5 days used for fine-tuning (past data only — unchanged)
        prev_days = unique_dates[start_idx - 5: start_idx]
        train_data = df_feat_predict_local[df_feat_predict_local['date'].isin(prev_days)].copy()

        if do_retrain:
            base_model = update_xgb_model_aggressive_with_callbacks(
                existing_model=base_model,
                new_data=train_data,
                target_col=TARGET_COL,
                scaler_X=scaler_X,
                scaler_y=scaler_y,
                columns_to_keep=columns_to_keep,
                cols_to_scale=cols_to_scale,
                extra_rounds=2000,
            )

        # ---- Forecast window with PERSISTED drivers ----
        window_dates = unique_dates[start_idx: start_idx + window_size]
        window_data = df_feat_predict_local[df_feat_predict_local['date'].isin(window_dates)].copy()

        source_day = df_feat_predict_local[df_feat_predict_local['date'] == prev_days[-1]]
        if source_day.empty:
            print(f"⚠️ No source day {prev_days[-1]} — skipping window at {window_dates[0]}")
            continue
        window_data = persist_drivers(window_data, source_day)

        # ---- NEW: persisted-driver MSIS scaffold for the same window ----
        window_data["msis_persist"] = msis_with_persisted_drivers(window_data, source_day)
        bad = ~(window_data["msis_persist"] > 0) | window_data["msis_persist"].isna()
        if bad.any():
            print(f"⚠️ {bad.sum()} invalid persisted-MSIS values in window at "
                  f"{window_dates[0]} — dropping those rows")
            window_data = window_data[~bad]
        if window_data.empty:
            continue

        # prepare features (scaling AFTER persistence substitution)
        X_to_scale = window_data[cols_to_scale]
        X_scaled = pd.DataFrame(scaler_X.transform(X_to_scale), columns=cols_to_scale, index=X_to_scale.index)
        X_unscaled = window_data[[c for c in columns_to_keep if c not in cols_to_scale]]
        X_final = pd.concat([X_scaled, X_unscaled], axis=1)[columns_to_keep]

        if do_retrain:
            feat_order = base_model.feature_names
            X_final = X_final[feat_order]
            pred_scaled = base_model.predict(xgb.DMatrix(X_final))
        else:
            feat_order = base_model.get_booster().feature_names
            X_final = X_final[feat_order]
            pred_scaled = base_model.predict(X_final)

        pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
        window_data['y_true_log'] = window_data['log_ratio']
        window_data['y_pred_log'] = pred_original.flatten()
        window_data['rho_true']   = window_data['msis_rho'] * np.exp(window_data['y_true_log'])
        # TRUE forecast: persisted-MSIS scaffold x persisted-driver correction
        window_data['rho_pred']         = window_data['msis_persist'] * np.exp(window_data['y_pred_log'])
        # reference: same correction on the observed-driver scaffold
        # (equals the on_track_persisted.py prediction)
        window_data['rho_pred_obsmsis'] = window_data['msis_rho'] * np.exp(window_data['y_pred_log'])

        keep_cols = ['date', 'time', 'y_true_log', 'y_pred_log', 'rho_true', 'rho_pred',
                     'rho_pred_obsmsis', 'msis_persist'] \
                    + columns_to_keep + ["msis_rho", "rho_obs"]
        all_preds.append(window_data[keep_cols])
        n_windows += 1

    # ---- 4) Concatenate & save ----
    if len(all_preds) == 0:
        raise RuntimeError(f"No predictions generated for run {tag}.")
    pred_df = pd.concat(all_preds).reset_index(drop=True)

    pred_df.to_csv(os.path.join(run_dir, f"predictions_{tag}.csv"), index=False)
    pred_df.to_pickle(os.path.join(run_dir, f"predictions_{tag}.pkl"))
    base_model.save_model(os.path.join(run_dir, f"xgb_model_updated_{tag}.json"))

    # ---- 5) Metrics ----
    #   (no prefix) : ML true forecast          = msis_persist * exp(corr_persisted)
    #   msisp_      : OPERATIONAL baseline      = MSIS with persisted drivers
    #   msis_       : reference                 = stored perfect-driver MSIS
    #   predobs_    : reference                 = corr_persisted on observed scaffold
    metrics = compute_metrics(pred_df, pred_col="rho_pred", obs_col="rho_obs")
    metrics.update(compute_metrics(pred_df, pred_col="msis_persist", obs_col="rho_obs", prefix="msisp_"))
    metrics.update(compute_metrics(pred_df, pred_col="msis_rho", obs_col="rho_obs", prefix="msis_"))
    metrics.update(compute_metrics(pred_df, pred_col="rho_pred_obsmsis", obs_col="rho_obs", prefix="predobs_"))
    metrics.update({
        "tag": tag,
        "do_retrain": int(do_retrain),
        "date_filter": date_filter,
        "horizon_days": int(window_size),
        "drivers": "persisted_full",
    })
    return pred_df, metrics

# %% ------------------------------ MAIN: 8 RUNS --------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--smoke", action="store_true",
                        help="quick validation: single no-retrain run, 3 windows")
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    msis_sanity_check(df)

    if args.smoke:
        combos = [(0, "pre2009", 1)]
        max_steps = 3
        summary_csv = os.path.join(OUTPUT_ROOT, "summary_metrics_true_smoke.csv")
    else:
        do_retrain_opts = [0, 1]
        date_filters    = ["pre2009", "post2016"]
        horizons        = [1, 3]
        combos = [(dr, dfilt, h) for dr in do_retrain_opts for dfilt in date_filters for h in horizons]
        max_steps = None
        summary_csv = os.path.join(OUTPUT_ROOT, "summary_metrics_true.csv")

    summary_rows = []
    for dr, dfilt, h in combos:
        tag = f"dr{dr}_{dfilt}_h{h}_true"
        print(f"\n===== Starting run: {tag} =====")
        pred_df, metrics = run_experiment(do_retrain=dr, date_filter=dfilt,
                                          window_size=h, tag=tag, max_steps=max_steps)
        summary_rows.append(metrics)
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)  # save progress after each run
        print(f"✅ Finished run {tag}")

    print(f"\n📄 Summary metrics → {summary_csv}\n")
