# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
ontrack.py
- Runs rolling warm-start forecasts on data outside the 2009–2016 training window.
- Fine-tunes the base model each step using the previous 5 days, with early stopping.
- Tests 8 combinations of retrain/no-retrain, pre/post-training dates, and 1 or 3-day horizons.
"""

# %% --------------------------------- IMPORTS ---------------------------------
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import gc
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

# Your helper module with feature engineering and plotting utilities
import feature_functions as ff

# %% ----------------------------- CONFIGURATION --------------------------------
# Turn plotting on/off for the initial MSIS vs Observed plot (not essential for batch runs)
PLOT = False

# The day for which you want to save an exact model snapshot during the rolling loop
# Set to any date present in your dataset; example kept from your original file.
# (You can change this freely.)
#DATE_TO_SAVE_MODEL = pd.to_datetime("2016-02-18").date()
DATE_TO_SAVE_MODEL = pd.to_datetime("2009-01-13").date()



TARGET_COL = "log_ratio"

# Model/scaler filenames (must already exist — these are from your current setup)
# Defaults below can be overridden via environment variables (see run_pipeline.sh)
MODEL_FILE = os.environ.get("ONTRACK_MODEL_FILE", "xgb_model_v3.json")           # base model
SCALER_X_FILE = os.environ.get("ONTRACK_SCALER_X_FILE", "scaler_xgboost_X_v3.joblib")  # fitted feature scaler
SCALER_Y_FILE = os.environ.get("ONTRACK_SCALER_Y_FILE", "scaler_xgboost_y_v3.joblib")  # fitted target scaler

# Data file (merged dataset). We will not globally filter; each run filters internally.
DATA_FILE = os.environ.get("ONTRACK_DATA_FILE", "grace_data_merged2.parquet")

# Output root folder for all run artifacts (CSV, PNG, updated models, etc.)
OUTPUT_ROOT = os.environ.get("ONTRACK_OUTPUT_ROOT", "runs")

# Optional tuned core-model parameters. When provided, warm-start uses the same
# tree-shape parameters for newly added trees instead of XGBoost load defaults.
ONTRACK_PARAMS_JSON = os.environ.get("ONTRACK_PARAMS_JSON", "").strip()
ONTRACK_TREE_PARAMS = None
if ONTRACK_PARAMS_JSON:
    with open(ONTRACK_PARAMS_JSON) as fh:
        _tuned = json.load(fh)
    ONTRACK_TREE_PARAMS = {
        "max_depth": int(_tuned["max_depth"]),
        "min_child_weight": float(_tuned["min_child_weight"]),
        "subsample": float(_tuned["subsample"]),
        "colsample_bytree": float(_tuned["colsample_bytree"]),
    }
    print(f"Warm-start tree params from {ONTRACK_PARAMS_JSON}: {ONTRACK_TREE_PARAMS}")

# TEC lag features: "rows" = historical shift(500)/shift(17280) (v3 model),
# "time" = exact, gap-robust lookups at t-2500s / t-24h (full-mission setup).
# Must match the mode the base model was trained with.
TEC_LAG_MODE = os.environ.get("TEC_LAG_MODE", "rows")

# %% ----------------------------- LOAD THE DATA --------------------------------
# Keep only the schema globally. Each run uses Parquet predicate pushdown so
# the 82.5M-row full-mission dataset is never kept resident in its entirety.
_RAW_NEEDED = sorted({
    "grace_time", "lat", "lon", "alt_km", "lst_h",
    "rho_obs", "msis_rho", "matched_tec_value",
    "f107", "f107a",
    "ap_daily", "ap_0h", "ap_m3h", "ap_m6h", "ap_m9h",
    "ap_avg12_33h", "ap_avg36_57h",
})
_available = pq.ParquetFile(DATA_FILE).schema.names
_read_columns = [c for c in _RAW_NEEDED if c in _available]


def _utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _filters_for_regime(date_filter: str):
    """Arrow filters matching the historical in-memory date cuts."""
    filters = {
        "pre2009": [("grace_time", "<", _utc("2009-06-06"))],
        "post2016": [("grace_time", ">", _utc("2016-01-01"))],
        "y2002": [("grace_time", "<", _utc("2003-01-01"))],
        "quiet2009": [
            ("grace_time", ">=", _utc("2009-01-01")),
            ("grace_time", "<", _utc("2009-06-06")),
        ],
        "storm2015": [
            ("grace_time", ">=", _utc("2015-03-01")),
            ("grace_time", "<", _utc("2015-04-15")),
        ],
    }
    try:
        return filters[date_filter]
    except KeyError as exc:
        raise ValueError(
            "date_filter must be 'pre2009', 'post2016', 'y2002', "
            "'quiet2009' or 'storm2015'"
        ) from exc


def _load_regime(date_filter: str) -> pd.DataFrame:
    """Load one evaluation regime without materializing the full mission."""
    frame = pd.read_parquet(
        DATA_FILE,
        columns=_read_columns,
        filters=_filters_for_regime(date_filter),
    )
    print(
        f"Loaded {date_filter}: {len(frame):,} rows "
        f"({frame.memory_usage(deep=True).sum() / 2**30:.2f} GiB in pandas)"
    )
    return frame


# %% ----------------------- FEATURE LISTS (YOUR SETUP) -------------------------
# Columns to scale (kept from your script)
cols_to_scale = [
    "f107", "ap_m6h", "lat", "f107a", "alt_km",
    "matched_tec_value", "ap_m3h", "vtec_matched_lag", "vtec_matched_lag2"
]

# Full ordered feature set used for training/prediction (kept from your script)
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

# Storm-history drivers (see CoreModel/config.py). Must match the feature set
# the base model was trained with — AP_HISTORY=1 for both training and here.
AP_HISTORY_FEATURES = ["ap_m9h", "ap_avg12_33h", "ap_avg36_57h"]
AP_HISTORY_FEATURES_FULL = ["ap_daily", "ap_0h"] + AP_HISTORY_FEATURES

_ap = os.environ.get("AP_HISTORY", "1")
if _ap != "0":
    _extra = AP_HISTORY_FEATURES_FULL if _ap == "full" else AP_HISTORY_FEATURES
    columns_to_keep = columns_to_keep + _extra
    cols_to_scale = cols_to_scale + _extra

# %% ----------------------- LR SCHEDULER (YOUR LOGIC) --------------------------
def lr_scheduler(current_round: int):
    """
    Learning rate scheduler for native XGBoost API.
    Mirrors your setup: ultra-low LR initially, then exponential decay.
    """
    initial_lr = 0.005
    if current_round < 4:
        initial_lr = 1e-7
    decay_factor = 0.9
    step_size = 20
    calculated_lr = initial_lr * (decay_factor ** (current_round // step_size))
    if current_round % 100 == 0:
        print(f"Round {current_round}: LR = {calculated_lr:.8f}")
    return calculated_lr

# %% ------------- AGGRESSIVE UPDATE WITH CALLBACKS (YOUR FUNCTION) -------------
def update_xgb_model_aggressive_with_callbacks(
    existing_model,
    new_data: pd.DataFrame,
    target_col: str,
    scaler_X,
    scaler_y,
    columns_to_keep,
    cols_to_scale,
    extra_rounds: int = 2000,
    patience_rounds: int = 60,
    lr_scheduler=lr_scheduler,
    tree_params: dict | None = None
):
    """
    Aggressive update:
    - Uses EarlyStopping + LR scheduler
    - Returns the BEST booster (not the last)
    - Ensures next step starts from best checkpoint
    (Code adapted directly from your original file.)
    """
    # 1) Data Preparation and Splitting
    new_data = new_data.sort_values(by=['date', 'time']).reset_index(drop=True)

    # Random block split (train/val)
    K = 6
    block_size = len(new_data) // K if len(new_data) >= K else 1
    np.random.seed(42)
    block_indices = np.arange(K)
    num_train_blocks = int(K * (2/3))
    train_block_indices = np.random.choice(block_indices, size=num_train_blocks, replace=False)
    val_block_indices = np.setdiff1d(block_indices, train_block_indices)

    train_chunks, val_chunks = [], []
    for i in range(K):
        start_idx = i * block_size
        end_idx = len(new_data) if i == K - 1 else start_idx + block_size
        block = new_data.iloc[start_idx:end_idx]
        (train_chunks if i in train_block_indices else val_chunks).append(block)

    train_chunk = pd.concat(train_chunks).sort_values(by=['date', 'time'])
    val_chunk   = pd.concat(val_chunks).sort_values(by=['date', 'time'])

    # 2) Feature Scaling and Alignment
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

    # Booster and feature alignment
    booster = existing_model if isinstance(existing_model, xgb.Booster) else existing_model.get_booster()
    feat_order = booster.feature_names
    X_train_final = X_train_final[feat_order]
    X_val_final   = X_val_final[feat_order]

    dtrain = xgb.DMatrix(X_train_final, label=y_train_scaled)
    dval   = xgb.DMatrix(X_val_final,   label=y_val_scaled)

    # 3) Train with callbacks (EarlyStopping + LR scheduler)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }
    if tree_params:
        params.update(tree_params)
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

    # 4) Save best checkpoint and return clean booster
    # Use a per-update temporary file. A fixed workspace filename retained an
    # unnecessary artifact and made concurrent/resumed runs overwrite it.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_model_path = tmp.name
    try:
        updated_booster.save_model(tmp_model_path)
        best_booster = xgb.Booster()
        best_booster.load_model(tmp_model_path)
    finally:
        os.unlink(tmp_model_path)
    return best_booster

# %% ----------------------------- METRICS HELPER -------------------------------
def compute_metrics(df: pd.DataFrame,
                    pred_col: str = "rho_pred",
                    obs_col: str = "rho_obs",
                    prefix: str = "") -> dict:
    """Paper-comparable metrics: RMSE, MAE, Top-5% error, MAPE, R2, log-RMSE, log-Top5."""
    y = df[obs_col].values
    yhat = df[pred_col].values
    err = yhat - y
    abs_err = np.abs(err)

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(abs_err))
    thr = np.quantile(abs_err, 0.95)
    top5 = float(np.sqrt(np.mean(err[abs_err >= thr] ** 2)))

    mask = y != 0
    mape = float(np.mean(abs_err[mask] / np.abs(y[mask])) * 100) if np.any(mask) else np.nan
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
        f"{prefix}mae": mae,
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
                   output_root: str = OUTPUT_ROOT):
    """
    Run the rolling fine-tune + forecast with chosen settings and save outputs.

    Args:
        do_retrain: 1 to fine-tune each step on previous 5 days; 0 to skip.
        date_filter: "pre2009" or "post2016".
        window_size: forecast horizon in days (1 or 3).
        tag: unique label used to distinguish outputs (e.g., dr1_post2016_h3).
        output_root: base folder to store all artifacts.

    Returns:
        pred_df: dataframe containing all predictions for this run.
        metrics: dict with overall metrics for this run.
    """
    DATE_TO_SAVE_MODEL = _utc("2009-01-10")
    print(tag)
    if "pre2009" in tag:
        DATE_TO_SAVE_MODEL = _utc("2009-01-13")
    elif "post2016" in tag:
        DATE_TO_SAVE_MODEL = _utc("2016-02-18")
    else:
        DATE_TO_SAVE_MODEL = None
    print(DATE_TO_SAVE_MODEL)
    # Prepare run directory
    run_dir = os.path.join(output_root, tag)
    os.makedirs(run_dir, exist_ok=True)

    # ---- 0) Select data subset for this run ----
    df_local = _load_regime(date_filter)

    # ---- 1) Feature engineering (your exact steps) ----
    # Work in place on df_local (already a filtered copy from step 0) instead
    # of taking further full-dataframe copies — post2016/quiet2009 are large
    # enough (15-20M rows) that each extra copy is a real memory cost.
    df_local["time"] = df_local['grace_time']
    df_local = ff.add_lst_doy_features(df_local, copy=False)
    df_local['lon_sin'] = np.sin(np.deg2rad(df_local['lon']))
    df_local['lon_cos'] = np.cos(np.deg2rad(df_local['lon']))
    df_local['lst_lat_cos'] = df_local['lst_cos'] * df_local['lat']
    if TEC_LAG_MODE == "time":
        df_local = ff.add_tec_time_lag_features(df_local, time_col="time")
    else:
        df_local['vtec_matched_lag']  = df_local['matched_tec_value'].shift(500)
        df_local['vtec_matched_lag2'] = df_local['matched_tec_value'].shift(17280)
    df_local['lst_lat_sin'] = df_local['lst_sin'] * df_local['lat']
    df_local['ap_change'] = df_local['ap_0h'] - df_local['ap_m3h']
    df_local[TARGET_COL] = np.log(df_local["rho_obs"] / df_local["msis_rho"])
    df_local.dropna(subset=sorted(set([
        "f107", "ap_m6h", "lat", "f107a", "alt_km",
        "matched_tec_value", "ap_m3h", "vtec_matched_lag", "vtec_matched_lag2", "log_ratio"
    ]) | set(cols_to_scale)), inplace=True)
    # Drop columns nothing downstream needs, before the rolling loop starts
    # copying per-window slices of this frame 100s of times.
    needed_cols = set(columns_to_keep) | set(cols_to_scale) | {
        "date", "time", TARGET_COL, "msis_rho", "rho_obs", "grace_time",
        "lst_cos", "lst_sin", "ap_0h",
    }
    # Drop in place rather than via df[keep].copy(): the copy would duplicate
    # the whole regime frame (post2016 is ~17M rows) and needs more headroom
    # than it saves.
    for c in [c for c in df_local.columns if c not in needed_cols]:
        del df_local[c]
    df_feat_predict_local = df_local
    gc.collect()

    # ---- 2) Load original model + scalers ----
    original_model = xgb.XGBRegressor()
    original_model.load_model(MODEL_FILE)
    original_model.save_model(os.path.join(run_dir, f"xgb_model_original_{tag}.json"))
    base_model = copy.deepcopy(original_model)

    scaler_X = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)

    # ---- 3) Rolling loop ----
    step_size = 1
    # datetime64 is much smaller than a column of Python ``date`` objects.
    df_feat_predict_local['date'] = pd.to_datetime(
        df_feat_predict_local['time'], utc=True
    ).dt.normalize()
    unique_dates = df_feat_predict_local['date'].drop_duplicates().sort_values().tolist()

    # Stage windows on disk. The old all_preds list retained every overlapping
    # h3 window, and pd.concat briefly duplicated the complete result.
    pred_parquet = os.path.join(run_dir, f".predictions_{tag}.staging.parquet")
    pred_writer = None
    step = 10

    for start_idx in range(15, len(unique_dates) - window_size + 1, step_size):
        step += 1
   

        if step % 4 == 0:
            # periodic reset to the original model
            base_model = copy.deepcopy(original_model)

        # previous 5 days used for fine-tuning
        prev_days = unique_dates[start_idx -14: start_idx]
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
                tree_params=ONTRACK_TREE_PARAMS,
            )
        del train_data

        current_forecast_start_date = unique_dates[start_idx]
        if (do_retrain == 1) and (current_forecast_start_date == DATE_TO_SAVE_MODEL):
            snapshot_date = current_forecast_start_date.date()
            snapshot_fn = os.path.join(run_dir, f"xgb_model_saved_{tag}_start_{snapshot_date}.json")
            base_model.save_model(snapshot_fn)
            print(f"\n💾 Saved snapshot for {tag} at {current_forecast_start_date} → {snapshot_fn}\n")

        # ---- Forecast window ----
        window_dates = unique_dates[start_idx: start_idx + window_size]
        window_data = df_feat_predict_local[df_feat_predict_local['date'].isin(window_dates)].copy()

        # prepare features
        X_to_scale = window_data[cols_to_scale]
        X_scaled = pd.DataFrame(scaler_X.transform(X_to_scale), columns=cols_to_scale, index=X_to_scale.index)
        X_unscaled = window_data[[c for c in columns_to_keep if c not in cols_to_scale]]
        X_final = pd.concat([X_scaled, X_unscaled], axis=1)[columns_to_keep]

        # match feature order & predict
        if do_retrain:
            # base_model is a Booster after update_xgb_model_aggressive_with_callbacks
            feat_order = base_model.feature_names
            X_final = X_final[feat_order]
            pred_scaled = base_model.predict(xgb.DMatrix(X_final))
        else:
            # base_model is XGBRegressor (wrapper) when not retraining
            feat_order = base_model.get_booster().feature_names
            X_final = X_final[feat_order]
            pred_scaled = base_model.predict(X_final)

        # back-transform to original target space
        pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
        window_data['y_true_log'] = window_data['log_ratio']
        window_data['y_pred_log'] = pred_original.flatten()
        window_data['rho_true']   = window_data['msis_rho'] * np.exp(window_data['y_true_log'])
        window_data['rho_pred']   = window_data['msis_rho'] * np.exp(window_data['y_pred_log'])

        keep_cols = ['date', 'time', 'y_true_log', 'y_pred_log', 'rho_true', 'rho_pred'] \
                    + columns_to_keep + ["msis_rho", "rho_obs"]
        pred_chunk = window_data[keep_cols]
        table = pa.Table.from_pandas(pred_chunk, preserve_index=False)
        if pred_writer is None:
            pred_writer = pq.ParquetWriter(
                pred_parquet, table.schema, compression="zstd"
            )
        pred_writer.write_table(table)
        del table, pred_chunk, window_data, X_to_scale, X_scaled, X_unscaled, X_final
        gc.collect()

    # ---- 4) Finalize disk-staged predictions for this run ----
    if pred_writer is None:
        raise RuntimeError(f"No predictions generated for run {tag}. "
                           f"Check that your date_filter '{date_filter}' yields data and that windowing is valid.")
    pred_writer.close()
    del pred_writer, df_feat_predict_local, df_local
    gc.collect()

    # Preserve the pickle contract used by the report scripts. At this point
    # the regime frame and rolling chunks have been released, so only one full
    # prediction frame is resident.
    pred_df = pd.read_parquet(pred_parquet)

    # ---- 5) Save per-run predictions ----
    # The pickle is what every downstream reader uses; the CSV is a convenience
    # copy. Writing CSV for the largest runs builds a multi-GB string buffer on
    # top of pred_df, so skip it there (override with ONTRACK_ALWAYS_CSV=1).
    pred_pkl = os.path.join(run_dir, f"predictions_{tag}.pkl")
    pred_df.to_pickle(pred_pkl)
    os.unlink(pred_parquet)

    csv_row_limit = int(os.environ.get("ONTRACK_CSV_MAX_ROWS", 10_000_000))
    if len(pred_df) <= csv_row_limit or os.environ.get("ONTRACK_ALWAYS_CSV") == "1":
        pred_df.to_csv(os.path.join(run_dir, f"predictions_{tag}.csv"), index=False)
    else:
        print(f"  (skipped CSV for {tag}: {len(pred_df):,} rows > {csv_row_limit:,}; "
              f"pickle written — set ONTRACK_ALWAYS_CSV=1 to force)")


    # ---- 6) Save the (potentially) updated model for this run ----
    base_model.save_model(os.path.join(run_dir, f"xgb_model_updated_{tag}.json"))

    # ---- 7) Figures: save with unique filenames tagged by the run ----
    # a) Validation densities + metrics figure
    ff.plot_val_densities_with_metrics(
        pred_df, time_col="time", sample_step=30,
        obs_col="rho_obs", msis_col="msis_rho", pred_col="rho_pred"
    )
    fig = plt.gcf()
    if fig and fig.axes:
        figpath1 = os.path.join(run_dir, f"val_densities_{tag}.png")
        fig.savefig(figpath1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    import pickle
    pkl_path = os.path.join(run_dir, f"val_densities_{tag}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(fig, f)





    if DATE_TO_SAVE_MODEL is None:
        plot_df = pred_df.iloc[0:0]
    else:
        plot_df = pred_df[
            pd.to_datetime(pred_df["date"], utc=True).dt.normalize()
            == DATE_TO_SAVE_MODEL
        ]


    if plot_df.empty:
        print(f"⚠️ No data for {DATE_TO_SAVE_MODEL} in run {tag}")
    else:

    # b) Diagnostic index plot
        ff.simple_index_plot(
            plot_df, y_pred="rho_pred",
            start_index=0, n_steps=5000,
            feature_list=["f107", "matched_tec_value", "ap_m6h", "doy_cos"],
            y_target="rho_obs", Title=f"all_{tag}"
        )
        fig = plt.gcf()
        if fig and fig.axes:
            figpath2 = os.path.join(run_dir, f"indexplot_all_{tag}.png")
            fig.savefig(figpath2, dpi=150, bbox_inches="tight")
  

        index_pkl = os.path.join(run_dir, f"indexplot_all_{tag}.pkl")
        with open(index_pkl, "wb") as f:
            pickle.dump(fig, f)

        plt.close(fig)

    # ---- 8) Compute & return metrics for summary CSV ----
    # (Runs for every combo — including those without a snapshot/plot date.)
    metrics = compute_metrics(pred_df, pred_col="rho_pred", obs_col="rho_obs")
    metrics.update(compute_metrics(pred_df, pred_col="msis_rho", obs_col="rho_obs", prefix="msis_"))
    metrics.update({
        "tag": tag,
        "do_retrain": int(do_retrain),
        "date_filter": date_filter,
        "horizon_days": int(window_size)
    })
    return pred_df, metrics

# %% ------------------------------ MAIN: 8 RUNS --------------------------------
if __name__ == "__main__":
    # Ensure output root exists
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Full 8 combinations (12 with the full-mission filters):
    do_retrain_opts = [0, 1]
    date_filters    = os.environ.get("ONTRACK_FILTERS", "pre2009,post2016").split(",")
    horizons = [
        int(value.strip())
        for value in os.environ.get("ONTRACK_HORIZONS", "1,3").split(",")
        if value.strip()
    ]
    invalid_horizons = sorted(set(horizons) - {1, 3})
    if not horizons or invalid_horizons:
        raise ValueError(
            "ONTRACK_HORIZONS must contain 1 and/or 3 "
            f"(got {os.environ.get('ONTRACK_HORIZONS', '')!r})"
        )
    combos = [(dr, dfilt, h) for dr in do_retrain_opts for dfilt in date_filters for h in horizons]

    summary_csv = os.path.join(OUTPUT_ROOT, "summary_metrics.csv")
    if os.path.exists(summary_csv):
        previous_summary = pd.read_csv(summary_csv)
        previous_by_tag = {
            str(row["tag"]): row.to_dict()
            for _, row in previous_summary.iterrows()
        }
        summary_order = previous_summary["tag"].astype(str).tolist()
    else:
        previous_by_tag = {}
        summary_order = []
    summary_by_tag = previous_by_tag.copy()

    for dr, dfilt, h in combos:
        tag = f"dr{dr}_{dfilt}_h{h}"
        existing_pkl = os.path.join(OUTPUT_ROOT, tag, f"predictions_{tag}.pkl")
        if os.path.exists(existing_pkl) and tag in previous_by_tag:
            # Do not reopen multi-GB prediction files merely to reproduce
            # metrics already persisted in the progress summary.
            print(f"\n===== Skipping run {tag} (saved metrics + predictions found) =====")
            metrics = previous_by_tag[tag]
        elif os.path.exists(existing_pkl):
            # The artifact was saved but the process died before its summary
            # row. Recover it once, then release it before starting dr1.
            print(f"\n===== Recovering metrics for {tag} from {existing_pkl} =====")
            pred_df = pd.read_pickle(existing_pkl)
            metrics = compute_metrics(pred_df, pred_col="rho_pred", obs_col="rho_obs")
            metrics.update(compute_metrics(pred_df, pred_col="msis_rho", obs_col="rho_obs", prefix="msis_"))
            metrics.update({"tag": tag, "do_retrain": int(dr),
                            "date_filter": dfilt, "horizon_days": int(h)})
            del pred_df
            gc.collect()
        else:
            print(f"\n===== Starting run: {tag} =====")
            pred_df, metrics = run_experiment(do_retrain=dr, date_filter=dfilt, window_size=h, tag=tag)
            print(f"✅ Finished run {tag}")
            del pred_df
            gc.collect()
        summary_by_tag[tag] = metrics
        if tag not in summary_order:
            summary_order.append(tag)
        # Keep metrics from horizons not selected in this invocation. Running
        # h1-only must not erase already-completed h3 summary rows.
        summary_rows = [summary_by_tag[t] for t in summary_order if t in summary_by_tag]
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print(f"\n📄 Wrote summary metrics → {summary_csv}\n")

    summary_pkl = os.path.join(OUTPUT_ROOT, "summary_metrics.pkl")
    pd.DataFrame(summary_rows).to_pickle(summary_pkl)
    print(f"📦 Wrote summary metrics (pickle) → {summary_pkl}\n")
