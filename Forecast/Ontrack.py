# %% --------------------------------- IMPORTS ---------------------------------
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

# Your helper module with feature engineering and plotting utilities
import Feature_functions as ff

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
MODEL_FILE = "xgb_model_v3.json"                 # base model
SCALER_X_FILE = "scaler_xgboost_X_v3.joblib"     # fitted feature scaler
SCALER_Y_FILE = "scaler_xgboost_y_v3.joblib"     # fitted target scaler

# Data file (merged dataset). We will not globally filter; each run filters internally.
DATA_FILE = "grace_data_merged2.parquet"

# Output root folder for all run artifacts (CSV, PNG, updated models, etc.)
OUTPUT_ROOT = "runs"

# %% ----------------------------- LOAD THE DATA --------------------------------
# IMPORTANT: Do NOT apply any global date filtering here. Each run will filter internally.
df = pd.read_parquet(DATA_FILE)

# Optional: Quick alias for time handling used later in feature engineering
# We'll rebuild 'time' inside each run to be safe.
# %% -

# --- Extract space-weather context for the illustrative day ---
# --- Extract space-weather context for the illustrative day ---
day_mask = (
    pd.to_datetime(df["grace_time"]).dt.date
    == DATE_TO_SAVE_MODEL
)
day_df = df.loc[day_mask]

if day_df.empty:
    print(f"⚠️ No space-weather data for {DATE_TO_SAVE_MODEL}")
else:
    ap_min, ap_max = day_df["ap_m6h"].min(), day_df["ap_m6h"].max()
    f107_min, f107_max = day_df["f107"].min(), day_df["f107"].max()

    print(
        f"{DATE_TO_SAVE_MODEL} | "
        f"Ap_m6h: {ap_min:.0f}–{ap_max:.0f}, "
        f"F10.7: {f107_min:.0f}–{f107_max:.0f}"
    )


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

# %% ----------------------- LR SCHEDULER (YOUR LOGIC) --------------------------
def lr_scheduler(current_round: int):
    """
    Learning rate scheduler for native XGBoost API.
    Mirrors your setup: ultra-low LR initially, then exponential decay.
    """
    initial_lr = 0.03
    if current_round < 4:
        initial_lr = 1e-7
    decay_factor = 0.9
    step_size = 12
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
    patience_rounds: int = 300,
    lr_scheduler=lr_scheduler
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
    tmp_model_path = "model_aggressive_best.json"
    updated_booster.save_model(tmp_model_path)
    best_booster = xgb.Booster()
    best_booster.load_model(tmp_model_path)
    return best_booster

# %% ----------------------------- METRICS HELPER -------------------------------
def compute_metrics(df: pd.DataFrame,
                    pred_col: str = "rho_pred",
                    obs_col: str = "rho_obs") -> dict:
    """Compute overall metrics for the run."""
    y = df[obs_col].values
    yhat = df[pred_col].values
    n = len(df)
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    mae  = float(np.mean(np.abs(yhat - y)))
    mask = y != 0
    mape = float(np.mean(np.abs((yhat[mask] - y[mask]) / y[mask])) * 100) if np.any(mask) else np.nan
    bias = float(np.mean(yhat - y))
    return {"n": n, "rmse": rmse, "mae": mae, "mape_pct": mape, "bias": bias}

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
    DATE_TO_SAVE_MODEL = pd.to_datetime("2009-01-10").date()
    print(tag)
    if "pre2009" in tag:
        DATE_TO_SAVE_MODEL = pd.to_datetime("2009-01-13").date()
    elif "post2016" in tag:
        DATE_TO_SAVE_MODEL = pd.to_datetime("2016-02-18").date()
    else:
        DATE_TO_SAVE_MODEL = None
    print(DATE_TO_SAVE_MODEL)
    # Prepare run directory
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

    # ---- 1) Feature engineering (your exact steps) ----
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
    original_model.save_model(os.path.join(run_dir, f"xgb_model_original_{tag}.json"))
    base_model = copy.deepcopy(original_model)

    scaler_X = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)

    # ---- 3) Rolling loop ----
    step_size = 1
    df_feat_predict_local['date'] = pd.to_datetime(df_feat_predict_local['time']).dt.date
    unique_dates = df_feat_predict_local['date'].drop_duplicates().sort_values().tolist()

    all_preds = []
    step = 10

    for start_idx in range(6, len(unique_dates) - window_size + 1, step_size):
        step += 1
   

        if step % 7 == 0:
            # periodic reset to the original model
            base_model = copy.deepcopy(original_model)

        # previous 5 days used for fine-tuning
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
          
        current_forecast_start_date = unique_dates[start_idx]
        if (do_retrain == 1) and (current_forecast_start_date == DATE_TO_SAVE_MODEL):
            snapshot_fn = os.path.join(run_dir, f"xgb_model_saved_{tag}_start_{current_forecast_start_date}.json")
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
        all_preds.append(window_data[keep_cols])

    # ---- 4) Concatenate predictions for this run ----
    if len(all_preds) == 0:
        raise RuntimeError(f"No predictions generated for run {tag}. "
                           f"Check that your date_filter '{date_filter}' yields data and that windowing is valid.")
    pred_df = pd.concat(all_preds).reset_index(drop=True)

    # ---- 5) Save per-run predictions as CSV ----
    pred_csv = os.path.join(run_dir, f"predictions_{tag}.csv")
    pred_df.to_csv(pred_csv, index=False)

    pred_pkl = os.path.join(run_dir, f"predictions_{tag}.pkl")
    pred_df.to_pickle(pred_pkl)


    # ---- 6) Save the (potentially) updated model for this run ----
    base_model.save_model(os.path.join(run_dir, f"xgb_model_updated_{tag}.json"))

    # ---- 7) Figures: save with unique filenames tagged by the run ----
    # a) Validation densities + metrics figure
    ff.plot_val_densities_with_metrics3(
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





    plot_df = pred_df[
        pd.to_datetime(pred_df["date"]).dt.date == DATE_TO_SAVE_MODEL
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
        metrics = compute_metrics(pred_df, pred_col="rho_pred", obs_col="rho_obs")
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

    # Full 8 combinations:
    do_retrain_opts = [0, 1]
    date_filters    = ["pre2009", "post2016"]
    horizons        = [1, 3]
    combos = [(dr, dfilt, h) for dr in do_retrain_opts for dfilt in date_filters for h in horizons]

    summary_rows = []
    summary_csv = os.path.join(OUTPUT_ROOT, "summary_metrics.csv")

    for dr, dfilt, h in combos:
        tag = f"dr{dr}_{dfilt}_h{h}"
        print(f"\n===== Starting run: {tag} =====")
        pred_df, metrics = run_experiment(do_retrain=dr, date_filter=dfilt, window_size=h, tag=tag)
        summary_rows.append(metrics)
        print(f"✅ Finished run {tag}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n📄 Wrote summary metrics → {summary_csv}\n")

    summary_pkl = os.path.join(OUTPUT_ROOT, "summary_metrics.pkl")
    summary_df.to_pickle(summary_pkl)
    print(f"📦 Wrote summary metrics (pickle) → {summary_pkl}\n")

