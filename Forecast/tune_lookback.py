# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
tune_lookback.py
- Runs the rolling warm-start forecast for lookback windows of 3, 5, and 7 days.
- All other settings are fixed: post2016 data, 3-day forecast horizon, retraining on.
- Prints a metrics table and saves a bar chart comparing RMSE and MAPE across lookbacks.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

import feature_functions as ff

# ---------------------------------------------------------------------------
# CONFIG — change these if needed
# ---------------------------------------------------------------------------
LOOKBACKS    = [3, 5, 7]       # days to look back for fine-tuning
HORIZON      = 3               # forecast horizon in days
DATE_FILTER  = "post2016"      # "pre2009" or "post2016"
EVAL_START   = "2016-02-01"    # start of evaluation window (needs lookback days before this)
EVAL_END     = "2016-02-28"    # end of evaluation window
STEP_SIZE    = 1               # rolling step in days
RESET_EVERY  = 7               # periodic model reset to base (every N steps)

MODEL_FILE   = "xgb_model_v3.json"
SCALER_X_FILE = "scaler_xgboost_X_v3.joblib"
SCALER_Y_FILE = "scaler_xgboost_y_v3.joblib"
DATA_FILE    = "grace_data_merged2.parquet"

TARGET_COL   = "log_ratio"
cols_to_scale = [
    "f107", "ap_m6h", "lat", "f107a", "alt_km",
    "matched_tec_value", "ap_m3h", "vtec_matched_lag", "vtec_matched_lag2",
]
columns_to_keep = [
    "f107a", "lat", "matched_tec_value", "lon_cos", "lon_sin",
    "lst_sin", "ap_m3h", "doy_sin", "doy_cos", "f107", "alt_km",
    "ap_m6h", "vtec_matched_lag", "vtec_matched_lag2", "lst_lat_sin",
]

# ---------------------------------------------------------------------------
# LR SCHEDULER
# ---------------------------------------------------------------------------
def lr_scheduler(current_round: int) -> float:
    initial_lr = 0.03 if current_round >= 4 else 1e-7
    return initial_lr * (0.9 ** (current_round // 12))


# ---------------------------------------------------------------------------
# FINE-TUNE
# ---------------------------------------------------------------------------
def fine_tune(booster, train_data, scaler_X, scaler_y):
    def prep(chunk):
        X_s = pd.DataFrame(
            scaler_X.transform(chunk[cols_to_scale]),
            columns=cols_to_scale, index=chunk.index,
        )
        X_u = chunk[[c for c in columns_to_keep if c not in cols_to_scale]]
        return pd.concat([X_s, X_u], axis=1)[booster.feature_names]

    K = 6
    block_size = max(1, len(train_data) // K)
    np.random.seed(42)
    train_idx = np.random.choice(np.arange(K), size=int(K * 2 / 3), replace=False)
    val_idx   = np.setdiff1d(np.arange(K), train_idx)

    train_chunks = [train_data.iloc[i * block_size: (i + 1) * block_size] for i in train_idx]
    val_chunks   = [train_data.iloc[i * block_size: (i + 1) * block_size] for i in val_idx]
    tr = pd.concat(train_chunks)
    vl = pd.concat(val_chunks)

    dtrain = xgb.DMatrix(prep(tr), label=scaler_y.transform(tr[TARGET_COL].values.reshape(-1, 1)).ravel())
    dval   = xgb.DMatrix(prep(vl), label=scaler_y.transform(vl[TARGET_COL].values.reshape(-1, 1)).ravel())

    updated = xgb.train(
        {"objective": "reg:squarederror", "eval_metric": "rmse"},
        dtrain, num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        xgb_model=booster,
        callbacks=[
            xgb.callback.EarlyStopping(rounds=300, save_best=True),
            xgb.callback.LearningRateScheduler(lr_scheduler),
        ],
        verbose_eval=False,
    )
    tmp = "tmp_tune_lookback.json"
    updated.save_model(tmp)
    b = xgb.Booster()
    b.load_model(tmp)
    return b


# ---------------------------------------------------------------------------
# ROLLING FORECAST FOR ONE LOOKBACK VALUE
# ---------------------------------------------------------------------------
def run_lookback(df_feat, original_booster, scaler_X, scaler_y, lookback: int) -> pd.DataFrame:
    df_feat = df_feat.copy()
    df_feat["date"] = pd.to_datetime(df_feat["time"]).dt.date
    unique_dates = df_feat["date"].drop_duplicates().sort_values().tolist()

    base = copy.deepcopy(original_booster)
    all_preds = []
    step = 0

    eval_start = pd.Timestamp(EVAL_START).date()
    eval_end   = pd.Timestamp(EVAL_END).date()

    for i in range(lookback, len(unique_dates) - HORIZON + 1, STEP_SIZE):
        if not (eval_start <= unique_dates[i] <= eval_end):
            continue
        step += 1
        if step % RESET_EVERY == 0:
            base = copy.deepcopy(original_booster)

        prev_days  = unique_dates[i - lookback: i]
        train_data = df_feat[df_feat["date"].isin(prev_days)].copy()
        base = fine_tune(base, train_data, scaler_X, scaler_y)

        window_dates = unique_dates[i: i + HORIZON]
        window_data  = df_feat[df_feat["date"].isin(window_dates)].copy()

        X_s = pd.DataFrame(
            scaler_X.transform(window_data[cols_to_scale]),
            columns=cols_to_scale, index=window_data.index,
        )
        X_u = window_data[[c for c in columns_to_keep if c not in cols_to_scale]]
        X   = pd.concat([X_s, X_u], axis=1)[base.feature_names]

        pred_s = base.predict(xgb.DMatrix(X))
        pred   = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()

        window_data["y_pred_log"] = pred
        window_data["rho_pred"]   = window_data["msis_rho"] * np.exp(pred)
        all_preds.append(window_data[["date", "time", "rho_obs", "msis_rho", "rho_pred"]])

    return pd.concat(all_preds).reset_index(drop=True)


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def metrics(df: pd.DataFrame) -> dict:
    y, yhat = df["rho_obs"].values, df["rho_pred"].values
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    mape = float(np.mean(np.abs((yhat - y) / y)) * 100)
    bias = float(np.mean(yhat - y))
    log_rmse = float(np.sqrt(np.mean((np.log(yhat) - np.log(y)) ** 2)))
    k = max(1, int(np.ceil(0.05 * len(y))))
    top5 = float(np.mean(np.sort(np.abs(y - yhat))[-k:]))
    return {"rmse": rmse, "mape_pct": mape, "bias": bias, "log_rmse": log_rmse, "top5": top5}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Loading {DATA_FILE} ...")
    df = pd.read_parquet(DATA_FILE)

    if DATE_FILTER == "post2016":
        df = df[df["grace_time"] > "2016-01-01"]
    else:
        df = df[df["grace_time"] < "2009-06-06"]

    # Trim to eval window + enough leading days for the longest lookback
    buffer = pd.Timedelta(days=max(LOOKBACKS) + 1)
    df = df[
        (df["grace_time"] >= pd.Timestamp(EVAL_START, tz="UTC") - buffer) &
        (df["grace_time"] <= pd.Timestamp(EVAL_END, tz="UTC"))
    ]

    # Feature engineering
    df["time"] = df["grace_time"]
    df = ff.add_lst_doy_features(df)
    df["lon_sin"]          = np.sin(np.deg2rad(df["lon"]))
    df["lon_cos"]          = np.cos(np.deg2rad(df["lon"]))
    df["lst_lat_sin"]      = df["lst_sin"] * df["lat"]
    df["vtec_matched_lag"]  = df["matched_tec_value"].shift(500)
    df["vtec_matched_lag2"] = df["matched_tec_value"].shift(17280)
    df[TARGET_COL]         = np.log(df["rho_obs"] / df["msis_rho"])
    df = df.dropna(subset=cols_to_scale + [TARGET_COL])

    scaler_X = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)

    base_model = xgb.Booster()
    base_model.load_model(MODEL_FILE)

    # Run for each lookback
    results = {}
    for lb in LOOKBACKS:
        print(f"\n--- Lookback = {lb} days ---")
        pred_df = run_lookback(df, base_model, scaler_X, scaler_y, lb)
        results[lb] = {"metrics": metrics(pred_df), "preds": pred_df}
        m = results[lb]["metrics"]
        print(f"  RMSE={m['rmse']:.4e}  MAPE={m['mape_pct']:.2f}%  "
              f"bias={m['bias']:.4e}  log-RMSE={m['log_rmse']:.4f}  top5={m['top5']:.4e}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Lookback':>10} {'RMSE':>12} {'MAPE (%)':>10} {'Bias':>12} {'log-RMSE':>10} {'Top5':>12}")
    for lb in LOOKBACKS:
        m = results[lb]["metrics"]
        print(f"{lb:>10} {m['rmse']:>12.4e} {m['mape_pct']:>10.2f} "
              f"{m['bias']:>12.4e} {m['log_rmse']:>10.4f} {m['top5']:>12.4e}")

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    lbs = LOOKBACKS
    for ax, key, label in zip(axes, ["rmse", "mape_pct", "top5"], ["RMSE", "MAPE (%)", "Top-5% error"]):
        vals = [results[lb]["metrics"][key] for lb in lbs]
        bars = ax.bar([str(lb) for lb in lbs], vals, color=["steelblue", "darkorange", "seagreen"])
        ax.bar_label(bars, fmt="%.4g", padding=3)
        ax.set_xlabel("Lookback (days)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs lookback window")
    fig.suptitle(f"Lookback sensitivity — {DATE_FILTER}, horizon={HORIZON}d")
    fig.tight_layout()
    plt.savefig("lookback_sensitivity.png", dpi=150)
    plt.show()
    print("\nSaved → lookback_sensitivity.png")
