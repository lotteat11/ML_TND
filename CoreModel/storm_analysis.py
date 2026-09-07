# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
storm_analysis.py
- Loads data around the 2016-06-14 geomagnetic storm (Ap = 94).
- Fine-tunes the model on the 5 days before the plot window.
- Plots 7 days of observed / MSIS / prediction with Ap index below.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import xgboost as xgb

import feature_functions as ff
from config import (PARQUET_FILE, MODEL_OUT, SCALER_X_OUT, SCALER_Y_OUT,
                    FEATURES, COLS_TO_SCALE, TARGET, TEC_LAGS, TEC_LAG_COLS)

STORM_DATE   = pd.Timestamp("2016-02-17", tz="UTC")
PLOT_START   = pd.Timestamp("2016-02-11", tz="UTC")
PLOT_END     = pd.Timestamp("2016-02-19", tz="UTC")
# Load extra days for the 24-hour TEC lag and the five-day fine-tuning window.
LOAD_START   = PLOT_START - pd.Timedelta(days=5 + 4)


def fine_tune(booster, train_df, scaler_X, scaler_y):
    K = 6
    block_size = max(1, len(train_df) // K)
    np.random.seed(42)
    tr_idx = np.random.choice(np.arange(K), size=4, replace=False)
    vl_idx = np.setdiff1d(np.arange(K), tr_idx)

    def prep(chunk):
        Xs = pd.DataFrame(scaler_X.transform(chunk[COLS_TO_SCALE]),
                          columns=COLS_TO_SCALE, index=chunk.index)
        Xu = chunk[[c for c in FEATURES if c not in COLS_TO_SCALE]]
        return pd.concat([Xs, Xu], axis=1)[booster.feature_names]

    tr = pd.concat([train_df.iloc[i * block_size:(i+1) * block_size] for i in tr_idx])
    vl = pd.concat([train_df.iloc[i * block_size:(i+1) * block_size] for i in vl_idx])

    dtrain = xgb.DMatrix(prep(tr), label=scaler_y.transform(tr[[TARGET]]).ravel())
    dval   = xgb.DMatrix(prep(vl), label=scaler_y.transform(vl[[TARGET]]).ravel())

    updated = xgb.train(
        {"objective": "reg:squarederror", "eval_metric": "rmse"},
        dtrain, num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        xgb_model=booster,
        callbacks=[xgb.callback.EarlyStopping(rounds=300, save_best=True)],
        verbose_eval=False,
    )
    updated.save_model("tmp_storm.json")
    b = xgb.Booster(); b.load_model("tmp_storm.json")
    return b


if __name__ == "__main__":

    # Load data
    print(f"Loading {PARQUET_FILE} ...")
    df = pd.read_parquet(PARQUET_FILE)
    df["time"] = pd.to_datetime(df["grace_time"], utc=True)
    df = df[(df["time"] >= LOAD_START) & (df["time"] <= PLOT_END)].sort_values("time").reset_index(drop=True)
    print(f"{len(df):,} rows  |  {df['time'].min().date()} → {df['time'].max().date()}")

    # Feature engineering
    df = ff.add_lst_doy_features(df)
    df["lon_sin"]           = np.sin(np.deg2rad(df["lon"]))
    df["lon_cos"]           = np.cos(np.deg2rad(df["lon"]))
    df["lst_lat_sin"]       = df["lst_sin"] * df["lat"]
    df = ff.add_tec_time_lag_features(df, lags=TEC_LAGS, names=TEC_LAG_COLS)
    df[TARGET]              = np.log(df["rho_obs"] / df["msis_rho"])
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    df["date"] = df["time"].dt.date
    df_plot    = df[df["time"] >= PLOT_START].copy()
    print(f"Plot rows: {len(df_plot):,}")

    # Load model and scalers
    scaler_X      = joblib.load(SCALER_X_OUT)
    scaler_y      = joblib.load(SCALER_Y_OUT)
    base_booster  = xgb.Booster()
    base_booster.load_model(str(MODEL_OUT))

    # Rolling day-by-day: fine-tune on previous 5 days, predict current day
    # Model carries forward between days (reset every 7 steps like on_track.py)
    unique_dates = sorted(df_plot["date"].unique())
    all_preds = []
    current_model = copy.deepcopy(base_booster)

    for i, day in enumerate(unique_dates):
        if i % 7 == 0:
            current_model = copy.deepcopy(base_booster)

        lookback_start = pd.Timestamp(day, tz="UTC") - pd.Timedelta(days=5)
        lookback_end   = pd.Timestamp(day, tz="UTC")
        df_finetune    = df[(df["time"] >= lookback_start) & (df["time"] < lookback_end)].copy()

        print(f"  {day}  fine-tune rows: {len(df_finetune):,}", end="  ")
        current_model = fine_tune(copy.deepcopy(current_model), df_finetune, scaler_X, scaler_y)

        df_day = df_plot[df_plot["date"] == day].copy()
        X = df_day[FEATURES].copy()
        X[COLS_TO_SCALE] = scaler_X.transform(X[COLS_TO_SCALE])
        pred_log = scaler_y.inverse_transform(
            current_model.predict(xgb.DMatrix(X[current_model.feature_names])).reshape(-1, 1)
        ).ravel()
        df_day["rho_pred"] = df_day["msis_rho"] * np.exp(pred_log)
        all_preds.append(df_day)
        print("done")

    df_plot = pd.concat(all_preds).reset_index(drop=True)

    # Plot
    step = max(1, len(df_plot) // 3000)
    dp = df_plot.iloc[::step]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(dp["time"], dp["rho_obs"],  color="black",      lw=0.8, label="Observed")
    ax1.plot(dp["time"], dp["msis_rho"], color="steelblue",  lw=1.0, label="MSIS")
    ax1.plot(dp["time"], dp["rho_pred"], color="darkorange", lw=1.0, label="Prediction")
    ax1.axvline(STORM_DATE, color="red", lw=1.2, ls="--", alpha=0.7, label=f"Storm peak ({STORM_DATE.date()})")
    ax1.set_yscale("log")
    ax1.set_ylabel("Density [kg m$^{-3}$]")
    ax1.set_title(f"Geomagnetic storm {STORM_DATE.date()}  (peak Ap = {dp['ap_m3h'].max():.0f})")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(dp["time"], dp["ap_m3h"], color="crimson", alpha=0.5, label="Ap (3-hr lag)")
    ax2.axvline(STORM_DATE, color="red", lw=1.2, ls="--", alpha=0.7)
    ax2.set_ylabel("Ap index")
    ax2.set_xlabel("Time (UTC)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.tight_layout()
    plt.savefig("storm_analysis.png", dpi=150)
    plt.show()
    print("Saved → storm_analysis.png")

    # Metrics
    print("\n=== Storm-period metrics ===")
    print(f"{'':>14} {'RMSE':>12} {'MAPE (%)':>10} {'Bias':>12} {'log-RMSE':>10} {'Top-5%':>12}")
    df_metrics = df_plot[df_plot["time"].dt.date == STORM_DATE.date()].copy()
    print(f"  (metrics on storm day only: {STORM_DATE.date()}, n={len(df_metrics):,})")
    obs = df_metrics["rho_obs"].values

    results = {}
    for label, pred in [("MSIS", df_metrics["msis_rho"].values), ("Prediction", df_metrics["rho_pred"].values)]:
        mask = (obs > 0) & (pred > 0)
        o, p = obs[mask], pred[mask]
        k = max(1, int(np.ceil(0.05 * len(o))))
        results[label] = {
            "rmse":     np.sqrt(np.mean((p - o) ** 2)),
            "mape":     np.mean(np.abs((p - o) / o)) * 100,
            "bias":     np.mean(p - o),
            "log_rmse": np.sqrt(np.mean((np.log(p) - np.log(o)) ** 2)),
            "top5":     np.mean(np.sort(np.abs(p - o))[-k:]),
        }
        m = results[label]
        print(f"  {label:>12}  {m['rmse']:>12.3e} {m['mape']:>10.2f} {m['bias']:>12.3e} {m['log_rmse']:>10.4f} {m['top5']:>12.3e}")

    print(f"\n  {'Improvement':>12}  ", end="")
    msis, pred = results["MSIS"], results["Prediction"]
    for key in ["rmse", "mape", "log_rmse", "top5"]:
        imp = (msis[key] - pred[key]) / msis[key] * 100
        print(f"  {imp:>+.1f}%", end="   ")
    print(f"\n  (positive = model better than MSIS)")
