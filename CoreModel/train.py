# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
train.py
- Loads GRACE merged data, engineers features including TEC lags and interaction terms.
- Splits into train/val/test using cyclic time blocks and scales features.
- Trains XGBoost (native API) to predict log(rho_obs/msis_rho); saves model and scalers.
"""

import gc
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from xgboost.callback import EarlyStopping, LearningRateScheduler

import feature_functions as ff

from config import (
    PARQUET_FILE, MODEL_OUT, SCALER_X_OUT, SCALER_Y_OUT,
    TIME_MIN, TIME_MAX, TIME_EXCLUDE, TEC_LAG_MODE, TARGET, FEATURES, COLS_TO_SCALE,
)
from plotting import (
    plot_feature_distributions, plot_split_targets, plot_training_curve,
)


def lr_scheduler(current_round: int) -> float:
    initial_lr = 0.08068
    decay_factor = 0.86866
    step_size = 15
    lr = initial_lr * (decay_factor ** (current_round // step_size))
    if current_round % 50 == 0:
        print(f"Round {current_round}: LR = {lr:.8f}")
    return lr


def summarize(name: str, df: pd.DataFrame) -> dict:
    return {
        "name":       name,
        "n":          len(df),
        "alt_km_min": df["alt_km"].min(),
        "alt_km_max": df["alt_km"].max(),
        "f107_5_95":  df["f107"].quantile([0.05, 0.95]).values,
        "ap_5_95":    df["ap_m3h"].quantile([0.05, 0.95]).values,
    }


def load_and_engineer(parquet_file: str) -> pd.DataFrame:
    # Read only what feature engineering and evaluation need: the full file is
    # 82.5M rows x 31 columns (~20 GB in pandas), and reading it all before the
    # time filter is applied is enough to exhaust memory on its own.
    raw_needed = sorted({
        "grace_time", "lat", "lon", "alt_km", "lst_h",
        "rho_obs", "msis_rho", "matched_tec_value",
        "f107", "f107a",
        "ap_daily", "ap_0h", "ap_m3h", "ap_m6h", "ap_m9h",
        "ap_avg12_33h", "ap_avg36_57h",
    })
    available = pq.ParquetFile(parquet_file).schema.names
    df = pd.read_parquet(parquet_file,
                         columns=[c for c in raw_needed if c in available])
    df["time"] = pd.to_datetime(df["grace_time"])
    df = df[(df["time"] > TIME_MIN) & (df["time"] < TIME_MAX)].sort_values("time")

    df = ff.add_lst_doy_features(df)
    df["lon_sin"]           = np.sin(np.deg2rad(df["lon"]))
    df["lon_cos"]           = np.cos(np.deg2rad(df["lon"]))
    df["lst_lat_sin"]       = df["lst_sin"] * df["lat"]
    if TEC_LAG_MODE == "time":
        df = ff.add_tec_time_lag_features(df)
    else:
        df["vtec_matched_lag"]  = df["matched_tec_value"].shift(500)
        df["vtec_matched_lag2"] = df["matched_tec_value"].shift(17280)
    df["log_ratio"]         = np.log(df["rho_obs"] / df["msis_rho"])
    # Interior holdouts are dropped AFTER the shift-based lags are built, so
    # rows just after a holdout keep their physically correct TEC lag values.
    if TIME_EXCLUDE is not None:
        for lo, hi in TIME_EXCLUDE:
            n_before = len(df)
            df = df[(df["time"] < lo) | (df["time"] >= hi)]
            print(f"Interior holdout {lo} .. {hi}: removed {n_before - len(df):,} rows")
    df = df.dropna()

    # Drop unused intermediate columns and downcast float64 -> float32 (XGBoost
    # converts to float32 internally anyway, so this costs nothing). Both are
    # done IN PLACE: df[keep].copy() would duplicate an ~11 GB frame and needs
    # more headroom than it saves. msis_rho/rho_obs stay float64 — they are
    # ~1e-13 and the log-ratio target is derived from them above.
    # Keep LST in hours for split-coverage diagnostics (Table 2). It is not
    # passed to XGBoost unless explicitly listed in FEATURES; the model uses
    # the cyclic LST representation instead.
    keep = set(FEATURES) | {
        TARGET, "time", "grace_time", "msis_rho", "rho_obs", "lst_h",
    }
    for c in [c for c in df.columns if c not in keep]:
        del df[c]
    protected = {TARGET, "msis_rho", "rho_obs"}
    for c in df.columns:
        if c not in protected and df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


if __name__ == "__main__":

    # 1. Load & engineer
    df_feat = load_and_engineer(PARQUET_FILE)

    X = df_feat[FEATURES]
    y = df_feat[[TARGET]]

    # 2. Time-block split
    X_train, X_test, X_val, y_train, y_test, y_val, idx_train, idx_test, idx_val = \
        ff.timeblock_split_repeated(
            X, y,
            fractions=(2/3, 1/6, 1/6),
            n_cycles=16,
            gap_before_val=1100,
            gap_before_test=1100,
            order=("train", "test", "val"),
            copy=False,
        )

    for name, idx in [("TRAIN", idx_train), ("VAL", idx_val), ("TEST", idx_test)]:
        print(summarize(name, df_feat.loc[idx]))

    # 3. Diagnostics
    plot_feature_distributions(X_train, X_val, X_test)
    plot_split_targets(idx_train, idx_val, idx_test, y_train, y_val, y_test)

    # 4. Scale
    # scale_simple copies each split, so the unscaled X/splits are dead weight
    # from here on — on the full mission that is >10 GB held for nothing.
    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, scaler_X, scaler_y = \
        ff.scale_simple(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            cols_to_scale=COLS_TO_SCALE,
        )

    del X, y, X_train, X_val, X_test, df_feat
    gc.collect()

    joblib.dump(scaler_X, SCALER_X_OUT)
    joblib.dump(scaler_y, SCALER_Y_OUT)

    # 5. Train (native API — matches paper hyperparameters)
    dtrain = xgb.DMatrix(X_train_s, label=y_train_s[TARGET])
    dtest  = xgb.DMatrix(X_test_s,  label=y_test_s[TARGET])
    del X_train_s, X_test_s, X_val_s
    gc.collect()

    params = {
        "max_depth":        6,
        "min_child_weight": 14.637,
        "subsample":        0.643,
        "colsample_bytree": 0.694,
        "eval_metric":      ["rmse"],
        "base_score":       float(y_train_s[TARGET].mean()),
        "tree_method":      "hist",
        "nthread":          -1,
    }

    # Optional tuned hyperparameters (from CoreModel/tune.py) via
    # TRAIN_PARAMS_JSON — overrides tree params and the LR schedule.
    lr_schedule = lr_scheduler
    if os.environ.get("TRAIN_PARAMS_JSON"):
        import json
        with open(os.environ["TRAIN_PARAMS_JSON"]) as fh:
            tuned = json.load(fh)
        params.update({
            "max_depth":        int(tuned["max_depth"]),
            "min_child_weight": float(tuned["min_child_weight"]),
            "subsample":        float(tuned["subsample"]),
            "colsample_bytree": float(tuned["colsample_bytree"]),
        })
        _lr0  = float(tuned["learning_rate"])
        _dec  = float(tuned["lr_decay_factor"])
        _step = int(tuned["lr_step_size"])

        def lr_schedule(current_round: int) -> float:
            return _lr0 * (_dec ** (current_round // _step))

        print(f"Using tuned hyperparameters from {os.environ['TRAIN_PARAMS_JSON']}: {tuned}")

    evals_result = {}
    callbacks = [
        LearningRateScheduler(lr_schedule),
        EarlyStopping(rounds=30, save_best=True, data_name="val", metric_name="rmse"),
    ]

    model = xgb.train(
        params, dtrain,
        num_boost_round=1360,
        evals=[(dtrain, "train"), (dtest, "val")],
        evals_result=evals_result,
        callbacks=callbacks,
        verbose_eval=10,
    )
    model.save_model(MODEL_OUT)
    print(f"Model saved → {MODEL_OUT}")

    # 6. Training curve
    plot_training_curve(evals_result)

    # 7. Feature importance
    scores = model.get_score(importance_type="gain")
    feat_imp = pd.DataFrame({
        "feature":    list(scores.keys()),
        "importance": list(scores.values()),
    }).sort_values("importance", ascending=False)
    print(feat_imp.head(15))
    plt.figure(figsize=(8, 6))
    xgb.plot_importance(model, importance_type="gain", max_num_features=20)
    plt.show()
