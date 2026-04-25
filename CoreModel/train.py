# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
train.py
- Loads GRACE merged data, engineers features including TEC lags and interaction terms.
- Splits into train/val/test using cyclic time blocks and scales features.
- Trains XGBoost (native API) to predict log(rho_obs/msis_rho); saves model and scalers.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.callback import EarlyStopping, LearningRateScheduler

import feature_functions as ff

from config import (
    PARQUET_FILE, MODEL_OUT, SCALER_X_OUT, SCALER_Y_OUT,
    TIME_MIN, TIME_MAX, TARGET, FEATURES, COLS_TO_SCALE,
)
from plotting import (
    plot_feature_distributions, plot_split_targets, plot_training_curve,
)


def lr_scheduler(current_round: int) -> float:
    initial_lr = 5e-4
    decay_factor = 0.8
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
    df = pd.read_parquet(parquet_file)
    df["time"] = pd.to_datetime(df["grace_time"])
    df = df[(df["time"] > TIME_MIN) & (df["time"] < TIME_MAX)].sort_values("time")

    df = ff.add_lst_doy_features(df)
    df["lon_sin"]           = np.sin(np.deg2rad(df["lon"]))
    df["lon_cos"]           = np.cos(np.deg2rad(df["lon"]))
    df["lst_lat_sin"]       = df["lst_sin"] * df["lat"]
    df["vtec_matched_lag"]  = df["matched_tec_value"].shift(500)
    df["vtec_matched_lag2"] = df["matched_tec_value"].shift(17280)
    df["log_ratio"]         = np.log(df["rho_obs"] / df["msis_rho"])
    return df.dropna()


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
            n_cycles=8,
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
    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, scaler_X, scaler_y = \
        ff.scale_simple(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            cols_to_scale=COLS_TO_SCALE,
        )

    joblib.dump(scaler_X, SCALER_X_OUT)
    joblib.dump(scaler_y, SCALER_Y_OUT)

    # 5. Train (native API — matches paper hyperparameters)
    dtrain = xgb.DMatrix(X_train_s, label=y_train_s[TARGET])
    dtest  = xgb.DMatrix(X_test_s,  label=y_test_s[TARGET])

    params = {
        "max_depth":        4,
        "min_child_weight": 300,
        "subsample":        0.5,
        "colsample_bytree": 0.6,
        "eval_metric":      ["rmse"],
        "base_score":       float(y_train_s[TARGET].mean()),
        "tree_method":      "hist",
        "nthread":          -1,
    }

    evals_result = {}
    callbacks = [
        LearningRateScheduler(lr_scheduler),
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
