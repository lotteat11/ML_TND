"""
Training script for the XGBoost density correction model.

Run:
    python train.py

Outputs:
    xgb_model_test.json          — trained model
    scaler_xgboost_X_test.joblib — feature scaler
    scaler_xgboost_y_test.joblib — target scaler
"""

import importlib

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

import Feature_functions as ff
importlib.reload(ff)

from config import (
    PARQUET_FILE, MODEL_OUT, SCALER_X_OUT, SCALER_Y_OUT,
    TIME_MIN, TIME_MAX, TARGET, FEATURES, COLS_TO_SCALE,
)
from plotting import (
    plot_feature_distributions, plot_split_targets, plot_training_curve,
)


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
    df["lon_sin"]   = np.sin(np.deg2rad(df["lon"]))
    df["lon_cos"]   = np.cos(np.deg2rad(df["lon"]))
    df["log_ratio"] = np.log(df["rho_obs"] / df["msis_rho"])
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
            n_cycles=7,
            gap_before_val=500,
            gap_before_test=500,
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

    # 5. Train
    model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.0004,
        max_depth=9,
        min_child_weight=4,
        subsample=1,
        base_score=float(y_train_s[TARGET].mean()),
        n_jobs=-1,
        eval_metric=["mae"],
    )
    model.fit(
        X_train_s, y_train_s[TARGET],
        eval_set=[(X_train_s, y_train_s[TARGET]), (X_test_s, y_test_s[TARGET])],
        verbose=50,
    )
    model.save_model(MODEL_OUT)
    print(f"Model saved → {MODEL_OUT}")

    # 6. Training curve
    plot_training_curve(model.evals_result())

    # 7. Feature importance
    feat_imp = pd.DataFrame({
        "feature":    X_train_s.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(feat_imp.head(13))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    xgb.plot_importance(model, importance_type="gain", max_num_features=20)
    plt.show()
