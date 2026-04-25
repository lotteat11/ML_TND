"""
Evaluation script for the XGBoost density correction model.

Loads the saved model and scalers, runs predictions on the validation
and test sets, and produces all diagnostic plots.

Run:
    python evaluate.py

Requires that train.py has already been run to produce:
    xgb_model_test.json, scaler_xgboost_X_test.joblib, scaler_xgboost_y_test.joblib
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

import Feature_functions as ff

from config import (
    PARQUET_FILE, MODEL_OUT, SCALER_X_OUT, SCALER_Y_OUT,
    TIME_MIN, TIME_MAX, TARGET, FEATURES, COLS_TO_SCALE,
)
from plotting import (
    plot_val_densities_with_metrics,
    plot_density_hist2d,
    plot_error_map,
    plot_residual_diagnostics,
)
from train import load_and_engineer


if __name__ == "__main__":

    # 1. Reload data and reproduce splits (same seed / order as train.py)
    df_feat = load_and_engineer(PARQUET_FILE)

    X = df_feat[FEATURES]
    y = df_feat[[TARGET]]

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

    # 2. Load scalers and transform (no refitting — uses the scalers from training)
    scaler_X = joblib.load(SCALER_X_OUT)
    scaler_y = joblib.load(SCALER_Y_OUT)

    X_val_s  = ff.scale_transform(X_val,  scaler_X, COLS_TO_SCALE)
    X_test_s = ff.scale_transform(X_test, scaler_X, COLS_TO_SCALE)
    y_val_s  = ff.scale_y_transform(y_val,  scaler_y)
    y_test_s = ff.scale_y_transform(y_test, scaler_y)

    # 3. Load model and predict
    model = xgb.XGBRegressor()
    model.load_model(MODEL_OUT)

    y_pred_val_s  = model.predict(X_val_s)
    y_pred_test_s = model.predict(X_test_s)

    rmse_val  = np.sqrt(mean_squared_error(y_val_s[TARGET],  y_pred_val_s))
    rmse_test = np.sqrt(mean_squared_error(y_test_s[TARGET], y_pred_test_s))
    print(f"Validation RMSE (log space): {rmse_val:.4f}")
    print(f"Test       RMSE (log space): {rmse_test:.4f}")

    # 4. Residual diagnostics (scaled space)
    plot_residual_diagnostics(
        np.asarray(y_val_s[TARGET]).ravel(),
        y_pred_val_s.ravel(),
    )

    # 5. Back-transform to physical density
    y_pred_val = ff.unscale_y_pred(y_pred_val_s, scaler_y, y_val_s)

    df_val = df_feat.loc[idx_val].copy()
    df_val["y_true_log"] = y_val[TARGET].values.ravel()
    df_val["y_pred_log"] = np.asarray(y_pred_val).ravel()
    df_val["rho_msis"]   = df_val["tnd_kg_m3"]
    df_val["rho_pred"]   = df_val["rho_msis"] * np.exp(df_val["y_pred_log"])
    if "rho_obs" in df_val.columns:
        df_val["ratio_pred"] = df_val["rho_pred"] / df_val["rho_obs"]

    # 6. Density time-series and parity plots
    plot_val_densities_with_metrics(df_val, sample_step=10)
    ff.plot_val_densities_with_metrics(df_val, time_col="time", sample_step=1,
                                       obs_col="rho_obs", msis_col="msis_rho",
                                       pred_col="rho_pred")

    # 7. 2D histograms
    plot_density_hist2d(df_val, obs_col="rho_obs", pred_col="rho_pred",
                        ymin=1e-13, ymax=1e-11, count_max=1e7)
    plot_density_hist2d(df_val, obs_col="rho_obs", pred_col="msis_rho",
                        ymin=1e-13, ymax=1e-11, count_max=1e7)

    # 8. Error maps
    df_val["diff"]       = (df_val["rho_obs"] - df_val["rho_pred"]).abs()
    df_val["diff_nmsis"] = (df_val["rho_obs"] - df_val["tnd_kg_m3"]).abs()

    df_val_small = df_val.iloc[::100]
    for xcol, ycol in [("lat", "alt_km"), ("rho_obs", "alt_km"), ("rho_obs", "ap_m3h")]:
        plot_error_map(df_val_small, xcol, ycol,
                       y_true="rho_obs", y_pred="rho_pred",
                       error_type="rel", cmap="seismic")
        plot_error_map(df_val_small, xcol, ycol,
                       y_true="rho_obs", y_pred="msis_rho",
                       error_type="rel", cmap="seismic")

    # 9. Threshold distribution plots
    for diff_col, cols in [
        ("diff",       ["lat", "alt_km", "f107", "rho_obs", "diff"]),
        ("diff_nmsis", ["lat", "alt_km", "f107", "rho_obs", "diff_nmsis"]),
    ]:
        ff.plot_distribution_by_threshold(
            df_val,
            columns=cols,
            threshold="9E-14",
            threshold_col=diff_col,
            bins=30, density=True,
            gt_color="darkorange", le_color="seagreen",
        )
