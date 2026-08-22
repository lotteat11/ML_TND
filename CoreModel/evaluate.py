# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
evaluate.py
- Loads the saved model and scalers produced by train.py (run that first).
- Runs predictions on the validation and test splits, back-transforms to physical density.
- Produces residual diagnostics, parity plots, error maps, and threshold distribution plots.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

import feature_functions as ff

from config import (
    PARQUET_FILE, MODEL_OUT, SCALER_X_OUT, SCALER_Y_OUT,
    TIME_MIN, TIME_MAX, TARGET, FEATURES, COLS_TO_SCALE,
)
from plotting import (
    plot_val_densities_with_metrics,
    plot_density_hist2d,
    plot_error_map,
    plot_residual_diagnostics,
    _compute_density_metrics,
)
from train import load_and_engineer


def _score_split(df, time_col="time", obs_col="rho_obs",
                 msis_col="rho_msis", pred_col="rho_pred",
                 sample_step=1) -> dict:
    """Metrics for one split without drawing anything.

    Mirrors the filtering plot_val_densities_with_metrics applies (same dropna,
    same stride, same positivity mask) so val and test rows are directly
    comparable, and reuses its metric code rather than restating it.
    """
    d = df[[time_col, obs_col, msis_col, pred_col]].dropna()
    if sample_step > 1:
        d = d.iloc[::sample_step]
    obs  = d[obs_col].to_numpy()
    msis = d[msis_col].to_numpy()
    pred = d[pred_col].to_numpy()
    mask = (obs > 0) & (msis > 0) & (pred > 0)
    return {"MSIS": _compute_density_metrics(obs[mask], msis[mask]),
            "Pred": _compute_density_metrics(obs[mask], pred[mask])}


if __name__ == "__main__":

    # 1. Reload data and reproduce splits (same seed / order as train.py)
    df_feat = load_and_engineer(PARQUET_FILE)

    X = df_feat[FEATURES]
    y = df_feat[[TARGET]]

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

    # 2. Load scalers and transform (no refitting — uses the scalers from training)
    scaler_X = joblib.load(SCALER_X_OUT)
    scaler_y = joblib.load(SCALER_Y_OUT)

    X_val_s  = ff.scale_transform(X_val,  scaler_X, COLS_TO_SCALE)
    X_test_s = ff.scale_transform(X_test, scaler_X, COLS_TO_SCALE)
    y_val_s  = ff.scale_y_transform(y_val,  scaler_y)
    y_test_s = ff.scale_y_transform(y_test, scaler_y)

    # 3. Load model and predict
    model = xgb.Booster()
    model.load_model(MODEL_OUT)

    y_pred_val_s  = model.predict(xgb.DMatrix(X_val_s))
    y_pred_test_s = model.predict(xgb.DMatrix(X_test_s))

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
    def to_physical(idx, y_true, y_pred_s, y_s):
        """Scaled prediction -> physical density, on one split."""
        y_pred = ff.unscale_y_pred(y_pred_s, scaler_y, y_s)
        d = df_feat.loc[idx].copy()
        d["y_true_log"] = y_true[TARGET].values.ravel()
        d["y_pred_log"] = np.asarray(y_pred).ravel()
        d["rho_msis"]   = d["tnd_kg_m3"] if "tnd_kg_m3" in d.columns else d["msis_rho"]
        d["rho_pred"]   = d["rho_msis"] * np.exp(d["y_pred_log"])
        if "rho_obs" in d.columns:
            d["ratio_pred"] = d["rho_pred"] / d["rho_obs"]
        return d

    df_val  = to_physical(idx_val,  y_val,  y_pred_val_s,  y_val_s)
    df_test = to_physical(idx_test, y_test, y_pred_test_s, y_test_s)

    # 6. Density time-series and parity plots (figures are for the val split;
    # the test split is scored for the table only, so it does not overwrite
    # the val figures under figs/).
    SAMPLE_STEP = int(os.environ.get("EVAL_SAMPLE_STEP", "10"))
    val_metrics  = plot_val_densities_with_metrics(df_val,  sample_step=SAMPLE_STEP)
    test_metrics = _score_split(df_test, sample_step=SAMPLE_STEP)

    # 6b. Persist val AND test metrics. They used to exist only as text baked
    # into the parity-plot labels, so quoting r (or any of them) meant reading a
    # number off a PNG — and the test split was predicted but never scored in
    # physical space at all. Written next to the model so the file names the run.
    metrics_csv = os.environ.get(
        "EVAL_METRICS_CSV",
        f"eval_metrics_{os.path.splitext(os.path.basename(str(MODEL_OUT)))[0]}.csv",
    )
    rows = []
    for split, d, mm in [("val", df_val, val_metrics),
                         ("test", df_test, test_metrics)]:
        for name, m in mm.items():
            rows.append({"model": os.path.basename(str(MODEL_OUT)),
                         "split": split, "source": name,
                         "n": len(d), "sample_step": SAMPLE_STEP, **m})
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)

    print(f"\nWrote val + test metrics -> {metrics_csv}")
    for split, m in [("val", val_metrics), ("test", test_metrics)]:
        for name in ("MSIS", "Pred"):
            s = m[name]
            print(f"  {split:>4} {name:>4}: RMSE={s['rmse_lin']:.3e} "
                  f"RMSE_log={s['rmse_log']:.4f} MAPE={s['mape']:.2f}% "
                  f"R2={s['r2']:.4f} r={s['r']:.4f}")
    print(f"  (full-split log-space RMSE: val {rmse_val:.4f}, test {rmse_test:.4f}; "
          f"the metrics above are on every {SAMPLE_STEP}th row)")

    # 7. 2D histograms
    plot_density_hist2d(df_val, obs_col="rho_obs", pred_col="rho_pred",
                        ymin=1e-13, ymax=1e-11, count_max=1e7)
    plot_density_hist2d(df_val, obs_col="rho_obs", pred_col="msis_rho",
                        ymin=1e-13, ymax=1e-11, count_max=1e7)

    # 8. Error maps
    df_val["diff"]       = (df_val["rho_obs"] - df_val["rho_pred"]).abs()
    df_val["diff_nmsis"] = (df_val["rho_obs"] - df_val["msis_rho"]).abs()

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
