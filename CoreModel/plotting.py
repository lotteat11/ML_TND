# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
Plotting functions for the XGBoost density correction pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


def plot_split_targets(idx_train, idx_val, idx_test,
                       y_train, y_val, y_test, target: str = "log_ratio") -> None:
    """Plot target values for each split to verify temporal coverage."""
    plt.figure(figsize=(12, 5))
    plt.plot(idx_val[::100],   y_val[target][::100],   ".", markersize=2, label="Validation")
    plt.plot(idx_test[::100],  y_test[target][::100],  ".", markersize=2, label="Test")
    plt.plot(idx_train[::100], y_train[target][::100], ".", markersize=2, label="Train")
    plt.xlabel("Sample index")
    plt.ylabel(f"Target ({target})")
    plt.title("Train / Test / Validation splits")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(X_train, X_val, X_test, sample_step: int = 520) -> None:
    """KDE per feature, comparing train / val / test distributions."""
    n_features = X_train.shape[1]
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols
    plt.figure(figsize=(5 * ncols, 4 * nrows))
    for i, col in enumerate(X_train.columns):
        plt.subplot(nrows, ncols, i + 1)
        sns.kdeplot(X_train.iloc[::sample_step][col], label="Train", fill=True, alpha=0.4)
        sns.kdeplot(X_val.iloc[::sample_step][col],   label="Val",   fill=True, alpha=0.4)
        sns.kdeplot(X_test.iloc[::sample_step][col],  label="Test",  fill=True, alpha=0.4)
        plt.title(col, fontsize=10)
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def _compute_density_metrics(obs: np.ndarray, model: np.ndarray) -> dict:
    """Return a dict of linear- and log-space metrics."""
    log_obs   = np.log(obs)
    log_model = np.log(model)
    rmse_lin  = np.sqrt(mean_squared_error(obs, model))
    rmse_log  = np.sqrt(mean_squared_error(log_obs, log_model))
    p95_lin   = np.percentile(np.abs(obs - model), 95)
    p95_log   = np.percentile(np.abs(log_obs - log_model), 95)
    mape      = np.mean(np.abs((obs - model) / obs)) * 100.0
    r2        = r2_score(obs, model)
    r, _      = pearsonr(obs, model)
    return dict(rmse_lin=rmse_lin, rmse_log=rmse_log,
                p95_lin=p95_lin,   p95_log=p95_log,
                mape=mape, r2=r2, r=r)


def _metric_label(name: str, m: dict) -> str:
    fe = np.exp(m["rmse_log"]); fe95 = np.exp(m["p95_log"])
    return (
        f"{name}\n"
        f"RMSE(log)={m['rmse_log']:.3f} (×{fe:.2f})\n"
        f"P95(log)={m['p95_log']:.3f} (×{fe95:.2f})\n"
        f"RMSE={m['rmse_lin']:.2e}, P95={m['p95_lin']:.2e}\n"
        f"MAPE={m['mape']:.1f}%, R²={m['r2']:.3f}, r={m['r']:.3f}"
    )


def plot_val_densities_with_metrics(
    df_val: pd.DataFrame,
    time_col: str = "time",
    obs_col: str  = "rho_obs",
    msis_col: str = "rho_msis",
    pred_col: str = "rho_pred",
    sample_step: int = 1,
    parity_alpha: float = 0.5,
) -> None:
    """Time-series and parity plot comparing observed, MSIS, and predicted density."""
    d = df_val[[time_col, obs_col, msis_col, pred_col]].dropna().copy()
    if sample_step > 1:
        d = d.iloc[::sample_step]

    t    = d[time_col]
    obs  = d[obs_col].to_numpy()
    msis = d[msis_col].to_numpy()
    pred = d[pred_col].to_numpy()

    mask = (obs > 0) & (msis > 0) & (pred > 0)
    obs, msis, pred, t = obs[mask], msis[mask], pred[mask], t[mask]

    m_msis = _compute_density_metrics(obs, msis)
    m_pred = _compute_density_metrics(obs, pred)

    # Time series
    plt.figure(figsize=(12, 5))
    plt.plot(t, obs,  label="Observed",  color="k",  lw=1.0, alpha=0.8)
    plt.plot(t, msis, label="MSIS",      color="C1", lw=1.0, alpha=0.8)
    plt.plot(t, pred, label="Predicted", color="C0", lw=1.0, alpha=0.8)
    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("Density [kg m$^{-3}$] (log scale)")
    plt.title("Validation: Observed vs MSIS vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Parity plot
    vmax = np.nanmax([obs.max(), msis.max(), pred.max()])
    vmin = max(1e-15, np.nanmin([obs.min(), msis.min(), pred.min()]))

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(obs, msis, s=8, alpha=parity_alpha, color="C1",
                label=_metric_label("MSIS",      m_msis))
    plt.scatter(obs, pred, s=8, alpha=parity_alpha, color="C0",
                label=_metric_label("Predicted", m_pred))
    plt.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)
    plt.xlabel("Observed density [kg m$^{-3}$]")
    plt.ylabel("Modeled density [kg m$^{-3}$]")
    plt.title("Parity (Validation)")
    plt.legend(loc="lower right", framealpha=0.9, fontsize="small")
    plt.tight_layout()
    plt.show()

    for name, m in [("MSIS", m_msis), ("Pred", m_pred)]:
        print(
            f"{name} : RMSE_log={m['rmse_log']:.3f} ×{np.exp(m['rmse_log']):.2f} | "
            f"P95_log={m['p95_log']:.3f} | RMSE={m['rmse_lin']:.3e} | "
            f"MAPE={m['mape']:.1f}% | R²={m['r2']:.3f}"
        )


def plot_training_curve(history: dict) -> None:
    """Plot RMSE training and validation curves from evals_result dict."""
    train_key = list(history.keys())[0]
    val_key   = list(history.keys())[1]
    metric    = "rmse" if "rmse" in history[train_key] else list(history[train_key].keys())[0]
    plt.figure(figsize=(8, 5))
    plt.plot(history[train_key][metric], label=f"Train {metric.upper()}")
    plt.plot(history[val_key][metric],   label=f"Validation {metric.upper()}")
    plt.xlabel("Boosting rounds")
    plt.ylabel(f"{metric.upper()} (log space)")
    plt.title("XGBoost Training vs Validation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_density_hist2d(df: pd.DataFrame,
                        obs_col: str = "rho_obs", pred_col: str = "rho_pred",
                        bins: int = 1000, cmap: str = "viridis", figsize=(8, 6),
                        ymin=None, ymax=None, count_max=None) -> None:
    """2D log-log histogram of observed vs predicted density."""
    obs  = df[obs_col].dropna()
    pred = df[pred_col].dropna()
    mask = (obs > 0) & (pred > 0)
    obs, pred = obs[mask], pred[mask]

    plt.figure(figsize=figsize)
    _, _, _, im = plt.hist2d(obs, pred, bins=bins, cmap=cmap,
                             norm=LogNorm(vmin=1, vmax=count_max))
    plt.colorbar(im, label="Count (log scale)")
    lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=2, label="1-to-1")
    plt.xscale("log"); plt.yscale("log")
    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)
    plt.xlabel("Observed Density [kg m$^{-3}$]")
    plt.ylabel("Predicted Density [kg m$^{-3}$]")
    plt.title("2D Histogram: Observed vs Predicted (log counts)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_map(df: pd.DataFrame, col_x: str, col_y: str,
                   y_true: str = "y_true", y_pred: str = "y_pred",
                   error_type: str = "abs", cmap: str = "coolwarm",
                   s: int = 5, alpha: float = 0.5) -> None:
    """Scatter plot coloured by prediction error across two feature axes."""
    residual = df[y_true] - df[y_pred]
    err   = residual.abs() if error_type == "abs" else residual
    label = "Absolute Error" if error_type == "abs" else "Residual (y_true - y_pred)"

    plt.figure(figsize=(8, 6))
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-2e-12, vmax=2e-12)
    sc   = plt.scatter(df[col_x], df[col_y], c=err, cmap=cmap, s=s, alpha=alpha, norm=norm)
    plt.colorbar(sc, label=label)
    plt.xlabel(col_x); plt.ylabel(col_y)
    plt.title(f"{label} across {col_x} vs {col_y}")
    plt.tight_layout()
    plt.show()


def plot_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray,
                               sample_size: int = 31_100_000) -> None:
    """Residual histogram, scatter, hexbin, and rolling MAE."""
    res = y_true - y_pred
    idx = np.random.choice(len(res), size=min(sample_size, len(res)), replace=False)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f}")

    plt.figure(figsize=(8, 4))
    plt.hist(res[idx], bins=200)
    plt.title("Residual distribution (y_true - y_pred)")
    plt.xlabel("Residual"); plt.ylabel("Count"); plt.show()

    plt.figure(figsize=(8, 4))
    plt.scatter(y_pred[idx], res[idx], s=1, alpha=0.3)
    plt.axhline(0, ls="--")
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.show()

    plt.figure(figsize=(8, 4))
    hb = plt.hexbin(y_pred[idx], res[idx], gridsize=440, cmap="viridis", bins="log", mincnt=1)
    plt.axhline(0, ls="--", color="red")
    plt.colorbar(hb, label="log10(N points)")
    plt.title("Residuals vs Predicted (hexbin density)")
    plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.show()

    window = 50_000
    roll = pd.Series(np.abs(res)).rolling(window, min_periods=1).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(roll.values, lw=1)
    plt.title(f"Rolling mean |residual| (window={window})")
    plt.xlabel("Sample index"); plt.ylabel("Rolling MAE"); plt.show()
