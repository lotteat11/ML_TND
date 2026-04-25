import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

PLOT_OUTPUT_DIR = "workflow_plots_xbg"


def save_plot(filename):
    """Saves the current matplotlib figure and closes it."""
    if not os.path.exists(PLOT_OUTPUT_DIR):
        os.makedirs(PLOT_OUTPUT_DIR)
    filepath = os.path.join(PLOT_OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plot to: {filepath}")


def simple_index_plot(df_test, y_pred="rho_pred", start_index=0, n_steps=100,
                      feature_list=None, y_target="rho_obs", Title="title"):
    """
    Plots target performance and key feature dynamics for a fixed number
    of steps (rows) starting at a specific index.
    """
    if feature_list is None:
        feature_list = ['ap_m3h', 'f107a']

    df_plot = df_test.copy()
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred, index=df_test.index)

    end_index = start_index + n_steps
    df_window = df_plot.iloc[start_index:end_index].copy()

    if df_window.empty:
        print(f"No data available from index {start_index} for {n_steps} steps.")
        return

    num_panels = 1 + len(feature_list)
    fig, axes = plt.subplots(num_panels, 1, figsize=(14, 3 * num_panels), sharex=False)
    if num_panels == 1:
        axes = [axes]

    if 'time' in df_window.columns:
        try:
            x_data = pd.to_datetime(df_window['time'], utc=True, errors='coerce')
            x_data = np.arange(start_index, start_index + n_steps) if x_data.isna().all() else x_data.to_numpy()
        except Exception:
            x_data = np.arange(start_index, start_index + n_steps)
    else:
        x_data = np.arange(start_index, start_index + n_steps)

    ax_target = axes[0]
    ax_target.plot(x_data, df_window[y_target], label='Observed (True)', color='black', linestyle='--')
    ax_target.plot(x_data, df_window[y_pred],   label='Prediction',      color='red',   linestyle='--')
    ax_target.plot(x_data, df_window['msis_rho'], label='MSIS',          color='blue',  linestyle='--')
    ax_target.set_title(f"Performance and Feature Dynamics (Rows {start_index} to {end_index-1})")
    ax_target.set_ylabel(y_target)
    ax_target.legend(loc='upper right')
    ax_target.grid(True, linestyle=':', alpha=0.6)

    print(pd.DataFrame({
        'MSIS_Rho':   df_window['msis_rho'],
        'Model_Pred': df_window[y_pred],
        'True_Obs':   df_window[y_target],
    }).head())

    for j, feature in enumerate(feature_list):
        ax_feature = axes[1 + j]
        ax_feature.plot(x_data, df_window[feature], label=feature, color='darkorange', alpha=0.8)
        ax_feature.set_ylabel(feature)
        ax_feature.grid(True, linestyle=':', alpha=0.6)
        if j == len(feature_list) - 1:
            ax_feature.tick_params(axis='x', rotation=45)
            ax_feature.set_xlabel("Index Label / Time Stamp")
        else:
            ax_feature.tick_params(axis='x', labelbottom=False)

    plt.tight_layout()

    out_dir = Path("figs/time_series")
    safe_title = "".join(c if c.isalnum() or c in "-." else "" for c in Title)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"time_series_dynamics{safe_title}.png"
    fig.savefig(png_path)
    print(f"Plot saved as '{png_path.name}'")

    pkl_path = out_dir / f"time_series_dynamics{safe_title}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"time_series": fig}, f)
    print(f"Figures saved as '{pkl_path.name}'")

    return fig, axes


def add_lst_doy_features(df: pd.DataFrame,
                         time_col="time",
                         lon_col="lon") -> pd.DataFrame:
    """
    Add Local Solar Time (LST) and Day-of-Year (DOY) features
    with sine/cosine encoding to a dataframe.
    """
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True)

    utc_hours = (
        d[time_col].dt.hour
        + d[time_col].dt.minute / 60.0
        + d[time_col].dt.second / 3600.0
    )
    lst_h = (utc_hours + d[lon_col] / 15.0) % 24.0
    d["lst_h"]   = lst_h
    d["lst_sin"] = np.sin(2 * np.pi * lst_h / 24.0)
    d["lst_cos"] = np.cos(2 * np.pi * lst_h / 24.0)

    doy = d[time_col].dt.dayofyear.astype(float)
    d["doy"]     = doy
    d["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)

    return d


def add_ap_lags(df, time_col="time", ap_col="ap_hourly", lags_hours=(3, 6, 12, 24)):
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True)

    ap_hour = (
        d[[time_col, ap_col]]
        .assign(hour=lambda x: x[time_col].dt.floor("H"))
        .groupby("hour", as_index=True)[ap_col]
        .mean()
        .sort_index()
        .asfreq("H")
        .ffill()
    )

    hour_key = d[time_col].dt.floor("h")
    for h in lags_hours:
        d[f"{ap_col}_lag{h}h"] = hour_key.map(ap_hour.shift(h))

    return d


def scale_simple(X_train, X_val, X_test,
                 y_train, y_val, y_test,
                 cols_to_scale, feature_range=(-1, 1)):
    """
    Fit scalers on training data and apply to all splits.
    Scales only `cols_to_scale` in X, and all columns in y.
    Returns scaled DataFrames and the fitted scalers.
    """
    scaler_X = MinMaxScaler(feature_range=feature_range)
    scaler_X.fit(X_train[cols_to_scale])

    def _scale_X(X):
        Xs = X.copy()
        Xs[cols_to_scale] = pd.DataFrame(
            scaler_X.transform(X[cols_to_scale]),
            columns=cols_to_scale,
            index=X.index,
        )
        return Xs

    X_train_s = _scale_X(X_train)
    X_val_s   = _scale_X(X_val)
    X_test_s  = _scale_X(X_test)

    scaler_y = MinMaxScaler(feature_range=feature_range)
    scaler_y.fit(y_train)

    def _scale_y(y):
        return pd.DataFrame(
            scaler_y.transform(y.values),
            columns=y.columns,
            index=y.index,
        )

    y_train_s = _scale_y(y_train)
    y_val_s   = _scale_y(y_val)
    y_test_s  = _scale_y(y_test)

    return (X_train_s, X_val_s, X_test_s,
            y_train_s, y_val_s, y_test_s,
            scaler_X, scaler_y)


def scale_transform(X, scaler_X, cols_to_scale):
    """Apply a pre-fitted X scaler without refitting."""
    Xs = X.copy()
    Xs[cols_to_scale] = pd.DataFrame(
        scaler_X.transform(X[cols_to_scale]),
        columns=cols_to_scale,
        index=X.index,
    )
    return Xs


def scale_y_transform(y, scaler_y):
    """Apply a pre-fitted y scaler without refitting."""
    return pd.DataFrame(
        scaler_y.transform(y.values),
        columns=y.columns,
        index=y.index,
    )


def plot_features_vs_index(df):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    axs[0, 0].plot(df.index, df['f107'],  label='f107')
    axs[0, 0].plot(df.index, df['f107a'], label='f107a')
    axs[0, 0].set_title('f107 & f107a vs Time')
    axs[0, 0].set_ylabel('Flux')
    axs[0, 0].legend()

    axs[0, 1].plot(df.index, df['alt_km'], color='tab:orange')
    axs[0, 1].set_title('Altitude vs Time')
    axs[0, 1].set_ylabel('Altitude (km)')

    axs[1, 0].plot(df.index, df['lat'], color='tab:green')
    axs[1, 0].set_title('Latitude vs Time')
    axs[1, 0].set_ylabel('Latitude')

    axs[1, 1].plot(df.index, df['Ap'],       label='ap')
    axs[1, 1].plot(df.index, df['ap_daily'], label='ap_daily')
    axs[1, 1].plot(df.index, df['Ap_lag3h'], label='ap lag 3')
    axs[1, 1].set_title('ap & ap_daily vs Time')
    axs[1, 1].set_ylabel('ap')
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


def plot_features_vs_time(df):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    axs[0, 0].plot(df['time'], df['f107'],  label='f107')
    axs[0, 0].plot(df['time'], df['f107a'], label='f107a')
    axs[0, 0].set_title('f107 & f107a vs Time')
    axs[0, 0].set_ylabel('Flux')
    axs[0, 0].legend()

    axs[0, 1].plot(df['time'], df['alt_km'], color='tab:orange')
    axs[0, 1].set_title('Altitude vs Time')
    axs[0, 1].set_ylabel('Altitude (km)')

    axs[1, 0].plot(df['time'], df['lat'], color='tab:green')
    axs[1, 0].set_title('Latitude vs Time')
    axs[1, 0].set_ylabel('Latitude')

    axs[1, 1].plot(df['time'], df['Ap'],       label='ap')
    axs[1, 1].plot(df['time'], df['ap_daily'], label='ap_daily')
    axs[1, 1].plot(df['time'], df['Ap_lag3h'], label='ap lag 3h')
    axs[1, 1].set_title('ap & ap_daily vs Time')
    axs[1, 1].set_ylabel('ap')
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


def build_and_train_regressor_cb(
    X_train, y_train,
    X_val, y_val,
    learning_rate=1e-3,
    epochs=200,
    batch_size=128,
    es_patience=10,
    rlrop_patience=5,
    rlrop_factor=0.5,
    min_lr=1e-5,
    monitor="val_loss",
    checkpoint_path="best.keras",
):
    import tensorflow as tf  # lazy import — avoids TF startup cost for XGBoost-only runs

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    X_val   = np.asarray(X_val,   dtype=np.float32)
    y_val   = np.asarray(y_val,   dtype=np.float32)

    n_features = X_train.shape[1]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(32,  activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mse", "mae"],
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=es_patience, restore_best_weights=True,
    )
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor, factor=rlrop_factor, patience=rlrop_patience,
        min_lr=min_lr, verbose=1,
    )
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor=monitor,
        save_best_only=True, mode="min", verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, rlrop, ckpt],
        verbose=1,
        shuffle=False,
    )

    model = tf.keras.models.load_model(checkpoint_path)
    return model, history


def plot_history_simple(history):
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE (loss)")
    plt.title("Training history")
    plt.legend(); plt.show()


def plot_parity_density(y_true, y_pred):
    plt.hexbin(y_true, y_pred, gridsize=40, cmap="viridis")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual (density)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar().set_label("Count")
    plt.show()


def plot_residuals(y_true, y_pred):
    res = y_pred - y_true
    plt.hist(res, bins=40)
    plt.axvline(0, color="r", linestyle="--")
    plt.xlabel("Residual (pred - actual)")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.show()


def plot_val_densities_with_metrics(
    df_val,
    time_col="time",
    obs_col="rho_obs",
    msis_col="rho_msis",
    pred_col="rho_pred",
    sample_step=1,
    parity_alpha=0.5,
):
    """
    Compare observed vs MSIS vs predicted densities on the validation set.
    Computes RMSE, Top-5% error, MAPE, R², and Pearson r in linear and log space.
    """
    cols = [time_col, obs_col, msis_col, pred_col]
    d = df_val[cols].dropna().copy()
    if sample_step > 1:
        d = d.iloc[::sample_step]

    t    = d[time_col]
    obs  = d[obs_col].to_numpy()
    msis = d[msis_col].to_numpy()
    pred = d[pred_col].to_numpy()

    mpos = (obs > 0) & (msis > 0) & (pred > 0)
    obs, msis, pred, t = obs[mpos], msis[mpos], pred[mpos], t[mpos]

    # Linear-space metrics
    rmse_msis_lin = np.sqrt(mean_squared_error(obs, msis))
    rmse_pred_lin = np.sqrt(mean_squared_error(obs, pred))
    mape_msis     = np.mean(np.abs((obs - msis) / obs)) * 100.0
    mape_pred     = np.mean(np.abs((obs - pred) / obs)) * 100.0
    r_msis, _     = pearsonr(obs, msis)
    r_pred, _     = pearsonr(obs, pred)
    r2_msis       = r2_score(obs, msis)
    r2_pred       = r2_score(obs, pred)

    # Log-space metrics
    log_obs  = np.log(obs)
    log_msis = np.log(msis)
    log_pred = np.log(pred)
    rmse_msis_log = np.sqrt(mean_squared_error(log_obs, log_msis))
    rmse_pred_log = np.sqrt(mean_squared_error(log_obs, log_pred))
    fe_msis = float(np.exp(rmse_msis_log))
    fe_pred = float(np.exp(rmse_pred_log))

    # Top-5% mean error
    n = obs.size
    k = max(1, int(np.ceil(0.05 * n)))
    abs_err_msis_lin = np.abs(obs - msis)
    abs_err_pred_lin = np.abs(obs - pred)
    abs_err_msis_log = np.abs(log_obs - log_msis)
    abs_err_pred_log = np.abs(log_obs - log_pred)
    top5_msis_lin = float(np.mean(np.sort(abs_err_msis_lin)[-k:]))
    top5_pred_lin = float(np.mean(np.sort(abs_err_pred_lin)[-k:]))
    top5_msis_log = float(np.mean(np.sort(abs_err_msis_log)[-k:]))
    top5_pred_log = float(np.mean(np.sort(abs_err_pred_log)[-k:]))
    fe_top5_msis  = float(np.exp(top5_msis_log))
    fe_top5_pred  = float(np.exp(top5_pred_log))

    msis_label = (
        f"MSIS\n"
        f"RMSE(log)={rmse_msis_log:.3f} (×{fe_msis:.2f})\n"
        f"Top5(log)={top5_msis_log:.3f} (×{fe_top5_msis:.2f})\n"
        f"RMSE={rmse_msis_lin:.2e}, Top5={top5_msis_lin:.2e}\n"
        f"MAPE={mape_msis:.1f}%, R²={r2_msis:.3f}, r={r_msis:.3f}"
    )
    pred_label = (
        f"Predicted\n"
        f"RMSE(log)={rmse_pred_log:.3f} (×{fe_pred:.2f})\n"
        f"Top5(log)={top5_pred_log:.3f} (×{fe_top5_pred:.2f})\n"
        f"RMSE={rmse_pred_lin:.2e}, Top5={top5_pred_lin:.2e}\n"
        f"MAPE={mape_pred:.1f}%, R²={r2_pred:.3f}, r={r_pred:.3f}"
    )

    vmax = np.nanmax([obs.max(), msis.max(), pred.max()])
    vmin = max(1e-15, np.nanmin([obs.min(), msis.min(), pred.min()]))

    fig1 = plt.figure(figsize=(12, 5))
    plt.plot(t, obs,  label="Observed",  color="k",  lw=1.0, alpha=0.8)
    plt.plot(t, msis, label="MSIS",      color="C1", lw=1.0, alpha=0.8)
    plt.plot(t, pred, label="Predicted", color="C0", lw=1.0, alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("Density [kg m$^{-3}$] (log scale)")
    plt.title("Validation: Observed vs MSIS vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig("val1.png", dpi=300)
    plt.show()

    fig2 = plt.figure(figsize=(6.5, 6.5))
    plt.scatter(obs, msis, s=8, alpha=parity_alpha, label=msis_label, color="C1")
    plt.scatter(obs, pred, s=8, alpha=parity_alpha, label=pred_label, color="C0")
    plt.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)
    plt.xlabel("Observed density [kg m$^{-3}$]")
    plt.ylabel("Modeled density [kg m$^{-3}$]")
    plt.title("Parity (Validation)")
    plt.legend(loc="lower right", framealpha=0.9, fontsize='small')
    plt.tight_layout()
    plt.savefig("density.png", dpi=300)
    plt.show()

    with open("val_density_plots.pkl", "wb") as f:
        pickle.dump({"time_series": fig1, "parity": fig2}, f)
    print("Figures saved as val_density_plots.pkl")

    print(
        f"MSIS  : RMSE_log={rmse_msis_log:.3f} ×{fe_msis:.2f} | Top5_log={top5_msis_log:.3f} ×{fe_top5_msis:.2f} | "
        f"RMSE={rmse_msis_lin:.3e} Top5={top5_msis_lin:.3e} | MAPE={mape_msis:.1f}% R²={r2_msis:.3f} r={r_msis:.3f}\n"
        f"Pred  : RMSE_log={rmse_pred_log:.3f} ×{fe_pred:.2f} | Top5_log={top5_pred_log:.3f} ×{fe_top5_pred:.2f} | "
        f"RMSE={rmse_pred_lin:.3e} Top5={top5_pred_lin:.3e} | MAPE={mape_pred:.1f}% R²={r2_pred:.3f} r={r_pred:.3f}"
    )


def unscale_y_pred(y_pred_s, scaler_y, like_y):
    """Inverse-transform scaled predictions back to original y-space."""
    y_pred_s = np.asarray(y_pred_s)
    n_targets = getattr(scaler_y, "n_features_in_", 1)
    if y_pred_s.ndim == 1:
        y_pred_s = y_pred_s.reshape(-1, n_targets)
    y_pred = scaler_y.inverse_transform(y_pred_s)
    return pd.DataFrame(y_pred, index=like_y.index, columns=like_y.columns)


def timeblock_split(
    X: pd.DataFrame,
    y,
    fractions: Tuple[float, float, float] = (2/3, 1/6, 1/6),
    order: Tuple[str, str, str] = ("train", "test", "val"),
    copy: bool = False,
):
    """Split X, y into three contiguous time-ordered blocks."""
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same number of rows (got {len(X)} and {len(y)})")

    roles = {"train", "val", "test"}
    if set(order) != roles:
        raise ValueError(f"`order` must be a permutation of {roles}, got {order}")

    f1, f2, f3 = fractions
    if any(f <= 0 for f in (f1, f2, f3)):
        raise ValueError("All fractions must be > 0.")
    if not 0.99 <= (f1 + f2 + f3) <= 1.01:
        raise ValueError("fractions must sum to ~1.0")

    n = len(X)
    if n < 3:
        raise ValueError("Need at least 3 rows to split into three blocks.")

    b1 = min(max(int(f1 * n), 1), n - 2)
    b2 = min(max(int((f1 + f2) * n), b1 + 1), n - 1)

    blocks_X   = (X.iloc[:b1],   X.iloc[b1:b2],   X.iloc[b2:])
    blocks_y   = (y.iloc[:b1],   y.iloc[b1:b2],   y.iloc[b2:])
    blocks_idx = (X.index[:b1],  X.index[b1:b2],  X.index[b2:])

    mapping = dict(zip(order, zip(blocks_X, blocks_y, blocks_idx)))
    X_train, y_train, idx_train = mapping["train"]
    X_val,   y_val,   idx_val   = mapping["val"]
    X_test,  y_test,  idx_test  = mapping["test"]

    if copy:
        X_train, y_train = X_train.copy(), y_train.copy()
        X_val,   y_val   = X_val.copy(),   y_val.copy()
        X_test,  y_test  = X_test.copy(),  y_test.copy()

    return X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test


def plot_columns_vs_time(df, columns, name="titel"):
    """Plots multiple columns against the 'time' column."""
    plt.figure(figsize=(10, 6))
    for col in columns:
        plt.plot(df['time'], df[col], label=col)
    plt.xlabel('Time'); plt.ylabel('Values'); plt.title(name)
    plt.legend(); plt.grid(True); plt.tight_layout()
    filename = f"{name.replace(' ', '_')}_plot.png"
    plt.savefig(filename)
    print(f"Plot saved as '{filename}'")


def plot_two_columns_vs_time(df, value1, value2, name="titel"):
    """Plots two columns against the 'time' column."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df[value1], label=value1)
    plt.plot(df['time'], df[value2], label=value2)
    plt.xlabel('Time'); plt.ylabel('Values'); plt.title(name)
    plt.legend(); plt.grid(True); plt.tight_layout()
    filename = f"{name.replace(' ', '_')}_plot.png"
    plt.savefig(filename)
    print(f"Plot saved as '{filename}'")


def plot_distribution_by_threshold(
    df,
    columns,
    threshold,
    threshold_col=None,
    time_col='time',
    bins=30,
    density=False,
    alpha=0.8,
    gt_color='darkorange',
    le_color='seagreen',
):
    """
    Plot distributions of selected columns split by a threshold.

    VALUE MODE: if `threshold_col` is provided, split by (threshold_col > threshold).
    TIME MODE:  if `threshold_col` is None, split by datetime on `time_col`.
    """
    def _parse_float(x):
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        if isinstance(x, str):
            return float(x.strip().replace(',', '.'))
        raise TypeError(f"Cannot parse threshold {x!r} to float.")

    if threshold_col is not None:
        thr = _parse_float(threshold)
        key = pd.to_numeric(df[threshold_col], errors='coerce')
        mask_gt = key > thr
        mask_le = ~mask_gt

        fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(8, 3.5 * len(columns)))
        if len(columns) == 1:
            axes = [axes]

        for ax, col in zip(axes, columns):
            v_gt = df.loc[mask_gt, col].dropna()
            v_le = df.loc[mask_le, col].dropna()
            all_vals = pd.concat([v_gt, v_le], ignore_index=True)
            if len(all_vals) == 0:
                ax.set_title(f"Distribution of {col} (no data)")
                ax.set_xlabel(col); ax.set_ylabel("Density" if density else "Frequency")
                ax.grid(alpha=0.3); continue
            bin_edges = np.histogram_bin_edges(all_vals, bins=bins) if isinstance(bins, int) else bins
            ax.hist(v_le, bins=bin_edges, color=le_color, alpha=alpha,
                    label=f"{threshold_col} ≤ {thr:g}", density=density)
            ax.hist(v_gt, bins=bin_edges, color=gt_color, alpha=alpha,
                    label=f"{threshold_col} > {thr:g}", density=density)
            ax.set_title(f"Distribution of {col} split by {threshold_col} threshold ({thr:g})")
            ax.set_xlabel(col); ax.set_ylabel("Density" if density else "Frequency")
            ax.grid(alpha=0.3)

        axes[0].legend()
        plt.tight_layout(); plt.show()
        return {'mode': 'value', 'threshold': thr, 'threshold_col': threshold_col,
                'n_gt': int(mask_gt.sum()), 'n_le': int(mask_le.sum())}

    time_series    = pd.to_datetime(df[time_col], errors='coerce')
    threshold_time = pd.to_datetime(threshold)

    if time_series.dt.tz is not None and threshold_time.tzinfo is None:
        threshold_time = threshold_time.tz_localize(time_series.dt.tz)
    elif time_series.dt.tz is None and threshold_time.tzinfo is not None:
        threshold_time = threshold_time.tz_convert(None)

    mask_ge = time_series >= threshold_time
    mask_lt = ~mask_ge

    fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(8, 3.5 * len(columns)))
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        v_ge = df.loc[mask_ge, col].dropna()
        v_lt = df.loc[mask_lt, col].dropna()
        all_vals = pd.concat([v_ge, v_lt], ignore_index=True)
        if len(all_vals) == 0:
            ax.set_title(f"Distribution of {col} (no data)")
            ax.set_xlabel(col); ax.set_ylabel("Density" if density else "Frequency")
            ax.grid(alpha=0.3); continue
        bin_edges = np.histogram_bin_edges(all_vals, bins=bins) if isinstance(bins, int) else bins
        ax.hist(v_lt, bins=bin_edges, color='darkorange', alpha=alpha,
                label='< threshold time', density=density)
        ax.hist(v_ge, bins=bin_edges, color='teal', alpha=alpha,
                label='≥ threshold time', density=density)
        ax.set_title(f"Distribution of {col} by Time Threshold @ {threshold_time}")
        ax.set_xlabel(col); ax.set_ylabel("Density" if density else "Frequency")
        ax.grid(alpha=0.3)

    axes[0].legend()
    plt.tight_layout(); plt.show()
    return {'mode': 'time', 'threshold_time': pd.to_datetime(threshold_time),
            'n_ge': int(mask_ge.sum()), 'n_lt': int(mask_lt.sum())}


def plot_with_threshold(df, columns, threshold, time_col='time', alpha=0.6):
    """Plot columns colored by whether they are above/below a datetime threshold."""
    time_series = pd.to_datetime(df[time_col], errors='coerce')
    threshold   = pd.to_datetime(threshold)
    mask_above  = time_series >= threshold
    mask_below  = ~mask_above

    fig, axes = plt.subplots(len(columns), 1, figsize=(8, 3.5 * len(columns)), sharex=True)
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        ax.plot(time_series, df[col], color='gray', alpha=alpha, label='Full')
        ax.plot(time_series[mask_above], df[col][mask_above], color='teal',       lw=2, label='≥ threshold')
        ax.plot(time_series[mask_below], df[col][mask_below], color='darkorange', lw=2, label='< threshold')
        ax.set_title(col); ax.grid(alpha=0.3)

    axes[0].legend()
    plt.tight_layout(); plt.show()


def plot_distribution_by_time_threshold(df, columns, threshold_time, time_col='time', alpha=0.6):
    """Plot distributions of selected columns split by a datetime threshold."""
    time_series    = pd.to_datetime(df[time_col], errors='coerce')
    threshold_time = pd.to_datetime(threshold_time)

    if time_series.dt.tz is not None and threshold_time.tzinfo is None:
        threshold_time = threshold_time.tz_localize(time_series.dt.tz)
    elif time_series.dt.tz is None and threshold_time.tzinfo is not None:
        threshold_time = threshold_time.tz_convert(None)

    mask_above = time_series >= threshold_time
    mask_below = ~mask_above

    fig, axes = plt.subplots(len(columns), 1, figsize=(8, 3.5 * len(columns)))
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        ax.hist(df.loc[mask_above, col].dropna(), bins=30, color='teal',
                alpha=0.8, label='≥ threshold time')
        ax.hist(df.loc[mask_below, col].dropna(), bins=30, color='darkorange',
                alpha=0.8, label='< threshold time')
        ax.set_title(f"Distribution of {col} by Time Threshold")
        ax.set_xlabel(col); ax.set_ylabel("Frequency"); ax.grid(alpha=0.3)

    axes[0].legend()
    plt.tight_layout(); plt.show()


def timeblock_split_repeated(
    X: pd.DataFrame,
    y,
    fractions: Tuple[float, float, float] = (2/3, 1/6, 1/6),
    order: Tuple[str, str, str] = ("train", "test", "val"),
    n_cycles: int = 5,
    gap_before_val: int = 0,
    gap_before_test: int = 0,
    copy: bool = False,
):
    """
    Repeating time-block split:
        [TRAIN | gap | VAL | gap | TEST] x n_cycles (contiguous in time per block)
    """
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same number of rows (got {len(X)} and {len(y)})")

    roles = {"train", "test", "val"}
    if set(order) != roles:
        raise ValueError(f"`order` must be a permutation of {roles}, got {order}")

    f1, f2, f3 = fractions
    if any(f <= 0 for f in (f1, f2, f3)):
        raise ValueError("All fractions must be > 0.")
    if not 0.99 <= (f1 + f2 + f3) <= 1.01:
        raise ValueError("fractions must sum to ~1.0")

    n = len(X)
    if n < 3 * n_cycles:
        raise ValueError(f"Need at least {3*n_cycles} rows to make {n_cycles} cycles.")

    train_len = max(1, int(round((f1 * n) / n_cycles)))
    val_len   = max(1, int(round((f2 * n) / n_cycles)))
    test_len  = max(1, int(round((f3 * n) / n_cycles)))

    if train_len + gap_before_val + val_len + gap_before_test + test_len <= 0:
        raise ValueError("Cycle length computed as zero; check lengths/gaps.")

    idx_train_list, idx_val_list, idx_test_list = [], [], []
    offset = 0
    for _ in range(n_cycles):
        remaining = n - offset
        needed = train_len + gap_before_val + val_len + gap_before_test + test_len
        if remaining < needed:
            scale = remaining / needed
            t_len = max(1, int(np.floor(train_len * scale)))
            v_len = max(1, int(np.floor(val_len   * scale)))
            s_len = max(1, int(np.floor(test_len  * scale)))
            while (t_len + gap_before_val + v_len + gap_before_test + s_len) > remaining and s_len > 1:
                s_len -= 1
        else:
            t_len, v_len, s_len = train_len, val_len, test_len

        t_start = offset
        t_end   = min(t_start + t_len, n)
        v_start = min(t_end   + gap_before_val,  n)
        v_end   = min(v_start + v_len, n)
        s_start = min(v_end   + gap_before_test, n)
        s_end   = min(s_start + s_len, n)

        if v_start >= n or v_end <= v_start or s_start >= n or s_end <= s_start:
            break

        idx_train_list.append(np.arange(t_start, t_end))
        idx_val_list.append(np.arange(v_start, v_end))
        idx_test_list.append(np.arange(s_start, s_end))

        offset = s_end
        if offset >= n:
            break

    role_blocks = {
        "train": np.concatenate(idx_train_list) if idx_train_list else np.array([], dtype=int),
        "val":   np.concatenate(idx_val_list)   if idx_val_list   else np.array([], dtype=int),
        "test":  np.concatenate(idx_test_list)  if idx_test_list  else np.array([], dtype=int),
    }

    idx_train = role_blocks["train"]
    idx_val   = role_blocks["val"]
    idx_test  = role_blocks["test"]

    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_val,   y_val   = X.iloc[idx_val],   y.iloc[idx_val]
    X_test,  y_test  = X.iloc[idx_test],  y.iloc[idx_test]

    if copy:
        X_train, y_train = X_train.copy(), y_train.copy()
        X_val,   y_val   = X_val.copy(),   y_val.copy()
        X_test,  y_test  = X_test.copy(),  y_test.copy()

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            X.index[idx_train], X.index[idx_val], X.index[idx_test])
