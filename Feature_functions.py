import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as npsour  
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import Feature_functions 
import numpy as np
PLOT_OUTPUT_DIR = "workflow_plots_xbg"

import pickle
import tensorflow as tf
from tensorflow.keras import layers, regularizers

import tensorflow as tf



def save_plot(filename):
    """Saves the current matplotlib figure and closes it."""
    # Ensure the output directory exists
    if not os.path.exists(PLOT_OUTPUT_DIR):
        os.makedirs(PLOT_OUTPUT_DIR)
        
    filepath = os.path.join(PLOT_OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close() # Close the figure to free memory
    print(f"   🖼️ Saved plot to: {filepath}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def simple_index_plot(df_test, y_pred="rho_pred", start_index=0, n_steps=100, 
                      feature_list=None, y_target="rho_obs", Title="title"):
    """
    Plots target performance and key feature dynamics for a fixed number 
    of steps (rows) starting at a specific index.

    Args:
        df_feat (pd.DataFrame): DataFrame with features and the target.
        y_pred (np.array or pd.Series): Predictions aligned to df_feat index.
        start_index (int): The starting row index in df_feat.
        n_steps (int): The number of steps (rows) to include in the plot.
        feature_list (list): List of feature column names to plot.
        y_target (str): Name of the target column.
    """

    if feature_list is None:
        feature_list = ['ap_m3h', 'f107a']  # Simple default features

    # --- 1. Prepare Data ---
    
    # Combine features and predictions
    df_plot = df_test.copy()
    
    # Ensure y_pred is a Series aligned with df_feat
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred, index=df_test.index)
        
   


    # Use iloc to slice by integer index: [start_index : start_index + n_steps]
    end_index = start_index + n_steps
    df_window = df_plot.iloc[start_index:end_index].copy()

    if df_window.empty:
        print(f"⚠️ No data available from index {start_index} for {n_steps} steps.")
        return

    # --- 2. Create Plot ---
    
    # Total number of subplots: 1 (target/pred) + len(features)
    num_panels = 1 + len(feature_list) 
    fig, axes = plt.subplots(num_panels, 1, 
                             figsize=(14, 3 * num_panels), 
                             sharex=False) # Do not share X-axis, as it is just row numbers
    
    # Ensure axes is iterable even for a single subplot
    if num_panels == 1:
        axes = [axes]
    
    # X-axis data (using the index labels or just range if index is complex)
    # X-axis: use 'time' if present and valid, else row index
    if 'time' in df_window.columns:
        try:
            x_data = pd.to_datetime(df_window['time'], utc=True, errors='coerce')
            if x_data.isna().all():
            # If conversion failed, fallback to row index
                x_data = np.arange(start_index, start_index + n_steps)
            else:
                x_data = x_data.to_numpy()
        except Exception:
            x_data = np.arange(start_index, start_index + n_steps)
    else:
        x_data = np.arange(start_index, start_index + n_steps)
    
    # --- TOP PANEL: TARGET PERFORMANCE ---
    ax_target = axes[0]
    ax_target.plot(x_data, df_window[y_target], 
                   label='Observed (True)', color='black',  linestyle='--')
    ax_target.plot(x_data, df_window[y_pred], 
                   label='Prediction', color='red', linestyle='--')
    ax_target.plot(x_data, df_window['msis_rho'], 
                   label='MSIS', color='blue', linestyle='--')
    ax_target.set_title(f"Performance and Feature Dynamics (Rows {start_index} to {end_index-1})")
    ax_target.set_ylabel(y_target)
    ax_target.legend(loc='upper right')
    ax_target.grid(True, linestyle=':', alpha=0.6)


    df_comparison = pd.DataFrame({
    'MSIS_Rho': df_window['msis_rho'],
    'Model_Pred': df_window[y_pred],
    'True_Obs': df_window[y_target]
    }   )

# Display the comparison DataFrame (this is the standard way to 'print' in a notebook environment)
    print(df_comparison.head())

    # --- FEATURE PANELS ---
    for j, feature in enumerate(feature_list):
        # We start at axes[1] for the first feature plot
        ax_feature = axes[1 + j] 
        ax_feature.plot(x_data, df_window[feature],
                        label=feature, color='darkorange', alpha=0.8) 
        ax_feature.set_ylabel(feature)
        ax_feature.grid(True, linestyle=':', alpha=0.6)

        # Label x-axis only for the very last subplot
        if j == len(feature_list) - 1:
            ax_feature.tick_params(axis='x', rotation=45)
            # Use 'Index/Time' depending on what the index represents
            ax_feature.set_xlabel("Index Label / Time Stamp") 
        else:
            # Hide x-axis ticks/labels for upper feature panels
            ax_feature.tick_params(axis='x', labelbottom=False)

    plt.tight_layout()

    # Optional: Save the plot
    from pathlib import Path
    out_dir = Path("figs/time_series")
    safe_title = "".join(c if c.isalnum() or c in "-." else "" for c in Title)


    out_dir.mkdir(parents=True, exist_ok=True)

# Save PNG
    png_path = out_dir / f"time_series_dynamics{safe_title}.png"
    fig.savefig(png_path)
    print(f"✅ Plot saved as '{png_path.name}'")

# Save pickle
    pkl_path = out_dir / f"time_series_dynamics{safe_title}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"time_series": fig}, f)
    print(f"✅ Figures saved as '{pkl_path.name}'")

    return fig, axes


def add_lst_doy_features(df: pd.DataFrame,
                         time_col="time",
                         lon_col="lon") -> pd.DataFrame:
    """
    Add Local Solar Time (LST) and Day-of-Year (DOY) features
    with sine/cosine encoding to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
          - `time_col`: UTC datetime
          - `lon_col`: longitude in degrees (-180..180 or 0..360).
    time_col : str
        Column name for datetime.
    lon_col : str
        Column name for longitude.

    Returns
    -------
    pd.DataFrame
        Original dataframe with new columns:
        - lst_h (0–24 hours), lst_sin, lst_cos
        - doy (1–366), doy_sin, doy_cos
    """
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True)

    # --- Local Solar Time (hours) ---
    utc_hours = (
        d[time_col].dt.hour
        + d[time_col].dt.minute/60.0
        + d[time_col].dt.second/3600.0
    )
    lst_h = (utc_hours + d[lon_col]/15.0) % 24.0
    d["lst_h"]   = lst_h
    d["lst_sin"] = np.sin(2 * np.pi * lst_h / 24.0)
    d["lst_cos"] = np.cos(2 * np.pi * lst_h / 24.0)

    # --- Day-of-Year (1–366) ---
    doy = d[time_col].dt.dayofyear.astype(float)
    d["doy"]     = doy
    d["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)

    return d



def add_ap_lags(df, time_col="time", ap_col="ap_hourly", lags_hours=(3, 6, 12, 24)):
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True)

    # 1) Build an hourly Ap time series (handles duplicates and gaps)
    #    - take the mean if multiple rows fall in the same hour
    ap_hour = (
        d[[time_col, ap_col]]
        .assign(hour=lambda x: x[time_col].dt.floor("H"))
        .groupby("hour", as_index=True)[ap_col]
        .mean()
        .sort_index()
        .asfreq("H")                 # put on exact hourly grid
        .ffill()                     # fill small gaps forward
    )

    # 2) For each requested lag, map the lagged hourly Ap back to rows
    hour_key = d[time_col].dt.floor("h")
    for h in lags_hours:
        d[f"{ap_col}_lag{h}h"] = hour_key.map(ap_hour.shift(h))

    return d

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np # Used if X.copy() is necessary outside of a function

def scale_simple2(X_train, X_val, X_test,
                 y_train, y_val, y_test,
                 cols_to_scale, feature_range=(-1, 1)):
    """
    Scale ONLY the specified feature columns in X, and scale ALL columns in y.
    """
    # --- Feature scalers ---
    scaler_X = MinMaxScaler(feature_range=feature_range)
    # Fit only on the training subset of columns to be scaled
    scaler_X.fit(X_train[cols_to_scale])

    def scale_X(X):
        # Apply the transformation and get the scaled numpy array
        X_scaled_values = scaler_X.transform(X[cols_to_scale])
        
        # Create a DataFrame for the scaled part, preserving original index/columns
        X_scaled_df = pd.DataFrame(
            X_scaled_values,
            columns=cols_to_scale,
            index=X.index
        )
        
        # Create the final result DataFrame
        Xs = X.copy()
        
        # Merge the scaled columns back into the main DataFrame
        # This replaces the need for the problematic Xs.loc[:, cols_to_scale] assignment.
        Xs[cols_to_scale] = X_scaled_df
        return Xs

    X_train_s = scale_X(X_train)
    X_val_s   = scale_X(X_val)
    X_test_s  = scale_X(X_test)

    # --- Target scaler (all columns) ---
    scaler_y = MinMaxScaler(feature_range=feature_range)
    scaler_y.fit(y_train)

    def scale_y(y):
        # Target scaling is correct, but use .values for robust numpy conversion
        ys = pd.DataFrame(
            scaler_y.transform(y.values),
            columns=y.columns,
            index=y.index
        )
        return ys

    y_train_s = scale_y(y_train)
    y_val_s   = scale_y(y_val)
    y_test_s  = scale_y(y_test)

    return (X_train_s, X_val_s, X_test_s,
            y_train_s, y_val_s, y_test_s,
            scaler_X, scaler_y)


def scale_simple(X_train, X_val, X_test,
                 y_train, y_val, y_test,
                 cols_to_scale, feature_range=(-1, 1)):
    """
    Scale ONLY the specified feature columns in X, and scale ALL columns in y.
    Assumes y is numeric (no datetimes).

    Returns:
      X_train_s, X_val_s, X_test_s (DataFrames)
      y_train_s, y_val_s, y_test_s (DataFrames)
      scaler_X, scaler_y
    """
    # --- Feature scalers ---
    scaler_X = MinMaxScaler(feature_range=feature_range)
    scaler_X.fit(X_train[cols_to_scale])

    def scale_X(X):
        Xs = X.copy()
        Xs.loc[:, cols_to_scale] = scaler_X.transform(X[cols_to_scale])
        return Xs

    X_train_s = scale_X(X_train)
    X_val_s   = scale_X(X_val)
    X_test_s  = scale_X(X_test)

    # --- Target scaler (all columns) ---
    scaler_y = MinMaxScaler(feature_range=feature_range)
    scaler_y.fit(y_train)

    def scale_y(y):
        ys = pd.DataFrame(
            scaler_y.transform(y),
            columns=y.columns,
            index=y.index
        )
        return ys

    y_train_s = scale_y(y_train)
    y_val_s   = scale_y(y_val)
    y_test_s  = scale_y(y_test)

    return (X_train_s, X_val_s, X_test_s,
            y_train_s, y_val_s, y_test_s,
            scaler_X, scaler_y)

def plot_features_vs_index(df):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    axs[0, 0].plot(df.index, df['f107'], label='f107')
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

    axs[1, 1].plot(df.index, df['Ap'], label='ap')
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

    axs[0, 0].plot(df['time'], df['f107'], label='f107')
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

    axs[1, 1].plot(df['time'], df['Ap'], label='ap')
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
    # callback knobs:
    es_patience=10,
    rlrop_patience=5,
    rlrop_factor=0.5,
    min_lr=1e-5,
    monitor="val_loss",
    checkpoint_path="best.keras",
):
    # Ensure correct dtype and shape
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    X_val   = np.asarray(X_val,   dtype=np.float32)
    y_val   = np.asarray(y_val,   dtype=np.float32)

    n_features = X_train.shape[1]
    print("y_train columns", y_train)
    print("X_train columns", X_train)
    print("n_features:", n_features)
    print("Devices:", tf.config.list_logical_devices())
    # Build model
    
    
    #with tf.device('/GPU:0'):  # Force GPU usage

    model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(128, activation="relu",),
            tf.keras.layers.Dense(128, activation="relu"),
       # tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1)  # regression output
    ])

    model.compile(
       # optimizer=tf.keras.optimizers.AdamW(1e-4, weight_decay=1e-4, clipnorm=1.0),
            optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate), #Adam 
            loss="mse", #tf.keras.losses.Huber(delta=0.05), 
            metrics=["mse","mae"]
        )
    ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            mode="min",
            verbose=1
     )

    # Callbacks: EarlyStopping + ReduceLROnPlateau
    es = tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=es_patience,
            restore_best_weights=True
        )
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=rlrop_factor,
            patience=rlrop_patience,
            min_lr=min_lr,
            verbose=1
    )

    print("Starting training...")
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

# 2) Predicted vs Actual
def plot_parity_density(y_true, y_pred):
   # lims = [0.5, 1]
    plt.hexbin(y_true, y_pred, gridsize=40, cmap="viridis")
   # plt.plot(lims, lims, "r--", label="y = x")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual (density)")
   # plt.xlim(lims); plt.ylim(lims)
    plt.gca().set_aspect("equal", adjustable="box")
    cb = plt.colorbar()
    cb.set_label("Count")
    plt.legend()
    plt.show()
# 3) Residuals histogram
def plot_residuals(y_true, y_pred):
    res = y_pred - y_true
    plt.hist(res, bins=40)
    plt.axvline(0, color="r", linestyle="--")
    plt.xlabel("Residual (pred - actual)")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


def plot_val_densities_with_metrics(
    df_val,
    time_col="time",
    obs_col="rho_obs",
    msis_col="rho_msis",
    pred_col="rho_pred",
    sample_step=1,    # e.g. 10 to plot every 10th point
    parity_alpha=0.5
):
    """
    Compare observed vs MSIS vs predicted densities on the validation set,
    with metrics and annotated parity plot.

    Expects df_val to contain columns:
      - time_col, obs_col, msis_col, pred_col
    """
    # ---- clean & optional subsample for plotting ----
    cols = [time_col, obs_col, msis_col, pred_col]
    d = df_val[cols].dropna().copy()
    if sample_step > 1:
        d = d.iloc[::sample_step]

    t   = d[time_col]
    obs = d[obs_col].to_numpy()
    msis= d[msis_col].to_numpy()
    pred= d[pred_col].to_numpy()

    # guard: keep strictly positive for log metrics
    mpos = (obs > 0) & (msis > 0) & (pred > 0)
    obs, msis, pred, t = obs[mpos], msis[mpos], pred[mpos], t[mpos]

    # ---- metrics (linear space) ----
    rmse_msis_lin  = mean_squared_error(obs, msis)
    rmse_pred_lin  = mean_squared_error(obs, pred)
    mape_msis      = np.mean(np.abs((obs - msis) / obs)) * 100.0
    mape_pred      = np.mean(np.abs((obs - pred) / obs)) * 100.0
    r_msis, _      = pearsonr(obs, msis)
    r_pred, _      = pearsonr(obs, pred)
    r2_msis        = r2_score(obs, msis)
    r2_pred        = r2_score(obs, pred)

    # ---- metrics (log space) ----
    log_obs  = np.log(obs)
    log_msis = np.log(msis)
    log_pred = np.log(pred)
    rmse_msis_log = mean_squared_error(log_obs, log_msis)
    rmse_pred_log = mean_squared_error(log_obs, log_pred)
    # factor error = exp(RMSE_log)
    fe_msis = float(np.exp(rmse_msis_log))
    fe_pred = float(np.exp(rmse_pred_log))

    # ========= 1) Time series =========
    plt.figure(figsize=(12,5))
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

    # ========= 2) Parity =========
    vmax = np.nanmax([obs.max(), msis.max(), pred.max()])
    vmin = max(1e-15, np.nanmin([obs.min(), msis.min(), pred.min()]))

    plt.figure(figsize=(6.5,6.5))
    plt.scatter(obs, msis, s=8, alpha=parity_alpha, label=f"MSIS\nRMSE(log)={rmse_msis_log:.3f} (×{fe_msis:.2f})\nRMSE={rmse_msis_lin:.2e}\nMAPE={mape_msis:.1f}%\nR²={r2_msis:.3f}, r={r_msis:.3f}", color="C1")
    plt.scatter(obs, pred, s=8, alpha=parity_alpha, label=f"Predicted\nRMSE(log)={rmse_pred_log:.3f} (×{fe_pred:.2f})\nRMSE={rmse_pred_lin:.2e}\nMAPE={mape_pred:.1f}%\nR²={r2_pred:.3f}, r={r_pred:.3f}", color="C0")

    # 1:1 line
    plt.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)

   # plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Observed density [kg m$^{-3}$]")
    plt.ylabel("Modeled density [kg m$^{-3}$]")
    plt.title("Parity (Validation)")
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.show()

    # ---- also print a concise summary to console ----
    print(
        f"MSIS  : RMSE_log={rmse_msis_log:.3f}  factor≈×{fe_msis:.2f}  RMSE={rmse_msis_lin:.3e}  MAPE={mape_msis:.1f}%  R²={r2_msis:.3f}  r={r_msis:.3f}\n"
        f"Pred  : RMSE_log={rmse_pred_log:.3f}  factor≈×{fe_pred:.2f}  RMSE={rmse_pred_lin:.3e}  MAPE={mape_pred:.1f}%  R²={r2_pred:.3f}  r={r_pred:.3f}"
    )


def unscale_y_pred(y_pred_s, scaler_y, like_y):
    """
    Inverse-transform scaled predictions back to original y-space.
    Works for 1D or multi-target.

    Parameters
    ----------
    y_pred_s : array-like, shape (n_samples,) or (n_samples, n_targets)
    scaler_y : fitted MinMaxScaler (fit on y_train)
    like_y   : pd.DataFrame with the correct columns and index (e.g., y_val)

    Returns
    -------
    pd.DataFrame aligned to like_y.index/columns
    """
    import numpy as np
    import pandas as pd

    y_pred_s = np.asarray(y_pred_s)
    n_targets = getattr(scaler_y, "n_features_in_", 1)

    # Ensure shape (n_samples, n_targets)
    if y_pred_s.ndim == 1:
        y_pred_s = y_pred_s.reshape(-1, n_targets)

    y_pred = scaler_y.inverse_transform(y_pred_s)
    return pd.DataFrame(y_pred, index=like_y.index, columns=like_y.columns)


import pandas as pd
from typing import Tuple
def timeblock_split(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    fractions: Tuple[float, float, float] = (2/3, 1/6, 1/6),
    order: Tuple[str, str, str] = ("train", "test", "val"),
    copy: bool = False,
):
    """
    Split X, y into three contiguous time-ordered blocks and assign them to
    (train, val, test) according to `order`.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test,
    index_train, index_val, index_test
        Slices and corresponding indices for each role.
    """

    # --- Validation ---
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

    # --- Compute boundaries ---
    b1 = int(f1 * n)
    b2 = int((f1 + f2) * n)
    b1 = min(max(b1, 1), n-2)
    b2 = min(max(b2, b1+1), n-1)

    # Build blocks
    blocks_X = (X.iloc[:b1], X.iloc[b1:b2], X.iloc[b2:])
    blocks_y = (y.iloc[:b1], y.iloc[b1:b2], y.iloc[b2:])
    blocks_idx = (X.index[:b1], X.index[b1:b2], X.index[b2:])

    # Map to roles
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
    """
    Plots multiple columns against the 'time' column in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data
    columns (list): List of column names to plot
    name (str): Title of the plot
    """
    plt.figure(figsize=(10, 6))
    for col in columns:
        plt.plot(df['time'], df[col], label=col)
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"{name.replace(' ', '_')}_plot.png"
    plt.savefig(filename)
    print(f"Plot saved as '{filename}'")


def plot_two_columns_vs_time(df, value1, value2, name="titel"):
    """
    Plots two columns against the 'time' column in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data
    value1 (str): Name of the first column to plot
    value2 (str): Name of the second column to plot
    name (str): Title of the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df[value1], label=value1)
    plt.plot(df['time'], df[value2], label=value2)
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"{name.replace(' ', '_')}_plot.png"
    plt.savefig(filename)
    print(f"Plot saved as '{filename}'")


def plot_distribution_by_threshold(
    df,
    columns,
    threshold,
    threshold_col=None,  # NEW: if provided, use numeric threshold on this column
    time_col='time',
    bins=30,
    density=False,
    alpha=0.8,
    gt_color='darkorange',  # color for values > threshold
    le_color='seagreen'     # color for values ≤ threshold
):
    """
    Plot distributions of selected columns split by a threshold.

    Two modes:
    1) VALUE MODE (numeric threshold):
       If `threshold_col` is provided, split rows by:
           - (threshold_col > threshold)  --> orange
           - (threshold_col ≤ threshold)  --> green

    2) TIME MODE (datetime threshold):
       If `threshold_col` is None, treat `threshold` as a datetime and split by:
           - (time_col >= threshold) --> teal/orange as in your original

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns to plot and the split column(s).
    columns : list[str]
        Columns to plot as histograms.
    threshold : float|str|pd.Timestamp
        In VALUE MODE: numeric threshold (e.g., 1e-11, '10E-12').
        In TIME MODE: datetime string or Timestamp.
    threshold_col : str or None
        Column name for VALUE MODE. If None, TIME MODE is used.
    time_col : str
        Datetime column for TIME MODE.
    bins : int or sequence
        Histogram bins. If int, common bins are computed from combined data.
    density : bool
        Plot density instead of frequency.
    alpha : float
        Histogram bar opacity.
    gt_color : str
        Color for the group where value/condition is "greater than" the threshold.
    le_color : str
        Color for the group where value/condition is "less or equal" the threshold.

    Returns
    -------
    split_info : dict
        - In VALUE MODE: {'mode': 'value', 'threshold': float, 'threshold_col': str,
                          'n_gt': int, 'n_le': int}
        - In TIME MODE:  {'mode': 'time', 'threshold_time': pd.Timestamp,
                          'n_ge': int, 'n_lt': int}
    """

    def _parse_float(x):
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        if isinstance(x, str):
            # allow formats like "10E-12" or "1e-11" and comma decimals
            s = x.strip().replace(',', '.')
            return float(s)  # raises ValueError if not parseable
        raise TypeError(f"Cannot parse threshold {x!r} to float.")

    # -------- VALUE MODE: threshold on `threshold_col` --------
    if threshold_col is not None:
        thr = _parse_float(threshold)
        key = pd.to_numeric(df[threshold_col], errors='coerce')

        mask_gt = key > thr
        mask_le = ~mask_gt  # includes NaN as False in mask_gt => NaN goes to mask_le

        gt_label = f"{threshold_col} > {thr:g}"
        le_label = f"{threshold_col} \u2264 {thr:g}"  # ≤

        # Plot
        fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(8, 3.5 * len(columns)))
        if len(columns) == 1:
            axes = [axes]

        for ax, col in zip(axes, columns):
            v_gt = df.loc[mask_gt, col].dropna()
            v_le = df.loc[mask_le, col].dropna()

            # common bins for fair comparison
            if isinstance(bins, int):
                all_vals = pd.concat([v_gt, v_le], ignore_index=True)
                if len(all_vals) == 0:
                    ax.set_title(f"Distribution of {col} (no data)")
                    ax.set_xlabel(col)
                    ax.set_ylabel("Density" if density else "Frequency")
                    ax.grid(alpha=0.3)
                    continue
                bin_edges = np.histogram_bin_edges(all_vals, bins=bins)
            else:
                bin_edges = bins

            ax.hist(v_le, bins=bin_edges, color=le_color, alpha=alpha,
                    label=le_label, density=density)
            ax.hist(v_gt, bins=bin_edges, color=gt_color, alpha=alpha,
                    label=gt_label, density=density)

            ax.set_title(
                f"Distribution of {col} split by {threshold_col} threshold ({thr:g})"
            )
            ax.set_xlabel(col)
            ax.set_ylabel("Density" if density else "Frequency")
            ax.grid(alpha=0.3)

        axes[0].legend()
        plt.tight_layout()
        plt.show()

        return {
            'mode': 'value',
            'threshold': thr,
            'threshold_col': threshold_col,
            'n_gt': int(mask_gt.sum()),
            'n_le': int(mask_le.sum()),
        }

    # -------- TIME MODE: threshold on datetime `time_col` --------
    time_series = pd.to_datetime(df[time_col], errors='coerce')
    threshold_time = pd.to_datetime(threshold)

    # Align timezone awareness
    if time_series.dt.tz is not None and threshold_time.tzinfo is None:
        threshold_time = threshold_time.tz_localize(time_series.dt.tz)
    elif time_series.dt.tz is None and threshold_time.tzinfo is not None:
        threshold_time = threshold_time.tz_convert(None)

    mask_ge = (time_series >= threshold_time)
    mask_lt = ~mask_ge

    fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(8, 3.5 * len(columns)))
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        v_ge = df.loc[mask_ge, col].dropna()
        v_lt = df.loc[mask_lt, col].dropna()

        if isinstance(bins, int):
            all_vals = pd.concat([v_ge, v_lt], ignore_index=True)
            if len(all_vals) == 0:
                ax.set_title(f"Distribution of {col} (no data)")
                ax.set_xlabel(col)
                ax.set_ylabel("Density" if density else "Frequency")
                ax.grid(alpha=0.3)
                continue
            bin_edges = np.histogram_bin_edges(all_vals, bins=bins)
        else:
            bin_edges = bins

        ax.hist(v_lt, bins=bin_edges, color='darkorange', alpha=alpha,
                label='< threshold time', density=density)
        ax.hist(v_ge, bins=bin_edges, color='teal', alpha=alpha,
                label='≥ threshold time', density=density)

        ax.set_title(f"Distribution of {col} by Time Threshold @ {pd.to_datetime(threshold_time)}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density" if density else "Frequency")
        ax.grid(alpha=0.3)

    axes[0].legend()
    plt.tight_layout()
    plt.show()

    return {
        'mode': 'time',
        'threshold_time': pd.to_datetime(threshold_time),
        'n_ge': int(mask_ge.sum()),
        'n_lt': int(mask_lt.sum()),
    }

    # Build the diff column

# Columns you want to compare

# %%

def plot_with_threshold(df, columns, threshold, time_col='time', alpha=0.6):
    """
    Plot selected columns with:
      - Full series in gray
      - Values >= threshold in green
      - Values < threshold in red

    Parameters:
    - df: DataFrame with a datetime column
    - columns: list of columns to plot
    - threshold: datetime string or Timestamp
    - time_col: name of the datetime column
    - alpha: line transparency
    """
    # Ensure datetime
    time_series = pd.to_datetime(df[time_col], errors='coerce')
    threshold = pd.to_datetime(threshold)

    # Masks
    mask_above = time_series >= threshold
    mask_below = ~mask_above

    # Plot
    fig, axes = plt.subplots(len(columns), 1, figsize=(8, 3.5 * len(columns)), sharex=True)
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        ax.plot(time_series, df[col], color='gray', alpha=alpha, label='Full')
        ax.plot(time_series[mask_above], df[col][mask_above], color='teal', lw=2, label='≥ threshold')
        ax.plot(time_series[mask_below], df[col][mask_below], color='darkorange', lw=2, label='< threshold')
        ax.set_title(col)
        ax.grid(alpha=0.3)

    axes[0].legend()
    plt.tight_layout()
    plt.show()



def plot_distribution_by_time_threshold(df, columns, threshold_time, time_col='time', alpha=0.6):
    """
    Plot distributions of selected columns:
      - Values >= threshold_time in teal
      - Values < threshold_time in darkorange

    Parameters:
    - df: DataFrame with a datetime column
    - columns: list of columns to plot
    - threshold_time: datetime string or Timestamp
    - time_col: name of the datetime column
    - alpha: transparency for histograms
    """
    # Ensure datetime and align timezones
    time_series = pd.to_datetime(df[time_col], errors='coerce')
    threshold_time = pd.to_datetime(threshold_time)

    # Align timezone awareness
    if time_series.dt.tz is not None and threshold_time.tzinfo is None:
        threshold_time = threshold_time.tz_localize(time_series.dt.tz)
    elif time_series.dt.tz is None and threshold_time.tzinfo is not None:
        threshold_time = threshold_time.tz_convert(None)

    # Masks based on time
    mask_above = time_series >= threshold_time
    mask_below = ~mask_above

    # Plot
    fig, axes = plt.subplots(len(columns), 1, figsize=(8, 3.5 * len(columns)))
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        values_above = df.loc[mask_above, col].dropna()
        values_below = df.loc[mask_below, col].dropna()

        ax.hist(values_above, bins=30, color='teal', alpha=0.8, label='≥ threshold time')
        ax.hist(values_below, bins=30, color='darkorange', alpha=0.8, label='< threshold time')
        ax.set_title(f"Distribution of {col} by Time Threshold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)

    axes[0].legend()
    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
from typing import Tuple
import numpy as np
import pandas as pd
from typing import Tuple

def timeblock_split_repeated(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    fractions: Tuple[float, float, float] = (2/3, 1/6, 1/6),
    order: Tuple[str, str, str] = ("train", "test", "val"),
    n_cycles: int = 5,
    gap_before_val: int = 0,   # embargo between train→val in each cycle
    gap_before_test: int = 0,  # embargo between val→test in each cycle
    copy: bool = False,
):
    """
    Repeating time-block split:
        [TRAIN | gap | VAL | gap | TEST] x n_cycles (contiguous in time per block)

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test,
    index_train, index_val, index_test
    """

    # --- Validation (same spirit as your function) ---
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

    # --- Per-cycle lengths (rounded so total ≈ n) ---
    # We first compute ideal per-cycle lengths then adjust last cycle to consume remainder.
    train_len = max(1, int(round((f1 * n) / n_cycles)))
    val_len   = max(1, int(round((f2 * n) / n_cycles)))
    test_len  = max(1, int(round((f3 * n) / n_cycles)))

    # Safety: ensure a cycle is feasible
    min_cycle = train_len + gap_before_val + val_len + gap_before_test + test_len
    if min_cycle <= 0:
        raise ValueError("Cycle length computed as zero; check lengths/gaps.")

    idx_train_list, idx_val_list, idx_test_list = [], [], []

    offset = 0
    for i in range(n_cycles):
        # For the last cycle, try to absorb any remaining samples proportionally into train
        remaining = n - offset
        needed = train_len + gap_before_val + val_len + gap_before_test + test_len
        if remaining < needed:
            # Scale down this last cycle proportionally but keep order: train → val → test
            scale = remaining / needed
            t_len = max(1, int(np.floor(train_len * scale)))
            v_len = max(1, int(np.floor(val_len   * scale)))
            s_len = max(1, int(np.floor(test_len  * scale)))
            # If rounding lost too much, push leftovers to test
            while (t_len + gap_before_val + v_len + gap_before_test + s_len) > remaining and s_len > 1:
                s_len -= 1
        else:
            t_len, v_len, s_len = train_len, val_len, test_len

        t_start = offset
        t_end   = min(t_start + t_len, n)

        v_start = min(t_end + gap_before_val, n)
        v_end   = min(v_start + v_len, n)

        s_start = min(v_end + gap_before_test, n)
        s_end   = min(s_start + s_len, n)

        # If we can't fit a full train|val|test in this cycle, stop.
        if v_start >= n or v_end <= v_start or s_start >= n or s_end <= s_start:
            break

        idx_train_list.append(np.arange(t_start, t_end))
        idx_val_list.append(np.arange(v_start, v_end))
        idx_test_list.append(np.arange(s_start, s_end))

        offset = s_end
        if offset >= n:
            break

    # Concatenate indices for each role and map to order
    role_blocks = {
        "train": np.concatenate(idx_train_list) if idx_train_list else np.array([], dtype=int),
        "val":   np.concatenate(idx_val_list)   if idx_val_list   else np.array([], dtype=int),
        "test":  np.concatenate(idx_test_list)  if idx_test_list  else np.array([], dtype=int),
    }

    # Build slices according to `order`
    # (We follow your return order exactly: train, val, test)
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

    return X_train, X_val, X_test, y_train, y_val, y_test, X.index[idx_train], X.index[idx_val], X.index[idx_test]


def plot_val_densities_with_metrics2(
    df_val,
    time_col="time",
    obs_col="rho_obs",
    msis_col="rho_msis",
    pred_col="rho_pred",
    sample_step=1,    # e.g. 10 to plot every 10th point
    parity_alpha=0.5
):
    """
    Compare observed vs MSIS vs predicted densities on the validation set,
    with metrics and annotated parity plot.
    
    Expects df_val to contain columns:
      - time_col, obs_col, msis_col, pred_col
    """
    # ---- clean & optional subsample for plotting ----
    cols = [time_col, obs_col, msis_col, pred_col]
    d = df_val[cols].dropna().copy()
    if sample_step > 1:
        d = d.iloc[::sample_step]

    t   = d[time_col]
    obs = d[obs_col].to_numpy()
    msis= d[msis_col].to_numpy()
    pred= d[pred_col].to_numpy()

    # guard: keep strictly positive for log metrics
    mpos = (obs > 0) & (msis > 0) & (pred > 0)
    obs, msis, pred, t = obs[mpos], msis[mpos], pred[mpos], t[mpos]

    # ---- metrics (linear space) ----
    mse_msis_lin = mean_squared_error(obs, msis)
    mse_pred_lin = mean_squared_error(obs, pred)
    rmse_msis_lin = np.sqrt(mse_msis_lin)  # Now correctly calculated
    rmse_pred_lin = np.sqrt(mse_pred_lin)  # Now correctly calculated# Use squared=False for true RMSE
    mape_msis      = np.mean(np.abs((obs - msis) / obs)) * 100.0
    mape_pred      = np.mean(np.abs((obs - pred) / obs)) * 100.0
    r_msis, _      = pearsonr(obs, msis)
    r_pred, _      = pearsonr(obs, pred)
    r2_msis        = r2_score(obs, msis)
    r2_pred        = r2_score(obs, pred)

    # ---- metrics (log space) ----
    log_obs  = np.log(obs)
    log_msis = np.log(msis)
    log_pred = np.log(pred)
    rmse_msis_log = mean_squared_error(log_obs, log_msis) # Use squared=False for true RMSE
    rmse_pred_log = mean_squared_error(log_obs, log_pred) # Use squared=False for true RMSE
    
    rmse_msis_log = np.sqrt(rmse_msis_log)  # Now correctly calculated
    rmse_pred_log = np.sqrt(rmse_pred_log)  # Now correctly calculated# Use squared=False for true RMSE
    
    # factor error = exp(RMSE_log)
    fe_msis = float(np.exp(rmse_msis_log))
    fe_pred = float(np.exp(rmse_pred_log))
    
    # ---- P95 Error (New Metric) ----
    # Absolute linear errors (for P95 linear)
    abs_err_msis_lin = np.abs(obs - msis)
    abs_err_pred_lin = np.abs(obs - pred)
    p95_msis_lin = np.percentile(abs_err_msis_lin, 95)
    p95_pred_lin = np.percentile(abs_err_pred_lin, 95)

    # Absolute log errors (for P95 log)
    abs_err_msis_log = np.abs(log_obs - log_msis)
    abs_err_pred_log = np.abs(log_obs - log_pred)
    p95_msis_log = np.percentile(abs_err_msis_log, 95)
    p95_pred_log = np.percentile(abs_err_pred_log, 95)
    # P95 factor error
    fe95_msis = float(np.exp(p95_msis_log))
    fe95_pred = float(np.exp(p95_pred_log))

    # ========= 1) Time series =========

    msis_label = (
        f"MSIS\n"
        f"RMSE(log)={rmse_msis_log:.3f} (×{fe_msis:.2f})\n"
        f"P95(log)={p95_msis_log:.3f} (×{fe95_msis:.2f})\n"
        f"RMSE={rmse_msis_lin:.2e}, P95={p95_msis_lin:.2e}\n"
        f"MAPE={mape_msis:.1f}%, R²={r2_msis:.3f}, r={r_msis:.3f}"
    )

    pred_label = (
        f"Predicted\n"
        f"RMSE(log)={rmse_pred_log:.3f} (×{fe_pred:.2f})\n"
        f"P95(log)={p95_pred_log:.3f} (×{fe95_pred:.2f})\n"
        f"RMSE={rmse_pred_lin:.2e}, P95={p95_pred_lin:.2e}\n"
        f"MAPE={mape_pred:.1f}%, R²={r2_pred:.3f}, r={r_pred:.3f}"
    )

    vmax = np.nanmax([obs.max(), msis.max(), pred.max()])
    vmin = max(1e-15, np.nanmin([obs.min(), msis.min(), pred.min()]))


    # ========= 1) Time series =========
    fig1 = plt.figure(figsize=(12,5))
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

# ========= 2) Parity =========
    fig2 = plt.figure(figsize=(6.5,6.5))
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

# ---- Save figures as pickle ----
    import pickle
    with open("val_density_plots.pkl", "wb") as f:
        pickle.dump({"time_series": fig1, "parity": fig2}, f)
    print("✅ Figures saved as val_density_plots.pkl")


    # ---- also print a concise summary to console ----
    print(
        f"MSIS  : RMSE_log={rmse_msis_log:.3f} factor≈×{fe_msis:.2f} | P95_log={p95_msis_log:.3f} factor95≈×{fe95_msis:.2f} | RMSE={rmse_msis_lin:.3e} P95={p95_msis_lin:.3e} | MAPE={mape_msis:.1f}% R²={r2_msis:.3f} r={r_msis:.3f}\n"
        f"Pred  : RMSE_log={rmse_pred_log:.3f} factor≈×{fe_pred:.2f} | P95_log={p95_pred_log:.3f} factor95≈×{fe95_pred:.2f} | RMSE={rmse_pred_lin:.3e} P95={p95_pred_lin:.3e} | MAPE={mape_pred:.1f}% R²={r2_pred:.3f} r={r_pred:.3f}"
    )



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

def plot_val_densities_with_metrics3(
    df_val,
    time_col="time",
    obs_col="rho_obs",
    msis_col="rho_msis",
    pred_col="rho_pred",
    sample_step=1,    # e.g. 10 to plot every 10th point
    parity_alpha=0.5
):
    """
    Compare observed vs MSIS vs predicted densities on the validation set,
    with metrics and annotated parity plot.

    Replaces P95 metrics with Top5 metrics (mean of worst 5% absolute errors)
    in both linear and log space.

    Expects df_val to contain columns:
      - time_col, obs_col, msis_col, pred_col
    """
    # ---- clean & optional subsample for plotting ----
    cols = [time_col, obs_col, msis_col, pred_col]
    d = df_val[cols].dropna().copy()
    if sample_step > 1:
        d = d.iloc[::sample_step]

    t   = d[time_col]
    obs = d[obs_col].to_numpy()
    msis= d[msis_col].to_numpy()
    pred= d[pred_col].to_numpy()

    # guard: keep strictly positive for log metrics
    mpos = (obs > 0) & (msis > 0) & (pred > 0)
    obs, msis, pred, t = obs[mpos], msis[mpos], pred[mpos], t[mpos]

    # ---- metrics (linear space) ----
    mse_msis_lin = mean_squared_error(obs, msis)
    mse_pred_lin = mean_squared_error(obs, pred)
    rmse_msis_lin = np.sqrt(mse_msis_lin)
    rmse_pred_lin = np.sqrt(mse_pred_lin)
    mape_msis      = np.mean(np.abs((obs - msis) / obs)) * 100.0
    mape_pred      = np.mean(np.abs((obs - pred) / obs)) * 100.0
    r_msis, _      = pearsonr(obs, msis)
    r_pred, _      = pearsonr(obs, pred)
    r2_msis        = r2_score(obs, msis)
    r2_pred        = r2_score(obs, pred)

    # ---- metrics (log space) ----
    log_obs  = np.log(obs)
    log_msis = np.log(msis)
    log_pred = np.log(pred)

    # RMSE in log space (use MSE then sqrt)
    rmse_msis_log = np.sqrt(mean_squared_error(log_obs, log_msis))
    rmse_pred_log = np.sqrt(mean_squared_error(log_obs, log_pred))

    # factor error = exp(RMSE_log)
    fe_msis = float(np.exp(rmse_msis_log))
    fe_pred = float(np.exp(rmse_pred_log))

    # ---- Top 5% mean error (replace P95) ----
    # Absolute linear errors
    abs_err_msis_lin = np.abs(obs - msis)
    abs_err_pred_lin = np.abs(obs - pred)
    n = abs_err_msis_lin.size
    k = max(1, int(np.ceil(0.05 * n)))  # worst 5% count (at least 1)

    # sort descending and take mean of top-k (tail severity)
    top5_msis_lin = float(np.mean(np.sort(abs_err_msis_lin)[-k:])) if n else np.nan
    top5_pred_lin = float(np.mean(np.sort(abs_err_pred_lin)[-k:])) if n else np.nan

    # Absolute log errors
    abs_err_msis_log = np.abs(log_obs - log_msis)
    abs_err_pred_log = np.abs(log_obs - log_pred)
    top5_msis_log = float(np.mean(np.sort(abs_err_msis_log)[-k:])) if n else np.nan
    top5_pred_log = float(np.mean(np.sort(abs_err_pred_log)[-k:])) if n else np.nan

    # Top5 factor error in log space
    fe_top5_msis = float(np.exp(top5_msis_log)) if np.isfinite(top5_msis_log) else np.nan
    fe_top5_pred = float(np.exp(top5_pred_log)) if np.isfinite(top5_pred_log) else np.nan

    # ========= 1) Time series =========
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

    # ========= 1) Time series =========
    fig1 = plt.figure(figsize=(12,5))
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

    # ========= 2) Parity =========
    fig2 = plt.figure(figsize=(6.5,6.5))
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

    # ---- Save figures as pickle ----
    import pickle
    with open("val_density_plots.pkl", "wb") as f:
        pickle.dump({"time_series": fig1, "parity": fig2}, f)
    print("✅ Figures saved as val_density_plots.pkl")

    # ---- concise summary to console ----
    print(
        f"MSIS  : RMSE_log={rmse_msis_log:.3f} ×{fe_msis:.2f} | Top5_log={top5_msis_log:.3f} ×{fe_top5_msis:.2f} | "
        f"RMSE={rmse_msis_lin:.3e} Top5={top5_msis_lin:.3e} | MAPE={mape_msis:.1f}% R²={r2_msis:.3f} r={r_msis:.3f}\n"
        f"Pred  : RMSE_log={rmse_pred_log:.3f} ×{fe_pred:.2f} | Top5_log={top5_pred_log:.3f} ×{fe_top5_pred:.2f} | "
        f"RMSE={rmse_pred_lin:.3e} Top5={top5_pred_lin:.3e} | MAPE={mape_pred:.1f}% R²={r2_pred:.3f} r={r_pred:.3f}"
    )
