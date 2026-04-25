import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import importlib
import platform
import pickle

# ML/XGBoost imports
import xgboost as xgb
from sklearn.metrics import mean_squared_error
# Note: EarlyStopping and LearningRateScheduler are used below
from xgboost.callback import EarlyStopping, LearningRateScheduler

# Custom Feature functions (Assumes Feature_functions.py is available)
# NOTE: Removed the 'import tensorflow as tf' from here, as it's only needed for the final check.
# Added platform import up top for the check.
import Feature_functions as ff 
importlib.reload(ff)

# --- CONFIGURATION ---
TARGET_COL = "log_ratio"
MODEL_FILE = "xgb_model_v3.json" #v2 good solid small - v3 with lag for 24 hours
SCALER_X_FILE = "scaler_xgboost_X_v3.joblib"
SCALER_Y_FILE = "scaler_xgboost_y_v3.joblib"
#df = pd.read_parquet("grace_dns_with_tnd_y200916_v4_0809.parquet", engine="pyarrow")
PROCESSED_DATA_FILE = "prodees.parquet"
#PROCESSED_DATA_FILE = "grace_data_featured.parquet"
PLOT_OUTPUT_DIR = "workflow_plots_xbg"

# EXECUTION FLAGS
TRAIN_FLAG = 1             # 1 to train and save the model, 0 to skip
TRAINING_MODE = 'custom'   # Options: 'custom' (uses mse_extreme_obj) or 'simple' (uses reg:squarederror)
PLOT_FLAG = 1            # 1 to generate all diagnostic plots, 0 to skip
SKIP_DATA_PROCESSING = 1   # 1 to load PROCESSED_DATA_FILE if it exists, 0 to force full run

# PLOT SETTINGS
PLOT_SAMPLED_STEP = 100    # Sampling rate for distributions
PLOT_TS_STEP = 100         # Sampling rate for time series plots



# ====================================================================
# --- CUSTOM FUNCTIONS (Loss and LR Scheduler) ---
# ====================================================================

def lr_scheduler(current_round):
    """Learning rate scheduler for native XGBoost API."""
    initial_lr = 5e-4
    decay_factor = 0.8
    step_size = 15
    calculated_lr = initial_lr * (decay_factor ** (current_round // step_size))
    if current_round % 50 == 0:
        print(f"Round {current_round}: LR = {calculated_lr:.8f}")
    return calculated_lr

def mse_extreme_obj(preds, dtrain, threshold=0.5, lambda_extreme=0.6):
    """Custom loss: MSE for bulk + extra penalty for extremes."""
    y = dtrain.get_label()
    r = preds - y
    grad = r
    hess = np.ones_like(r)
    extreme_mask = np.abs(r) > threshold
    grad_extreme = lambda_extreme * np.sign(r) * extreme_mask
    hess_extreme = np.ones_like(r)
    grad += grad_extreme
    hess += hess_extreme
    return grad, hess

# ====================================================================
# --- 1. DATA LOADING, CACHING, AND FEATURE ENGINEERING ---
# ====================================================================

def clamp_outliers(df_feat, col, percentile=0.94):
    """Clamps extreme values in a column to a specific percentile threshold."""
    # Find the upper threshold
    upper_threshold = df_feat[col].quantile(percentile)
    
    # Clamp values above the threshold
    df_feat[col] = np.clip(df_feat[col], a_min=None, a_max=upper_threshold)
    return df_feat, df

def load_and_feature_engineer_data(file_path):
    """
    Loads, cleans, and engineers features. Uses caching to skip long steps
    if PROCESSED_DATA_FILE exists and SKIP_DATA_PROCESSING is enabled.
    """
    if SKIP_DATA_PROCESSING and os.path.exists(PROCESSED_DATA_FILE):
        print(f"⏩ Loading cached data from {PROCESSED_DATA_FILE}")
        df_feat = pd.read_parquet(PROCESSED_DATA_FILE) 
        print(f"✅ Loaded cached DataFrame shape: {df_feat.shape}")
        return df_feat, df_feat

    print("⏳ Starting FULL data processing (Loading, Cleaning, Feature Engineering)...")

    # A. LOAD AND CLEAN
    df = pd.read_parquet(file_path)
    df['time'] = pd.to_datetime(df['grace_time'])
    df = df[df['time'] < '2016-01-01']
    df = df[df['time'] > '2009-06-01']

    print(f"Loaded and cleaned data shape: {df.shape}")
    
    # B. FEATURE ENGINEERING
    df_feat = df.copy()
    df_feat = ff.add_lst_doy_features(df_feat)
    df_feat['lon_sin'] = np.sin(np.deg2rad(df_feat['lon']))
    df_feat['lon_cos'] = np.cos(np.deg2rad(df_feat['lon']))
    df_feat['lst_lat_cos'] = df_feat['lst_cos'] * df_feat['lat']
    # LST_sin * Latitude (Helps capture the shift/skewness of the bulge)
    df_feat['lst_lat_sin'] = df_feat['lst_sin'] * df_feat['lat']
    df_feat['vtec_matched_lag'] = df_feat['matched_tec_value'].shift(500)
    df_feat['vtec_matched_lag2'] = df_feat['matched_tec_value'].shift(17280) #1075)
    df_feat['ap_change'] = df_feat['ap_0h'] - df_feat['ap_m3h']
    df_feat[TARGET_COL] = np.log(df_feat["rho_obs"] / df_feat["msis_rho"])

   # df_feat = clamp_outliers(df_feat, 'f107')
   # df_feat = clamp_outliers(df_feat, 'f107a')
   # df_feat = clamp_outliers(df_feat, 'ap_m3h')
   # df_feat = clamp_outliers(df_feat, 'ap_m6h')
   # df_feat = clamp_outliers(df_feat, 'ap_change')
    # C. FINAL CLEANUP AND CACHE
    initial_nans = df_feat.isna().sum().sum()
    df_feat = df_feat.dropna()
    print(f"ℹ️ Dropped {initial_nans - df_feat.isna().sum().sum()} rows with NaNs. Final size: {df_feat.shape}")
    df_feat.reset_index(drop=True, inplace=True) 

    try:
        # NOTE: Using index=False for better cross-system compatibility with Parquet
        df_feat.to_parquet(PROCESSED_DATA_FILE, index=False) 
        print(f"💾 Cached processed data to {PROCESSED_DATA_FILE}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save cached file: {e}")

    return df_feat, df

# ====================================================================
# --- 2. DATA SPLITTING AND FILTERING ---
# ====================================================================

def split_and_filter_data(df_feat):
    """Splits data into train/val/test using time block splitting."""
    print("⏳ Splitting data...")
    
    columns_to_keep = [
        "f107a", "lat", 
        "matched_tec_value",
        "lon_cos",
        "lon_sin" , 
        "lst_sin", 
        "ap_m3h",
        "doy_sin", "doy_cos", "f107", "alt_km", 
        "ap_m6h", #"
        "vtec_matched_lag", "vtec_matched_lag2", 
       # "ap_0h",
        #'ap_change',
        "lst_lat_sin"#, "lst_lat_cos",  "lst_cos"
    ]
    df_feat_filtered = df_feat[columns_to_keep]
    df_target = df_feat[[TARGET_COL]]

    results = ff.timeblock_split_repeated(
        df_feat_filtered, df_target,
        fractions=(2/3, 1/6, 1/6),
        n_cycles=8,
        gap_before_val=1100,
        gap_before_test=1100,
        order=("train", "test", "val"),
        copy=False
    )
    X_train, X_test, X_val, y_train, y_test, y_val, idx_train, idx_test, idx_val = results
    print(f"✅ Split sizes: Train={X_train.shape[0]}, Test={X_test.shape[0]}, Val={X_val.shape[0]}")
    
    return X_train, X_test, X_val, y_train, y_test, y_val, idx_train, idx_test, idx_val

# ====================================================================
# --- 3. SCALING ---
# ====================================================================

def scale_data(X_train, X_test, X_val, y_train, y_test, y_val):
    """Scales features and target, and saves scalers."""
    print("⏳ Scaling data and saving scalers...")
    
    cols_to_scale = ["f107","ap_m6h","lat", "f107a", "alt_km","matched_tec_value", "ap_m3h", "vtec_matched_lag", "vtec_matched_lag2" ] #,  'lst_lat_cos', 'lst_lat_sin', "ap_0h"] 
   # cols_to_scale = ["f107","ap_m6h","lat", "f107a", "alt_km", "matched_tec_value", "vtec_matched_lag"] 
    results = ff.scale_simple(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        cols_to_scale=cols_to_scale
    )
    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, scaler_X, scaler_y = results

    joblib.dump(scaler_y, SCALER_Y_FILE)
    joblib.dump(scaler_X, SCALER_X_FILE)
    
    print(f"✅ Scalers saved to {SCALER_X_FILE} and {SCALER_Y_FILE}")
    return X_train_s, X_test_s, X_val_s, y_train_s, y_test_s, y_val_s, scaler_X, scaler_y




# ====================================================================
# --- 5. NEW: FEATURE IMPORTANCE ANALYSIS ---
# ====================================================================

def analyze_feature_importance(model, X_train_cols, mode, plot_flag):
    """
    Calculates, prints, and plots feature importance for the trained model 
    (supports both XGBRegressor and Booster objects).
    """
    if not plot_flag:
        return

    print("\n🔍 Calculating Feature Importance...")

    if mode == 'simple':
        # --- Wrapper API (XGBRegressor) ---
        importances = model.feature_importances_
        title = "XGBoost Feature Importance (Simple Mode - Wrapper API)"
        importance_type = "weight" # Default for feature_importances_ property
        
    elif mode == 'custom':
        # --- Native API (Booster) ---
        # Use 'gain' importance as it's typically more informative for tree models
        importances = model.get_score(importance_type='gain')
        title = "XGBoost Feature Importance (Custom Mode - Native API)"
        
        # Convert the native dict score to a consistent list/array format
        importances = pd.Series(importances).reindex(X_train_cols).fillna(0).values 
        
    else:
        print("⚠️ Warning: Unknown mode, skipping feature importance analysis.")
        return

    # Put into a DataFrame for readability (Common logic for both modes)
    feat_imp = pd.DataFrame({
        "feature": X_train_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    print(feat_imp.head(13).to_markdown(index=False))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # We must plot based on the original model type
    if mode == 'simple':
        xgb.plot_importance(model, importance_type="gain", max_num_features=20, ax=ax, title=title)
        ff.save_plot('4a_feature_importance_simple.png')
    elif mode == 'custom':
        # For native Booster, importance_type must be passed to get_score() first. 
        # Then, plotting needs the 'gain' type specified.
        # Note: We must call get_score() again here, or use the pre-calculated dict, which is slightly more complex.
        # Sticking to `get_score` and plotting the correct way is simplest.
        xgb.plot_importance(model, importance_type="gain", max_num_features=20, ax=ax, title=title)
        ff.save_plot('4a_feature_importance_custom.png')

        fig, ax = plt.subplots(figsize=(8, 6))
        xgb.plot_importance(model, importance_type="gain", max_num_features=20, ax=ax, title=title)

# Save the figure object
        with open('feature_importance_fig.pkl', 'wb') as f:
            pickle.dump(fig, f)



# ====================================================================
# --- 4. UNIFIED MODEL TRAINING ---
# ====================================================================

def train_model(X_train_s, X_test_s, y_train_s, y_test_s, mode):
    """
    Trains the XGBoost model using either 'custom' (Native API) or 'simple' (Wrapper API).
    """
    target = TARGET_COL
    history = {}

    
    if mode == 'custom':
        print(f"⏳ Starting **Custom** training (Native API,j)...")
        dtrain = xgb.DMatrix(X_train_s, label=y_train_s[target])
        dval   = xgb.DMatrix(X_test_s, label=y_test_s[target]) 

        params = {
            'max_depth': 4, 'min_child_weight': 300, 'subsample': 0.5,
            'nthread': -1, 'eval_metric': ["rmse"],'colsample_bytree': 0.6,
            'base_score': float(y_train_s[target].mean()), 'tree_method': 'hist'
        }
        #callbacks = [LearningRateScheduler(lr_scheduler)]
        callbacks=[LearningRateScheduler(lr_scheduler), EarlyStopping(rounds=30, save_best=True, data_name="val", metric_name="rmse")]

        watchlist = [(dtrain, "train"), (dval, "val")]
        
        bst = xgb.train(params, dtrain, num_boost_round=1360,# obj=mse_extreme_obj,
                        evals=watchlist, callbacks=callbacks, verbose_eval=10)
        
       # history = bst.evals_result()
        bst.save_model(MODEL_FILE)
        model = bst
       
    
        
    elif mode == 'simple':
        print(f"⏳ Starting **Simple** training (Wrapper API, Objective: reg:squarederror)...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=57, learning_rate=0.002061004, max_depth=9, min_child_weight=3,
            subsample=1, base_score=float(y_train_s[target].mean()), 
            n_jobs=-1,
            eval_metric=["mae", "rmse"], tree_method="hist" 
        )

        xgb_model.fit(X_train_s, y_train_s[target],
                      eval_set=[(X_train_s, y_train_s[target]), (X_test_s, y_test_s[target])],
                      verbose=10)
        
        xgb_model.save_model(MODEL_FILE)
        history = xgb_model.evals_result()
        model = xgb_model # Assign model object

    else:
        raise ValueError(f"Unknown training mode: {mode}. Must be 'custom' or 'simple'.")

   
    analyze_feature_importance(model, X_train_s.columns, mode, PLOT_FLAG)

    print(f"✅ Model saved as {MODEL_FILE}")
    return history

# ====================================================================
# --- 5. PLOTTING FUNCTIONS ---
# ====================================================================

def plot_train_val_test_distributions(X_train, X_val, X_test, y_train, y_val, y_test, step=PLOT_SAMPLED_STEP):
    """Plots feature and target distributions for Train/Val/Test splits."""
    print("🎨 Plotting feature distributions...")
    
    X_train_sampled = X_train.iloc[::step, :]; X_val_sampled = X_val.iloc[::step, :]
    X_test_sampled = X_test.iloc[::step, :]
    y_train_sampled = y_train.iloc[::step, :]; y_val_sampled = y_val.iloc[::step, :]
    y_test_sampled = y_test.iloc[::step, :]
    
    n_features = X_train.shape[1]
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols

    # Feature KDE Plots
    plt.figure(figsize=(5*ncols, 4*nrows))
    for i, col in enumerate(X_train.columns):
        plt.subplot(nrows, ncols, i+1)
        sns.kdeplot(X_train_sampled[col], label="Train", fill=True, alpha=0.4)
        sns.kdeplot(X_val_sampled[col], label="Val", fill=True, alpha=0.4)
        sns.kdeplot(X_test_sampled[col], label="Test", fill=True, alpha=0.4)
        plt.title(col, fontsize=10); plt.legend(fontsize=8)
    plt.tight_layout(); 
    ff.save_plot('training.png')

    # Target Density Plots
    n_targets = y_train.shape[1]
    plt.figure(figsize=(5*ncols, 4*n_targets))
    for i, col in enumerate(y_train.columns):
        plt.subplot(1, 1, i+1) # Assuming one target
        for data, color, label in [
            (y_train_sampled[col], "blue", "Train"), (y_val_sampled[col], "green", "Val"), (y_test_sampled[col], "red", "Test")
        ]:
            density, bins = np.histogram(data, bins=50, density=True)
            centers = (bins[1:] + bins[:-1]) / 2
            plt.plot(centers, density, color=color, label=label, linewidth=2, alpha=0.8)
        plt.title(col, fontsize=10); plt.legend(fontsize=8)
    plt.tight_layout(); plt.show()


def plot_time_series_splits(y_train, y_test, y_val, idx_train, idx_test, idx_val, step=PLOT_TS_STEP):
    """Plots target values over sample index to visualize splits."""
    print("🎨 Plotting time series splits (index view)...")
    target1 = TARGET_COL
    
    plt.figure(figsize=(12,5))
    plt.plot(idx_val[::step],   y_val[target1].iloc[::step],   ".", markersize=2, label="Validation")
    plt.plot(idx_test[::step],  y_test[target1].iloc[::step],  ".", markersize=2, label="Test")
    plt.plot(idx_train[::step], y_train[target1].iloc[::step], ".", markersize=2, label="Train")
    plt.xlabel("Sample index"); plt.ylabel(f"Target ({target1}, Unscaled)")
    plt.title("Train / Test / Validation splits (Time Index View)")
    plt.legend(); plt.grid(True); 
    ff.save_plot('3_time_series_splits.png')

def plot_density_time_series(df_feat, versus='msis_rho', step=PLOT_TS_STEP):
    """Plots observed vs MSIS density over time."""
    print("🎨 Plotting density time series (Obs vs MSIS)...")
    df_down = df_feat.iloc[::step]
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_down['time'], df_down['rho_obs'], label='rho_obs')
    plt.plot(df_down['time'], df_down[versus], label='msis_rho')
    plt.title('MSIS and Observed Density vs. Time'); plt.xlabel('Time')
    plt.ylabel('Density [kg/m³]'+versus); plt.legend(); plt.grid(True); 
    ff.save_plot('plot_density_time_series'+versus+'.png')

def run_all_plots(df_feat, X_train, X_test, X_val, y_train, y_test, y_val, idx_train, idx_test, idx_val):
    """Executes all defined plotting functions based on PLOT_FLAG."""
    if PLOT_FLAG == 0:
        return
        
    print("\n--- Starting All Plotting Diagnostics ---")
    plot_density_time_series(df_feat)
    plot_train_val_test_distributions(X_train, X_val, X_test, y_train, y_val, y_test)
    plot_time_series_splits(y_train, y_test, y_val, idx_train, idx_test, idx_val)

    print("--- Finished Plotting Diagnostics ---\n")

# ====================================================================
# --- 6. MODEL PREDICTION AND REPORTING (FIXED DYNAMIC LOADING) ---
# ====================================================================

def predict_and_report(df_feat, X_val_s, y_val, idx_val, scaler_y, history=None):
    """
    Loads the model using dynamic loading (Booster or XGBRegressor),
    predicts, unscales, and runs diagnostics.
    """
    print("\n⏳ Loading model and evaluating...")
    
    # 1. DYNAMIC MODEL LOADING
    if not os.path.exists(MODEL_FILE):
        print(f"🛑 Error: Model file {MODEL_FILE} not found.")
        return

    model = None
    
    # Try loading as Scikit-learn wrapper (saved by 'simple' mode)
    try:
        model = xgb.XGBRegressor()
        model.load_model(MODEL_FILE)
        print(MODEL_FILE)
        print("✅ Model loaded successfully using XGBRegressor wrapper.")
        
    except xgb.core.XGBoostError:
        # Fallback to Native Booster API (saved by 'custom' mode)
        try:
            model = xgb.Booster()
            model.load_model(MODEL_FILE)
            print("✅ Model loaded successfully using Native Booster API.")
        except Exception as e:
            print(f"🛑 FATAL ERROR: Could not load model using either API: {e}")
            return
    
    # 2. PREDICTION STEP
    if isinstance(model, xgb.XGBRegressor):
        y_pred_val_scale = model.predict(X_val_s)
    elif isinstance(model, xgb.Booster):
        dval_s = xgb.DMatrix(X_val_s)
        y_pred_val_scale = model.predict(dval_s)
    else:
        # This should ideally not be reached due to the loading logic, but serves as a final safeguard
        print("🛑 Prediction failed: Loaded model type is unrecognized.")
        return

    # 3. UNSCALING AND REPORTING
    y_pred_val_unscaled = ff.unscale_y_pred(
        pd.Series(y_pred_val_scale, index=X_val_s.index, name=TARGET_COL),
        scaler_y, y_val.copy()
    ).squeeze()

    df_val = df_feat.loc[idx_val].copy()
    y_pred_val_unscaled = pd.Series(y_pred_val_unscaled, index=df_val.index)

    df_val["y_true_log"] = df_val[TARGET_COL]
    df_val["y_pred_log"] = y_pred_val_unscaled
    
    df_val["rho_pred"] = df_val["msis_rho"] * np.exp(df_val["y_pred_log"])
    df_val["rho_true"] = df_val["msis_rho"] * np.exp(df_val["y_true_log"])
    
    rmse_val_log = np.sqrt(mean_squared_error(df_val["y_true_log"], df_val["y_pred_log"]))
    print(f"Validation RMSE (Log Space): {rmse_val_log:.4f}")

    # Plot Training History (if available)
    if history and 'val' in history and 'mse' in history['val']:
        plt.figure(figsize=(8,5))
        plt.plot(history['train']['mae'], label="Train MAE")
        plt.plot(history['val']['mae'], label="Validation MAE")
        plt.xlabel("Boosting rounds"); plt.ylabel("MAE") 
        plt.title(f"XGBoost Training History ({TRAINING_MODE} mode)")
        plt.legend(); plt.grid(True); plt.show()
    
    # Call the final plotting function for model diagnostics
    ff.plot_val_densities_with_metrics3(
        df_val, time_col="time", sample_step=1,
        obs_col="rho_obs", msis_col="msis_rho", pred_col="rho_pred"
    )
    return df_val["y_pred_log"], idx_val, df_val
    print("✅ Evaluation complete.")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TARGET_COL = "log_ratio" 

def visualize_time_series_dynamics(df_feat, y_pred, year, n_days=5, feature_list=None):
    """
    Selects N random 24-hour periods, plotting target performance and key feature dynamics.
    
    Args:
        df_feat (pd.DataFrame): DataFrame containing original features and the target.
        y_pred (np.array/pd.Series): XGBoost predictions aligned to the df_feat index.
        year (int): The year to sample days from.
        n_days (int): Number of random 24-hour periods to plot.
        feature_list (list): List of feature column names to plot (e.g., ['ap_m3h', 'f107a']).
    """
    
    if feature_list is None:
        feature_list = ['ap_m3h', 'f107a', 'lst_lat_cos'] # Default features for diagnosis

    # 1. Combine data and predictions for easier slicing
    df_plot = df_feat.copy()
    # Ensure y_pred is aligned. Assuming y_pred is aligned to the index of df_feat used for prediction (Test/Val set)
    df_plot['prediction'] = y_pred 
    df_plot['time'] = df_plot.index # Assuming the index is a DatetimeIndex

    # Filter data for the specified year
    df_year = df_plot[df_plot['time'].dt.year == year]
    
    if df_year.empty:
        print(f"⚠️ No data found for year {year}. Skipping.")
        return

    # 2. Select N random start times for 24-hour periods
    unique_days = df_year['time'].dt.normalize().unique()
    if len(unique_days) < n_days:
        n_days = len(unique_days)
    
    random_days = np.random.choice(unique_days, size=n_days, replace=False)

    # 3. Create the multi-panel plots
    # Total panels per day is 1 (Target) + len(feature_list)
    num_panels_per_day = 1 + len(feature_list) 
    
    fig, axes = plt.subplots(n_days * num_panels_per_day, 1, 
                             figsize=(16, 3 * n_days * num_panels_per_day), 
                             sharex='col')
    
    if n_days * num_panels_per_day == 1:
         axes = [axes] # Ensure axes is iterable

    for i, start_time in enumerate(random_days):
        end_time = start_time + pd.Timedelta(days=1)
        
        # Slice the data for the 24-hour window
        df_window = df_year[(df_year['time'] >= start_time) & (df_year['time'] < end_time)]
        
        if df_window.empty: continue
            
        # --- TOP PANEL: TARGET PERFORMANCE ---
        ax_target = axes[i * num_panels_per_day]
        ax_target.plot(df_window['time'], df_window[TARGET_COL], 
                       label='Observed (True Ratio)', color='black', alpha=0.7)
        ax_target.plot(df_window['time'], df_window['prediction'], 
                       label='XGBoost Prediction', color='red', linestyle='--', alpha=0.8)
        ax_target.axhline(0, color='blue', linestyle=':', 
                          label='MSIS Baseline (log_ratio=0)', alpha=0.5)

        ax_target.set_title(f"Target Performance and Feature Dynamics: {start_time.strftime('%Y-%m-%d')}")
        ax_target.set_ylabel("Log Density Ratio (ln(Obs/MSIS))")
        ax_target.legend(loc='upper right')
        ax_target.grid(True, linestyle='--', alpha=0.6)
        
        # --- FEATURE PANELS ---
        for j, feature in enumerate(feature_list):
            ax_feature = axes[i * num_panels_per_day + 1 + j]
            
            # Plot the raw, unscaled feature value
            ax_feature.plot(df_window['time'], df_window[feature], 
                            label=feature, color='darkgreen', alpha=0.8)
            
            ax_feature.set_ylabel(feature)
            ax_feature.grid(True, linestyle='--', alpha=0.6)

            # Only set the x-label on the bottom-most panel for the day
            if j == len(feature_list) - 1:
                 ax_feature.tick_params(axis='x', rotation=45)
                 ax_feature.set_xlabel(f"Time on {start_time.strftime('%Y-%m-%d')}")
            else:
                 ax_feature.tick_params(axis='x', labelbottom=False)

    plt.tight_layout()
    plt.show()

# Example Usage (assuming you use the Test set and want to plot ap_change and lst_lat_cos):
# visualize_time_series_dynamics(df_test_data, y_pred_test, 
#                                year=2015, n_days=5, 
#                                feature_list=['ap_change', 'lst_lat_cos', 'f107a'])   

# ====================================================================
# --- 7. MAIN EXECUTION FUNCTION ---
# ====================================================================

def main():
    """Main function to run the entire machine learning workflow."""
    
    # --- Data Path Check ---
    data_path = "grace_data_merged.parquet"
    if not os.path.exists(data_path):
        print(f"🛑 Error: Data file not found at {data_path}. Please place it in the current directory.")
        return

    # 1. & 2. LOAD, CLEAN, AND FEATURE ENGINEER (CACHING LOGIC)
    df_feat, df = load_and_feature_engineer_data(data_path)
    print(df_feat.columns)


    # 3. SPLIT DATA
    X_train, X_test, X_val, y_train, y_test, y_val, idx_train, idx_test, idx_val = split_and_filter_data(df_feat)
    
    # 4. RUN PRE-TRAINING PLOTS (If PLOT_FLAG is 1)
    run_all_plots(df_feat, X_train, X_test, X_val, y_train, y_test, y_val, idx_train, idx_test, idx_val)
    
    # 5. SCALE DATA
    results_s = scale_data(X_train, X_test, X_val, y_train, y_test, y_val)
    X_train_s, X_test_s, X_val_s, y_train_s, y_test_s, y_val_s, scaler_X, scaler_y = results_s

    def check_dataframe_health(df, name):
    #"""Prints size, data types, and checks for NaNs."""

        print(f"\n--- 🩺 Health Check: {name} ---")
    
    # 1. Shape and Sample
        print(f"Shape: {df.shape}")
        print(f"Index Head: {df.head(2).index.tolist()} (Should be contiguous)")
    
    # 2. NaN Check
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            print(f"🛑 CRITICAL ERROR: Found {nan_count:,} NaN values in {name}!")
            print(f"   NaNs per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        else:
            print("✅ NaN Check: PASSED (0 NaNs)")
        
    # 3. Data Type Check
    # Check if all dtypes are float, which is required by XGBoost
        non_float_cols = df.select_dtypes(exclude=['float', 'float64']).columns.tolist()
        if non_float_cols:
            print(f"⚠️ DType WARNING: Found non-float columns in {name}: {non_float_cols}")
            print(df.dtypes)
        else:
            print("✅ DType Check: PASSED (All columns are float)")

    # 4. Data Range Check (Verify Scaling)
        if not df.empty:
            min_val = df.min().min()
            max_val = df.max().max()
            print(f"Range (Min/Max): {min_val:.4f} / {max_val:.4f} (Should be near -1 to 1)")
        

   # check_dataframe_health(X_train_s, "X_train_s (Features)")
   # check_dataframe_health(y_train_s, "y_train_s (Target)")
   # check_dataframe_health(X_test_s, "X_test_s (Features)")
   # check_dataframe_health(y_test_s, "y_test_s (Target)")

   # plot_train_val_test_distributions(X_train, X_val, X_test, y_train, y_val, y_test)
    plot_train_val_test_distributions(X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s)



    # 6. TRAIN MODEL (If TRAIN_FLAG is set)
    history = None
    if TRAIN_FLAG:
        history = train_model(X_train_s, X_test_s, y_train_s, y_test_s, mode=TRAINING_MODE)
    else:
        print("⚠️ Training skipped as TRAIN_FLAG is 0.")

    # 7. EVALUATE AND REPORT
    # NOTE: Moved the model existence check inside predict_and_report for cleaner logic
    y_pred_log_val, idx_val_ret,df_val  = predict_and_report(df_feat, X_val_s, y_val, idx_val, scaler_y, history=history)
    print(y_pred_log_val)
   
    # 8. Time-Series Visualization (Using the Validation Set Prediction)
    # The visualize_time_series_dynamics function needs the original df_feat
    # and a prediction series *aligned* to its indices.

    # Filter df_feat to only include the Validation set data for plotting context
    
    COLUMNS_TO_KEEP = df_feat.columns.tolist() + [TARGET_COL] + ['time'] # Add the time column name!

# 2. Re-slice the full original data based on the returned indices, ensuring the time column is present.
# (This step depends entirely on what you named your original full data frame)
# Assuming df_feat is the closest representation of the validation data *with* the features:
    df_val_data_context = df_feat.loc[idx_val_ret].copy() 

    

# Since the column is missing in df_feat, you need to use the full, original time index/column.
# If your original time index was called 'time' in the full data set, you might need to merge it back:

# --- Assuming you can access the original full DataFrame/Index ---

# If the original time index was the index itself:
    plot_density_time_series(df_val,  versus='rho_pred')
    print(df_val_data_context.head)
    # Use the captured prediction series
   

# Calculate n_steps dynamically
    start_index = 200
    n_steps = len(df_val) - start_index
# Call the plotting function
    ff.simple_index_plot(
        df_val,
        y_pred="rho_pred",
        start_index=start_index,
        n_steps=n_steps,
        feature_list=["f107","matched_tec_value", "ap_m6h","doy_cos"],
        y_target="rho_obs",
        Title="Full_Validation_Plot"
    )

    ff.simple_index_plot(df_val, y_pred="rho_pred", start_index=100, n_steps=5000,  feature_list=["f107","matched_tec_value", "ap_m6h","doy_cos"], y_target="rho_obs", Title="zero")
    ff.simple_index_plot(df_val, y_pred="rho_pred", start_index=1000000 , n_steps=5000,  feature_list=["f107","matched_tec_value", "ap_m6h", "doy_cos"], y_target="rho_obs", Title="FIRST")
    ff.simple_index_plot(df_val, y_pred="rho_pred", start_index=2000000 , n_steps=5000,  feature_list=["f107","matched_tec_value", "ap_m6h", "doy_cos"], y_target="rho_obs", Title="second")
    ff.simple_index_plot(df_val, y_pred="rho_pred", start_index=3000000 , n_steps=5000,  feature_list=["f107","matched_tec_value", "ap_m6h", "doy_cos"], y_target="rho_obs", Title="thrds")

    ff.simple_index_plot(df_val, y_pred="rho_pred", start_index=4000000 , n_steps=5000,  feature_list=["f107","matched_tec_value", "ap_m6h", "doy_cos"], y_target="rho_obs", Title="four")
    ff.simple_index_plot(df_val, y_pred="rho_pred", start_index=5000000 , n_steps=5000,  feature_list=["f107","matched_tec_value", "ap_m6h", "doy_cos"], y_target="rho_obs", Title="fith")
    ff.simple_index_plot(df_val, y_pred="rho_pred", start_index=5800000 , n_steps=5000,  feature_list=["f107","matched_tec_value", "ap_m6h", "doy_cos"], y_target="rho_obs", Title="fithhalf")

    

if __name__ == "__main__":
    # Environment Check (requires TensorFlow import only here)
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        gpus = tf.config.list_physical_devices('GPU')
    except ImportError:
        tf_version = "N/A (TensorFlow not installed)"
        gpus = "N/A"

    print("\n--- Environment Check ---")
    print(f"tf: {tf_version} | python: {platform.python_version()}")
    print(f"GPUs: {gpus}")
    print("-------------------------\n")
    
    main()