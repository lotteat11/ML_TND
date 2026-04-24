# %%

import os
# Setting the environment variable to only expose device '1' 
# (which TensorFlow will then call '/GPU:0' internally)
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import polars as pl
from datetime import timedelta
import tensorflow as tf
import os
#os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
#os.environ["TF_NUM_INTEROP_THREADS"] = "2"


# %%
from typing import List
from tensorflow.keras.models import load_model
import joblib
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
import Feature_functions as ff
import numpy as np
import importlib
importlib.reload(ff)
# %%
import tensorflow as tf, os, platform
print("tf:", tf.__version__, "| python:", platform.python_version())
#print("CPUs:", tf.config.threading.get_intra_op_parallelism_threads())
print("GPUs:", tf.config.list_physical_devices('GPU'))



# This should now only show one GPU device (index 0).

# %%
#df = pd.read_parquet("grace_dns_with_tnd_y200910_v1_2408.parquet", engine="pyarrow")
#df = pd.read_parquet("grace_dns_with_tnd_y200912_v1_2608.parquet", engine="pyarrow")
df = pd.read_parquet("grace_dns_with_tnd_y200916_v4_0809.parquet", engine="pyarrow")
#df = df.drop_duplicates(subset=["time", "lat", "lon"])
df['time'] = pd.to_datetime(df['time'])
df = df[df['time'] < '2016-01-01']
df = df[df['time'] > '2009-06-06']
#df = df[df['time'] < '2016-01-01']
# %%
#df = df[df['time'] < '2010-01-01']
print(df.head(100))
print(df.tail(100))
import pandas as pd
import matplotlib.pyplot as plt

# Ensure time column is datetime (and sorted)
df = df.copy()
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')
# %%
# Take the last 3 days available in the data
end_time = df['time'].max()
start_time = end_time - pd.Timedelta(days=1)
mask = (df['time'] >= start_time) & (df['time'] <= end_time)
print(start_time)
print(end_time)
# %%
df_3d = df.loc[mask]
print(df_3d.head(100))
print(df_3d.tail(100))
# %%
plt.figure(figsize=(12, 6))
plt.plot(df_3d['time'], df_3d['msis_rho'], label='MSIS Density', color='orange', alpha=0.7)
plt.plot(df_3d['time'], df_3d['rho_obs'], label='Observed Density', color='blue', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Density (kg/m³)')
plt.title('MSIS vs Observed Atmospheric Density — Last 3 Days')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
#df = df.sort_values('time')

# Compute monthly averages
#df['month'] = df['time'].dt.to_period('M')
#monthly_avg = df.groupby('month').agg({
 #   'rho_obs': 'mean',
  #  'msis_rho': 'mean'
#}).reset_index()

# Convert month back to timestamp for plotting
#monthly_avg['month'] = monthly_avg['month'].dt.to_timestamp()

# Plot monthly averages
#plt.figure(figsize=(12, 6))
#plt.plot(monthly_avg['month'], monthly_avg['rho_obs'], label='Observed Density (Monthly Avg)', color='blue')
#plt.plot(monthly_avg['month'], monthly_avg['msis_rho'], label='MSIS Density (Monthly Avg)', color='orange')
#plt.xlabel('Time')
#plt.ylabel('Density (kg/m³)')
#plt.title('Monthly Average of MSIS vs Observed Atmospheric Density')
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.show()

# %%

import pandas as pd
import matplotlib.pyplot as plt

# Assuming df has columns: 'time', 'rho_obs', 'msis_rho'
# Ensure time is datetime and sort
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')


# %%
#CSV_FILENAME = "tec_codg_2009-2017_doy1-365.csv"
#df_tec = pd.read_csv(
 #       CSV_FILENAME,
  #      parse_dates=['epoch'],
   #     index_col=False
    #)
#print(df_tec(100))

# %%

PARQUET_FILENAME = "tec_codg_2009-2017_doy1-365.parquet"  # update extension

# Preferred engine is pyarrow; fallback to fastparquet if needed
try:
    df_tec = pd.read_parquet(PARQUET_FILENAME, engine="pyarrow")
except Exception:
    df_tec = pd.read_parquet(PARQUET_FILENAME, engine="fastparquet")



print(df_tec.head(100))
print(df_tec.tail(100))
print(df_tec.dtypes)  # useful to confirm column types

# %%

import polars as pl
from datetime import timedelta
# Assuming df and df_tec are Pandas DataFrames loaded earlier

# --- Initial Conversion to Polars ---
grace_df_pl = pl.from_pandas(df)
df_tec_pl = pl.from_pandas(df_tec)

# --- 1) GRACE: Standardize, Localize, and Rename ---
grace_df_pl = (
    grace_df_pl
      .with_columns(
          # Chain cast and time zone replacement on the original 'time' column
          pl.col("time")
            # 1. Cast to the required microsecond resolution
            .dt.cast_time_unit("us")
            # 2. Assign the UTC timezone (Polars standard method)
            .dt.replace_time_zone("UTC")
            .alias("grace_time"),
          # sequential index
          pl.arange(0, pl.len()).alias("original_index"),
      )
      .drop("time") # Drop the original column
      .sort("grace_time")
)

# --- 2) TEC: Standardize and Localize ---
df_tec_pl = (
    df_tec_pl
      .with_columns(
          # Chain cast and time zone replacement on the original 'epoch' column
          pl.col("epoch")
            .dt.cast_time_unit("us") # Ensure microsecond to match GRACE
            .dt.replace_time_zone("UTC")
            .alias("epoch_tec")
      )
      .drop("epoch") # Drop the original column
      .sort("epoch_tec")
)

# --- 3) Join Asof ---
tec_epochs_only = df_tec_pl.select("epoch_tec").unique().sort("epoch_tec")

temp_merged_time = grace_df_pl.join_asof(
    other=tec_epochs_only,
    left_on="grace_time",
    right_on="epoch_tec",
    strategy="nearest",
    tolerance=timedelta(hours=3),
)

# %%



# %%

print(temp_merged_time.columns)
# %%
#temp_merged_time.rename(columns={'epoch': 'epoch_tec_key'}, inplace=True) 

#print(f"✅ Step 1 Complete. Time-matched rows (index preserved): {len(temp_merged_time):,}")
#temp_merged_time.rename(columns={'epoch_tec': 'epoch'}, inplace=True)
# temp_merged_time now contains: [GRACE data] + [epoch_tec]. It is clean.
print(f"Step 1 Complete. Time-matched rows: {len(temp_merged_time):,}")
print(f"Step 1 Complete. Time-matched rows: {len(df):,}")
print(tec_epochs_only)# %%
print(temp_merged_time.columns)# %%

# %%¨

print(df_tec.columns)
# %%¨
#pl_left  = pl.from_pandas(temp_merged_time)  # has column 'epoch_y'
#pl_right = pl.from_pandas(df_tec)
#pl_right = pl_right.unique(subset=["epoch"], maintain_order=True)
#temp_merged_final_pl = pl_left.join(
 #   pl_right,
  #  left_on="epoch_tec",
   # right_on="epoch",
    #how="left"


# %%¨# %%
temp_merged_pl=temp_merged_time
print(temp_merged_pl.columns)
# %%
# %%





# %%
import polars as pl
import numpy as np
from scipy.spatial import cKDTree
from datetime import timedelta

# --- ASSUMED EXTERNAL DEFINITIONS ---

# 1. Coordinate Conversion Function (The haversine_tree must be defined)
def haversine_tree(coords: np.ndarray) -> np.ndarray:
    """Converts (lat, lon) in radians to 3D Cartesian coordinates (X, Y, Z)."""
    lat, lon = coords[:, 0], coords[:, 1]
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.vstack((x, y, z)).T

# 2. Parameters
MAX_CHORD_DISTANCE = 4.15 # Your established quality control threshold

# 3. Clean Input DataFrames (MUST be pre-loaded: grace_df_pl and tec_df_pl)
# The following lines are placeholders for your loaded data:
# grace_df_pl = ... # Load your main GRACE DataFrame here
# tec_df_pl = ...   # Load your clean, fully parsed TEC DataFrame here

# --------------------------------------------------------------------
# 🚀 CORE K-D TREE MATCHING LOGIC
# --------------------------------------------------------------------

#grace_df_pl  = pl.from_pandas(temp_merged_time)  # has column 'epoch_y'
pl_right = df_tec_pl 

# 1. Define Unique TEC Epochs (The loop driver)
unique_tec_epochs = pl_right.select(pl.col("epoch_tec")).unique().sort("epoch_tec").to_series().to_list()
final_matched_results = []
TIME_WINDOW = timedelta(hours=3) # +/- 1 hour around the 2-hour epoch
k_grace=0
print(f"Starting K-D Tree matching for {len(unique_tec_epochs)} epochs...")
print(f"Total epochs to process: {len(unique_tec_epochs)}")
for i, epoch in enumerate(unique_tec_epochs):
    
    # --- A. Prepare TEC Grid (Search Space) ---
    # Filter the clean TEC data to get the full grid (approx. 2880 points) for this epoch
    tec_grid_for_epoch = pl_right.filter(pl.col("epoch_tec") == epoch)

    # Convert to NumPy for K-D Tree: [latitude, longitude, tec_value]
    tec_grid_np = tec_grid_for_epoch.select(["latitude", "longitude", "tec_value"]).to_numpy()
    
    # CRITICAL CHECK: Ensure we have the full grid
    if tec_grid_np.shape[0] < 100: 
        print(f"Warning: Skipping epoch {epochp} due to missing TEC grid points ({tec_grid_np.shape[0]} found).")
        continue

    # TEC Coordinates in Radians
    tec_coords_rad = np.radians(tec_grid_np[:, :2]) 
    
    # --- B. Prepare GRACE Targets (Query Points) ---
    # Filter GRACE points that temporally match this epoch's window
    grace_targets_pl = grace_df_pl.filter(
        (pl.col("grace_time") >= epoch - TIME_WINDOW) &
        (pl.col("grace_time") < epoch + TIME_WINDOW)
    ).select(["original_index", "lat", "lon"])

    if grace_targets_pl.height == 0:
      #  print(epoch)
     #   print("no grace")
        k_grace=k_grace+1
        continue # No GRACE points in this time window
        
    # GRACE Coordinates in Radians
    grace_coords_np = grace_targets_pl.select(["lat", "lon"]).to_numpy()
    grace_coords_rad = np.radians(grace_coords_np)
    
    # --- C. K-D Tree Operations (Core Spatial Search) ---
    # 1. Build the K-D Tree
    tec_tree = cKDTree(haversine_tree(tec_coords_rad))
    # 2. Convert the GRACE query points
    grace_cartesian = haversine_tree(grace_coords_rad)
    
    # 3. Query Nearest Neighbor with threshold
    distances, indices = tec_tree.query(
        grace_cartesian, 
        k=1, 
        distance_upper_bound=MAX_CHORD_DISTANCE
    )
    
    # --- D. Consolidate and Handle NaNs ---
    # cKDTree.n is the size of the search space (tec_tree.n == tec_grid_np.shape[0])
    i# --- D. Consolidate and Handle NaNs (Refined and Corrected) ---
    # cKDTree.n is the size of the search space (tec_tree.n == tec_grid_np.shape[0])
    invalid_mask = (indices == tec_tree.n) 
    valid_mask = ~invalid_mask  # New: Explicitly define the valid mask
    num_grace_points = grace_targets_pl.height
    
    # Initialize result array with NaNs: [lat, lon, vtec_value]
    matched_tec_data_np = np.full((num_grace_points, 3), np.nan) 
    
    # --- D1. Assign Valid Matches (Vectorized) ---
    
    valid_indices = indices[valid_mask] # Indices into tec_grid_np
    
    # Look up the TEC data using the valid indices from the query result
    # Columns: [latitude, longitude, tec_value]
    matched_data_raw = tec_grid_np[valid_indices] 
    
    # Assign data back into the result array for *valid* matches
    matched_tec_data_np[valid_mask, 0] = matched_data_raw[:, 0] # matched_tec_latitude
    matched_tec_data_np[valid_mask, 1] = matched_data_raw[:, 1] # matched_tec_longitude
    matched_tec_data_np[valid_mask, 2] = matched_data_raw[:, 2] # matched_tec_value

    # Initialize quality flag
    quality_flag = np.zeros(num_grace_points, dtype=np.int8)
    
    # --- D2. Apply Fallback Logic (Iterative over the GRACE points) ---
    # Fills invalid (NaN) TEC values with the last valid TEC value found.
    last_valid_tec_value = np.nan 
    
    for idx in range(num_grace_points):
        if invalid_mask[idx]:
            # If distance > MAX_CHORD_DISTANCE, assign fallback value
            matched_tec_data_np[idx, 2] = last_valid_tec_value  
            quality_flag[idx] = 1 # Mark that a fallback was used
        else:
            # Update the last_valid_tec_value with the current valid match
            last_valid_tec_value = matched_tec_data_np[idx, 2]

    # Create final Polars DataFrame for this epoch
    matched_tec_data_pl = pl.DataFrame({
        "original_index": grace_targets_pl["original_index"].to_list(),
        "matched_tec_value": matched_tec_data_np[:, 2],
        "matched_tec_latitude": matched_tec_data_np[:, 0],
        "matched_tec_longitude": matched_tec_data_np[:, 1],
        "chord_distance": distances,
        "fallback_used": quality_flag
    })
    
    final_matched_results.append(matched_tec_data_pl)
    
    if i % 500 == 0 and i > 0:
        print(f"Progress: {i}/{len(unique_tec_epochs)} epochs matched.")
# --------------------------------------------------------------------
# 2. FINAL CONSOLIDATION
# --------------------------------------------------------------------

print("Consolidating final results...")
final_matched_pl = pl.concat(final_matched_results)


final_matched_pl = final_matched_pl.sort(["original_index", "chord_distance"])

final_matched_pl = final_matched_pl.unique(subset=["original_index"], keep="first")

# Join the matched results back to the original GRACE data
grace_df_final_pl = grace_df_pl.join(
    final_matched_pl,
    on="original_index",
    how="left"
)

# You can now proceed to drop NaNs on grace_df_final_pl
print(f"Final Data Shape: {grace_df_final_pl.shape}")
print(f"INput grace Data Shape: {df.shape}")
print(f"INput grace Data Shape: {df_tec.shape}")
print("Matching complete. Final DataFrame includes all GRACE points, with NaNs for rejected TEC matches.")
# %%
print(grace_df_final_pl)
print(k_grace)
# %%
grace_df_final_pl.write_parquet("grace_data_merged_v3.parquet")

# %%
