"""
config.py
- Central config for file paths, feature list, and target variable.
- Shared by train.py and evaluate.py; update model/scaler paths here if switching versions.
- Feature set and scaling columns match the v3 model (15 features, includes TEC lags).
"""

PARQUET_FILE  = "grace_data_merged_v3.parquet"
MODEL_OUT     = "xgb_model_v3.json"
SCALER_X_OUT  = "scaler_xgboost_X_v3.joblib"
SCALER_Y_OUT  = "scaler_xgboost_y_v3.joblib"

TIME_MIN      = "2009-06-01"
TIME_MAX      = "2016-01-01"

TARGET        = "log_ratio"

FEATURES = [
    "f107a", "lat",
    "matched_tec_value",
    "lon_cos", "lon_sin",
    "lst_sin",
    "ap_m3h",
    "doy_sin", "doy_cos", "f107", "alt_km",
    "ap_m6h",
    "vtec_matched_lag", "vtec_matched_lag2",
    "lst_lat_sin",
]

COLS_TO_SCALE = [
    "f107", "ap_m6h", "lat", "f107a", "alt_km",
    "matched_tec_value", "ap_m3h", "vtec_matched_lag", "vtec_matched_lag2",
]
