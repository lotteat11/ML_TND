"""
Shared configuration for the XGBoost density correction pipeline.
"""

PARQUET_FILE  = "grace_data_merged_v3.parquet"
MODEL_OUT     = "xgb_model_test.json"
MODEL_IN      = "xgb_model.json"
SCALER_X_OUT  = "scaler_xgboost_X_test.joblib"
SCALER_Y_OUT  = "scaler_xgboost_y_test.joblib"

TIME_MIN      = "2009-06-06"
TIME_MAX      = "2016-01-01"

TARGET        = "log_ratio"

FEATURES = [
    "f107a", "lat",
    "matched_tec_value",
    "lon_sin", "lon_cos", "doy_sin", "doy_cos",
    "f107", "alt_km", "lst_cos", "lst_sin", "ap_m3h", "ap_m6h",
]

COLS_TO_SCALE = [
    "f107", "ap_m3h", "ap_m6h", "lat", "f107a", "alt_km", "matched_tec_value",
]
