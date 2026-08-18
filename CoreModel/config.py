# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
config.py
- Central config for file paths, feature list, and target variable.
- Shared by train.py and evaluate.py; update model/scaler paths here if switching versions.
- Feature set and scaling columns match the v3 model (15 features, includes TEC lags).
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Defaults below can be overridden via environment variables (see run_pipeline.sh)
PARQUET_FILE  = ROOT / os.environ.get("TRAIN_PARQUET_FILE", "grace_data_merged_v3.parquet")
MODEL_OUT     = ROOT / os.environ.get("TRAIN_MODEL_OUT", "xgb_model_v3.json")
SCALER_X_OUT  = ROOT / os.environ.get("TRAIN_SCALER_X_OUT", "scaler_xgboost_X_v3.joblib")
SCALER_Y_OUT  = ROOT / os.environ.get("TRAIN_SCALER_Y_OUT", "scaler_xgboost_y_v3.joblib")

TIME_MIN      = os.environ.get("TRAIN_TIME_MIN", "2009-06-01")
TIME_MAX      = os.environ.get("TRAIN_TIME_MAX", "2016-01-01")

# Optional interior holdouts kept OUT of training, e.g. the quiet-2009 test
# period and the March-2015 storm. Format: one or more "start,end" pairs
# (start inclusive, end exclusive) separated by ";"; unset = no holdout.
#   "2009-01-01,2009-06-06;2015-03-01,2015-04-15"
_excl = os.environ.get("TRAIN_TIME_EXCLUDE", "").strip()
TIME_EXCLUDE = ([tuple(p.split(",")) for p in _excl.split(";") if p.strip()]
                if _excl else None)

# TEC lag features: "rows" = historical shift(500)/shift(17280) (v3 model),
# "time" = exact, gap-robust lookups at t-2500s / t-24h (full-mission setup).
TEC_LAG_MODE  = os.environ.get("TEC_LAG_MODE", "rows")

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

# Geomagnetic storm-history drivers. The v3/v5 feature set sees only ap_m3h and
# ap_m6h (a 6-hour window), while NRLMSIS-2.1 itself consumes a 7-element ap
# vector spanning ~57 hours: storm-time density depends on INTEGRATED Joule
# heating, not the instantaneous index. Enable with AP_HISTORY=1 to give the
# correction model the same driver history as the baseline it corrects.
# AP_HISTORY=1 adds the three that carry genuinely new information: ap_m9h
# extends the short-range history, and the two averaged terms are the
# integrated-heating drivers (they peak ~a day after the instantaneous index).
# AP_HISTORY=full also adds ap_daily and ap_0h, which are largely redundant
# with the averages and with ap_m3h respectively.
AP_HISTORY_FEATURES = [
    "ap_m9h", "ap_avg12_33h", "ap_avg36_57h",
]
AP_HISTORY_FEATURES_FULL = ["ap_daily", "ap_0h"] + AP_HISTORY_FEATURES

_ap = os.environ.get("AP_HISTORY", "1")
if _ap != "0":
    _extra = AP_HISTORY_FEATURES_FULL if _ap == "full" else AP_HISTORY_FEATURES
    FEATURES = FEATURES + _extra
    COLS_TO_SCALE = COLS_TO_SCALE + _extra
