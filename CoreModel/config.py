# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
config.py
- Central config for file paths, feature list, and target variable.
- Single source of truth for FEATURES/COLS_TO_SCALE: CoreModel and Forecast both
  import from here, so a feature change lands in one place.
- 14 base features + a single time-based TEC lag; AP_HISTORY=1 (the default)
  brings the live set to 17.
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

# The model sees the current TEC map plus one lagged map per entry below: exact,
# gap-robust lookups matched within the same satellite track (see
# feature_functions.add_tec_time_lag_features). The 3 h lag carries the recent
# ionospheric state and is the production feature set: it is what the shipped
# v8_storm_ap model and its scalers were fitted on, so changing this default
# silently invalidates them.
# Add lags to re-open the choice, e.g. TEC_LAGS=3h,24h — the 24 h lag carries
# the same local-time geometry one day earlier, the timescale the density
# response to Joule heating peaks on. Any such run needs a RETRAINED model and
# scalers; the 17-feature production files reject the extra column.
_lags = os.environ.get("TEC_LAGS", "3h")
TEC_LAGS     = tuple(x.strip() for x in _lags.split(",") if x.strip())
if not TEC_LAGS:
    raise ValueError("TEC_LAGS must name at least one lag, e.g. '3h,24h'")
# Column names are positional and historical: the first lag is
# vtec_matched_lag, the second vtec_matched_lag2, and so on.
TEC_LAG_COLS = tuple("vtec_matched_lag" + ("" if i == 0 else str(i + 1))
                     for i in range(len(TEC_LAGS)))
# Kept for callers that assume a single lag.
TEC_LAG      = TEC_LAGS[0]
TEC_LAG_COL  = TEC_LAG_COLS[0]

TARGET        = "log_ratio"

FEATURES = [
    "f107a", "lat",
    "matched_tec_value",
    "lon_cos", "lon_sin",
    "lst_sin",
    "ap_m3h",
    "doy_sin", "doy_cos", "f107", "alt_km",
    "ap_m6h",
    *TEC_LAG_COLS,
    "lst_lat_sin",
]

COLS_TO_SCALE = [
    "f107", "ap_m6h", "lat", "f107a", "alt_km",
    "matched_tec_value", "ap_m3h", *TEC_LAG_COLS,
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

# NO_AP=1 drops EVERY ap column, including the ap_m3h/ap_m6h pair that is part
# of the base feature list and therefore survives AP_HISTORY=0. Ablation only:
# it removes the model's sole geomagnetic driver, so a drop in skill is
# expected. Use it to measure how much of the TEC signal is really ap acting
# through the ionosphere, since TEC and ap are strongly correlated during
# storms and an apparent TEC gain can be ap arriving by another route.
if os.environ.get("NO_AP", "0") == "1":
    FEATURES = [f for f in FEATURES if not f.startswith("ap")]
    COLS_TO_SCALE = [c for c in COLS_TO_SCALE if not c.startswith("ap")]
