#!/bin/bash
# =============================================================================
# run_test_lb3.sh — ontrack with a 3-day warm-start lookback
#
# WHY: the 2002-2003 sweep (lookback_sensitivity_2002_2003/) found 3 and 5 days
# indistinguishable in aggregate (log-RMSE 0.1641 vs 0.1643) with 7 slightly
# worse. That sweep ran INSIDE the training window, so it compared lookbacks
# against each other rather than measuring held-out skill. This run repeats the
# comparison on the three real holdout regimes, and lets us check the storm
# ONSET specifically: a shorter lookback carries less quiet-day history into a
# rising storm, so it may reduce the onset overshoot documented for dr1.
#
# BASELINE: runs_v14_report_h1h3 used ONTRACK_LOOKBACK_DAYS=5 (inferred from
# its first forecast date: data starts day 1, first prediction day 7, and the
# loop starts at LOOKBACK_DAYS+1). Note on_track.py's built-in default is 14 —
# the 5 was supplied from the environment and is not recorded in the repo.
#
# Everything else matches that run exactly: same model, scalers, data, tuned
# warm-start params, reset cadence, regimes and horizon. Only the lookback
# differs, so any change is attributable to it.
#
# Writes only runs_test_lb3/ — nothing existing is touched.
#
# Usage:  ./run_test_lb3.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

if [[ -x "ven_2404/bin/python" ]]; then PY=${PY:-ven_2404/bin/python}; else PY=${PY:-python3}; fi
export MPLBACKEND=Agg

# ---- the variable under test -------------------------------------------------
export ONTRACK_LOOKBACK_DAYS=3        # <-- the ONLY intended difference (baseline 5)

# ---- everything below matches the v14 report run -----------------------------
export TEC_LAGS="3h"
export AP_HISTORY=1
export ONTRACK_RESET_EVERY=4

export ONTRACK_DATA_FILE="grace_data_merged_v5_full.parquet"
export ONTRACK_MODEL_FILE="xgb_model_v8_storm_ap_2002train.json"
export ONTRACK_SCALER_X_FILE="scaler_xgboost_X_v8_storm_ap_2002train.joblib"
export ONTRACK_SCALER_Y_FILE="scaler_xgboost_y_v8_storm_ap_2002train.joblib"
export ONTRACK_PARAMS_JSON="tuning_v13_tec3h_depth3_10/best_params.json"

export ONTRACK_OUTPUT_ROOT="runs_test_lb3"
export ONTRACK_FILTERS="quiet2009,storm2015,post2016"
export ONTRACK_HORIZONS="1"

echo "lookback : $ONTRACK_LOOKBACK_DAYS days (baseline run used 5)"
echo "model    : $ONTRACK_MODEL_FILE"
echo "runs     : $ONTRACK_OUTPUT_ROOT"
echo "regimes  : $ONTRACK_FILTERS  h=$ONTRACK_HORIZONS"
echo

$PY -u Forecast/on_track.py

echo
echo "test_lb3 done -> $ONTRACK_OUTPUT_ROOT"
