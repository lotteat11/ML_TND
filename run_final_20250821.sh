#!/bin/bash
# =============================================================================
# run_final_20250821.sh — final reported rolling evaluation
#
# Three held-out regimes x {core, warm-start} x horizons {1, 3} = 12 runs,
# at the adopted 3-day warm-start lookback.
#
# Every setting below is now the default in the code (Forecast/on_track.py and
# CoreModel/config.py); they are restated here so this script is a complete,
# readable record of what produced the reported table.
#
# Output: runs_final_20250821/
#
# Usage:  ./run_final_20250821.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

if [[ -x "ven_2404/bin/python" ]]; then PY=${PY:-ven_2404/bin/python}; else PY=${PY:-python3}; fi
export MPLBACKEND=Agg

# ---- feature set (must match the model) --------------------------------------
export TEC_LAGS="3h"
export AP_HISTORY=1

# ---- model / scalers / data --------------------------------------------------
export ONTRACK_MODEL_FILE="xgb_model_v8_storm_ap_2002train.json"
export ONTRACK_SCALER_X_FILE="scaler_xgboost_X_v8_storm_ap_2002train.joblib"
export ONTRACK_SCALER_Y_FILE="scaler_xgboost_y_v8_storm_ap_2002train.joblib"
export ONTRACK_DATA_FILE="grace_data_merged_v5_full.parquet"
export ONTRACK_PARAMS_JSON="tuning_v13_tec3h_depth3_10/best_params.json"

# ---- warm-start ---------------------------------------------------------------
export ONTRACK_LOOKBACK_DAYS=3
export ONTRACK_RESET_EVERY=4
export ONTRACK_WARMSTART_LR=0.005
export ONTRACK_WARMSTART_LR_DECAY=0.9
export ONTRACK_WARMSTART_LR_STEP=20
export ONTRACK_WARMSTART_ROUNDS=2000
export ONTRACK_WARMSTART_PATIENCE=60

# ---- what to evaluate ---------------------------------------------------------
export ONTRACK_FILTERS="quiet2009,storm2015,post2016"
export ONTRACK_HORIZONS="1,3"
export ONTRACK_OUTPUT_ROOT="runs_final_20250821"

# The post-2016 h=3 prediction frames run to ~18M rows; the CSV twin of each is
# ~2 GB of pure duplication on top of the pickle every reader actually uses.
export ONTRACK_CSV_MAX_ROWS=5000000

echo "lookback : $ONTRACK_LOOKBACK_DAYS d   horizons: $ONTRACK_HORIZONS"
echo "model    : $ONTRACK_MODEL_FILE"
echo "regimes  : $ONTRACK_FILTERS"
echo "output   : $ONTRACK_OUTPUT_ROOT"
echo

$PY -u Forecast/on_track.py

echo
echo "Building results table ..."
$PY Forecast/make_table_regimes.py --run "$ONTRACK_OUTPUT_ROOT" --horizon 1 \
    --out "$ONTRACK_OUTPUT_ROOT/table_regimes_h1.tex"
$PY Forecast/make_table_regimes.py --run "$ONTRACK_OUTPUT_ROOT" --horizon 3 \
    --out "$ONTRACK_OUTPUT_ROOT/table_regimes_h3.tex"

echo
echo "done -> $ONTRACK_OUTPUT_ROOT"
