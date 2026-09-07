#!/bin/bash
# =============================================================================
# run_snapshot_20160216.sh — warm-start snapshot for the off-track map
#
# Forecast/off_track.py builds its global map from a warm-start model snapshot
# saved during a rolling run, so the map can only be made for a day that has
# one. This run produces the snapshot for 2016-02-16, the day shown in the
# on-track showcase figure (panel c, ap_m6h peaking at 56), so the on-track and
# off-track figures depict the same epoch.
#
# Everything except the snapshot date and the output root is identical to
# run_final_20250821.sh, and to run_snapshot_20160307.sh alongside it. Only the
# snapshot is wanted; the metrics it produces should reproduce the reported
# ones.
#
# Usage:  ./run_snapshot_20160216.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

if [[ -x "ven_2404/bin/python" ]]; then PY=${PY:-ven_2404/bin/python}; else PY=${PY:-python3}; fi
export MPLBACKEND=Agg

# ---- the only intended difference -------------------------------------------
export ONTRACK_SNAPSHOT_POST2016="2016-02-16"

# ---- identical to run_final_20250821.sh --------------------------------------
export TEC_LAGS="3h"
export AP_HISTORY=1
export ONTRACK_MODEL_FILE="xgb_model_v8_storm_ap_2002train.json"
export ONTRACK_SCALER_X_FILE="scaler_xgboost_X_v8_storm_ap_2002train.joblib"
export ONTRACK_SCALER_Y_FILE="scaler_xgboost_y_v8_storm_ap_2002train.joblib"
export ONTRACK_DATA_FILE="grace_data_merged_v5_full.parquet"
export ONTRACK_PARAMS_JSON="tuning_v13_tec3h_depth3_10/best_params.json"
export ONTRACK_LOOKBACK_DAYS=3
export ONTRACK_RESET_EVERY=4
export ONTRACK_WARMSTART_LR=0.005
export ONTRACK_WARMSTART_LR_DECAY=0.9
export ONTRACK_WARMSTART_LR_STEP=20
export ONTRACK_WARMSTART_ROUNDS=2000
export ONTRACK_WARMSTART_PATIENCE=60

# Only the one run that carries the snapshot: dr1 writes it, dr0 does not.
export ONTRACK_FILTERS="post2016"
export ONTRACK_HORIZONS="1"
export ONTRACK_RETRAIN="1"
export ONTRACK_OUTPUT_ROOT="runs_snapshot_20160216"

# The prediction CSV is a ~2 GB write-only duplicate of the pickle.
export ONTRACK_CSV_MAX_ROWS=1000

echo "snapshot date : $ONTRACK_SNAPSHOT_POST2016"
echo "output        : $ONTRACK_OUTPUT_ROOT  (runs_final_20250821 untouched)"
echo

$PY -u Forecast/on_track.py

echo
echo "snapshot written:"
ls -la "$ONTRACK_OUTPUT_ROOT"/dr1_post2016_h1/xgb_model_saved_* 2>/dev/null || echo "  (none — check the log)"
