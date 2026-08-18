#!/bin/bash
# =============================================================================
# run_storm_pipeline.sh — train + evaluate the storm-holdout setup end to end,
# then print the March-2015 report. Chains the three commands used to test
# the March-2015 holdout, so the whole thing can run unattended overnight.
#
# Usage:
#   ./run_storm_pipeline.sh                 # 15 features, train includes 2002, h1 only
#   AP_HISTORY=1 ./run_storm_pipeline.sh    # + ap storm-history features, train includes 2002, h1 only
#   ONTRACK_HORIZONS=1,3 ./run_storm_pipeline.sh  # explicitly run h1 and h3
#   USE_TUNED=1 ./run_storm_pipeline.sh     # tuned hyperparameters
#
# Each feature-set variant uses its own model, run directory and logs, so runs
# never overwrite each other.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Flush progress into tee immediately (so a SIGKILL log shows the exact run)
# and use the non-interactive plotting backend for unattended execution.
export PYTHONUNBUFFERED=1
export MPLBACKEND="${MPLBACKEND:-Agg}"
export ONTRACK_HORIZONS="${ONTRACK_HORIZONS:-1}"

if [[ -x "ven_2404/bin/python" ]]; then PY=${PY:-ven_2404/bin/python}; else PY=${PY:-python3}; fi

# Must match the variant naming in run_pipeline.sh's storm block.
if [[ "${AP_HISTORY:-1}" == "1" ]]; then V="v8_storm_ap_2002train"; else V="v8_storm_2002train"; fi
echo "variant: $V  (AP_HISTORY=${AP_HISTORY:-1}, USE_TUNED=${USE_TUNED:-0}, HORIZONS=$ONTRACK_HORIZONS)"

caffeinate -is ./run_pipeline.sh storm train,eval 2>&1 | tee "pipeline_train_${V}.log"
caffeinate -is ./run_pipeline.sh storm ontrack    2>&1 | tee "pipeline_ontrack_${V}.log"

# --runs pins the directory: the prompt can't be answered through a pipe.
$PY Forecast/make_table_storm.py --daily --runs "runs_${V}" 2>&1 | tee "pipeline_report_${V}.log"

echo
echo "✅ storm pipeline done — report above and in pipeline_report_${V}.log"
