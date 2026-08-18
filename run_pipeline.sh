#!/bin/bash
# =============================================================================
# run_pipeline.sh — end-to-end pipeline runner
#
# Usage:
#   ./run_pipeline.sh new  [stages]     full-mission setup (train 2002-2015,
#                                       holdouts: quiet-2009, post-2016)
#   ./run_pipeline.sh storm [stages]    as 'new', plus the Mar-2015 G4 storm
#                                       held out of training and evaluated
#   ./run_pipeline.sh old  [stages]     published setup (train 2009-2016,
#                                       holdouts: pre-2009, post-2016)
#
# stages: comma-separated subset of  dns,tec,msis,merge,train,eval,ontrack
#         (default: all, in that order). Example:
#   ./run_pipeline.sh new dns,tec            # only download data
#   ./run_pipeline.sh new train,eval,ontrack # data already prepared
#
# Notes:
#   - 'tec' needs an Earthdata login in ~/.netrc (CDDIS).
#   - 'new' downloads ~3 GB of GRACE zips on first run; TEC IONEX files are
#     shared with the old setup's folder, so only 2002-2008 is fetched anew.
#   - Nothing from the published setup is overwritten: 'new' writes *_v5_full
#     model/scaler files, its own parquets, and runs_full/.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -lt 1 ]]; then
  echo "usage: ./run_pipeline.sh new|storm|old [stages]" >&2
  exit 1
fi
MODE=$1
STAGES=${2:-dns,tec,msis,merge,train,eval,ontrack}

# Python: prefer the project venv
if [[ -x "ven_2404/bin/python" ]]; then PY=${PY:-ven_2404/bin/python}; else PY=${PY:-python3}; fi
export MPLBACKEND=Agg   # scripts call plt.show(); keep batch runs headless

case "$MODE" in
  new|storm)
    # ---- data: full mission 2002-2017 ----
    export DNS_MISSION="GRACE"
    export DNS_YEARS="2002,2017"
    export DNS_OUTDIR="GRACE_0217_v02"
    export DNS_PARQUET_OUT="grace_dns_2002_2017.parquet"

    export TEC_START_YEAR=2002
    export TEC_END_YEAR=2017
    export TEC_OUT_DIR="ionex_files_0917_v4"     # reuse already-downloaded 2009-2017 IONEX

    export PYMSIS_INPUT="grace_dns_2002_2017.parquet"
    export PYMSIS_OUTPUT="grace_dns_with_tnd_y200217_v5.parquet"
    export PYMSIS_TIME_MIN="2002-01-01"
    export PYMSIS_TIME_MAX="2018-01-01"

    export MERGE_GRACE_PARQUET="grace_dns_with_tnd_y200217_v5.parquet"
    export MERGE_TEC_PARQUET="tec_codg_2002-2017_doy1-365_v2.parquet"
    export MERGE_OUTPUT="grace_data_merged_v5_full.parquet"
    export MERGE_TIME_MAX="2018-01-01"
    export MERGE_CHUNKED=1       # year-by-year merge; full-frame load OOMs on 24 GB

    # ---- training: 2002-2015, quiet-2009 kept out as interior holdout ----
    export TEC_LAG_MODE="time"   # exact t-2500s / t-24h TEC lags (gap-robust)
    # Storm-history ap features (ap_daily, ap_0h, ap_m9h, ap_avg12_33h,
    # ap_avg36_57h) — the integrated-heating drivers NRLMSIS itself uses.
    # Opt in with AP_HISTORY=1; must be set for BOTH train and ontrack.
    export AP_HISTORY="${AP_HISTORY:-1}"
    export TRAIN_PARQUET_FILE="grace_data_merged_v5_full.parquet"
    export TRAIN_TIME_MIN="2002-01-01"
    export TRAIN_TIME_MAX="2016-01-01"
    export TRAIN_TIME_EXCLUDE="2009-01-01,2009-06-06"
    export TRAIN_MODEL_OUT="xgb_model_v8_full_2002train.json"
    export TRAIN_SCALER_X_OUT="scaler_xgboost_X_v8_full_2002train.joblib"
    export TRAIN_SCALER_Y_OUT="scaler_xgboost_y_v8_full_2002train.joblib"
    # Adopt tuned hyperparameters only when explicitly asked for, so results
    # stay comparable with earlier runs by default:
    #   USE_TUNED=1 ./run_pipeline.sh storm train,eval
    PARAMS_JSON="${TRAIN_PARAMS_JSON:-tuning_v5/best_params.json}"
    if [[ "${USE_TUNED:-0}" == "1" && -f "$PARAMS_JSON" ]]; then
      export TRAIN_PARAMS_JSON="$PARAMS_JSON"
      export ONTRACK_PARAMS_JSON="$TRAIN_PARAMS_JSON"
      echo "  (using tuned hyperparameters from $TRAIN_PARAMS_JSON)"
    fi

    # ---- rolling out-of-sample evaluation: held-out regimes ----
    export ONTRACK_DATA_FILE="grace_data_merged_v5_full.parquet"
    export ONTRACK_MODEL_FILE="$TRAIN_MODEL_OUT"
    export ONTRACK_SCALER_X_FILE="$TRAIN_SCALER_X_OUT"
    export ONTRACK_SCALER_Y_FILE="$TRAIN_SCALER_Y_OUT"
    export ONTRACK_OUTPUT_ROOT="runs_v8_full_2002train"
    export ONTRACK_FILTERS="quiet2009,post2016"

    # 'storm' differs from 'new' ONLY in the extra Mar-2015 interior holdout
    # (and its own model/outputs), so the two are directly comparable: any
    # change in quiet-2009 / post-2016 controls is attributable to that cut.
    if [[ "$MODE" == "storm" ]]; then
      export TRAIN_TIME_EXCLUDE="2009-01-01,2009-06-06;2015-03-01,2015-04-15"
      # Feature-set variants write to separate files/roots so a run with the
      # storm-history drivers never overwrites (or, via ontrack's resume-skip,
      # silently reuses) the 15-feature results.
      if [[ "${AP_HISTORY:-1}" == "1" ]]; then
        _V="v8_storm_ap_2002train"
      else
        _V="v8_storm_2002train"
      fi
      export TRAIN_MODEL_OUT="xgb_model_${_V}.json"
      export TRAIN_SCALER_X_OUT="scaler_xgboost_X_${_V}.joblib"
      export TRAIN_SCALER_Y_OUT="scaler_xgboost_y_${_V}.joblib"
      export ONTRACK_MODEL_FILE="$TRAIN_MODEL_OUT"
      export ONTRACK_SCALER_X_FILE="$TRAIN_SCALER_X_OUT"
      export ONTRACK_SCALER_Y_FILE="$TRAIN_SCALER_Y_OUT"
      export ONTRACK_OUTPUT_ROOT="runs_${_V}"
      export ONTRACK_FILTERS="quiet2009,storm2015,post2016"
    fi
    ;;
  old)
    # Published setup — everything below matches the scripts' built-in defaults
    # except the DNS download, whose file defaults now point at 2002-2008.
    export DNS_MISSION="GRACE"
    export DNS_YEARS="2009,2016"
    export DNS_OUTDIR="GRACE_0916_v02"
    export DNS_PARQUET_OUT="grace_dns_2009_2016.parquet"
    # tec/msis/merge/train/eval/ontrack all run on their published defaults.
    ;;
  *)
    echo "unknown mode: $MODE (expected 'new', 'storm' or 'old')" >&2; exit 1 ;;
esac

banner() { echo; echo "=================================================================="; echo "  [$MODE] stage: $1"; echo "=================================================================="; }

for stage in ${STAGES//,/ }; do
  case "$stage" in
    dns)    banner "download GRACE DNS (TU Delft)";  $PY DataPreparation/download_dns.py ;;
    tec)    banner "download TEC IONEX (CDDIS)";     $PY DataPreparation/download_tec.py ;;
    msis)   banner "run NRLMSISE-2.1 (pymsis)";      $PY DataPreparation/run_pymsis.py ;;
    merge)  banner "merge TEC onto GRACE track";     $PY DataPreparation/merge_tec_grace.py ;;
    tune)   banner "hyperparameter random search";   $PY CoreModel/tune.py ;;
    train)  banner "train core XGBoost model";       $PY CoreModel/train.py ;;
    eval)   banner "evaluate core model";            $PY CoreModel/evaluate.py ;;
    ontrack) banner "rolling out-of-sample runs";    $PY Forecast/on_track.py ;;
    *) echo "unknown stage: $stage" >&2; exit 1 ;;
  esac
done

echo
echo "✅ pipeline ($MODE) done: $STAGES"
