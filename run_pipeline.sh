#!/bin/bash
set -e

PYTHON="$(dirname "$0")/ven_2404/bin/python"
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "======================================================"
echo " Thermospheric Density Pipeline"
echo " Working directory: $ROOT"
echo "======================================================"
echo ""

# ------------------------------------------------------------------
# Step 1 — Download GRACE/Swarm satellite data
# Requires: FTP access to thermosphere.tudelft.nl
# Edit MISSION, YEARS, OUTDIR in GettingData.py before running.
# ------------------------------------------------------------------
echo "[1/9] Downloading satellite density data..."
$PYTHON "$ROOT/DataPreparation/GettingData.py"
echo "Done."
echo ""

# ------------------------------------------------------------------
# Step 2 — Download TEC (IONEX) files
# Requires: Earthdata login credentials saved in ~/.netrc
# Edit START_YEAR, END_YEAR, CENTER, OUT_DIR in ImportTec.py.
# ------------------------------------------------------------------
echo "[2/9] Downloading TEC data..."
$PYTHON "$ROOT/DataPreparation/ImportTec.py"
echo "Done."
echo ""

# ------------------------------------------------------------------
# Step 3 — Run NRLMSISE-2.1 on GRACE data
# Input:  grace_dns_2009_2016.parquet (in project root)
# Output: grace_dns_with_tnd_y200916_v4_0809.parquet
# ------------------------------------------------------------------
echo "[3/9] Running MSIS on GRACE data..."
cd "$ROOT"
$PYTHON "$ROOT/DataPreparation/run_pymsis.py"
echo "Done."
echo ""

# ------------------------------------------------------------------
# Step 4 — Merge TEC with GRACE (K-D tree spatial matching)
# Input:  grace_dns_with_tnd_y200916_v4_0809.parquet
#         tec_codg_2009-2017_doy1-365.parquet
# Output: grace_data_merged_v3.parquet
# ------------------------------------------------------------------
echo "[4/9] Merging TEC with GRACE..."
cd "$ROOT"
$PYTHON "$ROOT/DataPreparation/MergeTecGrace2.py"
echo "Done."
echo ""

# ------------------------------------------------------------------
# Step 5 — Train XGBoost model
# Input:  grace_data_merged_v3.parquet
# Output: xgb_model_v3.json, scaler_xgboost_X_v3.joblib, scaler_xgboost_y_v3.joblib
# ------------------------------------------------------------------
echo "[5/9] Training XGBoost model..."
cd "$ROOT"
$PYTHON "$ROOT/CoreModel/train.py"
echo "Done."
echo ""

# ------------------------------------------------------------------
# Step 6 — Evaluate model on val/test splits
# Requires: step 5 outputs
# ------------------------------------------------------------------
echo "[6/9] Evaluating model..."
cd "$ROOT"
$PYTHON "$ROOT/CoreModel/evaluate.py"
echo "Done."
echo ""

# ------------------------------------------------------------------
# Step 7 — Rolling warm-start forecast (pre-2009 and post-2016)
# Input:  grace_data_merged2.parquet, xgb_model_v3.json
# Output: runs/ directory with predictions and plots
# ------------------------------------------------------------------
echo "[7/9] Running rolling forecast..."
cd "$ROOT"
$PYTHON "$ROOT/Forecast/Ontrack.py"
echo "Done."
echo ""

# ------------------------------------------------------------------
# Step 8 — Global density map for a snapshot date
# Input:  xgb_model_saved_dr1_post2016_h3_start_2016-02-18.json
#         tec_codg_2009-2017_doy1-365.parquet
# Output: result_df.csv, scaled_swarm.csv, PNG maps
# ------------------------------------------------------------------
echo "[8/9] Generating global density map..."
cd "$ROOT"
$PYTHON "$ROOT/Forecast/off_track.py"
echo "Done."
echo ""

# ------------------------------------------------------------------
# Step 9 — Compare model vs Swarm observations
# Input:  result_df.csv, scaled_swarm.csv (from step 8)
# Output: swarm_vs_model_collocated.csv, diagnostic plots
# ------------------------------------------------------------------
echo "[9/9] Running Swarm validation..."
cd "$ROOT"
$PYTHON "$ROOT/Forecast/swarm_validation.py" \
    --result_df "$ROOT/result_df.csv" \
    --scaled_swarm "$ROOT/scaled_swarm.csv" \
    --out_csv "$ROOT/swarm_vs_model_collocated.csv" \
    --plot
echo "Done."
echo ""

echo "======================================================"
echo " Pipeline complete."
echo "======================================================"
