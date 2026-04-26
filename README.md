# Adaptive Thermospheric Density Modeling with TEC Coupling

XGBoost model that corrects NRLMSISE-2.1 density predictions using GRACE satellite observations and ionospheric TEC. The model learns the log-ratio `log(rho_obs / rho_msis)` and applies it as a correction on top of MSIS at inference time.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```
ML_TND/
├── DataPreparation/
│   ├── download_dns.py      # Download GRACE/Swarm DNS from TU Delft FTP
│   ├── download_tec.py      # Download and parse IONEX TEC files from NASA CDDIS
│   ├── run_pymsis.py        # Run NRLMSISE-2.1 on GRACE data, add simulated values
│   ├── run_pymsis_swarm.py  # Run NRLMSISE-2.1 on Swarm data, add simulated values
│   ├── pymsis_utils.py      # Helper functions for MSIS and space weather fetching
│   └── merge_tec_grace.py   # Match TEC to GRACE using K-D tree, save merged parquet
├── CoreModel/
│   ├── config.py            # Shared config: file paths, features, target
│   ├── train.py             # Train XGBoost model, save model + scalers
│   ├── evaluate.py          # Load saved model, run diagnostics and plots
│   └── plotting.py          # Plotting utilities for train/eval
├── Forecast/
│   ├── on_track.py           # Rolling warm-start forecast on out-of-sample data
│   ├── off_track.py         # Global grid prediction + Swarm validation
│   └── swarm_validation.py  # Collocate Swarm obs to model grid, compute metrics
└── feature_functions.py     # Feature engineering, splitting, scaling, shared plots
```

---

## Pipeline — run in this order

### 1. Download satellite density data
```bash
python DataPreparation/download_dns.py
```
Downloads GRACE DNS files from TU Delft FTP. Set `MISSION = "GRACE"`, `YEARS = (2009, 2016)`, and `PARQUET_OUT` at the top of the file. Outputs `GRACE_RAW`.

### 2. Download TEC data
```bash
python DataPreparation/download_tec.py
```
Downloads CODE GIM IONEX files from NASA CDDIS. Requires Earthdata credentials in `~/.netrc`. Outputs `TEC_RAW`.

### 3. Add MSIS density to GRACE data
```bash
python DataPreparation/run_pymsis.py
```
Reads `GRACE_RAW` (Step 1), fetches F10.7 and Ap indices via pymsis, runs NRLMSISE-2.1 for each GRACE point, and saves `GRACE_MSIS`.

### 4. Merge TEC with GRACE
```bash
python DataPreparation/merge_tec_grace.py
```
Spatially matches TEC grid cells to GRACE points using a K-D tree (±3 hour window). Reads `GRACE_MSIS` and `TEC_RAW`. Outputs `GRACE_MERGED`.

### 5. Train the model
```bash
cd CoreModel && python train.py
```
Reads `GRACE_MERGED`, engineers features, splits into train/val/test using cyclic time blocks, and trains XGBoost. Saves `MODEL`, `SCALER_X`, `SCALER_Y`.

### 6. Evaluate
```bash
cd CoreModel && python evaluate.py
```
Loads `MODEL`, `SCALER_X`, `SCALER_Y`, runs on val/test splits, and produces diagnostic plots. Requires step 5 to have been run first.

### 7. Rolling forecast (out-of-sample)
```bash
cd Forecast && python on_track.py
```
Runs on data outside the training window (pre-2009 or post-2016). Fine-tunes the model on the previous 5 days at each step, then predicts 1 or 3 days ahead. Tests 8 combinations (retrain on/off × pre2009/post2016 × horizon 1/3) and writes outputs to `runs/`:

```
runs/
├── summary_metrics.csv                          # cross-run comparison
└── <tag>/                                       # e.g. dr1_post2016_h3
    ├── xgb_model_original_<tag>.json            # model before warm-start
    ├── xgb_model_saved_<tag>_start_<date>.json  # mid-run snapshot (used by Step 8)
    ├── xgb_model_updated_<tag>.json             # fully updated final model
    ├── predictions_<tag>.csv
    └── val_densities_<tag>.png
```

During the rolling forecast, a warm-start model snapshot is saved for **one specific date** controlled by `DATE_TO_SAVE_MODEL` at the top of `on_track.py`. That snapshot captures the model state after it has been fine-tuned on data up to that date, and is the input for Step 8. To use a different date:
1. Set `DATE_TO_SAVE_MODEL` in `on_track.py` and re-run Step 7.
2. Set `model_file` in the `Config` at the top of `off_track.py` to the path of the new snapshot.

### 7b. Add MSIS density to Swarm data (needed for Step 8)
```bash
python DataPreparation/run_pymsis_swarm.py
```
Reads `SWARM_RAW` (download via `download_dns.py` with `MISSION = "Swarm"`, `YEARS = (2015, 2016)`), joins hourly space weather indices, and runs NRLMSISE-2.1 at both satellite altitude and 400 km. Saves `SWARM_MSIS`.

### 8. Global density map
```bash
cd Forecast && python off_track.py
```
Builds a global lat/lon grid for a chosen UTC snapshot, runs MSIS, and applies a warm-start model snapshot to predict density worldwide. Optionally overlays Swarm observations.

### 9. Compare model vs Swarm
```bash
cd Forecast && python swarm_validation.py --result_df result_df.csv --scaled_swarm scaled_swarm.csv --plot
```
Collocates Swarm observations (scaled to GRACE altitude) to the model grid and computes bias, RMSE, MAPE, and log-space metrics for both the prediction and MSIS baseline. Outputs a collocated CSV and diagnostic plots.

---

## Key files needed (not in repo — too large)

All paths are defined as constants in [`paths.py`](paths.py). Actual filenames are listed there.

| Name in `paths.py` | Created by | Used by |
|---|---|---|
| `GRACE_RAW` | Step 1 | Step 3 |
| `TEC_RAW` | Step 2 | Step 4 |
| `SWARM_RAW` | Step 1 (Swarm mission) | Step 7b |
| `GRACE_MSIS` | Step 3 | Step 4 |
| `SWARM_MSIS` | Step 7b | Step 8 |
| `GRACE_MERGED` | Step 4 | Step 5 |
| `MODEL` | Step 5 | Step 6, 7, 8 |
| `SCALER_X` | Step 5 | Step 6, 7, 8 |
| `SCALER_Y` | Step 5 | Step 6, 7, 8 |

---

## Model details

- **Target:** `log(rho_obs / rho_msis)` — correction on top of MSIS
- **Features:** 15 inputs including F10.7, Ap, altitude, latitude, LST/DOY trig encodings, TEC, and TEC lags
- **Split:** cyclic time-block (7 cycles, 2/3 train / 1/6 val / 1/6 test, 1100-obs buffer)
- **Hyperparameters:** `max_depth=4`, `min_child_weight=300`, `subsample=0.5`, `colsample_bytree=0.6`, `num_boost_round=1360`
