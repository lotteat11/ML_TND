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
│   ├── GettingData.py       # Download GRACE/Swarm DNS from TU Delft FTP
│   ├── ImportTec.py         # Download and parse IONEX TEC files from NASA CDDIS
│   ├── run_pymsis.py        # Run NRLMSISE-2.1 on GRACE data, add msis_rho column
│   ├── pymsis_utils.py      # Helper functions for MSIS and space weather fetching
│   └── MergeTecGrace2.py    # Match TEC to GRACE using K-D tree, save merged parquet
├── CoreModel/
│   ├── config.py            # Shared config: file paths, features, target
│   ├── train.py             # Train XGBoost model, save model + scalers
│   ├── evaluate.py          # Load saved model, run diagnostics and plots
│   └── plotting.py          # Plotting utilities for train/eval
├── Forecast/
│   ├── Ontrack.py           # Rolling warm-start forecast on out-of-sample data
│   ├── off_track.py         # Global grid prediction + Swarm validation
│   └── swarm_validation.py  # Collocate Swarm obs to model grid, compute metrics
└── Feature_functions.py     # Feature engineering, splitting, scaling, shared plots
```

---

## Pipeline — run in this order

### 1. Download satellite density data
```bash
python DataPreparation/GettingData.py
```
Downloads GRACE or Swarm DNS files from TU Delft FTP. Configure `MISSION`, `YEARS`, and output paths at the top of the file.

### 2. Download TEC data
```bash
python DataPreparation/ImportTec.py
```
Downloads CODE GIM IONEX files from NASA CDDIS. Requires Earthdata credentials in `~/.netrc`. Outputs `tec_codg_2009-2017_doy1-365_v2.parquet`.

### 3. Add MSIS density to GRACE data
```bash
python DataPreparation/run_pymsis.py
```
Fetches F10.7 and Ap indices via pymsis, runs NRLMSISE-2.1 for each GRACE point, and saves `grace_dns_with_tnd_y200916_v4_0809.parquet`.

### 4. Merge TEC with GRACE
```bash
python DataPreparation/MergeTecGrace2.py
```
Spatially matches TEC grid cells to GRACE points using a K-D tree (±3 hour window). Outputs `grace_data_merged_v3.parquet`.

### 5. Train the model
```bash
cd CoreModel && python train.py
```
Loads `grace_data_merged_v3.parquet`, engineers features, splits into train/val/test using cyclic time blocks, and trains XGBoost. Saves `xgb_model_v3.json`, `scaler_xgboost_X_v3.joblib`, `scaler_xgboost_y_v3.joblib`.

### 6. Evaluate
```bash
cd CoreModel && python evaluate.py
```
Loads the saved model and scalers, runs on val/test splits, and produces diagnostic plots. Requires step 5 to have been run first.

### 7. Rolling forecast (out-of-sample)
```bash
cd Forecast && python Ontrack.py
```
Runs on data outside the training window (pre-2009 or post-2016). Fine-tunes the model on the previous 5 days at each step, then predicts 1 or 3 days ahead. Tests 8 combinations and saves results to `runs/`.

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

| File | Created by |
|---|---|
| `grace_dns_with_tnd_y200916_v4_0809.parquet` | Step 3 |
| `tec_codg_2009-2017_doy1-365_v2.parquet` | Step 2 |
| `grace_data_merged_v3.parquet` | Step 4 |
| `xgb_model_v3.json` | Step 5 |
| `scaler_xgboost_X_v3.joblib` | Step 5 |
| `scaler_xgboost_y_v3.joblib` | Step 5 |

---

## Model details

- **Target:** `log(rho_obs / rho_msis)` — correction on top of MSIS
- **Features:** 15 inputs including F10.7, Ap, altitude, latitude, LST/DOY trig encodings, TEC, and TEC lags
- **Split:** cyclic time-block (8 cycles, 2/3 train / 1/6 val / 1/6 test, 1100-obs buffer)
- **Hyperparameters:** `max_depth=4`, `min_child_weight=300`, `subsample=0.5`, `colsample_bytree=0.6`, `num_boost_round=1360`
