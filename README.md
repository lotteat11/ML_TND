# Adaptive machine-learning correction of NRLMSIS-2.1

Code for the study *Adaptive machine-learning correction of NRLMSIS-2.1 using
GRACE thermospheric density observations*.

An XGBoost model corrects NRLMSIS-2.1 thermospheric neutral density using
accelerometer-derived GRACE observations, ionospheric TEC maps, and solar and
geomagnetic indices. The model learns the log-ratio
`log(rho_obs / rho_msis)` — a relative correction to the empirical baseline
rather than an absolute density — and applies it on top of MSIS at inference:

```
rho_pred = rho_msis * exp(log_ratio)
```

A core model is trained on the historical GRACE record. It is then fine-tuned
each day on the preceding three days of observations (warm start), which lets
it track the current thermospheric state and instrument calibration without
retraining from scratch.

---

## Setup

```bash
pip install -r requirements.txt
```

Downloads require an Earthdata login in `~/.netrc` (mode 600):

```
machine urs.earthdata.nasa.gov login USERNAME password PASSWORD
```

---

## Quick start

The whole pipeline runs from one entry point. Stages are selectable, so a
prepared dataset does not have to be rebuilt:

```bash
./run_pipeline.sh new                  # all stages, in order
./run_pipeline.sh new dns,tec          # downloads only (slow, resumable)
./run_pipeline.sh new train,eval       # data already prepared
./run_pipeline.sh storm                # adds the March-2015 storm holdout
```

Modes:

| Mode | Training window | Holdouts |
|---|---|---|
| `new` | 2002–2015, full mission | quiet-2009, post-2016 |
| `storm` | as `new`, March 2015 also excluded | quiet-2009, storm-2015, post-2016 |
| `old` | 2009–2016 | pre-2009, post-2016 |

Stages: `dns, tec, msis, merge, train, eval, tune, ontrack`.

The two setups write separate model, scaler and output files, so they never
overwrite one another. Long stages are wrapped in `caffeinate -is`, which
keeps the machine awake while the lid is open.

---

## Repository structure

```
ML_TND/
├── DataPreparation/          # acquire, model and merge the input data
├── CoreModel/                # train, tune, evaluate and plot the core model
├── Forecast/                 # adaptive updating, on-track and off-track runs
├── config in CoreModel/config.py, paths in paths.py
└── make_agu_figures.py       # assemble the manuscript figure files
```

Each directory has its own README with the run order, inputs and outputs,
and the environment variables for that step:
[DataPreparation](DataPreparation/README.md) ·
[CoreModel](CoreModel/README.md) · [Forecast](Forecast/README.md).

### DataPreparation — building the dataset

| File | What it does |
|---|---|
| `download_dns.py` | Downloads GRACE or Swarm DNS density files from the TU Delft HTTPS service, parses the ASCII format, stacks all years into one parquet. |
| `download_tec.py` | Downloads CODE GIM IONEX files from NASA CDDIS and parses them into epoch/lat/lon/TEC rows. |
| `run_pymsis.py` | Runs NRLMSIS-2.1 along the GRACE track via PyMSIS, adding the `msis_rho` baseline column. |
| `run_pymsis_swarm.py` | The same for Swarm, run both at satellite altitude and at a 400 km reference altitude so densities can be scaled between orbits. |
| `pymsis_utils.py` | Dask helpers: loads density data, fetches and interpolates F10.7 and Ap to hourly cadence, wraps `pymsis.calculate`. |
| `merge_tec_grace.py` | Matches each satellite point to the nearest TEC grid cell with a k-d tree (±3 h window) and writes the merged dataset. |
| `diagnose_tec_missing.py` | Reports where TEC matches are missing and distinguishes absent IONEX epochs from failed spatial matches. |
| `plot_density_altitude.py` | Weekly observed and MSIS density, and the GRACE altitude decay (manuscript Fig. 2). |

### CoreModel — the core model

| File | What it does |
|---|---|
| `config.py` | Single source of truth for the feature list, target, and TEC-lag settings. Both `CoreModel` and `Forecast` import from here, so a feature change lands in one place. |
| `train.py` | Engineers features, applies the cyclic time-block split, trains the XGBoost booster on `log_ratio`, saves the model and both scalers. |
| `evaluate.py` | Loads a saved model, predicts on the validation and test splits, back-transforms to physical density, and writes metrics and diagnostic plots. |
| `tune.py` | Seeded random hyperparameter search over tree shape and the learning-rate schedule. Writes `tuning_trials.csv` and `best_params.json`. |
| `tec_lag_sensitivity.py` | Compares TEC-lag feature sets on identical rows, splits and settings, so candidates differ only in which TEC columns they receive. |
| `check_lst.py` | Reports LST, F10.7 and ap coverage across the train/val/test splits, as full range and P5–P95. |
| `losses.py` | Custom XGBoost objectives and the learning-rate scheduler. |
| `plotting.py` | Shared plotting helpers and the metric computation used by train and evaluate. |
| `storm_analysis.py` | Fine-tunes on the days before a chosen storm and plots the response with the ap index beneath. |

Figure scripts, each writing one manuscript figure as PNG and PDF:

| File | Figure |
|---|---|
| `plot_feature_importance.py` | Gain-based feature importance (Fig. 3) |
| `plot_parity.py` | Observed against modelled density, per regime (Fig. 4) |
| `plot_storm_timeseries.py` | Along-track density on one day (Figs. 5 and 9) |
| `plot_tuning_summary.py` | Hyperparameter influence and the accuracy/overfitting trade-off (Fig. 7) |
| `plot_skill_regimes.py` | Daily RMSE-log over all three holdout periods (Fig. 8) |
| `plot_tuning_hyperparams.py` | One panel per searched hyperparameter |
| `plot_regime_overview.py` | Whole-period overview for one regime: density, daily skill, ap |
| `plot_msis_residuals.py` | Distribution of the MSIS log-residual, the training target |

### Forecast — adaptive updating and validation

| File | What it does |
|---|---|
| `on_track.py` | The main experiment. Rolls day by day through a holdout period: fine-tunes the core model on the preceding `ONTRACK_LOOKBACK_DAYS` days with early stopping, then specifies density along the following day's track. Resets to the core model every `ONTRACK_RESET_EVERY` iterations so accumulated update trees cannot outweigh it. |
| `off_track.py` | Builds a global lat/lon grid for one UTC epoch, runs MSIS on it, applies a saved warm-start model, and optionally overlays Swarm observations (Fig. 6). |
| `swarm_validation.py` | Collocates Swarm observations, scaled to the GRACE altitude, onto the prediction grid and scores prediction and baseline against them. |
| `tune_lookback.py` | Sweeps the fine-tuning window length and the reset cadence, scoring every configuration on identical observations. |
| `make_table_regimes.py` | Per-regime results table (Table 3). |
| `make_table_storm.py` | March-2015 main-phase table (Table 4). |
| `lookback_tables_latex.py` | Lookback and reset sweeps as LaTeX (Tables 8 and 9). |

### Top level

| File | What it does |
|---|---|
| `feature_functions.py` | Feature engineering (LST/DOY trigonometric encodings, TEC lag columns, longitude encoding), the cyclic time-block split, scaling and inverse transforms. |
| `paths.py` | Central registry of large data files and model artifacts. Import these constants rather than hardcoding filenames. |
| `make_agu_figures.py` | Combines multi-panel figures into one PDF per figure, into `agu_figures/`. |

---

## Model details

- **Target:** `log(rho_obs / rho_msis)`, a relative correction to NRLMSIS-2.1.
- **Features (17):** F10.7 and F10.7a; ap at 3, 6 and 9 h lag plus means over
  12–33 h and 36–57 h; latitude; sine/cosine of longitude; sine of local solar
  time and an LST-latitude interaction; sine/cosine of day of year; altitude;
  collocated VTEC and VTEC lagged 3 h.
- **Split:** cyclic time blocks — 16 contiguous cycles, each divided 2/3 train,
  1/6 validation, 1/6 test. GRACE altitude decays across the mission, so a
  plain chronological split would sort altitude regimes into separate subsets.
- **Core hyperparameters:** `max_depth=8`, `min_child_weight=282`,
  `subsample=0.440`, `colsample_bytree=0.610`, initial learning rate `0.0295`
  decaying by `0.815` every 96 rounds, up to 1360 boosting rounds.
- **Warm start:** learning rate `0.005` decaying by `0.9` every 20 rounds, up
  to 2000 added rounds, early stopping after 60 rounds without improvement.

---

## Environment variables

Defaults reproduce the reported results; every variable is optional.

| Variable | Default | Effect |
|---|---|---|
| `TEC_LAGS` | `3h` | TEC lag set. Extra lags need a retrained model and scalers. |
| `AP_HISTORY` | `1` | `1` adds the three storm-history ap features; `full` adds five. |
| `USE_TUNED` | `0` | `1` loads tuned hyperparameters from `best_params.json`. |
| `TRAIN_TIME_EXCLUDE` | unset | Interior holdouts, as `"start,end;start,end"`. |
| `ONTRACK_FILTERS` | `quiet2009,storm2015,post2016` | Which evaluation regimes to run. |
| `ONTRACK_HORIZONS` | `1,3` | Update latency in days. |
| `ONTRACK_LOOKBACK_DAYS` | `3` | Days of history each fine-tuning step sees. |
| `ONTRACK_RESET_EVERY` | `4` | Restore the working model to the core model every N iterations. |
| `ONTRACK_PARAMS_JSON` | `tuning_v13_tec3h_depth3_10/best_params.json` | Tree shape for the trees warm start adds. |
| `MERGE_CHUNKED` | unset | `1` merges year by year; required for the full mission on a 24 GB machine. |
| `TUNE_TRIALS` | `32` | Random-search trial count. |
| `FIG_DIR` | `figs` | Where figures are written. |

Per-stage path overrides (`TRAIN_PARQUET_FILE`, `TRAIN_MODEL_OUT`,
`ONTRACK_OUTPUT_ROOT`, and so on) are listed at the top of each script.

---

## Manuscript figures

```bash
python make_agu_figures.py
```

Writes `agu_figures/figure02.pdf` … `figure09.pdf`, one file per figure with
all panels combined and labelled. Panels keep their vector content. Figure 1
is a TikZ flowchart in the manuscript source and has no file here.

---

## Data availability

The processed data files are too large for the repository and are archived at
**https://doi.org/10.6084/m9.figshare.32241219**.

`paths.py` names every file. The dependency chain is:

| Constant | Written by | Read by |
|---|---|---|
| `GRACE_RAW` | `download_dns.py` | `run_pymsis.py` |
| `TEC_RAW` | `download_tec.py` | `merge_tec_grace.py` |
| `SWARM_RAW` | `download_dns.py` (Swarm) | `run_pymsis_swarm.py` |
| `GRACE_MSIS` | `run_pymsis.py` | `merge_tec_grace.py` |
| `SWARM_MSIS` | `run_pymsis_swarm.py` | `off_track.py` |
| `GRACE_MERGED` | `merge_tec_grace.py` | `train.py` |
| `MODEL`, `SCALER_X`, `SCALER_Y` | `train.py` | `evaluate.py`, `on_track.py`, `off_track.py` |
