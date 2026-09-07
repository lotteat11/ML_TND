# CoreModel

Trains, tunes and evaluates the core XGBoost model on the merged dataset from
`DataPreparation`. The target is `log(rho_obs / rho_msis)`; corrected density
is `rho_msis * exp(prediction)`.

```bash
./run_pipeline.sh new train,eval        # or the scripts directly:
python CoreModel/train.py
python CoreModel/evaluate.py
TUNE_TRIALS=200 python CoreModel/tune.py
```

## Configuration — `config.py`

Single source of truth for the feature list, the target and the TEC-lag
settings. `Forecast` imports from here too, so a feature change lands in one
place. The shipped model and scalers were fitted on the default 17-feature
set; changing `TEC_LAGS` or `AP_HISTORY` requires retraining, because the
saved scalers reject a different column set.

| Variable | Default | Effect |
|---|---|---|
| `TEC_LAGS` | `3h` | TEC lag columns, comma-separated |
| `AP_HISTORY` | `1` | `1` adds ap at 3/6/9 h and the 12–33 h and 36–57 h means; `full` adds five |

## Training — `train.py`

Engineers the features, splits the data into cyclic time blocks (16 cycles,
each 2/3 train, 1/6 validation, 1/6 test, with a buffer between blocks),
scales the columns listed in `config.COLS_TO_SCALE`, and trains the booster
with early stopping on the validation blocks. Saves the model and both
scalers.

The cyclic split exists because GRACE altitude decays across the mission; a
chronological split would sort altitude regimes into separate subsets.

| Variable | Default | Effect |
|---|---|---|
| `TRAIN_PARQUET_FILE` | `grace_data_merged_v3.parquet` | Input |
| `TRAIN_TIME_MIN` / `_MAX` | `2009-06-01` / `2016-01-01` | Training window |
| `TRAIN_TIME_EXCLUDE` | unset | Interior holdouts, `"start,end;start,end"` |
| `TRAIN_MODEL_OUT` | `xgb_model_v3.json` | Saved booster |
| `TRAIN_SCALER_X_OUT` / `_Y_OUT` | `scaler_xgboost_{X,y}_v3.joblib` | Saved scalers |
| `USE_TUNED` | `0` | `1` loads `best_params.json` from the tuning directory |

`losses.py` holds the custom objectives and the decaying learning-rate
scheduler used by both training and warm-start.

## Evaluation — `evaluate.py`

Loads the saved model and scalers, predicts on the validation and test
splits, back-transforms to physical density and writes metrics (RMSE,
log-RMSE, MAPE, R², Top-5 %) to CSV next to the model. Diagnostic plots go to
`FIG_DIR`.

| Variable | Default | Effect |
|---|---|---|
| `EVAL_START` / `EVAL_END` | training window | Period scored |
| `EVAL_SAMPLE_STEP` | `10` | Subsampling for the plots |
| `EVAL_METRICS_CSV` | next to the model | Metrics output |
| `FIG_DIR` | `figs` | Figure output |

`check_lst.py` reports local-solar-time, F10.7 and ap coverage per split
(manuscript Table 2).

## Hyperparameter search — `tune.py`

Seeded random search over tree depth, minimum child weight, row and column
subsampling, and the learning-rate schedule (initial rate, decay factor, decay
interval). Every eighth observation is retained to keep it tractable. Writes
`tuning_trials.csv` and `best_params.json` to `TUNE_OUT`.

| Variable | Default | Effect |
|---|---|---|
| `TUNE_TRIALS` | `32` | Number of configurations |
| `TUNE_SEED` | `42` | Random seed |
| `TUNE_OUT` | `tuning_v5` | Output directory |
| `TUNE_N_CYCLES` | `16` | Split cycles |
| `TUNE_MAX_ROUNDS` / `TUNE_ES_ROUNDS` | `3000` / `50` | Round cap and early-stopping patience |

`tec_lag_sensitivity.py` compares TEC-lag feature sets on identical rows,
splits and settings (Appendix B).

## Figures

Each `plot_*.py` writes one figure as PNG and PDF into `figs/`.

| Script | Figure |
|---|---|
| `plot_feature_importance.py` | Gain-based feature importance (Fig. 3) |
| `plot_parity.py` | Observed against modelled density per regime (Fig. 4) |
| `plot_storm_timeseries.py` | Along-track density on one day (Figs. 5, 9) |
| `plot_tuning_summary.py` | Hyperparameter influence and accuracy/overfitting trade-off (Fig. 7) |
| `plot_skill_regimes.py` | Daily log-RMSE over the three holdouts (Fig. 8) |
| `plot_tuning_hyperparams.py` | One panel per searched hyperparameter |
| `plot_regime_overview.py` | Whole-period density, daily skill and ap for one regime |
| `plot_msis_residuals.py` | Distribution of the MSIS log-residual |
| `storm_analysis.py` | Response around a single storm with ap beneath |
