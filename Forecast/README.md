# Forecast

Adaptive updating of the core model and its validation, both along the
GRACE track and on a global grid checked against Swarm.

```bash
./run_pipeline.sh new ontrack           # or directly:
python Forecast/on_track.py
python Forecast/off_track.py
python Forecast/swarm_validation.py --result_df result_df.csv \
                                    --scaled_swarm scaled_swarm.csv --plot
```

## On-track — `on_track.py`

The main experiment. Rolls one day at a time through each holdout period:

1. Fine-tune the working model on the preceding `ONTRACK_LOOKBACK_DAYS` days
   of observations. Those days are sorted, cut into six ~12 h blocks, four
   for training and two for early-stopping validation, with a fixed seed.
2. Specify density along the following day's track with the updated model.
3. Every `ONTRACK_RESET_EVERY` iterations, restore the working model to a
   deep copy of the core model before fine-tuning, so accumulated update
   trees cannot outweigh the long-term structure.

Warm start appends trees to the existing ensemble. The added trees take the
core model's tree shape from `ONTRACK_PARAMS_JSON`; without that they would
fall back to XGBoost defaults and not match the base model.

Each regime × configuration × latency combination is written to its own
directory under `ONTRACK_OUTPUT_ROOT` with predictions, metrics, the model
before and after updating, and the snapshot dated `ONTRACK_SNAPSHOT_*` that
`off_track.py` consumes. `summary_metrics.csv` collects every run.

| Variable | Default | Effect |
|---|---|---|
| `ONTRACK_FILTERS` | `quiet2009,storm2015,post2016` | Regimes to run |
| `ONTRACK_RETRAIN` | `0,1` | `0` static core model, `1` warm start |
| `ONTRACK_HORIZONS` | `1,3` | Update latency in days |
| `ONTRACK_LOOKBACK_DAYS` | `3` | History each update sees |
| `ONTRACK_RESET_EVERY` | `4` | Reset cadence in iterations |
| `ONTRACK_PARAMS_JSON` | `tuning_v13_tec3h_depth3_10/best_params.json` | Tree shape for added trees |
| `ONTRACK_WARMSTART_LR` | `0.005` | Initial learning rate for the update |
| `ONTRACK_WARMSTART_LR_DECAY` / `_LR_STEP` | `0.9` / `20` | Decay factor and interval |
| `ONTRACK_WARMSTART_ROUNDS` / `_PATIENCE` | `2000` / `60` | Round cap and early-stopping patience |
| `ONTRACK_DATA_FILE` | `grace_data_merged_v5_full.parquet` | Input |
| `ONTRACK_OUTPUT_ROOT` | `runs` | Output root |
| `ONTRACK_SNAPSHOT_PRE2009` / `_POST2016` | `2009-01-13` / `2016-02-18` | Model snapshot dates |

## Window and reset sweeps — `tune_lookback.py`

Runs the warm-start scheme for several lookback windows and reset cadences,
scoring every configuration on identical days and observations so the
comparison is not confounded by sampling (Appendix D).

| Variable | Default | Effect |
|---|---|---|
| `LOOKBACKS` | `3,5,7` | Windows swept, at a fixed reset |
| `RESET_INTERVALS` | `4,7,10` | Cadences swept, at `RESET_LOOKBACK` |
| `RESET_LOOKBACK` | `3` | Window used for the reset sweep |
| `EVAL_START` / `EVAL_END` | `2016-01-01` / `2016-03-01` | Period |
| `HORIZON` / `STEP_SIZE` | `1` / `1` | Latency and roll step in days |
| `LOOKBACK_OUTPUT_DIR` | `lookback_sensitivity` | Output |

## Off-track — `off_track.py` and `swarm_validation.py`

`off_track.py` builds a global grid (0.2° latitude, 0.09° longitude) for one
UTC epoch, runs NRLMSIS-2.1 on it, bilinearly interpolates the TEC map onto
it, and applies the saved warm-start snapshot to predict density everywhere.
The epoch, altitude and model file are set in the `Config` block at the top.
Global maps are written as PNG and PDF (manuscript Fig. 6).

`swarm_validation.py` scales Swarm densities to the GRACE altitude with
NRLMSIS-2.1 transfer factors, collocates every Swarm sample within the hour
to the nearest grid cell, and scores prediction and baseline against them:
bias, MAE, RMSE, MAPE, R² and the log-space equivalents. Writes the collocated
CSV and, with `--plot`, residual and latitude-band diagnostics.

## Tables

| Script | Output |
|---|---|
| `make_table_regimes.py` | Per-regime results, core and warm start (Table 3) |
| `make_table_storm.py` | March-2015 main phase, `--days` to choose them, `--daily` per day (Table 4) |
| `lookback_tables_latex.py` | Lookback and reset sweeps as LaTeX (Tables 8, 9) |
