# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
tune_lookback.py
- Runs the rolling warm-start forecast for lookback windows of 3, 5 and 7 days
  and reports how the warm-start skill depends on how much recent history each
  fine-tuning step sees.
- Also runs a matched reset-cadence sensitivity test with a fixed three-day
  lookback and reset intervals of 4, 7 and 10 daily iterations.
- Everything except the lookback is held at the CURRENT production setup, so the
  three runs differ in exactly one knob:
    * model/scalers   xgb_model_v8_storm_ap_2002train (17 features: single 3 h
                      TEC lag + the AP_HISTORY storm-history drivers)
    * warm-start tree params  tuning_v13_tec3h_depth3_10/best_params.json
    * LR schedule / patience / reset cadence  same defaults as Forecast/on_track.py
- Evaluation window defaults to 2002-2003 (solar maximum; GRACE data starts
  2002-04-04 and has a Jun-Jul 2002 gap). This period is inside the training
  window of the 2002-train model, so the numbers below compare lookbacks
  against each other rather than measuring out-of-sample skill.

Usage:
    ven_2404/bin/python Forecast/tune_lookback.py
    LOOKBACKS=3,5,7,14 EVAL_END=2003-01-01 ven_2404/bin/python Forecast/tune_lookback.py
    RESET_INTERVALS=4,7,10 RESET_LOOKBACK=3 ven_2404/bin/python Forecast/tune_lookback.py
"""

import os
import sys
import tempfile
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "CoreModel"))

import copy
import gc
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

import feature_functions as ff
from config import FEATURES, COLS_TO_SCALE, TEC_LAGS, TEC_LAG_COLS

# ---------------------------------------------------------------------------
# CONFIG — every value is overridable from the environment
# ---------------------------------------------------------------------------
LOOKBACKS = [int(v) for v in
             os.environ.get("LOOKBACKS", "3,5,7").split(",") if v.strip()]
if not LOOKBACKS or min(LOOKBACKS) < 1:
    raise ValueError("LOOKBACKS must be a comma-separated list of days >= 1")

HORIZON     = int(os.environ.get("HORIZON", "1"))       # forecast horizon in days
EVAL_START  = os.environ.get("EVAL_START", "2002-01-01")
EVAL_END    = os.environ.get("EVAL_END",   "2004-01-01")  # exclusive
STEP_SIZE   = int(os.environ.get("STEP_SIZE", "1"))       # rolling step in days

# Same warm-start knobs (and defaults) as Forecast/on_track.py, so a lookback
# picked here transfers straight into the production rolling runs.
RESET_EVERY        = int(os.environ.get("ONTRACK_RESET_EVERY", "4"))
RESET_INTERVALS = [int(v) for v in
                   os.environ.get("RESET_INTERVALS", "4,7,10").split(",")
                   if v.strip()]
RESET_LOOKBACK = int(os.environ.get("RESET_LOOKBACK", "3"))
if not RESET_INTERVALS or min(RESET_INTERVALS) < 1:
    raise ValueError(
        "RESET_INTERVALS must be a comma-separated list of iterations >= 1"
    )
if RESET_LOOKBACK < 1:
    raise ValueError("RESET_LOOKBACK must be >= 1")
WARMSTART_LR       = float(os.environ.get("ONTRACK_WARMSTART_LR", "0.005"))
WARMSTART_LR_DECAY = float(os.environ.get("ONTRACK_WARMSTART_LR_DECAY", "0.9"))
WARMSTART_LR_STEP  = int(os.environ.get("ONTRACK_WARMSTART_LR_STEP", "20"))
WARMSTART_ROUNDS   = int(os.environ.get("ONTRACK_WARMSTART_ROUNDS", "2000"))
WARMSTART_PATIENCE = int(os.environ.get("ONTRACK_WARMSTART_PATIENCE", "60"))

# Current model/scalers/data — same defaults the storm pipeline mode exports.
MODEL_FILE    = os.environ.get("ONTRACK_MODEL_FILE",
                               "xgb_model_v8_storm_ap_2002train.json")
SCALER_X_FILE = os.environ.get("ONTRACK_SCALER_X_FILE",
                               "scaler_xgboost_X_v8_storm_ap_2002train.joblib")
SCALER_Y_FILE = os.environ.get("ONTRACK_SCALER_Y_FILE",
                               "scaler_xgboost_y_v8_storm_ap_2002train.joblib")
DATA_FILE     = os.environ.get("ONTRACK_DATA_FILE",
                               "grace_data_merged_v5_full.parquet")

OUTPUT_DIR = os.environ.get("LOOKBACK_OUTPUT_DIR", "lookback_sensitivity")

# Tuned tree shape for the trees warm-start ADDS. Without this the new trees
# fall back to XGBoost load defaults and no longer match the base model, which
# would confound the lookback comparison.
PARAMS_JSON = os.environ.get("ONTRACK_PARAMS_JSON",
                             "tuning_v13_tec3h_depth3_10/best_params.json")
TREE_PARAMS = None
if PARAMS_JSON and os.path.exists(PARAMS_JSON):
    with open(PARAMS_JSON) as fh:
        _tuned = json.load(fh)
    TREE_PARAMS = {
        "max_depth": int(_tuned["max_depth"]),
        "min_child_weight": float(_tuned["min_child_weight"]),
        "subsample": float(_tuned["subsample"]),
        "colsample_bytree": float(_tuned["colsample_bytree"]),
    }
elif PARAMS_JSON:
    raise FileNotFoundError(f"ONTRACK_PARAMS_JSON not found: {PARAMS_JSON}")

TARGET_COL = "log_ratio"
# Imported from CoreModel/config.py so training and inference cannot drift.
cols_to_scale   = list(COLS_TO_SCALE)
columns_to_keep = list(FEATURES)


# ---------------------------------------------------------------------------
# LR SCHEDULER — identical to on_track.py's
# ---------------------------------------------------------------------------
def lr_scheduler(current_round: int) -> float:
    initial_lr = 1e-7 if current_round < 4 else WARMSTART_LR
    return initial_lr * (WARMSTART_LR_DECAY ** (current_round // WARMSTART_LR_STEP))


# ---------------------------------------------------------------------------
# FINE-TUNE — mirrors on_track.update_xgb_model_aggressive_with_callbacks
# ---------------------------------------------------------------------------
def fine_tune(booster, train_data, scaler_X, scaler_y):
    train_data = train_data.sort_values(by=["date", "time"]).reset_index(drop=True)

    K = 6
    block_size = len(train_data) // K if len(train_data) >= K else 1
    np.random.seed(42)
    train_idx = np.random.choice(np.arange(K), size=int(K * 2 / 3), replace=False)

    train_chunks, val_chunks = [], []
    for i in range(K):
        start = i * block_size
        end = len(train_data) if i == K - 1 else start + block_size
        block = train_data.iloc[start:end]
        (train_chunks if i in train_idx else val_chunks).append(block)

    tr = pd.concat(train_chunks).sort_values(by=["date", "time"])
    vl = pd.concat(val_chunks).sort_values(by=["date", "time"])

    def prep(chunk):
        X_s = pd.DataFrame(
            scaler_X.transform(chunk[cols_to_scale]),
            columns=cols_to_scale, index=chunk.index,
        )
        X_u = chunk[[c for c in columns_to_keep if c not in cols_to_scale]]
        return pd.concat([X_s, X_u], axis=1)[booster.feature_names]

    dtrain = xgb.DMatrix(prep(tr),
                         label=scaler_y.transform(tr[TARGET_COL].values.reshape(-1, 1)).ravel())
    dval = xgb.DMatrix(prep(vl),
                       label=scaler_y.transform(vl[TARGET_COL].values.reshape(-1, 1)).ravel())

    params = {"objective": "reg:squarederror", "eval_metric": "rmse"}
    if TREE_PARAMS:
        params.update(TREE_PARAMS)

    updated = xgb.train(
        params, dtrain, num_boost_round=WARMSTART_ROUNDS,
        evals=[(dtrain, "train"), (dval, "val")],
        xgb_model=booster,
        callbacks=[
            xgb.callback.EarlyStopping(rounds=WARMSTART_PATIENCE, save_best=True),
            xgb.callback.LearningRateScheduler(lr_scheduler),
        ],
        verbose_eval=False,
    )

    # Round-trip through a temp file so the next step starts from the BEST
    # checkpoint rather than the last (and so no fixed workspace file is left).
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        updated.save_model(tmp_path)
        best = xgb.Booster()
        best.load_model(tmp_path)
    finally:
        os.unlink(tmp_path)
    return best


# ---------------------------------------------------------------------------
# ROLLING FORECAST FOR ONE LOOKBACK VALUE
# ---------------------------------------------------------------------------
def run_lookback(df_feat, original_booster, scaler_X, scaler_y, lookback: int,
                 start_idx: int | None = None,
                 reset_every: int = RESET_EVERY) -> pd.DataFrame:
    unique_dates = df_feat["date"].drop_duplicates().sort_values().tolist()

    base = copy.deepcopy(original_booster)
    all_preds = []
    step = 0

    # Every lookback must forecast the SAME days, or the comparison confounds
    # the lookback with the period scored: a longer lookback consumes more
    # lead-in, so left to itself it would start later and skip the hardest
    # early days. start_idx is therefore gated by the LONGEST lookback in the
    # sweep (max(LOOKBACKS) + 1), not by this run's own.
    if start_idx is None:
        start_idx = lookback + 1
    if start_idx < lookback:
        raise ValueError(
            f"start_idx={start_idx} leaves less than {lookback} days of history"
        )
    for i in range(start_idx, len(unique_dates) - HORIZON + 1, STEP_SIZE):
        step += 1
        if step % reset_every == 0:
            base = copy.deepcopy(original_booster)

        prev_days  = unique_dates[i - lookback: i]
        train_data = df_feat[df_feat["date"].isin(prev_days)].copy()
        base = fine_tune(base, train_data, scaler_X, scaler_y)
        del train_data

        window_dates = unique_dates[i: i + HORIZON]
        window_data  = df_feat[df_feat["date"].isin(window_dates)].copy()

        X_s = pd.DataFrame(
            scaler_X.transform(window_data[cols_to_scale]),
            columns=cols_to_scale, index=window_data.index,
        )
        X_u = window_data[[c for c in columns_to_keep if c not in cols_to_scale]]
        X   = pd.concat([X_s, X_u], axis=1)[base.feature_names]

        pred_s = base.predict(xgb.DMatrix(X))
        pred   = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()

        window_data["y_pred_log"] = pred
        window_data["rho_pred"]   = window_data["msis_rho"] * np.exp(pred)
        all_preds.append(
            window_data[["date", "time", "rho_obs", "msis_rho", "rho_pred"]]
        )
        del window_data, X_s, X_u, X
        if step % 25 == 0:
            print(f"    step {step}: through {unique_dates[i].date()}", flush=True)
            gc.collect()

    if not all_preds:
        raise RuntimeError(
            f"No forecast windows for lookback={lookback}. The evaluation "
            f"window holds only {len(unique_dates)} days; it needs more than "
            f"lookback + horizon = {lookback + HORIZON}."
        )
    return pd.concat(all_preds).reset_index(drop=True)


# ---------------------------------------------------------------------------
# METRICS — same definitions as on_track.compute_metrics
# ---------------------------------------------------------------------------
def metrics(df: pd.DataFrame, pred_col: str = "rho_pred") -> dict:
    y, yhat = df["rho_obs"].values, df[pred_col].values
    err = yhat - y
    abs_err = np.abs(err)

    thr = np.quantile(abs_err, 0.95)
    mask = y != 0
    lmask = (y > 0) & (yhat > 0)
    lerr = np.log(yhat[lmask]) - np.log(y[lmask])
    lthr = np.quantile(np.abs(lerr), 0.95)
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    return {
        "n": len(df),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(abs_err)),
        "top5": float(np.sqrt(np.mean(err[abs_err >= thr] ** 2))),
        "mape_pct": float(np.mean(abs_err[mask] / np.abs(y[mask])) * 100),
        "bias": float(np.mean(err)),
        "r2": 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else np.nan,
        "rmse_log": float(np.sqrt(np.mean(lerr ** 2))),
        "top5_log": float(np.sqrt(np.mean(lerr[np.abs(lerr) >= lthr] ** 2))),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Model    : {MODEL_FILE}")
    print(f"Scalers  : {SCALER_X_FILE} / {SCALER_Y_FILE}")
    print(f"Data     : {DATA_FILE}")
    print(f"Params   : {PARAMS_JSON} -> {TREE_PARAMS}")
    print(f"TEC lags : {TEC_LAGS} -> {TEC_LAG_COLS}")
    print(f"Features : {len(columns_to_keep)} -> {columns_to_keep}")
    print(f"Window   : {EVAL_START} .. {EVAL_END} (exclusive), horizon={HORIZON}d, "
          f"reset every {RESET_EVERY} steps")
    print(f"Lookbacks: {LOOKBACKS}\n")
    print(f"Reset sensitivity: lookback={RESET_LOOKBACK} d, "
          f"intervals={RESET_INTERVALS}\n")

    # Push the date cut into Parquet so the 82.5M-row full-mission file is
    # never materialized in full.
    df = pd.read_parquet(
        DATA_FILE,
        filters=[
            ("grace_time", ">=", pd.Timestamp(EVAL_START, tz="UTC")),
            ("grace_time", "<",  pd.Timestamp(EVAL_END, tz="UTC")),
        ],
    )
    print(f"Loaded {len(df):,} rows "
          f"({df['grace_time'].min()} .. {df['grace_time'].max()})")

    # Feature engineering — same order as on_track.py
    df["time"] = df["grace_time"]
    df = ff.add_lst_doy_features(df, copy=False)
    df["lon_sin"]     = np.sin(np.deg2rad(df["lon"]))
    df["lon_cos"]     = np.cos(np.deg2rad(df["lon"]))
    df["lst_lat_sin"] = df["lst_sin"] * df["lat"]
    df = ff.add_tec_time_lag_features(df, time_col="time",
                                      lags=TEC_LAGS, names=TEC_LAG_COLS)
    df[TARGET_COL] = np.log(df["rho_obs"] / df["msis_rho"])
    df.dropna(subset=sorted(set(columns_to_keep) | {TARGET_COL}), inplace=True)

    needed = set(columns_to_keep) | set(cols_to_scale) | {
        "date", "time", TARGET_COL, "msis_rho", "rho_obs"}
    df["date"] = pd.to_datetime(df["time"], utc=True).dt.normalize()
    for c in [c for c in df.columns if c not in needed]:
        del df[c]
    gc.collect()
    print(f"After feature engineering: {len(df):,} rows, "
          f"{df['date'].nunique()} distinct days\n")

    scaler_X = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)

    base_model = xgb.Booster()
    base_model.load_model(MODEL_FILE)
    # The base model, the scalers and config.py must agree on the feature set.
    # A TEC_LAGS or AP_HISTORY override that does not match the loaded model
    # otherwise fails deep inside the first scaler.transform, several minutes
    # into the run. Fail here instead, naming both sides.
    missing = [f for f in base_model.feature_names if f not in columns_to_keep]
    extra   = [f for f in columns_to_keep if f not in base_model.feature_names]
    scaler_feats = list(getattr(scaler_X, "feature_names_in_", cols_to_scale))
    scale_mismatch = sorted(set(cols_to_scale) ^ set(scaler_feats))
    if missing or extra or scale_mismatch:
        raise ValueError(
            f"Feature set mismatch — check TEC_LAGS / AP_HISTORY.\n"
            f"  model {MODEL_FILE}: {len(base_model.feature_names)} features\n"
            f"  config.py         : {len(columns_to_keep)} features "
            f"(TEC_LAGS={TEC_LAGS})\n"
            f"  model wants, config does not build: {missing}\n"
            f"  config builds, model does not want: {extra}\n"
            f"  scaler/config disagree on scaled cols: {scale_mismatch}\n"
            f"  Hint: the v8_storm_ap model needs TEC_LAGS=3h."
        )

    # Gate every lookback on the longest one so all of them forecast exactly
    # the same days. Without this each run starts at its own lookback+1 and
    # scores a different (and for short lookbacks, longer and earlier) period,
    # which confounds the lookback with the days evaluated.
    shared_start = max(LOOKBACKS) + 1
    print(f"Shared start index: {shared_start} "
          f"(gated by longest lookback = {max(LOOKBACKS)} d) — "
          f"all lookbacks forecast identical days\n")

    results = {}
    for lb in LOOKBACKS:
        print(f"--- Lookback = {lb} days ---", flush=True)
        pred_df = run_lookback(df, base_model, scaler_X, scaler_y, lb,
                               start_idx=shared_start)
        m = metrics(pred_df)
        m_msis = metrics(pred_df, pred_col="msis_rho")
        results[lb] = {"metrics": m, "msis": m_msis}
        pred_df.to_pickle(os.path.join(OUTPUT_DIR, f"predictions_lb{lb}.pkl"))
        print(f"  n={m['n']:,}  RMSE={m['rmse']:.4e}  MAPE={m['mape_pct']:.2f}%  "
              f"bias={m['bias']:.4e}  log-RMSE={m['rmse_log']:.4f}  "
              f"top5={m['top5']:.4e}  R2={m['r2']:.4f}")
        print(f"  MSIS baseline: RMSE={m_msis['rmse']:.4e}  "
              f"MAPE={m_msis['mape_pct']:.2f}%  log-RMSE={m_msis['rmse_log']:.4f}\n",
              flush=True)
        del pred_df
        gc.collect()

    # The whole point of shared_start is that every lookback scored the same
    # rows. Verify it rather than trust it: an unequal n means the comparison
    # is confounded and the table below would be misleading.
    counts = {lb: results[lb]["metrics"]["n"] for lb in LOOKBACKS}
    if len(set(counts.values())) != 1:
        raise RuntimeError(
            f"Lookbacks scored different row counts {counts} — the comparison "
            f"is not matched. Expected identical n from shared_start="
            f"{shared_start}."
        )
    print(f"✓ all lookbacks scored the same {next(iter(counts.values())):,} rows\n")

    # Summary table + CSV
    summary = pd.DataFrame(
        [{"lookback_days": lb, **results[lb]["metrics"]} for lb in LOOKBACKS]
    )
    summary_csv = os.path.join(OUTPUT_DIR, "lookback_summary.csv")
    summary.to_csv(summary_csv, index=False)

    print("=== Summary ===")
    print(f"{'Lookback':>9} {'n':>12} {'RMSE':>12} {'MAPE (%)':>9} "
          f"{'Bias':>12} {'log-RMSE':>9} {'Top5':>12} {'R2':>8}")
    for lb in LOOKBACKS:
        m = results[lb]["metrics"]
        print(f"{lb:>9} {m['n']:>12,} {m['rmse']:>12.4e} {m['mape_pct']:>9.2f} "
              f"{m['bias']:>12.4e} {m['rmse_log']:>9.4f} {m['top5']:>12.4e} "
              f"{m['r2']:>8.4f}")
    m0 = results[LOOKBACKS[0]]["msis"]
    print(f"{'MSIS':>9} {m0['n']:>12,} {m0['rmse']:>12.4e} {m0['mape_pct']:>9.2f} "
          f"{m0['bias']:>12.4e} {m0['rmse_log']:>9.4f} {m0['top5']:>12.4e} "
          f"{m0['r2']:>8.4f}")

    # Bar chart
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    keys = [("rmse", "RMSE [kg/m³]"), ("mape_pct", "MAPE (%)"),
            ("rmse_log", "log-RMSE"), ("top5", "Top-5% error [kg/m³]")]
    for ax, (key, label) in zip(axes, keys):
        vals = [results[lb]["metrics"][key] for lb in LOOKBACKS]
        bars = ax.bar([str(lb) for lb in LOOKBACKS], vals,
                      color=["steelblue", "darkorange", "seagreen",
                             "indianred", "slateblue"][:len(LOOKBACKS)])
        ax.bar_label(bars, fmt="%.4g", padding=3)
        ax.set_xlabel("Lookback (days)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.margins(y=0.15)
    fig.suptitle(f"Lookback sensitivity — {EVAL_START} to {EVAL_END}, "
                 f"horizon={HORIZON}d, reset every {RESET_EVERY} steps")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"lookback_sensitivity.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # RESET-CADENCE SENSITIVITY
    # -----------------------------------------------------------------------
    # Use the same shared start as the lookback sweep. This guarantees that
    # reset=4/7/10 are evaluated on exactly the same forecast dates and rows as
    # the lookback comparison above. The lookback is fixed at three days so
    # only the reset cadence changes.
    if shared_start < RESET_LOOKBACK:
        raise ValueError(
            f"shared_start={shared_start} leaves less than RESET_LOOKBACK="
            f"{RESET_LOOKBACK} days of history"
        )

    reset_results = {}
    for reset_interval in RESET_INTERVALS:
        print(f"--- Reset every {reset_interval} iterations "
              f"(lookback={RESET_LOOKBACK} days) ---", flush=True)

        # The default lookback run already computed this exact combination;
        # reuse its metrics and prediction file rather than repeat an expensive
        # rolling warm-start run.
        if reset_interval == RESET_EVERY and RESET_LOOKBACK in results:
            reset_results[reset_interval] = results[RESET_LOOKBACK]
            source = os.path.join(
                OUTPUT_DIR, f"predictions_lb{RESET_LOOKBACK}.pkl"
            )
            print(f"  reused matched lookback run from {source}")
        else:
            pred_df = run_lookback(
                df, base_model, scaler_X, scaler_y, RESET_LOOKBACK,
                start_idx=shared_start, reset_every=reset_interval,
            )
            m = metrics(pred_df)
            m_msis = metrics(pred_df, pred_col="msis_rho")
            reset_results[reset_interval] = {"metrics": m, "msis": m_msis}
            pred_df.to_pickle(os.path.join(
                OUTPUT_DIR,
                f"predictions_lb{RESET_LOOKBACK}_reset{reset_interval}.pkl",
            ))
            del pred_df
            gc.collect()

        m = reset_results[reset_interval]["metrics"]
        print(f"  n={m['n']:,}  RMSE={m['rmse']:.4e}  "
              f"MAPE={m['mape_pct']:.2f}%  bias={m['bias']:.4e}  "
              f"log-RMSE={m['rmse_log']:.4f}  top5={m['top5']:.4e}  "
              f"R2={m['r2']:.4f}\n", flush=True)

    reset_counts = {
        interval: result["metrics"]["n"]
        for interval, result in reset_results.items()
    }
    if len(set(reset_counts.values())) != 1:
        raise RuntimeError(
            f"Reset intervals scored different row counts {reset_counts}; "
            "the comparison is not matched."
        )
    print(f"✓ all reset intervals scored the same "
          f"{next(iter(reset_counts.values())):,} rows\n")

    reset_summary = pd.DataFrame([
        {
            "lookback_days": RESET_LOOKBACK,
            "reset_every_iterations": interval,
            **reset_results[interval]["metrics"],
        }
        for interval in RESET_INTERVALS
    ])
    reset_summary_csv = os.path.join(OUTPUT_DIR, "reset_summary.csv")
    reset_summary.to_csv(reset_summary_csv, index=False)

    print("=== Reset sensitivity (fixed lookback = "
          f"{RESET_LOOKBACK} days) ===")
    print(f"{'Reset':>9} {'n':>12} {'RMSE':>12} {'MAPE (%)':>9} "
          f"{'Bias':>12} {'log-RMSE':>9} {'Top5':>12} {'R2':>8}")
    for interval in RESET_INTERVALS:
        m = reset_results[interval]["metrics"]
        print(f"{interval:>9} {m['n']:>12,} {m['rmse']:>12.4e} "
              f"{m['mape_pct']:>9.2f} {m['bias']:>12.4e} "
              f"{m['rmse_log']:>9.4f} {m['top5']:>12.4e} "
              f"{m['r2']:>8.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (key, label) in zip(axes, keys):
        vals = [reset_results[r]["metrics"][key] for r in RESET_INTERVALS]
        bars = ax.bar(
            [str(r) for r in RESET_INTERVALS], vals,
            color=["steelblue", "darkorange", "seagreen",
                   "indianred", "slateblue"][:len(RESET_INTERVALS)],
        )
        ax.bar_label(bars, fmt="%.4g", padding=3)
        ax.set_xlabel("Reset interval (daily iterations)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.margins(y=0.15)
    fig.suptitle(
        f"Reset sensitivity — lookback={RESET_LOOKBACK}d, "
        f"{EVAL_START} to {EVAL_END}, horizon={HORIZON}d"
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"reset_sensitivity.{ext}"),
            dpi=150, bbox_inches="tight",
        )
    plt.close(fig)

    print(f"\nSaved → {summary_csv}")
    print(f"Saved → {os.path.join(OUTPUT_DIR, 'lookback_sensitivity.png')} (+ .pdf)")
    print(f"Saved → {reset_summary_csv}")
    print(f"Saved → {os.path.join(OUTPUT_DIR, 'reset_sensitivity.png')} (+ .pdf)")
