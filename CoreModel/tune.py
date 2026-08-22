# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
tune.py — documented random hyperparameter search for the core XGBoost model.

The search lives in `random_search(...)` and takes all settings as function
arguments, so it can be imported and driven directly:

    from tune import random_search
    trials, best = random_search(n_trials=16, stride=8, out_dir="tuning_test")

- Data, features, split and scaling are IDENTICAL to train.py (same
  load_and_engineer, same 16-cycle time-block split, same scalers).
  Selection RMSE is computed on the pooled early-stopping partition — i.e.
  8 temporally separated blocks spread across the whole training window —
  while the remaining partition stays untouched for final reporting.
- Trial 0 is always the published configuration, so the search doubles as a
  stability check of the existing choice.
- The search runs on a row-stride subsample (default every 4th row) to keep
  each trial to minutes; re-run top configurations with train.py (full data)
  before adopting one.

Outputs (in out_dir):
    tuning_trials.csv       one row per trial (params, sel-RMSE, rounds, time)
    tuning_stability.png    sorted trial scores + published config marker
    tuning_sensitivity.png  sel-RMSE vs each hyperparameter

Pipeline entry point (env-var driven):   ./run_pipeline.sh new tune
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.callback import EarlyStopping

import feature_functions as ff
from config import PARQUET_FILE, TARGET, FEATURES, COLS_TO_SCALE
from train import load_and_engineer

# Published configuration (v3 paper) — always evaluated as trial 0.
# ALL trials train with a decaying learning-rate schedule of the same FORM as
# the published one (initial_lr * decay^(round // step)); the search varies
# the initial rate AND the decay factor / step size. Decay near 1.0 with a
# large step approximates a constant learning rate, so that regime is
# included in the space too.
BASELINE = {
    "max_depth": 4, "min_child_weight": 300, "subsample": 0.5,
    "colsample_bytree": 0.6, "learning_rate": 5e-4,
    "lr_decay_factor": 0.8, "lr_step_size": 15,
}

# Schedule keys are consumed by make_lr_scheduler, not passed to XGBoost.
SCHEDULE_KEYS = ("learning_rate", "lr_decay_factor", "lr_step_size")


def make_lr_scheduler(initial_lr: float, decay_factor: float = 0.8,
                      step_size: int = 15):
    """Decay schedule of the published form with configurable parameters."""
    def sched(current_round: int) -> float:
        return initial_lr * (decay_factor ** (current_round // step_size))
    return sched

# Default search space: name -> callable(rng) drawing one value.
DEFAULT_SPACE = {
    "max_depth":        lambda rng: int(rng.integers(3, 11)),             # 3 .. 10
    "min_child_weight": lambda rng: float(10 ** rng.uniform(1, 3)),      # 10 .. 1000
    "subsample":        lambda rng: float(rng.uniform(0.4, 0.9)),
    "colsample_bytree": lambda rng: float(rng.uniform(0.4, 1.0)),
    "learning_rate":    lambda rng: float(10 ** rng.uniform(-2.5, -1)),  # 0.003 .. 0.1
    "lr_decay_factor":  lambda rng: float(rng.uniform(0.7, 0.995)),      # 0.995 ~ near-constant
    "lr_step_size":     lambda rng: int(rng.integers(10, 101)),
}


def random_search(parquet_file=None,
                  n_trials: int = 32,
                  stride: int = 4,
                  seed: int = 42,
                  out_dir: str = "tuning_v5",
                  space: dict | None = None,
                  baseline: dict | None = None,
                  max_rounds: int = 3000,
                  es_rounds: int = 50,
                  n_cycles: int = 16,
                  df_feat: pd.DataFrame | None = None,
                  make_plots: bool = True):
    """
    Seeded random search over XGBoost hyperparameters.

    Args:
        parquet_file: merged dataset; defaults to config.PARQUET_FILE.
        n_trials:     number of trials incl. the baseline as trial 0.
        stride:       row-stride subsample (1 = full data).
        seed:         RNG seed (drives the whole search; rerun = same trials).
        out_dir:      where CSV/plots are written.
        space:        dict name -> callable(rng); defaults to DEFAULT_SPACE.
        baseline:     configuration evaluated as trial 0 (default: published).
        max_rounds / es_rounds: boosting budget and early-stopping patience.
        n_cycles:     cycles in the time-block split (must match train.py).
        df_feat:      pre-engineered dataframe — pass to skip the (slow)
                      load_and_engineer step when calling repeatedly.
        make_plots:   write the stability/sensitivity figures.

    Returns:
        (trials_df, best_params): all trials sorted by selection RMSE, and
        the best configuration as a dict.
    """
    space = space or DEFAULT_SPACE
    baseline = baseline or BASELINE
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    if df_feat is None:
        print("Loading + engineering (identical to train.py)...")
        df_feat = load_and_engineer(parquet_file or PARQUET_FILE)
    if stride > 1:
        df_feat = df_feat.iloc[::stride]
        print(f"Row-stride subsample 1/{stride}: {len(df_feat):,} rows")

    X = df_feat[FEATURES]
    y = df_feat[[TARGET]]

    # Same 16-cycle block split as train.py; the gap (rows) shrinks with the
    # stride so it covers the same amount of TIME as in full-data training.
    X_train, X_test, X_val, y_train, y_test, y_val, *_ = \
        ff.timeblock_split_repeated(
            X, y, fractions=(2/3, 1/6, 1/6), n_cycles=n_cycles,
            gap_before_val=max(1, 1100 // stride),
            gap_before_test=max(1, 1100 // stride),
            order=("train", "test", "val"), copy=False,
        )

    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, scaler_X, scaler_y = \
        ff.scale_simple(X_train, X_val, X_test, y_train, y_val, y_test,
                        cols_to_scale=COLS_TO_SCALE)

    dtrain = xgb.DMatrix(X_train_s, label=y_train_s[TARGET])
    dsel   = xgb.DMatrix(X_test_s,  label=y_test_s[TARGET])
    dval   = xgb.DMatrix(X_val_s,   label=y_val_s[TARGET])
    base_score = float(y_train_s[TARGET].mean())

    trials = []
    for i in range(n_trials):
        hp = dict(baseline) if i == 0 else {k: draw(rng) for k, draw in space.items()}
        params = {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "tree_method": "hist", "nthread": -1, "base_score": base_score,
            **{k: v for k, v in hp.items() if k not in SCHEDULE_KEYS},
            "eta": hp["learning_rate"],
        }
        callbacks = [EarlyStopping(rounds=es_rounds, save_best=True,
                                   data_name="sel", metric_name="rmse"),
                     xgb.callback.LearningRateScheduler(make_lr_scheduler(
                         hp["learning_rate"], hp["lr_decay_factor"], hp["lr_step_size"]))]
        t0 = time.time()
        evals_result = {}
        booster = xgb.train(
            params, dtrain, num_boost_round=max_rounds,
            evals=[(dtrain, "train"), (dsel, "sel"), (dval, "val")],
            evals_result=evals_result,
            callbacks=callbacks,
            verbose_eval=False,
        )
        sel_rmse = float(np.min(evals_result["sel"]["rmse"]))
        best_round = int(booster.best_iteration)
        train_rmse = float(evals_result["train"]["rmse"][best_round])
        val_rmse = float(evals_result["val"]["rmse"][best_round])
        val_minus_sel = val_rmse - sel_rmse
        val_minus_train = val_rmse - train_rmse
        row = {**hp, "sel_rmse": sel_rmse,
               "train_rmse": train_rmse,
               "val_rmse": val_rmse,
               "val_minus_sel": val_minus_sel,
               "val_minus_train": val_minus_train,
               "val_over_sel": val_rmse / sel_rmse if sel_rmse else np.nan,
               "val_over_train": val_rmse / train_rmse if train_rmse else np.nan,
               "best_round": best_round,
               "seconds": round(time.time() - t0, 1),
               "is_baseline": int(i == 0)}
        trials.append(row)
        pd.DataFrame(trials).to_csv(os.path.join(out_dir, "tuning_trials.csv"), index=False)
        print(f"[{i+1:>2}/{n_trials}] train-RMSE={train_rmse:.5f} "
              f"sel-RMSE={sel_rmse:.5f} val-RMSE={val_rmse:.5f} "
              f"val-train={val_minus_train:+.5f} val-sel={val_minus_sel:+.5f} "
              f"rounds={row['best_round']:>4} "
              f"({row['seconds']:.0f}s)  {hp}")

    tdf = pd.DataFrame(trials).sort_values("sel_rmse").reset_index(drop=True)
    print("\nTop 5 configurations:")
    print(tdf.head(5).to_string(index=False))

    if make_plots:
        _plot_stability(tdf, n_trials, seed, out_dir)
        _plot_sensitivity(tdf, list(space.keys()), out_dir)
        print(f"\n📄 {out_dir}/tuning_trials.csv, tuning_stability.png, tuning_sensitivity.png")

    best_params = {k: tdf.iloc[0][k] for k in space.keys()}
    return tdf, best_params


def _plot_stability(tdf: pd.DataFrame, n_trials: int, seed: int, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tdf["sel_rmse"].values, marker="o", lw=1)
    base_rank = tdf.index[tdf["is_baseline"] == 1][0]
    ax.axhline(tdf.loc[base_rank, "sel_rmse"], color="red", ls="--",
               label=f"published config (rank {base_rank + 1})")
    ax.set_xlabel("Trial (sorted by selection RMSE)")
    ax.set_ylabel("Selection RMSE (scaled target)")
    ax.set_title(f"Random search, {n_trials} trials, seed {seed}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tuning_stability.png"), dpi=150)
    plt.close(fig)


def _plot_sensitivity(tdf: pd.DataFrame, keys: list, out_dir: str):
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), keys):
        ax.scatter(tdf[k], tdf["sel_rmse"], s=25)
        if k in ("min_child_weight", "learning_rate"):
            ax.set_xscale("log")
        ax.set_xlabel(k)
    np.atleast_1d(axes)[0].set_ylabel("Selection RMSE")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tuning_sensitivity.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # Thin env-var wrapper so the pipeline runner can drive the search.
    random_search(
        n_trials=int(os.environ.get("TUNE_TRIALS", 32)),
        stride=int(os.environ.get("TUNE_STRIDE", 4)),
        seed=int(os.environ.get("TUNE_SEED", 42)),
        out_dir=os.environ.get("TUNE_OUT", "tuning_v5"),
        max_rounds=int(os.environ.get("TUNE_MAX_ROUNDS", 3000)),
        es_rounds=int(os.environ.get("TUNE_ES_ROUNDS", 50)),
        n_cycles=int(os.environ.get("TUNE_N_CYCLES", 16)),
    )
