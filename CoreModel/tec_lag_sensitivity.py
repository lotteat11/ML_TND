#!/usr/bin/env python3
"""Fair TEC-lag test using the CoreModel training setup.

All candidates use identical complete rows, non-TEC features, splits, scaling,
and XGBoost settings. Candidates are selected on validation data. The test
split is reserved and never scored here, so it stays untouched for a single
final evaluation once the feature set is settled.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from xgboost.callback import EarlyStopping, LearningRateScheduler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import feature_functions as ff
from config import COLS_TO_SCALE, FEATURES, TARGET, TEC_LAG_COLS
from check_lst import VARIABLES, coverage_table

# TEC columns stripped from config.FEATURES to build the no_TEC baseline; the
# candidates below rebuild their own TEC features from --lags.
CORE_TEC = {"matched_tec_value", *TEC_LAG_COLS}
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / os.environ.get(
    "TRAIN_PARQUET_FILE", "grace_data_merged_v5_full.parquet")
# Match the production CoreModel window from run_pipeline.sh, which exports
# TRAIN_TIME_MIN=2002-01-01 / TRAIN_TIME_MAX=2016-01-01 (see the v8
# "..._2002train" model names). config.TIME_MIN's 2009-06-01 is only a
# bare-invocation fallback, so it is NOT the window to mirror here.
# 2002-2004 is solar maximum, where the ionosphere is active and TEC carries
# the most signal.  The grid that fixed the defaults below was run on it.  The
# full production window (TRAIN_TIME_MIN/MAX, 2002-2016) still applies when
# those variables are exported, e.g. by run_pipeline.sh.
DEFAULT_START = os.environ.get("TRAIN_TIME_MIN", "2002-01-01")
DEFAULT_END = os.environ.get("TRAIN_TIME_MAX", "2004-01-01")
RAW_COLUMNS = {
    "grace_time", "source", "lat", "lon", "alt_km", "rho_obs", "msis_rho",
    "matched_tec_value", "f107", "f107a", "ap_daily", "ap_0h", "ap_m3h",
    "ap_m6h", "ap_m9h", "ap_avg12_33h", "ap_avg36_57h",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--lags", default="2500s,3h,6h,9h,12h,24h")
    parser.add_argument("--n-cycles", type=int, default=9,
                        help="Time blocks per split. 9 beat 7 across the whole "
                             "2002-2004 grid (mean TEC gain -9.4%% vs -5.7%%), "
                             "and every cell where the lag feature lost was a "
                             "7-cycle cell.")
    parser.add_argument("--gap-rows", type=int, default=1100,
                        help="Row-count gap, used only by --split-mode rows.")
    parser.add_argument("--split-mode", choices=("time", "rows"), default="time",
                        help="'time' cuts blocks on real time so GA and GB stay "
                             "together and gaps are real quiet periods; 'rows' "
                             "reproduces the historical row-position split.")
    parser.add_argument("--gap-time", default="3d",
                        help="Quiet period between blocks for --split-mode time.")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Maximum total split rows; 0 uses all rows.")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Seed for --max-rows block subsampling.")
    parser.add_argument("--rounds", type=int, default=3000,
                        help="Upper bound only; early stopping ends most runs "
                             "well before this (~165 rounds at the default "
                             "learning rate).")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="XGBoost tree depth. 4 gave the lowest absolute "
                             "error and the largest TEC gain across the "
                             "2002-2004 grid at 9 cycles on a fixed row basis; "
                             "depth 8 was consistently worst.")
    parser.add_argument("--lr", type=float, default=LR_INITIAL,
                        help="Initial learning rate.")
    parser.add_argument("--lr-decay", type=float, default=LR_DECAY,
                        help="Multiplied into the rate every --lr-step rounds.")
    parser.add_argument("--lr-step", type=int, default=LR_STEP,
                        help="Rounds between decay steps. The default of 15 "
                             "decays very fast; tuning_v5/best_params.json "
                             "uses 81.")
    parser.add_argument("--early-stopping-rounds", type=int, default=200,
                        help="Early-stopping patience. Too small a value lets "
                             "candidates stop after ~40 rounds while others run "
                             "300+, so ranking gaps reflect stopping noise "
                             "rather than the TEC features under test.")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated seeds; scores are averaged over "
                             "them. The default of one seed matches how the "
                             "rest of the study reports single runs, and "
                             "reports a spread of zero -- which is not the "
                             "same as a small one. Candidate gaps here sit "
                             "near fit noise, and a single seed OVERSTATES the "
                             "TEC gain: at these defaults the best variant "
                             "measures -9.7%% on seed 42 but -3.5%% averaged "
                             "over 42,7,13, mostly because the no_TEC baseline "
                             "happens to land badly on that seed. Pass "
                             "--seeds 42,7,13 for any number being reported "
                             "as an effect size.")
    parser.add_argument("--row-basis", choices=("common", "none"), default="none",
                        help="Which rows every candidate is scored on. "
                             "'common' keeps only rows where all --lags "
                             "resolve, so the row set shifts when --lags "
                             "changes and results are not comparable across "
                             "runs. 'none' fixes the row set independently of "
                             "--lags, making runs comparable and charging no "
                             "candidate for lags it does not use.")
    parser.add_argument("--output-dir", type=Path, default=Path("tec_lag_sensitivity"))
    return parser.parse_args()


def lag_column(lag):
    return "tec_lag_" + lag.replace(" ", "")


def variants(lags, base):
    models = {"no_TEC": base, "current_TEC": base + ["matched_tec_value"]}
    for lag in lags:
        models[f"current+{lag}"] = base + ["matched_tec_value", lag_column(lag)]
    if "24h" in lags:
        for lag in lags:
            if lag != "24h":
                models[f"current+{lag}+24h"] = base + [
                    "matched_tec_value", lag_column(lag), lag_column("24h")]
    return models


def prepare_data(args, lags):
    available = set(pq.ParquetFile(args.data).schema.names)
    needed_raw = RAW_COLUMNS - {"ap_daily", "ap_0h"}
    missing = sorted(needed_raw - available)
    if missing:
        raise KeyError(f"Missing required parquet columns: {missing}")
    start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
    if end <= start:
        raise ValueError(
            f"--end ({args.end}) must be after --start ({args.start}); the "
            "interval is half-open [start, end), so equal values select no rows.")
    padding = max(pd.Timedelta(lag) for lag in lags)
    df = pd.read_parquet(
        args.data, columns=sorted(RAW_COLUMNS & available),
        filters=[("grace_time", ">=", start - padding), ("grace_time", "<", end)])
    df["time"] = pd.to_datetime(df["grace_time"], utc=True)
    df = df.sort_values("time", kind="mergesort").reset_index(drop=True)
    df = ff.add_lst_doy_features(df, copy=False)
    df["lon_sin"] = np.sin(np.deg2rad(df["lon"]))
    df["lon_cos"] = np.cos(np.deg2rad(df["lon"]))
    df["lst_lat_sin"] = df["lst_sin"] * df["lat"]
    df[TARGET] = np.log(df["rho_obs"] / df["msis_rho"])
    df = ff.add_tec_time_lag_features(
        df, lags=tuple(lags), names=tuple(lag_column(x) for x in lags),
        tolerance="10min")
    df = df[(df["time"] >= start) & (df["time"] < end)]
    base = [x for x in FEATURES if x not in CORE_TEC]
    required = base + ["matched_tec_value", TARGET, "rho_obs", "msis_rho"]
    # Which lags must resolve for a row to be kept.  Requiring EVERY requested
    # lag makes the row set a function of --lags: over 2002-2004 a single 3h
    # lag drops 1.5% of rows, the production 2500s+24h pair 3.9%, and five lags
    # 8.4%.  Two runs with different --lags therefore score different data, and
    # even within one run no_TEC and current_TEC -- which use no lag at all --
    # get charged for lags they never read.  The dropped rows are not random:
    # only rows inside long unbroken tracks can resolve a long lag, so the
    # survivors are the easy, gap-free stretches.
    #   basis=common (default): all candidates share the rows that resolve
    #     every requested lag.  Comparable within a run, not across runs.
    #   basis=none: the row set ignores lags entirely, so it is identical
    #     across runs regardless of --lags.  Candidates using a lag then carry
    #     NaNs, which XGBoost routes with its default direction -- fine here,
    #     and the price of a fixed denominator.
    if args.row_basis == "common":
        required += [lag_column(x) for x in lags]
    before = len(df)
    if before == 0:
        raise ValueError(
            f"No rows in {args.data} over {start} .. {end}. Check --start/--end "
            "(and TRAIN_TIME_MIN/TRAIN_TIME_MAX if exported).")
    print(f"Input: {args.data}")
    print(f"Requested interval: {start} .. {end} ({before:,} rows before complete-case filtering)")
    print("Non-null coverage of TEC inputs:")
    for column in ["matched_tec_value"] + [lag_column(x) for x in lags]:
        count = int(df[column].notna().sum())
        percentage = 100 * count / before if before else 0.0
        print(f"  {column:24s} {count:12,d}/{before:,} ({percentage:6.2f}%)")
    finite = df.replace([np.inf, -np.inf], np.nan)
    # A single all-NaN required column silently zeroes the joint mask, which
    # otherwise surfaces only as an unexplained "0 common complete-case rows"
    # failure in split_data. Name the offenders here instead.
    empty = [c for c in required if c in finite and not finite[c].notna().any()]
    if empty:
        raise ValueError(
            f"These required columns are entirely missing over {start} .. {end}: "
            f"{empty}. Check the time window, --lags, and AP_HISTORY "
            f"(config.FEATURES currently has {len(FEATURES)} features).")
    complete = finite.dropna(subset=required)
    if len(complete) == 0:
        worst = sorted(((float(finite[c].notna().mean()), c) for c in required
                        if c in finite))[:5]
        raise ValueError(
            "No rows have every required column present. Lowest-coverage "
            "columns: " + ", ".join(f"{c}={cov*100:.2f}%" for cov, c in worst))
    # The joint mask requires EVERY requested lag to resolve within tolerance,
    # so only rows inside long unbroken tracks survive. That subsample is not
    # neutral with respect to orbit geometry or season, and the no_TEC baseline
    # is then fitted on rows selected by TEC availability. Report the shift so
    # a biased comparison cannot be read as a clean one.
    retained = 100 * len(complete) / before if before else 0.0
    print(f"Complete-case rows (--row-basis {args.row_basis}): "
          f"{len(complete):,}/{before:,} ({retained:.2f}%)")
    # Always report the size of the other basis, so a run cannot be compared
    # with one that used a different rule without the difference being visible.
    lag_cols = [lag_column(x) for x in lags if lag_column(x) in finite]
    if lag_cols:
        both = len(finite.dropna(subset=required + lag_cols))
        if args.row_basis == "none":
            print(f"  (--row-basis common would keep {both:,}, "
                  f"{len(complete) - both:,} fewer)")
        else:
            loose = len(finite.dropna(subset=[c for c in required
                                              if c not in lag_cols]))
            print(f"  (--row-basis none would keep {loose:,}, "
                  f"{loose - len(complete):,} more)")
        for column in lag_cols:
            nn = int(complete[column].notna().sum()) if column in complete else 0
            if nn < len(complete):
                print(f"  {column:24s} NaN in {len(complete) - nn:,} kept rows "
                      f"({100 * (len(complete) - nn) / max(1, len(complete)):.2f}%)")
    if len(complete) and retained < 99.0:
        print("Complete-case selection bias (all rows -> complete cases):")
        for column in ["lst_h", "lat", "alt_km", "doy"]:
            if column not in df or column not in complete:
                continue
            print(f"  {column:8s} mean {df[column].mean():9.3f} -> "
                  f"{complete[column].mean():9.3f}   "
                  f"std {df[column].std():8.3f} -> {complete[column].std():8.3f}")
    return complete.reset_index(drop=True)


def sample_indices(indices, max_rows, seed=42, block=2048):
    """Downsample each split by keeping whole contiguous blocks.

    An evenly spaced stride (the previous np.linspace approach) aliases against
    the ~93-minute orbital period at GRACE's ~10 s cadence, which manufactures
    periodic local-time and longitude structure and flattens the short-lag TEC
    gradients this script is meant to measure. Sampling contiguous blocks keeps
    the within-block cadence intact so the lag features stay meaningful.
    """
    total = sum(map(len, indices))
    if max_rows <= 0 or total <= max_rows:
        return indices
    rng = np.random.default_rng(seed)
    result = []
    for index in indices:
        count = max(1, round(max_rows * len(index) / total))
        if count >= len(index):
            result.append(index)
            continue
        starts = np.arange(0, len(index), block)
        n_blocks = max(1, int(np.ceil(count / block)))
        chosen = np.sort(rng.choice(starts, size=min(n_blocks, len(starts)),
                                    replace=False))
        picked = np.concatenate([index[start:start + block] for start in chosen])
        result.append(picked[:count])
    return tuple(result)


def timeblock_split_by_time(df, n_cycles, gap, fractions=(2 / 3, 1 / 6, 1 / 6)):
    """Cyclic train/val/test split cut on TIME rather than on row position.

    GRACE A and B fly in formation and share timestamps to the second, so the
    rows arrive interleaved GA/GB/GA/GB after a plain sort on time. A split
    that cuts on row position therefore puts GA and GB samples of the SAME
    instant into different splits, and a row-count gap of 1100 spans only
    ~1.5 hours of real time. Measured on 2009-06, that left 98.9% of test rows
    within 60 s of a train row: the thermosphere does not change on that scale,
    so validation and test were largely re-measuring the training data.

    Cutting on time keeps every spacecraft sampling the same instant in the
    same split, and makes the gap a real quiet period between blocks.
    """
    # Work in integer seconds throughout: mixing numpy datetime64/timedelta64
    # with pandas Timedelta and float fractions silently produces wrong block
    # boundaries (it put every block inside a single 40-minute window).
    # Convert via datetime64[s] rather than dividing raw int64: the column is
    # datetime64[us] here but datetime64[ns] elsewhere, and a hard-coded 10**9
    # silently mis-scales one of them (it collapsed a 30-day span to 43 min).
    seconds = df["time"].to_numpy().astype("datetime64[s]").astype("int64")
    start, end = int(seconds.min()), int(seconds.max())
    span = end - start
    cycle = span / n_cycles
    gap_s = int(pd.Timedelta(gap).total_seconds())
    # Three gaps, not two: before val, before test, and after test so the NEXT
    # cycle's train block cannot start immediately against this cycle's test.
    usable = cycle - 3 * gap_s
    if usable <= 0:
        raise ValueError(
            f"--gap-time {gap} is too large for {n_cycles} cycles over "
            f"{span / 86400:.1f} days: each cycle would be fully consumed by "
            "its three gaps.")
    train_idx, val_idx, test_idx = [], [], []
    for cycle_number in range(n_cycles):
        cycle_start = start + cycle_number * cycle
        train_end = cycle_start + usable * fractions[0]
        val_start = train_end + gap_s
        val_end = val_start + usable * fractions[1]
        test_start = val_end + gap_s
        test_end = test_start + usable * fractions[2]
        train_idx.append((seconds >= cycle_start) & (seconds < train_end))
        val_idx.append((seconds >= val_start) & (seconds < val_end))
        test_idx.append((seconds >= test_start) & (seconds < test_end))
    combine = lambda masks: np.flatnonzero(np.logical_or.reduce(masks))
    return combine(train_idx), combine(val_idx), combine(test_idx)


def split_data(df, args):
    minimum = 3 * args.n_cycles
    if len(df) < minimum:
        raise ValueError(
            f"Only {len(df)} common complete-case rows remain, but {minimum} "
            f"are required for {args.n_cycles} cycles. Check the coverage "
            "printed above, widen --start/--end, or remove an unavailable "
            "candidate from --lags.")
    if args.split_mode == "time":
        indices = timeblock_split_by_time(df, args.n_cycles, args.gap_time)
    else:
        dummy = df[[TARGET]]
        split = ff.timeblock_split_repeated(
            dummy, dummy, fractions=(2 / 3, 1 / 6, 1 / 6),
            n_cycles=args.n_cycles, gap_before_val=args.gap_rows,
            gap_before_test=args.gap_rows, order=("train", "test", "val"),
            copy=False)
        indices = tuple(np.asarray(index, dtype=int) for index in split[6:9])
    if min(len(index) for index in indices) == 0:
        raise ValueError(
            "One of the splits is empty; reduce --n-cycles or --gap-time.")
    return sample_indices(indices, args.max_rows, seed=args.sample_seed), indices


# Learning-rate schedule, overridable from the command line.  Defaults are a
# CONSTANT rate of 0.05 (decay 1.0), which is the setting the 2002-2004 grid
# sweep was run under and the one its conclusions rest on -- a decaying rate
# would make new runs incomparable with those results.  A decay schedule is
# still available via --lr-decay/--lr-step; note that decaying every 15 rounds
# drives the rate to ~5e-5 by round 800, so past roughly round 500 the model
# stops learning and a large --rounds buys nothing.
LR_INITIAL = 0.05
LR_DECAY = 1.0
LR_STEP = 40


def lr_scheduler(round_number):
    return LR_INITIAL * (LR_DECAY ** (round_number // LR_STEP))


def fit_predict(df, features, train_idx, eval_idx, rounds, early_stop,
                seed=42, early_stopping_rounds=100, max_depth=5):
    scale_cols = [x for x in COLS_TO_SCALE if x in features]
    X_train, X_eval = df.loc[train_idx, features], df.loc[eval_idx, features]
    y_train, y_eval = df.loc[train_idx, [TARGET]], df.loc[eval_idx, [TARGET]]
    X_train_s, _, X_eval_s, y_train_s, _, _, scaler_X, scaler_y = ff.scale_simple(
        X_train, X_eval, X_eval, y_train, y_eval, y_eval, scale_cols)
    dtrain = xgb.DMatrix(X_train_s, label=y_train_s[TARGET])
    deval = xgb.DMatrix(X_eval_s)
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "max_depth": max_depth,
        "min_child_weight": 14.637, "subsample": 0.643, "colsample_bytree": 0.694,
        "tree_method": "hist", "nthread": -1, "seed": seed,
        "base_score": float(y_train_s[TARGET].mean())}
    callbacks = [LearningRateScheduler(lr_scheduler)]
    evals = []
    if early_stop:
        # Early stopping requires validation labels, but test fitting does not.
        deval = xgb.DMatrix(X_eval_s, label=scaler_y.transform(y_eval).ravel())
        callbacks.append(EarlyStopping(rounds=early_stopping_rounds, save_best=True,
                                       data_name="validation", metric_name="rmse"))
        evals = [(deval, "validation")]
    model = xgb.train(params, dtrain, num_boost_round=rounds, evals=evals,
                      callbacks=callbacks, verbose_eval=False)
    prediction_s = model.predict(deval)
    prediction = scaler_y.inverse_transform(prediction_s.reshape(-1, 1)).ravel()
    best_rounds = model.best_iteration + 1 if early_stop else rounds
    return prediction, best_rounds, model, scaler_X, scaler_y


def safe_name(name):
    """Return a filesystem-safe, still-readable model name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def plot_feature_importance(model, model_name, output_dir):
    """Save gain importance for every feature used by one candidate model."""
    importance = model.get_score(importance_type="gain")
    features = model.feature_names or []
    values = np.array([importance.get(feature, 0.0) for feature in features])
    order = np.argsort(values)
    fig_height = max(5, 0.34 * len(features) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(np.asarray(features)[order], values[order], color="#3976a8")
    ax.set_xlabel("XGBoost gain")
    ax.set_title(f"Feature importance — {model_name}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = output_dir / f"{safe_name(model_name)}_feature_importance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_all_feature_importances(importances, output_dir):
    """Save one normalized feature-importance comparison across candidates."""
    feature_names = sorted({
        feature for model_scores in importances.values() for feature in model_scores
    })
    model_names = list(importances)
    matrix = np.array([
        [importances[model].get(feature, 0.0) for feature in feature_names]
        for model in model_names
    ], dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals != 0) * 100
    fig_width = max(11, 0.62 * len(feature_names) + 4)
    fig_height = max(6, 0.48 * len(model_names) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(np.arange(len(feature_names)), feature_names,
                  rotation=55, ha="right")
    ax.set_yticks(np.arange(len(model_names)), model_names)
    ax.set_xlabel("Feature")
    ax.set_ylabel("TEC test variant")
    ax.set_title("Feature importance for all TEC tests (gain normalized per model)")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Gain importance (%)")
    fig.tight_layout()
    path = output_dir / "feature_importance_all_tests.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_first_three_days(df, features, model, scaler_X, scaler_y,
                          model_name, output_dir):
    """Plot observed/predicted density and inputs over the first three days."""
    period_start = df["time"].min()
    period = df[df["time"] < period_start + pd.Timedelta(days=3)]
    if period.empty:
        raise ValueError("No observations are available for the three-day plot")
    scale_cols = [column for column in COLS_TO_SCALE if column in features]
    X_period = ff.scale_transform(period[features], scaler_X, scale_cols)
    prediction_scaled = model.predict(xgb.DMatrix(X_period))
    prediction_log = scaler_y.inverse_transform(
        prediction_scaled.reshape(-1, 1)).ravel()
    predicted_density = period["msis_rho"].to_numpy() * np.exp(prediction_log)

    n_panels = 1 + len(features)
    fig_height = max(8, 1.75 * n_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, fig_height),
                             sharex=True, constrained_layout=True)
    axes[0].plot(period["time"], period["rho_obs"], color="black",
                 linewidth=1.15, label="Observed (true)")
    axes[0].plot(period["time"], predicted_density, color="red",
                 linestyle="--", linewidth=1.0, label="Prediction")
    axes[0].plot(period["time"], period["msis_rho"], color="blue",
                 linewidth=1.0, label="MSIS")
    axes[0].set_ylabel("rho")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3, linestyle=":")
    for ax, column in zip(axes[1:], features):
        ax.plot(period["time"], period[column], color="#ff8c1a", linewidth=0.9)
        ax.set_ylabel(column, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.3, linestyle=":")
    axes[0].set_title(
        f"Performance and feature dynamics — {model_name}\n"
        f"first three data days from {period_start:%Y-%m-%d}")
    axes[-1].set_xlabel("Time (UTC)")
    path = output_dir / f"{safe_name(model_name)}_first_3_days_timeseries.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_altitude_split(df, train_idx, val_idx, test_idx, output_dir):
    """Plot altitude over the full data period, coloured by split role."""
    fig, ax = plt.subplots(figsize=(16, 6))
    roles = [
        ("train", train_idx, "#2878b5"),
        ("validation", val_idx, "#e69f00"),
        ("test", test_idx, "#c33c54"),
    ]
    for label, index, color in roles:
        ax.scatter(df.loc[index, "time"], df.loc[index, "alt_km"],
                   s=1.2, alpha=0.20, color=color, label=f"{label} (n={len(index):,})",
                   rasterized=True)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("CoreModel split coverage: raw altitude versus time")
    ax.grid(alpha=0.2)
    ax.legend(markerscale=5)
    fig.tight_layout()
    path = output_dir / "altitude_vs_time_train_validation_test.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_altitude_by_satellite(df, output_dir):
    """Compare daily GA/GB altitude and quantify their same-day difference."""
    diagnostic = df[["time", "alt_km", "source"]].copy()
    source = diagnostic["source"].astype("string")
    diagnostic["satellite"] = source.str.extract(
        r"(?:^|[/\\])((?:GA|GB|GC|GD))(?:_|-)", expand=False)
    missing = diagnostic["satellite"].isna()
    if missing.any():
        diagnostic.loc[missing, "satellite"] = source[missing].str.extract(
            r"^([A-Za-z]{2})(?:_|-)", expand=False)
    diagnostic = diagnostic.dropna(subset=["satellite"])
    daily = (diagnostic.set_index("time")
             .groupby("satellite")["alt_km"].resample("1D").median()
             .unstack(0).sort_index())
    satellites = [name for name in ["GA", "GB", "GC", "GD"] if name in daily]
    if not satellites:
        print("WARNING: Could not infer satellite IDs from source; skipping GA/GB altitude plot")
        return None

    fig, (ax, delta_ax) = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    colors = {"GA": "#2878b5", "GB": "#e69f00", "GC": "#299d8f", "GD": "#c33c54"}
    for satellite in satellites:
        raw = diagnostic[diagnostic["satellite"] == satellite]
        ax.scatter(raw["time"], raw["alt_km"], color=colors[satellite],
                   s=1.0, alpha=0.25, rasterized=True,
                   label=f"{satellite} raw (n={len(raw):,})")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Raw altitude by GRACE spacecraft")
    ax.grid(alpha=0.25)
    ax.legend()

    pairs = [("GA", "GB"), ("GC", "GD")]
    plotted_delta = False
    for first, second in pairs:
        if first not in daily or second not in daily:
            continue
        difference = (daily[first] - daily[second]).dropna()
        if difference.empty:
            continue
        absolute = difference.abs()
        print(
            f"Altitude consistency {first}-{second}: overlapping days={len(difference):,}, "
            f"median signed difference={difference.median():.3f} km, "
            f"median absolute difference={absolute.median():.3f} km, "
            f"95th percentile absolute difference={absolute.quantile(0.95):.3f} km")
        delta_ax.plot(difference.index, difference, color=colors[first], linewidth=0.9,
                      label=f"{first} - {second}")
        plotted_delta = True
    delta_ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    delta_ax.set_ylabel("Difference (km)")
    delta_ax.set_xlabel("Time (UTC)")
    delta_ax.grid(alpha=0.25)
    if plotted_delta:
        delta_ax.legend()
    else:
        delta_ax.text(0.5, 0.5, "No overlapping spacecraft pair found",
                      transform=delta_ax.transAxes, ha="center", va="center")
    path = output_dir / "altitude_daily_by_satellite.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def report_split_coverage(df, train_idx, val_idx, test_idx, output_dir):
    """Table 2 coverage (LST / F10.7 / Ap) for the splits this script fits on.

    Reuses check_lst.coverage_table so the numbers match the paper's table
    exactly. A solar-activity range that differs across splits, or that is
    narrow relative to the full mission, changes which features carry gain --
    so this is read alongside the feature-importance plots, not separately.
    """
    missing = [c for c in VARIABLES.values() if c not in df.columns]
    if missing:
        print(f"WARNING: coverage table skipped, missing columns: {missing}")
        return None
    splits = {"train": df.loc[train_idx], "val": df.loc[val_idx],
              "test": df.loc[test_idx]}
    table = coverage_table(splits)
    print("\nSplit coverage (full range and descriptive central 90%):")
    print(table.to_string(index=False))
    path = output_dir / "split_coverage.csv"
    table.to_csv(path, index=False)
    return path


def score(df, index, prediction):
    observed_log = df.loc[index, TARGET].to_numpy()
    observed_density = df.loc[index, "rho_obs"].to_numpy()
    predicted_density = df.loc[index, "msis_rho"].to_numpy() * np.exp(prediction)
    return {
        "log_rmse": float(np.sqrt(np.mean((prediction - observed_log) ** 2))),
        "density_rmse": float(np.sqrt(np.mean(
            (predicted_density - observed_density) ** 2))),
        "mape_pct": float(100 * np.mean(
            np.abs((predicted_density - observed_density) / observed_density)))}


def main():
    args = arguments()
    if args.n_cycles < 1 or args.gap_rows < 0:
        raise ValueError("n-cycles must be positive and gap-rows non-negative")
    lags = [x.strip() for x in args.lags.split(",") if x.strip()]
    if not lags:
        raise ValueError("--lags must contain at least one lag")
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")
    global LR_INITIAL, LR_DECAY, LR_STEP
    LR_INITIAL, LR_DECAY, LR_STEP = args.lr, args.lr_decay, args.lr_step
    if LR_STEP < 1:
        raise ValueError("--lr-step must be at least 1")
    print(f"Tree depth: {args.max_depth}")
    print(f"Learning rate: {LR_INITIAL} x {LR_DECAY}^(round//{LR_STEP}); "
          f"at round {args.rounds} it is "
          f"{LR_INITIAL * LR_DECAY ** (args.rounds // LR_STEP):.3e}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_data(args, lags)
    (train_idx, val_idx, test_idx), raw_split_indices = split_data(df, args)
    print(f"Split rows: train={len(train_idx):,}, validation={len(val_idx):,}, "
          f"untouched test={len(test_idx):,}")
    split_plot_path = plot_altitude_split(
        df, *raw_split_indices, args.output_dir)
    print(f"Split overview: {split_plot_path}")
    satellite_plot_path = plot_altitude_by_satellite(df, args.output_dir)
    if satellite_plot_path is not None:
        print(f"Satellite altitude check: {satellite_plot_path}")
    coverage_path = report_split_coverage(df, *raw_split_indices, args.output_dir)
    if coverage_path is not None:
        print(f"Split coverage table: {coverage_path}")
    base = [x for x in FEATURES if x not in CORE_TEC]
    models, rows = variants(lags, base), []
    all_importances = {}
    for name, features in models.items():
        # Average across seeds: candidate gaps here are small enough that a
        # single seed cannot distinguish a real TEC effect from fit noise.
        seed_log, seed_density, seed_mape, seed_rounds = [], [], [], []
        model = scaler_X = scaler_y = None
        for seed in seeds:
            prediction, best_rounds, model, scaler_X, scaler_y = fit_predict(
                df, features, train_idx, val_idx, args.rounds, True,
                seed=seed, early_stopping_rounds=args.early_stopping_rounds,
                max_depth=args.max_depth)
            metric = score(df, val_idx, prediction)
            seed_log.append(metric["log_rmse"])
            seed_density.append(metric["density_rmse"])
            seed_mape.append(metric["mape_pct"])
            seed_rounds.append(best_rounds)
        spread = float(np.std(seed_log)) if len(seed_log) > 1 else 0.0
        rows.append({"model": name, "n_features": len(features),
                     "validation_log_rmse": float(np.mean(seed_log)),
                     "validation_log_rmse_std": spread,
                     "validation_density_rmse": float(np.mean(seed_density)),
                     "validation_mape_pct": float(np.mean(seed_mape)),
                     "selected_rounds": int(round(float(np.mean(seed_rounds)))),
                     "n_seeds": len(seeds)})
        print(f"{name:22s} log-RMSE={np.mean(seed_log):.6f}+-{spread:.6f} "
              f"density-RMSE={np.mean(seed_density):.6e} kg/m^3 "
              f"MAPE={np.mean(seed_mape):.3f}% "
              f"rounds={int(round(float(np.mean(seed_rounds))))}")
        diagnostics_dir = args.output_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        importance_path = plot_feature_importance(
            model, name, diagnostics_dir)
        gain = model.get_score(importance_type="gain")
        all_importances[name] = {
            feature: gain.get(feature, 0.0) for feature in (model.feature_names or [])}
        timeseries_path = plot_first_three_days(
            df, features, model, scaler_X, scaler_y, name, diagnostics_dir)
        print(f"  importance: {importance_path}")
        print(f"  first 3 days: {timeseries_path}")
    # Ranked on validation, the only split this script reads labels from.
    # The test split is built and reserved but never scored here, so it stays
    # untouched for a single final evaluation once the feature set is settled.
    results = pd.DataFrame(rows).sort_values("validation_log_rmse").reset_index(drop=True)
    winner_name = str(results.iloc[0]["model"])
    results["selected"] = results["model"].eq(winner_name)
    results.to_csv(args.output_dir / "results.csv", index=False)
    combined_importance_path = plot_all_feature_importances(
        all_importances, args.output_dir)

    display = results.copy()
    display["validation_log_rmse"] = display["validation_log_rmse"].map(
        lambda value: f"{value:.6f}")
    display["validation_log_rmse_std"] = display["validation_log_rmse_std"].map(
        lambda value: f"{value:.6f}")
    display["validation_density_rmse"] = display["validation_density_rmse"].map(
        lambda value: f"{value:.6e}")
    display["validation_mape_pct"] = display["validation_mape_pct"].map(
        lambda value: f"{value:.3f}")
    print("\nAll TEC variants (ranked by validation log-RMSE):")
    print(display.to_string(index=False))
    print(f"\nCombined feature importance: {combined_importance_path}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(results["model"], results["validation_log_rmse"], color="#3976a8")
    ax.invert_yaxis()
    ax.set_xlabel("Validation log-RMSE (lower is better)")
    ax.set_title("TEC lag sensitivity — CoreModel setup")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "lag_sensitivity.png", dpi=180)
    plt.close(fig)
    print(f"\nSelected on validation: {winner_name}")
    print("Test split reserved and not scored; evaluate it once the feature "
          "set is final.")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
