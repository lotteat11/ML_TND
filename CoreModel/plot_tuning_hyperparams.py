# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
plot_tuning_hyperparams.py — AGU-style hyperparameter sensitivity figure for a
random search produced by tune.py.

Reads tuning_trials.csv and draws one panel per searched hyperparameter, each
showing selection RMSE against that parameter, with a running median to make
the trend visible through the scatter. A final panel shows the learning-rate
schedule's effective lifetime (rounds until the rate has decayed to 1% of its
initial value), coloured by tree depth.

The baseline trial (is_baseline == 1) is excluded: it is the published
configuration carried over as trial 0 and sits far above the searched range,
which would otherwise set the y-axis for every panel.

Usage:
    python CoreModel/plot_tuning_hyperparams.py                       # v13
    python CoreModel/plot_tuning_hyperparams.py --trials DIR/tuning_trials.csv
    python CoreModel/plot_tuning_hyperparams.py --ylim 0.045 0.050
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRIALS = ROOT / "tuning_v13_tec3h_depth3_10" / "tuning_trials.csv"

# Panels in draw order: (column, axis label, log-x).
PARAMS = [
    ("max_depth",        "Maximum tree depth",        False),
    ("min_child_weight", "Minimum child weight",      True),
    ("subsample",        "Row subsample fraction",    False),
    ("colsample_bytree", "Feature subsample fraction", False),
    ("learning_rate",    "Initial learning rate",     True),
    ("lr_decay_factor",  "Learning-rate decay factor", False),
    ("lr_step_size",     "Decay step size (rounds)",  False),
]

# AGU/JGR: sans-serif, small type, thin rules, no heavy chrome. Two-column
# width is 7.5 in; the grid below is sized to sit inside that.
AGU_STYLE = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          8,
    "axes.titlesize":     8,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "pdf.fonttype":       42,   # embed TrueType, not Type 3 — AGU requires it
    "ps.fonttype":        42,
}

POINT = dict(s=9, c="0.55", alpha=0.55, linewidths=0, rasterized=True)
TREND = dict(color="#1f6feb", lw=1.3, zorder=3)


def running_median(x, y, bins=9, log_x=False):
    """Median of y within equal-count bins of x; returns (bin centre, median).

    Equal-count rather than equal-width bins: the search draws several
    parameters log-uniformly, so equal-width bins would leave the sparse end
    with too few trials to take a median over.
    """
    order = np.argsort(x)
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    n = len(xs)
    if n < bins * 2:
        bins = max(2, n // 2)
    edges = np.linspace(0, n, bins + 1).astype(int)
    cx, cy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        seg_x, seg_y = xs[lo:hi], ys[lo:hi]
        cx.append(np.exp(np.mean(np.log(seg_x))) if log_x else np.mean(seg_x))
        cy.append(np.median(seg_y))
    return np.array(cx), np.array(cy)


def lr_lifetime(decay, step):
    """Rounds until the scheduled learning rate falls to 1% of its initial value.

    The schedule is lr0 * decay ** (round // step), so the count is
    step * log(0.01) / log(decay). Decay >= 1 would never reach the threshold.
    """
    decay = np.asarray(decay, dtype=float)
    step = np.asarray(step, dtype=float)
    out = np.full(decay.shape, np.nan)
    ok = (decay > 0) & (decay < 1)
    out[ok] = step[ok] * np.log(0.01) / np.log(decay[ok])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS,
                        help="tuning_trials.csv written by tune.py.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output basename; defaults to the trials directory.")
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                        help="Selection-RMSE axis limits, e.g. --ylim 0.0445 0.0500.")
    parser.add_argument("--bins", type=int, default=9,
                        help="Equal-count bins behind each running median.")
    args = parser.parse_args()

    if not args.trials.is_file():
        raise SystemExit(f"Not found: {args.trials}")
    df = pd.read_csv(args.trials)
    n_all = len(df)
    if "is_baseline" in df:
        df = df[df["is_baseline"] != 1]
    df = df.dropna(subset=["sel_rmse"])
    print(f"{args.trials}: {n_all} trials, {len(df)} searched "
          f"(baseline excluded), depth {df.max_depth.min()}-{df.max_depth.max()}")

    df = df.assign(lr_life=lr_lifetime(df["lr_decay_factor"], df["lr_step_size"]))

    # Default the y-axis to the bulk of the trials so the panels resolve the
    # differences that matter; a handful of poor configurations sit above it.
    if args.ylim:
        lo, hi = args.ylim
    else:
        lo = df["sel_rmse"].min() - 0.0002
        hi = float(np.percentile(df["sel_rmse"], 95))
    clipped = int(((df["sel_rmse"] < lo) | (df["sel_rmse"] > hi)).sum())

    plt.rcParams.update(AGU_STYLE)
    fig, axes = plt.subplots(2, 4, figsize=(7.5, 4.0), sharey=True)
    flat = axes.ravel()

    for ax, (col, label, log_x) in zip(flat, PARAMS):
        ax.scatter(df[col], df["sel_rmse"], **POINT)
        mx, my = running_median(df[col].to_numpy(), df["sel_rmse"].to_numpy(),
                                bins=args.bins, log_x=log_x)
        ax.plot(mx, my, **TREND)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.tick_params(length=2.5, pad=2)

    # Final panel: schedule lifetime, coloured by depth. Depth is the one
    # parameter whose effect is entangled with the schedule (deep trees need
    # fewer rounds), so showing it as colour here rather than as a 9th panel.
    ax = flat[7]
    sc = ax.scatter(df["lr_life"], df["sel_rmse"], c=df["max_depth"],
                    cmap="viridis", s=9, alpha=0.85, linewidths=0,
                    rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("Rounds to 1% of initial rate")
    ax.tick_params(length=2.5, pad=2)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.06)
    cb.set_label("Tree depth", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.6)

    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel("Selection RMSE")
    flat[0].set_ylim(lo, hi)

    fig.tight_layout(w_pad=0.9, h_pad=1.1)

    out = args.out or (args.trials.parent / "tuning_hyperparameters")
    for ext in ("png", "pdf"):
        path = out.with_suffix(f".{ext}")
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)

    if clipped:
        print(f"note: {clipped} trial(s) outside the y-range "
              f"[{lo:.5f}, {hi:.5f}] are drawn off-panel")


if __name__ == "__main__":
    main()
