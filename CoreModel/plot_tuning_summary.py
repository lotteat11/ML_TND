# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
plot_tuning_summary.py — AGU-style two-panel summary of a tune.py random search.

(a) Which hyperparameters matter: |Spearman rho| between each searched
    parameter and selection RMSE, sorted. Also shown is the derived lifetime of
    the learning-rate schedule (rounds until the rate reaches 1% of its initial
    value), which summarises the initial rate, the decay factor and the step
    size as a single quantity.
(b) What the choice costs: validation RMSE against the validation-minus-
    training gap, coloured by tree depth. Both axes are "lower is better", so
    the preferred corner is the lower left. The Pareto front and the selected
    configuration are marked.

Usage:
    python CoreModel/plot_tuning_summary.py
    python CoreModel/plot_tuning_summary.py --trials DIR/tuning_trials.csv
    python CoreModel/plot_tuning_summary.py --select-row 123
    python CoreModel/plot_tuning_summary.py --select best-val
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRIALS = ROOT / "tuning_v13_tec3h_depth3_10" / "tuning_trials.csv"

PARAM_LABELS = {
    "lr_life":          "Schedule lifetime*",
    "learning_rate":    "Initial learning rate",
    "lr_decay_factor":  "Decay factor",
    "max_depth":        "Tree depth",
    "lr_step_size":     "Decay step size",
    "colsample_bytree": "Feature subsample",
    "min_child_weight": "Min. child weight",
    "subsample":        "Row subsample",
}
SEARCHED = ["max_depth", "min_child_weight", "subsample", "colsample_bytree",
            "learning_rate", "lr_decay_factor", "lr_step_size"]

AGU_STYLE = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          8,
    "axes.titlesize":     8,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    6.5,
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
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
}

ACCENT = "#1f6feb"
MARK = "#d1495b"


def lr_lifetime(decay, step):
    """Rounds until lr0 * decay ** (round // step) falls to 1% of lr0."""
    decay = np.asarray(decay, dtype=float)
    step = np.asarray(step, dtype=float)
    out = np.full(decay.shape, np.nan)
    ok = (decay > 0) & (decay < 1)
    out[ok] = step[ok] * np.log(0.01) / np.log(decay[ok])
    return out


def pareto_front(df, cols=("val_rmse", "val_minus_train")):
    """Rows not dominated on both columns (lower is better on each)."""
    pts = df[list(cols)].to_numpy()
    keep = ~np.array([
        np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)) for p in pts
    ])
    return df[keep]


def panel_influence(ax, df):
    rows = []
    for c in SEARCHED + ["lr_life"]:
        rho = df[c].corr(df["sel_rmse"], method="spearman")
        rows.append((PARAM_LABELS.get(c, c), abs(rho), c == "lr_life"))
    rows.sort(key=lambda r: r[1])
    vals = [r[1] for r in rows]
    colors = ["0.62" if r[2] else ACCENT for r in rows]

    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, height=0.66)
    ax.set_yticks(y, [r[0] for r in rows])
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=6.5,
                color="0.25")
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xlabel("Influence on selection RMSE  ($|\\rho|$)")
    ax.tick_params(length=2.5, pad=2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def panel_tradeoff(ax, fig, df, sel_row):
    sc = ax.scatter(df["val_rmse"], df["val_minus_train"], c=df["max_depth"],
                    cmap="viridis", s=12, alpha=0.85, linewidths=0,
                    rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("Tree depth", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.6)

    front = pareto_front(df).sort_values("val_rmse")
    ax.plot(front["val_rmse"], front["val_minus_train"], color="0.35",
            lw=0.8, ls="-", zorder=2, label="best possible trade-off")

    if sel_row is not None:
        ax.plot([sel_row["val_rmse"]], [sel_row["val_minus_train"]],
                marker="o", ms=7, mfc="none", mec=MARK, mew=1.4, zorder=6,
                label="selected")

    ax.set_xlabel("Validation RMSE")
    ax.set_ylabel("Overfitting: validation $-$ training RMSE")
    ax.tick_params(length=2.5, pad=2)
    ax.legend(loc="upper right", frameon=False, handlelength=1.4)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--select", choices=("compromise", "best-sel", "best-val", "none"),
                   default="compromise",
                   help="Which configuration to circle. 'compromise' takes the "
                        "smallest overfitting gap on the Pareto front among "
                        "trials within --val-tol of the best validation RMSE.")
    p.add_argument("--select-row", type=int, default=None,
                   help="Circle this CSV row index instead (overrides --select).")
    p.add_argument("--val-tol", type=float, default=1.3,
                   help="Percent of validation RMSE to give up, for "
                        "--select compromise.")
    p.add_argument("--write-params", action="store_true",
                   help="Write the marked configuration to best_params.json.")
    args = p.parse_args()

    if not args.trials.is_file():
        raise SystemExit(f"Not found: {args.trials}")
    raw = pd.read_csv(args.trials)
    df = raw[raw.get("is_baseline", 0) != 1].dropna(subset=["sel_rmse"]).copy()
    df["lr_life"] = lr_lifetime(df["lr_decay_factor"], df["lr_step_size"])

    if args.select_row is not None:
        sel_row = raw.loc[args.select_row]
    elif args.select == "none":
        sel_row = None
    elif args.select == "best-sel":
        sel_row = df.loc[df["sel_rmse"].idxmin()]
    elif args.select == "best-val":
        sel_row = df.loc[df["val_rmse"].idxmin()]
    else:
        front = pareto_front(df)
        near = front[front["val_rmse"] <=
                     df["val_rmse"].min() * (1 + args.val_tol / 100)]
        sel_row = near.loc[near["val_minus_train"].idxmin()]

    print(f"{args.trials}: {len(raw)} trials, {len(df)} searched")
    if sel_row is not None:
        best_val, best_sel = df["val_rmse"].min(), df["sel_rmse"].min()
        worst_gap = df.loc[df["val_rmse"].idxmin(), "val_minus_train"]
        print("marked configuration:")
        for k in SEARCHED:
            print(f"  {k:18s} {sel_row[k]}")
        print(f"  sel_rmse           {sel_row['sel_rmse']:.5f} "
              f"({100 * (sel_row['sel_rmse'] / best_sel - 1):+.2f}% vs best)")
        print(f"  val_rmse           {sel_row['val_rmse']:.5f} "
              f"({100 * (sel_row['val_rmse'] / best_val - 1):+.2f}% vs best)")
        print(f"  val-train gap      {sel_row['val_minus_train']:.5f} "
              f"({100 * (sel_row['val_minus_train'] / worst_gap - 1):+.0f}% vs "
              f"the best-validation trial)")
        print(f"  boosting rounds    {int(sel_row['best_round'])}")

        if args.write_params:
            params = {k: (int(sel_row[k]) if k in ("max_depth", "lr_step_size")
                          else float(sel_row[k])) for k in SEARCHED}
            path = args.trials.parent / "best_params.json"
            path.write_text(json.dumps(params, indent=2) + "\n")
            print(f"wrote {path}")

    plt.rcParams.update(AGU_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    panel_influence(axes[0], df)
    panel_tradeoff(axes[1], fig, df, sel_row)

    for ax, tag in zip(axes, "ab"):
        ax.set_title(f"({tag})", loc="left", fontweight="bold", pad=4)

    fig.tight_layout(w_pad=2.0)
    fig.text(0.01, -0.02,
             "*rounds until the scheduled learning rate reaches 1% of its "
             "initial value", fontsize=6, color="0.35")

    out = args.out or (args.trials.parent / "tuning_summary")
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"))
        print(f"wrote {out.with_suffix('.' + ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
