# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
plot_parity.py — AGU-style density parity plots for the on-track runs.

One panel per (regime, adaptation) pair: observed density on the x-axis against
NRLMSIS-2.1 and the corrected model on the y-axis, with the 1:1 line. Points on
the line are perfect; NRLMSIS-2.1 sitting above or below it shows the direction
of its bias in that regime.

Panels are drawn on log axes because density spans two decades between solar
minimum and storm conditions, and are subsampled so the vector output stays
manageable.

Usage:
    python CoreModel/plot_parity.py
    python CoreModel/plot_parity.py --horizon 3
    python CoreModel/plot_parity.py --max-points 60000
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = ROOT / "runs_final_20250821"

REGIMES = [("quiet2009", "2009"),
           ("storm2015", "Period 2015"),
           ("post2016", "Post-2016")]

C_MSIS = "#2e8b57"
C_PRED = "#1f6feb"

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


def rmse_log(obs, model):
    k = (obs > 0) & (model > 0)
    return float(np.sqrt(np.mean((np.log(obs[k]) - np.log(model[k])) ** 2)))


def panel(ax, d, title, max_points, rng, scale="linear", clip=None, top=None):
    obs = d["rho_obs"].to_numpy()
    msis = d["msis_rho"].to_numpy()
    pred = d["rho_pred"].to_numpy()
    keep = (obs > 0) & (msis > 0) & (pred > 0)
    obs, msis, pred = obs[keep], msis[keep], pred[keep]

    scores = (rmse_log(obs, msis), rmse_log(obs, pred))

    # Subsample for drawing only; the scores above use every sample.
    if max_points and len(obs) > max_points:
        idx = rng.choice(len(obs), max_points, replace=False)
        o, m, p = obs[idx], msis[idx], pred[idx]
    else:
        o, m, p = obs, msis, pred

    ax.scatter(o, m, s=.7, alpha=.10, c=C_MSIS, linewidths=0, rasterized=True,
               label="NRLMSIS-2.1")
    ax.scatter(o, p, s=.7, alpha=.10, c=C_PRED, linewidths=0, rasterized=True,
               label="ML model")

    if scale == "log":
        lo = float(np.percentile(obs, 0.2))
        hi = float(np.percentile(obs, 99.8))
        ax.set_xscale("log"); ax.set_yscale("log")
    else:
        # Linear axes start at zero and run past the bulk of the samples, so a
        # constant multiplicative bias reads as a fan opening from the origin.
        # Without --clip the range covers every sample, matching the published
        # figures, where matplotlib autoscaled to the full extent.
        lo = 0.0
        allv = np.concatenate([obs, msis, pred])
        hi = float(np.percentile(allv, clip) if clip else allv.max())
        if top is not None:
            hi = top
    ax.plot([lo, hi], [lo, hi], color="0.25", lw=.8, ls="--", zorder=5,
            label="1:1 line")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=4)
    ax.tick_params(length=2.5, pad=2)
    # Label in units of 1e-12 kg/m3 so the ticks stay short on both scales.
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_formatter(ticker.NullFormatter())
        axis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v / 1e-12:g}"))
    if scale == "log" and np.log10(hi) - np.log10(lo) < 1.2:
        ticks = [t for t in ticker.MaxNLocator(nbins=4).tick_values(lo, hi)
                 if lo <= t <= hi]
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())
    elif scale == "linear":
        if top is not None:
            ticks = np.linspace(0, top, 5)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    leg = ax.legend(loc="upper left", frameon=False, handlelength=.9,
                    handletextpad=.5, borderaxespad=.3)
    leg._legend_box.align = "left"
    for h in leg.legend_handles:
        if hasattr(h, "set_sizes"):
            h.set_sizes([14]); h.set_alpha(.9)
    return len(obs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--horizon", type=int, default=1, choices=(1, 3))
    p.add_argument("--variant", choices=("core", "adaptive", "both"),
                   default="both",
                   help="Plot the core model (dr0), adaptive model (dr1), or both.")
    p.add_argument("--max-points", type=int, default=150000,
                   help="Points drawn per panel per series; 0 draws every "
                        "sample (large vector output).")
    p.add_argument("--scale", choices=("linear", "log"), default="linear",
                   help="Axis scale; linear matches the published figure.")
    p.add_argument("--limit", action="append", default=[], metavar="REGIME=MAX",
                   help="Fix the axis maximum for one regime, in units of "
                        "1e-12 kg/m3, e.g. --limit quiet2009=0.7. Repeatable.")
    p.add_argument("--panel-limit", action="append", default=[], metavar="PANEL=MAX",
                   help="Override the axis maximum for a lettered panel, in "
                        "units of 1e-12 kg/m3, e.g. --panel-limit e=9.")
    p.add_argument("--clip", type=float, default=None,
                   help="Percentile to cut the linear axes at, e.g. 99.9. "
                        "Omit to show the full range of the data.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    limits = {}
    for item in args.limit:
        name, _, value = item.partition("=")
        limits[name] = float(value) * 1e-12
    panel_limits = {}
    for item in args.panel_limit:
        name, _, value = item.partition("=")
        panel_limits[name.lower()] = float(value) * 1e-12

    rng = np.random.default_rng(args.seed)
    plt.rcParams.update(AGU_STYLE)
    # Panel labels match the LaTeX tables, which call dr1 "warm-start".
    variants = {
        "core": [(0, "Core")],
        "adaptive": [(1, "Warm-start")],
        "both": [(0, "Core"), (1, "Warm-start")],
    }[args.variant]
    nrows = len(variants)
    fig, axes = plt.subplots(nrows, 3, figsize=(7.5, 2.9 if nrows == 1 else 5.4),
                             squeeze=False)

    tags = "abcdef"
    for col, (reg, label) in enumerate(REGIMES):
        for row, (dr, kind) in enumerate(variants):
            tag = f"dr{dr}_{reg}_h{args.horizon}"
            f = args.run / tag / f"predictions_{tag}.pkl"
            ax = axes[row][col]
            if not f.is_file():
                ax.text(.5, .5, "not available", ha="center", va="center",
                        transform=ax.transAxes, color="0.5", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            # With one variant the model name is the same in every panel, so it
            # belongs in the caption; the title then carries only the period.
            title = (f"({tags[col]}) {label}" if nrows == 1
                     else f"({tags[row * 3 + col]}) {kind} — {label}")
            n = panel(ax, pd.read_pickle(f), title,
                      args.max_points, rng, args.scale, args.clip,
                      panel_limits.get(tags[row * 3 + col], limits.get(reg)))
            print(f"  {tag:26s} {n:>10,} samples")

    for ax in axes[-1]:
        ax.set_xlabel(r"Observed density  [10$^{-12}$ kg m$^{-3}$]")
    for row in axes:
        row[0].set_ylabel(r"Modeled density  [10$^{-12}$ kg m$^{-3}$]")

    # No figure title: the horizon and dataset belong in the LaTeX caption, and
    # a suptitle duplicates them in the printed figure.
    fig.tight_layout(w_pad=1.4, h_pad=1.6)

    suffix = "" if args.scale == "linear" else f"_{args.scale}"
    variant_suffix = "" if args.variant == "both" else f"_{args.variant}"
    out = args.out or (ROOT / f"parity{variant_suffix}_h{args.horizon}{suffix}")
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"))
        print(f"wrote {out.with_suffix('.' + ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
