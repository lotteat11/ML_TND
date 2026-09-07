# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
plot_feature_importance.py — AGU-style feature importance figure for a trained
core model.

Reads a saved XGBoost booster and plots gain importance: the mean improvement
in the loss function contributed by each feature across the splits that use it.
This is the quantity train.py prints after fitting.

Bars are grouped by driver family (solar, geomagnetic, TEC, geometry/season)
so the figure answers which class of driver carries the correction, not just
which individual column ranks highest.

Usage:
    python CoreModel/plot_feature_importance.py
    python CoreModel/plot_feature_importance.py --model xgb_model_v3.json
    python CoreModel/plot_feature_importance.py --normalise
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "xgb_model_v8_storm_ap_2002train.json"

# Driver families. Order sets the legend order.
FAMILIES = {
    "Solar":        (["f107", "f107a"], "#e8a838"),
    "Geomagnetic":  (["ap_daily", "ap_0h", "ap_m3h", "ap_m6h", "ap_m9h",
                      "ap_avg12_33h", "ap_avg36_57h"], "#d1495b"),
    "Ionospheric (TEC)": (["matched_tec_value", "vtec_matched_lag",
                           "vtec_matched_lag2"], "#1f6feb"),
    "Geometry / season": (["lat", "lon_sin", "lon_cos", "lst_sin", "lst_cos",
                           "lst_lat_sin", "lst_lat_cos", "doy_sin", "doy_cos",
                           "alt_km"], "#6c757d"),
}

PRETTY = {
    "f107": "F10.7", "f107a": "F10.7a",
    "matched_tec_value": "VTEC (current)",
    "vtec_matched_lag": "VTEC ($t-3$ h)",
    "vtec_matched_lag2": "VTEC ($t-24$ h)",
    "ap_daily": "Ap (daily)", "ap_0h": "ap ($t$)",
    "ap_m3h": "ap ($t-3$ h)", "ap_m6h": "ap ($t-6$ h)",
    "ap_m9h": "ap ($t-9$ h)",
    "ap_avg12_33h": "ap (mean 12–33 h)",
    "ap_avg36_57h": "ap (mean 36–57 h)",
    "lat": "Latitude", "alt_km": "Altitude",
    "lon_sin": "sin(lon)", "lon_cos": "cos(lon)",
    "lst_sin": "sin(LST)", "lst_cos": "cos(LST)",
    "lst_lat_sin": "sin(LST)$\\times$lat",
    "lst_lat_cos": "cos(LST)$\\times$lat",
    "doy_sin": "sin(DOY)", "doy_cos": "cos(DOY)",
}

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
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
}


def family_of(feature):
    for name, (members, colour) in FAMILIES.items():
        if feature in members:
            return name, colour
    return "Other", "#adb5bd"


def plot_importance(scores, ax, normalise=False):
    """Horizontal bars, most important at the top, coloured by driver family."""
    items = sorted(scores.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = np.array([v for _, v in items], dtype=float)
    if normalise:
        vals = 100 * vals / vals.sum()

    colours = [family_of(n)[1] for n in names]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colours, height=0.7)
    ax.set_yticks(y, [PRETTY.get(n, n) for n in names])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    for yi, v in zip(y, vals):
        ax.text(v + vals.max() * 0.015, yi,
                f"{v:.1f}" + ("%" if normalise else ""),
                va="center", fontsize=6.5, color="0.3")

    ax.set_xlim(0, vals.max() * 1.16)
    ax.set_xlabel("Share of total gain (%)" if normalise
                  else "Gain importance")
    ax.tick_params(length=2.5, pad=2)

    present = []
    for fam, (members, colour) in FAMILIES.items():
        if any(n in members for n in names):
            present.append(mpatches.Patch(facecolor=colour, label=fam))
    ax.legend(handles=present, loc="lower right", frameon=False,
              handlelength=1.1, borderaxespad=0.6)
    return names, vals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--normalise", action="store_true",
                   help="Show each feature's share of total gain, in percent.")
    args = p.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Not found: {args.model}")
    booster = xgb.Booster()
    booster.load_model(str(args.model))
    scores = booster.get_score(importance_type="gain")
    if not scores:
        raise SystemExit("Model reports no gain importance.")

    print(f"{args.model.name}: {len(scores)} features with nonzero gain")

    plt.rcParams.update(AGU_STYLE)
    height = max(2.6, 0.23 * len(scores) + 1.0)
    fig, ax = plt.subplots(figsize=(4.6, height))
    names, vals = plot_importance(scores, ax, normalise=args.normalise)
    fig.tight_layout()

    out = args.out or (ROOT / f"feature_importance_{args.model.stem}")
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"))
        print(f"wrote {out.with_suffix('.' + ext)}")
    plt.close(fig)

    # Family totals: the individual ranking hides that several moderate ap
    # terms together can outweigh a single high-ranking column.
    total = sum(scores.values())
    print("\nshare of total gain by driver family:")
    by_fam = {}
    for k, v in scores.items():
        by_fam[family_of(k)[0]] = by_fam.get(family_of(k)[0], 0.0) + v
    for fam, v in sorted(by_fam.items(), key=lambda kv: -kv[1]):
        print(f"  {fam:20s} {100 * v / total:5.1f}%")


if __name__ == "__main__":
    main()
