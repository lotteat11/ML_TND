# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
plot_msis_residuals.py — AGU-style diagnostic of the MSIS log-residual.

Characterises r = log(rho_obs / rho_msis), the model's training target, over the
full-mission window: where it is centred, how far it departs from normality, and
how much of it a single constant offset would account for.

Prints the moments, normality tests, and per-year / per-ap breakdowns, then
draws four panels:

  (a) histogram of r with mean/median marked and a same-moment normal overlaid,
      showing the offset and the departure from normality together;
  (b) Q-Q plot against a normal, where the heavy left tail shows;
  (c) yearly median of r against F10.7, resolving the offset by solar activity;
  (d) median of r binned by ap, resolving the offset by geomagnetic forcing.

Panels (c) and (d) show how much the offset varies with geophysical state, as
opposed to sitting at a fixed scale factor: a single constant removes only a
fifth of the mean-square residual.

Usage:
    python CoreModel/plot_msis_residuals.py
    python CoreModel/plot_msis_residuals.py --parquet grace_data_merged_v5_full.parquet
    python CoreModel/plot_msis_residuals.py --out figs/msis_residuals
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = ROOT / "grace_data_merged_v5_full.parquet"
DEFAULT_OUT = ROOT / "figs" / "msis_residuals"

# Mirrors the training window in config.py: the full-mission model trains on
# 2002-2016 with the quiet-2009 test period held out.
TIME_MIN = pd.Timestamp("2002-01-01")
TIME_MAX = pd.Timestamp("2016-01-01")
TIME_EXCLUDE = [(pd.Timestamp("2009-01-01"), pd.Timestamp("2009-06-06"))]

NEEDED = ["grace_time", "rho_obs", "msis_rho", "f107", "ap_m3h", "alt_km"]

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

C_HIST = "#2e8b57"
C_MEAN = "#d1242f"
C_MED  = "#1f6feb"
C_REF  = "#57606a"

AP_BINS = [0, 10, 20, 40, 80, 400]


def load(parquet: Path) -> pd.DataFrame:
    """Stream the merged file and return the log-residual plus its predictors.

    The full-mission file is 82.5M rows; reading it whole costs ~20 GB in
    pandas, so batches are filtered down before being concatenated.
    """
    available = pq.ParquetFile(parquet).schema.names
    cols = [c for c in NEEDED if c in available]
    chunks = []
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=2_000_000,
                                                      columns=cols):
        d = batch.to_pandas()
        d["time"] = pd.to_datetime(d["grace_time"])
        if getattr(d["time"].dt, "tz", None) is not None:
            d["time"] = d["time"].dt.tz_localize(None)
        d = d[(d["time"] > TIME_MIN) & (d["time"] < TIME_MAX)]
        for lo, hi in TIME_EXCLUDE:
            d = d[(d["time"] < lo) | (d["time"] >= hi)]
        d = d[(d["rho_obs"] > 0) & (d["msis_rho"] > 0)]
        d = d.dropna(subset=["rho_obs", "msis_rho"])
        if len(d) == 0:
            continue
        d["log_ratio"] = np.log(d["rho_obs"].astype(float)
                                / d["msis_rho"].astype(float))
        chunks.append(d[["time", "log_ratio", "f107", "ap_m3h", "alt_km"]])
    return pd.concat(chunks, ignore_index=True)


def stats_table(df: pd.DataFrame) -> dict:
    r = df["log_ratio"].to_numpy()
    mean, median = float(np.mean(r)), float(np.median(r))
    # Share of the mean-square residual a single constant offset would remove.
    const_share = mean ** 2 / float(np.mean(r ** 2))
    return {
        "n":           len(r),
        "mean":        mean,
        "median":      median,
        "sd":          float(np.std(r)),
        "skew":        float(stats.skew(r)),
        "exkurt":      float(stats.kurtosis(r)),
        "mean_factor": float(np.exp(mean)),
        "const_share": const_share,
    }


def panel_hist(ax, df, s):
    r = df["log_ratio"].to_numpy()
    lo, hi = np.quantile(r, [0.001, 0.999])
    ax.hist(r, bins=200, range=(lo, hi), density=True,
            color=C_HIST, alpha=0.75, linewidth=0)

    # A normal with the same mean and sd is what a log-normal density would
    # imply; the gap between it and the histogram is the departure from it.
    x = np.linspace(lo, hi, 400)
    ax.plot(x, stats.norm.pdf(x, s["mean"], s["sd"]),
            color=C_REF, linewidth=0.9, linestyle="--",
            label="normal fit")

    ax.axvline(0.0, color="black", linewidth=0.7)
    ax.axvline(s["mean"], color=C_MEAN, linewidth=0.9,
               label=f"mean {s['mean']:+.3f}")
    ax.axvline(s["median"], color=C_MED, linewidth=0.9, linestyle=":",
               label=f"median {s['median']:+.3f}")

    ax.set_xlabel(r"$\log(\rho_{\mathrm{obs}}\,/\,\rho_{\mathrm{MSIS}})$")
    ax.set_ylabel("density")
    ax.set_title(f"(a) Residual distribution  (n = {s['n']:,})", loc="left")
    ax.text(0.02, 0.60,
            f"skew {s['skew']:+.2f}\nex. kurt {s['exkurt']:+.2f}\n"
            f"MSIS $\\times${s['mean_factor']:.3f}",
            transform=ax.transAxes, va="top", fontsize=6.5, color=C_REF)
    ax.legend(loc="upper left", frameon=False)


def panel_qq(ax, df, s, rng, n=20000):
    r = df["log_ratio"].to_numpy()
    sub = rng.choice(r, size=min(n, len(r)), replace=False)
    z = (sub - s["mean"]) / s["sd"]
    z.sort()
    q = stats.norm.ppf((np.arange(len(z)) + 0.5) / len(z))

    ax.plot(q, z, ".", markersize=1.2, color=C_HIST, alpha=0.5)
    lim = [min(q[0], z[0]) - 0.3, max(q[-1], z[-1]) + 0.3]
    ax.plot(lim, lim, color=C_REF, linewidth=0.8, linestyle="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("normal quantile")
    ax.set_ylabel("standardised residual")
    ax.set_title("(b) Q–Q against a normal", loc="left")
    ax.text(0.04, 0.94,
            "left tail heavier\nthan log-normal",
            transform=ax.transAxes, va="top", fontsize=6.5, color=C_REF)


def panel_solar(ax, df):
    g = df.groupby(df["time"].dt.year).agg(
        med=("log_ratio", "median"), f107=("f107", "mean"))
    g = g.dropna()

    # Join in time order first: the trajectory shows the offset following the
    # cycle down into the 2009 minimum and back out, which the scatter alone
    # does not make obvious.
    gt = g.sort_index()
    ax.plot(gt["f107"], gt["med"], "-", color=C_REF, linewidth=0.6,
            alpha=0.5, zorder=2)
    sc = ax.scatter(g["f107"], g["med"], c=g.index, cmap="viridis",
                    s=18, zorder=3, linewidth=0)
    for yr, row in g.iterrows():
        ax.annotate(str(yr)[2:], (row["f107"], row["med"]),
                    textcoords="offset points", xytext=(4.5, 1.5),
                    fontsize=5.5, color=C_REF, zorder=4)
    ax.axhline(0.0, color="black", linewidth=0.7)

    ax.set_xlabel("yearly mean F10.7 [sfu]")
    ax.set_ylabel("median log-residual")
    ax.set_title("(c) Offset varies with solar activity", loc="left")
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("year", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.6)


def panel_ap(ax, df):
    labels, meds, los, his = [], [], [], []
    for lo, hi in zip(AP_BINS[:-1], AP_BINS[1:]):
        g = df[(df["ap_m3h"] >= lo) & (df["ap_m3h"] < hi)]["log_ratio"]
        if len(g) < 1000:
            continue
        labels.append(f"{lo}–{hi}" if hi < AP_BINS[-1] else f">{lo}")
        meds.append(g.median())
        q25, q75 = g.quantile([0.25, 0.75])
        los.append(g.median() - q25)
        his.append(q75 - g.median())

    x = np.arange(len(labels))
    # IQR is drawn faint: the spread is wide in every bin, and the point of the
    # panel is the trend in the medians, not the scatter within a bin.
    ax.errorbar(x, meds, yerr=[los, his], fmt="none",
                ecolor=C_REF, elinewidth=0.6, capsize=2, alpha=0.35)
    ax.plot(x, meds, "-o", markersize=3.5, linewidth=0.9, color=C_MED)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"$a_p$ (3 h before sample)")
    ax.set_ylabel("median log-residual")
    ax.set_title("(d) Offset varies with geomagnetic forcing", loc="left")
    ax.text(0.04, 0.10, "MSIS over-estimates\n(quiet)",
            transform=ax.transAxes, fontsize=6, color=C_REF)
    ax.text(0.62, 0.86, "MSIS under-\nestimates (storm)",
            transform=ax.transAxes, fontsize=6, color=C_REF)


def print_report(df, s):
    print(f"\nrows: {s['n']:,}")
    print(f"mean   {s['mean']:+.4f}  (MSIS x{s['mean_factor']:.3f}, "
          f"{100 * (s['mean_factor'] - 1):+.1f}%)")
    print(f"median {s['median']:+.4f}  (x{np.exp(s['median']):.3f})")
    print(f"sd {s['sd']:.4f}   skew {s['skew']:+.3f}   "
          f"excess kurtosis {s['exkurt']:+.3f}")
    print(f"share of mean-square residual a constant would remove: "
          f"{s['const_share']:.3f}")

    rng = np.random.default_rng(0)
    r = df["log_ratio"].to_numpy()
    sub = rng.choice(r, size=min(5000, len(r)), replace=False)
    sh = stats.shapiro(sub)
    k2 = stats.normaltest(rng.choice(r, size=min(200_000, len(r)),
                                     replace=False))
    print(f"Shapiro-Wilk (n=5000): W={sh.statistic:.4f} p={sh.pvalue:.3g}")
    print(f"D'Agostino K2 (n=200k): {k2.statistic:.1f} p={k2.pvalue:.3g}")

    print("\nyearly median:")
    for yr, g in df.groupby(df["time"].dt.year):
        m = g["log_ratio"].median()
        print(f"  {yr}  {m:+.4f}  (x{np.exp(m):.3f})  n={len(g):,}")

    print("\nmedian by ap:")
    for lo, hi in zip(AP_BINS[:-1], AP_BINS[1:]):
        g = df[(df["ap_m3h"] >= lo) & (df["ap_m3h"] < hi)]["log_ratio"]
        if len(g) < 1000:
            continue
        print(f"  ap {lo:>3}-{hi:<4} {g.median():+.4f}  "
              f"(x{np.exp(g.median()):.3f})  n={len(g):,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output stem; .png and .pdf are both written.")
    args = ap.parse_args()

    print(f"reading {args.parquet}")
    df = load(args.parquet)
    s = stats_table(df)
    print_report(df, s)

    plt.rcParams.update(AGU_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    rng = np.random.default_rng(0)
    panel_hist(axes[0, 0], df, s)
    panel_qq(axes[0, 1], df, s, rng)
    panel_solar(axes[1, 0], df)
    panel_ap(axes[1, 1], df)
    fig.tight_layout(pad=0.8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = args.out.with_suffix(f".{ext}")
        fig.savefig(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
