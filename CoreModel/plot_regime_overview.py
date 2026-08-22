# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
plot_regime_overview.py — whole-period overview for one evaluation regime.

The along-track panels show hand-picked hours and are illustrative; this plot
is the representative counterpart: it covers EVERY day of a holdout period, so
the good and bad days are both visible.

Three stacked panels sharing a time axis:
  (a) daily median density -- observed, predicted, MSIS, with the observed
      10-90 percentile band for spread. Medians, not means: density is
      log-distributed and a few storm samples otherwise drag the mean.
  (b) daily RMSE_log for MSIS and the prediction -- the skill trace. Days where
      the prediction is worse than MSIS are shaded.
  (c) daily max ap, for context on what drove those days.

    python CoreModel/plot_regime_overview.py --regime quiet2009
    python CoreModel/plot_regime_overview.py --regime storm2015 --dr 1
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

C_OBS, C_PRED, C_MSIS = "#1a1a1a", "#e8762c", "#3b8ede"
C_BAD = "#d94f4f"

AGU_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

TITLES = {
    "quiet2009": "Quiet 2009 (deep solar minimum)",
    "storm2015": "March 2015 G4 storm window",
    "post2016":  "Post-2016 holdout",
}


def rmse_log(obs, model):
    k = (obs > 0) & (model > 0)
    if k.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((np.log(obs[k]) - np.log(model[k])) ** 2)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs_final_20250821")
    p.add_argument("--regime", required=True,
                   choices=("quiet2009", "storm2015", "post2016"))
    p.add_argument("--dr", type=int, default=1, choices=(0, 1))
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tag = f"dr{args.dr}_{args.regime}_h{args.horizon}"
    pkl = ROOT / args.run / tag / f"predictions_{tag}.pkl"
    if not pkl.is_file():
        raise SystemExit(f"Not found: {pkl}")

    df = pd.read_pickle(pkl)
    df["t"] = pd.to_datetime(df["time"], utc=True)
    df["day"] = df["t"].dt.floor("1D")

    # Daily aggregates. One pass per day rather than several groupbys, so the
    # 6M-row post-2016 frame is only traversed once.
    recs = []
    for day, g in df.groupby("day", sort=True):
        o = g["rho_obs"].to_numpy()
        recs.append({
            "day": day,
            "obs": np.median(o),
            "pred": np.median(g["rho_pred"].to_numpy()),
            "msis": np.median(g["msis_rho"].to_numpy()),
            "lo": np.percentile(o, 10), "hi": np.percentile(o, 90),
            "rl_pred": rmse_log(o, g["rho_pred"].to_numpy()),
            "rl_msis": rmse_log(o, g["msis_rho"].to_numpy()),
            "ap": g["ap_m3h"].max() if "ap_m3h" in g else np.nan,
        })
    d = pd.DataFrame(recs)

    overall_p = rmse_log(df["rho_obs"].to_numpy(), df["rho_pred"].to_numpy())
    overall_m = rmse_log(df["rho_obs"].to_numpy(), df["msis_rho"].to_numpy())
    variant = "warm-start" if args.dr else "core model"

    plt.rcParams.update(AGU_STYLE)
    fig, (a1, a2, a3) = plt.subplots(
        3, 1, figsize=(7.2, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.3, 0.9], "hspace": 0.16})

    # (a) daily median density
    a1.fill_between(d.day, d.lo, d.hi, color=C_OBS, alpha=0.12, lw=0,
                    label="Observed 10–90 %")
    a1.plot(d.day, d.obs,  color=C_OBS,  lw=0.9, label="Observed", zorder=4)
    a1.plot(d.day, d.pred, color=C_PRED, lw=1.1, label="Prediction", zorder=3)
    a1.plot(d.day, d.msis, color=C_MSIS, lw=1.1, ls="--", label="MSIS", zorder=2)
    a1.set_yscale("log")
    a1.set_ylabel(r"daily median $\rho$  [kg m$^{-3}$]")
    a1.legend(loc="upper left", frameon=False, ncol=4, handlelength=1.8,
              borderaxespad=0.2)
    a1.set_title(f"{TITLES[args.regime]} — {variant}, h = {args.horizon} d "
                 f"(RMSE$_{{log}}$: MSIS {overall_m:.3f} to model {overall_p:.3f}, "
                 f"{100*(overall_p/overall_m-1):+.0f}%)", pad=6)

    # (b) daily skill
    a2.plot(d.day, d.rl_msis, color=C_MSIS, lw=1.0, ls="--", label="MSIS")
    a2.plot(d.day, d.rl_pred, color=C_PRED, lw=1.0, label="Prediction")
    worse = d.rl_pred > d.rl_msis
    a2.fill_between(d.day, 0, 1, where=worse, transform=a2.get_xaxis_transform(),
                    color=C_BAD, alpha=0.12, lw=0)
    a2.set_ylabel("daily RMSE$_{log}$")
    a2.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.8,
              borderaxespad=0.2)
    a2.text(0.995, 0.92,
            f"model worse than MSIS on {int(worse.sum())} of {len(d)} days",
            transform=a2.transAxes, ha="right", va="top", fontsize=6.5,
            color=C_BAD)

    # (c) ap context
    a3.fill_between(d.day, 0, d.ap, color="#6b6b6b", alpha=0.45, lw=0)
    a3.set_ylabel("daily max $a_p$")
    a3.set_xlabel("Date")

    for ax in (a1, a2, a3):
        ax.grid(True, axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, pad=2)
    a3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"
                                 if len(d) > 120 else "%d %b"))
    fig.autofmt_xdate(rotation=0, ha="center")

    out = Path(args.out) if args.out else ROOT / "figs" / f"overview_{args.regime}_dr{args.dr}"
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"))
        print(f"wrote {out.with_suffix('.' + ext)}")
    plt.close(fig)

    print(f"  {len(d)} days | MSIS {overall_m:.4f} | model {overall_p:.4f} "
          f"({100*(overall_p/overall_m-1):+.1f}%) | "
          f"model worse on {int(worse.sum())} days")


if __name__ == "__main__":
    main()
