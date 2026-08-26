# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
plot_skill_regimes.py — daily skill trace for all three holdout regimes.

One panel per regime, stacked: daily RMSE_log for NRLMSIS-2.1 and for the
model, over every day of the period. The along-track figures show hand-picked
hours; this one covers every day, so the days the model loses are visible too.

Each panel has its own time axis -- the three periods are years apart and a
shared axis would compress 2009 and 2015 to slivers.

    python CoreModel/plot_skill_regimes.py
    python CoreModel/plot_skill_regimes.py --dr 0 --horizon 3
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

C_PRED, C_MSIS = "#e8762c", "#3b8ede"
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

REGIMES = [("quiet2009", "2009"),
           ("storm2015", "Period 2015"),
           ("post2016",  "Post-2016")]


def rmse_log(obs, model):
    k = (obs > 0) & (model > 0)
    if k.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((np.log(obs[k]) - np.log(model[k])) ** 2)))


def daily(pkl):
    df = pd.read_pickle(pkl)
    df["day"] = pd.to_datetime(df["time"], utc=True).dt.floor("1D")
    recs = []
    for day, g in df.groupby("day", sort=True):
        o = g["rho_obs"].to_numpy()
        recs.append({"day": day,
                     "rl_pred": rmse_log(o, g["rho_pred"].to_numpy()),
                     "rl_msis": rmse_log(o, g["msis_rho"].to_numpy())})
    overall = (rmse_log(df["rho_obs"].to_numpy(), df["msis_rho"].to_numpy()),
               rmse_log(df["rho_obs"].to_numpy(), df["rho_pred"].to_numpy()))
    return pd.DataFrame(recs), overall


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs_final_20250821")
    p.add_argument("--dr", type=int, default=1, choices=(0, 1))
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--losses", action="store_true",
                   help="Also label each panel with the count of days the "
                        "model loses to MSIS.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    plt.rcParams.update(AGU_STYLE)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4))

    for ax, (reg, label) in zip(axes, REGIMES):
        tag = f"dr{args.dr}_{reg}_h{args.horizon}"
        pkl = ROOT / args.run / tag / f"predictions_{tag}.pkl"
        if not pkl.is_file():
            raise SystemExit(f"Not found: {pkl}")
        d, (m_all, p_all) = daily(pkl)

        ax.plot(d.day, d.rl_msis, color=C_MSIS, lw=1.0, ls="--",
                label="NRLMSIS-2.1")
        ax.plot(d.day, d.rl_pred, color=C_PRED, lw=1.0, label="ML model")
        worse = d.rl_pred > d.rl_msis
        if args.losses:
            ax.fill_between(d.day, 0, 1, where=worse,
                            transform=ax.get_xaxis_transform(),
                            color=C_BAD, alpha=0.12, lw=0)

        ax.set_ylabel("daily RMSE$_{log}$")
        ax.set_title(label, loc="left", pad=3)
        if ax is axes[0]:
            ax.legend(loc="upper left", frameon=False, ncol=2,
                      handlelength=1.8, borderaxespad=0.2)
        # Period skill goes in the caption, not on the panel: three annotations
        # compete with the traces they describe.
        if args.losses:
            ax.text(0.995, 0.94,
                    f"model worse than MSIS on {int(worse.sum())} of "
                    f"{len(d)} days",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color=C_BAD)
        ax.grid(True, axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, pad=2)
        # Month stamps once a period runs longer than about four months.
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m" if len(d) > 120 else "%d %b"))

        print(f"  {tag:24s} {len(d):4d} days | MSIS {m_all:.4f} | "
              f"model {p_all:.4f} | {100*(p_all/m_all-1):+5.1f}% | "
              f"worse on {int(worse.sum())}")

    axes[-1].set_xlabel("Date")
    fig.tight_layout(h_pad=1.4)

    out = Path(args.out) if args.out else (
        ROOT / "figs" / f"skill_regimes_dr{args.dr}_h{args.horizon}")
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"))
        print(f"wrote {out.with_suffix('.' + ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
