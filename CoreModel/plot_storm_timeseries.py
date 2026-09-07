# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
plot_storm_timeseries.py — along-track thermospheric neutral density for one
day, comparing the observations, NRLMSIS-2.1 and the corrected model.

Plots the raw along-track samples rather than orbit means, so the ~90-minute
day/night cycle stays visible and the curves are directly comparable with the
reported RMSE, which is also computed on raw samples.

GRACE-A and GRACE-B fly the same track minutes apart, so their samples
interleave in time. Plotting them as one series draws a vertical stripe at
every epoch; --satellite picks one (default GA).

Usage:
    python CoreModel/plot_storm_timeseries.py
    python CoreModel/plot_storm_timeseries.py --date 2015-03-18 --hours 0 7
    python CoreModel/plot_storm_timeseries.py --run runs_v13_storm_lb3_reset7 --dr 1
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
DEFAULT_RUN = ROOT / "runs_final_20250821"

C_OBS = "#1a1a1a"
C_PRED = "#e8762c"
C_MSIS = "#3b8ede"
# Second forecast lead, when one is overlaid: a purple that reads as distinct
# from the orange prediction at line width, in colour and in greyscale.
C_LEAD = "#8250c4"

AGU_STYLE = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          8,
    "axes.titlesize":     8.5,
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


def rmse_log(obs, model):
    k = (obs > 0) & (model > 0)
    return float(np.sqrt(np.mean((np.log(obs[k]) - np.log(model[k])) ** 2)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--dr", type=int, default=0, choices=(0, 1),
                   help="0 = no warm-start, 1 = with warm-start.")
    p.add_argument("--tag", default="storm2015_h1")
    p.add_argument("--date", default="2015-03-17")
    p.add_argument("--hours", type=float, nargs=2, default=(0, 7),
                   metavar=("FROM", "TO"),
                   help="UTC hour range within the day (default 0 7).")
    p.add_argument("--satellite", default="GA",
                   help="Track to plot; 'both' interleaves GA and GB.")
    p.add_argument("--days", type=float, default=None,
                   help="Span this many days from the start of --hours, "
                        "instead of ending at the second --hours value.")
    p.add_argument("--stack", action="store_true",
                   help="Draw one panel per day stacked vertically. A multi-day "
                        "window on a single axis crowds ~15 orbits per day into "
                        "one width and the curves stop being separable.")
    p.add_argument("--overlay-lead", type=int, default=None,
                   help="Draw a second forecast at this lead time alongside "
                        "--lead, so the effect of lead time is visible in one "
                        "panel (observations and MSIS do not depend on it).")
    p.add_argument("--lead", type=int, default=None,
                   help="For an h>1 run, keep only the forecasts at this lead "
                        "time in days (1 = made one day ahead). Without it "
                        "every lead is drawn, which thickens the curve.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    pkl = args.run / f"dr{args.dr}_{args.tag}" / f"predictions_dr{args.dr}_{args.tag}.pkl"
    if not pkl.is_file():
        raise SystemExit(f"Not found: {pkl}")
    df = pd.read_pickle(pkl).reset_index(drop=True)
    df["t"] = pd.to_datetime(df["time"], utc=True)

    # An h-day run stores h forecasts of every epoch, one per lead time, as
    # consecutive blocks with identical time/lat and differing rho_pred. No
    # column records the lead, so it is recovered from the block order: the
    # last block reproduces the h=1 run exactly, hence block i is lead h - i.
    extra_leads = {}
    if args.lead is not None or args.overlay_lead is not None:
        horizon = int(args.tag.rsplit("_h", 1)[1])
        full = df
        blk = full.groupby(["t", "lat"]).cumcount()

        def _block(lead):
            if not 1 <= lead <= horizon:
                raise SystemExit(f"lead must be 1..{horizon} for {args.tag}")
            # Always index the unfiltered frame: blk is aligned to it.
            return full[(blk % horizon) == (horizon - lead)]

        primary = args.lead if args.lead is not None else horizon
        df = _block(primary)
        if args.overlay_lead is not None:
            # The blocks are not equal in length: epochs near the start and end
            # of a period are covered by fewer than h rolling windows, so only
            # the interior has every lead. Pair on (t, lat, occurrence) and keep
            # the epochs both leads cover, rather than aligning by position.
            def _keyed(frame):
                k = frame.groupby(["t", "lat"]).cumcount()
                return frame.assign(_k=k).set_index(["t", "lat", "_k"])

            a, b = _keyed(df), _keyed(_block(args.overlay_lead))
            common = a.index.intersection(b.index)
            dropped = len(a) - len(common)
            if dropped:
                print(f"  overlay: {dropped:,} epochs lack a lead "
                      f"{args.overlay_lead} forecast and are dropped")
            df = a.loc[common].copy()
            df[f"rho_pred_lead{args.overlay_lead}"] = b.loc[common, "rho_pred"]
            df = df.reset_index()
            extra_leads[args.overlay_lead] = None
        args.lead = primary

    day = pd.Timestamp(args.date, tz="UTC")
    lo = day + pd.Timedelta(hours=args.hours[0])
    hi = (lo + pd.Timedelta(days=args.days) if args.days
          else day + pd.Timedelta(hours=args.hours[1]))
    d = df[(df["t"] >= lo) & (df["t"] < hi)].sort_values("t")
    if d.empty:
        raise SystemExit(f"No rows in {lo} .. {hi}")

    # Predictions carry no 'source' column, so split the interleaved tracks on
    # the sampling gap instead: consecutive samples from one satellite are 10 s
    # apart, and the two tracks alternate.
    if args.satellite.lower() != "both" and len(d) > 1:
        dt = d["t"].diff().dt.total_seconds()
        d = d[(dt.isna()) | (dt >= 5)]

    obs = d["rho_obs"].to_numpy()
    scores = {"MSIS": rmse_log(obs, d["msis_rho"].to_numpy()),
              "Corrected": rmse_log(obs, d["rho_pred"].to_numpy())}
    for lead in extra_leads:
        col = d[f"rho_pred_lead{lead}"]
        scores[f"lead {lead} d"] = rmse_log(obs[col.notna().to_numpy()],
                                            col.dropna().to_numpy())
    print(f"{pkl}\n  {len(d):,} samples, {d.t.min()} .. {d.t.max()}")
    for k, v in scores.items():
        print(f"  RMSE(log) {k:10s} {v:.4f}  (factor {np.exp(v):.2f})")
    print(f"  improvement {100 * (scores['Corrected'] / scores['MSIS'] - 1):+.1f}%")

    plt.rcParams.update(AGU_STYLE)

    # A multi-day window is named by its span, not by the first day alone.
    last = hi - pd.Timedelta(seconds=1)
    period = (f"{day:%d %B %Y}" if last.date() == day.date()
              else f"{day:%d}--{last:%d %B %Y}" if day.month == last.month
              else f"{day:%d %b} -- {last:%d %b %Y}")

    # One panel per day when stacking, otherwise the whole window on one axis.
    # Panels start at the window's own start hour, not at midnight, so each
    # covers a full 24 h; splitting on calendar days would leave the first and
    # last panels short whenever the window starts mid-day.
    if args.stack:
        edges = pd.date_range(lo, hi, freq="D")
        spans = [(s, min(hi, s + pd.Timedelta(days=1)))
                 for s in edges if s < hi]
    else:
        spans = [(lo, hi)]

    fig, axes = plt.subplots(len(spans), 1, squeeze=False,
                             figsize=(7.0, 2.9 if len(spans) == 1
                                      else 1.75 * len(spans) + 0.6))
    axes = axes[:, 0]

    # Shared y-limits keep the panels comparable; without them each day is
    # autoscaled and a quiet day looks as active as a storm.
    _ycols = ["rho_obs", "rho_pred", "msis_rho"] + [
        f"rho_pred_lead{lead}" for lead in extra_leads]
    ymax = max(d[c].max() for c in _ycols)
    ymin = min(d[c].min() for c in _ycols)

    for i, (ax, (s, e)) in enumerate(zip(axes, spans)):
        w = d[(d["t"] >= s) & (d["t"] < e)]
        # Scores stay on stdout (and in the caption) rather than in the legend:
        # the labels are read at figure size, where a 3-decimal metric is noise.
        ax.plot(w["t"], w["rho_obs"], color=C_OBS, lw=0.7, label="Observed",
                zorder=4)
        pred_label = ("ML model" if not extra_leads
                      else f"ML model ({args.lead} d lead)")
        ax.plot(w["t"], w["rho_pred"], color=C_PRED, lw=1.0, label=pred_label,
                zorder=3)
        for lead in extra_leads:
            ax.plot(w["t"], w[f"rho_pred_lead{lead}"], color=C_LEAD, lw=1.0,
                    label=f"ML model ({lead} d lead)", zorder=3)
        ax.plot(w["t"], w["msis_rho"], color=C_MSIS, lw=1.0, ls="--",
                label="NRLMSIS-2.1", zorder=2)

        ax.set_ylabel(r"$\rho$  [kg m$^{-3}$]")
        ax.set_ylim(ymin - 0.04 * (ymax - ymin), ymax + 0.26 * (ymax - ymin))
        # Headroom above carries the legend, so only the top panel needs one.
        if i == 0:
            ax.legend(loc="upper right", frameon=False, handlelength=1.8,
                      ncol=3 + len(extra_leads), columnspacing=1.2,
                      borderaxespad=0.2)
        ax.tick_params(length=2.5, pad=2)
        ax.grid(True, axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        # Hourly ticks are unreadable beyond about a day, so widen the interval
        # and date-stamp the ticks once a panel spans more than one.
        span_h = (e - s).total_seconds() / 3600
        step = max(1, int(round(span_h / 12)))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(
            "%d %b\n%H:%M" if span_h > 26 else "%H:%M"))
        ax.set_xlim(s, e)
        if args.stack:
            # Each panel is one day, so name it in the panel and label the
            # x-axis once, under the bottom one.
            e_last = e - pd.Timedelta(seconds=1)
            ax.set_title(f"{s:%d %B %Y}" if e_last.date() == s.date()
                         else f"{s:%d}--{e_last:%d %B %Y}", loc="left", pad=3)
            if i == len(spans) - 1:
                ax.set_xlabel("UTC time")
        else:
            ax.set_xlabel("UTC time" if last.date() != day.date()
                          else f"UTC time ({day:%d %b %Y})")

    fig.suptitle("Along-track thermospheric neutral density — "
                 f"{period}", y=0.995 if args.stack else 1.0)
    if args.stack:
        fig.tight_layout(h_pad=1.2, rect=(0, 0, 1, 0.985))

    out = args.out or (ROOT / f"alongtrack_{args.date}_dr{args.dr}")
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"))
        print(f"wrote {out.with_suffix('.' + ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
