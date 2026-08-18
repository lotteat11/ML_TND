# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
make_table_storm.py
Reports the March-2015 G4 storm evaluation from runs_v6_storm/, and compares the
2002 / quiet-2009 control regimes against the earlier run to show the extra
interior holdout did not change them.

The rolling protocol needs a lead-in (5 fine-tuning days) and therefore runs
over the whole 1 Mar - 15 Apr 2015 holdout window, but the storm itself is
reported on the G4 main-phase days only (17-18 March by default). Those rows
are sliced from the saved predictions -- no re-run needed.

    python Forecast/make_table_storm.py                  # storm + controls
    python Forecast/make_table_storm.py --days 2015-03-17,2015-03-18
    python Forecast/make_table_storm.py --daily          # per-day breakdown
"""

import argparse
import os

import numpy as np
import pandas as pd

RUNS_STORM = "runs_v6_storm"
RUNS_REF   = "runs_full_pretune"          # earlier run, for the control check
STORM_DAYS = ["2015-03-17", "2015-03-18"]

METRIC_ROWS = [
    ("RMSE_log",       "rmse_log", "log"),
    ("Top5_log",       "top5_log", "log"),
    ("RMSE (kg m^-3)", "rmse",     "sci"),
    ("MAPE (%)",       "mape_pct", "flt"),
    ("R2",             "r2",       "flt"),
]


def metrics(df, pred_col="rho_pred", obs_col="rho_obs"):
    """Same definitions as on_track.py's compute_metrics."""
    y, yhat = df[obs_col].values, df[pred_col].values
    err = yhat - y
    abs_err = np.abs(err)
    thr = np.quantile(abs_err, 0.95)
    mask = y > 0
    lmask = (y > 0) & (yhat > 0)
    lerr = np.log(yhat[lmask]) - np.log(y[lmask])
    lthr = np.quantile(np.abs(lerr), 0.95) if lerr.size else np.nan
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": len(df),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "top5": float(np.sqrt(np.mean(err[abs_err >= thr] ** 2))),
        "mape_pct": float(np.mean(abs_err[mask] / y[mask]) * 100) if mask.any() else np.nan,
        "r2": 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else np.nan,
        "rmse_log": float(np.sqrt(np.mean(lerr ** 2))) if lerr.size else np.nan,
        "top5_log": float(np.sqrt(np.mean(lerr[np.abs(lerr) >= lthr] ** 2))) if lerr.size else np.nan,
    }


def fmt(v, kind):
    if not np.isfinite(v):
        return "-"
    if kind == "sci":
        return f"{v:.3e}"
    if kind == "log":
        return f"{v:.3f} (x{np.exp(v):.2f})"
    return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"


def pct(base, val):
    if not np.isfinite(base) or base == 0 or not np.isfinite(val):
        return "-"
    return f"{100.0 * (val - base) / abs(base):+.1f}%"


def load_run(root, tag):
    p = os.path.join(root, tag, f"predictions_{tag}.pkl")
    return pd.read_pickle(p) if os.path.exists(p) else None


def find_run_dirs():
    """Directories holding storm2015 runs, newest first."""
    found = []
    for d in sorted(os.listdir(".")):
        if os.path.isdir(d) and d.startswith("runs"):
            hits = [t for t in os.listdir(d)
                    if t.startswith("dr") and "storm2015" in t
                    and os.path.exists(os.path.join(d, t, f"predictions_{t}.pkl"))]
            if hits:
                found.append((d, len(hits), os.path.getmtime(d)))
    return sorted(found, key=lambda x: -x[2])


def choose_run_dir(default=None):
    """Ask which run directory to report on; Enter accepts the first/default."""
    dirs = find_run_dirs()
    if not dirs:
        raise SystemExit("No run directory with storm2015 predictions found.")
    if len(dirs) == 1:
        print(f"Using {dirs[0][0]} ({dirs[0][1]} storm runs)")
        return dirs[0][0]

    names = [d for d, _, _ in dirs]
    pick = default if default in names else names[0]
    print("\nAvailable run directories:")
    for i, (d, n, _) in enumerate(dirs, 1):
        mark = " (default)" if d == pick else ""
        print(f"  {i}. {d}  — {n} storm runs{mark}")
    try:
        ans = input(f"Select [1-{len(dirs)}, Enter for {pick}]: ").strip()
    except EOFError:                      # non-interactive: take the default
        ans = ""
    if ans.isdigit() and 1 <= int(ans) <= len(dirs):
        return names[int(ans) - 1]
    return ans if ans in names else pick


def storm_table(days, horizon=1):
    print(f"\n{'='*78}")
    print(f"  MARCH 2015 G4 STORM — main phase {', '.join(days)}  (h={horizon}d)")
    print(f"{'='*78}")

    wanted = {pd.Timestamp(d).date() for d in days}
    for dr, label in [(0, "Core model (no fine-tuning)"), (1, "Warm-start (daily fine-tuning)")]:
        tag = f"dr{dr}_storm2015_h{horizon}"
        df = load_run(RUNS_STORM, tag)
        if df is None:
            print(f"\n--- {label}: {tag} not found (run the storm pipeline first) ---")
            continue
        sub = df[pd.to_datetime(df["date"]).dt.date.isin(wanted)]
        if sub.empty:
            print(f"\n--- {label}: no rows on {days} ---")
            continue
        m = metrics(sub)
        b = metrics(sub, pred_col="msis_rho")
        print(f"\n--- {label} — {m['n']:,} points ---")
        print(f"{'Metric':<16} {'NRLMSISE-2.1':>20} {'Pred':>20} {'Diff':>10}")
        for lbl, key, kind in METRIC_ROWS:
            print(f"{lbl:<16} {fmt(b[key], kind):>20} {fmt(m[key], kind):>20} {pct(b[key], m[key]):>10}")


def daily_table(horizon=1):
    print(f"\n{'='*78}")
    print(f"  DAY-BY-DAY THROUGH THE STORM WINDOW  (h={horizon}d, RMSE_log)")
    print(f"{'='*78}")
    frames = {}
    for dr in (0, 1):
        df = load_run(RUNS_STORM, f"dr{dr}_storm2015_h{horizon}")
        if df is not None:
            frames[dr] = df
    if not frames:
        print("  no storm runs found")
        return

    print(f"{'Day':<12} {'ap_max':>7} {'MSIS':>8} {'Core':>8} {'Warm':>8}")
    ref = frames[max(frames)]
    for day, g in ref.groupby(pd.to_datetime(ref["date"]).dt.date):
        ap = g["ap_m3h"].max() if "ap_m3h" in g else np.nan
        row = f"{str(day):<12} {ap:>7.0f} {metrics(g, pred_col='msis_rho')['rmse_log']:>8.3f}"
        for dr in (0, 1):
            if dr in frames:
                d = frames[dr]
                gg = d[pd.to_datetime(d["date"]).dt.date == day]
                row += f" {metrics(gg)['rmse_log']:>8.3f}" if not gg.empty else f" {'-':>8}"
            else:
                row += f" {'-':>8}"
        print(row)


def control_check(horizon=1):
    """2002 / quiet-2009 should be ~unchanged by the extra March-2015 holdout."""
    print(f"\n{'='*78}")
    print(f"  CONTROL REGIMES — storm-holdout model vs earlier run  (h={horizon}d)")
    print(f"{'='*78}")
    print(f"{'Regime':<22} {'variant':<6} {'RMSE_log ref':>13} {'RMSE_log new':>13} {'change':>9}")
    for regime in ("y2002", "quiet2009", "post2016"):
        for dr in (0, 1):
            tag = f"dr{dr}_{regime}_h{horizon}"
            new, ref = load_run(RUNS_STORM, tag), load_run(RUNS_REF, tag)
            if new is None or ref is None:
                continue
            a, b = metrics(ref)["rmse_log"], metrics(new)["rmse_log"]
            print(f"{regime:<22} {'dr'+str(dr):<6} {a:>13.4f} {b:>13.4f} {pct(a, b):>9}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default=",".join(STORM_DAYS),
                    help="comma-separated storm days to report")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--daily", action="store_true", help="per-day breakdown of the window")
    ap.add_argument("--runs", default=None,
                    help="run directory to report on (skips the prompt)")
    args = ap.parse_args()

    RUNS_STORM = args.runs or choose_run_dir(RUNS_STORM)

    storm_table(args.days.split(","), args.horizon)
    if args.daily:
        daily_table(args.horizon)
    control_check(args.horizon)
    print()