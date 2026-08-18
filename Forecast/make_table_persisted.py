# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
make_table_persisted.py
Formats runs_persisted/summary_metrics_persisted.csv into Table 3/4-style rows
(NRLMSISE-2.1 vs Pred with percentage differences) for the persisted-driver
forecast experiment.

Run from the repo root after on_track_persisted.py has finished (it can also be
run while the 8 runs are still in progress — it formats whatever rows exist):

    python Forecast/make_table_persisted.py
"""

import os
import numpy as np
import pandas as pd

SUMMARY_CSV = os.path.join("runs_persisted", "summary_metrics_persisted.csv")

SCENARIO_NAMES = {
    ("pre2009", 0):  "Forecast (quiet): Training on core [persisted drivers]",
    ("pre2009", 1):  "Forecast (quiet): Warm-start [persisted drivers]",
    ("post2016", 0): "Forecast (disturbed): Training on core [persisted drivers]",
    ("post2016", 1): "Forecast (disturbed): Warm-start [persisted drivers]",
}

# (label, msis column, pred column, format, lower_is_better)
METRIC_ROWS = [
    ("RMSE_log",        "msis_rmse_log", "rmse_log", "log",  True),
    ("Top5_log",        "msis_top5_log", "top5_log", "log",  True),
    ("RMSE (kg m^-3)",  "msis_rmse",     "rmse",     "sci",  True),
    ("Top5 (kg m^-3)",  "msis_top5",     "top5",     "sci",  True),
    ("MAPE (%)",        "msis_mape_pct", "mape_pct", "flt",  True),
    ("R2",              "msis_r2",       "r2",       "flt",  False),
]


def fmt(value: float, kind: str) -> str:
    if kind == "sci":
        return f"{value:.3e}"
    if kind == "log":
        return f"{value:.3f} (x{np.exp(value):.2f})"
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"


def diff_pct(msis: float, pred: float) -> str:
    if msis == 0 or not np.isfinite(msis) or not np.isfinite(pred):
        return "-"
    return f"{100.0 * (pred - msis) / abs(msis):+.1f}%"


def main():
    if not os.path.exists(SUMMARY_CSV):
        raise SystemExit(f"Not found: {SUMMARY_CSV} — run Forecast/on_track_persisted.py first.")
    df = pd.read_csv(SUMMARY_CSV)

    for horizon in sorted(df["horizon_days"].unique()):
        print(f"\n{'='*72}")
        print(f"  {horizon}-DAY FORECASTS — PERSISTED DRIVERS")
        print(f"{'='*72}")
        for (dfilt, dr), name in SCENARIO_NAMES.items():
            row = df[(df["date_filter"] == dfilt)
                     & (df["do_retrain"] == dr)
                     & (df["horizon_days"] == horizon)]
            if row.empty:
                continue
            r = row.iloc[0]
            print(f"\n--- {name} ---")
            print(f"{'Metric':<18} {'NRLMSISE-2.1':>20} {'Pred':>20} {'Diff (%)':>10}")
            for label, mcol, pcol, kind, _ in METRIC_ROWS:
                print(f"{label:<18} {fmt(r[mcol], kind):>20} {fmt(r[pcol], kind):>20} "
                      f"{diff_pct(r[mcol], r[pcol]):>10}")


if __name__ == "__main__":
    main()
