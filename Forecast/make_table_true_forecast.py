# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
make_table_true_forecast.py
Formats runs_true_forecast/summary_metrics_true.csv into Table 3/4-style rows
for the FULLY OPERATIONAL forecast experiment (persisted drivers for both the
MSIS scaffold and the ML correction).

Columns per scenario:
  MSIS(perfect)  : stored msis_rho, observed drivers        (upper bound)
  MSIS(persist)  : MSIS re-run with issue-time drivers      (operational baseline)
  Pred(obsMSIS)  : msis_rho * exp(corr_persisted)           (= on_track_persisted.py)
  Pred(true)     : msis_persist * exp(corr_persisted)       (operational forecast)
  Diff column    : Pred(true) vs MSIS(persist) — the apples-to-apples comparison.

Run from the repo root after on_track_true_forecast.py has finished (it can
also be run while the 8 runs are still in progress — it formats whatever rows
exist):

    python Forecast/make_table_true_forecast.py
"""

import os
import numpy as np
import pandas as pd

SUMMARY_CSV = os.path.join("runs_true_forecast", "summary_metrics_true.csv")

SCENARIO_NAMES = {
    ("pre2009", 0):  "Forecast (quiet): Training on core [true forecast]",
    ("pre2009", 1):  "Forecast (quiet): Warm-start [true forecast]",
    ("post2016", 0): "Forecast (disturbed): Training on core [true forecast]",
    ("post2016", 1): "Forecast (disturbed): Warm-start [true forecast]",
}

# (label, metric column suffix, format)
METRIC_ROWS = [
    ("RMSE_log",        "rmse_log", "log"),
    ("Top5_log",        "top5_log", "log"),
    ("RMSE (kg m^-3)",  "rmse",     "sci"),
    ("Top5 (kg m^-3)",  "top5",     "sci"),
    ("MAPE (%)",        "mape_pct", "flt"),
    ("R2",              "r2",       "flt"),
]


def fmt(value: float, kind: str) -> str:
    if kind == "sci":
        return f"{value:.3e}"
    if kind == "log":
        return f"{value:.3f} (x{np.exp(value):.2f})"
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"


def diff_pct(base: float, pred: float) -> str:
    if base == 0 or not np.isfinite(base) or not np.isfinite(pred):
        return "-"
    return f"{100.0 * (pred - base) / abs(base):+.1f}%"


def main():
    if not os.path.exists(SUMMARY_CSV):
        raise SystemExit(f"Not found: {SUMMARY_CSV} — run Forecast/on_track_true_forecast.py first.")
    df = pd.read_csv(SUMMARY_CSV)

    for horizon in sorted(df["horizon_days"].unique()):
        print(f"\n{'='*100}")
        print(f"  {horizon}-DAY FORECASTS — TRUE FORECAST (PERSISTED DRIVERS INCL. MSIS SCAFFOLD)")
        print(f"{'='*100}")
        for (dfilt, dr), name in SCENARIO_NAMES.items():
            row = df[(df["date_filter"] == dfilt)
                     & (df["do_retrain"] == dr)
                     & (df["horizon_days"] == horizon)]
            if row.empty:
                continue
            r = row.iloc[0]
            print(f"\n--- {name} ---")
            print(f"{'Metric':<18} {'MSIS(perfect)':>20} {'MSIS(persist)':>20} "
                  f"{'Pred(obsMSIS)':>20} {'Pred(true)':>20} {'vs persist':>11}")
            for label, suffix, kind in METRIC_ROWS:
                msis_obs = r[f"msis_{suffix}"]
                msis_per = r[f"msisp_{suffix}"]
                pred_obs = r[f"predobs_{suffix}"]
                pred     = r[suffix]
                print(f"{label:<18} {fmt(msis_obs, kind):>20} {fmt(msis_per, kind):>20} "
                      f"{fmt(pred_obs, kind):>20} {fmt(pred, kind):>20} "
                      f"{diff_pct(msis_per, pred):>11}")


if __name__ == "__main__":
    main()
