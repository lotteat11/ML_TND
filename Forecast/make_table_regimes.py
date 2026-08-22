# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
make_table_regimes.py
Builds the per-regime results table (NRLMSISE-2.1 vs Predicted, with % diff)
for the rolling out-of-sample runs, as LaTeX and as plain text.

One block per regime x variant:
    <Regime> (Core)        -> dr0, core model only
    <Regime> (Warm-start)  -> dr1, warm-start with daily fine-tuning

Reads the run's summary_metrics.csv, which already carries both the model and
the paired msis_* baseline columns, so nothing is recomputed here.

    python Forecast/make_table_regimes.py --run runs_test_lb3
    python Forecast/make_table_regimes.py --run runs_v14_report_h1h3 --horizon 1
    python Forecast/make_table_regimes.py --run runs_test_lb3 --out table_lb3.tex
"""

import argparse
import os

import pandas as pd

# (label, model column, msis column, format)
METRIC_ROWS = [
    (r"RMSE$_{\log}$",          "rmse_log",  "msis_rmse_log",  "log"),
    (r"Top5$_{\log}$",          "top5_log",  "msis_top5_log",  "log"),
    (r"RMSE (kg\,m$^{-3}$)",    "rmse",      "msis_rmse",      "sci"),
    (r"Top5 (kg\,m$^{-3}$)",    "top5",      "msis_top5",      "sci"),
    (r"MAPE (\%)",              "mape_pct",  "msis_mape_pct",  "flt"),
    (r"R$^2$",                  "r2",        "msis_r2",        "flt"),
]

REGIME_LABELS = {
    "quiet2009": "Quiet 2009",
    "storm2015": "Storm 2015",
    "post2016":  "Post-2016",
    "pre2009":   "Pre-2009",
    "y2002":     "Solar max 2002",
}


def fmt(value: float, kind: str) -> str:
    if pd.isna(value):
        return "--"
    if kind == "sci":
        s = f"{value:.2e}"
        mant, exp = s.split("e")
        return f"${mant}\\times10^{{{int(exp)}}}$"
    if kind == "log":
        # RMSE_log doubles as a multiplicative factor: exp(x).
        import math
        return f"{value:.3f} ($\\times${math.exp(value):.2f})"
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"


def pct(model: float, msis: float) -> str:
    """Signed change of the model against the MSIS baseline."""
    if pd.isna(model) or pd.isna(msis) or msis == 0:
        return "--"
    return f"{100.0 * (model / msis - 1.0):+.1f}\\%"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs_test_lb3",
                   help="run directory holding summary_metrics.csv")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--regimes", default="quiet2009,storm2015,post2016")
    p.add_argument("--out", default=None, help="write LaTeX here (default: <run>/table_regimes.tex)")
    args = p.parse_args()

    csv = os.path.join(args.run, "summary_metrics.csv")
    if not os.path.isfile(csv):
        raise SystemExit(f"Not found: {csv}")
    df = pd.read_csv(csv)
    df = df[df["horizon_days"] == args.horizon]
    by_tag = {str(r["tag"]): r for _, r in df.iterrows()}

    tex, txt = [], []
    tex.append(r"\begin{tabular}{lrrrl}")
    tex.append(r"\hline")
    tex.append(r"\textbf{Metric} & \textbf{NRLMSISE-2.1} & \textbf{Predicted} "
               r"& \textbf{Diff (\%)} & \textbf{Training} \\")
    tex.append(r"\hline")

    for regime in args.regimes.split(","):
        for dr, variant in ((0, "Core"), (1, "Warm-start")):
            tag = f"dr{dr}_{regime}_h{args.horizon}"
            if tag not in by_tag:
                print(f"  (skipping {tag}: not in {csv})")
                continue
            row = by_tag[tag]
            label = REGIME_LABELS.get(regime, regime)
            head = f"{label} ({variant})"
            tex.append(rf"\textbf{{{head}}} & & & & \\")
            txt.append(f"\n{head}")
            txt.append(f"  {'Metric':<22}{'MSIS':>14}{'Predicted':>14}{'Diff':>10}")
            for lab, mcol, bcol, kind in METRIC_ROWS:
                mv, bv = row.get(mcol), row.get(bcol)
                tex.append(f"{lab} & {fmt(bv, kind)} & {fmt(mv, kind)} & "
                           f"{pct(mv, bv)} & {variant} \\\\")
                plain = lab.replace(r"$_{\log}$", "_log").replace(r"\%", "%") \
                           .replace(r"R$^2$", "R2").replace(r"\,", " ") \
                           .replace(r"(kg\,m$^{-3}$)", "(kg/m3)")
                txt.append(f"  {plain:<22}{bv:>14.4g}{mv:>14.4g}"
                           f"{pct(mv, bv).replace(chr(92)+'%','%'):>10}")
            tex.append(r"\hline")

    tex.append(r"\end{tabular}")

    out = args.out or os.path.join(args.run, "table_regimes.tex")
    with open(out, "w") as fh:
        fh.write("\n".join(tex) + "\n")

    print("\n".join(txt))
    print(f"\nWrote LaTeX -> {out}")
    print(f"  source: {csv}  (horizon h={args.horizon})")


if __name__ == "__main__":
    main()
