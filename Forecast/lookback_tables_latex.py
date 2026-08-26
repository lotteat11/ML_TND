# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
lookback_tables_latex.py
- Turns the CSVs written by Forecast/tune_lookback.py into LaTeX tables ready to
  paste into the thesis.
- Emits two tables: the lookback sensitivity sweep and the reset-cadence sweep.
- The bias column is dropped on purpose; it is not reported anywhere else in the
  text, so carrying it here would be the only place a reader meets it.
- The MSIS baseline row is read back from a prediction pickle so the table shows
  what the warm-start is measured against, not just the sweep internals.

Usage:
    ven_2404/bin/python Forecast/lookback_tables_latex.py
    LOOKBACK_OUTPUT_DIR=lookback_sensitivity_2016jf \
        ven_2404/bin/python Forecast/lookback_tables_latex.py
"""

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "Forecast"))

OUTPUT_DIR = os.environ.get("LOOKBACK_OUTPUT_DIR", "lookback_sensitivity")
EVAL_START = os.environ.get("EVAL_START", "2016-01-01")
EVAL_END = os.environ.get("EVAL_END", "2016-03-01")

# Columns to show, in table order. Bias and n are deliberately absent: bias is
# not reported anywhere else in the text, and every sweep scores identical rows
# by construction, so a repeated n carries no information.
COLUMNS = [
    ("rmse", r"RMSE [kg\,m$^{-3}$]", lambda v: _sci(v)),
    ("mape_pct", r"MAPE [\%]", lambda v: f"{v:.2f}"),
    ("rmse_log", "log-RMSE", lambda v: f"{v:.4f}"),
    ("top5", r"Top-5\,\% [kg\,m$^{-3}$]", lambda v: _sci(v)),
    ("r2", "$R^2$", lambda v: f"{v:.4f}"),
]


def _sci(value: float, digits: int = 3) -> str:
    """Format as LaTeX scientific notation, e.g. 3.193e-13 -> $3.193\\times10^{-13}$."""
    if value == 0 or not np.isfinite(value):
        return "--"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return f"${mantissa:.{digits}f}\\times10^{{{exponent}}}$"


def _msis_baseline() -> dict | None:
    """Recompute the MSIS baseline from any prediction pickle in OUTPUT_DIR.

    Every sweep scored identical rows, so whichever pickle is found first gives
    the same baseline. Returns None if no pickle is present.
    """
    from tune_lookback import metrics  # noqa: PLC0415 — needs OUTPUT_DIR resolved first

    for name in sorted(os.listdir(OUTPUT_DIR)):
        if name.startswith("predictions_lb") and name.endswith(".pkl"):
            df = pd.read_pickle(os.path.join(OUTPUT_DIR, name))
            return metrics(df, pred_col="msis_rho")
    return None


def _rows(df: pd.DataFrame, label_col: str, label_fmt) -> list[str]:
    rows = []
    for _, row in df.iterrows():
        cells = [label_fmt(row[label_col])]
        cells += [fmt(row[key]) for key, _, fmt in COLUMNS]
        rows.append("    " + " & ".join(cells) + r" \\")
    return rows


def build_table(df: pd.DataFrame, label_col: str, label_head: str, label_fmt,
                caption: str, tag: str, baseline: dict | None) -> str:
    align = "l" + "r" * len(COLUMNS)
    head = " & ".join([label_head] + [head for _, head, _ in COLUMNS])

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{tab:{tag}}}",
        f"  \\begin{{tabular}}{{{align}}}",
        r"    \hline",
        "    " + head + r" \\",
        r"    \hline",
    ]
    lines += _rows(df, label_col, label_fmt)

    if baseline is not None:
        lines.append(r"    \hline")
        cells = ["NRLMSIS-2.1"]
        cells += [fmt(baseline[key]) for key, _, fmt in COLUMNS]
        lines.append("    " + " & ".join(cells) + r" \\")

    lines += [
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if not os.path.isdir(OUTPUT_DIR):
        raise FileNotFoundError(
            f"{OUTPUT_DIR} not found — run Forecast/tune_lookback.py first."
        )

    lookback_csv = os.path.join(OUTPUT_DIR, "lookback_summary.csv")
    reset_csv = os.path.join(OUTPUT_DIR, "reset_summary.csv")
    missing = [p for p in (lookback_csv, reset_csv) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} — the sweep has not finished writing its CSVs."
        )

    baseline = _msis_baseline()
    window = f"{EVAL_START} to {EVAL_END}"

    lookback_df = pd.read_csv(lookback_csv)
    reset_df = pd.read_csv(reset_csv)
    reset_lookback = int(reset_df["lookback_days"].iloc[0])

    # n is no longer a column, so the matched row count moves into the caption:
    # it is what makes the comparison valid and should not be lost.
    n_rows = sorted(set(lookback_df["n"]) | set(reset_df["n"]))
    if len(n_rows) != 1:
        raise RuntimeError(
            f"Sweeps scored different row counts {n_rows}; the tables would "
            f"compare runs evaluated on different data."
        )
    n_txt = f"{int(n_rows[0]):,}".replace(",", "\\,")

    tables = [
        build_table(
            lookback_df, "lookback_days", "Lookback [d]",
            lambda v: f"{int(v)}",
            caption=(
                f"Warm-start skill against the fine-tuning lookback window, "
                f"{window}. All lookbacks forecast identical days and score "
                f"the same {n_txt} observations."
            ),
            tag="lookback_sensitivity",
            baseline=baseline,
        ),
        build_table(
            reset_df, "reset_every_iterations", "Reset [iter.]",
            lambda v: f"{int(v)}",
            caption=(
                f"Warm-start skill against the reset cadence at a fixed "
                f"{reset_lookback}-day lookback, {window}. Scored on the same "
                f"{n_txt} observations as the lookback sweep."
            ),
            tag="reset_sensitivity",
            baseline=baseline,
        ),
    ]

    out_tex = os.path.join(OUTPUT_DIR, "sensitivity_tables.tex")
    body = "\n\n".join(tables) + "\n"
    with open(out_tex, "w") as fh:
        fh.write(body)

    print(body)
    print(f"Saved → {out_tex}")
