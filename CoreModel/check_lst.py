# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
check_lst.py
- Loads the new full-mission article dataset with the same feature settings as
  the model, runs the cyclic time-block split, and reports coverage for LST,
  F10.7, and Ap.
- Exports both the complete min--max range and the descriptive P5--P95 interval;
  no distribution tails are removed.

Example
-------
    ven_2404/bin/python CoreModel/check_lst.py --no-plot
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

VARIABLES = {
    "Local solar time (h)": "lst_h",
    "F10.7 (sfu)": "f107",
    "Ap (3-h lag)": "ap_m3h",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report full and central-90% coverage for the article dataset splits."
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parent.parent / "grace_data_merged_v5_full.parquet",
        help="Merged model dataset (default: new full-mission v5 dataset).",
    )
    parser.add_argument("--start", default="2002-01-01", help="Exclusive model-period start.")
    parser.add_argument("--end", default="2016-01-01", help="Exclusive model-period end.")
    parser.add_argument(
        "--exclude", default="2009-01-01,2009-06-06",
        help="Interior holdout(s), as start,end pairs separated by semicolons.",
    )
    parser.add_argument(
        "--ap-history", choices=("0", "1", "full"), default="1",
        help="Ap-history feature configuration used by the model.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path.cwd(),
        help="Directory for dataset_coverage.csv and dataset_coverage.tex.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not create the KDE plot.")
    return parser.parse_args()


def coverage_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build Table 2 statistics without excluding either distribution tail."""
    rows = []
    for variable, column in VARIABLES.items():
        for split, df in splits.items():
            values = df[column].dropna()
            p5, p95 = values.quantile([0.05, 0.95]).values
            rows.append({
                "Variable": variable,
                "Split": split.title(),
                "N": len(values),
                "Full range (min–max)": f"{values.min():.1f}–{values.max():.1f}",
                "Central 90% (P5–P95)": f"{p5:.1f}–{p95:.1f}",
            })
    return pd.DataFrame(rows)


def save_coverage_table(table: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dataset_coverage.csv"
    tex_path = output_dir / "dataset_coverage.tex"
    table.to_csv(csv_path, index=False)
    caption = (
        "Coverage of selected variables across dataset splits. The central 90\\% "
        "interval is descriptive only; all samples in the full range, including "
        "the lower and upper 5\\% tails, were retained."
    )
    table.to_latex(
        tex_path, index=False, escape=False, caption=caption,
        label="tab:dataset_coverage",
    )
    print(table.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {tex_path}")


def plot_distributions(splits: dict) -> None:
    variables = [
        ("lst_h",    "Local Solar Time (h)"),
        ("f107",     "F10.7 (sfu)"),
        ("ap_m3h",   "Ap 3-hour"),
        ("log_ratio","log(rho_obs / rho_msis)"),
    ]
    colors = {"TRAIN": "steelblue", "VAL": "darkorange", "TEST": "seagreen"}

    fig, axes = plt.subplots(1, len(variables), figsize=(14, 4))
    for ax, (col, label) in zip(axes, variables):
        for name, df in splits.items():
            vals = df[col].dropna().values
            xs = np.linspace(vals.min(), vals.max(), 500)
            kde = gaussian_kde(vals, bw_method=0.15)
            ax.plot(xs, kde(xs), label=name, color=colors[name], lw=1.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend()
    fig.suptitle("Distribution overlap across splits")
    fig.tight_layout()
    plt.savefig("split_distributions.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    # Set the article configuration before importing config/train: both modules
    # resolve these environment variables at import time.
    os.environ["TRAIN_PARQUET_FILE"] = str(input_path)
    os.environ["TRAIN_TIME_MIN"] = args.start
    os.environ["TRAIN_TIME_MAX"] = args.end
    os.environ["TRAIN_TIME_EXCLUDE"] = args.exclude
    os.environ["AP_HISTORY"] = args.ap_history

    import feature_functions as ff
    from config import FEATURES, TARGET, TEC_LAGS, TEC_LAG_COLS
    from train import load_and_engineer

    print("Article dataset configuration:")
    print(f"  input:       {input_path}")
    print(f"  period:      {args.start} < time < {args.end}")
    print(f"  holdout:     {args.exclude or 'none'}")
    print(f"  TEC lags:    {', '.join(f'{l} -> {c}' for l, c in zip(TEC_LAGS, TEC_LAG_COLS))}")
    print(f"  Ap history:  {args.ap_history}")
    print(f"\nLoading {input_path} ...")
    df = load_and_engineer(input_path)

    X = df[FEATURES]
    y = df[[TARGET]]

    _, _, _, _, _, _, idx_train, idx_val, idx_test = ff.timeblock_split_repeated(
        X, y,
        fractions=(2/3, 1/6, 1/6),
        n_cycles=16,
        gap_before_val=1100,
        gap_before_test=1100,
        order=("train", "test", "val"),
        copy=False,
    )

    splits = {
        "TRAIN": df.loc[idx_train],
        "VAL":   df.loc[idx_val],
        "TEST":  df.loc[idx_test],
    }

    print("\nSplit diagnostics (all tails retained):")
    table = coverage_table(splits)
    save_coverage_table(table, args.output_dir)

    if not args.no_plot:
        plot_distributions(splits)
