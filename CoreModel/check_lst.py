# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
check_lst.py
- Loads and engineers features, runs the cyclic time-block split,
  and prints LST coverage statistics for each split.
  Run this instead of the full train.py pipeline when diagnosing splits.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import feature_functions as ff
from train import load_and_engineer
from config import PARQUET_FILE, FEATURES, TARGET


def print_stats(name: str, df: pd.DataFrame) -> None:
    h = df["lst_h"]
    p5, p25, p75, p95 = h.quantile([0.05, 0.25, 0.75, 0.95]).values
    std = h.std()
    uniform_std = 24 / (2 * np.sqrt(3))  # std of uniform U(0,24) ≈ 6.93
    coverage = std / uniform_std * 100

    f107_5, f107_95 = df["f107"].quantile([0.05, 0.95]).values
    ap_5,   ap_95   = df["ap_m3h"].quantile([0.05, 0.95]).values

    print(
        f"{name:>6}  n={len(df):>8,}  "
        f"LST 5–95th: [{p5:4.1f}–{p95:4.1f}] h  "
        f"IQR: [{p25:4.1f}–{p75:4.1f}] h  "
        f"std={std:.2f} h ({coverage:.0f}% of uniform)  |  "
        f"F10.7 5–95th: [{f107_5:.1f}–{f107_95:.1f}]  "
        f"Ap 5–95th: [{ap_5:.1f}–{ap_95:.1f}]"
    )


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
    print(f"Loading {PARQUET_FILE} ...")
    df = load_and_engineer(PARQUET_FILE)

    X = df[FEATURES]
    y = df[[TARGET]]

    _, _, _, _, _, _, idx_train, idx_val, idx_test = ff.timeblock_split_repeated(
        X, y,
        fractions=(2/3, 1/6, 1/6),
        n_cycles=8,
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

    print("\nSplit diagnostics:")
    for name, sub in splits.items():
        print_stats(name, sub)

    plot_distributions(splits)
