"""Create the AGU-style weekly-density and altitude overview figures.

The parquet files are read in batches so the full 2002--2017 dataset does not
need to fit in memory.  Density is averaged by calendar week; altitude is
evenly downsampled only for display.

Examples
--------
    python DataPreparation/plot_density_altitude.py
    python DataPreparation/plot_density_altitude.py data_1.parquet data_2.parquet
    python DataPreparation/plot_density_altitude.py --start 2009 --end 2017
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "grace_dns_with_tnd_y200217_v5.parquet"
REQUIRED_COLUMNS = ["time", "rho_obs", "msis_rho", "alt_km"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot weekly observed/MSIS density and observed altitude."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[DEFAULT_INPUT],
        help="One or more parquet files (default: expanded 2002--2017 file).",
    )
    parser.add_argument("--start", help="Optional inclusive start date/year.")
    parser.add_argument("--end", help="Optional inclusive end date/year.")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT, help="Directory for the PNG files."
    )
    parser.add_argument(
        "--altitude-points",
        type=int,
        default=200_000,
        help="Approximate maximum number of altitude points drawn.",
    )
    return parser.parse_args()


def _date_bound(value: str | None, *, end: bool) -> pd.Timestamp | None:
    if value is None:
        return None
    if len(value) == 4 and value.isdigit():
        return pd.Timestamp(f"{value}-12-31 23:59:59.999999999" if end else f"{value}-01-01")
    return pd.Timestamp(value)


def load_plot_data(
    paths: list[Path],
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    altitude_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scans: list[pl.LazyFrame] = []
    total_rows = 0
    for path in paths:
        scan = pl.scan_parquet(path)
        missing = sorted(set(REQUIRED_COLUMNS) - set(scan.collect_schema().names()))
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        total_rows += int(scan.select(pl.len()).collect(engine="streaming").item())
        scans.append(scan.select(REQUIRED_COLUMNS))

    data = pl.concat(scans)
    if start is not None:
        data = data.filter(pl.col("time") >= start.to_pydatetime())
    if end is not None:
        data = data.filter(pl.col("time") <= end.to_pydatetime())

    weekly_pl = (
        data.drop_nulls(["time", "rho_obs", "msis_rho"])
        .with_columns(pl.col("time").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(pl.col("rho_obs").mean(), pl.col("msis_rho").mean())
        .sort("week")
        .collect(engine="streaming")
    )
    if weekly_pl.is_empty():
        raise ValueError("No valid density observations were found in the requested period.")

    weekly = pd.DataFrame(weekly_pl.to_dict(as_series=False)).set_index("week")

    # A missing week remains visually blank instead of connecting data periods.
    full_weeks = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-MON")
    weekly = weekly.reindex(full_weeks)

    altitude_stride = max(1, int(np.ceil(total_rows / altitude_points)))
    altitude_pl = (
        data.drop_nulls(["time", "alt_km"])
        .with_row_index("row_number")
        .filter(pl.col("row_number") % altitude_stride == 0)
        .select("time", "alt_km")
        .sort("time")
        .collect(engine="streaming")
    )
    altitude = pd.DataFrame(altitude_pl.to_dict(as_series=False))
    return weekly, altitude


def apply_agu_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "lines.linewidth": 1.2,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
        }
    )


def format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=35)


def plot_density(weekly: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5 / 2.54, 6 / 2.54))
    ax.plot(weekly.index, weekly["rho_obs"] / 1e-12, color="#0072B2", label="Observed Density")
    ax.plot(
        weekly.index,
        weekly["msis_rho"] / 1e-12,
        color="#009E73",
        linestyle="--",
        label="MSIS Model",
    )
    ax.set_title("Observed vs MSIS Density (Weekly Average)")
    ax.set_xlabel("Time [Year]")
    ax.set_ylabel(r"Density [$10^{-12}$ kg/m$^3$]")
    ax.legend(frameon=False)
    format_time_axis(ax)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_altitude(altitude: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5 / 2.54, 6 / 2.54))
    ax.plot(altitude["time"], altitude["alt_km"], color="#0072B2", linewidth=0.65)
    ax.set_title("Observed Altitude Over Time")
    ax.set_xlabel("Time [Year]")
    ax.set_ylabel("Altitude [km]")
    format_time_axis(ax)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = [path.expanduser().resolve() for path in args.inputs]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.altitude_points <= 0:
        raise ValueError("--altitude-points must be greater than zero")

    weekly, altitude = load_plot_data(
        paths,
        _date_bound(args.start, end=False),
        _date_bound(args.end, end=True),
        args.altitude_points,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    density_output = args.output_dir / "rho_vs_msis_weekly_AGU.png"
    altitude_output = args.output_dir / "figure_altvstime.png"
    apply_agu_style()
    plot_density(weekly, density_output)
    plot_altitude(altitude, altitude_output)
    print(f"Saved: {density_output}")
    print(f"Saved: {altitude_output}")


if __name__ == "__main__":
    main()
