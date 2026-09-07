#!/usr/bin/env python3
"""Diagnose missing TEC matches without changing the merge pipeline.

The report distinguishes between

1. rows for which the merge produced no spatial/temporal match record, and
2. rows matched to a TEC grid node whose TEC value is missing (for example an
   IONEX fill marker converted to NaN during parsing).

It also checks whether the raw TEC parquet contains any epochs on dates with
missing merged TEC values. All calculations are streamed from parquet files;
the full data sets are not loaded into memory.

Usage
-----
    ven_2404/bin/python DataPreparation/diagnose_tec_missing.py
    ven_2404/bin/python DataPreparation/diagnose_tec_missing.py \
        --merged grace_data_merged_v5_full.parquet \
        --tec tec_codg_2002-2017_doy1-365_v2.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MERGED = ROOT / "grace_data_merged_v5_full.parquet"
DEFAULT_TEC = ROOT / "tec_codg_2002-2017_doy1-365_v2.parquet"


def missing(column: str) -> pl.Expr:
    """True for either Arrow nulls or floating-point NaNs."""
    return pl.col(column).is_null() | pl.col(column).is_nan()


def build_report(merged_path: Path, tec_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    merged = pl.scan_parquet(merged_path)

    value_missing = missing("matched_tec_value")
    match_metadata_missing = (
        missing("matched_tec_latitude")
        | missing("matched_tec_longitude")
        | missing("chord_distance")
    )
    status = (
        pl.when(~value_missing)
        .then(pl.lit("matched_value"))
        .when(match_metadata_missing)
        .then(pl.lit("no_match_record"))
        .otherwise(pl.lit("matched_node_missing_value"))
        .alias("tec_status")
    )

    classified = merged.with_columns(status)
    totals = (
        classified.group_by("tec_status")
        .agg(pl.len().cast(pl.UInt64).alias("rows"))
        .collect(engine="streaming")
        .with_columns(
            (100.0 * pl.col("rows") / pl.col("rows").sum().cast(pl.Float64))
            .alias("percent_of_all_rows")
        )
        .sort("tec_status")
    )

    missing_dates = (
        classified.filter(value_missing)
        .with_columns(
            pl.col("grace_time").dt.date().alias("date"),
            pl.col("grace_time").dt.ordinal_day().alias("day_of_year"),
        )
        .group_by(["year", "date", "day_of_year", "tec_status"])
        .agg(
            pl.len().cast(pl.UInt64).alias("missing_rows"),
            pl.col("grace_time").min().alias("first_missing_time"),
            pl.col("grace_time").max().alias("last_missing_time"),
        )
        .collect(engine="streaming")
        .sort(["date", "tec_status"])
    )

    # Only distinct epochs are materialized (about tens of thousands), not the
    # hundreds of millions of spatial grid rows in the raw TEC parquet.
    tec_epochs = (
        pl.scan_parquet(tec_path)
        .select("epoch")
        .unique()
        .collect(engine="streaming")
        .with_columns(pl.col("epoch").dt.date().alias("date"))
    )
    tec_by_date = tec_epochs.group_by("date").agg(
        pl.len().cast(pl.UInt64).alias("tec_epochs_on_date"),
        pl.col("epoch").min().alias("first_tec_epoch"),
        pl.col("epoch").max().alias("last_tec_epoch"),
    )
    missing_dates = (
        missing_dates.join(tec_by_date, on="date", how="left")
        .with_columns(pl.col("tec_epochs_on_date").fill_null(0))
    )

    fallbacks = (
        merged.select(
            pl.col("fallback_used").fill_null(0).cast(pl.UInt64).sum().alias("fallback_rows")
        )
        .collect(engine="streaming")
        .item()
    )

    total_rows = int(totals["rows"].sum())
    missing_rows = int(missing_dates["missing_rows"].sum())
    missing_pct = 100.0 * missing_rows / total_rows
    fill_rows = int(
        totals.filter(pl.col("tec_status") == "matched_node_missing_value")["rows"].sum()
    )

    print(f"Merged data: {merged_path}")
    print(f"Raw TEC data: {tec_path}")
    print(f"Total merged rows: {total_rows:,}")
    print(f"Rows without matched_tec_value: {missing_rows:,} ({missing_pct:.6f}%)")
    print(f"Missing values at an assigned grid node: {fill_rows:,}")
    print(f"Spatial fallback rows: {fallbacks:,}")
    print("\nMatch-status totals")
    print(totals)
    print("\nMissing rows by date")
    print(missing_dates)

    return totals, missing_dates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--tec", type=Path, default=DEFAULT_TEC)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Optional prefix for CSV copies of the totals and date-level report.",
    )
    args = parser.parse_args()

    for path in (args.merged, args.tec):
        if not path.is_file():
            raise SystemExit(f"Not found: {path}")

    totals, missing_dates = build_report(args.merged, args.tec)
    if args.output_prefix is not None:
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        totals_path = args.output_prefix.with_name(args.output_prefix.name + "_totals.csv")
        dates_path = args.output_prefix.with_name(args.output_prefix.name + "_dates.csv")
        totals.write_csv(totals_path)
        missing_dates.write_csv(dates_path)
        print(f"\nWrote {totals_path}")
        print(f"Wrote {dates_path}")


if __name__ == "__main__":
    main()
