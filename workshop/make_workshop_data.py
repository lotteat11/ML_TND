"""
make_workshop_data.py

Run this script once to generate the smaller workshop dataset from the full
GRACE merged parquet. Needs to run from the repo root (where paths.py lives).

Usage:
    python workshop/make_workshop_data.py

Output:
    grace_workshop.parquet  (placed next to paths.py)

Dataset composition
-------------------
Core training years  :  full 2010, 2012, 2014
Edge year — quiet    :  Jan–Mar 2009   (high GRACE altitude, quiet conditions)
Edge year — disturbed:  Jan–Mar 2016   (low GRACE altitude, storm activity;
                                        includes the Feb 18 2016 storm from NB5)
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from paths import GRACE_MERGED

OUT = ROOT / "grace_workshop.parquet"

print(f"Reading {GRACE_MERGED} ...")
df = pd.read_parquet(GRACE_MERGED)
df["time"] = pd.to_datetime(df["grace_time"])
df = df.sort_values("time").reset_index(drop=True)

print(f"Full dataset: {len(df):,} rows  "
      f"({df['time'].min().date()} → {df['time'].max().date()})")

# ── Date filters ─────────────────────────────────────────────────────────────
masks = [
    # Edge year — quiet (Jan–Mar 2009)
    (df["time"] >= "2009-01-01") & (df["time"] < "2009-04-01"),
    # Core training — 2010
    (df["time"] >= "2010-01-01") & (df["time"] < "2011-01-01"),
    # Core training — 2012
    (df["time"] >= "2012-01-01") & (df["time"] < "2013-01-01"),
    # Core training — 2014
    (df["time"] >= "2014-01-01") & (df["time"] < "2015-01-01"),
    # Edge year — disturbed (Jan–Mar 2016, includes Feb 18 storm)
    (df["time"] >= "2016-01-01") & (df["time"] < "2016-04-01"),
]

combined = masks[0]
for m in masks[1:]:
    combined = combined | m

df_ws = df[combined].reset_index(drop=True)

print(f"\nWorkshop dataset: {len(df_ws):,} rows")
print(f"  Jan–Mar 2009 : {(masks[0]).sum():>7,} rows")
print(f"  Full 2010    : {(masks[1]).sum():>7,} rows")
print(f"  Full 2012    : {(masks[2]).sum():>7,} rows")
print(f"  Full 2014    : {(masks[3]).sum():>7,} rows")
print(f"  Jan–Mar 2016 : {(masks[4]).sum():>7,} rows")
print(f"\nAltitude range: {df_ws['alt_km'].min():.0f} km "
      f"→ {df_ws['alt_km'].max():.0f} km")

size_mb = df_ws.memory_usage(deep=True).sum() / 1e6
print(f"Estimated size in memory: {size_mb:.0f} MB")

df_ws.to_parquet(OUT, index=False)
print(f"\nSaved → {OUT}")
print("\nUpdate paths.py or the notebook GRACE_MERGED variable to point to this file.")
