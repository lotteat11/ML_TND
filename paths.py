# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
paths.py
Central registry of all large data files and model artifacts.
Import these constants instead of hardcoding filenames across scripts.

Mapping of logical names to actual filenames:

    GRACE_RAW    <- grace_dns_2009_2016.parquet
    TEC_RAW      <- tec_codg_2009-2017_doy1-365_v2.parquet
    SWARM_RAW    <- swarm_dns_2015_2016_03092025.parquet
    GRACE_MSIS   <- grace_dns_with_tnd_y200916_v4_0809.parquet
    SWARM_MSIS   <- swarm_dns_with_tnd_y2001516_v1_0309.parquet
    GRACE_MERGED <- grace_data_merged_v3.parquet
    MODEL        <- xgb_model_v3.json
    SCALER_X     <- scaler_xgboost_X_v3.joblib
    SCALER_Y     <- scaler_xgboost_y_v3.joblib
"""

from pathlib import Path

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Raw downloads (pipeline steps 1–2)
# ---------------------------------------------------------------------------
GRACE_RAW = ROOT / "grace_dns_2009_2016.parquet"
TEC_RAW   = ROOT / "tec_codg_2009-2017_doy1-365_v2.parquet"
SWARM_RAW = ROOT / "swarm_dns_2015_2016_03092025.parquet"

# ---------------------------------------------------------------------------
# After NRLMSISE-2.1 processing (pipeline steps 3, 7b)
# ---------------------------------------------------------------------------
GRACE_MSIS = ROOT / "grace_dns_with_tnd_y200916_v4_0809.parquet"
SWARM_MSIS = ROOT / "swarm_dns_with_tnd_y2001516_v1_0309.parquet"

# ---------------------------------------------------------------------------
# After TEC merge (pipeline step 4)
# ---------------------------------------------------------------------------
GRACE_MERGED = ROOT / "grace_data_merged_v3.parquet"

# ---------------------------------------------------------------------------
# Model artifacts (pipeline step 5)
# ---------------------------------------------------------------------------
MODEL    = ROOT / "xgb_model_v3.json"
SCALER_X = ROOT / "scaler_xgboost_X_v3.joblib"
SCALER_Y = ROOT / "scaler_xgboost_y_v3.joblib"
