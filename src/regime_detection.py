"""
regime_detection.py — Classify weekly observations into Fed rate regimes.

Method: Rolling 26-week change in FEDFUNDS and DGS2.
No look-ahead bias: only uses information available up to each week.

Regime labels:
    "Hiking_or_Restrictive"  — FEDFUNDS or DGS2 is rising over the past 26 weeks
    "Pausing_or_Cutting"     — both are flat or declining

Output:
    data/results/regime_labels.csv
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import REGIME_LOOKBACK_WEEKS, PATHS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

REGIME_HIKING  = "Hiking_or_Restrictive"
REGIME_CUTTING = "Pausing_or_Cutting"


def classify_regime(macro_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each week using rolling 26-week change in FEDFUNDS and DGS2.
    Returns a DataFrame with columns: date, regime, fedfunds_roll_chg, dgs2_roll_chg.
    """
    df = macro_weekly.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Use level columns (from macro_weekly.csv which has raw weekly last values)
    fedfunds_col = "FEDFUNDS" if "FEDFUNDS" in df.columns else None
    dgs2_col     = "DGS2"     if "DGS2"     in df.columns else None

    if fedfunds_col is None and dgs2_col is None:
        log.error("Neither FEDFUNDS nor DGS2 found in macro_weekly. "
                  "Cannot classify regimes.")
        raise ValueError("FEDFUNDS or DGS2 required for regime detection.")

    result = pd.DataFrame(index=df.index)

    if fedfunds_col:
        result["fedfunds_roll_chg"] = df[fedfunds_col] - df[fedfunds_col].shift(REGIME_LOOKBACK_WEEKS)
    else:
        result["fedfunds_roll_chg"] = 0.0

    if dgs2_col:
        result["dgs2_roll_chg"] = df[dgs2_col] - df[dgs2_col].shift(REGIME_LOOKBACK_WEEKS)
    else:
        result["dgs2_roll_chg"] = 0.0

    # Regime: Hiking if either rate is rising over the 26-week window
    result["regime"] = result.apply(
        lambda row: REGIME_HIKING
        if (row["fedfunds_roll_chg"] >= 0 or row["dgs2_roll_chg"] >= 0)
        else REGIME_CUTTING,
        axis=1,
    )

    # First 26 weeks will be NaN rolling — mark as unknown
    n_nan = result[["fedfunds_roll_chg", "dgs2_roll_chg"]].isna().any(axis=1).sum()
    result.loc[result[["fedfunds_roll_chg", "dgs2_roll_chg"]].isna().any(axis=1), "regime"] = "Unknown"

    result.index.name = "date"
    return result.reset_index()


def run_regime_detection() -> pd.DataFrame:
    macro_weekly = pd.read_csv(PATHS["macro_weekly"], index_col=0, parse_dates=True)

    regime_df = classify_regime(macro_weekly)
    regime_df.to_csv(PATHS["regime_labels"], index=False)

    counts = regime_df["regime"].value_counts()
    log.info("Regime counts:\n%s", counts.to_string())
    log.info("Saved: %s", PATHS["regime_labels"])

    return regime_df


if __name__ == "__main__":
    run_regime_detection()
