"""
feature_engineering.py — Transform cleaned weekly data into analysis-ready features.

Transformations:
  - Stocks / ETFs / FX / Index  →  weekly log return
  - Yields / Interest rates     →  weekly change (first difference)
  - VIX                         →  VIX_CHANGE (log return) + VIX_LEVEL (kept raw)
  - Derived factors:
      US_YIELD_CURVE   = DGS10 - DGS2
      THAI_YIELD_CURVE = thai_gov_10y - thai_gov_2y (if available)
      THAI_BANK_BASKET = equal-weighted mean of 7 bank log returns
      BOT_RATE_CHANGE  = weekly change in BOT policy rate
      BOT_RATE_HIKE    = 1 if BOT_RATE_CHANGE > 0
      BOT_RATE_CUT     = 1 if BOT_RATE_CHANGE < 0

  - Outlier flags re-run on returns/changes
  - Ensure first row (NaN from differencing) is dropped before saving

Entry point: run_feature_engineering()
Output:
    data/processed/weekly_returns.csv  (prices → log returns only)
    data/processed/final_weekly_dataset.csv  (all features merged)
    data/processed/outlier_flags.csv updated with return-based flags
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    THAI_BANK_TICKERS,
    ETF_TICKERS,
    FX_TICKERS,
    INDEX_TICKERS,
    OUTLIER_ZSCORE_THRESHOLD,
    PATHS,
)
from src.data_cleaning import flag_outliers, outlier_flag_long

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# These columns get log returns
PRICE_COLS = THAI_BANK_TICKERS + ETF_TICKERS + FX_TICKERS + INDEX_TICKERS

# These FRED columns get first-difference (rate change)
RATE_COLS = ["FEDFUNDS", "DGS2", "DGS10", "MORTGAGE30US"]

# VIX handled separately
VIX_COL = "VIXCLS"

# BOT columns that need first-difference
BOT_RATE_COL = "bot_policy_rate"


# ─────────────────────────────────────────────────────────────────────────────
# Transformations
# ─────────────────────────────────────────────────────────────────────────────

def log_return(prices: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Compute weekly log returns for price columns."""
    available = [c for c in cols if c in prices.columns]
    missing   = [c for c in cols if c not in prices.columns]
    if missing:
        log.warning("log_return: columns not found → %s", missing)

    ret = np.log(prices[available] / prices[available].shift(1))

    # Rename: BBL.BK → BBL_ret  (strip .BK, ^, =X suffixes)
    rename = {}
    for c in available:
        clean = c.replace(".BK", "").replace("^", "").replace("=X", "")
        rename[c] = f"{clean}_ret"
    return ret.rename(columns=rename)


def rate_change(macro: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Compute weekly first-difference for rate columns."""
    available = [c for c in cols if c in macro.columns]
    missing   = [c for c in cols if c not in macro.columns]
    if missing:
        log.warning("rate_change: columns not found → %s", missing)

    chg = macro[available].diff()
    rename = {c: f"{c}_chg" for c in available}
    return chg.rename(columns=rename)


def vix_features(macro: pd.DataFrame) -> pd.DataFrame:
    """VIX_CHANGE (log return) + VIX_LEVEL (raw level for context)."""
    result = pd.DataFrame(index=macro.index)
    if VIX_COL not in macro.columns:
        log.warning("VIXCLS not found — skipping VIX features.")
        return result

    result["VIX_CHANGE"] = np.log(macro[VIX_COL] / macro[VIX_COL].shift(1))
    result["VIX_LEVEL"]  = macro[VIX_COL]   # kept for interpretation; NOT for correlation
    return result


def derived_factors(macro: pd.DataFrame, bank_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute derived factor columns."""
    result = pd.DataFrame(index=macro.index)

    # US Yield Curve
    if "DGS10" in macro.columns and "DGS2" in macro.columns:
        spread = macro["DGS10"] - macro["DGS2"]
        result["US_YIELD_CURVE_chg"] = spread.diff()
        log.info("US_YIELD_CURVE_chg created.")
    else:
        log.warning("DGS10 or DGS2 missing — US_YIELD_CURVE_chg skipped.")

    # Thai Yield Curve (optional)
    if "thai_gov_10y" in macro.columns and "thai_gov_2y" in macro.columns:
        thai_spread = macro["thai_gov_10y"] - macro["thai_gov_2y"]
        result["THAI_YIELD_CURVE_chg"] = thai_spread.diff()
        log.info("THAI_YIELD_CURVE_chg created.")
    else:
        log.info("Thai yield data not available — THAI_YIELD_CURVE_chg skipped.")

    # BOT Rate Change + dummies
    if BOT_RATE_COL in macro.columns:
        chg = macro[BOT_RATE_COL].diff()
        result["BOT_RATE_CHANGE"] = chg
        result["BOT_RATE_HIKE"]   = (chg > 0).astype(int)
        result["BOT_RATE_CUT"]    = (chg < 0).astype(int)
        log.info("BOT_RATE_CHANGE, HIKE, CUT dummies created.")
    else:
        log.warning("bot_policy_rate not found — BOT_RATE_* features skipped.")

    # Thai Bank Basket (equal-weighted return)
    bank_ret_cols = [c for c in bank_returns.columns if c.endswith("_ret")]
    if bank_ret_cols:
        result["THAI_BANK_BASKET_ret"] = bank_returns[bank_ret_cols].mean(axis=1)
        log.info("THAI_BANK_BASKET_ret created from %d bank returns.", len(bank_ret_cols))
    else:
        log.warning("No bank return columns found — THAI_BANK_BASKET_ret skipped.")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_engineering():
    """Transform weekly prices and macro into analysis-ready features."""
    log.info("=" * 60)
    log.info("Phase 2 — Feature Engineering")
    log.info("=" * 60)

    # ── Load weekly cleaned data ──────────────────────────────────────────────
    weekly_prices = pd.read_csv(PATHS["weekly_prices"], index_col=0, parse_dates=True)
    weekly_macro  = pd.read_csv(PATHS["macro_weekly"],  index_col=0, parse_dates=True)

    # ── Transform prices → log returns ───────────────────────────────────────
    bank_returns  = log_return(weekly_prices, THAI_BANK_TICKERS)
    etf_returns   = log_return(weekly_prices, ETF_TICKERS)
    fx_returns    = log_return(weekly_prices, FX_TICKERS)
    index_returns = log_return(weekly_prices, INDEX_TICKERS)

    all_price_returns = pd.concat([bank_returns, etf_returns, fx_returns, index_returns], axis=1)

    # Save weekly_returns (price columns only, renamed)
    all_price_returns.dropna(how="all").to_csv(PATHS["weekly_returns"])
    log.info("Saved: %s  shape=%s", PATHS["weekly_returns"], all_price_returns.shape)

    # ── Transform macro → rate changes ───────────────────────────────────────
    rate_changes = rate_change(weekly_macro, RATE_COLS)
    vix_feats    = vix_features(weekly_macro)
    derived      = derived_factors(weekly_macro, bank_returns)

    # ── Assemble final dataset ────────────────────────────────────────────────
    final = pd.concat(
        [all_price_returns, rate_changes, vix_feats, derived],
        axis=1
    )

    # Drop first row (NaN row from differencing) and any rows with ALL NaN
    final = final.iloc[1:]
    final = final.dropna(how="all")

    # ── Confirm no raw price columns contaminate the final dataset ────────────
    raw_price_cols = [c for c in final.columns
                      if not c.endswith("_ret")
                      and not c.endswith("_chg")
                      and not c.endswith("_CHANGE")
                      and not c.endswith("_HIKE")
                      and not c.endswith("_CUT")
                      and c not in ("VIX_LEVEL",)]
    if raw_price_cols:
        log.error("RAW PRICE COLUMNS DETECTED IN FINAL DATASET: %s", raw_price_cols)
        log.error("These will be dropped to prevent using levels in correlation.")
        final = final.drop(columns=raw_price_cols)
    else:
        log.info("Confirmed: no raw price columns in final dataset.")

    # ── Outlier flags on returns/changes ──────────────────────────────────────
    # Exclude VIX_LEVEL from outlier check (it's a level series, kept for context)
    cols_for_outlier = [c for c in final.columns if c != "VIX_LEVEL"]
    flags_matrix = flag_outliers(final[cols_for_outlier], OUTLIER_ZSCORE_THRESHOLD)
    flags_long   = outlier_flag_long(flags_matrix)
    flags_long.to_csv(PATHS["outlier_flags"], index=False)
    log.info("Saved (updated): %s  (%d flagged rows)", PATHS["outlier_flags"], len(flags_long))

    # ── Final dataset summary ─────────────────────────────────────────────────
    log.info("\nFinal dataset shape: %s", final.shape)
    log.info("Date range: %s → %s", final.index[0].date(), final.index[-1].date())
    log.info("Columns:\n  %s", "\n  ".join(final.columns.tolist()))

    # ── Save ──────────────────────────────────────────────────────────────────
    final.to_csv(PATHS["final_weekly_dataset"])
    log.info("Saved: %s", PATHS["final_weekly_dataset"])

    log.info("Feature engineering complete.")
    return final


if __name__ == "__main__":
    run_feature_engineering()
