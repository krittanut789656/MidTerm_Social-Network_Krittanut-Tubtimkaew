"""
data_cleaning.py — Clean and align raw data to weekly frequency.

Steps:
  1. Remove duplicate dates and columns
  2. Resample to weekly (Friday / last available)
  3. Forward-fill macro variables (max 5 business days → 1 week)
  4. Do NOT forward-fill stock prices
  5. Generate missing value summary
  6. Flag outliers by z-score (do not remove)
  7. Check SCB data continuity from May 2022

Entry point: run_cleaning()
Outputs:
    data/processed/weekly_prices.csv
    data/processed/macro_weekly.csv
    data/processed/missing_value_summary.csv
    data/processed/outlier_flags.csv
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
    START_DATE,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

PRICE_TICKERS = THAI_BANK_TICKERS + ETF_TICKERS + FX_TICKERS + INDEX_TICKERS
MACRO_SERIES  = ["FEDFUNDS", "DGS2", "DGS10", "MORTGAGE30US", "VIXCLS"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _drop_duplicate_dates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    n_dups = df.index.duplicated().sum()
    if n_dups:
        log.warning("%s: %d duplicate date(s) removed.", label, n_dups)
        df = df[~df.index.duplicated(keep="last")]
    return df


def _drop_duplicate_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    n_dups = df.columns.duplicated().sum()
    if n_dups:
        log.warning("%s: %d duplicate column(s) removed.", label, n_dups)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def _resample_prices_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Use last available price in each week (ending Friday)."""
    return df.resample("W-FRI").last()


def _resample_macro_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use last available value in each week.
    Forward-fill up to 7 calendar days (handles FRED reporting lags).
    """
    # First fill daily gaps (≤7 days) then resample
    df_filled = df.asfreq("D").ffill(limit=7)
    return df_filled.resample("W-FRI").last()


def _align_to_price_index(macro_weekly: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex macro to match price week-end dates, forward-fill ≤1 week."""
    return macro_weekly.reindex(price_index).ffill(limit=1)


# ─────────────────────────────────────────────────────────────────────────────
# Missing value summary
# ─────────────────────────────────────────────────────────────────────────────

def make_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-column missing value report."""
    total = len(df)
    summary = pd.DataFrame({
        "column":         df.columns,
        "total_rows":     total,
        "missing_count":  df.isna().sum().values,
        "missing_pct":    (df.isna().mean() * 100).round(2).values,
        "first_valid":    [df[c].first_valid_index() for c in df.columns],
        "last_valid":     [df[c].last_valid_index()  for c in df.columns],
    })
    summary = summary.sort_values("missing_pct", ascending=False).reset_index(drop=True)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Outlier flagging
# ─────────────────────────────────────────────────────────────────────────────

def flag_outliers(df: pd.DataFrame, threshold: float = OUTLIER_ZSCORE_THRESHOLD) -> pd.DataFrame:
    """
    Flag cells where |z-score| > threshold.
    Returns a boolean DataFrame (True = flagged outlier).
    Does NOT remove any values.
    """
    flags = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        z = (s - s.mean()) / s.std(ddof=1)
        outlier_dates = z[z.abs() > threshold].index
        flags.loc[outlier_dates, col] = True

    n_flagged = flags.sum().sum()
    pct = n_flagged / (df.shape[0] * df.shape[1]) * 100
    log.info("Outlier flags: %d cells flagged (%.2f%% of dataset, z > %.1f)",
             n_flagged, pct, threshold)
    return flags


def outlier_flag_long(flags: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean flag matrix to a long table for the report."""
    records = []
    for col in flags.columns:
        flagged_dates = flags.index[flags[col]]
        for d in flagged_dates:
            records.append({"date": d, "column": col, "outlier_flag": True})
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# SCB continuity check
# ─────────────────────────────────────────────────────────────────────────────

def check_scb_continuity(prices_weekly: pd.DataFrame) -> None:
    col = "SCB.BK"
    if col not in prices_weekly.columns:
        log.warning("SCB.BK not found in price data — skipping continuity check.")
        return

    scb = prices_weekly[col].dropna()
    if scb.empty:
        log.error("SCB.BK has NO valid price data.")
        return

    # Check for large weekly price gaps that could indicate structural break
    log.info("SCB.BK price check: first=%s (%.2f)  last=%s (%.2f)  n_obs=%d",
             scb.index[0].date(), scb.iloc[0],
             scb.index[-1].date(), scb.iloc[-1],
             len(scb))

    pct_changes = scb.pct_change().dropna()
    extreme = pct_changes[pct_changes.abs() > 0.20]
    if not extreme.empty:
        log.warning("SCB.BK has %d weekly price change(s) > 20%% — verify for structural breaks:",
                    len(extreme))
        for d, v in extreme.items():
            log.warning("  %s  pct_change=%.2f%%", d.date(), v * 100)
    else:
        log.info("SCB.BK: no weekly price jumps > 20%% — continuity looks OK.")


# ─────────────────────────────────────────────────────────────────────────────
# Main cleaning pipeline
# ─────────────────────────────────────────────────────────────────────────────

def clean_prices(raw_prices: pd.DataFrame) -> pd.DataFrame:
    df = raw_prices.copy()
    df.index = pd.to_datetime(df.index)
    df = _drop_duplicate_dates(df, "raw_prices")
    df = _drop_duplicate_columns(df, "raw_prices")
    df = df.sort_index()
    df = df[df.index >= START_DATE]

    # Keep only expected price tickers (ignore extra columns)
    available = [t for t in PRICE_TICKERS if t in df.columns]
    missing_tickers = [t for t in PRICE_TICKERS if t not in df.columns]
    if missing_tickers:
        log.warning("Missing price tickers: %s", missing_tickers)
    df = df[available]

    weekly = _resample_prices_weekly(df)
    log.info("Prices resampled to weekly: shape=%s", weekly.shape)
    return weekly


def clean_macro(raw_macro: pd.DataFrame) -> pd.DataFrame:
    df = raw_macro.copy()
    df.index = pd.to_datetime(df.index)
    df = _drop_duplicate_dates(df, "raw_macro")
    df = _drop_duplicate_columns(df, "raw_macro")
    df = df.sort_index()
    df = df[df.index >= START_DATE]

    weekly = _resample_macro_weekly(df)
    log.info("Macro resampled to weekly: shape=%s", weekly.shape)
    return weekly


def run_cleaning():
    """Load raw CSVs, clean, and produce weekly price and macro files."""
    log.info("=" * 60)
    log.info("Phase 2 — Data Cleaning")
    log.info("=" * 60)

    # ── Load raw ──────────────────────────────────────────────────────────────
    raw_prices = pd.read_csv(PATHS["raw_prices"],  index_col=0, parse_dates=True)
    raw_macro  = pd.read_csv(PATHS["macro_raw"],   index_col=0, parse_dates=True)

    log.info("raw_prices shape: %s", raw_prices.shape)
    log.info("raw_macro  shape: %s", raw_macro.shape)

    # ── Clean ─────────────────────────────────────────────────────────────────
    weekly_prices = clean_prices(raw_prices)
    weekly_macro  = clean_macro(raw_macro)

    # Align macro to price weekly index
    weekly_macro = _align_to_price_index(weekly_macro, weekly_prices.index)

    # ── SCB continuity check ──────────────────────────────────────────────────
    check_scb_continuity(weekly_prices)

    # ── Missing value summary ─────────────────────────────────────────────────
    combined_for_summary = weekly_prices.join(weekly_macro, how="outer")
    missing_summary = make_missing_summary(combined_for_summary)
    missing_summary.to_csv(PATHS["missing_value_summary"], index=False)
    log.info("Saved: %s", PATHS["missing_value_summary"])
    log.info("\n%s", missing_summary[missing_summary["missing_pct"] > 0].to_string(index=False))

    # ── Outlier flags on raw price levels ────────────────────────────────────
    # NOTE: Outlier detection on price levels here (for reporting).
    # Actual trading is done on returns (in feature_engineering.py).
    # We flag but never remove.
    flags_matrix = flag_outliers(combined_for_summary)
    flags_long   = outlier_flag_long(flags_matrix)
    flags_long.to_csv(PATHS["outlier_flags"], index=False)
    log.info("Saved: %s  (%d flagged rows)", PATHS["outlier_flags"], len(flags_long))

    # ── Save weekly cleaned files ─────────────────────────────────────────────
    weekly_prices.to_csv(PATHS["weekly_prices"])
    weekly_macro.to_csv(PATHS["macro_weekly"])
    log.info("Saved: %s  shape=%s", PATHS["weekly_prices"], weekly_prices.shape)
    log.info("Saved: %s  shape=%s", PATHS["macro_weekly"],  weekly_macro.shape)

    log.info("Cleaning complete.")
    return weekly_prices, weekly_macro


if __name__ == "__main__":
    run_cleaning()
