"""
data_download.py — Download raw data from Yahoo Finance, FRED, and BOT.

Entry point:  run_download()
Outputs:
    data/raw/raw_prices.csv
    data/raw/macro_raw.csv
"""

import logging
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── import config after adding project root to sys.path ──────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    ALL_PRICE_TICKERS,
    FRED_SERIES, FRED_API_KEY,
    BOT_API_KEY, BOT_MANUAL_PATH, BOT_SERIES,
    START_DATE, END_DATE,
    PATHS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Yahoo Finance
# ─────────────────────────────────────────────────────────────────────────────

def download_yahoo(tickers: list[str], start: str, end=None) -> pd.DataFrame:
    """Download daily adjusted close prices from Yahoo Finance."""
    log.info("Downloading Yahoo Finance tickers: %s", tickers)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    # Report download success per ticker
    for t in tickers:
        if t in prices.columns:
            n_valid = prices[t].notna().sum()
            log.info("  %-14s  %d rows", t, n_valid)
        else:
            log.warning("  %-14s  NOT FOUND in download", t)

    return prices


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FRED
# ─────────────────────────────────────────────────────────────────────────────

def download_fred(series_dict: dict, api_key: str, start: str, end=None) -> pd.DataFrame:
    """Download macro series from FRED using fredapi."""
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("fredapi not installed — run: pip install fredapi")

    if not api_key:
        raise ValueError("FRED_API_KEY is empty. Set it in your .env file.")

    fred = Fred(api_key=api_key)
    frames = {}

    for series_id, label in series_dict.items():
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s.name = series_id
            frames[series_id] = s
            log.info("  FRED %-16s  %d obs", series_id, s.notna().sum())
        except Exception as e:
            log.warning("  FRED %-16s  FAILED: %s", series_id, e)

    if not frames:
        log.error("No FRED series downloaded. Check FRED_API_KEY and internet connection.")
        return pd.DataFrame()   # return empty — pipeline continues with warning

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BOT (Bank of Thailand)
# ─────────────────────────────────────────────────────────────────────────────

def _bot_api(api_key: str, start: str, end=None) -> pd.DataFrame | None:
    """
    Attempt to download BOT data via official API.
    BOT API endpoint: https://apiportal.bot.or.th/bot/public/
    Returns DataFrame or None if unavailable.
    """
    import requests

    # BOT API series IDs (Financial Markets / Monetary Policy)
    # These IDs are subject to change — verify at bot.or.th API portal.
    BOT_ENDPOINT = "https://apiportal.bot.or.th/bot/public/financial-institutions-business/v1/"
    POLICY_RATE_URL = (
        "https://apiportal.bot.or.th/bot/public/financial-markets/v1/"
        "interestrate/policy-rate"
    )

    if not api_key:
        log.info("BOT API key not set — skipping API attempt.")
        return None

    headers = {"X-IBM-Client-Id": api_key}
    params  = {"start_period": start.replace("-", ""), "end_period": "99991231"}

    try:
        resp = requests.get(POLICY_RATE_URL, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Parse typical BOT JSON structure
        records = data.get("result", {}).get("data", [])
        if not records:
            log.warning("BOT API returned empty data.")
            return None

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["period"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
        df = df[["value"]].rename(columns={"value": "bot_policy_rate"})
        df["bot_policy_rate"] = pd.to_numeric(df["bot_policy_rate"], errors="coerce")
        log.info("BOT API  bot_policy_rate  %d obs", df["bot_policy_rate"].notna().sum())
        return df

    except Exception as e:
        log.warning("BOT API failed: %s — will try manual CSV.", e)
        return None


def _bot_manual(path: Path) -> pd.DataFrame | None:
    """Load BOT data from a manually downloaded CSV or XLSX file."""
    if not path.exists():
        log.warning("BOT manual file not found at: %s", path)
        return None

    log.info("Loading BOT data from manual file: %s", path)
    suffix = path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        log.error("Failed to read BOT manual file: %s", e)
        return None

    # Normalise column names to lowercase, strip whitespace
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Try to detect date column
    date_candidates = [c for c in df.columns if "date" in c or "period" in c or "year" in c]
    if not date_candidates:
        log.error("No date column found in BOT manual file. Columns: %s", df.columns.tolist())
        return None

    df["date"] = pd.to_datetime(df[date_candidates[0]], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    # Keep only columns that match BOT_SERIES keys (flexible partial match)
    kept = {}
    for key in BOT_SERIES.keys():
        matches = [c for c in df.columns if key in c or c in key]
        if matches:
            kept[key] = df[matches[0]].copy()

    if not kept:
        log.error("No recognisable BOT columns found. Expected keys: %s", list(BOT_SERIES.keys()))
        return None

    result = pd.DataFrame(kept)
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
        log.info("  BOT manual %-22s  %d obs", col, result[col].notna().sum())

    return result


def download_bot(api_key: str, manual_path: Path, start: str, end=None) -> pd.DataFrame:
    """Download BOT data: API first, manual CSV fallback, empty DataFrame last resort."""
    df = _bot_api(api_key, start, end)
    if df is None:
        df = _bot_manual(manual_path)
    if df is None:
        log.warning("BOT data unavailable from both API and manual file. "
                    "Creating empty placeholder.")
        df = pd.DataFrame(columns=list(BOT_SERIES.keys()))
        df.index = pd.DatetimeIndex([], name="date")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_download():
    """Download all raw data and save to data/raw/."""
    log.info("=" * 60)
    log.info("Phase 2 — Data Download")
    log.info("=" * 60)

    # ── Yahoo Finance ─────────────────────────────────────────────────────────
    prices = download_yahoo(ALL_PRICE_TICKERS, start=START_DATE, end=END_DATE)
    prices.to_csv(PATHS["raw_prices"])
    log.info("Saved: %s  shape=%s", PATHS["raw_prices"], prices.shape)

    # ── FRED ──────────────────────────────────────────────────────────────────
    macro = download_fred(FRED_SERIES, api_key=FRED_API_KEY, start=START_DATE, end=END_DATE)

    # ── BOT ───────────────────────────────────────────────────────────────────
    bot = download_bot(BOT_API_KEY, BOT_MANUAL_PATH, start=START_DATE, end=END_DATE)

    # ── Merge macro + bot into one raw macro file ─────────────────────────────
    if macro.empty and bot.empty:
        log.warning("Both FRED and BOT downloads failed. macro_raw will be empty.")
        macro_raw = pd.DataFrame()
    elif macro.empty:
        macro_raw = bot.copy()
    elif bot.empty:
        macro_raw = macro.copy()
    else:
        macro_raw = macro.join(bot, how="outer")

    macro_raw.to_csv(PATHS["macro_raw"])
    log.info("Saved: %s  shape=%s", PATHS["macro_raw"], macro_raw.shape)

    log.info("Download complete.")
    return prices, macro_raw


if __name__ == "__main__":
    run_download()
