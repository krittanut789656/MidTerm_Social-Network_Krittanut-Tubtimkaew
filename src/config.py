"""
config.py — Central configuration for Thai Bank Graph Analysis project.
All paths, tickers, series IDs, and constants live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Project root ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ─── Data paths ──────────────────────────────────────────────────────────────
DATA_RAW        = ROOT_DIR / "data" / "raw"
DATA_PROCESSED  = ROOT_DIR / "data" / "processed"
DATA_GRAPH      = ROOT_DIR / "data" / "graph"
DATA_RESULTS    = ROOT_DIR / "data" / "results"
OUTPUTS_DIR     = ROOT_DIR / "outputs"
FIGURES_DIR     = OUTPUTS_DIR / "figures"
TABLES_DIR      = OUTPUTS_DIR / "tables"

for _p in [DATA_RAW, DATA_PROCESSED, DATA_GRAPH, DATA_RESULTS, FIGURES_DIR, TABLES_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ─── Sample period ────────────────────────────────────────────────────────────
START_DATE = "2022-05-01"   # SCB/SCBX continuity — never go earlier
END_DATE   = None           # None = latest available

# ─── Thai banking stocks ─────────────────────────────────────────────────────
THAI_BANK_TICKERS = [
    "BBL.BK",
    "KBANK.BK",
    "KKP.BK",
    "KTB.BK",
    "SCB.BK",
    "TISCO.BK",
    "TTB.BK",
]

# ─── Global financial ETFs ───────────────────────────────────────────────────
ETF_TICKERS = ["XLF", "EUFN"]

# ─── FX and market index ─────────────────────────────────────────────────────
FX_TICKERS    = ["USDTHB=X"]
INDEX_TICKERS = ["^SET.BK"]

# All Yahoo Finance tickers combined
ALL_PRICE_TICKERS = THAI_BANK_TICKERS + ETF_TICKERS + FX_TICKERS + INDEX_TICKERS

# ─── FRED macro series ───────────────────────────────────────────────────────
FRED_SERIES = {
    "FEDFUNDS":      "Fed Funds Rate",
    "DGS2":          "US 2Y Treasury Yield",
    "DGS10":         "US 10Y Treasury Yield",
    "MORTGAGE30US":  "US 30Y Mortgage Rate",
    "VIXCLS":        "VIX Close",
}

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ─── BOT (Bank of Thailand) ───────────────────────────────────────────────────
# Primary: attempt BOT API. Fallback: user-supplied CSV/XLSX.
BOT_API_KEY      = os.getenv("BOT_API_KEY", "")
BOT_MANUAL_PATH  = DATA_RAW / "bot_manual.csv"   # user drops file here if API fails

BOT_SERIES = {
    "bot_policy_rate":    "BOT Policy Rate",
    "bot_interbank_rate": "Thai Interbank Overnight Rate",
    "bibor_1m":           "BIBOR 1M",
    "thai_gov_2y":        "Thai 2Y Government Bond Yield",
    "thai_gov_10y":       "Thai 10Y Government Bond Yield",
}

# Minimum required BOT variable (others are best-effort)
BOT_REQUIRED = "bot_policy_rate"

# ─── Derived / computed columns ──────────────────────────────────────────────
DERIVED_FACTORS = {
    "US_YIELD_CURVE":    ("DGS10", "DGS2"),         # DGS10 - DGS2
    "THAI_YIELD_CURVE":  ("thai_gov_10y", "thai_gov_2y"),  # optional
    "THAI_BANK_BASKET":  THAI_BANK_TICKERS,          # equal-weighted return
}

# ─── Correlation / edge thresholds ───────────────────────────────────────────
CORR_THRESHOLD_FDR      = 0.20   # validated network
CORR_THRESHOLD_SIMPLE   = 0.35   # simple threshold network (teaching)
PARTIAL_CORR_THRESHOLD  = 0.15
LAGGED_CORR_THRESHOLD   = 0.15   # exploratory threshold for graph edges
OLS_PVALUE_THRESHOLD    = 0.05

# ─── Outlier detection ───────────────────────────────────────────────────────
OUTLIER_ZSCORE_THRESHOLD = 3.5   # flag but do not remove

# ─── Regime detection ────────────────────────────────────────────────────────
REGIME_LOOKBACK_WEEKS = 26

# ─── Neo4j connection ────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ─── Output file paths (canonical) ───────────────────────────────────────────
PATHS = {
    # raw
    "raw_prices":               DATA_RAW        / "raw_prices.csv",
    "macro_raw":                DATA_RAW        / "macro_raw.csv",
    # processed
    "weekly_prices":            DATA_PROCESSED  / "weekly_prices.csv",
    "weekly_returns":           DATA_PROCESSED  / "weekly_returns.csv",
    "macro_weekly":             DATA_PROCESSED  / "macro_weekly.csv",
    "final_weekly_dataset":     DATA_PROCESSED  / "final_weekly_dataset.csv",
    "missing_value_summary":    DATA_PROCESSED  / "missing_value_summary.csv",
    "outlier_flags":            DATA_PROCESSED  / "outlier_flags.csv",
    # graph
    "corr_edges_validated":     DATA_GRAPH      / "correlation_edges_validated.csv",
    "corr_edges_simple":        DATA_GRAPH      / "correlation_edges_simple_threshold.csv",
    "partial_corr_edges":       DATA_GRAPH      / "partial_correlation_edges.csv",
    "partial_corr_matrix":      DATA_GRAPH      / "partial_correlation_matrix.csv",
    "lagged_corr_edges":        DATA_GRAPH      / "lagged_correlation_edges.csv",
    "lagged_corr_all_pairs":    DATA_GRAPH      / "lagged_correlation_all_pairs.csv",
    "factor_exposure_edges":    DATA_GRAPH      / "factor_exposure_edges.csv",
    "raw_vs_partial":           DATA_GRAPH      / "raw_vs_partial_comparison.csv",
    "neo4j_nodes":              DATA_GRAPH      / "neo4j_nodes.csv",
    "neo4j_edges":              DATA_GRAPH      / "neo4j_edges.csv",
    # results
    "centrality_results":       DATA_RESULTS    / "centrality_results.csv",
    "community_louvain":        DATA_RESULTS    / "community_results_louvain.csv",
    "community_kmeans":         DATA_RESULTS    / "community_results_kmeans.csv",
    "regime_labels":            DATA_RESULTS    / "regime_labels.csv",
    "regime_comparison":        DATA_RESULTS    / "regime_comparison_summary.csv",
}
