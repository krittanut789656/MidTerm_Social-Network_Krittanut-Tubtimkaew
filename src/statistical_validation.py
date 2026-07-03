"""
statistical_validation.py — Pairwise correlations with FDR (Benjamini-Hochberg) correction.

Outputs:
    data/graph/correlation_edges_validated.csv   (FDR p<0.05, |r|>=0.20)
    data/graph/correlation_edges_simple_threshold.csv  (|r|>=0.35, no stat filter)
"""

import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CORR_THRESHOLD_FDR, CORR_THRESHOLD_SIMPLE, PATHS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# These columns must NOT enter correlation analysis
# FEDFUNDS_chg excluded: monthly FRED series → weekly ffill(limit=7) → std=0 after .diff() → NaN corr
EXCLUDE_FROM_CORR = {"VIX_LEVEL", "BOT_RATE_HIKE", "BOT_RATE_CUT", "FEDFUNDS_chg"}


def get_corr_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_FROM_CORR]


def pairwise_correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = get_corr_cols(df)
    records = []
    for c1, c2 in combinations(cols, 2):
        sub = df[[c1, c2]].dropna()
        n = len(sub)
        if n < 20:
            continue
        r_p, p_p = stats.pearsonr(sub[c1], sub[c2])
        r_s, p_s = stats.spearmanr(sub[c1], sub[c2])
        records.append({
            "source": c1, "target": c2,
            "n_obs": n,
            "correlation_pearson":  round(r_p, 6),
            "p_value_pearson":      p_p,
            "correlation_spearman": round(r_s, 6),
            "p_value_spearman":     p_s,
        })
    return pd.DataFrame(records)


def apply_fdr(corr_df: pd.DataFrame) -> pd.DataFrame:
    if corr_df.empty:
        corr_df["p_adjusted_pearson"]  = pd.Series(dtype=float)
        corr_df["p_adjusted_spearman"] = pd.Series(dtype=float)
        return corr_df
    _, p_adj_p, _, _ = multipletests(corr_df["p_value_pearson"].fillna(1),  method="fdr_bh")
    _, p_adj_s, _, _ = multipletests(corr_df["p_value_spearman"].fillna(1), method="fdr_bh")
    corr_df["p_adjusted_pearson"]  = p_adj_p
    corr_df["p_adjusted_spearman"] = p_adj_s
    return corr_df


def run_statistical_validation(df: pd.DataFrame | None = None,
                                label: str = "full") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    df    : pre-loaded final_weekly_dataset (optional, loaded from PATHS if None)
    label : used for logging (e.g. 'full', 'regime_hiking', 'regime_cutting')
    """
    if df is None:
        df = pd.read_csv(PATHS["final_weekly_dataset"], index_col=0, parse_dates=True)

    start_date = str(df.index[0].date())
    end_date   = str(df.index[-1].date())
    n_obs      = len(df)

    log.info("[%s] Computing pairwise correlations for %d obs...", label, n_obs)

    corr_df = pairwise_correlations(df)
    corr_df = apply_fdr(corr_df)

    corr_df["sign"]       = corr_df["correlation_pearson"].apply(lambda x: "positive" if x >= 0 else "negative")
    corr_df["weight"]     = corr_df["correlation_pearson"].abs()
    corr_df["method"]     = "fdr_validated_pearson_weekly"
    corr_df["start_date"] = start_date
    corr_df["end_date"]   = end_date
    corr_df["frequency"]  = "weekly"

    # FDR-validated edges
    validated = corr_df[
        (corr_df["p_adjusted_pearson"] < 0.05) &
        (corr_df["correlation_pearson"].abs() >= CORR_THRESHOLD_FDR)
    ].copy().reset_index(drop=True)

    # Simple threshold edges (no stat filter — for teaching/explainability)
    simple = corr_df[
        corr_df["correlation_pearson"].abs() >= CORR_THRESHOLD_SIMPLE
    ].copy().reset_index(drop=True)

    log.info("[%s] Pairs=%d | Validated=%d | Simple(>=%.2f)=%d",
             label, len(corr_df), len(validated), CORR_THRESHOLD_SIMPLE, len(simple))

    if label == "full":
        validated.to_csv(PATHS["corr_edges_validated"], index=False)
        simple.to_csv(PATHS["corr_edges_simple"], index=False)
        log.info("Saved: %s", PATHS["corr_edges_validated"])
        log.info("Saved: %s", PATHS["corr_edges_simple"])

    return corr_df, validated, simple


if __name__ == "__main__":
    run_statistical_validation()
