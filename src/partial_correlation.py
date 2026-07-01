"""
partial_correlation.py — Partial correlation network via GraphicalLassoCV.
Falls back to LedoitWolf if Lasso is unstable.

Outputs:
    data/graph/partial_correlation_matrix.csv
    data/graph/partial_correlation_edges.csv
    data/graph/raw_vs_partial_comparison.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PARTIAL_CORR_THRESHOLD, PATHS
from src.statistical_validation import EXCLUDE_FROM_CORR

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def compute_partial_corr(df: pd.DataFrame) -> np.ndarray:
    """
    Compute partial correlation matrix from precision matrix (inverse covariance).
    partial_corr[i,j] = -precision[i,j] / sqrt(precision[i,i] * precision[j,j])
    """
    # Standardise
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    # Try GraphicalLassoCV first
    try:
        model = GraphicalLassoCV(cv=5, max_iter=500)
        model.fit(X)
        precision = model.precision_
        log.info("GraphicalLassoCV succeeded.")
    except Exception as e:
        log.warning("GraphicalLassoCV failed (%s) — falling back to LedoitWolf.", e)
        lw = LedoitWolf()
        lw.fit(X)
        precision = lw.precision_

    # Partial correlation from precision matrix
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)
    return partial_corr


def run_partial_correlation(df: pd.DataFrame | None = None,
                             raw_edge_count: int | None = None,
                             label: str = "full") -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None:
        df = pd.read_csv(PATHS["final_weekly_dataset"], index_col=0, parse_dates=True)

    cols = [c for c in df.columns if c not in EXCLUDE_FROM_CORR]
    data = df[cols].dropna(how="all")

    # Drop columns with too many missing values (>30%)
    threshold = 0.70
    data = data.loc[:, data.notna().mean() >= threshold]
    data = data.dropna()
    cols = data.columns.tolist()
    n_obs = len(data)

    log.info("[%s] Partial corr: %d obs × %d cols", label, n_obs, len(cols))

    partial_corr = compute_partial_corr(data)

    # Build matrix DataFrame
    pcorr_df = pd.DataFrame(partial_corr, index=cols, columns=cols)

    # Build edge list
    edges = []
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if j <= i:
                continue
            pc = partial_corr[i, j]
            if abs(pc) >= PARTIAL_CORR_THRESHOLD:
                edges.append({
                    "source":              c1,
                    "target":              c2,
                    "partial_correlation": round(pc, 6),
                    "weight":              round(abs(pc), 6),
                    "sign":                "positive" if pc >= 0 else "negative",
                    "method":              "graphical_lasso_weekly",
                    "n_obs":               n_obs,
                })

    edges_df = pd.DataFrame(edges)
    log.info("[%s] Partial corr edges (|pc|>=%.2f): %d",
             label, PARTIAL_CORR_THRESHOLD, len(edges_df))

    if label == "full":
        pcorr_df.to_csv(PATHS["partial_corr_matrix"])
        edges_df.to_csv(PATHS["partial_corr_edges"], index=False)
        log.info("Saved: %s", PATHS["partial_corr_matrix"])
        log.info("Saved: %s", PATHS["partial_corr_edges"])

        # Raw vs partial comparison
        if raw_edge_count is not None:
            survival_rate = len(edges_df) / raw_edge_count * 100 if raw_edge_count > 0 else 0
            comparison = pd.DataFrame([{
                "network":           "Raw Correlation (validated)",
                "edge_count":        raw_edge_count,
                "threshold":         f"|r|>={PARTIAL_CORR_THRESHOLD}",
            }, {
                "network":           "Partial Correlation",
                "edge_count":        len(edges_df),
                "threshold":         f"|pc|>={PARTIAL_CORR_THRESHOLD}",
            }, {
                "network":           "Edge Survival Rate",
                "edge_count":        f"{survival_rate:.1f}%",
                "threshold":         "partial / raw × 100",
            }])
            comparison.to_csv(PATHS["raw_vs_partial"], index=False)
            log.info("Edge survival rate: %.1f%%", survival_rate)

    return pcorr_df, edges_df


if __name__ == "__main__":
    _, validated, _ = __import__("src.statistical_validation",
                                  fromlist=["run_statistical_validation"]).run_statistical_validation()
    run_partial_correlation(raw_edge_count=len(validated))
