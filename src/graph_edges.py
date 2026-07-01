"""
graph_edges.py — Build lagged correlation edges, OLS factor exposure edges,
regime-aware networks, and assemble Neo4j node/edge CSVs.

Outputs:
    data/graph/lagged_correlation_edges.csv
    data/graph/factor_exposure_edges.csv
    data/graph/neo4j_nodes.csv
    data/graph/neo4j_edges.csv
    data/results/regime_comparison_summary.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import LAGGED_CORR_THRESHOLD, OLS_PVALUE_THRESHOLD, PATHS
from src.statistical_validation import run_statistical_validation, EXCLUDE_FROM_CORR
from src.partial_correlation import run_partial_correlation
from src.regime_detection import REGIME_HIKING, REGIME_CUTTING

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Node schema ───────────────────────────────────────────────────────────────
THAI_BANKS = ["BBL_ret", "KBANK_ret", "KKP_ret", "KTB_ret", "SCB_ret", "TISCO_ret", "TTB_ret"]

COL_TO_NODE = {
    "BBL_ret":            "BBL",
    "KBANK_ret":          "KBANK",
    "KKP_ret":            "KKP",
    "KTB_ret":            "KTB",
    "SCB_ret":            "SCB",
    "TISCO_ret":          "TISCO",
    "TTB_ret":            "TTB",
    "XLF_ret":            "XLF",
    "EUFN_ret":           "EUFN",
    "USDTHB_ret":         "USDTHB",
    "SET_ret":            "SET",
    "FEDFUNDS_chg":       "FEDFUNDS",
    "DGS2_chg":           "DGS2",
    "DGS10_chg":          "DGS10",
    "MORTGAGE30US_chg":   "MORTGAGE30US",
    "VIX_CHANGE":         "VIX",
    "US_YIELD_CURVE_chg": "US_YIELD_CURVE",
    "BOT_RATE_CHANGE":    "BOT_RATE",
    "THAI_BANK_BASKET_ret": "THAI_BANK_BASKET",
}

NODE_META = {
    "BBL":             ("Bank",          "BBL",            "BBL.BK"),
    "KBANK":           ("Bank",          "KBANK",          "KBANK.BK"),
    "KKP":             ("Bank",          "KKP",            "KKP.BK"),
    "KTB":             ("Bank",          "KTB",            "KTB.BK"),
    "SCB":             ("Bank",          "SCB",            "SCB.BK"),
    "TISCO":           ("Bank",          "TISCO",          "TISCO.BK"),
    "TTB":             ("Bank",          "TTB",            "TTB.BK"),
    "XLF":             ("ETF",           "XLF",            "XLF"),
    "EUFN":            ("ETF",           "EUFN",           "EUFN"),
    "USDTHB":          ("FX",            "USDTHB",         "USDTHB=X"),
    "SET":             ("Index",         "SET Index",      "^SET.BK"),
    "FEDFUNDS":        ("MacroFactor",   "Fed Funds Rate", "FEDFUNDS"),
    "DGS2":            ("MacroFactor",   "US 2Y Yield",    "DGS2"),
    "DGS10":           ("MacroFactor",   "US 10Y Yield",   "DGS10"),
    "MORTGAGE30US":    ("MacroFactor",   "US 30Y Mortgage","MORTGAGE30US"),
    "VIX":             ("MacroFactor",   "VIX",            "VIXCLS"),
    "US_YIELD_CURVE":  ("DerivedFactor", "US Yield Curve", "DGS10-DGS2"),
    "BOT_RATE":        ("MacroFactor",   "BOT Policy Rate","BOT_RATE"),
    "THAI_BANK_BASKET":("DerivedFactor", "Thai Bank Basket","THAI_BANK_BASKET"),
}

# OLS factor columns (order matters for model)
OLS_FACTORS = [
    "XLF_ret", "EUFN_ret", "SET_ret", "USDTHB_ret",
    "VIX_CHANGE", "DGS2_chg", "DGS10_chg",
    "US_YIELD_CURVE_chg", "MORTGAGE30US_chg", "BOT_RATE_CHANGE",
]


# ─────────────────────────────────────────────────────────────────────────────
# Lagged correlation edges
# ─────────────────────────────────────────────────────────────────────────────

def build_lagged_edges(df: pd.DataFrame, lags: range = range(1, 5)):
    """
    For each non-bank factor and each bank return:
        corr(factor_{t-k}, bank_t) for k in lags
    Returns:
        edges_df    — pairs with |best_corr| >= LAGGED_CORR_THRESHOLD (for graph)
        all_pairs   — all factor-bank pairs regardless of threshold (for display)
    """
    bank_cols   = [c for c in THAI_BANKS if c in df.columns]
    factor_cols = [c for c in df.columns
                   if c not in EXCLUDE_FROM_CORR
                   and c not in THAI_BANKS
                   and c not in {"THAI_BANK_BASKET_ret"}]

    all_records = []
    for fc in factor_cols:
        for bc in bank_cols:
            best_corr, best_lag = 0.0, 0
            for k in lags:
                sub = df[[fc, bc]].copy()
                sub[fc] = sub[fc].shift(k)   # factor_{t-k}
                sub = sub.dropna()
                if len(sub) < 15:
                    continue
                r = sub[fc].corr(sub[bc])
                if not np.isfinite(r):
                    continue
                if abs(r) > abs(best_corr):
                    best_corr = r
                    best_lag  = k

            src_node = COL_TO_NODE.get(fc, fc)
            tgt_node = COL_TO_NODE.get(bc, bc)
            all_records.append({
                "source":      fc,
                "target":      bc,
                "source_node": src_node,
                "target_node": tgt_node,
                "correlation": round(best_corr, 6),
                "weight":      round(abs(best_corr), 6),
                "sign":        "positive" if best_corr >= 0 else "negative",
                "lag_weeks":   best_lag,
                "method":      "lagged_corr_weekly",
                "above_threshold": abs(best_corr) >= LAGGED_CORR_THRESHOLD,
            })

    all_pairs = pd.DataFrame(all_records)
    edges_df  = all_pairs[all_pairs["above_threshold"]].drop(columns=["above_threshold"]).reset_index(drop=True)
    all_pairs = all_pairs.drop(columns=["above_threshold"])

    log.info("Lagged correlation — all pairs: %d, above threshold (|r|>=%.2f): %d",
             len(all_pairs), LAGGED_CORR_THRESHOLD, len(edges_df))
    return edges_df, all_pairs


# ─────────────────────────────────────────────────────────────────────────────
# OLS factor exposure edges
# ─────────────────────────────────────────────────────────────────────────────

def run_ols_per_bank(df: pd.DataFrame) -> pd.DataFrame:
    """OLS: each bank ~ OLS_FACTORS. Return significant factor-to-bank edges."""
    available_factors = [f for f in OLS_FACTORS if f in df.columns]
    bank_cols = [c for c in THAI_BANKS if c in df.columns]

    records = []
    for bc in bank_cols:
        cols = [bc] + available_factors
        sub = df[cols].dropna()
        if len(sub) < 20:
            log.warning("OLS: too few obs for %s (%d) — skipped.", bc, len(sub))
            continue

        y = sub[bc]
        X = sm.add_constant(sub[available_factors])
        try:
            res = sm.OLS(y, X).fit()
        except Exception as e:
            log.warning("OLS failed for %s: %s", bc, e)
            continue

        cond_number = np.linalg.cond(X.values)
        if cond_number > 1e10:
            log.warning("OLS for %s: high condition number (%.0f) — multicollinearity present.", bc, cond_number)

        for fc in available_factors:
            if fc not in res.params.index:
                continue
            beta  = res.params[fc]
            pval  = res.pvalues[fc]
            tstat = res.tvalues[fc]
            if pval <= OLS_PVALUE_THRESHOLD:
                src_node = COL_TO_NODE.get(fc, fc)
                tgt_node = COL_TO_NODE.get(bc, bc)
                records.append({
                    "source":      fc,
                    "target":      bc,
                    "source_node": src_node,
                    "target_node": tgt_node,
                    "beta":        round(beta, 6),
                    "t_stat":      round(tstat, 4),
                    "p_value":     round(pval, 6),
                    "weight":      round(abs(beta), 6),
                    "sign":        "positive" if beta >= 0 else "negative",
                    "method":      "OLS_weekly",
                    "n_obs":       len(sub),
                    "r_squared":   round(res.rsquared, 4),
                })

    edges_df = pd.DataFrame(records)
    log.info("OLS factor exposure edges (p<=%.2f): %d", OLS_PVALUE_THRESHOLD, len(edges_df))
    return edges_df


# ─────────────────────────────────────────────────────────────────────────────
# Regime-aware comparison
# ─────────────────────────────────────────────────────────────────────────────

def regime_comparison(df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    """Build validated correlation and partial corr for each regime."""
    regime_df["date"] = pd.to_datetime(regime_df["date"])
    regime_df = regime_df.set_index("date")

    df_indexed = df.copy()
    df_indexed.index = pd.to_datetime(df_indexed.index)

    records = []
    for regime_label in [REGIME_HIKING, REGIME_CUTTING]:
        regime_dates = regime_df[regime_df["regime"] == regime_label].index
        sub = df_indexed[df_indexed.index.isin(regime_dates)]

        if len(sub) < 20:
            log.warning("Regime %s has only %d obs — skipping.", regime_label, len(sub))
            continue

        log.info("Regime '%s': %d weeks", regime_label, len(sub))

        _, val_edges, _ = run_statistical_validation(sub, label=regime_label)
        _, pcorr_edges  = run_partial_correlation(sub, raw_edge_count=len(val_edges), label=regime_label)

        # Top correlated pair
        if not val_edges.empty:
            top = val_edges.nlargest(1, "weight").iloc[0]
            top_pair = f"{top['source']} ↔ {top['target']} ({top['weight']:.3f})"
        else:
            top_pair = "N/A"

        records.append({
            "regime":           regime_label,
            "n_weeks":          len(sub),
            "validated_edges":  len(val_edges),
            "partial_edges":    len(pcorr_edges),
            "edge_survival_pct": round(len(pcorr_edges) / len(val_edges) * 100, 1)
                                  if len(val_edges) > 0 else 0,
            "top_corr_pair":    top_pair,
        })

    comparison_df = pd.DataFrame(records)
    comparison_df.to_csv(PATHS["regime_comparison"], index=False)
    log.info("Saved: %s", PATHS["regime_comparison"])
    return comparison_df


# ─────────────────────────────────────────────────────────────────────────────
# Assemble Neo4j node / edge CSVs
# ─────────────────────────────────────────────────────────────────────────────

def build_neo4j_nodes(df_cols: list[str]) -> pd.DataFrame:
    node_ids = set()
    for col in df_cols:
        if col in EXCLUDE_FROM_CORR:
            continue
        node_id = COL_TO_NODE.get(col, col)
        node_ids.add(node_id)

    rows = []
    for nid in sorted(node_ids):
        if nid in NODE_META:
            ntype, name, ticker = NODE_META[nid]
        else:
            ntype, name, ticker = "Unknown", nid, nid
        rows.append({
            "node_id": nid,
            "name":    name,
            "ticker":  ticker,
            "type":    ntype,
            "label":   ntype,
        })
    return pd.DataFrame(rows)


def build_neo4j_edges(corr_edges: pd.DataFrame,
                       partial_edges: pd.DataFrame,
                       lagged_edges: pd.DataFrame,
                       factor_edges: pd.DataFrame) -> pd.DataFrame:
    all_edges = []

    for _, row in corr_edges.iterrows():
        all_edges.append({
            "source":      COL_TO_NODE.get(row["source"], row["source"]),
            "target":      COL_TO_NODE.get(row["target"], row["target"]),
            "rel_type":    "CORRELATED_WITH",
            "weight":      row.get("weight", 0),
            "sign":        row.get("sign", ""),
            "method":      row.get("method", ""),
            "corr_pearson":row.get("correlation_pearson", ""),
            "p_adj":       row.get("p_adjusted_pearson", ""),
        })

    for _, row in partial_edges.iterrows():
        all_edges.append({
            "source":   COL_TO_NODE.get(row["source"], row["source"]),
            "target":   COL_TO_NODE.get(row["target"], row["target"]),
            "rel_type": "PARTIAL_CORRELATED_WITH",
            "weight":   row.get("weight", 0),
            "sign":     row.get("sign", ""),
            "method":   row.get("method", ""),
        })

    for _, row in lagged_edges.iterrows():
        all_edges.append({
            "source":   row.get("source_node", row["source"]),
            "target":   row.get("target_node", row["target"]),
            "rel_type": "LAGGED_CORRELATED_WITH",
            "weight":   row.get("weight", 0),
            "sign":     row.get("sign", ""),
            "method":   row.get("method", ""),
            "lag_weeks":row.get("lag_weeks", ""),
        })

    for _, row in factor_edges.iterrows():
        all_edges.append({
            "source":   row.get("source_node", row["source"]),
            "target":   row.get("target_node", row["target"]),
            "rel_type": "INFLUENCES",
            "weight":   row.get("weight", 0),
            "sign":     row.get("sign", ""),
            "method":   row.get("method", ""),
            "beta":     row.get("beta", ""),
            "p_value":  row.get("p_value", ""),
        })

    return pd.DataFrame(all_edges)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_graph_edges():
    log.info("=" * 60)
    log.info("Phase 3 — Graph Edge Construction")
    log.info("=" * 60)

    # Load data
    df = pd.read_csv(PATHS["final_weekly_dataset"], index_col=0, parse_dates=True)
    regime_df = pd.read_csv(PATHS["regime_labels"], parse_dates=["date"])

    # 1. Validated correlation edges (full dataset)
    _, validated, _ = run_statistical_validation(df, label="full")

    # 2. Partial correlation edges
    _, partial_edges = run_partial_correlation(df, raw_edge_count=len(validated))

    # 3. Lagged correlation edges
    lagged_edges, lagged_all_pairs = build_lagged_edges(df)

    # Always save all-pairs (for app display table — no threshold)
    # Derive path directly to avoid stale config .pyc issues
    _all_pairs_path = PATHS["lagged_corr_edges"].parent / "lagged_correlation_all_pairs.csv"
    if lagged_all_pairs.empty:
        pd.DataFrame(columns=[
            "source", "target", "source_node", "target_node",
            "correlation", "weight", "sign", "lag_weeks", "method"
        ]).to_csv(_all_pairs_path, index=False)
    else:
        lagged_all_pairs.to_csv(_all_pairs_path, index=False)
    log.info("Saved all-pairs: %s (%d rows)", _all_pairs_path, len(lagged_all_pairs))

    # Save threshold-filtered edges (for graph + Neo4j)
    if lagged_edges.empty:
        log.warning("No lagged edges above threshold (%.2f). "
                    "Saving empty CSV with headers.", LAGGED_CORR_THRESHOLD)
        pd.DataFrame(columns=[
            "source", "target", "source_node", "target_node",
            "correlation", "weight", "sign", "lag_weeks", "method"
        ]).to_csv(PATHS["lagged_corr_edges"], index=False)
    else:
        lagged_edges.to_csv(PATHS["lagged_corr_edges"], index=False)
    log.info("Saved: %s (%d edges)", PATHS["lagged_corr_edges"], len(lagged_edges))

    # 4. OLS factor exposure edges
    factor_edges = run_ols_per_bank(df)
    if factor_edges.empty:
        log.warning("No OLS factor exposure edges (p<=%.2f). Saving empty CSV.", OLS_PVALUE_THRESHOLD)
        pd.DataFrame(columns=[
            "source", "target", "source_node", "target_node",
            "beta", "t_stat", "p_value", "weight", "sign", "method", "n_obs", "r_squared"
        ]).to_csv(PATHS["factor_exposure_edges"], index=False)
    else:
        factor_edges.to_csv(PATHS["factor_exposure_edges"], index=False)
    log.info("Saved: %s (%d edges)", PATHS["factor_exposure_edges"], len(factor_edges))

    # 5. Regime comparison
    regime_comparison(df, regime_df)

    # 6. Neo4j node/edge CSVs
    nodes_df = build_neo4j_nodes(df.columns.tolist())
    edges_df = build_neo4j_edges(validated, partial_edges, lagged_edges, factor_edges)

    nodes_df.to_csv(PATHS["neo4j_nodes"], index=False)
    edges_df.to_csv(PATHS["neo4j_edges"], index=False)
    log.info("Saved: %s (%d nodes)", PATHS["neo4j_nodes"], len(nodes_df))
    log.info("Saved: %s (%d edges)", PATHS["neo4j_edges"], len(edges_df))

    log.info("Phase 3 complete.")
    return validated, partial_edges, lagged_edges, factor_edges


if __name__ == "__main__":
    run_graph_edges()
