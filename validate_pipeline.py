"""
validate_pipeline.py — Automated sanity-check script for the Thai Bank Graph Analysis pipeline.

Checks every stage: data integrity, statistical validity, graph consistency, and report accuracy.

Usage:
    python validate_pipeline.py

Output: printed PASS/FAIL for each check, plus a summary table.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import PATHS, CORR_THRESHOLD_FDR, PARTIAL_CORR_THRESHOLD, LAGGED_CORR_THRESHOLD

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results = []


def check(name: str, passed: bool, detail: str = "", warn_only: bool = False):
    status = PASS if passed else (WARN if warn_only else FAIL)
    results.append((status, name, detail))
    icon = "✅" if status == PASS else ("⚠️" if status == WARN else "❌")
    print(f"  {icon} [{status}] {name}" + (f" — {detail}" if detail else ""))


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. FILE EXISTENCE
# ─────────────────────────────────────────────────────────────────────────────
section("1. File Existence")

required_files = {
    "final_weekly_dataset":  PATHS["final_weekly_dataset"],
    "macro_weekly":          PATHS["macro_weekly"],
    "missing_value_summary": PATHS["missing_value_summary"],
    "outlier_flags":         PATHS["outlier_flags"],
    "corr_edges_validated":  PATHS["corr_edges_validated"],
    "partial_corr_edges":    PATHS["partial_corr_edges"],
    "partial_corr_matrix":   PATHS["partial_corr_matrix"],
    "lagged_corr_edges":     PATHS["lagged_corr_edges"],
    "factor_exposure_edges": PATHS["factor_exposure_edges"],
    "neo4j_nodes":           PATHS["neo4j_nodes"],
    "neo4j_edges":           PATHS["neo4j_edges"],
    "regime_labels":         PATHS["regime_labels"],
    "regime_comparison":     PATHS["regime_comparison"],
    "centrality_results":    PATHS["centrality_results"],
    "community_louvain":     PATHS["community_louvain"],
    "community_kmeans":      PATHS["community_kmeans"],
}

for name, path in required_files.items():
    check(f"File exists: {path.name}", path.exists())


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────
section("2. Data Integrity")

try:
    df = pd.read_csv(PATHS["final_weekly_dataset"], index_col=0, parse_dates=True)

    # Weekly frequency
    check("Weekly index (W-FRI freq)",
          df.index.dayofweek.value_counts().idxmax() == 4,
          f"Most common day: {df.index.dayofweek.value_counts().idxmax()} (4=Friday)")

    # Sample starts at or after 2022-05-01
    check("Start date >= 2022-05-01",
          df.index[0] >= pd.Timestamp("2022-05-01"),
          str(df.index[0].date()))

    # Minimum 100 weeks
    check("At least 100 weeks of data",
          len(df) >= 100,
          f"{len(df)} weeks")

    # No raw price columns (column names should end in _ret, _chg, _CHANGE, _LEVEL)
    allowed_suffixes = ("_ret", "_chg", "_CHANGE", "_LEVEL")
    raw_cols = [c for c in df.columns if not any(c.endswith(s) for s in allowed_suffixes)]
    check("No raw price columns in final dataset",
          len(raw_cols) == 0,
          f"Unexpected: {raw_cols}" if raw_cols else "All columns correctly transformed")

    # Key columns present
    expected_cols = ["BBL_ret", "KBANK_ret", "VIX_CHANGE", "FEDFUNDS_chg",
                     "DGS2_chg", "DGS10_chg", "US_YIELD_CURVE_chg", "USDTHB_ret"]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    check("Key feature columns present",
          len(missing_cols) == 0,
          f"Missing: {missing_cols}" if missing_cols else f"{df.shape[1]} columns total")

    # No all-NaN rows (after first row)
    all_nan_rows = df.iloc[1:].isnull().all(axis=1).sum()
    check("No all-NaN rows",
          all_nan_rows == 0,
          f"{all_nan_rows} all-NaN rows found")

    # Missing per column — monthly series (FEDFUNDS) may have up to 60%
    # because ffill(limit=7) doesn't bridge a full month between Fed reports
    miss = df.isnull().mean()
    non_monthly = miss.drop("FEDFUNDS_chg", errors="ignore")
    bad_cols = non_monthly[non_monthly > 0.30]
    check("Non-monthly columns: missing <= 30%",
          len(bad_cols) == 0,
          f"High-missing cols: {bad_cols.to_dict()}" if len(bad_cols) else "OK")
    fedfunds_miss = miss.get("FEDFUNDS_chg", 0)
    check("FEDFUNDS_chg missing <= 60% (monthly series — expected)",
          fedfunds_miss <= 0.60,
          f"FEDFUNDS_chg missing: {fedfunds_miss:.1%}",
          warn_only=fedfunds_miss <= 0.60)

    # Returns are weekly-magnitude (not daily or annualized)
    bank_cols = [c for c in df.columns if c.endswith("_ret") and
                 c.replace("_ret", "") in ["BBL", "KBANK", "KKP", "KTB", "SCB", "TISCO", "TTB"]]
    if bank_cols:
        abs_mean = df[bank_cols].abs().mean().mean()
        check("Bank returns are weekly magnitude (0.001–0.05 typical)",
              0.001 <= abs_mean <= 0.10,
              f"Mean |return| = {abs_mean:.4f}",
              warn_only=not (0.001 <= abs_mean <= 0.10))

    # US_YIELD_CURVE = DGS10 - DGS2 (verify derivation)
    if "US_YIELD_CURVE_chg" in df.columns and "DGS10_chg" in df.columns and "DGS2_chg" in df.columns:
        macro = pd.read_csv(PATHS["macro_weekly"])
        if "DGS10" in macro.columns and "DGS2" in macro.columns:
            curve_raw = macro["DGS10"].ffill() - macro["DGS2"].ffill()
            curve_chg = curve_raw.diff()
            corr = np.corrcoef(curve_chg.dropna(), df["US_YIELD_CURVE_chg"].dropna()[:len(curve_chg.dropna())])[0, 1]
            check("US_YIELD_CURVE_chg derived correctly from DGS10-DGS2",
                  corr > 0.95,
                  f"Correlation with DGS10-DGS2 diff = {corr:.4f}")

except Exception as e:
    check("Data integrity checks", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 3. STATISTICAL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
section("3. Statistical Validation — Correlation Edges")

try:
    val = pd.read_csv(PATHS["corr_edges_validated"])

    # All edges meet threshold
    below_thresh = (val["weight"] < CORR_THRESHOLD_FDR).sum()
    check(f"All validated edges |r| >= {CORR_THRESHOLD_FDR}",
          below_thresh == 0,
          f"{below_thresh} edges below threshold")

    # p_adjusted <= 0.05
    if "p_adjusted_pearson" in val.columns:
        above_p = (val["p_adjusted_pearson"] > 0.05).sum()
        check("All validated edges p_adj <= 0.05",
              above_p == 0,
              f"{above_p} edges above p=0.05")

    # No self-loops
    self_loops = (val["source"] == val["target"]).sum()
    check("No self-loops in validated edges", self_loops == 0, f"{self_loops} self-loops")

    # Reasonable edge count (not zero, not suspiciously large)
    n_edges = len(val)
    check("Validated edge count reasonable (10–200)",
          10 <= n_edges <= 200,
          f"{n_edges} edges")

    # sign column consistent with correlation value
    if "sign" in val.columns and "correlation_pearson" in val.columns:
        sign_mismatch = ((val["sign"] == "positive") & (val["correlation_pearson"] < 0)).sum() + \
                        ((val["sign"] == "negative") & (val["correlation_pearson"] > 0)).sum()
        check("sign column consistent with correlation_pearson",
              sign_mismatch == 0,
              f"{sign_mismatch} mismatches")

    # Symmetric? (A-B and B-A should not both appear)
    pairs = set(frozenset([r.source, r.target]) for _, r in val.iterrows())
    check("No duplicate pairs (both A-B and B-A)",
          len(pairs) == len(val),
          f"Unique pairs: {len(pairs)}, total rows: {len(val)}")

except Exception as e:
    check("Validated correlation checks", False, str(e))

try:
    pc = pd.read_csv(PATHS["partial_corr_edges"])

    below = (pc["weight"].abs() < PARTIAL_CORR_THRESHOLD).sum()
    check(f"All partial corr edges |pc| >= {PARTIAL_CORR_THRESHOLD}",
          below == 0,
          f"{below} edges below threshold")

    survival = len(pc) / len(val) * 100 if len(val) > 0 else 0
    check("Partial edge survival rate <= raw edges",
          len(pc) <= len(val),
          f"{len(pc)}/{len(val)} = {survival:.1f}% survival")

    # Partial corr matrix is symmetric
    matrix = pd.read_csv(PATHS["partial_corr_matrix"], index_col=0)
    diff = (matrix.values - matrix.values.T)
    check("Partial correlation matrix is symmetric",
          np.max(np.abs(diff)) < 1e-6,
          f"Max asymmetry: {np.max(np.abs(diff)):.2e}")

    # Diagonal should be 1.0
    diag_max_dev = np.max(np.abs(np.diag(matrix.values) - 1.0))
    check("Partial corr matrix diagonal = 1.0",
          diag_max_dev < 1e-4,
          f"Max diagonal deviation: {diag_max_dev:.2e}")

except Exception as e:
    check("Partial correlation checks", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 4. LAGGED CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
section("4. Lagged Correlation")

try:
    lagged = pd.read_csv(PATHS["lagged_corr_edges"])

    if len(lagged) > 0:
        # lag_weeks must be 1-4
        valid_lags = lagged["lag_weeks"].between(1, 4).all()
        check("All lag_weeks in range 1–4",
              valid_lags,
              f"Unique lags: {sorted(lagged['lag_weeks'].unique())}")

        # targets should be bank columns only
        bank_targets = [c.replace("_ret", "") for c in
                        ["BBL_ret","KBANK_ret","KKP_ret","KTB_ret","SCB_ret","TISCO_ret","TTB_ret"]]
        non_bank_targets = [t for t in lagged["target_node"].unique() if t not in bank_targets]
        check("Lagged edge targets are Thai banks only",
              len(non_bank_targets) == 0,
              f"Non-bank targets: {non_bank_targets}" if non_bank_targets else "OK")
    else:
        check("Lagged edges file exists (may be empty — valid finding)",
              True,
              "No edges above threshold — valid finding at weekly frequency",
              warn_only=True)

except Exception as e:
    check("Lagged correlation checks", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 5. OLS FACTOR EXPOSURE
# ─────────────────────────────────────────────────────────────────────────────
section("5. OLS Factor Exposure")

try:
    factors = pd.read_csv(PATHS["factor_exposure_edges"])

    if len(factors) > 0:
        # p_value <= 0.05
        above_p = (factors["p_value"] > 0.05).sum()
        check("All OLS edges p <= 0.05",
              above_p == 0,
              f"{above_p} edges above p=0.05")

        # Targets should be banks
        bank_nodes = ["BBL","KBANK","KKP","KTB","SCB","TISCO","TTB"]
        non_bank = [t for t in factors["target_node"].unique() if t not in bank_nodes]
        check("OLS edge targets are Thai banks",
              len(non_bank) == 0,
              f"Non-bank targets: {non_bank}" if non_bank else "OK")

        # Beta values are finite
        all_finite = np.isfinite(factors["beta"]).all()
        check("All beta values finite",
              all_finite,
              f"Non-finite betas: {(~np.isfinite(factors['beta'])).sum()}")

        # n_obs should be >= 20
        if "n_obs" in factors.columns:
            min_obs = factors["n_obs"].min()
            check("All OLS regressions have n_obs >= 20",
                  min_obs >= 20,
                  f"Min n_obs: {min_obs}")
    else:
        check("OLS factor exposure edges present",
              False,
              "Empty — run Phase 3",
              warn_only=True)

except Exception as e:
    check("OLS factor exposure checks", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 6. REGIME DETECTION — NO LOOK-AHEAD BIAS
# ─────────────────────────────────────────────────────────────────────────────
section("6. Regime Detection — Look-Ahead Bias Check")

try:
    regime = pd.read_csv(PATHS["regime_labels"], parse_dates=["date"])

    # NaN in first 26 weeks (lookback window)
    first_26 = regime.head(26)
    has_unknowns = (first_26["regime"] == "Unknown").sum()
    check("First 26 weeks labeled 'Unknown' (lookback window)",
          has_unknowns >= 1,
          f"{has_unknowns} Unknown in first 26 weeks")

    # No look-ahead: rolling_change at time t should only use data up to t
    # Verify by checking fedfunds_roll_chg = FEDFUNDS[t] - FEDFUNDS[t-26]
    macro = pd.read_csv(PATHS["macro_weekly"], index_col=0, parse_dates=True)
    if "FEDFUNDS" in macro.columns:
        ff = macro["FEDFUNDS"].ffill()
        roll_expected = ff - ff.shift(26)
        roll_expected.name = "expected"
        regime_indexed = regime.set_index("date")
        aligned = roll_expected.reindex(regime_indexed.index)
        if "fedfunds_roll_chg" in regime_indexed.columns:
            diff = (aligned - regime_indexed["fedfunds_roll_chg"]).abs().dropna()
            check("fedfunds_roll_chg matches FEDFUNDS[t] - FEDFUNDS[t-26]",
                  diff.max() < 0.01,
                  f"Max deviation: {diff.max():.4f}")

    # Regime coverage
    hiking = (regime["regime"] == "Hiking_or_Restrictive").sum()
    cutting = (regime["regime"] == "Pausing_or_Cutting").sum()
    unknown = (regime["regime"] == "Unknown").sum()
    check("Regime labels cover full sample",
          hiking + cutting + unknown == len(regime),
          f"Hiking={hiking}, Cutting={cutting}, Unknown={unknown}, Total={len(regime)}")

except Exception as e:
    check("Regime detection checks", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 7. GRAPH CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────
section("7. Graph Consistency")

try:
    nodes = pd.read_csv(PATHS["neo4j_nodes"])
    edges = pd.read_csv(PATHS["neo4j_edges"])
    node_ids = set(nodes["node_id"].astype(str))

    # All edge endpoints exist in node list
    missing_src = set(edges["source"].astype(str)) - node_ids
    missing_tgt = set(edges["target"].astype(str)) - node_ids
    check("All edge sources exist in node list",
          len(missing_src) == 0,
          f"Missing: {missing_src}" if missing_src else f"{len(edges)} edges checked")
    check("All edge targets exist in node list",
          len(missing_tgt) == 0,
          f"Missing: {missing_tgt}" if missing_tgt else "OK")

    # All 7 Thai banks are nodes
    bank_nodes = {"BBL","KBANK","KKP","KTB","SCB","TISCO","TTB"}
    missing_banks = bank_nodes - node_ids
    check("All 7 Thai bank nodes present",
          len(missing_banks) == 0,
          f"Missing: {missing_banks}" if missing_banks else "BBL,KBANK,KKP,KTB,SCB,TISCO,TTB all present")

    # Relationship types are valid
    valid_rel_types = {"CORRELATED_WITH","PARTIAL_CORRELATED_WITH",
                       "LAGGED_CORRELATED_WITH","INFLUENCES"}
    actual_types = set(edges["rel_type"].unique())
    unexpected = actual_types - valid_rel_types
    check("All relationship types are valid",
          len(unexpected) == 0,
          f"Unexpected: {unexpected}" if unexpected else f"Types: {actual_types}")

    # Weight column is numeric and positive
    check("Edge weights are positive",
          (edges["weight"].fillna(0) >= 0).all(),
          f"Negative weights: {(edges['weight'] < 0).sum()}")

    # Node types cover expected categories
    expected_types = {"Bank", "ETF", "MacroFactor", "FX", "Index", "DerivedFactor"}
    actual_node_types = set(nodes["type"].unique())
    missing_types = expected_types - actual_node_types
    check("All expected node types present",
          len(missing_types) == 0,
          f"Missing types: {missing_types}" if missing_types else f"Types: {actual_node_types}")

except Exception as e:
    check("Graph consistency checks", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 8. GDS SIMULATION RESULTS
# ─────────────────────────────────────────────────────────────────────────────
section("8. GDS Simulation Results")

try:
    cent = pd.read_csv(PATHS["centrality_results"])

    required_metrics = ["degree_centrality", "betweenness_centrality", "pagerank"]
    for m in required_metrics:
        check(f"Centrality column '{m}' exists",
              m in cent.columns,
              f"Columns: {cent.columns.tolist()}" if m not in cent.columns else "OK")

    # All centrality values in [0, 1]
    for m in required_metrics:
        if m in cent.columns:
            in_range = cent[m].between(0, 1).all()
            check(f"{m} values in [0,1]",
                  in_range,
                  f"Range: [{cent[m].min():.4f}, {cent[m].max():.4f}]")

    # All nodes have centrality
    n_nodes_expected = len(pd.read_csv(PATHS["neo4j_nodes"]))
    check("Centrality computed for all nodes",
          len(cent) == n_nodes_expected,
          f"Centrality rows: {len(cent)}, nodes: {n_nodes_expected}")

    louvain = pd.read_csv(PATHS["community_louvain"])
    n_communities = louvain["communityId"].nunique()
    check("Louvain detected at least 2 communities",
          n_communities >= 2,
          f"{n_communities} communities")

    # Banks should be in same community (high intra-cluster correlation)
    bank_nodes = {"BBL","KBANK","KKP","KTB","SCB","TISCO","TTB"}
    bank_comm = louvain[louvain["node_id"].isin(bank_nodes)]["communityId"]
    bank_majority_comm = bank_comm.mode()[0] if len(bank_comm) > 0 else -1
    banks_together = (bank_comm == bank_majority_comm).sum()
    check("Majority of Thai banks in same Louvain community",
          banks_together >= 5,
          f"{banks_together}/7 banks in community {bank_majority_comm}")

    kmeans = pd.read_csv(PATHS["community_kmeans"])
    check("K-Means produced 4 clusters",
          kmeans["communityId"].nunique() == 4,
          f"Actual clusters: {kmeans['communityId'].nunique()}")

except Exception as e:
    check("GDS simulation checks", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 9. REPORT ACCURACY — Numbers match pipeline outputs
# ─────────────────────────────────────────────────────────────────────────────
section("9. Report Accuracy — Numbers vs Pipeline Outputs")

try:
    report_path = Path("outputs/midterm_report_draft.md")
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")

        # Check validated edge count
        val = pd.read_csv(PATHS["corr_edges_validated"])
        actual_val_count = str(len(val))
        check(f"Report validated edge count matches CSV ({actual_val_count})",
              actual_val_count in report_text,
              f"Expected '{actual_val_count}' in report")

        # Check partial edge count
        pc = pd.read_csv(PATHS["partial_corr_edges"])
        actual_pc_count = str(len(pc))
        check(f"Report partial edge count matches CSV ({actual_pc_count})",
              actual_pc_count in report_text,
              f"Expected '{actual_pc_count}' in report")

        # Check n_obs (week count)
        df = pd.read_csv(PATHS["final_weekly_dataset"], index_col=0, parse_dates=True)
        actual_weeks = str(len(df))
        check(f"Report observation count matches dataset ({actual_weeks} weeks)",
              actual_weeks in report_text,
              f"Expected '{actual_weeks}' in report")

        # Check top centrality node
        cent = pd.read_csv(PATHS["centrality_results"])
        top_node = cent.nlargest(1, "degree_centrality").iloc[0]["name"]
        check(f"Top centrality node '{top_node}' appears in report",
              top_node in report_text,
              f"Looking for '{top_node}'")

        # Check community count
        louvain = pd.read_csv(PATHS["community_louvain"])
        n_comm = str(louvain["communityId"].nunique())
        check(f"Report community count matches Louvain ({n_comm})",
              n_comm in report_text,
              f"Expected '{n_comm}' in report")

    else:
        check("Report file exists", False, "outputs/midterm_report_draft.md not found")

except Exception as e:
    check("Report accuracy checks", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY")

n_pass = sum(1 for r in results if r[0] == PASS)
n_warn = sum(1 for r in results if r[0] == WARN)
n_fail = sum(1 for r in results if r[0] == FAIL)
total  = len(results)

print(f"\n  Total checks : {total}")
print(f"  ✅ PASS      : {n_pass}")
print(f"  ⚠️  WARN      : {n_warn}")
print(f"  ❌ FAIL      : {n_fail}")

if n_fail == 0 and n_warn == 0:
    print("\n  🎉 All checks passed! Pipeline results are consistent.")
elif n_fail == 0:
    print(f"\n  ✅ No critical failures. {n_warn} warning(s) to review.")
else:
    print(f"\n  ❌ {n_fail} check(s) failed. Review items above.")
    print("\n  Failed checks:")
    for status, name, detail in results:
        if status == FAIL:
            print(f"    - {name}: {detail}")

print()
sys.exit(0 if n_fail == 0 else 1)
