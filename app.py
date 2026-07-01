"""
app.py — Streamlit application for Thai Bank Graph Analysis (9 pages).

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import PATHS
from src import visualization as viz

st.set_page_config(
    page_title="Thai Bank Graph Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_csv(path, **kwargs):
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, **kwargs)
        return df
    except pd.errors.EmptyDataError:
        # File exists but is empty or header-only — return empty DataFrame
        return pd.DataFrame()
    except Exception:
        return None


def file_warning(name: str):
    st.warning(f"⚠️ File not found: `{name}`. Run `python run_pipeline.py` first.")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

PAGES = [
    "1 · Project Overview",
    "2 · Data Quality & Cleansing",
    "3 · Validated Correlation Network",
    "4 · Partial Correlation Network",
    "5 · Regime-aware Network",
    "6 · Lagged Relationship Network",
    "7 · Factor Exposure Network",
    "8 · Neo4j GDS Results",
    "9 · Report Findings",
]

with st.sidebar:
    st.markdown("## 🏦 Thai Bank Graph")
    st.markdown("*Social Network & Media Analysis*")
    st.divider()
    page = st.radio("Navigate", PAGES, index=0)
    st.divider()
    st.caption("Period: May 2022 – Latest\nFrequency: Weekly")
    st.divider()
    if st.button("🔄 Clear Cache & Reload"):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Page 1: Project Overview
# ─────────────────────────────────────────────────────────────────────────────

if page == PAGES[0]:
    st.title("🏦 Interest Rate and Risk Sentiment Network")
    st.subheader("Graph Analysis of Thai SET50 Banking Stocks")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Research Question**
> How do global interest rates, risk sentiment, FX movement, and financial sector ETFs
> form a network of influence on Thai SET50 banking stocks?

**Data Period:** May 2022 – Latest Available
**Frequency:** Weekly

**Thai Banking Stocks**
BBL · KBANK · KKP · KTB · SCB · TISCO · TTB
        """)

    with col2:
        st.markdown("""
**Global Variables**
- XLF (US Financial ETF)
- EUFN (European Financial ETF)
- USDTHB (FX)
- SET Index

**FRED Macro**
- FEDFUNDS, DGS2, DGS10, MORTGAGE30US, VIX

**BOT Variables**
- BOT Policy Rate, BOT Rate Change

**Derived Factors**
- US Yield Curve (DGS10–DGS2)
- Thai Bank Basket (equal-weighted return)
        """)

    st.divider()
    st.markdown("""
**Methodology**
| Step | Method |
|---|---|
| 1 | Statistically Validated Correlation Network (Pearson + FDR) |
| 2 | Partial Correlation Network (GraphicalLassoCV) |
| 3 | Lagged Correlation Network (lag 1–4 weeks) |
| 4 | Factor Exposure Network (OLS regression) |
| 5 | Regime-Aware Network (Fed hiking vs. pausing) |
| 6 | Neo4j GDS: Degree, Betweenness, PageRank, Louvain, FastRP+KMeans |

**Important Limitations**
- Edges represent statistical relationships, not direct causal links.
- Correlation does not imply causation.
- Lagged correlation does not prove causality.
- This is an ex-post network analysis, not a forecasting model.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2: Data Quality & Cleansing
# ─────────────────────────────────────────────────────────────────────────────

elif page == PAGES[1]:
    st.title("2 · Data Quality & Cleansing")

    # Missing value summary
    st.subheader("Missing Value Summary")
    missing = load_csv(PATHS["missing_value_summary"])
    if missing is not None:
        styled = missing.style.background_gradient(subset=["missing_pct"], cmap="Reds")
        st.dataframe(styled, use_container_width=True)
    else:
        file_warning("missing_value_summary.csv")

    # Outlier flags
    st.subheader("Outlier Flags (z-score > 3.5)")
    outliers = load_csv(PATHS["outlier_flags"])
    if outliers is not None and not outliers.empty:
        st.metric("Total Flagged Cells", len(outliers))
        st.dataframe(outliers.head(50), use_container_width=True)
    elif outliers is not None:
        st.success("No outliers flagged above threshold.")
    else:
        file_warning("outlier_flags.csv")

    # Final dataset preview
    st.subheader("Final Weekly Dataset Preview")
    final = load_csv(PATHS["final_weekly_dataset"], index_col=0, parse_dates=True)
    if final is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows (weeks)", final.shape[0])
        c2.metric("Columns (features)", final.shape[1])
        c3.metric("Date Range", f"{final.index[0].date()} → {final.index[-1].date()}")

        st.dataframe(final.tail(10).style.format("{:.4f}"), use_container_width=True)

        st.subheader("Weekly Return Time Series")
        bank_cols = [c for c in final.columns if c.endswith("_ret") and "BASKET" not in c and "ETF" not in c
                     and c.replace("_ret","") in ["BBL","KBANK","KKP","KTB","SCB","TISCO","TTB"]]
        bank_cols = [c for c in final.columns if c.endswith("_ret")
                     and c.replace("_ret","") in ["BBL","KBANK","KKP","KTB","SCB","TISCO","TTB"]]
        if bank_cols:
            st.plotly_chart(viz.plot_time_series(final, bank_cols, "Thai Bank Weekly Log Returns"),
                           use_container_width=True)
    else:
        file_warning("final_weekly_dataset.csv")

    st.subheader("Transformation Rules Applied")
    st.markdown("""
    | Variable Type | Transformation |
    |---|---|
    | Stock / ETF / FX / Index | `log(price_t / price_{t-1})` |
    | Yield / Interest Rate | `rate_t − rate_{t-1}` |
    | VIX | `log(VIX_t / VIX_{t-1})` (VIX_CHANGE) |
    | BOT Rate | `rate_t − rate_{t-1}` (BOT_RATE_CHANGE) |
    | US Yield Curve | `(DGS10 − DGS2)_t − (DGS10 − DGS2)_{t-1}` |
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Page 3: Validated Correlation Network
# ─────────────────────────────────────────────────────────────────────────────

elif page == PAGES[2]:
    st.title("3 · Validated Correlation Network")
    st.caption("Pearson + FDR (Benjamini-Hochberg) correction  |  threshold: |r|≥0.20, p_adj<0.05")

    final = load_csv(PATHS["final_weekly_dataset"], index_col=0, parse_dates=True)
    validated = load_csv(PATHS["corr_edges_validated"])

    if final is not None:
        st.subheader("Pearson Correlation Matrix")
        st.plotly_chart(viz.plot_corr_heatmap(final), use_container_width=True)

    if validated is not None:
        st.subheader("Validated Edges")
        st.metric("Validated Correlation Edges", len(validated))
        st.dataframe(
            validated[["source","target","correlation_pearson","correlation_spearman",
                        "p_adjusted_pearson","sign","n_obs"]]
            .sort_values("correlation_pearson", key=abs, ascending=False)
            .head(30).style.format({"correlation_pearson":"{:.4f}",
                                    "correlation_spearman":"{:.4f}",
                                    "p_adjusted_pearson":"{:.4f}"}),
            use_container_width=True,
        )

        # Network
        st.subheader("Correlation Network Graph")
        nodes = load_csv(PATHS["neo4j_nodes"])
        if nodes is not None:
            val_for_viz = validated.copy()
            val_for_viz["rel_type"] = "CORRELATED_WITH"
            html = viz.build_pyvis_html(nodes, val_for_viz, title="Validated Correlation Network")
            components.html(html, height=620, scrolling=False)
        else:
            file_warning("neo4j_nodes.csv")

        st.subheader("Top 10 Correlated Pairs")
        st.dataframe(validated.nlargest(10, "weight")[["source","target","correlation_pearson","sign"]],
                     use_container_width=True)
    else:
        file_warning("correlation_edges_validated.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Page 4: Partial Correlation Network
# ─────────────────────────────────────────────────────────────────────────────

elif page == PAGES[3]:
    st.title("4 · Partial Correlation Network")
    st.caption("GraphicalLassoCV (fallback: LedoitWolf)  |  threshold: |pc|≥0.15")

    pcorr_matrix = load_csv(PATHS["partial_corr_matrix"], index_col=0)
    pcorr_edges  = load_csv(PATHS["partial_corr_edges"])
    raw_vs       = load_csv(PATHS["raw_vs_partial"])
    validated    = load_csv(PATHS["corr_edges_validated"])
    nodes        = load_csv(PATHS["neo4j_nodes"])

    if pcorr_matrix is not None:
        st.subheader("Partial Correlation Matrix")
        st.plotly_chart(viz.plot_partial_heatmap(pcorr_matrix), use_container_width=True)

    if pcorr_edges is not None and validated is not None:
        st.subheader("Raw vs Partial Edge Comparison")
        col1, col2, col3 = st.columns(3)
        col1.metric("Raw Validated Edges", len(validated))
        col2.metric("Partial Correlation Edges", len(pcorr_edges))
        survival = len(pcorr_edges) / len(validated) * 100 if len(validated) > 0 else 0
        col3.metric("Edge Survival Rate", f"{survival:.1f}%")
        st.plotly_chart(viz.plot_edge_comparison(len(validated), len(pcorr_edges)),
                       use_container_width=True)

        st.subheader("Partial Correlation Network Graph")
        if nodes is not None:
            pc_for_viz = pcorr_edges.copy()
            pc_for_viz["rel_type"] = "PARTIAL_CORRELATED_WITH"
            html = viz.build_pyvis_html(nodes, pc_for_viz, title="Partial Correlation Network")
            components.html(html, height=620, scrolling=False)

        st.subheader("Partial Correlation Edges")
        st.dataframe(pcorr_edges.sort_values("weight", ascending=False).head(30),
                     use_container_width=True)
    else:
        file_warning("partial_correlation_edges.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Page 5: Regime-aware Network
# ─────────────────────────────────────────────────────────────────────────────

elif page == PAGES[4]:
    st.title("5 · Regime-Aware Network")
    st.caption("Fed regime: rolling 26-week change in FEDFUNDS and DGS2. No look-ahead bias.")

    regime_labels  = load_csv(PATHS["regime_labels"], parse_dates=["date"])
    regime_compare = load_csv(PATHS["regime_comparison"])

    if regime_labels is not None:
        st.subheader("Regime Labels")
        counts = regime_labels["regime"].value_counts().reset_index()
        counts.columns = ["Regime", "Weeks"]
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(counts, use_container_width=True)
        with col2:
            fig = viz.plot_time_series(
                regime_labels.set_index("date")[["fedfunds_roll_chg", "dgs2_roll_chg"]],
                ["fedfunds_roll_chg", "dgs2_roll_chg"],
                "26-Week Rolling Change: FEDFUNDS & DGS2"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Regime Timeline")
        regime_labels["regime_num"] = regime_labels["regime"].map(
            {"Hiking_or_Restrictive": 1, "Pausing_or_Cutting": 0, "Unknown": -1}
        )
        fig2 = viz.plot_time_series(
            regime_labels.set_index("date")[["regime_num"]],
            ["regime_num"],
            "Regime Over Time (1=Hiking, 0=Cutting, -1=Unknown)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    if regime_compare is not None and not regime_compare.empty:
        st.subheader("Regime Comparison Summary")
        st.dataframe(regime_compare, use_container_width=True)
        st.plotly_chart(viz.plot_regime_comparison(regime_compare), use_container_width=True)
    else:
        file_warning("regime_comparison_summary.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Page 6: Lagged Relationship Network
# ─────────────────────────────────────────────────────────────────────────────

elif page == PAGES[5]:
    st.title("6 · Lagged Relationship Network")
    st.caption("corr(factor_{t-k}, bank_return_t) for k=1,2,3,4  |  graph threshold: |r|≥0.15")
    st.warning("Lagged correlation does NOT prove causality. It indicates that past movement "
               "in one variable is associated with later movement in another.")

    lagged      = load_csv(PATHS["lagged_corr_edges"])
    # lagged_corr_all_pairs may not be in PATHS if running old config .pyc — derive directly
    _all_pairs_path = PATHS["lagged_corr_edges"].parent / "lagged_correlation_all_pairs.csv"
    lagged_all  = load_csv(_all_pairs_path)
    nodes       = load_csv(PATHS["neo4j_nodes"])

    # Use all-pairs for the heatmap/table; fall back to filtered if all-pairs unavailable
    display_df = lagged_all if (lagged_all is not None and not lagged_all.empty) else lagged

    if display_df is None:
        file_warning("lagged_correlation_all_pairs.csv")
        st.info("Run `python run_pipeline.py --phase 3` to generate this file.")
    else:
        st.subheader("Best-Lag Correlation Heatmap (all factor-bank pairs)")
        st.caption("Shows the lag (k=1–4) with the highest |correlation| for each factor→bank pair. "
                   "Graph edges are drawn only for |r| ≥ 0.15.")
        if not display_df.empty:
            st.plotly_chart(viz.plot_lagged_heatmap(display_df), use_container_width=True)
        else:
            st.info("No lagged correlation data computed yet. Run Phase 3.")

        st.subheader("All Factor–Bank Best-Lag Correlations")
        if display_df is not None and not display_df.empty:
            show_cols = [c for c in ["source","target","correlation","lag_weeks","sign","weight"]
                         if c in display_df.columns]
            st.dataframe(
                display_df[show_cols].sort_values("weight", ascending=False)
                .style.format({"correlation": "{:.4f}", "weight": "{:.4f}"}),
                use_container_width=True,
            )

        # Graph section — only if above-threshold edges exist
        if lagged is not None and not lagged.empty:
            st.subheader(f"Directed Lagged Network (|r| ≥ 0.15 edges only, {len(lagged)} edges)")
            if nodes is not None and not nodes.empty:
                lagged_renamed = lagged.rename(columns={"source_node": "source", "target_node": "target"})
                lagged_renamed["rel_type"] = "LAGGED_CORRELATED_WITH"
                html = viz.build_pyvis_html(nodes, lagged_renamed, title="Lagged Correlation Network (Directed)")
                components.html(html, height=620, scrolling=False)
        else:
            st.info("No edges above the 0.15 threshold — the heatmap above shows the raw magnitudes. "
                    "This is a valid finding: global macro factors do not significantly lead "
                    "Thai banking stock returns at weekly lags 1–4 over this sample period.")


# ─────────────────────────────────────────────────────────────────────────────
# Page 7: Factor Exposure Network
# ─────────────────────────────────────────────────────────────────────────────

elif page == PAGES[6]:
    st.title("7 · Factor Exposure Network")
    st.caption("OLS: Bank_Return ~ Global Macro & Financial Factors  |  p≤0.05 edges shown")
    st.warning("OLS factor exposure means statistical association, not guaranteed causal impact.")

    factors = load_csv(PATHS["factor_exposure_edges"])
    nodes   = load_csv(PATHS["neo4j_nodes"])

    if factors is not None and not factors.empty:
        col1, col2 = st.columns(2)
        col1.metric("Significant Factor→Bank Edges", len(factors))
        col2.metric("Unique Banks with Significant Factors",
                    factors["target"].nunique())

        st.subheader("Beta Heatmap")
        st.plotly_chart(viz.plot_factor_heatmap(factors), use_container_width=True)

        st.subheader("OLS Results Table")
        display_cols = ["source","target","beta","t_stat","p_value","sign","r_squared","n_obs"]
        available_cols = [c for c in display_cols if c in factors.columns]
        st.dataframe(
            factors[available_cols].sort_values("p_value").style.format(
                {"beta":"{:.4f}", "t_stat":"{:.3f}", "p_value":"{:.4f}", "r_squared":"{:.3f}"}
            ),
            use_container_width=True,
        )

        st.subheader("Factor Exposure Network Graph")
        if nodes is not None:
            factors_renamed = factors.copy()
            factors_renamed["rel_type"] = "INFLUENCES"
            if "source_node" in factors_renamed.columns:
                factors_renamed["source"] = factors_renamed["source_node"]
                factors_renamed["target"] = factors_renamed["target_node"]
            html = viz.build_pyvis_html(nodes, factors_renamed, title="Factor Exposure Network")
            components.html(html, height=620, scrolling=False)
    else:
        file_warning("factor_exposure_edges.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Page 8: Neo4j GDS Results
# ─────────────────────────────────────────────────────────────────────────────

elif page == PAGES[7]:
    st.title("8 · Neo4j GDS Results")

    centrality = load_csv(PATHS["centrality_results"])
    louvain    = load_csv(PATHS["community_louvain"])
    kmeans     = load_csv(PATHS["community_kmeans"])

    if centrality is not None and not centrality.empty:
        st.subheader("Degree Centrality")
        st.plotly_chart(viz.plot_centrality_bar(centrality, "degree_centrality",
                                                title="Degree Centrality — Top Nodes"),
                       use_container_width=True)

        st.subheader("Betweenness Centrality")
        st.plotly_chart(viz.plot_centrality_bar(centrality, "betweenness_centrality",
                                                title="Betweenness Centrality — Bridge Nodes"),
                       use_container_width=True)

        st.subheader("PageRank")
        st.plotly_chart(viz.plot_centrality_bar(centrality, "pagerank",
                                                title="PageRank"),
                       use_container_width=True)

        st.subheader("Full Centrality Table")
        st.dataframe(centrality.sort_values("degree_centrality", ascending=False).reset_index(drop=True),
                     use_container_width=True)
    else:
        file_warning("centrality_results.csv")
        st.info("Run `python src/neo4j_loader.py` then `python src/gds_algorithms.py` to generate results.")

    st.divider()

    if louvain is not None and not louvain.empty:
        st.subheader("Louvain Community Detection")
        st.plotly_chart(viz.plot_community_table(louvain, "communityId"), use_container_width=True)
    else:
        file_warning("community_results_louvain.csv")

    if kmeans is not None and not kmeans.empty:
        st.subheader("K-Means Clustering (FastRP Embeddings, k=4)")
        st.plotly_chart(viz.plot_community_table(kmeans, "communityId"), use_container_width=True)
        if "silhouette" in kmeans.columns:
            avg_sil = kmeans["silhouette"].mean()
            st.metric("Average Silhouette Score", f"{avg_sil:.4f}")
    else:
        file_warning("community_results_kmeans.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Page 9: Report Findings
# ─────────────────────────────────────────────────────────────────────────────

elif page == PAGES[8]:
    st.title("9 · Report Findings")

    # Load available data for dynamic summaries
    centrality = load_csv(PATHS["centrality_results"])
    louvain    = load_csv(PATHS["community_louvain"])
    validated  = load_csv(PATHS["corr_edges_validated"])
    partial    = load_csv(PATHS["partial_corr_edges"])
    regime_cmp = load_csv(PATHS["regime_comparison"])
    factors    = load_csv(PATHS["factor_exposure_edges"])

    st.markdown("## Key Findings")
    st.markdown("*Evidence-based interpretation from graph analysis. "
                "Statistical relationships — not causal claims.*")

    # Finding 1
    with st.expander("Finding 1 · Most Central Node", expanded=True):
        if centrality is not None and not centrality.empty and "degree_centrality" in centrality.columns:
            top = centrality.nlargest(1, "degree_centrality").iloc[0]
            st.markdown(f"""
**Evidence:** Degree Centrality = `{top.get('degree_centrality', 'N/A'):.4f}`
**Node:** `{top.get('name', 'N/A')}` (Type: `{top.get('type', 'N/A')}`)

**Interpretation:** This node is connected to the most other variables in the network,
suggesting it is most broadly associated with price movements across the financial system.
            """)
        else:
            st.info("Run GDS algorithms to populate centrality results.")

    # Finding 2
    with st.expander("Finding 2 · Bridge Variable (Betweenness)"):
        if centrality is not None and "betweenness_centrality" in centrality.columns:
            top_b = centrality.nlargest(1, "betweenness_centrality").iloc[0]
            st.markdown(f"""
**Evidence:** Betweenness Centrality = `{top_b.get('betweenness_centrality','N/A'):.4f}`
**Node:** `{top_b.get('name','N/A')}`

**Interpretation:** This variable acts as a bridge connecting different clusters in the network,
potentially serving as a transmission channel for global macro shocks to Thai banking stocks.
            """)

    # Finding 3
    with st.expander("Finding 3 · Most Globally Sensitive Thai Bank"):
        if factors is not None and not factors.empty:
            bank_factor_count = factors.groupby("target").size().sort_values(ascending=False)
            most_sensitive = bank_factor_count.idxmax()
            st.markdown(f"""
**Evidence:** `{most_sensitive}` has {bank_factor_count.max()} significant factor exposures (OLS p≤0.05)

**Interpretation:** This bank's weekly returns show the strongest statistical association with
global macro and financial conditions among all Thai SET50 banking stocks analyzed.
            """)

    # Finding 4
    with st.expander("Finding 4 · Community Structure"):
        if louvain is not None and not louvain.empty:
            n_communities = louvain["communityId"].nunique()
            st.markdown(f"""
**Evidence:** Louvain detected `{n_communities}` communities.

**Interpretation:** Thai banking stocks and global macro variables form distinct clusters.
Variables within the same community tend to move together more strongly than those across communities.
            """)

    # Finding 5
    with st.expander("Finding 5 · Raw vs Partial Correlation"):
        if validated is not None and partial is not None:
            raw_n = len(validated)
            par_n = len(partial)
            survival = par_n / raw_n * 100 if raw_n > 0 else 0
            st.markdown(f"""
**Evidence:**
- Raw validated edges: `{raw_n}`
- Partial correlation edges: `{par_n}`
- Edge survival rate: `{survival:.1f}%`

**Interpretation:** After conditioning on all other variables, only {survival:.0f}% of raw correlations
remain. The edges that disappear were likely driven by common factors. The edges that survive
represent more direct associations between pairs of variables.
            """)

    # Finding 6
    with st.expander("Finding 6 · Regime Sensitivity"):
        if regime_cmp is not None and not regime_cmp.empty:
            st.dataframe(regime_cmp, use_container_width=True)
            st.markdown("""
**Interpretation:** The network structure changes between Fed hiking and cutting regimes.
Edge density and community membership may differ, suggesting that global rate tightening
is associated with shifts in how Thai banking stocks co-move with global factors.
            """)

    # Finding 7
    with st.expander("Finding 7 · Risk Sentiment Transmission"):
        st.markdown("""
**Evidence:** VIX, USDTHB, XLF, EUFN edges in the validated and lagged correlation networks.

**Interpretation:** Global risk sentiment (VIX), FX movement (USDTHB), and international
financial sector performance (XLF, EUFN) appear linked to Thai banking stock returns.
VIX and USDTHB may act as channels through which global risk sentiment reaches Thai markets.
        """)

    # Finding 8
    with st.expander("Finding 8 · Methodological Caution"):
        st.markdown("""
**Important caveats:**
1. This is not a traditional social network — it is a financial behavior network derived from time series.
2. Edges represent statistical relationships, not direct causal links.
3. Correlation ≠ Causation. Lagged correlation does not prove Granger causality.
4. OLS factor exposure indicates association, not guaranteed causal impact.
5. Weekly data reduces noise but may mask intra-week dynamics.
6. Sample starts May 2022 to reduce SCB/SCBX structural break issues.
7. Regime labels use only past 26-week information — no look-ahead bias.
        """)

    st.divider()
    st.markdown("## 2-Page Report Draft")
    report_path = Path("outputs/midterm_report_draft.md")
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("Run `python src/report_generator.py` to generate the report draft.")
