"""
visualization.py — Plotly and Pyvis chart builders for the Streamlit app.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# Time series
# ─────────────────────────────────────────────────────────────────────────────

def plot_time_series(df: pd.DataFrame, cols: list[str], title: str = "Weekly Returns") -> go.Figure:
    available = [c for c in cols if c in df.columns]
    fig = go.Figure()
    for col in available:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode="lines"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Value",
                      legend=dict(orientation="h", y=-0.2), height=450)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_corr_heatmap(df: pd.DataFrame,
                      exclude: set = None,
                      title: str = "Pearson Correlation Matrix") -> go.Figure:
    exclude = exclude or {"VIX_LEVEL", "BOT_RATE_HIKE", "BOT_RATE_CUT"}
    cols = [c for c in df.columns if c not in exclude]
    data = df[cols].dropna(how="all")
    corr = data.corr(method="pearson")

    # Shorten labels for display
    labels = [c.replace("_ret", "").replace("_chg", "").replace("_CHANGE", "") for c in corr.columns]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
    ))
    fig.update_layout(title=title, height=600, width=700)
    return fig


def plot_partial_heatmap(pcorr_df: pd.DataFrame, title: str = "Partial Correlation Matrix") -> go.Figure:
    labels = [c.replace("_ret", "").replace("_chg", "").replace("_CHANGE", "") for c in pcorr_df.columns]
    fig = go.Figure(go.Heatmap(
        z=pcorr_df.values,
        x=labels, y=labels,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(pcorr_df.values, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
    ))
    fig.update_layout(title=title, height=600, width=700)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Edge count comparison bar
# ─────────────────────────────────────────────────────────────────────────────

def plot_edge_comparison(raw_count: int, partial_count: int) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["Raw Correlation\n(validated)", "Partial Correlation"],
        y=[raw_count, partial_count],
        marker_color=["steelblue", "darkorange"],
        text=[str(raw_count), str(partial_count)],
        textposition="outside",
    ))
    survival = partial_count / raw_count * 100 if raw_count > 0 else 0
    fig.update_layout(
        title=f"Raw vs Partial Correlation Edge Count<br>(Edge Survival Rate: {survival:.1f}%)",
        yaxis_title="Number of Edges",
        height=400,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Network graph (Pyvis → HTML string)
# ─────────────────────────────────────────────────────────────────────────────

TYPE_COLORS = {
    "Bank":          "#2196F3",
    "ETF":           "#4CAF50",
    "MacroFactor":   "#F44336",
    "FX":            "#FF9800",
    "Index":         "#9C27B0",
    "DerivedFactor": "#009688",
    "Unknown":       "#9E9E9E",
}



# Column name → node_id mapping (dataset column names → Neo4j node IDs)
_COL_TO_NODE: dict[str, str] = {
    "BBL_ret": "BBL", "KBANK_ret": "KBANK", "KKP_ret": "KKP",
    "KTB_ret": "KTB", "SCB_ret": "SCB", "TISCO_ret": "TISCO", "TTB_ret": "TTB",
    "XLF_ret": "XLF", "EUFN_ret": "EUFN",
    "USDTHB_ret": "USDTHB", "SET_ret": "SET",
    "FEDFUNDS_chg": "FEDFUNDS", "DGS2_chg": "DGS2", "DGS10_chg": "DGS10",
    "MORTGAGE30US_chg": "MORTGAGE30US",
    "VIX_CHANGE": "VIX",
    "US_YIELD_CURVE_chg": "US_YIELD_CURVE",
    "BOT_RATE_CHANGE": "BOT_RATE",
    "THAI_BANK_BASKET_ret": "THAI_BANK_BASKET",
}


def _resolve_node_id(raw: str) -> str:
    """Translate a dataset column name to its node_id, or return raw if already a node_id."""
    return _COL_TO_NODE.get(str(raw), str(raw))


def build_pyvis_html(nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                     height: str = "600px", width: str = "100%",
                     title: str = "") -> str:
    """Return Pyvis network as an HTML string."""
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p>pyvis not installed. Run: pip install pyvis</p>"

    net = Network(height=height, width=width, bgcolor="#1a1a2e", font_color="white",
                  directed=False, notebook=False)
    net.set_options("""
    {
      "physics": { "forceAtlas2Based": { "gravitationalConstant": -40 },
                   "minVelocity": 0.75, "solver": "forceAtlas2Based" },
      "edges": { "smooth": false }
    }
    """)

    # Build set of valid node IDs for edge validation
    valid_nodes: set[str] = set()
    for _, row in nodes_df.iterrows():
        color = TYPE_COLORS.get(row.get("type", "Unknown"), "#9E9E9E")
        label = row.get("name", row.get("node_id", "?"))
        nid   = str(row.get("node_id", label))
        net.add_node(nid, label=label, color=color,
                     title=f"{label} [{row.get('type', '')}]",
                     size=20, font={"size": 12})
        valid_nodes.add(nid)

    skipped = 0
    for _, row in edges_df.iterrows():
        # Prefer source_node/target_node columns (lagged/factor edges); fall back to source/target
        raw_src = row.get("source_node") or row.get("source", "")
        raw_tgt = row.get("target_node") or row.get("target", "")
        src = _resolve_node_id(str(raw_src))
        tgt = _resolve_node_id(str(raw_tgt))

        if src not in valid_nodes or tgt not in valid_nodes:
            skipped += 1
            continue

        w      = float(row.get("weight", 0.1) or 0.1)
        color  = "#4CAF50" if row.get("sign", "") == "positive" else "#F44336"
        rel    = str(row.get("rel_type", ""))
        net.add_edge(src, tgt, value=max(w * 5, 0.5), color=color,
                     title=f"{rel} w={w:.3f}")

    if skipped:
        import logging
        logging.getLogger(__name__).warning("build_pyvis_html: skipped %d edges (node not in graph)", skipped)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        net.save_graph(f.name)
        return open(f.name, encoding="utf-8").read()


# ─────────────────────────────────────────────────────────────────────────────
# Centrality bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_centrality_bar(df: pd.DataFrame, metric: str = "degree_centrality",
                        top_n: int = 15, title: str = "") -> go.Figure:
    if metric not in df.columns:
        return go.Figure().update_layout(title=f"{metric} not available")
    sub = df[["name", "type", metric]].dropna().nlargest(top_n, metric)
    colors = [TYPE_COLORS.get(t, "#9E9E9E") for t in sub["type"]]
    fig = go.Figure(go.Bar(
        x=sub["name"], y=sub[metric],
        marker_color=colors,
        text=sub[metric].round(4),
        textposition="outside",
    ))
    fig.update_layout(title=title or metric.replace("_", " ").title(),
                      xaxis_title="Node", yaxis_title=metric, height=420)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Community visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_community_table(df: pd.DataFrame, community_col: str = "communityId") -> go.Figure:
    if community_col not in df.columns:
        return go.Figure().update_layout(title=f"{community_col} not available")

    palette = px.colors.qualitative.Set2
    df = df.sort_values([community_col, "name"]).reset_index(drop=True)
    df["color_idx"] = df[community_col].astype("category").cat.codes % len(palette)

    fig = go.Figure(go.Table(
        header=dict(values=["Name", "Type", "Community"],
                    fill_color="#1f2937", font=dict(color="white", size=12)),
        cells=dict(
            values=[df["name"], df.get("type", ""), df[community_col]],
            fill_color=[[palette[i % len(palette)] for i in df["color_idx"]]] * 3,
            font=dict(color="black", size=11),
        ),
    ))
    fig.update_layout(title=f"Community Detection — {community_col}", height=500)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Factor exposure heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_factor_heatmap(factor_edges: pd.DataFrame, title: str = "OLS Factor Exposure (Beta)") -> go.Figure:
    if factor_edges.empty:
        return go.Figure().update_layout(title="No factor exposure data")

    pivot = factor_edges.pivot_table(index="source", columns="target", values="beta", aggfunc="mean")
    labels_x = [c.replace("_ret", "") for c in pivot.columns]
    labels_y = [c.replace("_ret","").replace("_chg","").replace("_CHANGE","") for c in pivot.index]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=labels_x, y=labels_y,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(pivot.values, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig.update_layout(title=title, height=450)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Lagged correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_lagged_heatmap(lagged_edges: pd.DataFrame) -> go.Figure:
    if lagged_edges.empty:
        return go.Figure().update_layout(title="No lagged correlation data")

    df = lagged_edges.copy()
    # Prefer clean node-ID columns for readability; fall back to raw column names
    src_col = "source_node" if "source_node" in df.columns else "source"
    tgt_col = "target_node" if "target_node" in df.columns else "target"

    pivot = df.pivot_table(index=src_col, columns=tgt_col, values="correlation", aggfunc="mean")
    labels_x = [str(c) for c in pivot.columns]
    labels_y = [str(r) for r in pivot.index]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=labels_x, y=labels_y,
        colorscale="RdBu_r", zmid=0,
        text=np.round(pivot.values, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig.update_layout(title="Lagged Correlation — Best Lag k=1–4 (Factors → Banks)", height=500)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Regime comparison bar
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_comparison(regime_df: pd.DataFrame) -> go.Figure:
    if regime_df.empty:
        return go.Figure().update_layout(title="No regime data")
    fig = go.Figure()
    for col in ["validated_edges", "partial_edges"]:
        if col in regime_df.columns:
            fig.add_trace(go.Bar(name=col, x=regime_df["regime"], y=regime_df[col]))
    fig.update_layout(barmode="group", title="Edge Count by Regime", height=400)
    return fig
