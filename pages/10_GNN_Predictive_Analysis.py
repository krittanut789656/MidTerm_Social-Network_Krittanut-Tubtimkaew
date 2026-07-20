"""
Page 10 · GNN Predictive Analysis & Explainability

แสดงผลลัพธ์จาก Financial GNN:
    1. Model Performance (Accuracy, AUC-ROC)
    2. Network Visualization พร้อม Captum Edge Attribution
    3. Prediction สำหรับสัปดาห์ล่าสุด
    4. Training History
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GNN Predictive Analysis",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS  = BASE_DIR / "data" / "results"

MODEL_PATH     = RESULTS / "gnn_model.pt"
METRICS_PATH   = RESULTS / "gnn_metrics.json"
ATTR_PATH      = RESULTS / "gnn_attributions.csv"
ALL_ATTR_PATH  = RESULTS / "gnn_all_attributions.csv"
PRED_PATH      = RESULTS / "gnn_predictions.csv"
LABEL_INFO     = RESULTS / "gnn_label_info.csv"

# ── Node positions สำหรับ network visualization ───────────────────────────────
NODE_POS = {
    "BBL":              (-2.0,  1.5),
    "KBANK":            (-1.0,  2.0),
    "KKP":              ( 0.0,  2.5),
    "KTB":              ( 1.0,  2.0),
    "SCB":              ( 2.0,  1.5),
    "TISCO":            ( 1.5,  0.5),
    "TTB":              (-1.5,  0.5),
    "THAI_BANK_BASKET": ( 0.0,  1.0),
    "SET":              ( 0.0,  0.0),
    "USDTHB":           (-3.0,  0.0),
    "VIX":              (-3.0,  1.5),
    "XLF":              (-2.5, -1.0),
    "EUFN":             (-1.5, -1.5),
    "DGS2":             ( 2.0, -1.0),
    "DGS10":            ( 3.0,  0.0),
    "US_YIELD_CURVE":   ( 3.0,  1.0),
    "FEDFUNDS":         ( 2.5, -1.5),
    "MORTGAGE30US":     ( 1.5, -2.0),
}

NODE_TYPE_COLOR = {
    "Bank":          "#4A90D9",
    "ETF":           "#F5A623",
    "MacroFactor":   "#7ED321",
    "Index":         "#D0021B",
    "DerivedFactor": "#9B59B6",
    "FX":            "#E67E22",
}

NODE_TYPES = {
    "BBL": "Bank", "KBANK": "Bank", "KKP": "Bank", "KTB": "Bank",
    "SCB": "Bank", "TISCO": "Bank", "TTB": "Bank",
    "XLF": "ETF", "EUFN": "ETF",
    "USDTHB": "FX",
    "SET": "Index",
    "FEDFUNDS": "MacroFactor", "DGS2": "MacroFactor", "DGS10": "MacroFactor",
    "MORTGAGE30US": "MacroFactor", "VIX": "MacroFactor",
    "US_YIELD_CURVE": "DerivedFactor", "THAI_BANK_BASKET": "DerivedFactor",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Load Results
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_attributions():
    if not ATTR_PATH.exists():
        return None
    return pd.read_csv(ATTR_PATH)


@st.cache_data
def load_all_attributions():
    if not ALL_ATTR_PATH.exists():
        return None
    return pd.read_csv(ALL_ATTR_PATH)


@st.cache_data
def load_predictions():
    if not PRED_PATH.exists():
        return None
    return pd.read_csv(PRED_PATH)


@st.cache_data
def load_label_info():
    if not LABEL_INFO.exists():
        return None
    return pd.read_csv(LABEL_INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Network Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_attribution_network(attr_df: pd.DataFrame, title: str = "Edge Attribution Network",
                              top_n: int = 20) -> go.Figure:
    """
    วาด network graph โดย:
    - ความหนาของ edge ∝ attribution score (edge ที่สำคัญจะหนากว่า)
    - สีของ edge: แดง = attribution สูง (important), เทา = attribution ต่ำ
    - ขนาด node ∝ degree centrality
    """
    fig = go.Figure()

    # ── Normalize attribution scores ─────────────────────────────────────────
    max_attr = attr_df["attribution_score"].max()
    if max_attr == 0:
        max_attr = 1.0

    # ── วาด Edges ─────────────────────────────────────────────────────────────
    top_attr = attr_df.head(top_n)
    rest_attr = attr_df.iloc[top_n:]

    def add_edge(row, is_top=True):
        src = row["src_node"]
        tgt = row["tgt_node"]
        if src not in NODE_POS or tgt not in NODE_POS:
            return

        x0, y0 = NODE_POS[src]
        x1, y1 = NODE_POS[tgt]
        score   = row["attribution_score"]
        norm    = score / max_attr

        width = 1.0 + norm * 8.0  # edge width: 1 ถึง 9
        if is_top:
            color = f"rgba({int(255*norm)}, {int(50*(1-norm))}, 50, {0.4 + 0.6*norm})"
        else:
            color = "rgba(180, 180, 180, 0.2)"

        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="text",
            text=f"{src} ↔ {tgt}<br>Attribution: {score:.4f}",
            showlegend=False,
        ))

    for _, row in rest_attr.iterrows():
        add_edge(row, is_top=False)
    for _, row in top_attr.iterrows():
        add_edge(row, is_top=True)

    # ── วาด Nodes ─────────────────────────────────────────────────────────────
    for node, (x, y) in NODE_POS.items():
        node_type = NODE_TYPES.get(node, "MacroFactor")
        color     = NODE_TYPE_COLOR.get(node_type, "#888888")
        size      = 25 if node_type == "Bank" else 20
        if node in ["USDTHB", "THAI_BANK_BASKET", "SET"]:
            size = 30

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=size, color=color,
                        line=dict(color="white", width=2)),
            text=[node],
            textposition="top center",
            textfont=dict(size=9),
            hoverinfo="text",
            hovertext=f"{node}<br>Type: {node_type}",
            showlegend=False,
        ))

    # ── Legend สำหรับ Node Types ─────────────────────────────────────────────
    for ntype, color in NODE_TYPE_COLOR.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=color),
            name=ntype,
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        showlegend=True,
        legend=dict(title="Node Type", x=1.01, y=0.5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=550,
        margin=dict(l=20, r=120, t=60, b=20),
        plot_bgcolor="rgba(15,17,26,0.95)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


def plot_training_history(history: list) -> go.Figure:
    epochs    = [h["epoch"]    for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    val_acc    = [h["val_acc"]    for h in history]
    val_auc    = [h["val_auc"]    for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss",
                             line=dict(color="#4A90D9", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss",
                             line=dict(color="#E74C3C", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, name="Val Accuracy",
                             line=dict(color="#2ECC71", width=2, dash="dash"),
                             yaxis="y2"))
    fig.add_trace(go.Scatter(x=epochs, y=val_auc, name="Val AUC",
                             line=dict(color="#F39C12", width=2, dash="dot"),
                             yaxis="y2"))

    fig.update_layout(
        title="Training History",
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Loss", side="left"),
        yaxis2=dict(title="Accuracy / AUC", side="right", overlaying="y",
                    range=[0, 1]),
        height=350,
        legend=dict(x=0.7, y=0.95),
        plot_bgcolor="rgba(15,17,26,0.95)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Page Layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("🧠 Page 10 · GNN Predictive Analysis & Explainability")
st.caption("Graph Snapshot Classification ด้วย PyTorch Geometric + Captum Integrated Gradients")

metrics     = load_metrics()
attr_df     = load_attributions()
all_attr_df = load_all_attributions()
pred_df     = load_predictions()
label_info  = load_label_info()

# ─────────────────────────────────────────────────────────────────────────────
# Section 0: Run Pipeline Button (ถ้ายังไม่มี results)
# ─────────────────────────────────────────────────────────────────────────────

if metrics is None:
    st.warning("⚠️ ยังไม่มีผลลัพธ์จากการเทรน กรุณารัน pipeline ก่อน")

    with st.expander("📋 วิธีรัน GNN Pipeline", expanded=True):
        st.code("""
# 1. ติดตั้ง dependencies
pip install torch-geometric captum

# 2. สร้าง Graph Snapshot Dataset
python src/gnn_dataset.py

# 3. เทรน GNN Model
python src/gnn_train.py

# 4. คำนวณ Captum Attribution
python src/gnn_explain.py

# 5. รีเฟรชหน้านี้
        """, language="bash")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Model Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────

st.header("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Test Accuracy",
              f"{metrics['test']['accuracy']:.1%}",
              delta=f"Val: {metrics['val']['accuracy']:.1%}")
with col2:
    st.metric("Test AUC-ROC",
              f"{metrics['test']['auc']:.3f}",
              delta=f"Val: {metrics['val']['auc']:.3f}")
with col3:
    st.metric("Best Epoch",
              metrics["best_epoch"],
              delta=f"Val Loss: {metrics['best_val_loss']:.4f}")
with col4:
    n_total = metrics["dataset"]["total_snapshots"]
    n_hr    = metrics["dataset"]["class1_high_risk"]
    st.metric("Dataset Size",
              f"{n_total} snapshots",
              delta=f"High Risk: {n_hr/n_total:.0%}")

# ── Dataset Info ──────────────────────────────────────────────────────────────
st.caption(
    f"Train: {metrics['dataset']['n_train']} | "
    f"Val: {metrics['dataset']['n_val']} | "
    f"Test: {metrics['dataset']['n_test']} snapshots "
    f"(time-aware split, ไม่ shuffle)"
)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Latest Week Prediction
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.header("🔮 Prediction — สัปดาห์ล่าสุด")

if attr_df is not None and len(attr_df) > 0:
    latest_date  = attr_df["date"].iloc[0]
    latest_pred  = int(attr_df["pred_label"].iloc[0])
    latest_prob  = float(attr_df["prob_high_risk"].iloc[0])

    col_pred, col_prob = st.columns([1, 2])
    with col_pred:
        if latest_pred == 1:
            st.error(f"⚠️ **HIGH RISK**\nสัปดาห์: {latest_date}")
        else:
            st.success(f"✅ **STABLE**\nสัปดาห์: {latest_date}")

    with col_prob:
        st.metric("Probability of High Risk", f"{latest_prob:.1%}")
        st.progress(latest_prob)
        st.caption("ยิ่งสูง = โมเดลมั่นใจว่า network กำลังเข้าสู่ภาวะ High Volatility")

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Edge Attribution Network
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.header("🕸️ Edge Attribution Network — Captum Integrated Gradients")

st.info(
    "**การอ่านกราฟ:** Edge ที่**หนากว่า**และ**สีแดงกว่า** = มีอิทธิพลสูงกว่าต่อการทำนาย High Risk\n\n"
    "Attribution Score คำนวณจาก Integrated Gradients — วัดว่าถ้า edge นั้นหายไป "
    "การทำนายจะเปลี่ยนไปมากแค่ไหน"
)

tab1, tab2 = st.tabs(["สัปดาห์ล่าสุด", "ค่าเฉลี่ย Test Set"])

with tab1:
    if attr_df is not None:
        fig_net = plot_attribution_network(
            attr_df, title=f"Edge Attribution — {latest_date}", top_n=15)
        st.plotly_chart(fig_net, use_container_width=True)

        st.subheader("Top 15 Most Important Edges")
        display_df = attr_df.head(15)[["rank", "src_node", "tgt_node", "attribution_score"]].copy()
        display_df.columns = ["Rank", "Node A", "Node B", "Attribution Score"]
        display_df["Attribution Score"] = display_df["Attribution Score"].map("{:.4f}".format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab2:
    if all_attr_df is not None:
        avg_attr = (all_attr_df
                    .groupby(["src_node", "tgt_node"])["attribution_score"]
                    .mean()
                    .reset_index()
                    .sort_values("attribution_score", ascending=False)
                    .reset_index(drop=True))
        avg_attr["rank"] = avg_attr.index + 1

        fig_avg = plot_attribution_network(
            avg_attr, title="Average Edge Attribution — Test Set", top_n=15)
        st.plotly_chart(fig_avg, use_container_width=True)

        st.subheader("Top 15 Edges — Average Attribution (Test Period)")
        display_avg = avg_attr.head(15)[["rank", "src_node", "tgt_node", "attribution_score"]].copy()
        display_avg.columns = ["Rank", "Node A", "Node B", "Avg Attribution Score"]
        display_avg["Avg Attribution Score"] = display_avg["Avg Attribution Score"].map("{:.4f}".format)
        st.dataframe(display_avg, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Test Set Predictions
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.header("📈 Test Set Predictions")

if pred_df is not None:
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["prob_high_risk"],
        mode="lines+markers", name="P(High Risk)",
        line=dict(color="#E74C3C", width=2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
    ))
    fig_pred.add_hline(y=0.5, line_dash="dash", line_color="gray",
                       annotation_text="Threshold = 0.50")

    # Mark actual High Risk weeks
    hr_weeks = pred_df[pred_df["true_label"] == 1]
    fig_pred.add_trace(go.Scatter(
        x=hr_weeks["date"], y=hr_weeks["prob_high_risk"],
        mode="markers", name="Actual High Risk",
        marker=dict(color="red", size=8, symbol="x"),
    ))

    fig_pred.update_layout(
        title="Predicted Probability of High Risk — Test Period",
        xaxis_title="Week", yaxis_title="P(High Risk)",
        yaxis=dict(range=[0, 1]),
        height=350,
        plot_bgcolor="rgba(15,17,26,0.95)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        correct = (pred_df["true_label"] == pred_df["pred_label"]).sum()
        st.metric("Correct Predictions", f"{correct}/{len(pred_df)}")
    with col_b:
        hr_correct = pred_df[pred_df["true_label"] == 1]
        if len(hr_correct) > 0:
            hr_acc = (hr_correct["pred_label"] == 1).mean()
            st.metric("High Risk Detection Rate", f"{hr_acc:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Training History
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.header("📉 Training History")

if metrics and "history" in metrics:
    fig_hist = plot_training_history(metrics["history"])
    st.plotly_chart(fig_hist, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Methodology Summary
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
with st.expander("📚 Methodology — วิธีการทำงานของโมเดล"):
    st.markdown("""
**Graph Snapshot Classification**
แปลงข้อมูลรายสัปดาห์แต่ละสัปดาห์ (t) เป็น 1 Graph โดยมี:
- **18 Nodes** = หุ้นธนาคารไทย 7 ตัว + ตัวแปรโลก 11 ตัว
- **Node Features (8 ต่อ node)** = 4W log returns + 4W rolling std + Degree, Betweenness, PageRank
- **60 Edges** = Validated Correlation edges (static topology)
- **Edge Weights** = Rolling 8W Pearson correlation (dynamic ต่อสัปดาห์)
- **Label** = Thai Bank Basket volatility สัปดาห์ถัดไป (t+1) เทียบ historical median

**GNN Architecture (FinancialGNN)**
```
GraphConv(8→64) → ReLU → Dropout(0.3)
GraphConv(64→64) → ReLU → Dropout(0.3)
global_mean_pool → Linear(64→32) → ReLU → Linear(32→2)
```

**Captum Integrated Gradients**
- คำนวณ gradient ของ P(High Risk) เทียบกับ edge_weight ของแต่ละ edge
- Average gradient จาก 50 interpolation steps (baseline = all zeros)
- Attribution Score สูง = edge นั้น drive การทำนาย High Risk มากที่สุด

**Time-aware Split** (ไม่ shuffle เพราะ time series)
- Train: 60% แรก | Val: 20% กลาง | Test: 20% หลังสุด
""")
