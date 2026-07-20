"""
gnn_dataset.py — แปลงข้อมูล Time Series รายสัปดาห์เป็น PyG Graph Snapshot Dataset

Design Decisions (ตามที่วางแผนไว้):
  1. Label    : Thai Bank Basket 4W rolling volatility ที่ t+1 vs historical median
                Class 1 = High Risk (volatility > median), Class 0 = Stable
  2. Edge     : 60 Validated Correlation edges (static topology)
                Edge weight = rolling 8W Pearson correlation (dynamic per snapshot)
  3. Features : 4W log returns + 4W rolling std + 3 centrality metrics = 8 features/node

Output:
    data/results/gnn_snapshots.pt   — list of PyG Data objects
    data/results/gnn_label_info.csv — label distribution info
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PATHS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ─── Mapping: node_id (short) → column name ใน final_weekly_dataset.csv ──────
NODE_COL_MAP = {
    "BBL":              "BBL_ret",
    "KBANK":            "KBANK_ret",
    "KKP":              "KKP_ret",
    "KTB":              "KTB_ret",
    "SCB":              "SCB_ret",
    "TISCO":            "TISCO_ret",
    "TTB":              "TTB_ret",
    "XLF":              "XLF_ret",
    "EUFN":             "EUFN_ret",
    "USDTHB":           "USDTHB_ret",
    "SET":              "SET_ret",
    "FEDFUNDS":         "FEDFUNDS_chg",
    "DGS2":             "DGS2_chg",
    "DGS10":            "DGS10_chg",
    "MORTGAGE30US":     "MORTGAGE30US_chg",
    "VIX":              "VIX_CHANGE",
    "US_YIELD_CURVE":   "US_YIELD_CURVE_chg",
    "THAI_BANK_BASKET": "THAI_BANK_BASKET_ret",
}

# Reverse mapping: column name → node_id
COL_NODE_MAP = {v: k for k, v in NODE_COL_MAP.items()}

# Rolling windows
LOOKBACK_RETURNS = 4    # สัปดาห์ย้อนหลังสำหรับ node features
ROLLING_EDGE_W   = 8    # สัปดาห์สำหรับคำนวณ rolling Pearson edge weight
LABEL_VOL_W      = 4    # สัปดาห์สำหรับคำนวณ realized volatility (label)

# Paths สำหรับ save/load
GNN_SNAPSHOTS_PATH = Path(__file__).resolve().parent.parent / "data" / "results" / "gnn_snapshots.pt"
GNN_LABEL_INFO_PATH = Path(__file__).resolve().parent.parent / "data" / "results" / "gnn_label_info.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: โหลดข้อมูลทั้งหมด
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data():
    """โหลด dataset, edges, centrality จาก midterm pipeline"""
    df = pd.read_csv(PATHS["final_weekly_dataset"], index_col=0, parse_dates=True)
    edges_df = pd.read_csv(PATHS["corr_edges_validated"])
    nodes_df = pd.read_csv(PATHS["neo4j_nodes"])
    cent_df  = pd.read_csv(PATHS["centrality_results"])
    log.info("Data loaded: %d weeks, %d nodes, %d validated edges",
             len(df), len(nodes_df), len(edges_df))
    return df, edges_df, nodes_df, cent_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: สร้าง Static Edge Index (60 edges → undirected = 120 directed edges)
# ─────────────────────────────────────────────────────────────────────────────

def build_static_edge_index(edges_df: pd.DataFrame, node_list: list[str]):
    """
    แปลง 60 validated correlation edges เป็น edge_index tensor สำหรับ PyG
    undirected graph → เพิ่มทั้ง (src→tgt) และ (tgt→src)

    Returns:
        edge_index : LongTensor [2, 2*60]
        edge_pairs : list of (src_idx, tgt_idx) สำหรับคำนวณ edge weight ทีหลัง
        edge_col_pairs : list of (src_col, tgt_col) column names
    """
    node_idx = {n: i for i, n in enumerate(node_list)}

    src_list, tgt_list = [], []
    edge_pairs = []      # (src_node_idx, tgt_node_idx)
    edge_col_pairs = []  # (src_col, tgt_col)

    for _, row in edges_df.iterrows():
        src_col = row["source"]  # e.g. "BBL_ret"
        tgt_col = row["target"]  # e.g. "KBANK_ret"

        src_node = COL_NODE_MAP.get(src_col)
        tgt_node = COL_NODE_MAP.get(tgt_col)

        if src_node is None or tgt_node is None:
            log.warning("Cannot map edge %s → %s, skipping", src_col, tgt_col)
            continue
        if src_node not in node_idx or tgt_node not in node_idx:
            log.warning("Node not in list: %s or %s, skipping", src_node, tgt_node)
            continue

        si, ti = node_idx[src_node], node_idx[tgt_node]

        # Undirected: ใส่ทั้งสองทิศทาง
        src_list.extend([si, ti])
        tgt_list.extend([ti, si])
        edge_pairs.append((si, ti))
        edge_col_pairs.append((src_col, tgt_col))

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    log.info("Static edge_index: %d unique edges → %d directed edges",
             len(edge_pairs), edge_index.shape[1])
    return edge_index, edge_pairs, edge_col_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: คำนวณ Dynamic Edge Weights (rolling 8W Pearson)
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge_weights_at_t(df: pd.DataFrame, edge_col_pairs: list,
                               t: int, window: int = ROLLING_EDGE_W) -> np.ndarray:
    """
    คำนวณ rolling Pearson correlation สำหรับแต่ละ edge pair ณ สัปดาห์ t
    ใช้ข้อมูลย้อนหลัง `window` สัปดาห์ (inclusive)

    Returns:
        weights : array [n_unique_edges] ค่า correlation [-1, 1]
                  NaN จะถูก fallback เป็น 0
    """
    start = max(0, t - window + 1)
    weights = []

    for src_col, tgt_col in edge_col_pairs:
        if src_col not in df.columns or tgt_col not in df.columns:
            weights.append(0.0)
            continue

        window_src = df[src_col].iloc[start: t + 1].dropna()
        window_tgt = df[tgt_col].iloc[start: t + 1].dropna()

        # ต้องมี observations ร่วมกันอย่างน้อย 5 จุด
        common_idx = window_src.index.intersection(window_tgt.index)
        if len(common_idx) < 5:
            weights.append(0.0)
            continue

        try:
            corr = np.corrcoef(window_src[common_idx].values,
                               window_tgt[common_idx].values)[0, 1]
            weights.append(float(corr) if not np.isnan(corr) else 0.0)
        except Exception:
            weights.append(0.0)

    # แต่ละ unique edge มี 2 directed edges → ใส่ weight ซ้ำสำหรับ reverse
    # order: [e0_fwd, e0_rev, e1_fwd, e1_rev, ...]
    directed_weights = []
    for w in weights:
        directed_weights.extend([w, w])

    return np.array(directed_weights, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: คำนวณ Node Features ณ สัปดาห์ t
# ─────────────────────────────────────────────────────────────────────────────

def compute_node_features_at_t(df: pd.DataFrame, node_list: list[str],
                                cent_map: dict, t: int,
                                lookback: int = LOOKBACK_RETURNS) -> np.ndarray:
    """
    สร้าง node feature matrix [N, 8] ณ สัปดาห์ t

    Features (8 ต่อ node):
        [0-3] log returns ย้อนหลัง 4 สัปดาห์ (t-3, t-2, t-1, t)
        [4]   rolling std 4W
        [5]   degree_centrality (static)
        [6]   betweenness_centrality (static)
        [7]   pagerank (static)
    """
    n_nodes = len(node_list)
    n_feat  = lookback + 1 + 3  # 4 returns + 1 std + 3 centrality = 8
    X = np.zeros((n_nodes, n_feat), dtype=np.float32)

    for i, node_id in enumerate(node_list):
        col = NODE_COL_MAP.get(node_id)

        # ── Returns ──────────────────────────────────────────────────────────
        if col and col in df.columns:
            start = max(0, t - lookback + 1)
            returns = df[col].iloc[start: t + 1].fillna(0.0).values

            # Pad ซ้ายด้วย 0 ถ้าข้อมูลไม่ครบ
            if len(returns) < lookback:
                returns = np.pad(returns, (lookback - len(returns), 0))

            X[i, :lookback] = returns[-lookback:]

            # ── Rolling Std ───────────────────────────────────────────────────
            X[i, lookback] = float(np.std(returns[-lookback:]) if len(returns) >= 2 else 0.0)
        # else: leave as 0 (FEDFUNDS_chg ที่มีปัญหา)

        # ── Centrality (static จาก midterm) ──────────────────────────────────
        c = cent_map.get(node_id, {})
        X[i, lookback + 1] = c.get("degree_centrality", 0.0)
        X[i, lookback + 2] = c.get("betweenness_centrality", 0.0)
        X[i, lookback + 3] = c.get("pagerank", 0.0)

    return X


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: คำนวณ Binary Label
# ─────────────────────────────────────────────────────────────────────────────

def compute_labels(df: pd.DataFrame, vol_window: int = LABEL_VOL_W) -> pd.Series:
    """
    คำนวณ label สำหรับแต่ละสัปดาห์ t:
        ดู Thai Bank Basket rolling volatility ที่ t+1
        ถ้า > historical median → 1 (High Risk), ไม่เกิน → 0 (Stable)

    Returns:
        labels : pd.Series indexed เหมือน df, ค่า 0 หรือ 1
                 NaN สำหรับสัปดาห์สุดท้ายที่ไม่มี t+1
    """
    col = NODE_COL_MAP["THAI_BANK_BASKET"]
    rolling_vol = df[col].rolling(window=vol_window, min_periods=2).std()

    # Shift ย้อนกลับ 1 สัปดาห์ → label ณ t = volatility ที่ t+1
    next_vol = rolling_vol.shift(-1)

    median_vol = rolling_vol.median()
    log.info("Thai Bank Basket volatility median: %.6f", median_vol)

    labels = (next_vol > median_vol).astype(float)
    labels[next_vol.isna()] = float("nan")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: สร้าง PyG Data Snapshots ทั้งหมด
# ─────────────────────────────────────────────────────────────────────────────

def build_snapshots() -> list[Data]:
    """
    Main function: สร้าง list of PyG Data objects

    แต่ละ Data มี:
        data.x          : FloatTensor [18, 8]  — node features
        data.edge_index : LongTensor  [2, 120] — static edge structure (undirected)
        data.edge_attr  : FloatTensor [120]    — dynamic edge weights
        data.y          : LongTensor  [1]      — label (0 or 1)
        data.week_date  : str                  — วันที่ของสัปดาห์ t
        data.snapshot_idx : int               — index ใน dataset
    """
    df, edges_df, nodes_df, cent_df = load_raw_data()

    # ── Node list และ centrality map ─────────────────────────────────────────
    node_list = nodes_df["node_id"].tolist()
    cent_map  = cent_df.set_index("node_id")[
        ["degree_centrality", "betweenness_centrality", "pagerank"]
    ].to_dict("index")

    # ── Static edge structure ─────────────────────────────────────────────────
    edge_index, edge_pairs, edge_col_pairs = build_static_edge_index(edges_df, node_list)

    # ── Labels ───────────────────────────────────────────────────────────────
    labels = compute_labels(df)

    # ── ช่วงเวลาที่ valid ────────────────────────────────────────────────────
    # ต้องการ: lookback 4W (features) + rolling 8W (edges) → เริ่มที่ index 7
    # ต้องมี label (t+1) → จบที่ index T-2
    min_t = max(LOOKBACK_RETURNS - 1, ROLLING_EDGE_W - 1)  # = 7
    max_t = len(df) - 2  # สัปดาห์สุดท้ายที่มี label

    snapshots = []
    label_counts = {0: 0, 1: 0}

    for t in range(min_t, max_t + 1):
        label_val = labels.iloc[t]
        if pd.isna(label_val):
            continue

        # ── Node features ─────────────────────────────────────────────────────
        X = compute_node_features_at_t(df, node_list, cent_map, t)

        # ── Edge weights ──────────────────────────────────────────────────────
        ew = compute_edge_weights_at_t(df, edge_col_pairs, t)

        # ── Assemble PyG Data object ──────────────────────────────────────────
        data = Data(
            x          = torch.tensor(X, dtype=torch.float),
            edge_index = edge_index.clone(),
            edge_attr  = torch.tensor(ew, dtype=torch.float),
            y          = torch.tensor([int(label_val)], dtype=torch.long),
        )
        data.week_date    = str(df.index[t].date())
        data.snapshot_idx = len(snapshots)

        snapshots.append(data)
        label_counts[int(label_val)] += 1

    log.info("Built %d snapshots: Class0=%d (Stable), Class1=%d (High Risk)",
             len(snapshots), label_counts[0], label_counts[1])
    log.info("Node features: [18 nodes × 8 features] per snapshot")
    log.info("Edge structure: %d directed edges (60 unique × 2)", edge_index.shape[1])

    return snapshots, node_list, edge_pairs, edge_col_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def save_snapshots(snapshots: list[Data], node_list: list, edge_pairs: list,
                   edge_col_pairs: list):
    GNN_SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "snapshots":      snapshots,
        "node_list":      node_list,
        "edge_pairs":     edge_pairs,
        "edge_col_pairs": edge_col_pairs,
    }, GNN_SNAPSHOTS_PATH)
    log.info("Saved: %s (%d snapshots)", GNN_SNAPSHOTS_PATH, len(snapshots))

    # Label distribution
    labels_arr = [s.y.item() for s in snapshots]
    dates_arr  = [s.week_date for s in snapshots]
    info_df = pd.DataFrame({"date": dates_arr, "label": labels_arr})
    info_df.to_csv(GNN_LABEL_INFO_PATH, index=False)
    log.info("Saved label info: %s", GNN_LABEL_INFO_PATH)


def load_snapshots():
    if not GNN_SNAPSHOTS_PATH.exists():
        raise FileNotFoundError(f"Snapshots not found. Run: python src/gnn_dataset.py first.")
    data = torch.load(GNN_SNAPSHOTS_PATH, weights_only=False)
    log.info("Loaded %d snapshots from %s", len(data["snapshots"]), GNN_SNAPSHOTS_PATH)
    return data["snapshots"], data["node_list"], data["edge_pairs"], data["edge_col_pairs"]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset_pipeline():
    log.info("=" * 60)
    log.info("GNN Dataset Pipeline — Graph Snapshot Builder")
    log.info("=" * 60)
    snapshots, node_list, edge_pairs, edge_col_pairs = build_snapshots()
    save_snapshots(snapshots, node_list, edge_pairs, edge_col_pairs)

    # Quick sanity check
    s0 = snapshots[0]
    log.info("Sample snapshot[0]: date=%s, y=%d, x.shape=%s, edge_attr.shape=%s",
             s0.week_date, s0.y.item(), tuple(s0.x.shape), tuple(s0.edge_attr.shape))
    return snapshots


if __name__ == "__main__":
    run_dataset_pipeline()
