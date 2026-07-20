"""
gnn_explain.py — Captum Edge Attribution สำหรับ Financial GNN

วิธีที่ใช้: Integrated Gradients (IG)
    - คำนวณ gradient ของ output prediction เทียบกับ edge_weight
    - IG integrate gradient ตาม path จาก baseline (edge_weight=0) ถึง input จริง
    - Attribution score สูง = edge นั้นมีอิทธิพลสูงต่อการทำนาย High Risk

ทำไม Integrated Gradients ดีกว่า Saliency:
    - Saliency ใช้แค่ gradient ณ จุดเดียว → sensitive ต่อ noise
    - IG average gradients ตาม interpolation path → robust และ satisfy
      Completeness Axiom (sum of attributions = output - baseline output)
    - ตรงกับที่ notebook 6 แนะนำสำหรับ edge-level attribution

Output:
    data/results/gnn_attributions.csv  — edge attribution scores ล่าสุด
    data/results/gnn_all_attributions.csv — attribution scores ทุก test snapshot
"""

import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gnn_dataset import load_snapshots, NODE_COL_MAP
from src.gnn_model import FinancialGNN

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "data" / "results"
MODEL_PATH      = BASE / "gnn_model.pt"
ATTR_PATH       = BASE / "gnn_attributions.csv"
ALL_ATTR_PATH   = BASE / "gnn_all_attributions.csv"
METRICS_PATH    = BASE / "gnn_metrics.json"


# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device) -> FinancialGNN:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run gnn_train.py first.")
    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location=device)
    model = FinancialGNN(
        in_channels=ckpt["in_channels"],
        hidden_channels=ckpt["hidden_channels"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Loaded model from epoch %d (val_loss=%.4f)", ckpt["epoch"], ckpt["val_loss"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Captum: model_forward wrapper สำหรับ edge mask
# ─────────────────────────────────────────────────────────────────────────────

def make_model_forward(model, data, device):
    """
    สร้าง wrapper function ที่รับ edge_mask เป็น input สำหรับ Captum
    Captum จะ differentiate เทียบกับ edge_mask แทน raw edge_weight

    Pattern เดียวกับ notebook 6:
        def model_forward(edge_mask, data):
            batch = torch.zeros(...)
            out = model(data.x, data.edge_index, batch, edge_mask)
            return out
    """
    def model_forward(edge_mask):
        batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
        out   = model(data.x, data.edge_index, batch, edge_mask)
        # Return probability ของ High Risk (class 1)
        return F.softmax(out, dim=1)[:, 1]

    return model_forward


# ─────────────────────────────────────────────────────────────────────────────
# Integrated Gradients
# ─────────────────────────────────────────────────────────────────────────────

def explain_snapshot_ig(model, data, device, n_steps: int = 50) -> np.ndarray:
    """
    คำนวณ Integrated Gradients attribution สำหรับ 1 snapshot

    IG formula:
        attr(e_i) = (w_i - w_baseline) × ∫₀¹ ∂f(baseline + α(w-baseline))/∂w_i dα
        w_baseline = 0 (no edges)

    Args:
        model    : trained GNN
        data     : PyG Data object (1 snapshot)
        device   : torch device
        n_steps  : จำนวน interpolation steps (default 50)

    Returns:
        attributions : np.ndarray [n_edges] — attribution score ต่อ edge
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        raise ImportError("Captum not installed. Run: pip install captum")

    data = data.to(device)
    model_forward = make_model_forward(model, data, device)

    # Edge mask input (current edge weights)
    edge_mask = data.edge_attr.clone().requires_grad_(True)

    # Baseline = zeros (ไม่มี edges)
    baseline = torch.zeros_like(edge_mask)

    ig = IntegratedGradients(model_forward)
    attributions = ig.attribute(
        edge_mask,
        baselines=baseline,
        n_steps=n_steps,
        internal_batch_size=1,
    )

    return attributions.detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate Bidirectional Edge Attributions
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_attributions(attributions: np.ndarray, edge_col_pairs: list,
                            node_list: list) -> pd.DataFrame:
    """
    รวม attribution จาก 2 ทิศทาง (src→tgt และ tgt→src) เป็น 1 ค่าต่อ unique edge
    ใช้ค่าเฉลี่ยของ absolute attribution (ทิศทางไม่สำคัญ สำคัญแค่ magnitude)

    Pattern จาก notebook 6:
        aggregate_edge_directions(edge_mask, data)

    Returns:
        df : DataFrame ที่มีคอลัมน์ src_node, tgt_node, src_col, tgt_col,
             attribution_score, abs_attribution
    """
    # COL → node_id mapping
    col_to_node = {v: k for k, v in NODE_COL_MAP.items()}

    # attributions มี 2 ค่าต่อ unique edge: [fwd, rev, fwd, rev, ...]
    n_unique = len(edge_col_pairs)
    records  = []

    for i, (src_col, tgt_col) in enumerate(edge_col_pairs):
        fwd_idx = i * 2
        rev_idx = i * 2 + 1

        if fwd_idx >= len(attributions) or rev_idx >= len(attributions):
            break

        attr_fwd = float(attributions[fwd_idx])
        attr_rev = float(attributions[rev_idx])
        attr_avg = (abs(attr_fwd) + abs(attr_rev)) / 2.0

        src_node = col_to_node.get(src_col, src_col)
        tgt_node = col_to_node.get(tgt_col, tgt_col)

        records.append({
            "src_node":        src_node,
            "tgt_node":        tgt_node,
            "src_col":         src_col,
            "tgt_col":         tgt_col,
            "attr_forward":    round(attr_fwd, 6),
            "attr_reverse":    round(attr_rev, 6),
            "attribution_score": round(attr_avg, 6),
        })

    df = pd.DataFrame(records)
    df = df.sort_values("attribution_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main: Run Explanation บน Test Set + Latest Snapshot
# ─────────────────────────────────────────────────────────────────────────────

def run_explanation():
    log.info("=" * 60)
    log.info("Captum Edge Attribution — Integrated Gradients")
    log.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)

    snapshots, node_list, edge_pairs, edge_col_pairs = load_snapshots()

    # โหลด metrics เพื่อรู้ว่า test set เริ่มที่ index ไหน
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    n_train = metrics["dataset"]["n_train"]
    n_val   = metrics["dataset"]["n_val"]
    test_start_idx = n_train + n_val
    test_snapshots = snapshots[test_start_idx:]

    log.info("Running IG on %d test snapshots...", len(test_snapshots))

    all_attr_records = []

    for snap in test_snapshots:
        snap_device = snap.to(device)

        # คำนวณ prediction ก่อน
        with torch.no_grad():
            batch = torch.zeros(snap_device.x.shape[0], dtype=torch.long, device=device)
            out   = model(snap_device.x, snap_device.edge_index, batch, snap_device.edge_attr)
            pred  = out.argmax(dim=1).item()
            prob  = F.softmax(out, dim=1)[0, 1].item()

        # IG Attribution
        try:
            attributions = explain_snapshot_ig(model, snap, device)
            attr_df = aggregate_attributions(attributions, edge_col_pairs, node_list)
            attr_df["date"]           = snap.week_date
            attr_df["pred_label"]     = pred
            attr_df["prob_high_risk"] = round(prob, 4)
            all_attr_records.append(attr_df)
        except Exception as e:
            log.warning("IG failed for %s: %s", snap.week_date, e)
            continue

    if not all_attr_records:
        log.error("No attributions computed.")
        return

    all_attr_df = pd.concat(all_attr_records, ignore_index=True)
    all_attr_df.to_csv(ALL_ATTR_PATH, index=False)
    log.info("Saved all test attributions: %s", ALL_ATTR_PATH)

    # ── Latest Snapshot Attribution (สัปดาห์ล่าสุด) ───────────────────────────
    latest_snap = snapshots[-1]
    log.info("Running IG on latest snapshot: %s", latest_snap.week_date)

    attributions_latest = explain_snapshot_ig(model, latest_snap, device)
    latest_attr_df = aggregate_attributions(attributions_latest, edge_col_pairs, node_list)

    with torch.no_grad():
        batch = torch.zeros(latest_snap.x.shape[0], dtype=torch.long, device=device)
        out   = model(latest_snap.x.to(device), latest_snap.edge_index.to(device),
                      batch, latest_snap.edge_attr.to(device))
        pred  = out.argmax(dim=1).item()
        prob  = F.softmax(out, dim=1)[0, 1].item()

    latest_attr_df["date"]           = latest_snap.week_date
    latest_attr_df["pred_label"]     = pred
    latest_attr_df["prob_high_risk"] = round(prob, 4)
    latest_attr_df.to_csv(ATTR_PATH, index=False)

    log.info("Saved latest attribution: %s", ATTR_PATH)
    log.info("Latest snapshot: date=%s, pred=%s (prob_high_risk=%.3f)",
             latest_snap.week_date,
             "High Risk" if pred == 1 else "Stable",
             prob)

    # ── Top 10 Most Important Edges ───────────────────────────────────────────
    log.info("\nTop 10 Most Important Edges (Latest Snapshot):")
    log.info("%s", "-" * 60)
    for _, row in latest_attr_df.head(10).iterrows():
        log.info("  #%2d  %-20s ↔ %-20s  score=%.4f",
                 int(row["rank"]), row["src_node"], row["tgt_node"],
                 row["attribution_score"])

    # ── Average Attribution Across All Test Snapshots ─────────────────────────
    avg_attr = (all_attr_df
                .groupby(["src_node", "tgt_node"])["attribution_score"]
                .mean()
                .reset_index()
                .sort_values("attribution_score", ascending=False))

    log.info("\nTop 10 Edges by Average Attribution (All Test Snapshots):")
    log.info("%s", "-" * 60)
    for _, row in avg_attr.head(10).iterrows():
        log.info("  %-20s ↔ %-20s  avg_score=%.4f",
                 row["src_node"], row["tgt_node"], row["attribution_score"])

    return latest_attr_df, all_attr_df


if __name__ == "__main__":
    run_explanation()
