"""
gnn_train.py — Training Pipeline สำหรับ Financial GNN

Key Decisions:
    - Time-aware split: train 60% / val 20% / test 20% (ไม่ shuffle เพราะ time series)
    - Loss: CrossEntropyLoss
    - Optimizer: Adam (lr=0.001)
    - Early stopping: patience=20 epochs (monitor val loss)
    - Metrics: Accuracy + AUC-ROC

Outputs:
    data/results/gnn_model.pt          — best model checkpoint
    data/results/gnn_metrics.json      — training history + final metrics
    data/results/gnn_predictions.csv   — predictions บน test set
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gnn_dataset import load_snapshots, run_dataset_pipeline, GNN_SNAPSHOTS_PATH
from src.gnn_model import create_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "data" / "results"
MODEL_PATH   = BASE / "gnn_model.pt"
METRICS_PATH = BASE / "gnn_metrics.json"
PRED_PATH    = BASE / "gnn_predictions.csv"

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_CHANNELS = 64
DROPOUT         = 0.3
LEARNING_RATE   = 0.001
BATCH_SIZE      = 16
MAX_EPOCHS      = 200
PATIENCE        = 20   # Early stopping patience


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Time-aware Train/Val/Test Split
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(snapshots: list[Data], train_ratio=0.60, val_ratio=0.20):
    """
    แบ่ง dataset แบบ time-aware (sequential, ไม่ shuffle)
    train: 60%, val: 20%, test: 20%

    ห้าม shuffle เพราะข้อมูลเป็น time series — ถ้า shuffle จะเกิด data leakage
    """
    n = len(snapshots)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_data = snapshots[:n_train]
    val_data   = snapshots[n_train: n_train + n_val]
    test_data  = snapshots[n_train + n_val:]

    log.info("Split: train=%d, val=%d, test=%d (total=%d)", len(train_data), len(val_data), len(test_data), n)
    log.info("Train period: %s → %s", train_data[0].week_date, train_data[-1].week_date)
    log.info("Val   period: %s → %s", val_data[0].week_date,   val_data[-1].week_date)
    log.info("Test  period: %s → %s", test_data[0].week_date,  test_data[-1].week_date)

    return train_data, val_data, test_data


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Training Functions
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device) -> float:
    """เทรน 1 epoch, return average loss"""
    model.train()
    total_loss = 0.0
    n_graphs   = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out  = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        loss = F.cross_entropy(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        n_graphs   += batch.num_graphs

    return total_loss / n_graphs


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """ประเมิน accuracy และ AUC-ROC"""
    model.eval()
    all_preds  = []
    all_probs  = []
    all_labels = []
    total_loss = 0.0
    n_graphs   = 0

    for batch in loader:
        batch = batch.to(device)
        out   = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        loss  = F.cross_entropy(out, batch.y)

        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        labels = batch.y.cpu().numpy()

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels)

        total_loss += loss.item() * batch.num_graphs
        n_graphs   += batch.num_graphs

    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return {
        "loss":     total_loss / n_graphs,
        "accuracy": acc,
        "auc":      auc,
        "preds":    all_preds,
        "probs":    all_probs,
        "labels":   all_labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_training():
    log.info("=" * 60)
    log.info("GNN Training — Financial Graph Snapshot Classification")
    log.info("=" * 60)

    # ── โหลด dataset ─────────────────────────────────────────────────────────
    if not GNN_SNAPSHOTS_PATH.exists():
        log.info("Snapshots not found. Building dataset first...")
        run_dataset_pipeline()

    snapshots, node_list, edge_pairs, edge_col_pairs = load_snapshots()

    # ── Label distribution ────────────────────────────────────────────────────
    labels = [s.y.item() for s in snapshots]
    n_class0 = labels.count(0)
    n_class1 = labels.count(1)
    log.info("Label distribution: Stable=%d (%.1f%%), High Risk=%d (%.1f%%)",
             n_class0, 100*n_class0/len(labels),
             n_class1, 100*n_class1/len(labels))

    # ── Split ─────────────────────────────────────────────────────────────────
    train_data, val_data, test_data = split_dataset(snapshots)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    in_channels = snapshots[0].x.shape[1]  # = 8
    model = create_model(
        in_channels=in_channels,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # ── Training Loop ─────────────────────────────────────────────────────────
    history = []
    best_val_loss = float("inf")
    best_epoch    = 0
    patience_count = 0

    log.info("Starting training (max %d epochs, patience %d)...", MAX_EPOCHS, PATIENCE)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_metrics["loss"], 6),
            "val_acc":    round(val_metrics["accuracy"], 4),
            "val_auc":    round(val_metrics["auc"], 4),
        })

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch    = epoch
            patience_count = 0
            torch.save({
                "model_state":   model.state_dict(),
                "in_channels":   in_channels,
                "hidden_channels": HIDDEN_CHANNELS,
                "dropout":       DROPOUT,
                "epoch":         epoch,
                "val_loss":      best_val_loss,
            }, MODEL_PATH)
        else:
            patience_count += 1

        if epoch % 20 == 0 or epoch == 1:
            log.info("Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f | val_auc=%.3f",
                     epoch, train_loss, val_metrics["loss"],
                     val_metrics["accuracy"], val_metrics["auc"])

        if patience_count >= PATIENCE:
            log.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
            break

    # ── Load Best Model และ Evaluate บน Test Set ──────────────────────────────
    log.info("Loading best model (epoch %d, val_loss=%.4f)...", best_epoch, best_val_loss)
    ckpt = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_metrics  = evaluate(model, test_loader,  device)
    train_metrics = evaluate(model, train_loader, device)
    val_metrics_final = evaluate(model, val_loader, device)

    log.info("=" * 50)
    log.info("FINAL RESULTS:")
    log.info("  Train: acc=%.3f, auc=%.3f", train_metrics["accuracy"], train_metrics["auc"])
    log.info("  Val  : acc=%.3f, auc=%.3f", val_metrics_final["accuracy"], val_metrics_final["auc"])
    log.info("  Test : acc=%.3f, auc=%.3f", test_metrics["accuracy"], test_metrics["auc"])
    log.info("=" * 50)

    # Classification Report
    print("\nTest Set Classification Report:")
    print(classification_report(test_metrics["labels"], test_metrics["preds"],
                                target_names=["Stable", "High Risk"]))

    # ── Save Predictions ──────────────────────────────────────────────────────
    test_dates = [snapshots[len(train_data) + len(val_data) + i].week_date
                  for i in range(len(test_data))]
    pred_df = pd.DataFrame({
        "date":      test_dates,
        "true_label": test_metrics["labels"],
        "pred_label": test_metrics["preds"],
        "prob_high_risk": [round(p, 4) for p in test_metrics["probs"]],
    })
    pred_df.to_csv(PRED_PATH, index=False)
    log.info("Saved predictions: %s", PRED_PATH)

    # ── Save Metrics ──────────────────────────────────────────────────────────
    metrics_out = {
        "best_epoch":    best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "train": {
            "accuracy": round(train_metrics["accuracy"], 4),
            "auc":      round(train_metrics["auc"], 4),
        },
        "val": {
            "accuracy": round(val_metrics_final["accuracy"], 4),
            "auc":      round(val_metrics_final["auc"], 4),
        },
        "test": {
            "accuracy": round(test_metrics["accuracy"], 4),
            "auc":      round(test_metrics["auc"], 4),
        },
        "dataset": {
            "total_snapshots":   len(snapshots),
            "n_train":           len(train_data),
            "n_val":             len(val_data),
            "n_test":            len(test_data),
            "class0_stable":     n_class0,
            "class1_high_risk":  n_class1,
        },
        "hyperparams": {
            "hidden_channels": HIDDEN_CHANNELS,
            "dropout":         DROPOUT,
            "lr":              LEARNING_RATE,
            "batch_size":      BATCH_SIZE,
        },
        "history": history,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    log.info("Saved metrics: %s", METRICS_PATH)

    return model, metrics_out


if __name__ == "__main__":
    run_training()
