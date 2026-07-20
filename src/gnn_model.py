"""
gnn_model.py — GNN Architecture สำหรับ Graph Snapshot Classification

Architecture:
    GraphConv(8 → 64) → ReLU → Dropout
    GraphConv(64 → 64) → ReLU → Dropout
    global_mean_pool (graph readout)
    Linear(64 → 32) → ReLU
    Linear(32 → 2)  → class logits

ทำไมถึงเลือก GraphConv (ไม่ใช่ GCNConv):
    - GraphConv ไม่มี symmetric normalization (1/sqrt(deg_i * deg_j))
    - ใน financial graph ที่ USDTHB มี degree สูงมาก normalization จะทำให้ signal
      จาก USDTHB ถูก dilute → GraphConv รักษา expressiveness ได้ดีกว่า
    - ตรงกับ notebook 3 (optional exercise) และ notebook 6 ที่แสดงว่า
      GraphConv ให้ผลดีกว่า GCNConv สำหรับ graph classification

ทำไมถึงใช้แค่ 2 layers:
    - Graph มีแค่ 18 nodes → 2 hops ครอบคลุม neighborhood ได้เกือบทั้งหมด
    - 3 layers อาจเกิด over-smoothing ทำให้ node embeddings เหมือนกันหมด
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GraphConv, global_mean_pool


class FinancialGNN(torch.nn.Module):
    """
    GNN สำหรับจำแนกประเภท Graph Snapshot (High Risk vs Stable)

    Args:
        in_channels     : จำนวน node features (default 8)
        hidden_channels : ขนาด hidden layer (default 64)
        out_channels    : จำนวน output classes (default 2)
        dropout         : dropout rate (default 0.3)
    """

    def __init__(self, in_channels: int = 8, hidden_channels: int = 64,
                 out_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        torch.manual_seed(42)

        self.dropout = dropout

        # ── Message Passing Layers ────────────────────────────────────────────
        # GraphConv รองรับ edge_weight → จำเป็นสำหรับ Captum attribution
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)

        # ── Readout + Classifier ──────────────────────────────────────────────
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x           : [N, in_channels]  node features
            edge_index  : [2, E]            edge indices
            batch       : [N]               batch vector (node → graph mapping)
            edge_weight : [E]               edge weights (optional, ใช้สำหรับ Captum)

        Returns:
            out : [batch_size, out_channels] raw logits (ใช้กับ CrossEntropyLoss)
        """
        # ── Layer 1 ───────────────────────────────────────────────────────────
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Layer 2 ───────────────────────────────────────────────────────────
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Global Readout (node embeddings → graph embedding) ────────────────
        # global_mean_pool รวม node embeddings ทั้งหมดในแต่ละกราฟเป็น 1 vector
        x = global_mean_pool(x, batch)

        # ── Classifier ────────────────────────────────────────────────────────
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x  # raw logits (CrossEntropyLoss expects logits, not softmax)

    def get_embedding(self, x: torch.Tensor, edge_index: torch.Tensor,
                      batch: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        ดึง graph-level embedding (ก่อน classifier) สำหรับ visualization
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return x


def create_model(in_channels: int = 8, hidden_channels: int = 64,
                 out_channels: int = 2, dropout: float = 0.3) -> FinancialGNN:
    """Factory function สร้าง model พร้อม log architecture"""
    model = FinancialGNN(in_channels, hidden_channels, out_channels, dropout)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FinancialGNN Architecture:")
    print(f"  GraphConv({in_channels} → {hidden_channels}) → ReLU → Dropout({dropout})")
    print(f"  GraphConv({hidden_channels} → {hidden_channels}) → ReLU → Dropout({dropout})")
    print(f"  global_mean_pool")
    print(f"  Linear({hidden_channels} → {hidden_channels//2}) → ReLU")
    print(f"  Linear({hidden_channels//2} → {out_channels})")
    print(f"  Total parameters: {n_params:,}")
    return model


if __name__ == "__main__":
    # Quick test
    model = create_model()

    # Dummy input (5 snapshots, 18 nodes each)
    from torch_geometric.data import Data, Batch

    graphs = []
    for _ in range(5):
        data = Data(
            x          = torch.randn(18, 8),
            edge_index = torch.randint(0, 18, (2, 120)),
            edge_attr  = torch.rand(120),
            y          = torch.randint(0, 2, (1,)),
        )
        graphs.append(data)

    batch = Batch.from_data_list(graphs)
    out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
    print(f"\nTest forward pass: input graphs=5, output shape={tuple(out.shape)}")
    print("Model test PASSED ✓")
