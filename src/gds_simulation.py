"""
gds_simulation.py — Compute GDS-equivalent metrics using networkx + sklearn.

Produces the same CSV outputs as gds_algorithms.py (Neo4j GDS) so that
Page 8 of the Streamlit app works without a running Neo4j instance.

Outputs (same schema as gds_algorithms.py):
    data/results/centrality_results.csv
    data/results/community_results_louvain.csv
    data/results/community_results_kmeans.csv

Run:
    python src/gds_simulation.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PATHS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Build networkx graph from neo4j CSVs
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """Build undirected weighted networkx graph from Neo4j node/edge CSVs."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx not installed. Run: pip install networkx")

    G = nx.Graph()

    for _, row in nodes_df.iterrows():
        G.add_node(row["node_id"],
                   name=row.get("name", row["node_id"]),
                   type=row.get("type", "Unknown"),
                   label=row.get("label", "Unknown"))

    for _, row in edges_df.iterrows():
        src = str(row.get("source", ""))
        tgt = str(row.get("target", ""))
        w   = float(row.get("weight", 0.1) or 0.1)
        if src in G and tgt in G and src != tgt:
            # For multigraph-like behavior keep highest weight per pair
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] = max(G[src][tgt]["weight"], w)
            else:
                G.add_edge(src, tgt, weight=w, rel_type=row.get("rel_type", ""))

    log.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Centrality metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_centrality(G, nodes_df: pd.DataFrame) -> pd.DataFrame:
    import networkx as nx

    degree      = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    pagerank    = nx.pagerank(G, weight="weight", alpha=0.85, max_iter=200)

    meta = nodes_df.set_index("node_id")[["name", "type"]].to_dict("index")

    records = []
    for nid in G.nodes():
        info = meta.get(nid, {"name": nid, "type": "Unknown"})
        records.append({
            "node_id":              nid,
            "name":                 info["name"],
            "type":                 info["type"],
            "degree_centrality":    round(degree.get(nid, 0), 6),
            "betweenness_centrality": round(betweenness.get(nid, 0), 6),
            "pagerank":             round(pagerank.get(nid, 0), 6),
        })

    df = pd.DataFrame(records).sort_values("degree_centrality", ascending=False).reset_index(drop=True)
    log.info("Centrality computed for %d nodes.", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Louvain community detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_louvain(G, nodes_df: pd.DataFrame) -> pd.DataFrame:
    try:
        import community as community_louvain
    except ImportError:
        raise ImportError("python-louvain not installed. Run: pip install python-louvain")

    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    meta = nodes_df.set_index("node_id")[["name", "type"]].to_dict("index")

    records = []
    for nid, comm_id in partition.items():
        info = meta.get(nid, {"name": nid, "type": "Unknown"})
        records.append({
            "node_id":     nid,
            "name":        info["name"],
            "type":        info["type"],
            "communityId": comm_id,
        })

    df = pd.DataFrame(records).sort_values(["communityId", "name"]).reset_index(drop=True)
    n_comm = df["communityId"].nunique()
    log.info("Louvain: %d communities detected.", n_comm)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Spectral embedding + K-Means (FastRP analogue)
# ─────────────────────────────────────────────────────────────────────────────

def compute_kmeans(G, nodes_df: pd.DataFrame, k: int = 4, embed_dim: int = 16) -> pd.DataFrame:
    import networkx as nx
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    nodes = list(G.nodes())
    n = len(nodes)
    if n < k:
        k = max(2, n // 2)

    # Build feature matrix from centrality + adjacency row
    # Use adjacency matrix (weighted) as a simple "embedding"
    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")

    # Dimensionality reduction via SVD if needed
    if A.shape[1] > embed_dim:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=embed_dim, random_state=42)
        features = svd.fit_transform(A)
    else:
        features = A

    features = StandardScaler().fit_transform(features)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(features)

    try:
        sil = silhouette_score(features, labels)
    except Exception:
        sil = float("nan")

    meta = nodes_df.set_index("node_id")[["name", "type"]].to_dict("index")
    records = []
    for nid, comm_id in zip(nodes, labels):
        info = meta.get(nid, {"name": nid, "type": "Unknown"})
        records.append({
            "node_id":     nid,
            "name":        info["name"],
            "type":        info["type"],
            "communityId": int(comm_id),
            "silhouette":  round(sil, 4),
        })

    df = pd.DataFrame(records).sort_values(["communityId", "name"]).reset_index(drop=True)
    log.info("K-Means (k=%d): silhouette=%.4f", k, sil)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_gds_simulation():
    log.info("=" * 60)
    log.info("GDS Simulation (networkx) — Phase 4 alternative")
    log.info("=" * 60)

    nodes_path = PATHS["neo4j_nodes"]
    edges_path = PATHS["neo4j_edges"]

    if not nodes_path.exists() or not edges_path.exists():
        log.error("neo4j_nodes.csv or neo4j_edges.csv not found. Run Phase 3 first.")
        return

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    G = build_graph(nodes_df, edges_df)

    # Centrality
    centrality_df = compute_centrality(G, nodes_df)
    centrality_df.to_csv(PATHS["centrality_results"], index=False)
    log.info("Saved: %s", PATHS["centrality_results"])

    # Louvain
    louvain_df = compute_louvain(G, nodes_df)
    louvain_df.to_csv(PATHS["community_louvain"], index=False)
    log.info("Saved: %s", PATHS["community_louvain"])

    # K-Means
    kmeans_df = compute_kmeans(G, nodes_df, k=4, embed_dim=16)
    kmeans_df.to_csv(PATHS["community_kmeans"], index=False)
    log.info("Saved: %s", PATHS["community_kmeans"])

    log.info("GDS simulation complete.")
    return centrality_df, louvain_df, kmeans_df


if __name__ == "__main__":
    run_gds_simulation()
