"""
gds_algorithms.py — Run Neo4j GDS algorithms and save results to CSV.

Algorithms:
    Degree Centrality, Betweenness Centrality, PageRank
    Louvain Community Detection
    FastRP Embedding + K-Means

Requires Neo4j running with GDS plugin installed.

Usage:
    python src/gds_algorithms.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE, PATHS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

GRAPH_NAME = "combinedGraph"
NODE_LABELS = ["Bank", "ETF", "MacroFactor", "FX", "Index", "DerivedFactor"]


class GDSRunner:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        self.db = NEO4J_DATABASE

    def close(self):
        self.driver.close()

    def run(self, query: str, params: dict = None) -> list:
        with self.driver.session(database=self.db) as session:
            result = session.run(query, params or {})
            return result.data()

    # ── Graph projection ──────────────────────────────────────────────────────

    def drop_graph_if_exists(self, name: str = GRAPH_NAME):
        check = self.run(f"RETURN gds.graph.exists('{name}') AS exists")[0]["exists"]
        if check:
            self.run(f"CALL gds.graph.drop('{name}')")
            log.info("Dropped existing projection: %s", name)

    def project_graph(self, name: str = GRAPH_NAME):
        self.drop_graph_if_exists(name)
        query = f"""
        CALL gds.graph.project(
          '{name}',
          {str(NODE_LABELS)},
          {{
            CORRELATED_WITH:         {{ orientation: 'UNDIRECTED', properties: 'weight' }},
            PARTIAL_CORRELATED_WITH: {{ orientation: 'UNDIRECTED', properties: 'weight' }},
            LAGGED_CORRELATED_WITH:  {{ orientation: 'NATURAL',    properties: 'weight' }},
            INFLUENCES:              {{ orientation: 'NATURAL',    properties: 'weight' }}
          }}
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """
        result = self.run(query)
        if result:
            r = result[0]
            log.info("Projected '%s': %d nodes, %d relationships",
                     r["graphName"], r["nodeCount"], r["relationshipCount"])
        return result

    # ── Centrality ────────────────────────────────────────────────────────────

    def degree_centrality(self, name: str = GRAPH_NAME) -> pd.DataFrame:
        query = f"""
        CALL gds.degree.stream('{name}')
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).name AS name,
               gds.util.asNode(nodeId).node_id AS node_id,
               gds.util.asNode(nodeId).type AS type,
               score AS degree_centrality
        ORDER BY score DESC
        """
        return pd.DataFrame(self.run(query))

    def betweenness_centrality(self, name: str = GRAPH_NAME) -> pd.DataFrame:
        query = f"""
        CALL gds.betweenness.stream('{name}')
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).name AS name,
               gds.util.asNode(nodeId).node_id AS node_id,
               gds.util.asNode(nodeId).type AS type,
               score AS betweenness_centrality
        ORDER BY score DESC
        """
        return pd.DataFrame(self.run(query))

    def pagerank(self, name: str = GRAPH_NAME) -> pd.DataFrame:
        query = f"""
        CALL gds.pageRank.stream('{name}', {{
          relationshipWeightProperty: 'weight',
          maxIterations: 20,
          dampingFactor: 0.85
        }})
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).name AS name,
               gds.util.asNode(nodeId).node_id AS node_id,
               gds.util.asNode(nodeId).type AS type,
               score AS pagerank
        ORDER BY score DESC
        """
        return pd.DataFrame(self.run(query))

    # ── Community detection ───────────────────────────────────────────────────

    def louvain(self, name: str = GRAPH_NAME) -> pd.DataFrame:
        query = f"""
        CALL gds.louvain.stream('{name}', {{
          relationshipWeightProperty: 'weight'
        }})
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).name AS name,
               gds.util.asNode(nodeId).node_id AS node_id,
               gds.util.asNode(nodeId).type AS type,
               communityId
        ORDER BY communityId, name
        """
        return pd.DataFrame(self.run(query))

    # ── FastRP + K-Means ──────────────────────────────────────────────────────

    def fastrp_mutate(self, name: str = GRAPH_NAME, dim: int = 16, seed: int = 42):
        query = f"""
        CALL gds.fastRP.mutate('{name}', {{
          embeddingDimension: {dim},
          randomSeed: {seed},
          mutateProperty: 'embedding'
        }})
        YIELD nodePropertiesWritten
        RETURN nodePropertiesWritten
        """
        result = self.run(query)
        log.info("FastRP mutate: %s", result)

    def kmeans(self, name: str = GRAPH_NAME, k: int = 4, seed: int = 42) -> pd.DataFrame:
        query = f"""
        CALL gds.kmeans.stream('{name}', {{
          nodeProperty: 'embedding',
          k: {k},
          randomSeed: {seed}
        }})
        YIELD nodeId, communityId, distanceFromCentroid, silhouette
        RETURN gds.util.asNode(nodeId).name AS name,
               gds.util.asNode(nodeId).node_id AS node_id,
               gds.util.asNode(nodeId).type AS type,
               communityId,
               distanceFromCentroid,
               silhouette
        ORDER BY communityId, name
        """
        return pd.DataFrame(self.run(query))


def run_gds_algorithms():
    log.info("=" * 60)
    log.info("Phase 4 — GDS Algorithms")
    log.info("=" * 60)

    gds = GDSRunner()
    try:
        # Project graph
        gds.project_graph()

        # Centrality
        log.info("Running Degree Centrality...")
        degree = gds.degree_centrality()

        log.info("Running Betweenness Centrality...")
        betweenness = gds.betweenness_centrality()

        log.info("Running PageRank...")
        pagerank = gds.pagerank()

        # Merge centrality results
        if not degree.empty and not betweenness.empty and not pagerank.empty:
            centrality = degree.merge(
                betweenness[["node_id", "betweenness_centrality"]], on="node_id", how="outer"
            ).merge(
                pagerank[["node_id", "pagerank"]], on="node_id", how="outer"
            )
        else:
            centrality = pd.concat([degree, betweenness, pagerank], axis=0)

        centrality.to_csv(PATHS["centrality_results"], index=False)
        log.info("Saved: %s", PATHS["centrality_results"])

        # Louvain
        log.info("Running Louvain...")
        louvain_df = gds.louvain()
        louvain_df.to_csv(PATHS["community_louvain"], index=False)
        log.info("Saved: %s", PATHS["community_louvain"])

        # FastRP + K-Means
        log.info("Running FastRP embedding...")
        gds.fastrp_mutate()

        log.info("Running K-Means (k=4)...")
        kmeans_df = gds.kmeans()
        kmeans_df.to_csv(PATHS["community_kmeans"], index=False)
        log.info("Saved: %s", PATHS["community_kmeans"])

        log.info("GDS algorithms complete.")

    finally:
        gds.close()


if __name__ == "__main__":
    run_gds_algorithms()
