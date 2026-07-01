"""
neo4j_loader.py — Load nodes and edges from CSV files into Neo4j.

Requires:
    Neo4j running with GDS plugin installed.
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD set in .env

Usage:
    python src/neo4j_loader.py
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


class Neo4jLoader:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        self.db = NEO4J_DATABASE
        log.info("Connected to Neo4j: %s", NEO4J_URI)

    def close(self):
        self.driver.close()

    def run(self, query: str, params: dict = None):
        with self.driver.session(database=self.db) as session:
            result = session.run(query, params or {})
            return result.data()

    # ── Constraints ──────────────────────────────────────────────────────────

    def create_constraints(self):
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Bank)          REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ETF)           REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:MacroFactor)   REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:FX)            REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Index)         REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DerivedFactor) REQUIRE n.node_id IS UNIQUE",
        ]
        for c in constraints:
            self.run(c)
        log.info("Constraints created.")

    # ── Clear graph ───────────────────────────────────────────────────────────

    def clear_graph(self):
        self.run("MATCH (n) DETACH DELETE n")
        log.info("Graph cleared.")

    # ── Load nodes ────────────────────────────────────────────────────────────

    def load_nodes(self, nodes_df: pd.DataFrame):
        for _, row in nodes_df.iterrows():
            label   = row["label"]
            node_id = row["node_id"]
            name    = row["name"]
            ticker  = row["ticker"]
            query = f"""
            MERGE (n:{label} {{node_id: $node_id}})
            SET n.name   = $name,
                n.ticker = $ticker,
                n.type   = $label
            """
            self.run(query, {"node_id": node_id, "name": name, "label": label, "ticker": ticker})
        log.info("Loaded %d nodes.", len(nodes_df))

    # ── Load edges ────────────────────────────────────────────────────────────

    def load_edges(self, edges_df: pd.DataFrame):
        loaded = 0
        skipped = 0

        # Lookup node labels
        node_labels = {}
        nodes_df = pd.read_csv(PATHS["neo4j_nodes"])
        for _, row in nodes_df.iterrows():
            node_labels[row["node_id"]] = row["label"]

        for _, row in edges_df.iterrows():
            src_id  = row["source"]
            tgt_id  = row["target"]
            rel     = row["rel_type"]
            weight  = float(row.get("weight", 0) or 0)
            sign    = str(row.get("sign", ""))
            method  = str(row.get("method", ""))

            src_label = node_labels.get(src_id)
            tgt_label = node_labels.get(tgt_id)

            if not src_label or not tgt_label:
                skipped += 1
                continue

            # Build optional properties
            props = {"weight": weight, "sign": sign, "method": method}
            for prop in ["corr_pearson", "p_adj", "lag_weeks", "beta", "p_value"]:
                if prop in row and pd.notna(row[prop]) and row[prop] != "":
                    props[prop] = float(row[prop]) if prop != "lag_weeks" else int(row[prop])

            prop_str = ", ".join(f"r.{k} = ${k}" for k in props.keys())
            query = f"""
            MATCH (a:{src_label} {{node_id: $src_id}})
            MATCH (b:{tgt_label} {{node_id: $tgt_id}})
            MERGE (a)-[r:{rel}]->(b)
            SET {prop_str}
            """
            params = {"src_id": src_id, "tgt_id": tgt_id, **props}
            try:
                self.run(query, params)
                loaded += 1
            except Exception as e:
                log.warning("Edge %s→%s [%s] failed: %s", src_id, tgt_id, rel, e)
                skipped += 1

        log.info("Edges loaded: %d  |  skipped: %d", loaded, skipped)


def run_neo4j_load():
    log.info("=" * 60)
    log.info("Phase 4 — Neo4j Load")
    log.info("=" * 60)

    nodes_df = pd.read_csv(PATHS["neo4j_nodes"])
    edges_df = pd.read_csv(PATHS["neo4j_edges"])

    loader = Neo4jLoader()
    try:
        loader.create_constraints()
        loader.clear_graph()
        loader.load_nodes(nodes_df)
        loader.load_edges(edges_df)
        log.info("Neo4j load complete.")
    finally:
        loader.close()


if __name__ == "__main__":
    run_neo4j_load()
