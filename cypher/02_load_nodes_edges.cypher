// 02_load_nodes_edges.cypher
// Load nodes and edges from CSV files.
// Update the file:// paths to match your Neo4j import directory,
// or use neo4j_loader.py (Python) for automated loading.

// ── CLEAR EXISTING GRAPH ────────────────────────────────────────────────────
MATCH (n) DETACH DELETE n;

// ── LOAD NODES ───────────────────────────────────────────────────────────────
// Place neo4j_nodes.csv in Neo4j's import folder (e.g. $NEO4J_HOME/import/)
LOAD CSV WITH HEADERS FROM 'file:///neo4j_nodes.csv' AS row
CALL apoc.create.node([row.label], {
  node_id: row.node_id,
  name:    row.name,
  ticker:  row.ticker,
  type:    row.type
}) YIELD node
RETURN count(node) AS nodes_created;

// ── LOAD CORRELATED_WITH EDGES ───────────────────────────────────────────────
LOAD CSV WITH HEADERS FROM 'file:///neo4j_edges.csv' AS row
WHERE row.rel_type = 'CORRELATED_WITH'
MATCH (a {node_id: row.source})
MATCH (b {node_id: row.target})
MERGE (a)-[r:CORRELATED_WITH]->(b)
SET r.weight        = toFloat(row.weight),
    r.sign          = row.sign,
    r.method        = row.method,
    r.corr_pearson  = toFloat(row.corr_pearson),
    r.p_adj         = toFloat(row.p_adj)
RETURN count(r) AS corr_edges;

// ── LOAD PARTIAL_CORRELATED_WITH EDGES ──────────────────────────────────────
LOAD CSV WITH HEADERS FROM 'file:///neo4j_edges.csv' AS row
WHERE row.rel_type = 'PARTIAL_CORRELATED_WITH'
MATCH (a {node_id: row.source})
MATCH (b {node_id: row.target})
MERGE (a)-[r:PARTIAL_CORRELATED_WITH]->(b)
SET r.weight = toFloat(row.weight),
    r.sign   = row.sign,
    r.method = row.method
RETURN count(r) AS partial_edges;

// ── LOAD LAGGED_CORRELATED_WITH EDGES ───────────────────────────────────────
LOAD CSV WITH HEADERS FROM 'file:///neo4j_edges.csv' AS row
WHERE row.rel_type = 'LAGGED_CORRELATED_WITH'
MATCH (a {node_id: row.source})
MATCH (b {node_id: row.target})
MERGE (a)-[r:LAGGED_CORRELATED_WITH]->(b)
SET r.weight    = toFloat(row.weight),
    r.sign      = row.sign,
    r.method    = row.method,
    r.lag_weeks = toInteger(row.lag_weeks)
RETURN count(r) AS lagged_edges;

// ── LOAD INFLUENCES EDGES ────────────────────────────────────────────────────
LOAD CSV WITH HEADERS FROM 'file:///neo4j_edges.csv' AS row
WHERE row.rel_type = 'INFLUENCES'
MATCH (a {node_id: row.source})
MATCH (b {node_id: row.target})
MERGE (a)-[r:INFLUENCES]->(b)
SET r.weight  = toFloat(row.weight),
    r.sign    = row.sign,
    r.method  = row.method,
    r.beta    = toFloat(row.beta),
    r.p_value = toFloat(row.p_value)
RETURN count(r) AS influence_edges;

// ── VERIFY ────────────────────────────────────────────────────────────────────
MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label;
MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count ORDER BY rel_type;
