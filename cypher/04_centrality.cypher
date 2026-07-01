// 04_centrality.cypher
// Run Degree, Betweenness, and PageRank centrality on combinedGraph.

// ── Degree Centrality ────────────────────────────────────────────────────────
CALL gds.degree.stream('combinedGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name    AS name,
       gds.util.asNode(nodeId).node_id AS node_id,
       gds.util.asNode(nodeId).type    AS type,
       score AS degree_centrality
ORDER BY degree_centrality DESC;

// ── Betweenness Centrality ───────────────────────────────────────────────────
CALL gds.betweenness.stream('combinedGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name    AS name,
       gds.util.asNode(nodeId).node_id AS node_id,
       gds.util.asNode(nodeId).type    AS type,
       score AS betweenness_centrality
ORDER BY betweenness_centrality DESC;

// ── PageRank ──────────────────────────────────────────────────────────────────
CALL gds.pageRank.stream('combinedGraph', {
  relationshipWeightProperty: 'weight',
  maxIterations: 20,
  dampingFactor: 0.85
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name    AS name,
       gds.util.asNode(nodeId).node_id AS node_id,
       gds.util.asNode(nodeId).type    AS type,
       score AS pagerank
ORDER BY pagerank DESC;

// ── Write back to nodes (optional) ───────────────────────────────────────────
CALL gds.degree.write('combinedGraph', { writeProperty: 'degree_centrality' })
YIELD nodePropertiesWritten;

CALL gds.betweenness.write('combinedGraph', { writeProperty: 'betweenness_centrality' })
YIELD nodePropertiesWritten;

CALL gds.pageRank.write('combinedGraph', {
  relationshipWeightProperty: 'weight',
  writeProperty: 'pagerank'
})
YIELD nodePropertiesWritten;

// ── Verify ────────────────────────────────────────────────────────────────────
MATCH (n)
WHERE n.degree_centrality IS NOT NULL
RETURN n.name AS name, n.type AS type,
       n.degree_centrality AS degree,
       n.betweenness_centrality AS betweenness,
       n.pagerank AS pagerank
ORDER BY degree DESC;
