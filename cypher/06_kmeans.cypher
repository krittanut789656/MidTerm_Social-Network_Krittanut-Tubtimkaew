// 06_kmeans.cypher
// FastRP embedding + K-Means clustering via Neo4j GDS.
// Run AFTER 03_gds_projection.cypher.

// ── Step 1: Generate FastRP embeddings (mutate into projection) ───────────────
CALL gds.fastRP.mutate('combinedGraph', {
  embeddingDimension: 16,
  randomSeed: 42,
  mutateProperty: 'embedding'
})
YIELD nodePropertiesWritten, computeMillis
RETURN nodePropertiesWritten, computeMillis;

// ── Step 2: K-Means clustering (k=4) ─────────────────────────────────────────
CALL gds.kmeans.stream('combinedGraph', {
  nodeProperty: 'embedding',
  k: 4,
  randomSeed: 42
})
YIELD nodeId, communityId, distanceFromCentroid, silhouette
RETURN gds.util.asNode(nodeId).name    AS name,
       gds.util.asNode(nodeId).node_id AS node_id,
       gds.util.asNode(nodeId).type    AS type,
       communityId,
       round(distanceFromCentroid, 4) AS distanceFromCentroid,
       round(silhouette, 4)           AS silhouette
ORDER BY communityId, name;

// ── Summary: cluster composition ─────────────────────────────────────────────
CALL gds.kmeans.stream('combinedGraph', {
  nodeProperty: 'embedding',
  k: 4,
  randomSeed: 42
})
YIELD nodeId, communityId
RETURN communityId,
       count(nodeId)                AS cluster_size,
       collect(gds.util.asNode(nodeId).name) AS members
ORDER BY communityId;

// ── Write back to nodes ───────────────────────────────────────────────────────
CALL gds.kmeans.write('combinedGraph', {
  nodeProperty: 'embedding',
  k: 4,
  randomSeed: 42,
  writeProperty: 'kmeans_cluster'
})
YIELD nodePropertiesWritten, computeMillis
RETURN nodePropertiesWritten;

// ── Verify ────────────────────────────────────────────────────────────────────
MATCH (n)
WHERE n.kmeans_cluster IS NOT NULL
RETURN n.kmeans_cluster AS cluster, collect(n.name) AS members
ORDER BY cluster;
