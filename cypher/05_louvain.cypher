// 05_louvain.cypher
// Louvain community detection on combinedGraph.

// ── Stream results ────────────────────────────────────────────────────────────
CALL gds.louvain.stream('combinedGraph', {
  relationshipWeightProperty: 'weight'
})
YIELD nodeId, communityId, intermediateCommunityIds
RETURN gds.util.asNode(nodeId).name    AS name,
       gds.util.asNode(nodeId).node_id AS node_id,
       gds.util.asNode(nodeId).type    AS type,
       communityId
ORDER BY communityId, name;

// ── Summary: community sizes ──────────────────────────────────────────────────
CALL gds.louvain.stream('combinedGraph', {
  relationshipWeightProperty: 'weight'
})
YIELD nodeId, communityId
RETURN communityId, count(nodeId) AS community_size
ORDER BY community_size DESC;

// ── Write back to nodes ───────────────────────────────────────────────────────
CALL gds.louvain.write('combinedGraph', {
  relationshipWeightProperty: 'weight',
  writeProperty: 'louvain_community'
})
YIELD communityCount, modularity, modularities
RETURN communityCount, modularity;

// ── Verify ────────────────────────────────────────────────────────────────────
MATCH (n)
WHERE n.louvain_community IS NOT NULL
RETURN n.louvain_community AS community, collect(n.name) AS members
ORDER BY community;
