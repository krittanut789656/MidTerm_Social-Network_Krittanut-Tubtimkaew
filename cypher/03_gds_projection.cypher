// 03_gds_projection.cypher
// Create the GDS in-memory graph projection.
// Run after loading nodes and edges.

// Drop existing projection if it exists
CALL gds.graph.drop('combinedGraph', false) YIELD graphName;

// Project all node types and all relationship types
CALL gds.graph.project(
  'combinedGraph',
  ['Bank', 'ETF', 'MacroFactor', 'FX', 'Index', 'DerivedFactor'],
  {
    CORRELATED_WITH: {
      orientation: 'UNDIRECTED',
      properties: 'weight'
    },
    PARTIAL_CORRELATED_WITH: {
      orientation: 'UNDIRECTED',
      properties: 'weight'
    },
    LAGGED_CORRELATED_WITH: {
      orientation: 'NATURAL',
      properties: 'weight'
    },
    INFLUENCES: {
      orientation: 'NATURAL',
      properties: 'weight'
    }
  }
)
YIELD graphName, nodeCount, relationshipCount, projectMillis
RETURN graphName, nodeCount, relationshipCount, projectMillis;

// Verify projection
CALL gds.graph.list() YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount;
