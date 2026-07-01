// 01_create_constraints.cypher
// Run once to set up uniqueness constraints before loading data.

CREATE CONSTRAINT IF NOT EXISTS FOR (n:Bank)          REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:ETF)           REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:MacroFactor)   REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:FX)            REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Index)         REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:DerivedFactor) REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Regime)        REQUIRE n.node_id IS UNIQUE;

// Verify
SHOW CONSTRAINTS;
