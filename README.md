# Thai Bank Graph Analysis
## Interest Rate and Risk Sentiment Network: Graph Analysis of Thai SET50 Banking Stocks

**Course:** Social Network & Media Analysis — Midterm Project
**Frequency:** Weekly | **Period:** May 2022 – Latest

---

## Project Objective

Build a financial behavior network linking global interest rates, risk sentiment, FX movement,
and financial sector ETFs to Thai SET50 banking stock returns.
Use Neo4j GDS for graph algorithms (Degree, Betweenness, PageRank, Louvain, FastRP + K-Means).

## Research Question

> How do global interest rates, risk sentiment, FX movement, and financial sector ETFs form a network of influence on Thai SET50 banking stocks?

---

## Variable List

| Variable | Source | Type | Transformation |
|---|---|---|---|
| BBL, KBANK, KKP, KTB, SCB, TISCO, TTB | Yahoo Finance | Thai Bank | log return |
| XLF, EUFN | Yahoo Finance | ETF | log return |
| USDTHB=X | Yahoo Finance | FX | log return |
| ^SET.BK | Yahoo Finance | Index | log return |
| FEDFUNDS, DGS2, DGS10, MORTGAGE30US | FRED | Macro | weekly change |
| VIXCLS | FRED | Macro | log return (VIX_CHANGE) |
| BOT Policy Rate | BOT / manual CSV | Macro | weekly change |
| US Yield Curve | DGS10–DGS2 | Derived | change in spread |
| Thai Bank Basket | Equal-weighted | Derived | mean log return |

---

## Installation

```bash
# 1. Clone / extract project
cd thai-bank-graph-analysis

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env: add FRED_API_KEY, NEO4J_PASSWORD
```

---

## Environment Variables (.env)

```
FRED_API_KEY=your_fred_api_key      # from fred.stlouisfed.org
BOT_API_KEY=                         # optional; leave blank to use manual CSV
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

---

## Data Pipeline (Phase 2)

```bash
# Run all three steps: download → clean → feature engineering
python run_pipeline.py
```

This produces:
- `data/raw/raw_prices.csv` — daily prices from Yahoo Finance
- `data/raw/macro_raw.csv` — FRED + BOT data
- `data/processed/weekly_prices.csv` — weekly resampled prices
- `data/processed/macro_weekly.csv` — weekly macro
- `data/processed/missing_value_summary.csv`
- `data/processed/outlier_flags.csv`
- `data/processed/final_weekly_dataset.csv` — analysis-ready features

**BOT Manual CSV Fallback:**
If the BOT API is unavailable, place a CSV/XLSX file at `data/raw/bot_manual.csv`.
The file must have a date column and columns matching: `bot_policy_rate`, `bibor_1m`,
`thai_gov_2y`, `thai_gov_10y` (any subset is fine).

---

## Statistical Edge Construction (Phase 3)

```bash
python -c "
from src.regime_detection import run_regime_detection
from src.graph_edges import run_graph_edges
run_regime_detection()
run_graph_edges()
"
```

Produces all CSV files in `data/graph/` and `data/results/`.

---

## Neo4j Setup

1. Download and install [Neo4j Desktop](https://neo4j.com/download/) or Neo4j Community.
2. Install the **Graph Data Science (GDS)** plugin from the Neo4j Plugin Manager.
3. Start Neo4j and set your password in `.env`.
4. (Optional) Install the **APOC** plugin for Cypher utility functions.

### Import Graph Data

**Option A — Python (recommended):**
```bash
python src/neo4j_loader.py
```

**Option B — Cypher scripts:**
Copy `data/graph/neo4j_nodes.csv` and `data/graph/neo4j_edges.csv` to Neo4j's import folder,
then run the Cypher scripts in order:
```
cypher/01_create_constraints.cypher
cypher/02_load_nodes_edges.cypher
```

---

## Neo4j GDS Algorithms

```bash
# Run GDS projection + all algorithms
python src/gds_algorithms.py
```

Or run Cypher scripts manually:
```
cypher/03_gds_projection.cypher
cypher/04_centrality.cypher
cypher/05_louvain.cypher
cypher/06_kmeans.cypher
```

Results saved to:
- `data/results/centrality_results.csv`
- `data/results/community_results_louvain.csv`
- `data/results/community_results_kmeans.csv`

---

## Streamlit App

```bash
streamlit run app.py
```

The app runs at `http://localhost:8501` with 9 pages:
1. Project Overview
2. Data Quality & Cleansing
3. Validated Correlation Network
4. Partial Correlation Network
5. Regime-aware Network
6. Lagged Relationship Network
7. Factor Exposure Network
8. Neo4j GDS Results
9. Report Findings

Pages show "file not found" warnings until the corresponding pipeline steps are run.

---

## Report Generation

```bash
python src/report_generator.py
```

Outputs:
- `outputs/midterm_report_draft.md`
- `outputs/midterm_report_draft.docx`

---

## Methodology Summary

| Step | Method | Output |
|---|---|---|
| Correlation edges | Pearson + Spearman + FDR (BH) | `correlation_edges_validated.csv` |
| Partial corr edges | GraphicalLassoCV (LedoitWolf fallback) | `partial_correlation_edges.csv` |
| Lagged edges | corr(factor_{t-k}, bank_t), k=1–4 | `lagged_correlation_edges.csv` |
| Factor exposure | OLS regression per bank | `factor_exposure_edges.csv` |
| Regime detection | Rolling 26-week FEDFUNDS/DGS2 change | `regime_labels.csv` |
| Centrality | Degree, Betweenness, PageRank (GDS) | `centrality_results.csv` |
| Community | Louvain (GDS) | `community_results_louvain.csv` |
| Embedding+cluster | FastRP + K-Means (GDS) | `community_results_kmeans.csv` |

---

## Limitations

1. Edges represent statistical relationships, **not causal links**.
2. Correlation ≠ Causation. Lagged correlation ≠ Granger causality.
3. OLS factor exposure = statistical association, not guaranteed causal impact.
4. Sample starts May 2022 to reduce SCB/SCBX structural break.
5. Weekly data may miss intra-week dynamics.
6. Regime labels based on rolling 26-week window only — no future information used.
7. Partial correlation via GraphicalLasso is sensitive to sample size.
8. Neo4j GDS requires separate Neo4j installation.

---

## File Structure

```
thai-bank-graph-analysis/
├── app.py                        Streamlit app (9 pages)
├── run_pipeline.py               Phase 2 entry point
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── raw/                      Downloaded raw data
│   ├── processed/                Weekly returns + features
│   ├── graph/                    Network edge/node CSVs
│   └── results/                  Centrality, community, regime
├── src/
│   ├── config.py                 All paths, tickers, constants
│   ├── data_download.py          Yahoo + FRED + BOT
│   ├── data_cleaning.py          Dedup, resample, missing, outliers
│   ├── feature_engineering.py    Log returns, rate changes, derived factors
│   ├── statistical_validation.py Pearson + FDR correction
│   ├── partial_correlation.py    GraphicalLasso partial correlation
│   ├── regime_detection.py       Fed regime classification
│   ├── graph_edges.py            Lagged corr + OLS + Neo4j CSVs
│   ├── neo4j_loader.py           Load graph to Neo4j
│   ├── gds_algorithms.py         Run GDS algorithms
│   ├── visualization.py          Plotly + Pyvis chart builders
│   └── report_generator.py       .md + .docx report
├── cypher/                       6 Cypher scripts
└── outputs/                      Report drafts + figures
```
