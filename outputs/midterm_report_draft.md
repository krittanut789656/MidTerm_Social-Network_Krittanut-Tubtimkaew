# Interest Rate and Risk Sentiment Network:
# Graph Analysis of Thai SET50 Banking Stocks
**การวิเคราะห์เครือข่ายผลกระทบของดอกเบี้ยโลก ค่าเงิน และความเสี่ยงตลาด
ต่อพฤติกรรมราคาหุ้นธนาคารไทยใน SET50**

---

## 1. Research Question

> How do global interest rates, risk sentiment, FX movement, and financial sector ETFs
> form a network of influence on Thai SET50 banking stocks?

---

## 2. Dataset and Variables

| Category | Variables |
|---|---|
| Thai Banking Stocks | BBL, KBANK, KKP, KTB, SCB, TISCO, TTB |
| Global ETFs | XLF (US Financial), EUFN (European Financial) |
| FX | USDTHB |
| Index | SET Index |
| FRED Macro | FEDFUNDS, DGS2, DGS10, MORTGAGE30US, VIX |
| BOT | BOT Policy Rate |
| Derived | US Yield Curve (DGS10-DGS2), Thai Bank Basket |

**Period:** 2022-05-13 to 2026-06-26
**Frequency:** Weekly
**Observations:** 216 weeks
**Note:** Sample starts May 2022 to avoid SCB/SCBX structural continuity issues.

---

## 3. Data Cleaning

- Removed duplicate dates and columns.
- Resampled all series to weekly (Friday/last available).
- Forward-filled macro variables up to 1 week; never forward-filled stock prices.
- Converted prices/FX/indices to weekly log returns: log(P_t / P_(t-1)).
- Converted rates/yields to weekly first differences: rate_t minus rate_(t-1).
- Flagged outliers by |z-score| > 3.5 (not removed).
- Verified SCB.BK price continuity from May 2022.

---

## 4. Graph Schema

**Node Types:** Bank, ETF, MacroFactor, FX, Index, DerivedFactor

**Relationship Types:**

| Relationship | Method | Threshold |
|---|---|---|
| CORRELATED_WITH | Pearson + FDR (BH) | p_adj < 0.05, |r| >= 0.20 |
| PARTIAL_CORRELATED_WITH | GraphicalLassoCV | |pc| >= 0.15 |
| LAGGED_CORRELATED_WITH | Lagged Pearson (k=1-4) | |r| >= 0.30 |
| INFLUENCES | OLS regression | p <= 0.05 |

---

## 5. Graph Analysis Results

### 5.1 Centrality Results

| Metric | Top Node | Score |
|---|---|---|
| Degree Centrality | USDTHB (0.7647) | Most connections |
| Betweenness Centrality | USDTHB (0.5147) | Bridge node |
| PageRank | Thai Bank Basket (0.0920) | Influence score |

**Interpretation:** The node with highest degree centrality is most broadly associated with
price movements across the financial network. The betweenness node acts as the primary
bridge between global macro conditions and Thai banking stock behavior.

### 5.2 Community Detection

**Louvain Communities:** 4 communities detected.

| Community | Members |
|---|---|
| 0 | EUFN, VIX, XLF |
| 1 | BBL, KBANK, KKP, KTB, SCB, SET Index, TISCO, TTB, Thai Bank Basket |
| 2 | US 10Y Yield, US 2Y Yield, US 30Y Mortgage, US Yield Curve, USDTHB |
| 3 | Fed Funds Rate |

**Interpretation:** Thai banking stocks and global macro/financial variables cluster into
communities. Variables within the same community exhibit stronger co-movement patterns.

### 5.3 Raw vs Partial Correlation

| Network | Edge Count |
|---|---|
| Raw Validated Correlation | 60 |
| Partial Correlation | 14 |
| Edge Survival Rate | 23.3% |

**Interpretation:** After conditioning on all other variables, 23.3% of raw correlation
edges survive. Edges that disappear were driven by common factors. Surviving edges
represent more direct associations between pairs of variables.

### 5.4 Factor Exposure (OLS)

**Most Globally Sensitive Bank:** KBANK_ret (2 significant factors)

OLS model: Bank_Return ~ XLF + EUFN + SET + USDTHB + VIX_CHANGE + DGS2_chg + DGS10_chg
           + US_YIELD_CURVE_chg + MORTGAGE30US_chg + BOT_RATE_CHANGE

Significant edges (p <= 0.05) retained as INFLUENCES relationships.

### 5.5 Regime-Aware Network

| regime | n_weeks | validated_edges | partial_edges | edge_survival_pct | top_corr_pair |
|---|---|---|---|---|---|
| Hiking_or_Restrictive | 83 | 58 | 14 | 24.1 | DGS2_chg ↔ DGS10_chg (0.832) |
| Pausing_or_Cutting | 38 | 50 | 15 | 30.0 | BBL_ret ↔ THAI_BANK_BASKET_ret (0.872) |

**Regime Classification:** Rolling 26-week change in FEDFUNDS and DGS2.
'Hiking or Restrictive' if either rate is rising over the past 26 weeks.
No look-ahead bias applied.

---

## 6. Key Findings

1. **Most central node:** USDTHB (0.7647) — most broadly connected variable.
2. **Bridge variable:** USDTHB (0.5147) — connects clusters, transmits macro shocks.
3. **Most sensitive bank:** KBANK_ret (2 significant factors)
4. **Community structure:** 4 distinct communities (Louvain).
5. **Edge survival:** Only 23.3% of raw correlation edges survive partial adjustment.
6. **Regime effect:** Network structure differs between Fed hiking and cutting regimes.
7. **Risk transmission:** VIX and USDTHB linked to Thai bank returns (potential channels).
8. **Methodological note:** All findings are association-based (ex-post). No causal claims.

---

## 7. Limitations

1. Financial behavior network derived from time series — not a traditional social network.
2. Edges represent statistical relationships, not causal links.
3. Correlation does not imply causation.
4. OLS with correlated regressors (DGS2, DGS10, US_YIELD_CURVE) may have multicollinearity.
5. Weekly data reduces noise but may miss intra-week dynamics.
6. Neo4j GDS results require Neo4j running with GDS plugin installed.
7. Sample size (~216 weeks) is sufficient but limited for high-dimensional partial correlation.

---

## 8. Conclusion

This study constructs a financial behavior network linking global interest rates, risk sentiment,
FX movement, and financial sector ETFs to Thai SET50 banking stock returns.

Graph analysis reveals a structured network with distinct communities, clear bridge variables,
and measurable differences in network topology between Fed tightening and easing regimes.

The partial correlation adjustment substantially reduces the raw edge count, demonstrating
that many apparent correlations are driven by common global factors rather than direct pairwise
relationships.

---
*Generated by Thai Bank Graph Analysis Pipeline — Social Network & Media Analysis, Midterm Project*
