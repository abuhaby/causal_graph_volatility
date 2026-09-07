# Causal Volatility

An enterprise-grade, modular quantitative finance library for systematic risk, causal discovery, and adaptive volatility management.

## Overview

The `causal_volatility` framework refactors monolithic econometric notebooks into a clean, reproducible Python architecture. It addresses critical numerical challenges in causal risk analysis (such as rank-deficient design matrices in bivariate Granger regressions) and expands the econometric framework to support multiple intraday realized volatility estimators and advanced asymmetric GARCH residualization models.

### Key Capabilities

1. **Systematic Risk Data Ingestion**:
   - Multi-source ingestion for equity market proxies (`OEF`), implied volatility (`^VIX`), and corporate bond credit stress (Moody's `BAA10Y`).
   - Resilient 3-tier fallback hierarchy: authenticated FRED JSON API $\to$ direct FRED CSV streaming $\to$ local deterministic econometric proxy.
   - Comprehensive offline support and fixture caching for deterministic air-gapped test execution.

2. **Realized Volatility Estimators**:
   - `GarmanKlassEstimator`: Extreme-value estimator with analytical zero-clipping preventing negative variance on anomalous gap days.
   - `ParkinsonEstimator`: Classical high-low price range variance estimator.
   - `RogersSatchellEstimator`: Drift-independent extreme-value variance estimator.
   - `YangZhangEstimator`: Minimum-variance unbiased estimator accounting for continuous drift and overnight price jumps.
   - `CloseToCloseEstimator`: Standard return deviation benchmark.

3. **Stationarity & Econometric Diagnostics**:
   - Logarithmic and first-difference transformations with $\epsilon = 10^{-8}$ safety against flat-market zero-variance days.
   - Dual stationarity validation suite: Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS).
   - Serial autocorrelation (Ljung-Box Q-test) and conditional heteroskedasticity (Engle ARCH LM-test) auditing.

4. **Volatility Residualization Engines**:
   - Modular `BaseVolatilityModel` interface.
   - Autoregressive lag optimization via AIC/BIC criteria.
   - Standard GARCH(p, q), Student's t GARCH, Exponential GARCH (EGARCH), and GJR-GARCH.

5. **Structural Causal Discovery & Dynamic Risk Multiplier**:
   - Full-rank bivariate Granger causality testing across multiple lag horizons without collinearity or singular design matrices.
   - Composite causal shock index construction with exponential weighted moving (EWM) smoothing.
   - 252-day rolling percentile mapping to bounded dynamic multiplier series $\lambda_t \in [\lambda_{min}, \lambda_0]$.

6. **High-Performance Backtesting Engine**:
   - Vectorized 1D ratchet trailing stop state machine with calm/stormy regime re-entry dynamics.
   - Full performance analytics: annualized return, annualized volatility, Sharpe ratio, maximum drawdown, and stop-out counts.

## Installation

```bash
# Clone repository
git clone https://github.com/abuhaby/causal_graph_volatility.git
cd causal_graph_volatility

# Editable installation
pip install -e .
```

## Repository Structure

```
causal_graph_volatility/
├── docs/
│   ├── flowcharts/                      # Interactive HTML & workflow designs
│   ├── reports/                         # Full quant diagnostic report
│   └── specs/                           # Project specs, test infra & audit trails
├── figures/
│   ├── eda/                             # Exploratory data analysis (1 to 5)
│   └── results/                         # Stage & backtest diagnostics (6 to 12)
├── notebooks/
│   ├── causal_volatility_framework.ipynb
│   ├── causal_volatility_framework_60_20_20.ipynb
│   ├── causal_volatility_framework_OOS.ipynb
│   ├── causal_volatility_framework_WALKFORWARD.ipynb
│   └── reference_logs/                  # Benchmark logs & baseline metrics
├── scripts/
│   ├── run_causal_volatility_analysis.py # End-to-end quantitative runner
│   └── benchmarks/                      # Legacy benchmarks, unit & timing scripts
├── src/
│   └── causal_volatility/               # Modular quantitative library
│       ├── data/                        # Market ingestion & realized volatility
│       ├── econometrics/                # Stationarity, ARCH/GARCH & Granger causality
│       ├── strategy/                    # Causal multiplier & trailing stop ratchet
│       ├── visualization/               # Publication-grade plotting routines
│       ├── cli.py                       # Command line interface
│       └── pipeline.py                  # End-to-end pipeline orchestrator
├── tests/                               # 4-tier test suite (100 tests)
│   ├── fixtures/
│   ├── tier1_unit/
│   ├── tier2_integration/
│   ├── tier3_regression/
│   └── tier4_e2e/
├── pyproject.toml
└── README.md
```

## Quick Start & Reproducibility

Execute the full quantitative pipeline and generate all 12 publication-grade EDA and stage diagnostic figures:

```bash
# Run complete end-to-end analysis & figure generation
python scripts/run_causal_volatility_analysis.py

# Or via installed CLI entrypoint
causal-volatility --model garch --split oos --offline
```

### Generated Diagnostics & Reports

* **Full Quantitative Report**: [`docs/reports/quant_diagnostic_report.md`](docs/reports/quant_diagnostic_report.md)
* **EDA Graphics (Stage 1-3)** (`figures/eda/`):
  * [`figures/eda/eda_1_correlation_matrix.png`](figures/eda/eda_1_correlation_matrix.png): Cross-asset correlation heatmap across differenced stationary inputs.
  * [`figures/eda/eda_2_return_distribution_qq.png`](figures/eda/eda_2_return_distribution_qq.png): Asset return distribution, excess kurtosis, and normal Q-Q plot.
  * [`figures/eda/eda_3_macro_overlay.png`](figures/eda/eda_3_macro_overlay.png): S&P 100 cumulative wealth overlay against VIX and Moody's Baa credit spreads.
  * [`figures/eda/eda_4_volatility_estimators_comparison.png`](figures/eda/eda_4_volatility_estimators_comparison.png): Multi-estimator realized volatility comparison (Garman-Klass, Parkinson, Rogers-Satchell, Close-to-Close).
  * [`figures/eda/eda_5_stationarity_transformation.png`](figures/eda/eda_5_stationarity_transformation.png): 4-panel ADF/KPSS stationarity transformation verification.
* **Stage Diagnostics & Backtest Results (Stage 4-8)** (`figures/results/`):
  * [`figures/results/res_6_garch_conditional_vol_residuals.png`](figures/results/res_6_garch_conditional_vol_residuals.png): AR-GARCH(1,1) conditional volatility fit and standardized innovation white-noise audits.
  * [`figures/results/res_7_causal_dag_pathways.png`](figures/results/res_7_causal_dag_pathways.png): Discovered Granger causal path coefficients and directed network topology.
  * [`figures/results/res_8_adaptive_multiplier_dynamics.png`](figures/results/res_8_adaptive_multiplier_dynamics.png): Dynamic adaptive multiplier ($\lambda_t$) contraction and composite stress tracking.
  * [`figures/results/res_9_is_equity_curve.png`](figures/results/res_9_is_equity_curve.png): In-Sample (TRAIN 2016–2020) strategy cumulative wealth vs baseline and Buy & Hold.
  * [`figures/results/res_10_oos_equity_curve.png`](figures/results/res_10_oos_equity_curve.png): Out-of-Sample (TEST 2021–2026) strategy cumulative equity curve.
  * [`figures/results/res_11_oos_drawdown.png`](figures/results/res_11_oos_drawdown.png): Out-of-Sample underwater drawdown curves showing maximum drawdown containment.
  * [`figures/results/res_12_regime_reentry_analysis.png`](figures/results/res_12_regime_reentry_analysis.png): Trailing stop ratchet execution, cash preservation intervals, and re-entry events.

## License

MIT License.


