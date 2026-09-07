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
├── docs/                                    # Documentation, specifications & research reports
│   ├── flowcharts/                          # Pipeline architecture & visual schematics
│   │   └── project-design.html              # Standalone interactive flowchart & workflow design
│   ├── reports/                             # Quantitative research & diagnostic audits
│   │   └── quant_diagnostic_report.md       # Comprehensive econometric, stationarity & backtest report
│   └── specs/                               # Engineering specifications & verification trails
│       ├── ORIGINAL_REQUEST.md              # Original project requirements & scope contract
│       ├── PROJECT.md                       # Architectural design & implementation roadmap
│       ├── TEST_INFRA.md                    # 4-tier testing infrastructure documentation
│       └── TEST_READY.md                    # Pre-flight checklist & test readiness sign-off
│
├── figures/                                 # Emitted publication-grade graphical outputs
│   ├── eda/                                 # Exploratory Data Analysis (Stage 1-3)
│   │   ├── eda_1_correlation_matrix.png     # Cross-asset correlation heatmap across differenced inputs
│   │   ├── eda_2_return_distribution_qq.png # Empirical return distribution, excess kurtosis & normal Q-Q
│   │   ├── eda_3_macro_overlay.png          # S&P 100 cumulative wealth vs VIX & Moody's Baa credit spreads
│   │   ├── eda_4_volatility_estimators_comparison.png # Multi-estimator realized volatility benchmark
│   │   └── eda_5_stationarity_transformation.png      # 4-panel ADF & KPSS stationarity verification
│   └── results/                             # Stage Diagnostics & Backtest Results (Stage 4-8)
│       ├── res_6_garch_conditional_vol_residuals.png  # AR-GARCH(1,1) volatility fit & innovation whitening
│       ├── res_7_causal_dag_pathways.png              # Discovered Granger causal network & path coefficients
│       ├── res_8_adaptive_multiplier_dynamics.png     # Dynamic adaptive multiplier (λ_t) & composite stress
│       ├── res_9_is_equity_curve.png                  # In-Sample (TRAIN 2016–2020) strategy cumulative wealth
│       ├── res_10_oos_equity_curve.png                # Out-of-Sample (TEST 2021–2026) cumulative equity curve
│       ├── res_11_oos_drawdown.png                    # Out-of-Sample underwater drawdown curves
│       └── res_12_regime_reentry_analysis.png         # Trailing stop ratchet state transitions & cash regimes
│
├── notebooks/                               # Research notebooks & legacy exploration
│   ├── causal_volatility_framework.ipynb    # Monolithic research notebook baseline
│   ├── causal_volatility_framework_60_20_20.ipynb # 60/20/20 train/validation/test split experiment
│   ├── causal_volatility_framework_OOS.ipynb      # Out-of-sample partition analysis
│   ├── causal_volatility_framework_WALKFORWARD.ipynb # Walk-forward rolling evaluation
│   └── reference_logs/                      # Historical execution logs & validation baselines
│       ├── causal_volatility_framework_OOS.txt      # Text transcript of legacy OOS run
│       ├── causal_volatility_framework_OOS.txt.bak  # Legacy transcript backup
│       └── output.log                       # Baseline numerical metrics log for regression tests
│
├── scripts/                                 # Pipeline execution & benchmark drivers
│   ├── run_causal_volatility_analysis.py    # Master runner: data ingestion, econometrics, backtest & plots
│   └── benchmarks/                          # Diagnostic micro-benchmarks & verification scripts
│       ├── cell1.py                         # Single cell diagnostic runner
│       ├── extract.py                       # Notebook extraction utility
│       ├── fix_date_col.py                  # Datetime index parsing patch
│       ├── fix_timeouts.py                  # Network timeout handler
│       ├── run_all_timed.py                 # Benchmarking execution timer
│       ├── run_garch_pq.py                  # GARCH(p,q) grid search benchmark
│       ├── run_garch_t.py                   # Student's t GARCH evaluation script
│       ├── run_sec1.py                      # Stage 1 execution driver
│       ├── run_sec12.py                     # Stages 1–2 execution driver
│       ├── run_sec123.py                    # Stages 1–3 execution driver
│       ├── scratch.py                       # Ad-hoc experimental scratchpad
│       ├── temp.py                          # Temporary execution buffer
│       ├── temp_search.py                   # Parameter search scratch script
│       ├── temp_source.py                   # Legacy source snapshot
│       ├── test_garch_pq.py                 # Ad-hoc GARCH order test
│       ├── test_garch_t.py                  # Ad-hoc Student's t test
│       ├── test_perf.py                     # Performance timing harness
│       ├── test_sec3.py                     # Ad-hoc stationarity test
│       ├── test_yf.py                       # Yahoo Finance connectivity test
│       ├── time_loop.py                     # Profiling execution timer
│       ├── time_test.py                     # Step-by-step latency check
│       └── verify_m1_adversarial.py         # M1 adversarial fixture verification
│
├── src/                                     # Core production Python package
│   └── causal_volatility/                   # Package namespace
│       ├── backtest/                        # Trailing stop ratchet & performance analytics
│       │   ├── engine.py                    # Vectorized 1D ratchet state machine (calm/stormy regimes)
│       │   ├── metrics.py                   # Annualized return, Sharpe ratio, MaxDD, and stop-out counts
│       │   └── validation.py                # 5-fold walk-forward validation without lookahead leakage
│       ├── causal/                          # Structural causal graph discovery & multiplier
│       │   ├── discovery.py                 # Full-rank bivariate Granger causality testing (R2 fix)
│       │   ├── multiplier.py                # Adaptive multiplier (λ_t ∈ [1.3, 2.0]) via rolling percentile
│       │   └── pathways.py                  # Directed causal DAG network analysis & filtering
│       ├── data/                            # Market data acquisition & alignment
│       │   ├── fetcher.py                   # 3-tier fallback data loader (FRED JSON -> CSV -> Synthetic)
│       │   ├── processor.py                 # Intraday feature alignment & liquidity proxy computation
│       │   └── storage.py                   # Local fixture caching & offline serialization
│       ├── estimators/                      # Intraday realized volatility estimators
│       │   ├── base.py                      # BaseRealizedVolatilityEstimator abstract contract
│       │   ├── close_to_close.py            # Standard deviation return variance benchmark
│       │   ├── garman_klass.py              # Garman-Klass extreme-value estimator with zero-clipping
│       │   ├── parkinson.py                 # Parkinson high-low range estimator
│       │   ├── rogers_satchell.py           # Rogers-Satchell drift-independent estimator
│       │   └── yang_zhang.py                # Yang-Zhang minimum-variance jump/drift estimator
│       ├── models/                          # Volatility residualization engines
│       │   ├── base.py                      # BaseVolatilityModel interface definition
│       │   ├── egarch.py                    # Exponential GARCH (asymmetric leverage modeling)
│       │   ├── factory.py                   # Dynamic model factory (garch, egarch, gjr, student_t)
│       │   ├── garch.py                     # Standard AR-GARCH(1,1) Gaussian volatility filter
│       │   ├── gjr_garch.py                 # Glosten-Jagannathan-Runkle GARCH
│       │   └── selection.py                 # Autoregressive memory order lag selection (AIC/BIC)
│       ├── stationarity/                    # Covariance stationarity validation
│       │   ├── diagnostics.py               # Dual suite: Augmented Dickey-Fuller (ADF) + KPSS
│       │   └── transform.py                 # ε-safe logarithmic & first-difference transformation
│       ├── visualization/                   # Publication-grade plotting modules
│       │   ├── eda.py                       # Seaborn/Matplotlib routines for EDA stages 1–5
│       │   └── stage_diagnostics.py         # Diagnostic plots for stages 6–12 (equity, DAG, drawdown)
│       ├── cli.py                           # Production Command-Line Interface (`causal-volatility`)
│       ├── config.py                        # Centralized pipeline hyperparameters & defaults
│       └── pipeline.py                      # Unified CausalVolatilityPipeline orchestrator
│
├── tests/                                   # Exhaustive 4-tier test suite (100 passing tests)
│   ├── conftest.py                          # Shared pytest fixtures & test configuration
│   ├── fixtures/                            # Deterministic air-gapped test datasets
│   │   └── market_data_2016_2026.csv        # 10-year multi-asset market fixture
│   ├── tier1_unit/                          # Tier 1: Mathematical invariants & analytical formulas
│   ├── tier2_integration/                   # Tier 2: Component integration & contract validation
│   ├── tier3_regression/                    # Tier 3: Numerical parity against historical logs
│   └── tier4_e2e/                            # Tier 4: Full pipeline execution & CLI validation
│
├── pyproject.toml                           # PEP 517/518 build metadata, dependencies & CLI entrypoints
└── README.md                                # Project overview, architecture & empirical results
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


