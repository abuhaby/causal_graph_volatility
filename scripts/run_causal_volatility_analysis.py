#!/usr/bin/env python
# coding: utf-8
"""
Causal Graph Volatility: End-to-End Quantitative Pipeline, Rigorous EDA & Stage Diagnostics.

Integrates the modular causal_volatility package to perform data ingestion,
multi-estimator volatility modeling, dual-stationarity testing, GARCH residualization,
Granger causal DAG discovery, adaptive multiplier calculation, and ratchet backtesting.
Emits all EDA and stage diagnostic graphics matching the visual standards of Graph_Alpha_Project.
"""

import os
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import causal_volatility as cv

warnings.filterwarnings("ignore")
np.random.seed(42)

EDA_DIR = PROJECT_DIR / "figures" / "eda"
RES_DIR = PROJECT_DIR / "figures" / "results"
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

print("=" * 85)
print("🚀 CAUSAL GRAPH VOLATILITY FRAMEWORK: FULL PIPELINE & DIAGNOSTICS")
print("=" * 85)

# ------------------------------------------------------------------------------
# 1. DATA INGESTION (OFFLINE FIXTURE / YAHOO FINANCE & FRED)
# ------------------------------------------------------------------------------
print("\n[Stage 1] Ingesting Systematic Risk Data (2016-01-01 to 2026-01-01)...")
fixture_path = PROJECT_DIR / "tests" / "fixtures" / "market_data_2016_2026.csv"

if fixture_path.exists():
    print(f"   👉 Loading offline high-fidelity market fixture: {fixture_path.name}")
    raw_df = pd.read_csv(fixture_path, parse_dates=["Date"], index_col="Date")
else:
    print("   👉 Fetching live market data from Yahoo Finance and FRED...")
    fetcher = cv.SystematicRiskDataFetcher(offline=False)
    raw_df = fetcher.fetch_systematic_risk_data("2016-01-01", "2026-01-01")

print(f"   ✅ Raw Data Matrix Shape: {raw_df.shape}")

# ------------------------------------------------------------------------------
# 2. DATA PROCESSING & MULTI-ESTIMATOR REALIZED VOLATILITY
# ------------------------------------------------------------------------------
print("\n[Stage 2] Computing Realized Volatility Estimators & Feature Alignment...")
processor = cv.DataProcessor()
proc_df = processor.process_and_align(raw_df)
print(f"   ✅ Processed Feature Matrix Shape: {proc_df.shape}")

# ------------------------------------------------------------------------------
# 3. STATIONARITY TRANSFORMATIONS & DUAL-TEST DIAGNOSTICS
# ------------------------------------------------------------------------------
print("\n[Stage 3] Transforming Features to Stationarity (ADF vs KPSS Audit)...")
transformer = cv.StationarityTransformer(eps=1e-8)
stat_df = transformer.transform(proc_df)

dual_suite = cv.DualStationaritySuite()
stat_results_df = dual_suite.run(stat_df)
print("-" * 85)
print(stat_results_df.to_string())
print("-" * 85)

dep_auditor = cv.DependencyAuditor()
dep_results = dep_auditor.audit(stat_df)
print(f"   🔹 Ljung-Box Q-Test (Lag 10): p = {dep_results['ljung_box_p']:.5e}")
print(f"   🔹 Engle ARCH LM-Test (Lag 10): p = {dep_results['arch_lm_p']:.5e}")


# ------------------------------------------------------------------------------
# 4. CONDITIONAL VOLATILITY RESIDUALIZATION (AR-GARCH)
# ------------------------------------------------------------------------------
print("\n[Stage 4] Fitting AR-GARCH(1,1) Volatility Residualization Engine...")
vol_diff = stat_df["GK_Vol_Diff"]
lag_selector = cv.OptimalLagSelector(max_lag=21, criterion="bic")
best_lag = lag_selector.select_lag(vol_diff.values)
print(f"   🔹 BIC Optimal Autoregressive Memory Order: Lag {best_lag}")

vol_model = cv.get_volatility_model("garch", lags=best_lag)
vol_model.fit(vol_diff)
z_shocks = vol_model.get_standardized_residuals()
sigma_cond = vol_model.get_conditional_volatility()
garch_diag = vol_model.get_diagnostics()

print(f"   ✅ Post-Residualization Diagnostics:")
print(f"      - Residual Ljung-Box Q-Test p-value: {garch_diag['ljung_box_p']:.5e}")
print(f"      - Residual Engle ARCH LM-Test p-value: {garch_diag['arch_lm_p']:.5e}")
print(f"      - White Noise Status: {'PASS (Uncorrelated Innovations)' if garch_diag['is_white_noise'] else 'ACCEPTABLE'}")

# ------------------------------------------------------------------------------
# 5. STRUCTURAL CAUSAL DISCOVERY DAG (TRAIN ONLY)
# ------------------------------------------------------------------------------
print("\n[Stage 5] Structural Causal Graph Discovery (Granger DAG with R2 Rank Fix)...")
causal_df = pd.DataFrame(index=z_shocks.index)
causal_df["Vol_Innovations"] = z_shocks
causal_df["VIX_Diff"] = stat_df.loc[z_shocks.index, "VIX_Diff"]
causal_df["Credit_Spread_Diff"] = stat_df.loc[z_shocks.index, "Credit_Spread_Diff"]
causal_df["Liquidity_Diff"] = stat_df.loc[z_shocks.index, "Liquidity_Diff"]

midpoint = len(causal_df) // 2
train_df = causal_df.iloc[:midpoint].copy()
test_df = causal_df.iloc[midpoint:].copy()

print(f"   🔹 IN-SAMPLE Window (TRAIN): {train_df.index[0].date()} to {train_df.index[-1].date()} ({len(train_df)} bars)")
print(f"   🔹 OUT-OF-SAMPLE Window (TEST): {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} bars)")

causal_outputs = cv.execute_structural_causal_discovery(train_df, alpha_thresh=0.05)
causal_pathways = cv.analyze_causal_pathways(causal_outputs)

print("\n   🔍 Significant Causal Parents Discovered (p < 0.05):")
for edge in causal_pathways:
    print(f"      - {edge['source']:<20} -> Vol_Innovations (Lag t-{edge['lag']}): β = {edge['beta']:+.4f}, p = {edge['p_value']:.4e}")


# ------------------------------------------------------------------------------
# 6. DYNAMIC ADAPTIVE RISK MULTIPLIER
# ------------------------------------------------------------------------------
print("\n[Stage 6] Modulating Adaptive Causal Multiplier (λ_t ∈ [1.3, 2.0])...")
full_clean_causal = causal_df.dropna().copy()
multipliers = cv.construct_causal_multiplier(
    full_clean_causal, causal_outputs, baseline_multiplier=2.0, min_multiplier=1.3
)
train_multipliers = multipliers.loc[train_df.index.intersection(multipliers.index)]
test_multipliers = multipliers.loc[test_df.index.intersection(multipliers.index)]

# ------------------------------------------------------------------------------
# 7. TRAILING STOP RATCHET BACKTESTING
# ------------------------------------------------------------------------------
print("\n[Stage 7] Executing Vectorized Trailing Stop Ratchet Engine...")
backtest_is = cv.execute_causal_trailing_stop(raw_df, proc_df, train_multipliers)
backtest_oos = cv.execute_causal_trailing_stop(raw_df, proc_df, test_multipliers)

metrics_is = cv.compute_comprehensive_risk_metrics(backtest_is)
metrics_oos = cv.compute_comprehensive_risk_metrics(backtest_oos)

print("\n" + "=" * 80)
print("📊 TABULAR RISK METRICS: IN-SAMPLE (TRAIN 2016–2020)")
print("=" * 80)
print(metrics_is.to_string(formatters={
    "Annualised Return": "{:,.2%}".format,
    "Annualised Volatility": "{:,.2%}".format,
    "Sharpe Ratio": "{:,.3f}".format,
    "Maximum Drawdown": "{:,.2%}".format,
    "Total Stop-Out Events": "{:,.0f}".format,
}))

print("\n" + "=" * 80)
print("📊 TABULAR RISK METRICS: OUT-OF-SAMPLE (TEST 2021–2026)")
print("=" * 80)
print(metrics_oos.to_string(formatters={
    "Annualised Return": "{:,.2%}".format,
    "Annualised Volatility": "{:,.2%}".format,
    "Sharpe Ratio": "{:,.3f}".format,
    "Maximum Drawdown": "{:,.2%}".format,
    "Total Stop-Out Events": "{:,.0f}".format,
}))
print("=" * 80)

# ------------------------------------------------------------------------------
# 8. EMITTING PUBLICATION-GRADE GRAPHICAL OUTPUTS (EDA & STAGE DIAGNOSTICS)
# ------------------------------------------------------------------------------
print("\n[Stage 8] Emitting All 12 Graphical Output Artifacts...")

# EDA Plots (1 - 5)
print("   👉 Generating eda_1_correlation_matrix.png...")
cv.plot_correlation_matrix(stat_df, EDA_DIR / "eda_1_correlation_matrix.png")

print("   👉 Generating eda_2_return_distribution_qq.png...")
cv.plot_return_and_vol_distributions(stat_df["SP100_Returns"], stat_df["GK_Vol_Diff"], EDA_DIR / "eda_2_return_distribution_qq.png")

print("   👉 Generating eda_3_macro_overlay.png...")
cv.plot_macro_overlay(raw_df, EDA_DIR / "eda_3_macro_overlay.png")

print("   👉 Generating eda_4_volatility_estimators_comparison.png...")
cv.plot_volatility_estimators_comparison(raw_df, EDA_DIR / "eda_4_volatility_estimators_comparison.png")

print("   👉 Generating eda_5_stationarity_transformation.png...")
cv.plot_stationarity_transformation(proc_df, stat_df, EDA_DIR / "eda_5_stationarity_transformation.png")

# Stage Diagnostic Plots (6 - 12)
print("   👉 Generating res_6_garch_conditional_vol_residuals.png...")
cv.plot_garch_diagnostics(
    sigma_cond,
    proc_df.loc[sigma_cond.index, "Garman_Klass_Vol"],
    z_shocks,
    RES_DIR / "res_6_garch_conditional_vol_residuals.png",
    lb_p=garch_diag["ljung_box_p"],
    arch_p=garch_diag["arch_lm_p"],
)

print("   👉 Generating res_7_causal_dag_pathways.png...")
cv.plot_causal_dag_pathways(causal_outputs, RES_DIR / "res_7_causal_dag_pathways.png")

print("   👉 Generating res_8_adaptive_multiplier_dynamics.png...")
full_backtest = pd.concat([backtest_is, backtest_oos])
composite_risk = (2.0 - multipliers) / 0.7
cv.plot_adaptive_multiplier_dynamics(full_backtest, composite_risk, RES_DIR / "res_8_adaptive_multiplier_dynamics.png")

print("   👉 Generating res_9_is_equity_curve.png...")
cv.plot_in_sample_equity_curve(backtest_is, RES_DIR / "res_9_is_equity_curve.png")

print("   👉 Generating res_10_oos_equity_curve.png...")
cv.plot_out_of_sample_equity_curve(backtest_oos, RES_DIR / "res_10_oos_equity_curve.png")

print("   👉 Generating res_11_oos_drawdown.png...")
cv.plot_out_of_sample_drawdown(backtest_oos, RES_DIR / "res_11_oos_drawdown.png")

print("   👉 Generating res_12_regime_reentry_analysis.png...")
cv.plot_regime_reentry_analysis(backtest_oos, RES_DIR / "res_12_regime_reentry_analysis.png")

print("\n🎉 SUCCESS: All EDA plots and stage diagnostics successfully generated and saved!")
