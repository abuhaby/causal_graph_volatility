# Original User Request

## Initial Request — 2026-09-05T05:11:17Z

# Teamwork Project Prompt — Draft

> Status: Ready for launch — awaiting user approval
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: Full team

Refactor the existing Causal Graph Volatility analysis code into a clean, modular Python package. Fix existing mathematical instability (e.g., 'Rank-deficient design matrix' warnings), and expand the framework to include alternative volatility models.

Working directory: ~/teamwork_projects/causal_graph_volatility
Integrity mode: development

## Verification Resources
You should use the existing output baselines in the original directory as reference for expected behavior:
- `/home/oem/Documents/causal graph volatility/output.log`
- `/home/oem/Documents/causal graph volatility/causal_volatility_framework_OOS.txt`
- The original notebooks and scripts in `/home/oem/Documents/causal graph volatility/`

## Requirements

### R1. Modular Refactoring
Convert the existing Jupyter Notebooks and standalone scripts into a well-structured Python package with distinct modules for data ingestion, feature engineering, causal discovery, and backtesting.

### R2. Bug Fixes & Stability
Identify and fix the 'Rank-deficient design matrix' warnings in the statistical models (statsmodels OLS) without degrading the pipeline's logic.

### R3. Framework Expansion
Introduce new alternative volatility models (e.g., advanced GARCH variations) to the backtesting pipeline.

## Acceptance Criteria

### Verification
- [ ] A programmatic test suite (pytest) is written that executes the core pipeline (ingestion, OLS fits, backtests) and passes successfully.
- [ ] The new package's backtest results closely match or improve upon the historical metrics in the existing `output.log` and OOS logs.
- [ ] The OLS fit process completes without throwing 'Rank-deficient design matrix' warnings.
- [ ] At least one new volatility model is implemented and executable in the backtest framework.
