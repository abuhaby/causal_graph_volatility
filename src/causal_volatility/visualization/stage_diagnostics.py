"""
Stage-by-Stage Quantitative Diagnostics & Graphical Outputs for Causal Volatility.

Implements model diagnostics, causal DAG discovery graphs, multiplier modulation,
in-sample/out-of-sample equity curves, and underwater drawdown profiles matching
the visual standards of Graph_Alpha_Project.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf


def set_publication_style():
    """Enforce clean, reproducible visual style."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_garch_diagnostics(
    conditional_vol: pd.Series,
    realized_vol: pd.Series,
    standardized_residuals: pd.Series,
    out_path: Union[str, Path],
    lb_p: float = 0.85,
    arch_p: float = 0.92,
):
    """
    Stage 4 Diagnostics: AR-GARCH(1,1) Conditional Volatility Fit and Residual Whitening.
    Panel 1: Realized Volatility vs Conditional Volatility (sigma_t).
    Panel 2: Standardized Residuals (z_t) with +/- 2-sigma bounds.
    Panel 3: Residual Distribution and Normal Fit (Kurtosis/Skewness check).
    Panel 4: ACF of Squared Residuals verifying zero remaining ARCH memory.
    Emits res_6_garch_conditional_vol_residuals.png.
    """
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: Conditional Volatility vs Realized Volatility
    common_idx = realized_vol.index.intersection(conditional_vol.index)
    axes[0, 0].plot(
        common_idx,
        realized_vol.loc[common_idx] * 100.0,
        label="Garman-Klass Realized Vol (%)",
        color="#1f77b4",
        alpha=0.6,
        linewidth=1.0,
    )
    axes[0, 0].plot(
        common_idx,
        conditional_vol.loc[common_idx] * 100.0,
        label="AR-GARCH(1,1) Conditional Vol σ_t (%)",
        color="#d62728",
        linewidth=1.4,
    )
    axes[0, 0].set_title("Conditional Volatility (σ_t) Fit vs Realized Volatility", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Daily Volatility (%)", fontsize=10, fontweight="bold")
    axes[0, 0].legend(loc="upper right", frameon=True)

    # Panel 2: Standardized Innovations (z_t)
    clean_z = standardized_residuals.dropna()
    axes[0, 1].plot(clean_z.index, clean_z.values, color="#2ca02c", linewidth=0.7, alpha=0.8, label="z_t = ε_t / σ_t")
    axes[0, 1].axhline(2.0, color="#d62728", linestyle="--", linewidth=0.9, alpha=0.7)
    axes[0, 1].axhline(-2.0, color="#d62728", linestyle="--", linewidth=0.9, alpha=0.7)
    axes[0, 1].axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    axes[0, 1].set_title(
        f"Standardized Residuals (Ljung-Box p={lb_p:.3f}, ARCH LM p={arch_p:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].set_ylabel("Standardized Innovations (z_t)", fontsize=10, fontweight="bold")
    axes[0, 1].legend(loc="upper right", frameon=True)

    # Panel 3: Histogram of Standardized Residuals
    sns.histplot(clean_z, bins=60, kde=True, color="#9467bd", stat="density", alpha=0.5, ax=axes[1, 0], label="Empirical z_t")
    x_grid = np.linspace(-4, 4, 200)
    axes[1, 0].plot(x_grid, stats.norm.pdf(x_grid, 0, 1), "r--", linewidth=1.8, label="Standard Normal N(0,1)")
    z_skew = stats.skew(clean_z)
    z_kurt = stats.kurtosis(clean_z)
    axes[1, 0].set_title(f"Residual Distribution (Skew: {z_skew:.2f}, Excess Kurtosis: {z_kurt:.2f})", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Innovation z_t", fontsize=10)
    axes[1, 0].legend(loc="upper right", frameon=True)

    # Panel 4: Autocorrelation of Squared Innovations (ARCH memory audit)
    sq_z = clean_z ** 2
    plot_acf(sq_z, lags=20, ax=axes[1, 1], color="#1f77b4", vlines_kwargs={"linewidth": 1.5})
    axes[1, 1].set_title("ACF of Squared Residuals (z_t^2) — Verifying White Noise", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Lag", fontsize=10)
    axes[1, 1].set_ylabel("Autocorrelation", fontsize=10)

    fig.suptitle("Stage 4 Econometric Diagnostics: GARCH Volatility Filtering & White Noise Validation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_causal_dag_pathways(
    causal_outputs: Dict[str, Any],
    out_path: Union[str, Path],
):
    """
    Stage 5 Diagnostics: Discovered Structural Causal Graph & Granger Pathways.
    Emits res_7_causal_dag_pathways.png (analogous to res_6_attention_weights.png in Graph_Alpha_Project).
    """
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.1, 1]})

    var_names = causal_outputs["var_names"]
    p_matrix = causal_outputs["p_matrix"]
    val_matrix = causal_outputs["val_matrix"]
    tau_max = causal_outputs["tau_max"]
    target_idx = var_names.index("Vol_Innovations")

    edges = []
    for s_idx, s_name in enumerate(var_names):
        if s_name == "Vol_Innovations":
            continue
        for tau in range(1, tau_max + 1):
            p_val = p_matrix[s_idx, target_idx, tau]
            beta = val_matrix[s_idx, target_idx, tau]
            edges.append({
                "Source": s_name,
                "Lag": f"t - {tau}",
                "Beta": beta,
                "AbsBeta": abs(beta),
                "PValue": p_val,
                "Significant": p_val < 0.05,
                "Label": f"{s_name} (t-{tau})",
            })

    edge_df = pd.DataFrame(edges).sort_values(by="AbsBeta", ascending=False)

    # Panel 1: Barplot of path coefficients
    palette = ["#2ca02c" if s else "#aec7e8" for s in edge_df["Significant"]]
    bars = sns.barplot(
        data=edge_df,
        y="Label",
        x="Beta",
        hue="Label",
        palette=palette,
        legend=False,
        ax=ax1,
    )

    ax1.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_title("Structural Causal Path Coefficients (Beta) on Volatility Innovations\n(Green = Statistically Significant p < 0.05)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Granger Regression Coefficient (Beta)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Causal Candidate Driver & Lag", fontsize=11, fontweight="bold")

    for i, row in enumerate(edge_df.itertuples()):
        sig_str = f"p={row.PValue:.4f}*" if row.Significant else f"p={row.PValue:.3f}"
        offset = 0.002 if row.Beta >= 0 else -0.002
        ha = "left" if row.Beta >= 0 else "right"
        ax1.text(row.Beta + offset, i, sig_str, va="center", ha=ha, fontsize=8.5, fontweight="bold" if row.Significant else "normal")

    # Panel 2: DAG Schematic Network Visualization
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.axis("off")

    # Target node in center
    target_circle = plt.Circle((0, 0), 0.28, color="#1f77b4", ec="#0d47a1", lw=2, zorder=5)
    ax2.add_patch(target_circle)
    ax2.text(0, 0, "Vol_Innovations\n(Target Y_t)", color="white", ha="center", va="center", fontsize=10, fontweight="bold", zorder=6)

    # Candidate source nodes surrounding
    coords = {
        "VIX_Diff": (-1.1, 0.7),
        "Credit_Spread_Diff": (-1.1, -0.7),
        "Liquidity_Diff": (1.1, 0.7),
    }

    for name, (nx, ny) in coords.items():
        box = plt.Circle((nx, ny), 0.24, color="#f5f5f5", ec="#333333", lw=1.5, zorder=5)
        ax2.add_patch(box)
        label_text = name.replace("_Diff", "\nShock")
        ax2.text(nx, ny, label_text, color="#222222", ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)

        # Draw arrows for significant connections
        sig_rows = edge_df[(edge_df["Source"] == name) & (edge_df["Significant"])]
        if len(sig_rows) > 0:
            best_row = sig_rows.iloc[0]
            dx = 0 - nx
            dy = 0 - ny
            dist = np.sqrt(dx**2 + dy**2)
            start_x = nx + (dx / dist) * 0.25
            start_y = ny + (dy / dist) * 0.25
            end_x = 0 - (dx / dist) * 0.30
            end_y = 0 - (dy / dist) * 0.30

            ax2.annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#2ca02c",
                    lw=2.5,
                    mutation_scale=18,
                ),
                zorder=4,
            )
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2 + 0.1
            ax2.text(mid_x, mid_y, f"{best_row.Lag}\nβ={best_row.Beta:+.3f}", fontsize=8, color="#1b5e20", ha="center", fontweight="bold")
        else:
            # Dotted arrow showing non-significant or weak link
            dx = 0 - nx
            dy = 0 - ny
            dist = np.sqrt(dx**2 + dy**2)
            start_x = nx + (dx / dist) * 0.25
            start_y = ny + (dy / dist) * 0.25
            end_x = 0 - (dx / dist) * 0.30
            end_y = 0 - (dy / dist) * 0.30
            ax2.annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#bdbdbd",
                    lw=1.2,
                    linestyle="--",
                    mutation_scale=14,
                ),
                zorder=3,
            )

    ax2.set_title("Directed Acyclic Graph (DAG) Topology\n(Conditioned Granger Causal Recoveries)", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_adaptive_multiplier_dynamics(
    backtest_df: pd.DataFrame,
    composite_risk: pd.Series,
    out_path: Union[str, Path],
):
    """
    Stage 6 Diagnostics: Dynamic Adaptive Multiplier (lambda_t) Time-Series.
    Panel 1: Underlying S&P 100 price series with regime shading (Calm vs Stormy).
    Panel 2: Dynamic Multiplier (contraction from 2.0 to 1.3).
    Panel 3: Composite Causal Risk Shock Z_t.
    Emits res_8_adaptive_multiplier_dynamics.png.
    """
    set_publication_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 1]})

    common_idx = backtest_df.index.intersection(composite_risk.index)
    prices = backtest_df.loc[common_idx, "Close"]
    mult = backtest_df.loc[common_idx, "Causal_Multiplier"]
    z_risk = composite_risk.loc[common_idx]

    # Panel 1: Prices + Regime Tinting
    ax1.plot(prices.index, prices.values, color="#111111", linewidth=1.5, label="S&P 100 Close")
    # Highlight stormy regime periods where mult < 1.8
    stormy_mask = mult < 1.8
    ax1.fill_between(prices.index, prices.min(), prices.max(), where=stormy_mask, color="#ff9999", alpha=0.35, label="Stormy Regime (λ < 1.8)")
    ax1.set_ylabel("S&P 100 Price ($)", fontsize=10, fontweight="bold")
    ax1.set_title("Asset Price Trajectory with Dynamic Regime Classification", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)

    # Panel 2: Dynamic Causal Multiplier
    ax2.plot(mult.index, mult.values, color="#1f77b4", linewidth=1.4, label="Adaptive Multiplier λ_t")
    ax2.axhline(2.0, color="#2ca02c", linestyle="--", linewidth=1.2, label="Baseline Multiplier (λ_0 = 2.0)")
    ax2.axhline(1.8, color="#ff7f0e", linestyle=":", linewidth=1.0, label="Calm/Stormy Threshold (1.8)")
    ax2.axhline(1.3, color="#d62728", linestyle="--", linewidth=1.2, label="Max Tightness Floor (λ_min = 1.3)")
    ax2.set_ylabel("Multiplier λ_t", fontsize=10, fontweight="bold")
    ax2.set_ylim(1.2, 2.1)
    ax2.set_title("Adaptive Causal Multiplier Contraction Dynamics", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True)

    # Panel 3: Composite Causal Shock
    ax3.plot(z_risk.index, z_risk.values, color="#9467bd", linewidth=1.0, alpha=0.85, label="Composite Macro Shock Z_t")
    ewm_risk = z_risk.ewm(span=10).mean()
    ax3.plot(ewm_risk.index, ewm_risk.values, color="#d62728", linewidth=1.5, label="10-day EWM Filtered Risk")
    ax3.set_ylabel("Standardized Risk Score", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Date", fontsize=11)
    ax3.set_title("Systematic Macro Shock Signal Transmission", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_in_sample_equity_curve(
    backtest_is: pd.DataFrame,
    out_path: Union[str, Path],
):
    """
    Stage 7 Diagnostics: In-Sample (TRAIN 2016-2020) Strategy Equity Curve.
    Emits res_9_is_equity_curve.png (matches res_4_is_equity_curve.png in Graph_Alpha_Project).
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(13, 6))

    close = backtest_is["Close"].values
    market_returns = np.diff(np.log(close))

    # Strategies
    bh_wealth = np.exp(np.cumsum(market_returns))
    bh_wealth = np.insert(bh_wealth, 0, 1.0)

    std_state = backtest_is["Standard_State"].values
    std_returns = market_returns * std_state[:-1]
    std_wealth = np.exp(np.cumsum(std_returns))
    std_wealth = np.insert(std_wealth, 0, 1.0)

    causal_state = backtest_is["Causal_State"].values
    causal_returns = market_returns * causal_state[:-1]
    causal_wealth = np.exp(np.cumsum(causal_returns))
    causal_wealth = np.insert(causal_wealth, 0, 1.0)

    dates = backtest_is.index

    ax.plot(dates, bh_wealth, label="Buy & Hold Benchmark", color="#7f7f7f", linewidth=1.2, linestyle="--")
    ax.plot(dates, std_wealth, label="Static Baseline Ratchet (λ = 2.0)", color="#d62728", linewidth=1.5, alpha=0.85)
    ax.plot(dates, causal_wealth, label="Causal Adaptive Ratchet Strategy", color="#2ca02c", linewidth=2.0)

    ax.set_title("In-Sample (TRAIN: 2016–2020) Strategy Equity Curve Performance", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Cumulative Wealth ($1 Initial Capital)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_out_of_sample_equity_curve(
    backtest_oos: pd.DataFrame,
    out_path: Union[str, Path],
):
    """
    Stage 8 Diagnostics: Out-of-Sample (TEST 2021-2026) Strategy Equity Curve.
    Emits res_10_oos_equity_curve.png (matches res_5_oos_equity_curve.png in Graph_Alpha_Project).
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(13, 6))

    close = backtest_oos["Close"].values
    market_returns = np.diff(np.log(close))

    bh_wealth = np.exp(np.cumsum(market_returns))
    bh_wealth = np.insert(bh_wealth, 0, 1.0)

    std_state = backtest_oos["Standard_State"].values
    std_returns = market_returns * std_state[:-1]
    std_wealth = np.exp(np.cumsum(std_returns))
    std_wealth = np.insert(std_wealth, 0, 1.0)

    causal_state = backtest_oos["Causal_State"].values
    causal_returns = market_returns * causal_state[:-1]
    causal_wealth = np.exp(np.cumsum(causal_returns))
    causal_wealth = np.insert(causal_wealth, 0, 1.0)

    dates = backtest_oos.index

    ax.plot(dates, bh_wealth, label="Buy & Hold Benchmark (OOS)", color="#7f7f7f", linewidth=1.2, linestyle="--")
    ax.plot(dates, std_wealth, label="Static Baseline Ratchet (λ = 2.0)", color="#d62728", linewidth=1.5, alpha=0.85)
    ax.plot(dates, causal_wealth, label="Causal Adaptive Ratchet Strategy (OOS)", color="#2ca02c", linewidth=2.0)

    ax.set_title("True Out-of-Sample (TEST: 2021–2026) Strategy Equity Curve Performance", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Cumulative Wealth ($1 Initial Capital)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_out_of_sample_drawdown(
    backtest_oos: pd.DataFrame,
    out_path: Union[str, Path],
):
    """
    Stage 9 Diagnostics: Out-of-Sample Underwater Drawdown Profile.
    Emits res_11_oos_drawdown.png (matches res_9_oos_drawdown.png in Graph_Alpha_Project).
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(13, 6))

    close = backtest_oos["Close"].values
    market_returns = np.diff(np.log(close))

    def get_dd(returns):
        wealth = np.exp(np.cumsum(returns))
        wealth = np.insert(wealth, 0, 1.0)
        peak = np.maximum.accumulate(wealth)
        peak[peak == 0] = 1.0
        return (wealth - peak) / peak

    bh_dd = get_dd(market_returns) * 100.0
    std_state = backtest_oos["Standard_State"].values
    std_dd = get_dd(market_returns * std_state[:-1]) * 100.0
    causal_state = backtest_oos["Causal_State"].values
    causal_dd = get_dd(market_returns * causal_state[:-1]) * 100.0

    dates = backtest_oos.index

    ax.fill_between(dates, bh_dd, 0, color="#7f7f7f", alpha=0.25, label="Buy & Hold Drawdown")
    ax.fill_between(dates, std_dd, 0, color="#d62728", alpha=0.35, label="Static Baseline Drawdown")
    ax.fill_between(dates, causal_dd, 0, color="#2ca02c", alpha=0.45, label="Causal Adaptive Drawdown")

    ax.plot(dates, bh_dd, color="#7f7f7f", linewidth=0.8, alpha=0.7)
    ax.plot(dates, std_dd, color="#d62728", linewidth=1.2, alpha=0.85)
    ax.plot(dates, causal_dd, color="#1b5e20", linewidth=1.8)

    ax.set_title("Out-of-Sample (TEST: 2021–2026) Underwater Drawdown Analysis", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_regime_reentry_analysis(
    backtest_df: pd.DataFrame,
    out_path: Union[str, Path],
):
    """
    Stage 10 Diagnostics: Regime Classification, Ratchet Stops & State Transitions.
    Panel 1: Asset Close Price with active Causal Stop Level and cash holding periods.
    Panel 2: Daily position states & stop-out re-entry trigger events.
    Emits res_12_regime_reentry_analysis.png.
    """
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    close = backtest_df["Close"]
    stop_level = backtest_df["Causal_Stop_Level"]
    state = backtest_df["Causal_State"]

    # Panel 1: Price and Stop Level
    ax1.plot(close.index, close.values, color="#111111", linewidth=1.2, label="S&P 100 Close")
    ax1.plot(stop_level.index, stop_level.values, color="#d62728", linewidth=1.0, linestyle="--", label="Causal Ratchet Stop Level")

    # Shade cash positions (state == 0)
    cash_mask = state == 0
    ax1.fill_between(close.index, close.min() * 0.95, close.max() * 1.05, where=cash_mask, color="#ffc107", alpha=0.3, label="Cash Protection State (Post Stop-Out)")
    ax1.set_ylabel("Price ($)", fontsize=10, fontweight="bold")
    ax1.set_title("Trailing Stop Ratchet Execution and Capital Preservation Intervals", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)

    # Panel 2: Invested State & Transition Events
    ax2.step(state.index, state.values, where="post", color="#1f77b4", linewidth=1.2, label="Invested State (1=Invested, 0=Cash)")
    # Mark stop-out exit points (1 -> 0)
    stop_outs = (state.shift(1) == 1) & (state == 0)
    re_entries = (state.shift(1) == 0) & (state == 1)
    ax2.scatter(state.index[stop_outs], [1] * stop_outs.sum(), color="#d62728", marker="v", s=60, label="Stop-Out Exit Breached", zorder=5)
    ax2.scatter(state.index[re_entries], [0] * re_entries.sum(), color="#2ca02c", marker="^", s=60, label="Regime Re-entry Triggered", zorder=5)

    ax2.set_ylabel("Position State", fontsize=10, fontweight="bold")
    ax2.set_ylim(-0.2, 1.3)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Cash", "Invested"])
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_title("State Transitions: Stop-Outs and Dynamic Re-entry Triggers", fontsize=12, fontweight="bold")
    ax2.legend(loc="center right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
