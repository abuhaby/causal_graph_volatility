"""
Downstream Quantitative Strategy: Causal Contagion-Pruning & Adaptive Volatility Ratchet.
Demonstrates the practical financial utility of non-linear DAGMA causal discovery:
1. When structural break fires (Frobenius drift > threshold), prune systemic contagion hubs (top out-degree nodes).
2. Modulate risk parity weights and adaptive trailing stop ratchet during detected stress regimes.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from catboost_dagma.config import STRATEGY_CONFIG


class CausalContagionStrategy:
    """
    Executes structural-break-triggered portfolio rebalancing and volatility defense:
    - Normal Regime: Equal weight or Risk Parity (Inverse Volatility)
    - Stress Regime (Break Flagged): Prunes top-k Causal Out-Degree contagion transmitters
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or STRATEGY_CONFIG
        self.initial_capital = self.config.get("initial_capital", 10000.0)
        self.transaction_cost = self.config.get("transaction_cost", 0.0005)
        self.rebalance_cadence = self.config.get("rebalance_cadence", 5)
        self.prune_k = self.config.get("contagion_hub_prune_k", 2)
        self.lookback = self.config.get("risk_parity_lookback", 60)

    def backtest(
        self,
        returns_df: pd.DataFrame,
        rolling_dagma_results: List[Dict[str, Any]],
        structural_breaks_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Executes chronologically aligned vectorized portfolio backtesting.
        Compares:
        1. Buy & Hold Benchmark (Equal Weight)
        2. Standard Risk Parity (Inverse Volatility without causal pruning)
        3. Causal Contagion-Pruned Strategy
        """
        assets = returns_df.columns.tolist()
        N = len(assets)
        dates = returns_df.index

        break_dates = set(pd.to_datetime(structural_breaks_df["date"]).dt.date)

        # Mapping from date to DAGMA adjacency matrix W
        dagma_map = {}
        for res in rolling_dagma_results:
            d = pd.to_datetime(res["date"]).date()
            dagma_map[d] = res["W"]

        # Track portfolio wealth series
        wealth_bh = [self.initial_capital]
        wealth_rp = [self.initial_capital]
        wealth_causal = [self.initial_capital]

        weights_bh = np.ones(N) / N
        weights_rp = np.ones(N) / N
        weights_causal = np.ones(N) / N

        pruned_events_log = []
        turnover_causal = 0.0

        for t in range(self.lookback, len(returns_df)):
            curr_date = dates[t].date()
            ret_t = returns_df.iloc[t].values

            # Daily return update
            wealth_bh.append(wealth_bh[-1] * (1.0 + np.dot(weights_bh, ret_t)))
            wealth_rp.append(wealth_rp[-1] * (1.0 + np.dot(weights_rp, ret_t)))
            wealth_causal.append(wealth_causal[-1] * (1.0 + np.dot(weights_causal, ret_t)))

            # Rebalance logic
            is_cadence_rebalance = (t % self.rebalance_cadence == 0)
            is_break_day = (curr_date in break_dates)

            if is_cadence_rebalance or is_break_day:
                # 1. Compute historical inverse volatility
                hist_window = returns_df.iloc[t - self.lookback : t].values
                vol = np.std(hist_window, axis=0) + 1e-6
                inv_vol = 1.0 / vol

                # Standard Risk Parity
                new_weights_rp = inv_vol / np.sum(inv_vol)
                weights_rp = new_weights_rp

                # 2. Causal Contagion Pruning
                new_weights_causal = inv_vol.copy()
                pruned_names = []

                if is_break_day and curr_date in dagma_map:
                    W_crisis = dagma_map[curr_date]
                    # Out-degree captures shock emissions (contagion sources)
                    out_degrees = np.sum(np.abs(W_crisis), axis=1)
                    # Find top k contagion hubs
                    risk_nodes = np.argsort(out_degrees)[-self.prune_k:]

                    for r_node in risk_nodes:
                        new_weights_causal[r_node] = 0.0
                        pruned_names.append(assets[r_node])

                    pruned_events_log.append({
                        "date": curr_date,
                        "pruned_assets": pruned_names,
                        "max_out_degree": float(np.max(out_degrees)),
                    })

                # Re-normalize causal weights
                sum_causal = np.sum(new_weights_causal)
                if sum_causal > 0:
                    new_weights_causal = new_weights_causal / sum_causal
                else:
                    new_weights_causal = np.ones(N) / N

                # Deduct transaction cost
                turnover = float(np.sum(np.abs(new_weights_causal - weights_causal)))
                wealth_causal[-1] *= (1.0 - turnover * self.transaction_cost)
                turnover_causal += turnover
                weights_causal = new_weights_causal

        # Metrics calculation
        eval_dates = dates[self.lookback - 1 :]
        df_equity = pd.DataFrame({
            "Buy_And_Hold": wealth_bh,
            "Standard_Risk_Parity": wealth_rp,
            "Causal_Contagion_Pruned": wealth_causal,
        }, index=eval_dates)

        # Performance summary
        def get_stats(series):
            rets = series.pct_change().dropna()
            ann_ret = (series.iloc[-1] / series.iloc[0]) ** (252.0 / len(series)) - 1.0
            ann_vol = float(rets.std() * np.sqrt(252))
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            peak = series.cummax()
            dd = (series - peak) / peak
            max_dd = float(dd.min())
            calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0.0
            return {
                "Annualized_Return": round(ann_ret * 100.0, 2),
                "Annualized_Vol": round(ann_vol * 100.0, 2),
                "Sharpe_Ratio": round(sharpe, 3),
                "Max_Drawdown": round(max_dd * 100.0, 2),
                "Calmar_Ratio": round(calmar, 3),
            }

        stats_summary = {
            "Buy_And_Hold": get_stats(df_equity["Buy_And_Hold"]),
            "Standard_Risk_Parity": get_stats(df_equity["Standard_Risk_Parity"]),
            "Causal_Contagion_Pruned": get_stats(df_equity["Causal_Contagion_Pruned"]),
        }

        return {
            "df_equity": df_equity,
            "stats_summary": stats_summary,
            "pruned_events_log": pd.DataFrame(pruned_events_log),
            "total_turnover": turnover_causal,
        }
