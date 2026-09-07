"""
Vectorized Trailing Stop Ratchet State Machine Engine (Feature 15).

Provides high-performance state machine simulation for adaptive risk bounds,
enforcing ratchet non-decreasing properties and regime-dependent re-entry logic.
"""

import numpy as np
import pandas as pd


def run_trailing_stop(
    prices: pd.Series,
    vol_series: pd.Series,
    multiplier: pd.Series,
    calm_threshold: float = 1.8,
) -> pd.DataFrame:
    """
    Execute 1D vectorized trailing stop ratchet state machine.

    Invariants:
    1. While invested (state == 1), Stop_t >= Stop_{t-1} (ratchet monotonicity).
    2. If Close_t < Stop_t, state immediately transitions to 0 (stop-out exit).
    3. Re-entry logic:
       - Calm regime (multiplier >= calm_threshold): Re-enters if Close_t > MA5_t.
       - Stormy regime (multiplier < calm_threshold): Requires Close_t > MA5_t AND Close_t > Close_{t-1}.

    Args:
        prices: Asset close price series.
        vol_series: Base realized volatility series (e.g., 5-day rolling Garman-Klass).
        multiplier: Dynamic risk multiplier series lambda_t.
        calm_threshold: Multiplier threshold separating calm vs stormy regimes (default: 1.8).

    Returns:
        DataFrame indexed by date containing:
            - 'Close': close prices
            - 'Base_Vol': volatility input
            - 'Multiplier': multiplier lambda_t
            - 'Close_MA5': 5-day moving average of close
            - 'Stop_Level': active ratchet stop level
            - 'Invested_State': binary position state (1 = invested, 0 = cash)
    """
    n = len(prices)
    close = prices.values.astype(float)
    vol = vol_series.values.astype(float)
    mult = multiplier.values.astype(float)

    ma5 = prices.rolling(window=5, min_periods=1).mean().values

    invested = np.zeros(n, dtype=int)
    stop_level = np.zeros(n, dtype=float)

    # Start invested on day 0
    curr_state = 1
    curr_stop = close[0] - mult[0] * vol[0] * close[0]

    for t in range(n):
        c = close[t]
        raw_stop = c - mult[t] * vol[t] * c

        if curr_state == 1:
            # Ratchet non-decreasing invariant while invested
            curr_stop = max(curr_stop, raw_stop)
            if c < curr_stop:
                # Stop breached -> transition to cash
                curr_state = 0
                curr_stop = raw_stop
        else:
            # Re-entry evaluation from cash state
            is_calm = mult[t] >= calm_threshold
            reenter = False
            if is_calm:
                if c > ma5[t]:
                    reenter = True
            else:
                prev_c = close[t - 1] if t > 0 else c
                if c > ma5[t] and c > prev_c:
                    reenter = True

            if reenter:
                curr_state = 1
                curr_stop = raw_stop

        invested[t] = curr_state
        stop_level[t] = curr_stop

    return pd.DataFrame(
        {
            "Close": close,
            "Base_Vol": vol,
            "Multiplier": mult,
            "Close_MA5": ma5,
            "Stop_Level": stop_level,
            "Invested_State": invested,
        },
        index=prices.index,
    )


def execute_causal_trailing_stop(
    df_raw: pd.DataFrame,
    df_processed: pd.DataFrame,
    lambda_t: pd.Series,
    static_multiplier: float = 2.0,
    calm_threshold: float = 1.8,
) -> pd.DataFrame:
    """
    Execute dual backtest comparing standard static baseline against causal dynamic stop.

    Args:
        df_raw: Raw DataFrame containing 'SP100_Close'.
        df_processed: Processed DataFrame containing 'Garman_Klass_Vol'.
        lambda_t: Dynamic multiplier series from causal module.
        static_multiplier: Fixed multiplier for baseline (default: 2.0).
        calm_threshold: Calm regime threshold (default: 1.8).

    Returns:
        DataFrame indexed by date with both Standard and Causal stop levels and states.
    """
    common_idx = df_processed.index.intersection(lambda_t.index)
    prices = df_raw.loc[common_idx, "SP100_Close"]
    vol = df_processed.loc[common_idx, "Garman_Klass_Vol"].rolling(5, min_periods=1).mean().bfill()
    mult_causal = lambda_t.loc[common_idx]
    mult_static = pd.Series(static_multiplier, index=common_idx)

    res_static = run_trailing_stop(prices, vol, mult_static, calm_threshold=calm_threshold)
    res_causal = run_trailing_stop(prices, vol, mult_causal, calm_threshold=calm_threshold)

    out_df = pd.DataFrame(
        {
            "Close": prices,
            "Base_Vol": vol,
            "Causal_Multiplier": mult_causal,
            "Close_MA5": res_causal["Close_MA5"],
            "Standard_Stop_Level": res_static["Stop_Level"],
            "Causal_Stop_Level": res_causal["Stop_Level"],
            "Standard_State": res_static["Invested_State"],
            "Causal_State": res_causal["Invested_State"],
        },
        index=common_idx,
    )
    return out_df
