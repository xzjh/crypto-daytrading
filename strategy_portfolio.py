"""Portfolio rotation strategy: dynamically weight BTC vs ETH.

Uses momentum-based relative strength with realistic constraints:
- Rebalance only every N bars (not every bar)
- Moderate tilt (not 100/0 switching)
- Transaction costs on rebalance
"""

import numpy as np
import pandas as pd
from backtesting import Backtest

import config
from data_fetcher import fetch_ohlcv
from indicators import add_indicators
from strategy_eth import add_eth_indicators
from strategy_robust import RobustTrendStrategy
from strategy_eth import ETHTrendStrategy


def run_rotation_backtest(btc_eq, eth_eq, lookback=120, strong_weight=0.65,
                          rebal_every=30, rebal_cost=0.001):
    """Run rotation backtest with realistic constraints.

    Args:
        btc_eq, eth_eq: equity curves from underlying strategies
        lookback: momentum lookback in bars
        strong_weight: weight on stronger asset (0.5 = equal, 1.0 = all-in)
        rebal_every: bars between rebalances
        rebal_cost: transaction cost per rebalance (as fraction)
    """
    common = btc_eq.index.intersection(eth_eq.index)
    btc_n = (btc_eq.loc[common] / btc_eq.loc[common].iloc[0])
    eth_n = (eth_eq.loc[common] / eth_eq.loc[common].iloc[0])

    btc_ret = btc_n.pct_change().fillna(0)
    eth_ret = eth_n.pct_change().fillna(0)

    weak_weight = 1.0 - strong_weight
    btc_w = np.full(len(common), 0.5)
    bars_since_rebal = rebal_every  # Allow immediate first rebalance

    for i in range(lookback, len(common)):
        bars_since_rebal += 1
        if bars_since_rebal >= rebal_every:
            btc_mom = btc_n.iloc[i] / btc_n.iloc[i - lookback] - 1
            eth_mom = eth_n.iloc[i] / eth_n.iloc[i - lookback] - 1
            new_w = strong_weight if btc_mom > eth_mom else weak_weight
            if new_w != btc_w[i - 1]:
                bars_since_rebal = 0
            btc_w[i] = new_w
        else:
            btc_w[i] = btc_w[i - 1]

    eth_w = 1.0 - btc_w

    # Apply rebalancing costs
    w_changes = np.abs(np.diff(btc_w, prepend=btc_w[0]))
    costs = w_changes * rebal_cost

    port_ret = btc_w * btc_ret.values + eth_w * eth_ret.values - costs
    port_eq = (1 + port_ret).cumprod()

    # Metrics
    total_ret = (port_eq[-1] - 1) * 100
    ann_vol = np.std(port_ret) * np.sqrt(365 * 6) * 100
    ann_ret = (port_eq[-1] ** (365 * 6 / len(port_eq)) - 1) * 100
    sharpe = (ann_ret - 5) / ann_vol if ann_vol > 0 else 0

    peak = np.maximum.accumulate(port_eq)
    dd = ((port_eq - peak) / peak * 100).min()

    n_rebalances = int(np.sum(w_changes > 0.01))
    btc_pct = (btc_w > 0.5).mean() * 100

    return {
        "total_return": round(total_ret, 1),
        "ann_return": round(ann_ret, 1),
        "sharpe": round(sharpe, 4),
        "max_dd": round(dd, 1),
        "n_rebalances": n_rebalances,
        "btc_overweight_pct": round(btc_pct, 0),
        "equity": pd.Series(port_eq, index=common),
        "weights_btc": pd.Series(btc_w, index=common),
    }


def walk_forward_rotation(btc_eq, eth_eq, lookback=120, strong_weight=0.65,
                           rebal_every=30, window_bars=1080):
    """Walk-forward validation of rotation strategy.

    Splits the equity curves into windows and tests rotation on each.
    """
    common = btc_eq.index.intersection(eth_eq.index)
    btc_n = (btc_eq.loc[common] / btc_eq.loc[common].iloc[0])
    eth_n = (eth_eq.loc[common] / eth_eq.loc[common].iloc[0])
    total = len(common)

    results = []
    start = 0
    while start + window_bars <= total:
        end = start + window_bars
        # Run rotation on this window
        btc_w = btc_n.iloc[start:end]
        eth_w = eth_n.iloc[start:end]
        # Re-normalize
        btc_w = btc_w / btc_w.iloc[0]
        eth_w = eth_w / eth_w.iloc[0]

        btc_ret = btc_w.pct_change().fillna(0)
        eth_ret = eth_w.pct_change().fillna(0)

        # Simple rotation
        weights = np.full(len(btc_w), 0.5)
        weak = 1.0 - strong_weight
        for i in range(lookback, len(btc_w)):
            btc_mom = btc_w.iloc[i] / btc_w.iloc[max(0, i - lookback)] - 1
            eth_mom = eth_w.iloc[i] / eth_w.iloc[max(0, i - lookback)] - 1
            weights[i] = strong_weight if btc_mom > eth_mom else weak

        port_ret = weights * btc_ret.values + (1 - weights) * eth_ret.values
        port_eq_val = (1 + port_ret).cumprod()
        window_ret = (port_eq_val[-1] - 1) * 100

        # B&H comparison (equal weight)
        bh_ret_val = (0.5 * btc_ret + 0.5 * eth_ret)
        bh_eq = (1 + bh_ret_val).cumprod()
        bh_ret = (bh_eq.iloc[-1] - 1) * 100

        period = f"{common[start].strftime('%Y-%m')} → {common[min(end-1, total-1)].strftime('%Y-%m')}"
        results.append({
            "period": period,
            "rotation_ret": round(window_ret, 1),
            "equal_ret": round(bh_ret, 1),
            "alpha": round(window_ret - bh_ret, 1),
        })
        start += window_bars

    return results
