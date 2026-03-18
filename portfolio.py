"""Portfolio-level optimization: combine BTC and ETH strategies.

Strategies:
1. Equal weight: 50/50 BTC/ETH
2. Relative strength rotation: overweight the stronger asset
3. Risk parity: weight inversely proportional to volatility
"""

import numpy as np
import pandas as pd


def combine_equity_curves(btc_equity: pd.Series, eth_equity: pd.Series,
                          method: str = "equal") -> pd.DataFrame:
    """Combine two equity curves into a portfolio.

    Args:
        btc_equity: BTC strategy equity curve (indexed by date)
        eth_equity: ETH strategy equity curve (indexed by date)
        method: 'equal', 'relative_strength', or 'risk_parity'

    Returns DataFrame with portfolio equity and metrics.
    """
    # Align indexes
    common = btc_equity.index.intersection(eth_equity.index)
    btc = btc_equity.loc[common]
    eth = eth_equity.loc[common]

    # Normalize to returns
    btc_ret = btc.pct_change().fillna(0)
    eth_ret = eth.pct_change().fillna(0)

    if method == "equal":
        port_ret = 0.5 * btc_ret + 0.5 * eth_ret

    elif method == "relative_strength":
        # Use 30-bar momentum to determine weights
        btc_mom = btc.pct_change(30).fillna(0)
        eth_mom = eth.pct_change(30).fillna(0)
        # Overweight the one with higher recent momentum
        btc_w = np.where(btc_mom > eth_mom, 0.7, 0.3)
        eth_w = 1.0 - btc_w
        port_ret = btc_w * btc_ret + eth_w * eth_ret

    elif method == "risk_parity":
        # Weight inversely proportional to rolling volatility
        btc_vol = btc_ret.rolling(30).std().fillna(btc_ret.std())
        eth_vol = eth_ret.rolling(30).std().fillna(eth_ret.std())
        total_inv_vol = (1 / btc_vol) + (1 / eth_vol)
        btc_w = (1 / btc_vol) / total_inv_vol
        eth_w = (1 / eth_vol) / total_inv_vol
        port_ret = btc_w * btc_ret + eth_w * eth_ret

    else:
        raise ValueError(f"Unknown method: {method}")

    # Build equity curve
    port_equity = (1 + port_ret).cumprod()

    return pd.DataFrame({
        "BTC": btc / btc.iloc[0],
        "ETH": eth / eth.iloc[0],
        "Portfolio": port_equity,
        "Port_Return": port_ret,
    })


def compute_portfolio_metrics(port_df: pd.DataFrame, label: str = "") -> dict:
    """Compute standard metrics for a portfolio equity curve."""
    equity = port_df["Portfolio"]
    returns = port_df["Port_Return"]

    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    ann_ret = ((1 + total_ret / 100) ** (365 * 6 / len(equity)) - 1) * 100  # 4H bars
    ann_vol = returns.std() * np.sqrt(365 * 6) * 100
    sharpe = (ann_ret - 5) / ann_vol if ann_vol > 0 else 0  # 5% risk-free

    peak = equity.expanding().max()
    dd = (equity - peak) / peak * 100
    max_dd = dd.min()

    return {
        "label": label,
        "total_return": round(total_ret, 2),
        "ann_return": round(ann_ret, 2),
        "ann_vol": round(ann_vol, 2),
        "sharpe": round(sharpe / 100, 4) if ann_vol > 0 else None,
        "max_dd": round(max_dd, 2),
        "start": str(equity.index[0]),
        "end": str(equity.index[-1]),
    }
