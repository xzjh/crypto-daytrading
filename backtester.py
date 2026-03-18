"""Run backtests and extract performance metrics."""

import warnings
import numpy as np
from backtesting import Backtest

import config
from strategy_robust import RobustTrendStrategy
from strategy_eth import ETHTrendStrategy

warnings.filterwarnings("ignore")


def _get_strategy(symbol):
    """Return the appropriate strategy class for a given symbol."""
    if "ETH" in symbol:
        return ETHTrendStrategy
    return RobustTrendStrategy


def run_backtest(df, symbol: str, plot: bool = False, **strategy_params) -> dict:
    """Run backtest on prepared DataFrame."""
    strategy_cls = _get_strategy(symbol)
    bt = Backtest(
        df, strategy_cls,
        cash=config.BACKTEST_CASH,
        commission=config.BACKTEST_COMMISSION,
        exclusive_orders=True,
        trade_on_close=True,
    )
    stats = bt.run(**strategy_params)

    def _safe(key):
        v = stats[key]
        return round(v, 2) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None

    metrics = {
        "symbol": symbol,
        "start": str(stats["Start"]),
        "end": str(stats["End"]),
        "duration": str(stats["Duration"]),
        "return_pct": _safe("Return [%]"),
        "buy_hold_return_pct": _safe("Buy & Hold Return [%]"),
        "sharpe_ratio": round(stats["Sharpe Ratio"], 4) if stats["Sharpe Ratio"] is not None else None,
        "max_drawdown_pct": _safe("Max. Drawdown [%]"),
        "win_rate_pct": _safe("Win Rate [%]"),
        "num_trades": stats["# Trades"],
        "profit_factor": round(stats["Profit Factor"], 4) if stats["Profit Factor"] is not None else None,
        "avg_trade_pct": _safe("Avg. Trade [%]"),
    }

    if plot:
        bt.plot(open_browser=True)

    return metrics, stats


def run_period_comparison(df, symbol: str, candles_per_quarter=540):
    """Split data into quarters and compare strategy vs buy-and-hold."""
    strategy_cls = _get_strategy(symbol)
    total = len(df)
    results = []
    start = 0

    while start + candles_per_quarter <= total:
        end = min(start + candles_per_quarter, total)
        df_period = df.iloc[start:end].copy()
        if len(df_period) < 100:
            break
        try:
            bt = Backtest(
                df_period, strategy_cls,
                cash=config.BACKTEST_CASH,
                commission=config.BACKTEST_COMMISSION,
                exclusive_orders=True, trade_on_close=True,
            )
            stats = bt.run()
            ret = stats["Return [%]"]
            bh = stats["Buy & Hold Return [%]"]
            results.append({
                "period": f"{df_period.index[0].strftime('%Y-%m')} → {df_period.index[-1].strftime('%Y-%m')}",
                "strategy_ret": round(ret, 2),
                "bh_ret": round(bh, 2),
                "alpha": round(ret - bh, 2),
                "sharpe": round(stats["Sharpe Ratio"], 2) if stats["Sharpe Ratio"] is not None else None,
                "max_dd": round(stats["Max. Drawdown [%]"], 2),
                "trades": stats["# Trades"],
                "market": "BULL" if bh > 5 else ("BEAR" if bh < -5 else "FLAT"),
            })
        except Exception:
            pass
        start += candles_per_quarter

    return results
