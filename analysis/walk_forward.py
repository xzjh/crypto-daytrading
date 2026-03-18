"""Walk-forward analysis to validate strategy is not overfit.

Splits data into rolling train/test windows:
- Train on N months, test on next M months, slide forward
- Optimizes params on train set, applies to test set
- Aggregates out-of-sample performance across all test windows
"""

import itertools
import warnings
import numpy as np
import pandas as pd
from backtesting import Backtest

from core import config
from core.data_fetcher import fetch_ohlcv
from core.indicators import add_indicators
from strategies.legacy import ConfluenceStrategy

warnings.filterwarnings("ignore")

# Candles per month at 4H
CANDLES_PER_MONTH_4H = 30 * 6  # ~180 candles


def _optimize_on_window(df_train):
    """Find best params on a training window. Returns best params dict."""
    param_grid = {
        "entry_bull_threshold": [2.5, 3.0, 3.15, 3.25, 3.5],
        "exit_bear_threshold": [3.0, 3.5, 4.0, 4.5],
        "atr_sl_mult": [2.5, 2.75, 2.85, 3.0, 3.25],
        "cooldown": [3, 4, 5],
        "use_trailing": [True],
    }
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    best_sharpe = -999
    best_params = None

    for values in combos:
        params = dict(zip(keys, values))
        try:
            bt = Backtest(df_train, ConfluenceStrategy, cash=config.BACKTEST_CASH,
                          commission=config.BACKTEST_COMMISSION, exclusive_orders=True,
                          trade_on_close=True)
            stats = bt.run(**params)
            sharpe = stats["Sharpe Ratio"]
            trades = stats["# Trades"]
            if trades >= 3 and sharpe is not None and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        except Exception:
            continue

    return best_params, best_sharpe


def _test_on_window(df_test, params):
    """Run strategy with fixed params on test window. Returns metrics."""
    try:
        bt = Backtest(df_test, ConfluenceStrategy, cash=config.BACKTEST_CASH,
                      commission=config.BACKTEST_COMMISSION, exclusive_orders=True,
                      trade_on_close=True)
        stats = bt.run(**params)
        bh_ret = stats["Buy & Hold Return [%]"]
        strat_ret = stats["Return [%]"]
        return {
            "return_pct": round(strat_ret, 2),
            "buy_hold_pct": round(bh_ret, 2),
            "alpha": round(strat_ret - bh_ret, 2),
            "sharpe": round(stats["Sharpe Ratio"], 4) if stats["Sharpe Ratio"] is not None else None,
            "max_dd": round(stats["Max. Drawdown [%]"], 2),
            "win_rate": round(stats["Win Rate [%]"], 2) if stats["Win Rate [%]"] is not None else None,
            "trades": stats["# Trades"],
            "params": params,
        }
    except Exception as e:
        return None


def walk_forward(symbol="BTC/USDT", days=1095, train_months=6, test_months=3):
    """Run walk-forward analysis with rolling windows."""
    print(f"\n{'#'*70}")
    print(f"  WALK-FORWARD ANALYSIS: {symbol}")
    print(f"  Train: {train_months}mo | Test: {test_months}mo | Data: {days} days")
    print(f"{'#'*70}")

    print("\nFetching data...")
    df_raw = fetch_ohlcv(symbol, config.SIGNAL_TIMEFRAME, days=days)
    df = add_indicators(df_raw)
    total_candles = len(df)
    print(f"Total candles: {total_candles}")

    train_size = train_months * CANDLES_PER_MONTH_4H
    test_size = test_months * CANDLES_PER_MONTH_4H
    step = test_size

    windows = []
    start = 0
    while start + train_size + test_size <= total_candles:
        train_end = start + train_size
        test_end = train_end + test_size
        windows.append((start, train_end, test_end))
        start += step

    print(f"Windows: {len(windows)}\n")

    oos_results = []  # Out-of-sample results
    is_results = []   # In-sample results

    for i, (train_start, train_end, test_end) in enumerate(windows):
        df_train = df.iloc[train_start:train_end].copy()
        df_test = df.iloc[train_end:test_end].copy()

        train_period = f"{df_train.index[0].strftime('%Y-%m-%d')} → {df_train.index[-1].strftime('%Y-%m-%d')}"
        test_period = f"{df_test.index[0].strftime('%Y-%m-%d')} → {df_test.index[-1].strftime('%Y-%m-%d')}"

        # Optimize on training data
        best_params, train_sharpe = _optimize_on_window(df_train)
        if best_params is None:
            print(f"  Window {i+1}: No valid params found on train set, skipping")
            continue

        # Test on out-of-sample data
        is_result = _test_on_window(df_train, best_params)
        oos_result = _test_on_window(df_test, best_params)

        if oos_result is None:
            print(f"  Window {i+1}: Test failed, skipping")
            continue

        is_results.append(is_result)
        oos_results.append(oos_result)

        print(f"  Window {i+1}/{len(windows)}: Train[{train_period}] Test[{test_period}]")
        print(f"    In-sample:     Ret={is_result['return_pct']:>7.1f}%  Sharpe={is_result['sharpe']}  Trades={is_result['trades']}")
        print(f"    Out-of-sample: Ret={oos_result['return_pct']:>7.1f}%  Sharpe={oos_result['sharpe']}  Alpha={oos_result['alpha']:>+.1f}%  Trades={oos_result['trades']}")
        print(f"    Params: entry={best_params['entry_bull_threshold']} atr={best_params['atr_sl_mult']} cd={best_params['cooldown']}")

    # Aggregate out-of-sample results
    if not oos_results:
        print("\nNo valid out-of-sample results!")
        return

    oos_returns = [r["return_pct"] for r in oos_results]
    oos_alphas = [r["alpha"] for r in oos_results]
    oos_sharpes = [r["sharpe"] for r in oos_results if r["sharpe"] is not None]
    oos_dds = [r["max_dd"] for r in oos_results]
    is_returns = [r["return_pct"] for r in is_results]

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD RESULTS (Out-of-Sample)")
    print(f"{'='*70}")
    print(f"  Windows tested:        {len(oos_results)}")
    print(f"  OOS Mean Return:       {np.mean(oos_returns):>+7.2f}%")
    print(f"  OOS Mean Alpha:        {np.mean(oos_alphas):>+7.2f}%")
    print(f"  OOS Mean Sharpe:       {np.mean(oos_sharpes):>7.4f}")
    print(f"  OOS Worst Return:      {min(oos_returns):>+7.2f}%")
    print(f"  OOS Best Return:       {max(oos_returns):>+7.2f}%")
    print(f"  OOS Worst Drawdown:    {min(oos_dds):>7.2f}%")
    print(f"  OOS Win Rate (windows):{sum(1 for r in oos_returns if r > 0)}/{len(oos_returns)}")
    print(f"  OOS Alpha+ (windows):  {sum(1 for a in oos_alphas if a > 0)}/{len(oos_alphas)}")

    # Overfitting check: compare in-sample vs out-of-sample
    is_mean = np.mean(is_returns)
    oos_mean = np.mean(oos_returns)
    degradation = ((is_mean - oos_mean) / abs(is_mean) * 100) if is_mean != 0 else 0

    print(f"\n  OVERFITTING CHECK:")
    print(f"  In-sample mean return:  {is_mean:>+7.2f}%")
    print(f"  Out-of-sample mean:     {oos_mean:>+7.2f}%")
    print(f"  Performance degradation:{degradation:>7.1f}%")
    if degradation < 30:
        print(f"  Verdict: LOW overfitting risk (degradation < 30%)")
    elif degradation < 50:
        print(f"  Verdict: MODERATE overfitting risk")
    else:
        print(f"  Verdict: HIGH overfitting risk (degradation > 50%)")
    print(f"{'='*70}")

    return oos_results


def param_sensitivity(symbol="BTC/USDT", days=1095):
    """Test how sensitive returns are to small param changes."""
    print(f"\n{'#'*70}")
    print(f"  PARAMETER SENSITIVITY ANALYSIS: {symbol}")
    print(f"{'#'*70}")

    df_raw = fetch_ohlcv(symbol, config.SIGNAL_TIMEFRAME, days=days)
    df = add_indicators(df_raw)

    base_params = {
        "entry_bull_threshold": 3.15,
        "exit_bear_threshold": 3.5,
        "atr_sl_mult": 2.85,
        "cooldown": 4,
        "use_trailing": True,
    }

    # Get baseline
    bt = Backtest(df, ConfluenceStrategy, cash=config.BACKTEST_CASH,
                  commission=config.BACKTEST_COMMISSION, exclusive_orders=True, trade_on_close=True)
    base_stats = bt.run(**base_params)
    base_ret = base_stats["Return [%]"]
    base_sharpe = base_stats["Sharpe Ratio"]
    print(f"\n  Baseline: Return={base_ret:.2f}%  Sharpe={base_sharpe:.4f}")

    # Perturb each parameter
    perturbations = {
        "entry_bull_threshold": [2.65, 2.9, 3.15, 3.4, 3.65],
        "exit_bear_threshold": [3.0, 3.25, 3.5, 3.75, 4.0],
        "atr_sl_mult": [2.35, 2.6, 2.85, 3.1, 3.35],
        "cooldown": [2, 3, 4, 5, 6],
    }

    print(f"\n  {'Parameter':<25} {'Values':>40}  {'Returns':>50}")
    print(f"  {'─'*120}")

    for param, values in perturbations.items():
        returns = []
        sharpes = []
        for v in values:
            test_params = base_params.copy()
            test_params[param] = v
            stats = bt.run(**test_params)
            returns.append(round(stats["Return [%]"], 1))
            s = stats["Sharpe Ratio"]
            sharpes.append(round(s, 2) if s is not None else 0)

        ret_str = "  ".join(f"{r:>7.1f}%" for r in returns)
        val_str = "  ".join(f"{v:>7}" for v in values)
        ret_range = max(returns) - min(returns)

        print(f"  {param:<25} {val_str}  {ret_str}  range={ret_range:.1f}%")

    print(f"\n  Low range = robust (not overfit). High range = sensitive (overfit risk).")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"

    walk_forward(symbol)
    param_sensitivity(symbol)
