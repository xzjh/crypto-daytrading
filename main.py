"""CLI entry point for the crypto day trading signal system."""

import argparse
import sys

import numpy as np

from core import config
from core.data_fetcher import fetch_ohlcv
from core.indicators import add_indicators
from strategies.robust import evaluate_signals as btc_evaluate_signals
from strategies.eth import evaluate_signals as eth_evaluate_signals, add_eth_indicators
from core.backtester import run_backtest, run_period_comparison
from strategies.portfolio import run_rotation_backtest


def print_signal(result: dict):
    """Pretty-print a signal result."""
    rec = result["recommendation"]
    colors = {"BUY": "\033[92m", "SELL": "\033[91m", "HOLD": "\033[93m"}
    reset = "\033[0m"
    color = colors.get(rec, "")

    print(f"\n{'='*55}")
    print(f"  {result['symbol']} | {result['time']}")
    print(f"  Price: ${result['price']:,.2f}  |  RSI: {result['rsi']:.1f}")
    print(f"  Entry Conditions: {result['conditions_met']}")
    print(f"  Signal: {color}{rec}{reset}")
    print(f"  {'─'*51}")
    for indicator, status in result["signals"].items():
        print(f"    {indicator:>16}: {status}")
    print(f"{'='*55}")


def print_metrics(metrics: dict):
    """Pretty-print backtest metrics."""
    ret = metrics["return_pct"]
    bh = metrics["buy_hold_return_pct"]
    alpha = round(ret - bh, 2) if ret is not None and bh is not None else None

    print(f"\n{'='*55}")
    print(f"  BACKTEST: {metrics['symbol']}")
    print(f"  {metrics['start']} → {metrics['end']}")
    print(f"  {'─'*51}")
    print(f"  Strategy Return: {ret:>8}%")
    print(f"  Buy & Hold:      {bh:>8}%")
    print(f"  Alpha:           {alpha:>+8}%")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:>8}")
    print(f"  Max Drawdown:    {metrics['max_drawdown_pct']:>8}%")
    print(f"  Win Rate:        {metrics['win_rate_pct']:>8}%")
    print(f"  # Trades:        {metrics['num_trades']:>8}")
    print(f"  Profit Factor:   {metrics['profit_factor']:>8}")
    print(f"  Avg Trade:       {metrics['avg_trade_pct']:>8}%")
    print(f"{'='*55}")


def print_period_comparison(results: list, symbol: str):
    """Print per-quarter comparison table."""
    if not results:
        return

    print(f"\n  QUARTERLY BREAKDOWN: {symbol}")
    print(f"  {'Period':<20} {'Market':>6} {'Strategy':>9} {'B&H':>9} {'Alpha':>8} {'Sharpe':>7} {'MaxDD':>7} {'Trades':>6}")
    print(f"  {'─'*80}")

    bull_alphas, bear_alphas, flat_alphas = [], [], []
    for r in results:
        sharpe_str = f"{r['sharpe']:.2f}" if r['sharpe'] is not None else "  N/A"
        print(f"  {r['period']:<20} {r['market']:>6} {r['strategy_ret']:>+8.1f}% {r['bh_ret']:>+8.1f}% {r['alpha']:>+7.1f}% {sharpe_str:>7} {r['max_dd']:>6.1f}% {r['trades']:>6}")
        if r["market"] == "BULL":
            bull_alphas.append(r["alpha"])
        elif r["market"] == "BEAR":
            bear_alphas.append(r["alpha"])
        else:
            flat_alphas.append(r["alpha"])

    print(f"  {'─'*80}")

    all_alphas = [r["alpha"] for r in results]
    all_rets = [r["strategy_ret"] for r in results]

    print(f"\n  SUMMARY:")
    print(f"  Quarters tested:      {len(results)}")
    print(f"  Strategy profitable:  {sum(1 for r in all_rets if r > 0)}/{len(results)}")
    print(f"  Alpha positive:       {sum(1 for a in all_alphas if a > 0)}/{len(results)}")
    print(f"  Mean Alpha:           {sum(all_alphas)/len(all_alphas):+.2f}%")

    if bull_alphas:
        print(f"\n  In BULL quarters ({len(bull_alphas)}): mean alpha = {sum(bull_alphas)/len(bull_alphas):+.1f}%")
    if bear_alphas:
        print(f"  In BEAR quarters ({len(bear_alphas)}): mean alpha = {sum(bear_alphas)/len(bear_alphas):+.1f}%")
    if flat_alphas:
        print(f"  In FLAT quarters ({len(flat_alphas)}): mean alpha = {sum(flat_alphas)/len(flat_alphas):+.1f}%")


def _prepare_data(symbol, days):
    """Fetch and prepare data with asset-specific indicators."""
    df = fetch_ohlcv(symbol, config.SIGNAL_TIMEFRAME, days=days)
    if "ETH" in symbol:
        return add_eth_indicators(df)
    return add_indicators(df)


def cmd_signal(args):
    """Fetch latest data and output current signals."""
    symbols = args.symbols if args.symbols else config.SYMBOLS

    # Collect equity curves for rotation signal
    btc_eq = eth_eq = None

    for symbol in symbols:
        print(f"\nFetching {symbol} data...")
        df = _prepare_data(symbol, 250)
        if "ETH" in symbol:
            result = eth_evaluate_signals(df, symbol)
        else:
            result = btc_evaluate_signals(df, symbol)
        print_signal(result)

    # Portfolio rotation signal (if both symbols present)
    if len(symbols) >= 2 and any("BTC" in s for s in symbols) and any("ETH" in s for s in symbols):
        print(f"\n{'='*55}")
        print(f"  PORTFOLIO ROTATION SIGNAL")
        print(f"  {'─'*51}")

        btc_df = _prepare_data("BTC/USDT", 250)
        eth_df = _prepare_data("ETH/USDT", 250)

        # Use recent returns as momentum proxy
        btc_ret_20d = btc_df["Close"].iloc[-1] / btc_df["Close"].iloc[-120] - 1
        eth_ret_20d = eth_df["Close"].iloc[-1] / eth_df["Close"].iloc[-120] - 1

        if btc_ret_20d > eth_ret_20d:
            alloc = "BTC 70% / ETH 30%"
        else:
            alloc = "BTC 30% / ETH 70%"

        print(f"  BTC 20d momentum:  {btc_ret_20d:>+.1%}")
        print(f"  ETH 20d momentum:  {eth_ret_20d:>+.1%}")
        print(f"  Recommended:       {alloc}")
        print(f"{'='*55}")


def cmd_backtest(args):
    """Run backtests on historical data."""
    symbols = args.symbols if args.symbols else config.SYMBOLS
    days = args.days

    equity_curves = {}
    for symbol in symbols:
        print(f"\nFetching {symbol} data ({days} days)...")
        df = _prepare_data(symbol, days)
        print(f"Running backtest on {len(df)} candles...")

        metrics, stats = run_backtest(df, symbol, plot=args.plot)
        print_metrics(metrics)
        equity_curves[symbol] = stats["_equity_curve"]["Equity"]

        period_results = run_period_comparison(df, symbol)
        print_period_comparison(period_results, symbol)

    # Portfolio rotation backtest (if both BTC and ETH present)
    btc_key = next((k for k in equity_curves if "BTC" in k), None)
    eth_key = next((k for k in equity_curves if "ETH" in k), None)

    if btc_key and eth_key:
        print(f"\n{'='*55}")
        print(f"  PORTFOLIO ROTATION BACKTEST")
        print(f"  {'─'*51}")
        print(f"  Config: LB=120 bars, Weight=70/30, Rebal=30 bars")

        result = run_rotation_backtest(
            equity_curves[btc_key], equity_curves[eth_key],
            lookback=120, strong_weight=0.70,
            rebal_every=30, rebal_cost=0.001,
        )

        print(f"  {'─'*51}")
        print(f"  Portfolio Return:  {result['total_return']:>+.1f}%")
        print(f"  Sharpe Ratio:      {result['sharpe']:.4f}")
        print(f"  Max Drawdown:      {result['max_dd']:.1f}%")
        print(f"  Rebalances:        {result['n_rebalances']}")
        print(f"  BTC overweight:    {result['btc_overweight_pct']:.0f}% of time")
        print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser(description="Crypto Day Trading Signal System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    sig_parser = subparsers.add_parser("signal", help="Get current trading signals")
    sig_parser.add_argument("--symbols", nargs="+", help="Trading pairs")

    bt_parser = subparsers.add_parser("backtest", help="Run historical backtest")
    bt_parser.add_argument("--days", type=int, default=1095, help="Days of history (default: 1095)")
    bt_parser.add_argument("--symbols", nargs="+", help="Trading pairs")
    bt_parser.add_argument("--plot", action="store_true", help="Open interactive plot")

    args = parser.parse_args()

    if args.command == "signal":
        cmd_signal(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
