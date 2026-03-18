"""Comprehensive strategy comparison on 6.5 years of data."""

import warnings
import numpy as np
import pandas as pd
from backtesting import Backtest

from core import config
from core.data_fetcher import fetch_ohlcv
from core.indicators import add_indicators
from strategies.eth import add_eth_indicators
from core.external_data import merge_external_data
from strategies.robust import RobustTrendStrategy
from strategies.eth import ETHTrendStrategy
from strategies.ml import MLStrategy, precompute_ml_signals
from strategies.portfolio import run_rotation_backtest, walk_forward_rotation

warnings.filterwarnings("ignore")

DAYS = 2400


def run_bt(df, strategy_cls):
    bt = Backtest(df, strategy_cls, cash=config.BACKTEST_CASH,
                  commission=config.BACKTEST_COMMISSION, exclusive_orders=True,
                  trade_on_close=True)
    stats = bt.run()
    equity = stats["_equity_curve"]["Equity"]
    return stats, equity


def fmt(stats, label):
    ret = stats["Return [%]"]
    bh = stats["Buy & Hold Return [%]"]
    sh = stats["Sharpe Ratio"]
    dd = stats["Max. Drawdown [%]"]
    tr = stats["# Trades"]
    sh_s = f"{sh:.4f}" if sh is not None else "N/A"
    return f"  {label:<28} Ret={ret:>+9.1f}%  B&H={bh:>+9.1f}%  Alpha={ret-bh:>+9.1f}%  Sharpe={sh_s:>8}  DD={dd:>7.1f}%  Tr={tr:>3}"


if __name__ == "__main__":
    print("=" * 85)
    print("  COMPREHENSIVE STRATEGY COMPARISON — 6.5 YEARS (2019.10 → 2026.03)")
    print("=" * 85)

    # ── Data ──
    print("\n[DATA]")
    btc_raw = fetch_ohlcv("BTC/USDT", "4h", days=DAYS)
    eth_raw = fetch_ohlcv("ETH/USDT", "4h", days=DAYS)

    btc = add_indicators(btc_raw)
    btc = merge_external_data(btc, "BTC/USDT")

    eth = add_eth_indicators(eth_raw)
    eth = merge_external_data(eth, "ETH/USDT")
    btc_trend = (btc[f"EMA_{config.EMA_MID}"] > btc[f"EMA_{config.EMA_SLOW}"]).astype(int)
    eth["BTC_Trend"] = btc_trend.reindex(eth.index, method="ffill").fillna(1)

    print(f"  BTC: {len(btc)} candles  ETH: {len(eth)} candles")
    print(f"  Period: {btc.index[0].strftime('%Y-%m-%d')} → {btc.index[-1].strftime('%Y-%m-%d')}")

    # ══════════════════════════════════════════════════════════
    print("\n" + "─" * 85)
    print("  [1] TREND FOLLOWING (production)")
    print("─" * 85)
    btc_tf, btc_tf_eq = run_bt(btc, RobustTrendStrategy)
    eth_tf, eth_tf_eq = run_bt(eth, ETHTrendStrategy)
    print(fmt(btc_tf, "BTC Trend Following"))
    print(fmt(eth_tf, "ETH Trend Following"))

    # ══════════════════════════════════════════════════════════
    print("\n" + "─" * 85)
    print("  [2] ML REGIME CLASSIFIER + TREND FOLLOWING")
    print("─" * 85)

    print("  Training BTC ML (walk-forward, retrain every 3mo)...")
    btc_ml_sig = precompute_ml_signals(btc, train_bars=2160, retrain_every=540, pred_horizon=24)
    btc_ml = btc.copy()
    btc_ml["ML_Signal"] = btc_ml_sig
    btc_ml_valid = btc_ml[btc_ml["ML_Signal"].notna()]
    if len(btc_ml_valid) > 100:
        btc_ml_stats, btc_ml_eq = run_bt(btc_ml_valid, MLStrategy)
        print(fmt(btc_ml_stats, "BTC ML+TF Hybrid"))

    print("  Training ETH ML (walk-forward, retrain every 3mo)...")
    eth_ml_sig = precompute_ml_signals(eth, train_bars=2160, retrain_every=540, pred_horizon=24)
    eth_ml = eth.copy()
    eth_ml["ML_Signal"] = eth_ml_sig
    eth_ml_valid = eth_ml[eth_ml["ML_Signal"].notna()]
    if len(eth_ml_valid) > 100:
        eth_ml_stats, eth_ml_eq = run_bt(eth_ml_valid, MLStrategy)
        print(fmt(eth_ml_stats, "ETH ML+TF Hybrid"))

    # ══════════════════════════════════════════════════════════
    print("\n" + "─" * 85)
    print("  [3] PORTFOLIO ROTATION (BTC+ETH relative strength)")
    print("─" * 85)

    # Test multiple configurations
    print(f"\n  {'Config':<35} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Rebal':>6}")
    print(f"  {'─'*72}")

    best_sharpe = 0
    best_config = None
    for lb in [60, 90, 120, 180]:
        for sw in [0.60, 0.65, 0.70]:
            for rb in [30, 60, 90]:
                r = run_rotation_backtest(btc_tf_eq, eth_tf_eq,
                                          lookback=lb, strong_weight=sw,
                                          rebal_every=rb, rebal_cost=0.001)
                label = f"LB={lb} W={sw:.0%} RB={rb}"
                if r["sharpe"] > 1.0:
                    print(f"  {label:<35} {r['total_return']:>+9.1f}% {r['sharpe']:>8.4f} {r['max_dd']:>7.1f}% {r['n_rebalances']:>5}")
                if r["sharpe"] > best_sharpe:
                    best_sharpe = r["sharpe"]
                    best_config = (lb, sw, rb)
                    best_result = r

    print(f"\n  Best: LB={best_config[0]} W={best_config[1]:.0%} RB={best_config[2]}")
    print(f"  Return: {best_result['total_return']:+.1f}%  Sharpe: {best_result['sharpe']:.4f}  MaxDD: {best_result['max_dd']:.1f}%")

    # Walk-forward validation for best rotation
    print(f"\n  Walk-forward validation (best config):")
    wf_results = walk_forward_rotation(btc_tf_eq, eth_tf_eq,
                                        lookback=best_config[0],
                                        strong_weight=best_config[1],
                                        rebal_every=best_config[2])
    for r in wf_results:
        print(f"    {r['period']}  Rotation={r['rotation_ret']:>+6.1f}%  Equal={r['equal_ret']:>+6.1f}%  Alpha={r['alpha']:>+5.1f}%")

    wf_alphas = [r["alpha"] for r in wf_results]
    wf_rets = [r["rotation_ret"] for r in wf_results]
    print(f"  WF Summary: mean ret={np.mean(wf_rets):+.1f}%  mean alpha={np.mean(wf_alphas):+.1f}%  alpha+: {sum(1 for a in wf_alphas if a > 0)}/{len(wf_alphas)}")

    # ══════════════════════════════════════════════════════════
    # ML portfolio rotation
    if btc_ml_eq is not None and eth_ml_eq is not None:
        print(f"\n  ML Portfolio Rotation:")
        for lb in [60, 90, 120]:
            for sw in [0.60, 0.65, 0.70]:
                r = run_rotation_backtest(btc_ml_eq, eth_ml_eq,
                                          lookback=lb, strong_weight=sw,
                                          rebal_every=60, rebal_cost=0.001)
                if r["sharpe"] > 0.8:
                    print(f"  ML LB={lb} W={sw:.0%}: Ret={r['total_return']:>+9.1f}%  Sharpe={r['sharpe']:.4f}  DD={r['max_dd']:.1f}%")

    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 85)
    print("  FINAL COMPARISON")
    print("=" * 85)
    print(f"\n  {'Strategy':<35} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print(f"  {'─'*65}")
    print(f"  {'BTC Buy & Hold':<35} {btc_tf['Buy & Hold Return [%]']:>+9.1f}% {'—':>8} {'~-50%':>8}")
    print(f"  {'ETH Buy & Hold':<35} {eth_tf['Buy & Hold Return [%]']:>+9.1f}% {'—':>8} {'~-65%':>8}")
    print(f"  {'BTC Trend Following':<35} {btc_tf['Return [%]']:>+9.1f}% {btc_tf['Sharpe Ratio']:.4f} {btc_tf['Max. Drawdown [%]']:>7.1f}%")
    print(f"  {'ETH Trend Following':<35} {eth_tf['Return [%]']:>+9.1f}% {eth_tf['Sharpe Ratio']:.4f} {eth_tf['Max. Drawdown [%]']:>7.1f}%")
    if btc_ml_stats is not None:
        print(f"  {'BTC ML+TF Hybrid':<35} {btc_ml_stats['Return [%]']:>+9.1f}% {btc_ml_stats['Sharpe Ratio']:.4f} {btc_ml_stats['Max. Drawdown [%]']:>7.1f}%")
    if eth_ml_stats is not None:
        print(f"  {'ETH ML+TF Hybrid':<35} {eth_ml_stats['Return [%]']:>+9.1f}% {eth_ml_stats['Sharpe Ratio']:.4f} {eth_ml_stats['Max. Drawdown [%]']:>7.1f}%")
    print(f"  {'TF Portfolio Rotation':<35} {best_result['total_return']:>+9.1f}% {best_result['sharpe']:>7.4f} {best_result['max_dd']:>7.1f}%")
