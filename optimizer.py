"""Automated strategy optimization — multi-round grid search."""

import itertools
import warnings
from backtesting import Backtest

import config
from data_fetcher import fetch_ohlcv
from indicators import add_indicators
from strategy import ConfluenceStrategy

warnings.filterwarnings("ignore")


def _run_combo(df, params):
    """Run a single parameter combo, return metrics or None."""
    try:
        bt = Backtest(df, ConfluenceStrategy, cash=config.BACKTEST_CASH,
                      commission=config.BACKTEST_COMMISSION, exclusive_orders=True, trade_on_close=True)
        stats = bt.run(**params)
        ret = round(stats["Return [%]"], 2)
        bh = round(stats["Buy & Hold Return [%]"], 2)
        sharpe = stats["Sharpe Ratio"]
        mdd = stats["Max. Drawdown [%]"]
        wr = stats["Win Rate [%]"]
        trades = stats["# Trades"]
        if trades < 5:
            return None
        return {
            "return_pct": ret, "buy_hold_pct": bh, "alpha": round(ret - bh, 2),
            "sharpe": round(sharpe, 4) if sharpe is not None else None,
            "max_dd": round(mdd, 2),
            "win_rate": round(wr, 2) if wr is not None else None,
            "trades": trades, "params": params,
        }
    except Exception:
        return None


def _grid_search(df, param_grid, label=""):
    """Run grid search, return sorted results."""
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    print(f"  {label} Testing {len(combos)} combinations...")

    results = []
    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        r = _run_combo(df, params)
        if r:
            results.append(r)
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{len(combos)}")

    results.sort(key=lambda x: x["alpha"], reverse=True)
    return results


def _print_top(results, n=5, label=""):
    """Print top N results."""
    print(f"\n  {label} TOP {min(n, len(results))} (by Alpha):")
    for rank, r in enumerate(results[:n], 1):
        print(f"  #{rank}  Ret: {r['return_pct']:>7.1f}%  B&H: {r['buy_hold_pct']:>7.1f}%  Alpha: {r['alpha']:>+7.1f}%  "
              f"Sharpe: {r['sharpe']}  DD: {r['max_dd']}%  WR: {r['win_rate']}%  Trades: {r['trades']}")
        print(f"       {r['params']}")


def optimize_multi_round(symbol="BTC/USDT", days=1095):
    """10-round optimization: progressively refining parameters."""
    print(f"\n{'#'*70}")
    print(f"  MULTI-ROUND OPTIMIZER: {symbol} | {days} days")
    print(f"{'#'*70}")

    print("\nFetching data...")
    df_raw = fetch_ohlcv(symbol, config.SIGNAL_TIMEFRAME, days=days)
    df = add_indicators(df_raw)
    print(f"Data ready: {len(df)} candles")

    # ── Round 1: Wide coarse search ──
    print(f"\n{'─'*70}")
    print(f"  ROUND 1: WIDE COARSE GRID")
    r1 = _grid_search(df, {
        "entry_bull_threshold": [2, 2.5, 3, 3.5, 4],
        "exit_bear_threshold": [3, 3.5, 4, 4.5, 5],
        "atr_sl_mult": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        "cooldown": [1, 2, 3, 4, 6],
        "use_trailing": [True],
    }, "R1")
    _print_top(r1, 5, "R1")
    best1 = r1[0]["params"]

    # ── Round 2: Fine grid around R1 best ──
    print(f"\n{'─'*70}")
    print(f"  ROUND 2: FINE GRID AROUND R1 BEST")
    e, x, a, c = best1["entry_bull_threshold"], best1["exit_bear_threshold"], best1["atr_sl_mult"], best1["cooldown"]
    r2 = _grid_search(df, {
        "entry_bull_threshold": sorted(set([max(1, e-0.5), e-0.25, e, e+0.25, e+0.5])),
        "exit_bear_threshold": sorted(set([max(1, x-0.5), x-0.25, x, x+0.25, x+0.5])),
        "atr_sl_mult": sorted(set([max(0.5, a-0.5), a-0.25, a, a+0.25, a+0.5])),
        "cooldown": sorted(set([max(1, c-1), c, c+1, c+2])),
        "use_trailing": [True],
    }, "R2")
    _print_top(r2, 5, "R2")
    best2 = r2[0]["params"]

    # ── Round 3: Ultra-fine around R2 best ──
    print(f"\n{'─'*70}")
    print(f"  ROUND 3: ULTRA-FINE AROUND R2 BEST")
    e, x, a = best2["entry_bull_threshold"], best2["exit_bear_threshold"], best2["atr_sl_mult"]
    r3 = _grid_search(df, {
        "entry_bull_threshold": sorted(set([max(0.5, e-0.25), e-0.1, e, e+0.1, e+0.25])),
        "exit_bear_threshold": sorted(set([max(0.5, x-0.25), x-0.1, x, x+0.1, x+0.25])),
        "atr_sl_mult": sorted(set([max(0.5, a-0.25), a-0.1, a, a+0.1, a+0.25])),
        "cooldown": sorted(set([max(1, best2["cooldown"]-1), best2["cooldown"], best2["cooldown"]+1])),
        "use_trailing": [True],
    }, "R3")
    _print_top(r3, 3, "R3")
    best3 = r3[0]["params"]

    # ── Round 4-5: Test indicator period variations ──
    print(f"\n{'─'*70}")
    print(f"  ROUND 4: TEST EMA PERIOD VARIATIONS")
    best_alpha = r3[0]["alpha"]
    best_overall = best3.copy()

    for ema_fast, ema_mid in [(10, 30), (15, 40), (20, 50), (25, 60), (30, 70)]:
        config.EMA_FAST = ema_fast
        config.EMA_MID = ema_mid
        df_test = add_indicators(df_raw)
        r = _run_combo(df_test, best3)
        if r:
            label = f"EMA({ema_fast},{ema_mid})"
            print(f"  {label:>12}: Ret={r['return_pct']:>7.1f}%  Alpha={r['alpha']:>+7.1f}%  DD={r['max_dd']}%")
            if r["alpha"] > best_alpha:
                best_alpha = r["alpha"]
                best_overall = best3.copy()
                best_overall["_ema_fast"] = ema_fast
                best_overall["_ema_mid"] = ema_mid

    ema_f = best_overall.pop("_ema_fast", 20)
    ema_m = best_overall.pop("_ema_mid", 50)
    config.EMA_FAST = ema_f
    config.EMA_MID = ema_m
    df = add_indicators(df_raw)
    print(f"  >>> Best EMAs: ({ema_f}, {ema_m})")

    # ── Round 5: Test RSI thresholds ──
    print(f"\n{'─'*70}")
    print(f"  ROUND 5: TEST RSI THRESHOLDS")
    for ob, os_ in [(65, 35), (70, 30), (75, 25), (80, 20)]:
        config.RSI_OVERBOUGHT = ob
        config.RSI_OVERSOLD = os_
        r = _run_combo(df, best_overall)
        if r:
            print(f"  RSI({os_}/{ob}): Ret={r['return_pct']:>7.1f}%  Alpha={r['alpha']:>+7.1f}%")
            if r["alpha"] > best_alpha:
                best_alpha = r["alpha"]
                config._best_rsi = (ob, os_)
    ob, os_ = getattr(config, "_best_rsi", (70, 30))
    config.RSI_OVERBOUGHT = ob
    config.RSI_OVERSOLD = os_
    print(f"  >>> Best RSI: oversold={os_}, overbought={ob}")

    # ── Round 6: ATR period ──
    print(f"\n{'─'*70}")
    print(f"  ROUND 6: TEST ATR PERIOD")
    for atr_p in [7, 10, 14, 20, 25]:
        config.ATR_PERIOD = atr_p
        df_test = add_indicators(df_raw)
        r = _run_combo(df_test, best_overall)
        if r:
            print(f"  ATR({atr_p}): Ret={r['return_pct']:>7.1f}%  Alpha={r['alpha']:>+7.1f}%")
            if r["alpha"] > best_alpha:
                best_alpha = r["alpha"]
                config._best_atr_p = atr_p
    config.ATR_PERIOD = getattr(config, "_best_atr_p", 14)
    df = add_indicators(df_raw)
    print(f"  >>> Best ATR period: {config.ATR_PERIOD}")

    # ── Round 7: BB parameters ──
    print(f"\n{'─'*70}")
    print(f"  ROUND 7: TEST BOLLINGER BAND PARAMS")
    for bb_p, bb_s in [(15, 1.5), (20, 2.0), (20, 2.5), (25, 2.0), (30, 2.0)]:
        config.BB_PERIOD = bb_p
        config.BB_STD = bb_s
        df_test = add_indicators(df_raw)
        r = _run_combo(df_test, best_overall)
        if r:
            print(f"  BB({bb_p},{bb_s}): Ret={r['return_pct']:>7.1f}%  Alpha={r['alpha']:>+7.1f}%")
            if r["alpha"] > best_alpha:
                best_alpha = r["alpha"]
                config._best_bb = (bb_p, bb_s)
    bb_p, bb_s = getattr(config, "_best_bb", (20, 2.0))
    config.BB_PERIOD = bb_p
    config.BB_STD = bb_s
    df = add_indicators(df_raw)
    print(f"  >>> Best BB: period={bb_p}, std={bb_s}")

    # ── Round 8: Final fine-tune with all optimized indicators ──
    print(f"\n{'─'*70}")
    print(f"  ROUND 8: FINAL FINE-TUNE WITH OPTIMIZED INDICATORS")
    e = best_overall["entry_bull_threshold"]
    a = best_overall["atr_sl_mult"]
    r8 = _grid_search(df, {
        "entry_bull_threshold": sorted(set([max(0.5, e-0.5), e-0.25, e, e+0.25, e+0.5])),
        "exit_bear_threshold": [3, 3.5, 4, 4.5, 5, 5.5],
        "atr_sl_mult": sorted(set([max(0.5, a-0.5), a-0.25, a, a+0.25, a+0.5, a+1.0])),
        "cooldown": [1, 2, 3, 4, 5, 6],
        "use_trailing": [True],
    }, "R8")
    _print_top(r8, 5, "R8")
    best8 = r8[0]["params"]

    # ── Round 9: BTC final verification ──
    print(f"\n{'─'*70}")
    print(f"  ROUND 9: BTC FINAL VERIFICATION")
    r_btc = _run_combo(df, best8)
    print(f"\n  BTC/USDT FINAL:")
    print(f"  Return: {r_btc['return_pct']:.2f}%  |  B&H: {r_btc['buy_hold_pct']:.2f}%  |  Alpha: {r_btc['alpha']:+.2f}%")
    print(f"  Sharpe: {r_btc['sharpe']}  |  MaxDD: {r_btc['max_dd']}%  |  WR: {r_btc['win_rate']}%  |  Trades: {r_btc['trades']}")

    # ── Round 10: ETH cross-validation ──
    print(f"\n{'─'*70}")
    print(f"  ROUND 10: ETH CROSS-VALIDATION")
    df_eth_raw = fetch_ohlcv("ETH/USDT", config.SIGNAL_TIMEFRAME, days=days)
    df_eth = add_indicators(df_eth_raw)
    r_eth = _run_combo(df_eth, best8)
    print(f"\n  ETH/USDT CROSS-VALIDATION:")
    print(f"  Return: {r_eth['return_pct']:.2f}%  |  B&H: {r_eth['buy_hold_pct']:.2f}%  |  Alpha: {r_eth['alpha']:+.2f}%")
    print(f"  Sharpe: {r_eth['sharpe']}  |  MaxDD: {r_eth['max_dd']}%  |  WR: {r_eth['win_rate']}%  |  Trades: {r_eth['trades']}")

    # ── Summary ──
    print(f"\n{'#'*70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'#'*70}")
    print(f"\n  Strategy Params: {best8}")
    print(f"  Indicator Config: EMA({config.EMA_FAST},{config.EMA_MID},{config.EMA_SLOW}) RSI({config.RSI_OVERSOLD}/{config.RSI_OVERBOUGHT}) ATR({config.ATR_PERIOD}) BB({config.BB_PERIOD},{config.BB_STD})")
    print(f"\n  BTC: {r_btc['return_pct']:+.2f}% vs B&H {r_btc['buy_hold_pct']:+.2f}%  (Alpha {r_btc['alpha']:+.2f}%)")
    print(f"  ETH: {r_eth['return_pct']:+.2f}% vs B&H {r_eth['buy_hold_pct']:+.2f}%  (Alpha {r_eth['alpha']:+.2f}%)")

    return best8, r_btc, r_eth


if __name__ == "__main__":
    optimize_multi_round()
