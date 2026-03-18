"""Comprehensive backtest + walk-forward for enhanced strategy."""

import warnings
import numpy as np
from backtesting import Backtest

import config
from data_fetcher import fetch_ohlcv
from indicators import add_indicators
from strategy_eth import add_eth_indicators
from external_data import merge_external_data, fetch_fear_greed, fetch_funding_rates
from strategy_enhanced import EnhancedBTCStrategy, EnhancedETHStrategy
from strategy_robust import RobustTrendStrategy
from strategy_eth import ETHTrendStrategy

from ta.trend import EMAIndicator

warnings.filterwarnings("ignore")

CANDLES_PER_MONTH = 180


def prepare_btc(days=1095):
    """Prepare BTC data with indicators + external data."""
    df = add_indicators(fetch_ohlcv("BTC/USDT", config.SIGNAL_TIMEFRAME, days=days))
    df = merge_external_data(df, "BTC/USDT")
    return df


def prepare_eth(days=1095):
    """Prepare ETH data with indicators + external data + BTC trend signal."""
    df = add_eth_indicators(fetch_ohlcv("ETH/USDT", config.SIGNAL_TIMEFRAME, days=days))
    df = merge_external_data(df, "ETH/USDT")

    # Add BTC trend signal for ETH
    btc = add_indicators(fetch_ohlcv("BTC/USDT", config.SIGNAL_TIMEFRAME, days=days))
    btc_trend = (btc[f"EMA_{config.EMA_MID}"] > btc[f"EMA_{config.EMA_SLOW}"]).astype(int)
    btc_trend = btc_trend.reindex(df.index, method="ffill").fillna(1)
    df["BTC_Trend"] = btc_trend
    return df


def run_bt(df, strategy_cls, **params):
    """Run a single backtest, return stats."""
    bt = Backtest(df, strategy_cls, cash=config.BACKTEST_CASH,
                  commission=config.BACKTEST_COMMISSION, exclusive_orders=True,
                  trade_on_close=True)
    return bt.run(**params)


def print_comparison(label, old_stats, new_stats):
    """Print side-by-side comparison."""
    def g(s, k):
        v = s[k]
        if v is None:
            return "N/A"
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    print(f"\n  {label}:")
    print(f"  {'':>20} {'OLD':>12} {'ENHANCED':>12} {'Change':>10}")
    print(f"  {'─'*58}")
    for key, fmt in [("Return [%]", ".1f"), ("Buy & Hold Return [%]", ".1f"),
                      ("Sharpe Ratio", ".4f"), ("Max. Drawdown [%]", ".1f"),
                      ("Win Rate [%]", ".1f"), ("# Trades", "d"),
                      ("Profit Factor", ".2f")]:
        ov = old_stats[key]
        nv = new_stats[key]
        if ov is not None and nv is not None and isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
            if fmt == "d":
                diff = f"{nv - ov:+d}"
                print(f"  {key:>20} {ov:>12d} {nv:>12d} {diff:>10}")
            else:
                diff = f"{nv - ov:+{fmt}}"
                print(f"  {key:>20} {ov:>12{fmt}} {nv:>12{fmt}} {diff:>10}")
        else:
            print(f"  {key:>20} {g(old_stats, key):>12} {g(new_stats, key):>12}")


def quarterly_breakdown(df, strategy_cls, label):
    """Run per-quarter comparison."""
    total = len(df)
    quarter_size = 540  # ~3 months of 4H
    results = []
    start = 0

    while start + quarter_size <= total:
        df_q = df.iloc[start:min(start + quarter_size, total)]
        if len(df_q) < 100:
            break
        try:
            s = run_bt(df_q, strategy_cls)
            ret = s["Return [%]"]
            bh = s["Buy & Hold Return [%]"]
            results.append({
                "period": f"{df_q.index[0].strftime('%Y-%m')} → {df_q.index[-1].strftime('%Y-%m')}",
                "ret": round(ret, 1), "bh": round(bh, 1), "alpha": round(ret - bh, 1),
                "sharpe": round(s["Sharpe Ratio"], 2) if s["Sharpe Ratio"] is not None else None,
                "dd": round(s["Max. Drawdown [%]"], 1),
                "trades": s["# Trades"],
                "market": "BULL" if bh > 5 else ("BEAR" if bh < -5 else "FLAT"),
            })
        except Exception:
            pass
        start += quarter_size

    print(f"\n  QUARTERLY: {label}")
    print(f"  {'Period':<20} {'Mkt':>4} {'Strat':>7} {'B&H':>7} {'Alpha':>7} {'Sh':>6} {'DD':>6} {'Tr':>3}")
    print(f"  {'─'*68}")
    for r in results:
        sh = f"{r['sharpe']:.2f}" if r["sharpe"] is not None else "N/A"
        print(f"  {r['period']:<20} {r['market']:>4} {r['ret']:>+6.1f}% {r['bh']:>+6.1f}% {r['alpha']:>+6.1f}% {sh:>6} {r['dd']:>5.1f}% {r['trades']:>3}")

    bull_a = [r["alpha"] for r in results if r["market"] == "BULL"]
    bear_a = [r["alpha"] for r in results if r["market"] == "BEAR"]
    flat_a = [r["alpha"] for r in results if r["market"] == "FLAT"]
    print(f"  {'─'*68}")
    if bull_a: print(f"  BULL ({len(bull_a)}): mean alpha = {np.mean(bull_a):+.1f}%")
    if bear_a: print(f"  BEAR ({len(bear_a)}): mean alpha = {np.mean(bear_a):+.1f}%")
    if flat_a: print(f"  FLAT ({len(flat_a)}): mean alpha = {np.mean(flat_a):+.1f}%")
    return results


def walk_forward(df, strategy_cls, label):
    """Walk-forward with fixed params."""
    total = len(df)
    train_sz, test_sz = 6 * CANDLES_PER_MONTH, 3 * CANDLES_PER_MONTH
    windows = []
    s = 0
    while s + train_sz + test_sz <= total:
        windows.append((s, s + train_sz, s + train_sz + test_sz))
        s += test_sz

    print(f"\n  WALK-FORWARD: {label} ({len(windows)} windows)")
    is_rets, oos_rets, oos_alphas = [], [], []

    for i, (ts, te, tte) in enumerate(windows):
        is_s = run_bt(df.iloc[ts:te], strategy_cls)
        oos_s = run_bt(df.iloc[te:tte], strategy_cls)
        bh = oos_s["Buy & Hold Return [%]"]
        is_rets.append(is_s["Return [%]"])
        oos_rets.append(oos_s["Return [%]"])
        oos_alphas.append(oos_s["Return [%]"] - bh)

        period = df.iloc[te:tte].index[0].strftime("%Y-%m") + " → " + df.iloc[te:tte].index[-1].strftime("%Y-%m")
        mkt = "BULL" if bh > 5 else ("BEAR" if bh < -5 else "FLAT")
        sh = oos_s["Sharpe Ratio"]
        sh_s = f"{sh:.2f}" if sh is not None else "N/A"
        print(f"  W{i+1} [{period}] {mkt:>4} IS={is_rets[-1]:>+6.1f}% OOS={oos_rets[-1]:>+6.1f}% B&H={bh:>+6.1f}% Alpha={oos_alphas[-1]:>+6.1f}% Sh={sh_s}")

    is_m, oos_m = np.mean(is_rets), np.mean(oos_rets)
    deg = (is_m - oos_m) / abs(is_m) * 100 if is_m != 0 else 0
    print(f"  ───")
    print(f"  IS={is_m:+.2f}%  OOS={oos_m:+.2f}% ({oos_m*4:+.0f}% ann)  Alpha={np.mean(oos_alphas):+.2f}%  Degrad={deg:.0f}%")
    print(f"  OOS profitable: {sum(1 for r in oos_rets if r > 0)}/{len(oos_rets)}  Alpha+: {sum(1 for a in oos_alphas if a > 0)}/{len(oos_alphas)}")

    # Bull/bear split
    bull_is, bull_oos, bear_is, bear_oos = [], [], [], []
    for i, (ts, te, tte) in enumerate(windows):
        oos_s = run_bt(df.iloc[te:tte], strategy_cls)
        bh = oos_s["Buy & Hold Return [%]"]
        if bh > 5:
            bull_is.append(is_rets[i])
            bull_oos.append(oos_rets[i])
        elif bh < -5:
            bear_is.append(is_rets[i])
            bear_oos.append(oos_rets[i])

    if bull_is:
        d = (np.mean(bull_is) - np.mean(bull_oos)) / abs(np.mean(bull_is)) * 100 if np.mean(bull_is) != 0 else 0
        print(f"  BULL degrad: {d:.0f}%  (IS={np.mean(bull_is):+.1f}% OOS={np.mean(bull_oos):+.1f}%)")
    if bear_is:
        d = (np.mean(bear_is) - np.mean(bear_oos)) / abs(np.mean(bear_is)) * 100 if np.mean(bear_is) != 0 else 0
        print(f"  BEAR degrad: {d:.0f}%  (IS={np.mean(bear_is):+.1f}% OOS={np.mean(bear_oos):+.1f}%)")

    verdict = "LOW" if abs(deg) < 30 else ("MODERATE" if abs(deg) < 50 else "HIGH")
    print(f"  Verdict: {verdict} overfitting risk")


if __name__ == "__main__":
    print("=" * 70)
    print("  ENHANCED STRATEGY COMPREHENSIVE TEST")
    print("=" * 70)

    # ── 1. Prepare data ──
    print("\n[1/4] Preparing data with external sources...")
    df_btc = prepare_btc()
    print(f"  BTC: {len(df_btc)} candles, FNG range: {df_btc['FNG'].min():.0f}-{df_btc['FNG'].max():.0f}")

    df_eth = prepare_eth()
    print(f"  ETH: {len(df_eth)} candles, BTC_Trend present: {'BTC_Trend' in df_eth.columns}")

    # ── 2. Full-period comparison ──
    print("\n[2/4] Full-period backtest: OLD vs ENHANCED...")
    old_btc = run_bt(df_btc, RobustTrendStrategy)
    new_btc = run_bt(df_btc, EnhancedBTCStrategy)
    print_comparison("BTC/USDT", old_btc, new_btc)

    old_eth = run_bt(df_eth, ETHTrendStrategy)
    new_eth = run_bt(df_eth, EnhancedETHStrategy)
    print_comparison("ETH/USDT", old_eth, new_eth)

    # ── 3. Quarterly breakdown ──
    print("\n[3/4] Quarterly breakdown...")
    quarterly_breakdown(df_btc, EnhancedBTCStrategy, "BTC Enhanced")
    quarterly_breakdown(df_eth, EnhancedETHStrategy, "ETH Enhanced")

    # ── 4. Walk-forward ──
    print("\n[4/4] Walk-forward validation...")
    walk_forward(df_btc, EnhancedBTCStrategy, "BTC Enhanced")
    walk_forward(df_eth, EnhancedETHStrategy, "ETH Enhanced")
