"""Dashboard server: FastAPI backend + background data refresh."""

import asyncio
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backtesting import Backtest

import config
from data_fetcher import fetch_ohlcv
from indicators import add_indicators
from strategy_eth import add_eth_indicators
from strategy_robust import RobustTrendStrategy
from strategy_eth import ETHTrendStrategy
from strategy_portfolio import run_rotation_backtest
from dashboard_trades import build_timeline

warnings.filterwarnings("ignore")

app = FastAPI()
app.mount("/static", StaticFiles(directory="dashboard"), name="static")

DATA = {}
LOCK = threading.Lock()
REFRESH_INTERVAL = 1800  # 30 min


def compute_data():
    """Fetch data, run backtests, extract everything for the dashboard."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Refreshing data...")

    btc_raw = fetch_ohlcv("BTC/USDT", config.SIGNAL_TIMEFRAME, days=2400, use_cache=True)
    eth_raw = fetch_ohlcv("ETH/USDT", config.SIGNAL_TIMEFRAME, days=2400, use_cache=True)
    btc_df = add_indicators(btc_raw)
    eth_df = add_eth_indicators(eth_raw)

    btc_stats = Backtest(btc_df, RobustTrendStrategy, cash=config.BACKTEST_CASH,
                         commission=config.BACKTEST_COMMISSION, exclusive_orders=True,
                         trade_on_close=True).run()
    eth_stats = Backtest(eth_df, ETHTrendStrategy, cash=config.BACKTEST_CASH,
                         commission=config.BACKTEST_COMMISSION, exclusive_orders=True,
                         trade_on_close=True).run()

    btc_equity = btc_stats["_equity_curve"]["Equity"]
    eth_equity = eth_stats["_equity_curve"]["Equity"]
    rotation = run_rotation_backtest(btc_equity, eth_equity,
                                     lookback=120, strong_weight=0.70,
                                     rebal_every=30, rebal_cost=0.001)

    # Downsample to daily for chart
    btc_d = btc_df.resample("1D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    eth_d = eth_df.resample("1D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()

    def ts(idx): return [int(d.timestamp() * 1000) for d in idx]

    # EMA overlays
    btc_ema50 = btc_df[f"EMA_{config.EMA_MID}"].resample("1D").last().dropna()
    btc_ema200 = btc_df[f"EMA_{config.EMA_SLOW}"].resample("1D").last().dropna()
    eth_ema40 = eth_df["EMA_40"].resample("1D").last().dropna()
    eth_ema100 = eth_df["EMA_100"].resample("1D").last().dropna()

    # Trades (completed + open positions)
    btc_trades = _extract_trades(btc_stats, "BTC", None, None)
    eth_trades = _extract_trades(eth_stats, "ETH", None, None)
    btc_strat = btc_stats._strategy if hasattr(btc_stats, "_strategy") else None
    eth_strat = eth_stats._strategy if hasattr(eth_stats, "_strategy") else None
    _add_open_positions(btc_trades, btc_stats, "BTC", btc_df, btc_strat)
    _add_open_positions(eth_trades, eth_stats, "ETH", eth_df, eth_strat)

    # Rebalances
    rebalances = _extract_rebalances(rotation["weights_btc"], btc_trades, eth_trades, btc_df["Close"], eth_df["Close"], rotation["equity"])

    # Build unified timeline (pass per-strategy equity for correct % calculation)
    timeline = build_timeline(btc_trades, eth_trades, rebalances,
                              rotation["equity"], rotation["weights_btc"],
                              btc_equity, eth_equity)

    # Equity curves (normalized to 100)
    eq_dates = ts(btc_equity.index)
    btc_eq_n = (btc_equity / btc_equity.iloc[0] * 100).round(2).tolist()
    eth_eq_n = (eth_equity / eth_equity.iloc[0] * 100).round(2).tolist()
    port_eq_n = (rotation["equity"] * 100).round(2).tolist()

    # B&H price lines (normalized to 100)
    btc_close = btc_df["Close"].reindex(btc_equity.index, method="ffill")
    eth_close = eth_df["Close"].reindex(btc_equity.index, method="ffill")
    btc_bh_n = (btc_close / btc_close.iloc[0] * 100).round(2).tolist()
    eth_bh_n = (eth_close / eth_close.iloc[0] * 100).round(2).tolist()

    # Current state
    bl, el = btc_df.iloc[-1], eth_df.iloc[-1]
    btc_gc = bool(bl[f"EMA_{config.EMA_MID}"] > bl[f"EMA_{config.EMA_SLOW}"])
    eth_gc = bool(el["EMA_40"] > el["EMA_100"])
    btc_mom = round((btc_df["Close"].iloc[-1] / btc_df["Close"].iloc[-120] - 1) * 100, 1)
    eth_mom = round((eth_df["Close"].iloc[-1] / eth_df["Close"].iloc[-120] - 1) * 100, 1)

    result = {
        "last_update": datetime.now(timezone.utc).isoformat(),
        "btc_ohlc": {"t": ts(btc_d.index), "o": btc_d["Open"].round(2).tolist(),
                      "h": btc_d["High"].round(2).tolist(), "l": btc_d["Low"].round(2).tolist(),
                      "c": btc_d["Close"].round(2).tolist()},
        "eth_ohlc": {"t": ts(eth_d.index), "o": eth_d["Open"].round(2).tolist(),
                      "h": eth_d["High"].round(2).tolist(), "l": eth_d["Low"].round(2).tolist(),
                      "c": eth_d["Close"].round(2).tolist()},
        "btc_ema": {"ema50_t": ts(btc_ema50.index), "ema50": btc_ema50.round(2).tolist(),
                     "ema200_t": ts(btc_ema200.index), "ema200": btc_ema200.round(2).tolist()},
        "eth_ema": {"ema40_t": ts(eth_ema40.index), "ema40": eth_ema40.round(2).tolist(),
                     "ema100_t": ts(eth_ema100.index), "ema100": eth_ema100.round(2).tolist()},
        "trades_btc": btc_trades,
        "trades_eth": eth_trades,
        "timeline": timeline,
        "equity": {"t": eq_dates, "btc": btc_eq_n, "eth": eth_eq_n, "portfolio": port_eq_n,
                   "btc_bh": btc_bh_n, "eth_bh": eth_bh_n},
        "current": {
            "btc_price": round(bl["Close"], 2), "eth_price": round(el["Close"], 2),
            "btc_rsi": round(bl[f"RSI_{config.RSI_PERIOD}"], 1),
            "eth_rsi": round(el[f"RSI_{config.RSI_PERIOD}"], 1),
            "btc_regime": "BULL" if btc_gc else "BEAR",
            "eth_regime": "BULL" if eth_gc else "BEAR",
            "btc_momentum": btc_mom, "eth_momentum": eth_mom,
            "btc_in_trade": bool(_is_in_trade(btc_stats)),
            "eth_in_trade": bool(_is_in_trade(eth_stats)),
        },
        "stats": {
            "btc_ret": round(btc_stats["Return [%]"], 1),
            "eth_ret": round(eth_stats["Return [%]"], 1),
            "btc_bh": round(btc_stats["Buy & Hold Return [%]"], 1),
            "eth_bh": round(eth_stats["Buy & Hold Return [%]"], 1),
            "btc_sharpe": round(btc_stats["Sharpe Ratio"], 2) if btc_stats["Sharpe Ratio"] else None,
            "eth_sharpe": round(eth_stats["Sharpe Ratio"], 2) if eth_stats["Sharpe Ratio"] else None,
            "btc_dd": round(btc_stats["Max. Drawdown [%]"], 1),
            "eth_dd": round(eth_stats["Max. Drawdown [%]"], 1),
            "btc_trades": btc_stats["# Trades"], "eth_trades": eth_stats["# Trades"],
            "port_ret": rotation["total_return"], "port_sharpe": rotation["sharpe"],
            "port_dd": rotation["max_dd"],
        },
    }

    with LOCK:
        DATA.update(result)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Data ready.")


def _is_in_trade(stats):
    """Check if the strategy has an open position at the end of data.

    Detects by comparing equity at last completed trade exit vs final equity.
    If they differ significantly, there's an open position.
    """
    trades = stats._trades
    equity = stats["_equity_curve"]["Equity"]
    if len(trades) == 0:
        return False
    last_exit = trades.iloc[-1]["ExitTime"]
    eq_at_exit = equity.loc[:last_exit].iloc[-1]
    eq_final = equity.iloc[-1]
    return abs(eq_final - eq_at_exit) > 100


def _add_open_positions(trades_list, stats, symbol, df, strategy_instance):
    """Detect and add open positions that backtesting.py hasn't closed yet."""
    equity = stats["_equity_curve"]["Equity"]
    completed_trades = stats._trades

    if len(completed_trades) == 0:
        return

    last_exit = completed_trades.iloc[-1]["ExitTime"]
    eq_at_exit = float(equity.loc[:last_exit].iloc[-1])
    eq_final = float(equity.iloc[-1])

    if abs(eq_final - eq_at_exit) <= 10000:
        return

    # Find the stable (flat) equity value after last trade settles
    eq_after = equity.loc[last_exit:]
    # Skip first few bars (settlement period), then find the flat value
    settle_start = min(6, len(eq_after) - 1)
    flat_value = float(eq_after.iloc[settle_start])

    # Now find the first bar where equity diverges from this flat value
    entry_time = None
    for i in range(settle_start + 1, len(eq_after)):
        if abs(float(eq_after.iloc[i]) - flat_value) > 10000:
            entry_time = eq_after.index[i]
            break

    # Re-check: is there really an open position (using flat value)?
    if abs(eq_final - flat_value) <= 10000:
        return

    if entry_time is None:
        return

    # Entry price = close at entry bar
    entry_price = float(df["Close"].asof(entry_time))
    current_price = float(df["Close"].iloc[-1])
    entry_ms = int(entry_time.timestamp() * 1000)
    now_ms = int(df.index[-1].timestamp() * 1000)

    # Get SL from strategy's entry_log (the last entry is the open position)
    sl_price = None
    if strategy_instance and hasattr(strategy_instance, "entry_log") and strategy_instance.entry_log:
        last_entry = strategy_instance.entry_log[-1]
        sl_price = round(last_entry["sl"], 2)
        entry_price = round(last_entry["price"], 2)  # Use exact price from strategy

    price_change_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
    equity_change = eq_final - flat_value
    estimated_value = abs(equity_change / price_change_pct) if abs(price_change_pct) > 0.001 else 0

    trades_list.append({
        "entry_time": entry_ms,
        "exit_time": now_ms,
        "entry_price": round(entry_price, 2),
        "exit_price": round(current_price, 2),
        "size": 0,
        "entry_value": round(estimated_value, 2),
        "exit_value": round(estimated_value * (1 + price_change_pct), 2),
        "pnl": round(equity_change, 2),
        "return_pct": round(price_change_pct * 100, 2),
        "sl": sl_price,
        "sl_triggered": False,
        "symbol": symbol,
        "is_open": True,
    })


def _extract_trades(stats, symbol, all_btc_trades_df, all_eth_trades_df):
    """Extract trades with actual dollar amounts."""
    trades = []
    for _, t in stats._trades.iterrows():
        entry_ms = int(t["EntryTime"].timestamp() * 1000)
        exit_ms = int(t["ExitTime"].timestamp() * 1000)
        size = abs(int(t["Size"]))  # Number of units
        entry_value = round(size * float(t["EntryPrice"]), 2)  # $ spent to buy
        exit_value = round(size * float(t["ExitPrice"]), 2)    # $ received on sell
        sl_price = round(float(t["SL"]), 2) if pd.notna(t["SL"]) else None
        exit_price = round(float(t["ExitPrice"]), 2)
        # Detect if exit was triggered by stop loss
        sl_triggered = False
        if sl_price is not None and exit_price <= sl_price * 1.005:
            sl_triggered = True

        trades.append({
            "entry_time": entry_ms,
            "exit_time": exit_ms,
            "entry_price": round(float(t["EntryPrice"]), 2),
            "exit_price": exit_price,
            "size": size,
            "entry_value": entry_value,
            "exit_value": exit_value,
            "pnl": round(float(t["PnL"]), 2),
            "return_pct": round(float(t["ReturnPct"]) * 100, 2),
            "sl": sl_price,
            "sl_triggered": sl_triggered,
            "symbol": symbol,
        })
    return trades


def _add_allocation_to_trades(trades_list, all_btc_trades, all_eth_trades, rotation_weights):
    """Add post-action allocation snapshot to each trade event.

    For BUY: the bought asset is now IN position (even if the trade just started).
    For SELL: the sold asset is now OUT of position (even though trade record says exit_time = now).
    """
    def is_in_position_excluding(symbol_trades, time_ms, exclude_trade=None):
        """Check if there's an active trade, optionally excluding a specific one."""
        for t in symbol_trades:
            if exclude_trade and t is exclude_trade:
                continue
            if t["entry_time"] <= time_ms <= t["exit_time"]:
                return True
        return False

    weight_index = rotation_weights.index
    weight_values = rotation_weights.values

    def get_weight_at(time_ms):
        ts = pd.Timestamp(time_ms, unit="ms", tz="UTC")
        idx = weight_index.searchsorted(ts) - 1
        return float(weight_values[max(0, idx)])

    for t in trades_list:
        sym = t["symbol"]
        ms_entry = t["entry_time"]
        ms_exit = t["exit_time"]
        btc_w = get_weight_at(ms_entry)

        # ── AFTER BUY: this asset is now in position ──
        if sym == "BTC":
            btc_in = True  # we just bought it
            eth_in = is_in_position_excluding(all_eth_trades, ms_entry)
        else:
            eth_in = True  # we just bought it
            btc_in = is_in_position_excluding(all_btc_trades, ms_entry)

        btc_alloc = round(btc_w * 100) if btc_in else 0
        eth_alloc = round((1 - btc_w) * 100) if eth_in else 0
        t["entry_btc_pct"] = btc_alloc
        t["entry_eth_pct"] = eth_alloc
        t["entry_cash_pct"] = 100 - btc_alloc - eth_alloc
        t["entry_action_size"] = btc_alloc if sym == "BTC" else eth_alloc

        # ── AFTER SELL: this asset is now OUT of position ──
        btc_w = get_weight_at(ms_exit)
        if sym == "BTC":
            btc_in = False  # we just sold it
            eth_in = is_in_position_excluding(all_eth_trades, ms_exit)
        else:
            eth_in = False  # we just sold it
            btc_in = is_in_position_excluding(all_btc_trades, ms_exit)

        btc_alloc = round(btc_w * 100) if btc_in else 0
        eth_alloc = round((1 - btc_w) * 100) if eth_in else 0
        t["exit_btc_pct"] = btc_alloc
        t["exit_eth_pct"] = eth_alloc
        t["exit_cash_pct"] = 100 - btc_alloc - eth_alloc
        t["exit_action_size"] = round(btc_w * 100) if sym == "BTC" else round((1 - btc_w) * 100)


def _add_position_legs(trades_list, rebalances):
    """For each trade, find all rebalances during its lifetime that changed this asset's weight.

    Builds a 'legs' list on each trade showing how the position was constructed:
    - First leg: the initial BUY
    - Subsequent legs: rebalances that added to or reduced the position
    """
    for t in trades_list:
        sym = t["symbol"]
        legs = [{
            "action": "BUY",
            "time": t["entry_time"],
            "price": t["entry_price"],
            "pct": t.get("entry_action_size", 0),
        }]

        # Find rebalances during this trade's lifetime
        for r in rebalances:
            if r["time"] <= t["entry_time"] or r["time"] >= t["exit_time"]:
                continue
            # Did this rebalance change our asset's weight?
            if sym == "BTC" and r["buy_asset"] == "BTC":
                legs.append({
                    "action": "REBAL +",
                    "time": r["time"],
                    "price": r["buy_price"],
                    "pct": r["buy_to"] - r["buy_from"],
                })
            elif sym == "BTC" and r["sell_asset"] == "BTC":
                legs.append({
                    "action": "REBAL -",
                    "time": r["time"],
                    "price": r["sell_price"],
                    "pct": -(r["sell_from"] - r["sell_to"]),
                })
            elif sym == "ETH" and r["buy_asset"] == "ETH":
                legs.append({
                    "action": "REBAL +",
                    "time": r["time"],
                    "price": r["buy_price"],
                    "pct": r["buy_to"] - r["buy_from"],
                })
            elif sym == "ETH" and r["sell_asset"] == "ETH":
                legs.append({
                    "action": "REBAL -",
                    "time": r["time"],
                    "price": r["sell_price"],
                    "pct": -(r["sell_from"] - r["sell_to"]),
                })

        t["legs"] = legs
        # Running total: initial buy + all rebalance changes = position at exit
        t["total_position_pct"] = sum(l["pct"] for l in legs)


def _add_nav_to_trades(trades_list, portfolio_equity):
    """Add NAV (net asset value, base=1.0) and fund total return at each trade time."""
    eq_index = portfolio_equity.index
    eq_values = portfolio_equity.values  # already normalized so that start = 1.0

    def nav_at(time_ms):
        ts = pd.Timestamp(time_ms, unit="ms", tz="UTC")
        idx = eq_index.searchsorted(ts) - 1
        if idx < 0:
            return 1.0
        return float(eq_values[max(0, idx)])

    for t in trades_list:
        entry_nav = nav_at(t["entry_time"])
        exit_nav = nav_at(t["exit_time"])
        t["entry_nav"] = round(entry_nav, 4)
        t["exit_nav"] = round(exit_nav, 4)
        t["entry_fund_return"] = round((entry_nav - 1) * 100, 2)
        t["exit_fund_return"] = round((exit_nav - 1) * 100, 2)


def _extract_rebalances(weights, btc_trades, eth_trades, btc_prices, eth_prices, portfolio_equity=None):
    """Extract meaningful rebalance events (only when both assets have positions).

    Each rebalance means: sell some of the overweight asset, buy the underweight.
    We compute the amounts and prices at rebalance time.
    """
    def is_in_pos(symbol_trades, time_ms):
        for t in symbol_trades:
            if t["entry_time"] <= time_ms <= t["exit_time"]:
                return True
        return False

    def get_price(prices_df, time_ms):
        ts = pd.Timestamp(time_ms, unit="ms", tz="UTC")
        idx = prices_df.index.searchsorted(ts) - 1
        if idx < 0:
            idx = 0
        return float(prices_df.iloc[idx])

    changes = []
    prev_w = 0.5
    for i in range(len(weights)):
        w = float(weights.iloc[i])
        if abs(w - prev_w) > 0.01:
            t_ms = int(weights.index[i].timestamp() * 1000)
            btc_in = is_in_pos(btc_trades, t_ms)
            eth_in = is_in_pos(eth_trades, t_ms)

            # Only emit rebalance when BOTH have positions
            if not (btc_in and eth_in):
                prev_w = w
                continue

            btc_new = round(w * 100)
            eth_new = round((1 - w) * 100)
            btc_old = round(prev_w * 100)
            eth_old = round((1 - prev_w) * 100)

            btc_price = get_price(btc_prices, t_ms)
            eth_price = get_price(eth_prices, t_ms)

            # What changed: delta in allocation %
            btc_delta = btc_new - btc_old  # positive = buying more BTC
            # Describe the action: e.g. "Sell ETH 40% → 30%, Buy BTC 60% → 70%"
            if btc_delta > 0:
                sell_asset = "ETH"
                buy_asset = "BTC"
                sell_price = eth_price
                buy_price = btc_price
            else:
                sell_asset = "BTC"
                buy_asset = "ETH"
                sell_price = btc_price
                buy_price = eth_price

            nav = 1.0
            fund_ret = 0.0
            if portfolio_equity is not None:
                ts_pd = pd.Timestamp(t_ms, unit="ms", tz="UTC")
                idx = portfolio_equity.index.searchsorted(ts_pd) - 1
                if idx >= 0:
                    nav = float(portfolio_equity.iloc[max(0, idx)])
                    fund_ret = round((nav - 1) * 100, 2)
                nav = round(nav, 4)

            changes.append({
                "time": t_ms,
                "btc_pct": btc_new, "eth_pct": eth_new, "cash_pct": 0,
                "sell_asset": sell_asset, "sell_price": round(sell_price, 2),
                "buy_asset": buy_asset, "buy_price": round(buy_price, 2),
                "sell_from": eth_old if sell_asset == "ETH" else btc_old,
                "sell_to": eth_new if sell_asset == "ETH" else btc_new,
                "buy_from": btc_old if buy_asset == "BTC" else eth_old,
                "buy_to": btc_new if buy_asset == "BTC" else eth_new,
                "nav": nav, "fund_return": fund_ret,
            })
            prev_w = w
    return changes


def _extract_weight_changes(weights, btc_trades, eth_trades):
    """Extract rebalancing events with position status."""
    changes = []
    prev_w = 0.5
    for i in range(len(weights)):
        w = float(weights.iloc[i])
        if abs(w - prev_w) > 0.01:
            t_ms = int(weights.index[i].timestamp() * 1000)
            # Check if BTC/ETH strategies are in a trade at this time
            btc_in = any(tr["entry_time"] <= t_ms <= tr["exit_time"] for tr in btc_trades)
            eth_in = any(tr["entry_time"] <= t_ms <= tr["exit_time"] for tr in eth_trades)
            changes.append({
                "t": t_ms,
                "btc_w": round(w * 100), "eth_w": round((1 - w) * 100),
                "btc_pos": "LONG" if btc_in else "CASH",
                "eth_pos": "LONG" if eth_in else "CASH",
            })
            prev_w = w
    return changes


def _refresh_loop():
    while True:
        try:
            compute_data()
        except Exception as e:
            print(f"Refresh error: {e}")
        time.sleep(REFRESH_INTERVAL)


@app.on_event("startup")
def startup():
    compute_data()
    threading.Thread(target=_refresh_loop, daemon=True).start()


@app.get("/")
def index():
    return FileResponse("dashboard/index.html")


@app.get("/api/data")
def api_data():
    with LOCK:
        return DATA


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard:app", host="0.0.0.0", port=8050, reload=False)
