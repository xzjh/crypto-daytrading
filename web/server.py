"""Dashboard server: FastAPI backend + background data refresh."""

import threading
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backtesting import Backtest

from core import config
from core.data_fetcher import fetch_ohlcv
from core.indicators import add_indicators
from strategies.eth import add_eth_indicators
from strategies.robust import RobustTrendStrategy
from strategies.eth import ETHTrendStrategy
from web.trades import build_timeline

warnings.filterwarnings("ignore")

app = FastAPI()
app.mount("/static", StaticFiles(directory="web/static"), name="static")

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
                         commission=0.0018, exclusive_orders=False,
                         trade_on_close=True, margin=1/1.3).run()
    eth_stats = Backtest(eth_df, ETHTrendStrategy, cash=config.BACKTEST_CASH,
                         commission=0.0018, exclusive_orders=False,
                         trade_on_close=True, margin=1/1.3).run()

    btc_equity = btc_stats["_equity_curve"]["Equity"]
    eth_equity = eth_stats["_equity_curve"]["Equity"]

    def ts(idx): return [int(d.timestamp() * 1000) for d in idx]

    def resample_ohlc(df, rule):
        r = df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        return {"t": ts(r.index), "o": r["Open"].round(2).tolist(), "h": r["High"].round(2).tolist(),
                "l": r["Low"].round(2).tolist(), "c": r["Close"].round(2).tolist()}

    def resample_ema(series, rule):
        r = series.resample(rule).last().dropna()
        return {"t": ts(r.index), "v": r.round(2).tolist()}

    # Multi-timeframe OHLCV + EMAs
    tf_rules = {"4h": None, "1d": "1D", "1w": "1W", "1m": "1MS"}
    btc_ohlc_tf, eth_ohlc_tf = {}, {}
    btc_ema_tf, eth_ema_tf = {}, {}

    for tf_key, rule in tf_rules.items():
        if rule is None:
            btc_ohlc_tf[tf_key] = {"t": ts(btc_df.index), "o": btc_df["Open"].round(2).tolist(),
                                    "h": btc_df["High"].round(2).tolist(), "l": btc_df["Low"].round(2).tolist(),
                                    "c": btc_df["Close"].round(2).tolist()}
            eth_ohlc_tf[tf_key] = {"t": ts(eth_df.index), "o": eth_df["Open"].round(2).tolist(),
                                    "h": eth_df["High"].round(2).tolist(), "l": eth_df["Low"].round(2).tolist(),
                                    "c": eth_df["Close"].round(2).tolist()}
            btc_ema_tf[tf_key] = {
                "ema50": {"t": ts(btc_df.index), "v": btc_df[f"EMA_{config.EMA_MID}"].round(2).tolist()},
                "ema200": {"t": ts(btc_df.index), "v": btc_df[f"EMA_{config.EMA_SLOW}"].round(2).tolist()},
            }
            eth_ema_tf[tf_key] = {
                "ema40": {"t": ts(eth_df.index), "v": eth_df["EMA_40"].round(2).tolist()},
                "ema100": {"t": ts(eth_df.index), "v": eth_df["EMA_100"].round(2).tolist()},
            }
        else:
            btc_ohlc_tf[tf_key] = resample_ohlc(btc_df, rule)
            eth_ohlc_tf[tf_key] = resample_ohlc(eth_df, rule)
            btc_ema_tf[tf_key] = {
                "ema50": resample_ema(btc_df[f"EMA_{config.EMA_MID}"], rule),
                "ema200": resample_ema(btc_df[f"EMA_{config.EMA_SLOW}"], rule),
            }
            eth_ema_tf[tf_key] = {
                "ema40": resample_ema(eth_df["EMA_40"], rule),
                "ema100": resample_ema(eth_df["EMA_100"], rule),
            }

    # Trades (completed + open positions)
    btc_trades = _extract_trades(btc_stats, "BTC")
    eth_trades = _extract_trades(eth_stats, "ETH")
    btc_strat = btc_stats._strategy if hasattr(btc_stats, "_strategy") else None
    eth_strat = eth_stats._strategy if hasattr(eth_stats, "_strategy") else None
    _add_open_positions(btc_trades, btc_stats, "BTC", btc_df, btc_strat)
    _add_open_positions(eth_trades, eth_stats, "ETH", eth_df, eth_strat)

    # Build unified timeline
    timeline = build_timeline(btc_trades, eth_trades, btc_equity, eth_equity)

    # Equity curves (normalized to 100)
    eq_dates = ts(btc_equity.index)
    btc_eq_n = (btc_equity / btc_equity.iloc[0] * 100).round(2).tolist()
    eth_eq_reindexed = eth_equity.reindex(btc_equity.index, method="ffill")
    eth_eq_n = (eth_eq_reindexed / eth_eq_reindexed.iloc[0] * 100).round(2).tolist()

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
        "btc_ohlc": btc_ohlc_tf,
        "eth_ohlc": eth_ohlc_tf,
        "btc_ema": btc_ema_tf,
        "eth_ema": eth_ema_tf,
        "trades_btc": btc_trades,
        "trades_eth": eth_trades,
        "timeline": timeline,
        "equity": {"t": eq_dates, "btc": btc_eq_n, "eth": eth_eq_n,
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
            "btc_position_pct": _current_position_pct(timeline, "BTC"),
            "eth_position_pct": _current_position_pct(timeline, "ETH"),
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
            "yearly": _compute_yearly(btc_equity, eth_equity,
                                       btc_df["Close"], eth_df["Close"]),
        },
    }

    with LOCK:
        DATA.update(result)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Data ready.")


def _compute_yearly(btc_eq, eth_eq, btc_close, eth_close):
    """Compute per-year returns, Sharpe, and MaxDD for both strategies."""
    years = []
    for year in range(btc_eq.index[0].year, btc_eq.index[-1].year + 1):
        s, e = f"{year}-01-01", f"{year}-12-31"
        bs = btc_eq.loc[s:e]
        es = eth_eq.loc[s:e]
        bc = btc_close.reindex(bs.index, method="ffill")
        ec = eth_close.reindex(es.index, method="ffill")
        if len(bs) < 10:
            continue

        btc_ret = round((bs.iloc[-1] / bs.iloc[0] - 1) * 100, 1)
        eth_ret = round((es.iloc[-1] / es.iloc[0] - 1) * 100, 1)
        btc_bh = round((bc.iloc[-1] / bc.iloc[0] - 1) * 100, 1)
        eth_bh = round((ec.iloc[-1] / ec.iloc[0] - 1) * 100, 1)

        def sharpe(eq_series):
            rets = eq_series.pct_change().dropna()
            if len(rets) < 20 or rets.std() == 0:
                return None
            ann = rets.mean() / rets.std() * np.sqrt(365 * 6)
            return round(ann, 2)

        def max_dd(eq_series):
            pk = eq_series.expanding().max()
            return round(((eq_series - pk) / pk).min() * 100, 1)

        years.append({
            "year": year,
            "btc_strat": btc_ret, "eth_strat": eth_ret,
            "btc_bh": btc_bh, "eth_bh": eth_bh,
            "btc_sharpe": sharpe(bs), "eth_sharpe": sharpe(es),
            "btc_dd": max_dd(bs), "eth_dd": max_dd(es),
        })
    return years


def _current_position_pct(timeline, symbol):
    """Get current position % from timeline events."""
    for e in timeline:
        if e["symbol"] == symbol:
            return min(e.get("remaining_pct", 0), 130)
    return 0


def _is_in_trade(stats):
    """Check if the strategy has an open position at the end of data."""
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

    eq_after = equity.loc[last_exit:]
    settle_start = min(6, len(eq_after) - 1)
    flat_value = float(eq_after.iloc[settle_start])

    entry_time = None
    for i in range(settle_start + 1, len(eq_after)):
        if abs(float(eq_after.iloc[i]) - flat_value) > 10000:
            entry_time = eq_after.index[i]
            break

    if abs(eq_final - flat_value) <= 10000:
        return

    if entry_time is None:
        return

    entry_price = float(df["Close"].asof(entry_time))
    current_price = float(df["Close"].iloc[-1])
    entry_ms = int(entry_time.timestamp() * 1000)
    now_ms = int(df.index[-1].timestamp() * 1000)

    sl_price = None
    if strategy_instance and hasattr(strategy_instance, "entry_log") and strategy_instance.entry_log:
        last_entry = strategy_instance.entry_log[-1]
        sl_price = round(last_entry["sl"], 2)
        entry_price = round(last_entry["price"], 2)

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


def _extract_trades(stats, symbol):
    """Extract trades with actual dollar amounts."""
    trades = []
    for _, t in stats._trades.iterrows():
        entry_ms = int(t["EntryTime"].timestamp() * 1000)
        exit_ms = int(t["ExitTime"].timestamp() * 1000)
        size = abs(int(t["Size"]))
        entry_value = round(size * float(t["EntryPrice"]), 2)
        exit_value = round(size * float(t["ExitPrice"]), 2)
        sl_price = round(float(t["SL"]), 2) if pd.notna(t["SL"]) else None
        exit_price = round(float(t["ExitPrice"]), 2)
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
    return FileResponse("web/static/index.html")


@app.get("/api/data")
def api_data():
    with LOCK:
        return DATA


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="0.0.0.0", port=8050, reload=False)
