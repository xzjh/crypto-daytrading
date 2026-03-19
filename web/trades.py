"""Build unified trade timeline for the dashboard."""

import pandas as pd
from datetime import datetime, timezone


def build_timeline(btc_trades, eth_trades, btc_equity, eth_equity):
    """Build BUY + SELL event timeline with correct position %."""
    events = []

    def strategy_equity_at(eq_series, ms):
        ts = pd.Timestamp(ms, unit="ms", tz="UTC")
        i = eq_series.index.searchsorted(ts, side="right") - 1
        i = max(0, min(i, len(eq_series) - 1))
        return float(eq_series.iloc[i])

    def trade_pct(trade_value, strategy_eq):
        if strategy_eq <= 0:
            return 0
        return round(trade_value / strategy_eq * 100, 1)

    def fmt_date(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    for t in btc_trades + eth_trades:
        sym = t["symbol"]
        ms_entry = t["entry_time"]
        ms_exit = t["exit_time"]

        # ── BUY ──
        strat_eq_entry = strategy_equity_at(btc_equity if sym == "BTC" else eth_equity, ms_entry)
        pct = trade_pct(t["entry_value"], strat_eq_entry)

        events.append({
            "time": ms_entry, "type": "BUY", "symbol": sym,
            "price": t["entry_price"], "trade_pct": pct,
            "sl": t.get("sl"), "trade_return": None,
            "sl_triggered": False,
            "detail": "",
            "_sort": 1,
        })

        # ── SELL ──
        if not t.get("is_open", False):
            strat_eq_exit = strategy_equity_at(btc_equity if sym == "BTC" else eth_equity, ms_exit)
            sell_pct = trade_pct(t["exit_value"], strat_eq_exit)
            detail = "Entry: $" + f"{t['entry_price']:,.0f}" + " on " + fmt_date(ms_entry)

            events.append({
                "time": ms_exit, "type": "SELL", "symbol": sym,
                "price": t["exit_price"], "trade_pct": sell_pct,
                "sl": None, "trade_return": t["return_pct"],
                "sl_triggered": t.get("sl_triggered", False),
                "detail": detail,
                "_sort": 0,
            })

    # Compute remaining position after each event using unit sizes
    for e in events:
        sym = e["symbol"]
        t = e["time"]
        sym_trades = btc_trades if sym == "BTC" else eth_trades
        eq_series = btc_equity if sym == "BTC" else eth_equity
        eq = strategy_equity_at(eq_series, t)
        price = e["price"]

        remaining_value = 0
        for tr in sym_trades:
            entry_ok = tr["entry_time"] <= t if e["type"] == "BUY" else tr["entry_time"] < t
            if entry_ok and (tr["exit_time"] > t or tr.get("is_open")):
                if tr["size"] > 0:
                    remaining_value += tr["size"] * price
                else:
                    remaining_value += tr["entry_value"]
        e["remaining_pct"] = round(remaining_value / eq * 100, 1) if eq > 0 else 0

    # Merge same-time same-asset same-type events into one
    events.sort(key=lambda ev: (ev["time"], ev["_sort"], ev["symbol"]))
    merged = []
    for e in events:
        if merged and merged[-1]["time"] == e["time"] and merged[-1]["symbol"] == e["symbol"] and merged[-1]["type"] == e["type"]:
            merged[-1]["trade_pct"] = round(merged[-1]["trade_pct"] + e["trade_pct"], 1)
            merged[-1]["remaining_pct"] = e["remaining_pct"]
        else:
            merged.append(dict(e))

    # Round to 1 decimal place
    for e in merged:
        if e.get("trade_pct"):
            e["trade_pct"] = round(e["trade_pct"], 1)
        e["remaining_pct"] = round(e.get("remaining_pct", 0), 1)

    # Tag events: Open/Close vs Add/Reduce
    prev_remaining = {}
    for e in merged:
        sym = e["symbol"]
        if e["type"] == "BUY":
            e["tag"] = "Open" if prev_remaining.get(sym, 0) == 0 else "Add"
        else:
            if e["remaining_pct"] == 0:
                e["tag"] = "Close"
            else:
                e["tag"] = "Reduce"
        prev_remaining[sym] = e["remaining_pct"]

    # Sort: descending time, SELL before BUY at same timestamp
    merged.sort(key=lambda ev: (-ev["time"], ev["_sort"]))
    return merged
