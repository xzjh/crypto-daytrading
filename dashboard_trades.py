"""Build unified trade timeline for the dashboard.

All percentages are relative to the TOTAL portfolio value.
real_pct = (trade_value / strategy_equity) * rotation_weight
"""

import pandas as pd
from datetime import datetime, timezone


def build_timeline(btc_trades, eth_trades, rebalances,
                   portfolio_equity, rotation_weights,
                   btc_equity, eth_equity):
    """Build BUY + SELL event timeline with correct portfolio %."""
    events = []

    # Portfolio equity (starts at 1.0)
    peq_idx, peq_vals = portfolio_equity.index, portfolio_equity.values

    def nav_at(ms):
        ts = pd.Timestamp(ms, unit="ms", tz="UTC")
        i = peq_idx.searchsorted(ts, side="right") - 1
        i = max(0, min(i, len(peq_vals) - 1))
        if i + 1 < len(peq_vals) and abs(peq_idx[i + 1] - ts) < abs(peq_idx[i] - ts):
            i = i + 1
        return round(float(peq_vals[i]), 4)

    # Per-strategy equity (starts at BACKTEST_CASH = 1M)
    def strategy_equity_at(eq_series, ms):
        ts = pd.Timestamp(ms, unit="ms", tz="UTC")
        i = eq_series.index.searchsorted(ts, side="right") - 1
        i = max(0, min(i, len(eq_series) - 1))
        return float(eq_series.iloc[i])

    # Rotation weight
    w_idx, w_vals = rotation_weights.index, rotation_weights.values

    def weight_at(ms):
        ts = pd.Timestamp(ms, unit="ms", tz="UTC")
        i = w_idx.searchsorted(ts, side="right") - 1
        return float(w_vals[max(0, i)]) if i >= 0 else 0.5

    def in_trade(symbol_trades, ms):
        for t in symbol_trades:
            if t["entry_time"] <= ms <= t["exit_time"]:
                return True
        return False

    def real_pct_of_portfolio(trade_value, strategy_eq, rotation_w):
        """What % of total portfolio does this trade represent?
        = (fraction of strategy capital used) * (strategy's weight in portfolio)
        """
        if strategy_eq <= 0:
            return 0
        return round(trade_value / strategy_eq * rotation_w * 100, 1)

    def compute_allocation(ms, btc_in, eth_in, btc_trades_l, eth_trades_l):
        """Compute real holding % of total portfolio for each asset."""
        w = weight_at(ms)

        real_btc = 0
        if btc_in:
            beq = strategy_equity_at(btc_equity, ms)
            for tr in btc_trades_l:
                if tr["entry_time"] <= ms <= tr["exit_time"]:
                    real_btc = real_pct_of_portfolio(tr["entry_value"], beq, w)
                    break

        real_eth = 0
        if eth_in:
            eeq = strategy_equity_at(eth_equity, ms)
            for tr in eth_trades_l:
                if tr["entry_time"] <= ms <= tr["exit_time"]:
                    real_eth = real_pct_of_portfolio(tr["entry_value"], eeq, 1 - w)
                    break

        real_cash = round(max(0, 100 - real_btc - real_eth), 1)
        target_btc = round(w * 100) if btc_in else 0
        target_eth = round((1 - w) * 100) if eth_in else 0

        return real_btc, real_eth, real_cash, target_btc, target_eth

    def fmt_date(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    for t in btc_trades + eth_trades:
        sym = t["symbol"]
        ms_entry = t["entry_time"]
        ms_exit = t["exit_time"]

        # ── BUY ──
        nav = nav_at(ms_entry)
        w_entry = weight_at(ms_entry)
        rot_w_entry = w_entry if sym == "BTC" else (1 - w_entry)
        strat_eq_entry = strategy_equity_at(btc_equity if sym == "BTC" else eth_equity, ms_entry)
        trade_pct = real_pct_of_portfolio(t["entry_value"], strat_eq_entry, rot_w_entry)

        other_in = in_trade(eth_trades if sym == "BTC" else btc_trades, ms_entry)
        btc_in = True if sym == "BTC" else other_in
        eth_in = True if sym == "ETH" else other_in
        rb, re, rc, tb, te = compute_allocation(ms_entry, btc_in, eth_in, btc_trades, eth_trades)

        events.append({
            "time": ms_entry, "type": "BUY", "symbol": sym,
            "price": t["entry_price"], "trade_pct": trade_pct,
            "sl": t.get("sl"), "trade_return": None,
            "sl_triggered": False,
            "nav": nav, "fund_return": round((nav - 1) * 100, 2),
            "real_btc": rb, "real_eth": re, "real_cash": rc,
            "target_btc": tb, "target_eth": te,
            "detail": "",
            "_sort": 1,  # BUY sorts after SELL at same timestamp
        })

        # ── SELL ──
        nav = nav_at(ms_exit)
        # Use entry-time rotation weight if no rebalance happened during trade
        # This keeps buy/sell % consistent for the same position
        had_rebal = any(ms_entry < r["time"] < ms_exit for r in rebalances)
        if had_rebal:
            w_sell = weight_at(ms_exit)
            rot_w_sell = w_sell if sym == "BTC" else (1 - w_sell)
        else:
            rot_w_sell = rot_w_entry  # Same weight as when we bought

        strat_eq_exit = strategy_equity_at(btc_equity if sym == "BTC" else eth_equity, ms_exit)
        sell_pct = real_pct_of_portfolio(t["exit_value"], strat_eq_exit, rot_w_sell)

        other_trades = eth_trades if sym == "BTC" else btc_trades
        other_in = any(ot["entry_time"] <= ms_exit <= ot["exit_time"] for ot in other_trades)
        btc_in = False if sym == "BTC" else other_in
        eth_in = False if sym == "ETH" else other_in
        rb, re, rc, tb, te = compute_allocation(ms_exit, btc_in, eth_in, btc_trades, eth_trades)

        detail = "Entry: $" + f"{t['entry_price']:,.0f}" + " on " + fmt_date(ms_entry)
        for r in rebalances:
            if ms_entry < r["time"] < ms_exit:
                if sym == r.get("buy_asset"):
                    detail += " | REBAL +" + str(r["buy_to"] - r["buy_from"]) + "% @ $" + f"{r['buy_price']:,.0f}" + " on " + fmt_date(r["time"])
                elif sym == r.get("sell_asset"):
                    detail += " | REBAL -" + str(r["sell_from"] - r["sell_to"]) + "% @ $" + f"{r['sell_price']:,.0f}" + " on " + fmt_date(r["time"])

        # Skip the fake "sell" for open positions — they're still held
        if not t.get("is_open", False):
            events.append({
                "time": ms_exit, "type": "SELL", "symbol": sym,
                "price": t["exit_price"], "trade_pct": sell_pct,
                "sl": None, "trade_return": t["return_pct"],
                "sl_triggered": t.get("sl_triggered", False),
                "is_open": False,
                "nav": nav, "fund_return": round((nav - 1) * 100, 2),
                "real_btc": rb, "real_eth": re, "real_cash": rc,
                "target_btc": tb, "target_eth": te,
                "detail": detail,
                "_sort": 0,
            })

    # Sort: descending time, SELL before BUY at same timestamp
    events.sort(key=lambda ev: (-ev["time"], ev["_sort"]))
    return events
