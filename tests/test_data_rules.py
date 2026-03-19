"""Backend data rules and sentry invariant tests.

Two categories:
  - Data Rules: shape, type, range checks on every field
  - Sentry Rules: business invariants — violation = bug

Compatible with both pytest and unittest (no external deps needed).
"""

import math
import unittest

from tests.conftest import DATA_SNAPSHOT

D = DATA_SNAPSHOT  # alias for brevity

# ── Required key sets ──

TIMELINE_KEYS = {
    "time", "type", "symbol", "price", "trade_pct", "remaining_pct",
    "sl", "trade_return", "sl_triggered",
    "detail", "tag", "_sort",
}

TRADE_KEYS = {
    "entry_time", "exit_time", "entry_price", "exit_price",
    "size", "entry_value", "exit_value", "pnl", "return_pct",
    "sl", "sl_triggered", "symbol",
}

EQUITY_KEYS = {"t", "btc", "eth", "btc_bh", "eth_bh"}

CURRENT_KEYS = {
    "btc_price", "eth_price", "btc_rsi", "eth_rsi",
    "btc_regime", "eth_regime", "btc_momentum", "eth_momentum",
    "btc_in_trade", "eth_in_trade", "btc_position_pct", "eth_position_pct",
}

STATS_KEYS = {
    "btc_ret", "eth_ret", "btc_bh", "eth_bh",
    "btc_sharpe", "eth_sharpe", "btc_dd", "eth_dd",
    "btc_trades", "eth_trades",
    "yearly",
}

YEARLY_KEYS = {
    "year", "btc_strat", "eth_strat", "btc_bh", "eth_bh",
    "btc_sharpe", "eth_sharpe", "btc_dd", "eth_dd",
}


# ─────────────────────────────────────────────────────────
# DATA RULES — field completeness, types, ranges
# ─────────────────────────────────────────────────────────

class TestTimelineDataRules(unittest.TestCase):

    def test_not_empty(self):
        self.assertGreater(len(D["timeline"]), 0)

    def test_required_fields(self):
        for e in D["timeline"]:
            missing = TIMELINE_KEYS - set(e.keys())
            self.assertFalse(missing, f"Missing: {missing} at t={e.get('time')}")

    def test_type_values(self):
        for e in D["timeline"]:
            self.assertIn(e["type"], {"BUY", "SELL"})

    def test_symbol_values(self):
        for e in D["timeline"]:
            self.assertIn(e["symbol"], {"BTC", "ETH"})

    def test_time_positive_int(self):
        for e in D["timeline"]:
            self.assertIsInstance(e["time"], int)
            self.assertGreater(e["time"], 0)

    def test_price_positive(self):
        for e in D["timeline"]:
            self.assertGreater(e["price"], 0)

    def test_trade_pct_range(self):
        for e in D["timeline"]:
            pct = e["trade_pct"]
            self.assertIsInstance(pct, (int, float))
            self.assertGreaterEqual(pct, 0)

    def test_remaining_pct_range(self):
        for e in D["timeline"]:
            pct = e["remaining_pct"]
            self.assertIsInstance(pct, (int, float))
            self.assertGreaterEqual(pct, 0)

    def test_sl_type(self):
        for e in D["timeline"]:
            if e["sl"] is not None:
                self.assertIsInstance(e["sl"], (int, float))
                self.assertGreater(e["sl"], 0)

    def test_trade_return_type(self):
        for e in D["timeline"]:
            if e["trade_return"] is not None:
                self.assertIsInstance(e["trade_return"], (int, float))

    def test_sl_triggered_is_bool(self):
        for e in D["timeline"]:
            self.assertIsInstance(e["sl_triggered"], bool)

    def test_tag_values(self):
        for e in D["timeline"]:
            self.assertIn(e["tag"], {"Open", "Close", "Add", "Reduce"})

    def test_sort_values(self):
        for e in D["timeline"]:
            self.assertIn(e["_sort"], {0, 1})


class TestTradeDataRules(unittest.TestCase):

    @property
    def all_trades(self):
        return D["trades_btc"] + D["trades_eth"]

    def test_not_empty(self):
        self.assertGreater(len(D["trades_btc"]), 0)
        self.assertGreater(len(D["trades_eth"]), 0)

    def test_required_fields(self):
        for t in self.all_trades:
            missing = TRADE_KEYS - set(t.keys())
            self.assertFalse(missing, f"Missing: {missing}")

    def test_times_positive_int(self):
        for t in self.all_trades:
            self.assertIsInstance(t["entry_time"], int)
            self.assertGreater(t["entry_time"], 0)
            self.assertIsInstance(t["exit_time"], int)
            self.assertGreater(t["exit_time"], 0)

    def test_prices_positive(self):
        for t in self.all_trades:
            self.assertGreater(t["entry_price"], 0)
            self.assertGreater(t["exit_price"], 0)

    def test_values_positive(self):
        for t in self.all_trades:
            self.assertGreater(t["entry_value"], 0)
            self.assertGreater(t["exit_value"], 0)

    def test_size_non_negative(self):
        for t in self.all_trades:
            self.assertGreaterEqual(t["size"], 0)

    def test_sl_type(self):
        for t in self.all_trades:
            if t["sl"] is not None:
                self.assertIsInstance(t["sl"], (int, float))
                self.assertGreater(t["sl"], 0)

    def test_sl_triggered_is_bool(self):
        for t in self.all_trades:
            self.assertIsInstance(t["sl_triggered"], bool)

    def test_symbol_matches_list(self):
        for t in D["trades_btc"]:
            self.assertEqual(t["symbol"], "BTC")
        for t in D["trades_eth"]:
            self.assertEqual(t["symbol"], "ETH")

    def test_return_pct_numeric(self):
        for t in self.all_trades:
            self.assertIsInstance(t["return_pct"], (int, float))

    def test_pnl_numeric(self):
        for t in self.all_trades:
            self.assertIsInstance(t["pnl"], (int, float))


class TestEquityDataRules(unittest.TestCase):

    def test_has_required_keys(self):
        missing = EQUITY_KEYS - set(D["equity"].keys())
        self.assertFalse(missing, f"Missing: {missing}")

    def test_arrays_same_length(self):
        """All equity arrays must have the same length as t."""
        eq = D["equity"]
        n = len(eq["t"])
        for key in ["btc", "eth", "btc_bh", "eth_bh"]:
            self.assertEqual(len(eq[key]), n, f"equity['{key}'] len {len(eq[key])} != {n}")

    def test_t_sorted_ascending(self):
        ts = D["equity"]["t"]
        for i in range(1, len(ts)):
            self.assertGreater(ts[i], ts[i - 1], f"Not ascending at {i}")

    def test_no_nan_or_none(self):
        for key in EQUITY_KEYS:
            for i, v in enumerate(D["equity"][key]):
                self.assertIsNotNone(v, f"equity['{key}'][{i}] is None")
                if isinstance(v, float):
                    self.assertFalse(math.isnan(v), f"equity['{key}'][{i}] NaN")

    def test_all_positive(self):
        for key in ["btc", "eth", "btc_bh", "eth_bh"]:
            for i, v in enumerate(D["equity"][key]):
                self.assertGreater(v, 0, f"equity['{key}'][{i}] = {v}")

    def test_t_positive_int(self):
        for v in D["equity"]["t"]:
            self.assertIsInstance(v, int)
            self.assertGreater(v, 0)


class TestCurrentDataRules(unittest.TestCase):

    def test_has_required_keys(self):
        missing = CURRENT_KEYS - set(D["current"].keys())
        self.assertFalse(missing, f"Missing: {missing}")

    def test_prices_positive(self):
        self.assertGreater(D["current"]["btc_price"], 0)
        self.assertGreater(D["current"]["eth_price"], 0)

    def test_rsi_range(self):
        for sym in ["btc", "eth"]:
            rsi = D["current"][f"{sym}_rsi"]
            self.assertGreaterEqual(rsi, 0)
            self.assertLessEqual(rsi, 100)

    def test_regime_values(self):
        self.assertIn(D["current"]["btc_regime"], {"BULL", "BEAR"})
        self.assertIn(D["current"]["eth_regime"], {"BULL", "BEAR"})

    def test_momentum_numeric(self):
        self.assertIsInstance(D["current"]["btc_momentum"], (int, float))
        self.assertIsInstance(D["current"]["eth_momentum"], (int, float))

    def test_in_trade_is_bool(self):
        self.assertIsInstance(D["current"]["btc_in_trade"], bool)
        self.assertIsInstance(D["current"]["eth_in_trade"], bool)

    def test_position_pct_range(self):
        for sym in ["btc", "eth"]:
            pct = D["current"][f"{sym}_position_pct"]
            self.assertGreaterEqual(pct, 0)
            self.assertLessEqual(pct, 130)


class TestStatsDataRules(unittest.TestCase):

    def test_has_required_keys(self):
        missing = STATS_KEYS - set(D["stats"].keys())
        self.assertFalse(missing, f"Missing: {missing}")

    def test_trade_counts_positive(self):
        self.assertGreater(D["stats"]["btc_trades"], 0)
        self.assertGreater(D["stats"]["eth_trades"], 0)

    def test_drawdowns_non_positive(self):
        self.assertLessEqual(D["stats"]["btc_dd"], 0)
        self.assertLessEqual(D["stats"]["eth_dd"], 0)

    def test_returns_numeric(self):
        for key in ["btc_ret", "eth_ret", "btc_bh", "eth_bh"]:
            self.assertIsInstance(D["stats"][key], (int, float))

    def test_sharpe_type(self):
        for key in ["btc_sharpe", "eth_sharpe"]:
            v = D["stats"][key]
            self.assertTrue(v is None or isinstance(v, (int, float)))

    def test_yearly_not_empty(self):
        self.assertGreater(len(D["stats"]["yearly"]), 0)

    def test_yearly_fields(self):
        for y in D["stats"]["yearly"]:
            missing = YEARLY_KEYS - set(y.keys())
            self.assertFalse(missing, f"Year {y.get('year')}: missing {missing}")

    def test_yearly_years_ascending(self):
        years = [y["year"] for y in D["stats"]["yearly"]]
        self.assertEqual(years, sorted(years))

    def test_yearly_dd_non_positive(self):
        for y in D["stats"]["yearly"]:
            self.assertLessEqual(y["btc_dd"], 0, f"Year {y['year']}")
            self.assertLessEqual(y["eth_dd"], 0, f"Year {y['year']}")


# ─────────────────────────────────────────────────────────
# SENTRY RULES — business invariants, violation = bug
# ─────────────────────────────────────────────────────────

class TestTimelineSentryRules(unittest.TestCase):

    def test_sorted_descending(self):
        """Sort key: (-time, _sort). Descending time; at same time _sort ascending."""
        tl = D["timeline"]
        for i in range(1, len(tl)):
            p, c = tl[i - 1], tl[i]
            if p["time"] == c["time"]:
                self.assertLessEqual(
                    p["_sort"], c["_sort"],
                    f"Same-time sort violation at index {i}: "
                    f"_sort {p['_sort']} > {c['_sort']}")
            else:
                self.assertGreater(
                    p["time"], c["time"],
                    f"Time not descending at index {i}: "
                    f"{p['time']} <= {c['time']}")

    def test_sell_has_no_sl(self):
        for e in D["timeline"]:
            if e["type"] == "SELL":
                self.assertIsNone(e["sl"], f"SELL at {e['time']} has sl")

    def test_buy_has_no_trade_return(self):
        for e in D["timeline"]:
            if e["type"] == "BUY":
                self.assertIsNone(e["trade_return"],
                                  f"BUY at {e['time']} has trade_return")

    def test_sell_sort_zero_buy_sort_one(self):
        for e in D["timeline"]:
            expected = 0 if e["type"] == "SELL" else 1
            self.assertEqual(e["_sort"], expected)

    def test_no_duplicate_events(self):
        seen = set()
        for e in D["timeline"]:
            key = (e["time"], e["type"], e["symbol"])
            self.assertNotIn(key, seen, f"Duplicate: {key}")
            seen.add(key)

    def test_buy_sl_below_price(self):
        """Long-only: stop-loss must be below entry price."""
        for e in D["timeline"]:
            if e["type"] == "BUY" and e["sl"] is not None:
                self.assertLess(e["sl"], e["price"],
                                f"BUY at {e['time']}: sl >= price")

    # ── Tag invariants ──

    def test_close_remaining_pct_zero(self):
        """Close = fully exited → remaining_pct must be 0."""
        for e in D["timeline"]:
            if e["tag"] == "Close":
                self.assertEqual(e["remaining_pct"], 0,
                                 f"Close at {e['time']} {e['symbol']}")

    def test_open_previous_remaining_zero(self):
        """Open = from zero → previous same-symbol remaining was 0."""
        # Chronological: ascending time, SELL before BUY at same time
        chrono = sorted(D["timeline"], key=lambda e: (e["time"], e["_sort"]))
        prev = {}
        for e in chrono:
            sym = e["symbol"]
            if e["tag"] == "Open":
                self.assertEqual(prev.get(sym, 0), 0,
                                 f"Open at {e['time']} {sym}: prev={prev.get(sym)}")
            prev[sym] = e["remaining_pct"]

    def test_buy_tag_is_open_or_add(self):
        for e in D["timeline"]:
            if e["type"] == "BUY":
                self.assertIn(e["tag"], {"Open", "Add"},
                              f"BUY at {e['time']}: tag={e['tag']}")

    def test_sell_tag_is_reduce_or_close(self):
        for e in D["timeline"]:
            if e["type"] == "SELL":
                self.assertIn(e["tag"], {"Reduce", "Close"},
                              f"SELL at {e['time']}: tag={e['tag']}")

    def test_add_previous_remaining_positive(self):
        """Add = adding to position → previous remaining > 0."""
        chrono = sorted(D["timeline"], key=lambda e: (e["time"], e["_sort"]))
        prev = {}
        for e in chrono:
            sym = e["symbol"]
            if e["tag"] == "Add":
                self.assertGreater(prev.get(sym, 0), 0,
                                   f"Add at {e['time']} {sym}: prev=0")
            prev[sym] = e["remaining_pct"]

    def test_reduce_remaining_positive(self):
        """Reduce = partial sell → remaining > 0 after."""
        for e in D["timeline"]:
            if e["tag"] == "Reduce":
                self.assertGreater(e["remaining_pct"], 0,
                                   f"Reduce at {e['time']} {e['symbol']}")


class TestTradeSentryRules(unittest.TestCase):

    def test_entry_before_exit(self):
        for t in D["trades_btc"] + D["trades_eth"]:
            if t.get("is_open"):
                self.assertLessEqual(t["entry_time"], t["exit_time"])
            else:
                self.assertLess(t["entry_time"], t["exit_time"],
                                f"{t['symbol']}: entry >= exit")

    def test_sl_triggered_implies_sl_exists(self):
        for t in D["trades_btc"] + D["trades_eth"]:
            if t["sl_triggered"]:
                self.assertIsNotNone(t["sl"],
                                     f"{t['symbol']}: triggered but no sl")

    def test_sl_triggered_price_near_sl(self):
        for t in D["trades_btc"] + D["trades_eth"]:
            if t["sl_triggered"] and t["sl"] is not None:
                self.assertLessEqual(t["exit_price"], t["sl"] * 1.01,
                                     f"{t['symbol']}: exit >> sl")

    def test_open_trade_has_zero_size(self):
        for t in D["trades_btc"] + D["trades_eth"]:
            if t.get("is_open"):
                self.assertEqual(t["size"], 0)


class TestCooldownRule(unittest.TestCase):
    """After selling, there must be a cooldown of at least 3 bars (12h) before buying again."""

    COOLDOWN_MS = 3 * 4 * 3600 * 1000  # 3 bars * 4h = 12 hours in ms

    def test_no_same_bar_sell_then_buy(self):
        """BUY must not happen on the same bar as a preceding SELL for the same symbol."""
        for sym in ["BTC", "ETH"]:
            events = sorted(
                [e for e in D["timeline"] if e["symbol"] == sym],
                key=lambda e: (e["time"], e["_sort"])
            )
            for i in range(len(events) - 1):
                if events[i]["type"] == "SELL" and events[i+1]["type"] == "BUY" \
                        and events[i]["time"] == events[i+1]["time"]:
                    self.fail(
                        f"{sym}: SELL and BUY on same bar at {events[i]['time']} "
                        f"— cooldown violated")

    def test_cooldown_between_sell_and_next_buy(self):
        """After a SELL, the next BUY for the same symbol must be >= 3 bars later."""
        for sym in ["BTC", "ETH"]:
            events = sorted(
                [e for e in D["timeline"] if e["symbol"] == sym],
                key=lambda e: (e["time"], e["_sort"])
            )
            last_sell_time = None
            for e in events:
                if e["type"] == "SELL" and e["tag"] == "Close":
                    last_sell_time = e["time"]
                elif e["type"] == "BUY" and e["tag"] == "Open" and last_sell_time is not None:
                    gap = e["time"] - last_sell_time
                    self.assertGreaterEqual(gap, self.COOLDOWN_MS,
                        f"{sym}: BUY at {e['time']} only {gap // 3600000}h after "
                        f"SELL at {last_sell_time} — need >= 12h cooldown")
                    last_sell_time = None


class TestEquitySentryRules(unittest.TestCase):

    def test_starts_at_100(self):
        for key in ["btc", "eth", "btc_bh", "eth_bh"]:
            self.assertEqual(D["equity"][key][0], 100,
                             f"equity['{key}'][0] != 100")

    def test_no_zero_values(self):
        for key in ["btc", "eth", "btc_bh", "eth_bh"]:
            for i, v in enumerate(D["equity"][key]):
                self.assertNotEqual(v, 0, f"equity['{key}'][{i}] is zero")

    def test_reasonable_range(self):
        for key in ["btc", "eth", "btc_bh", "eth_bh"]:
            for i, v in enumerate(D["equity"][key]):
                self.assertTrue(1 <= v <= 100000,
                                f"equity['{key}'][{i}] = {v}")


class TestCrossConsistency(unittest.TestCase):

    def test_current_position_matches_timeline(self):
        for sym in ["BTC", "ETH"]:
            cur_pct = D["current"][f"{sym.lower()}_position_pct"]
            timeline_pct = 0
            for e in D["timeline"]:
                if e["symbol"] == sym:
                    timeline_pct = min(e["remaining_pct"], 130)
                    break
            self.assertEqual(cur_pct, timeline_pct,
                             f"{sym}: current={cur_pct} vs timeline={timeline_pct}")

    def test_closed_trade_count_matches_stats(self):
        for sym, key in [("BTC", "btc_trades"), ("ETH", "eth_trades")]:
            closed = [t for t in D[f"trades_{sym.lower()}"]
                       if not t.get("is_open")]
            self.assertEqual(len(closed), D["stats"][key],
                             f"{sym}: {len(closed)} closed != stats {D['stats'][key]}")

    def test_buy_count_lte_total_trades(self):
        """BUY events <= total trades (merging reduces count)."""
        for sym in ["BTC", "ETH"]:
            buys = sum(1 for e in D["timeline"]
                       if e["symbol"] == sym and e["type"] == "BUY")
            trades = len(D[f"trades_{sym.lower()}"])
            self.assertLessEqual(buys, trades,
                                 f"{sym}: {buys} BUYs > {trades} trades")
            # But shouldn't be drastically fewer
            self.assertGreater(buys, 0, f"{sym}: no BUY events")

    def test_sell_count_lte_closed_trades(self):
        """SELL events <= closed trades (merging reduces count)."""
        for sym in ["BTC", "ETH"]:
            sells = sum(1 for e in D["timeline"]
                        if e["symbol"] == sym and e["type"] == "SELL")
            closed = len([t for t in D[f"trades_{sym.lower()}"]
                          if not t.get("is_open")])
            self.assertLessEqual(sells, closed,
                                 f"{sym}: {sells} SELLs > {closed} closed")


if __name__ == "__main__":
    unittest.main()
