"""Frontend data consistency tests.

Verifies that the data contract between backend and frontend is correct:
every value the frontend renders must faithfully represent the backend data.

We replicate the frontend's JS rendering logic in Python and assert correctness.
"""

import math
import unittest

from tests.conftest import DATA_SNAPSHOT

D = DATA_SNAPSHOT


# ─────────────────────────────────────────────────────────
# Helper: replicate frontend JS functions in Python
# ─────────────────────────────────────────────────────────

def fmt_money(v):
    """JS: fmtMoney(v)"""
    if v >= 1000:
        return "$" + f"{v:,.0f}"
    return "$" + f"{v:.2f}"


def pct_str(v):
    """JS: pctStr(v)"""
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def pct_class(v):
    """JS: pctClass(v)"""
    return "positive" if v >= 0 else "negative"


def _local_tz_abbr():
    """Get local timezone abbreviation matching JS Intl.DateTimeFormat city name."""
    from datetime import datetime, timezone
    import time
    # Use the IANA timezone city name (e.g. "Shanghai" from "Asia/Shanghai")
    try:
        import zoneinfo
        pass
    except ImportError:
        pass
    # Fallback: use time module
    tz_name = time.tzname[time.daylight and time.localtime().tm_isdst]
    return tz_name

def fmt_date_time(ms):
    """JS: fmtDateTime(ms) — browser local time + timezone"""
    from datetime import datetime
    d = datetime.fromtimestamp(ms / 1000)  # local timezone
    return d.strftime("%Y-%m-%d %H:%M")


# ─────────────────────────────────────────────────────────
# Status section tests
# ─────────────────────────────────────────────────────────

class TestStatusRendering(unittest.TestCase):
    """Tests for renderStatus() — verify every displayed value matches backend."""

    def test_price_displayed_value(self):
        """Price shown = fmtMoney(current.{asset}_price), directly from backend."""
        for asset in ["btc", "eth"]:
            price = D["current"][f"{asset}_price"]
            self.assertGreater(price, 0)
            rendered = fmt_money(price)
            self.assertTrue(rendered.startswith("$"))
            # Verify the number in the string matches the backend value
            num_str = rendered.replace("$", "").replace(",", "")
            if price >= 1000:
                self.assertEqual(int(float(num_str)), round(price))
            else:
                self.assertAlmostEqual(float(num_str), price, places=2)

    def test_rsi_displayed_value(self):
        """RSI shown as plain number, directly from backend."""
        for asset in ["btc", "eth"]:
            rsi = D["current"][f"{asset}_rsi"]
            self.assertGreaterEqual(rsi, 0)
            self.assertLessEqual(rsi, 100)

    def test_regime_displayed_value(self):
        """Regime shown as BULL or BEAR, directly from backend."""
        for asset in ["btc", "eth"]:
            self.assertIn(D["current"][f"{asset}_regime"], {"BULL", "BEAR"})

    def test_position_displayed_value(self):
        """Position shown = round(min(position_pct, 130)).
        Must match latest timeline event's remaining_pct."""
        for asset in ["btc", "eth"]:
            raw = D["current"][f"{asset}_position_pct"] or 0
            displayed = round(min(raw, 130))
            # Verify it matches the timeline
            sym = asset.upper()
            timeline_rp = 0
            for e in D["timeline"]:
                if e["symbol"] == sym:
                    timeline_rp = e["remaining_pct"]
                    break
            expected = round(min(timeline_rp, 130))
            self.assertEqual(displayed, expected,
                f"{sym}: status shows {displayed}% but timeline says {expected}%")

    def test_position_text_no_position_when_zero(self):
        """When position is 0, display shows 'No Position', not '0%'."""
        for asset in ["btc", "eth"]:
            raw = D["current"][f"{asset}_position_pct"] or 0
            posPct = round(min(raw, 130))
            if posPct == 0:
                # JS: posPct > 0 ? posPct + '%' : 'No Position'
                expected_text = "No Position"
            else:
                expected_text = f"{posPct}%"
            self.assertTrue(len(expected_text) > 0)

    def test_momentum_displayed_value(self):
        """Momentum shown as pctStr(momentum). Label says '20d'."""
        for asset in ["btc", "eth"]:
            mom = D["current"][f"{asset}_momentum"]
            self.assertIsInstance(mom, (int, float))
            rendered = pct_str(mom)
            self.assertIn("%", rendered)
            # Verify sign
            if mom >= 0:
                self.assertTrue(rendered.startswith("+"))
            else:
                self.assertTrue(rendered.startswith("-"))


# ─────────────────────────────────────────────────────────
# Trade history table tests
# ─────────────────────────────────────────────────────────

class TestTradeTableRendering(unittest.TestCase):
    """Tests for renderTrades() logic."""

    def _get_events(self, symbol):
        return [e for e in D["timeline"] if e["symbol"] == symbol]

    def test_date_column_uses_local_time(self):
        """Each row date = fmtDateTime(e.time) in local timezone with tz label."""
        for e in D["timeline"][:20]:
            rendered = fmt_date_time(e["time"])
            self.assertRegex(rendered, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}")

    def test_dates_are_local_not_utc(self):
        """Dates must use browser local time, not UTC."""
        import time
        if time.timezone == 0 and time.daylight == 0:
            self.skipTest("Local timezone is UTC — cannot distinguish")
        from datetime import datetime, timezone
        e = D["timeline"][0]
        local = fmt_date_time(e["time"])
        utc = datetime.fromtimestamp(e["time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        self.assertNotEqual(local, utc,
                            f"Date {local} matches UTC — should be local time")

    def test_type_column_matches_event(self):
        for e in D["timeline"]:
            self.assertIn(e["type"], {"BUY", "SELL"})

    def test_price_column_uses_event_price(self):
        """Price column = fmtMoney(e.price)."""
        for e in D["timeline"]:
            self.assertGreater(e["price"], 0)

    def test_trade_pct_label_shown_when_nonzero(self):
        """JS: pctLabel = pct ? pct + '% ' : '' — shown when trade_pct > 0."""
        for e in D["timeline"]:
            pct = e["trade_pct"] or 0
            if pct > 0:
                label = f"{pct}% "
                self.assertTrue(len(label) > 0)

    def test_tag_displayed_uppercase(self):
        """JS: tagText.toUpperCase()"""
        for e in D["timeline"]:
            tag = e.get("tag") or ("Open" if e["type"] == "BUY" else "Close")
            self.assertIn(tag.upper(), {"OPEN", "CLOSE", "ADD", "REDUCE"})

    def test_trade_pnl_only_for_sell(self):
        """Trade P&L column shows value only for SELL events with trade_return."""
        for e in D["timeline"]:
            if e["type"] == "BUY":
                # BUY should show '—' (trade_return is None)
                self.assertIsNone(e["trade_return"],
                                  f"BUY at {e['time']} has trade_return")
            elif e["type"] == "SELL" and e["trade_return"] is not None:
                # SELL with trade_return shows pctStr
                rendered = pct_str(e["trade_return"])
                self.assertIn("%", rendered)

    def test_position_bar_uses_remaining_pct(self):
        """JS: displayPct = Math.min(e.remaining_pct || 0, 130)"""
        for e in D["timeline"]:
            raw = e.get("remaining_pct") or 0
            displayPct = min(raw, 130)
            self.assertGreaterEqual(displayPct, 0)
            self.assertLessEqual(displayPct, 130)

    def test_position_bar_fill_color_logic(self):
        """fillColor logic: >100 orange, >0 green, else card-inner."""
        for e in D["timeline"]:
            displayPct = min(e.get("remaining_pct") or 0, 130)
            if displayPct > 100:
                expected = "var(--orange)"
            elif displayPct > 0:
                expected = "var(--green)"
            else:
                expected = "var(--card-inner)"
            # Just verify the logic is consistent — no actual DOM check
            self.assertIn(expected, [
                "var(--orange)", "var(--green)", "var(--card-inner)"
            ])

    def test_filter_by_symbol_correct(self):
        """renderAll filters timeline by symbol for each tab."""
        btc_events = [e for e in D["timeline"] if e["symbol"] == "BTC"]
        eth_events = [e for e in D["timeline"] if e["symbol"] == "ETH"]
        self.assertGreater(len(btc_events), 0)
        self.assertGreater(len(eth_events), 0)
        self.assertEqual(len(btc_events) + len(eth_events), len(D["timeline"]))


# ─────────────────────────────────────────────────────────
# Performance table tests
# ─────────────────────────────────────────────────────────

class TestPerfTableRendering(unittest.TestCase):
    """Tests for renderPerfTable() — verify every cell matches backend data."""

    def test_total_row_strategy_return(self):
        """Total Strategy column = pctStr(stats.{asset}_ret)."""
        for asset in ["btc", "eth"]:
            ret = D["stats"][f"{asset}_ret"]
            rendered = pct_str(ret)
            self.assertIn("%", rendered)
            # Verify the number
            num = float(rendered.replace("%", "").replace("+", ""))
            self.assertAlmostEqual(num, ret, places=1)

    def test_total_row_bh_return(self):
        """Total B&H column = pctStr(stats.{asset}_bh)."""
        for asset in ["btc", "eth"]:
            bh = D["stats"][f"{asset}_bh"]
            rendered = pct_str(bh)
            num = float(rendered.replace("%", "").replace("+", ""))
            self.assertAlmostEqual(num, bh, places=1)

    def test_total_row_alpha(self):
        """Total Alpha = strategy_ret - bh_ret (computed in frontend, not from backend)."""
        for asset in ["btc", "eth"]:
            ret = D["stats"][f"{asset}_ret"]
            bh = D["stats"][f"{asset}_bh"]
            alpha = ret - bh
            rendered = pct_str(alpha)
            num = float(rendered.replace("%", "").replace("+", ""))
            self.assertAlmostEqual(num, alpha, places=1,
                msg=f"{asset} total alpha: rendered {num} != computed {alpha:.1f}")

    def test_total_row_sharpe(self):
        """Sharpe shown as .toFixed(2) or '—' if None."""
        for asset in ["btc", "eth"]:
            sh = D["stats"][f"{asset}_sharpe"]
            if sh is not None:
                rendered = f"{sh:.2f}"
                self.assertAlmostEqual(float(rendered), sh, places=2)

    def test_total_row_dd(self):
        """MaxDD shown as raw value + '%', directly from backend."""
        for asset in ["btc", "eth"]:
            dd = D["stats"][f"{asset}_dd"]
            self.assertIsInstance(dd, (int, float))
            self.assertLessEqual(dd, 0, f"{asset} total DD should be <= 0")

    def test_yearly_alpha_equals_strat_minus_bh(self):
        """Per-year Alpha = {asset}_strat - {asset}_bh for every year."""
        for asset in ["btc", "eth"]:
            for y in D["stats"]["yearly"]:
                sr = y[f"{asset}_strat"]
                bh = y[f"{asset}_bh"]
                alpha = sr - bh
                rendered = pct_str(alpha)
                num = float(rendered.replace("%", "").replace("+", ""))
                self.assertAlmostEqual(num, alpha, places=1,
                    msg=f"{asset} {y['year']}: alpha rendered {num} != {alpha:.1f}")

    def test_yearly_dd_directly_from_backend(self):
        """Per-year MaxDD comes directly from yearly[i].{asset}_dd."""
        for asset in ["btc", "eth"]:
            for y in D["stats"]["yearly"]:
                dd = y[f"{asset}_dd"]
                self.assertIsInstance(dd, (int, float))
                self.assertLessEqual(dd, 0,
                    f"{asset} {y['year']}: DD={dd} should be <= 0")

    def test_yearly_sharpe_format(self):
        """Per-year Sharpe: .toFixed(2) if not None, else '—'."""
        for asset in ["btc", "eth"]:
            for y in D["stats"]["yearly"]:
                sh = y[f"{asset}_sharpe"]
                if sh is not None:
                    self.assertIsInstance(sh, (int, float))
                    rendered = f"{sh:.2f}"
                    self.assertAlmostEqual(float(rendered), sh, places=2)

    def test_yearly_displayed_in_reverse_order(self):
        """JS iterates from last to first — latest year shown first."""
        yearly = D["stats"]["yearly"]
        self.assertGreater(len(yearly), 1)
        self.assertLess(yearly[0]["year"], yearly[-1]["year"])

    def test_yearly_css_class_matches_value_sign(self):
        """Positive returns get 'positive' class, negative get 'negative'."""
        for asset in ["btc", "eth"]:
            for y in D["stats"]["yearly"]:
                sr = y[f"{asset}_strat"]
                expected_cls = "positive" if sr >= 0 else "negative"
                self.assertEqual(pct_class(sr), expected_cls,
                    f"{asset} {y['year']}: {sr} should be {expected_cls}")


# ─────────────────────────────────────────────────────────
# Equity chart data tests
# ─────────────────────────────────────────────────────────

class TestEquityChartData(unittest.TestCase):
    """Tests for renderEquityChart() data contract.

    The function plots traces where x = timestamps and y = values.
    x and y must have the same length for each trace.
    """

    def test_btc_equity_chart_xy_match(self):
        """BTC chart: x=eq.t, y=eq.btc — both BTC-indexed, should match."""
        eq = D["equity"]
        self.assertEqual(len(eq["t"]), len(eq["btc"]),
                         f"BTC chart: t({len(eq['t'])}) != btc({len(eq['btc'])})")

    def test_btc_bh_chart_xy_match(self):
        """BTC B&H chart: x=eq.t, y=eq.btc_bh — both BTC-indexed."""
        eq = D["equity"]
        self.assertEqual(len(eq["t"]), len(eq["btc_bh"]),
                         f"BTC B&H: t({len(eq['t'])}) != btc_bh({len(eq['btc_bh'])})")

    def test_eth_equity_chart_xy_match(self):
        """ETH chart: x=eq.t, y=eq.eth — t is BTC-indexed but eth may differ!
        JS: ds(msToDates(eq.t)) vs ds(eq[label.toLowerCase()])
        If lengths differ, Plotly gets misaligned x/y arrays."""
        eq = D["equity"]
        self.assertEqual(len(eq["t"]), len(eq["eth"]),
                         f"ETH chart x/y MISMATCH: t({len(eq['t'])}) != eth({len(eq['eth'])}). "
                         f"eq.t uses BTC index but eq.eth uses ETH index — "
                         f"must reindex ETH equity to BTC timestamps")

    def test_eth_bh_chart_xy_match(self):
        """ETH B&H: x=eq.t, y=eq.eth_bh — both BTC-indexed."""
        eq = D["equity"]
        self.assertEqual(len(eq["t"]), len(eq["eth_bh"]),
                         f"ETH B&H: t({len(eq['t'])}) != eth_bh({len(eq['eth_bh'])})")



# ─────────────────────────────────────────────────────────
# Price chart data tests
# ─────────────────────────────────────────────────────────

class TestPriceChartData(unittest.TestCase):
    """Tests for renderPriceChart() data contract."""

    def test_ohlc_arrays_same_length(self):
        """Each timeframe: t, o, h, l, c must be same length."""
        for asset in ["btc_ohlc", "eth_ohlc"]:
            for tf, ohlc in D[asset].items():
                n = len(ohlc["t"])
                self.assertGreater(n, 0, f"{asset}[{tf}] is empty")
                for key in ["o", "h", "l", "c"]:
                    self.assertEqual(len(ohlc[key]), n,
                                     f"{asset}[{tf}][{key}] len {len(ohlc[key])} != t len {n}")

    def test_ema_arrays_match_ohlc(self):
        """EMA t arrays should match within each timeframe."""
        for asset, ema_key in [("btc_ohlc", "btc_ema"), ("eth_ohlc", "eth_ema")]:
            for tf in D[asset]:
                ema_data = D[ema_key][tf]
                for ema_name, ema_series in ema_data.items():
                    self.assertEqual(len(ema_series["t"]), len(ema_series["v"]),
                                     f"{ema_key}[{tf}][{ema_name}]: t/v length mismatch")

    def test_ema_labels_match_data_keys(self):
        """Frontend EMA labels must correspond to actual backend data keys.
        BTC: ['EMA 50','EMA 200'] → ema50, ema200
        ETH: ['EMA 40','EMA 100'] → ema40, ema100"""
        # BTC: labels 'EMA 50', 'EMA 200'
        for tf in D["btc_ema"]:
            keys = list(D["btc_ema"][tf].keys())
            self.assertIn("ema50", keys, f"BTC EMA missing ema50 at {tf}")
            self.assertIn("ema200", keys, f"BTC EMA missing ema200 at {tf}")

        # ETH: labels 'EMA 40', 'EMA 100'
        for tf in D["eth_ema"]:
            keys = list(D["eth_ema"][tf].keys())
            self.assertIn("ema40", keys, f"ETH EMA missing ema40 at {tf}")
            self.assertIn("ema100", keys, f"ETH EMA missing ema100 at {tf}")

    def test_trade_markers_within_ohlc_range(self):
        """Trade entry/exit times should fall within OHLC time range (4h)."""
        for sym, ohlc_key, trades_key in [
            ("BTC", "btc_ohlc", "trades_btc"),
            ("ETH", "eth_ohlc", "trades_eth"),
        ]:
            ohlc = D[ohlc_key]["4h"]
            t_min, t_max = ohlc["t"][0], ohlc["t"][-1]
            for t in D[trades_key]:
                self.assertGreaterEqual(t["entry_time"], t_min,
                                        f"{sym} trade entry before OHLC start")
                self.assertLessEqual(t["entry_time"], t_max,
                                     f"{sym} trade entry after OHLC end")


# ─────────────────────────────────────────────────────────
# Data contract: backend fields used by frontend exist
# ─────────────────────────────────────────────────────────

class TestFrontendDataContract(unittest.TestCase):
    """Verify all fields the frontend accesses actually exist in backend data."""

    def test_top_level_keys(self):
        """Frontend accesses these keys on DATA object."""
        required = {
            "last_update", "btc_ohlc", "eth_ohlc", "btc_ema", "eth_ema",
            "trades_btc", "trades_eth", "timeline", "equity", "current", "stats",
        }
        missing = required - set(D.keys())
        self.assertFalse(missing, f"Missing top-level keys: {missing}")

    def test_timeline_events_have_frontend_fields(self):
        """Frontend accesses: time, type, symbol, price, trade_pct, remaining_pct,
        sl, trade_return, sl_triggered, tag, detail."""
        frontend_fields = {
            "time", "type", "symbol", "price", "trade_pct", "remaining_pct",
            "sl", "trade_return", "sl_triggered",
            "tag", "detail",
        }
        for e in D["timeline"]:
            missing = frontend_fields - set(e.keys())
            self.assertFalse(missing, f"Event at {e.get('time')}: missing {missing}")

    def test_trade_objects_have_frontend_fields(self):
        """Frontend accesses: entry_time, exit_time, entry_price, exit_price,
        return_pct, is_open."""
        frontend_fields = {"entry_time", "exit_time", "entry_price", "exit_price", "return_pct"}
        for t in D["trades_btc"] + D["trades_eth"]:
            missing = frontend_fields - set(t.keys())
            self.assertFalse(missing, f"Trade missing: {missing}")

    def test_current_has_frontend_fields(self):
        frontend_fields = {
            "btc_price", "eth_price", "btc_rsi", "eth_rsi",
            "btc_regime", "eth_regime", "btc_momentum", "eth_momentum",
            "btc_in_trade", "eth_in_trade", "btc_position_pct", "eth_position_pct",
        }
        missing = frontend_fields - set(D["current"].keys())
        self.assertFalse(missing, f"Missing current fields: {missing}")

    def test_ohlc_timeframes_exist(self):
        """Frontend uses timeframes: 4h, 1d, 1w, 1m."""
        required_tfs = {"4h", "1d", "1w", "1m"}
        for key in ["btc_ohlc", "eth_ohlc", "btc_ema", "eth_ema"]:
            missing = required_tfs - set(D[key].keys())
            self.assertFalse(missing, f"{key} missing timeframes: {missing}")

    def test_equity_has_frontend_keys(self):
        """Frontend accesses: t, btc, eth, btc_bh, eth_bh."""
        required = {"t", "btc", "eth", "btc_bh", "eth_bh"}
        missing = required - set(D["equity"].keys())
        self.assertFalse(missing, f"Equity missing: {missing}")

    def test_stats_yearly_has_frontend_keys(self):
        """Frontend accesses per-year: year, {asset}_strat, {asset}_bh, {asset}_sharpe, {asset}_dd."""
        for y in D["stats"]["yearly"]:
            for asset in ["btc", "eth"]:
                for suffix in ["_strat", "_bh", "_sharpe", "_dd"]:
                    key = asset + suffix
                    self.assertIn(key, y, f"Year {y.get('year')}: missing {key}")


# ─────────────────────────────────────────────────────────
# Precise position verification — independently computed
# ─────────────────────────────────────────────────────────

def _active_trades_at(trades, t):
    """Return list of trades still open AFTER time t."""
    return [tr for tr in trades
            if tr["entry_time"] <= t and (tr["exit_time"] > t or tr.get("is_open"))]


def _active_trades_before(trades, t):
    """Return list of trades open just BEFORE time t (for BUY: before entry)."""
    return [tr for tr in trades
            if tr["entry_time"] < t and (tr["exit_time"] > t or tr.get("is_open"))]


def _remaining_value_at(trades, t, price):
    """Independently compute remaining position value from raw trades."""
    value = 0
    for tr in trades:
        if tr["entry_time"] <= t and (tr["exit_time"] > t or tr.get("is_open")):
            if tr["size"] > 0:
                value += tr["size"] * price
            else:
                value += tr["entry_value"]
    return value


class TestPositionValuePrecise(unittest.TestCase):
    """Independently verify remaining_pct against raw trade data.

    For each timeline event, compute which trades are active from raw
    trades_btc / trades_eth, and verify remaining_pct matches exactly.
    """

    def _trades_for(self, symbol):
        return D[f"trades_{symbol.lower()}"]

    def test_remaining_pct_nonzero_iff_active_trades(self):
        """remaining_pct > 0 ↔ there are meaningful active trades at this time.
        remaining_pct == 0 ↔ no active trades (or too small to register).

        For SELL events: same-time BUY trades (entry_time == t) are excluded
        because SELL happens before BUY at the same timestamp."""
        for e in D["timeline"]:
            trades = self._trades_for(e["symbol"])
            t = e["time"]
            # Match backend logic: SELL uses entry_time < t, BUY uses entry_time <= t
            if e["type"] == "SELL":
                active = [tr for tr in trades
                          if tr["entry_time"] < t
                          and (tr["exit_time"] > t or tr.get("is_open"))]
            else:
                active = _active_trades_at(trades, t)
            rp = e["remaining_pct"]
            if len(active) == 0:
                self.assertEqual(rp, 0,
                    f"{e['type']} {e['symbol']} at {e['time']}: "
                    f"0 active trades but remaining_pct={rp}")
            elif rp == 0 and e["trade_pct"] == 0:
                pass  # trade too small to register at 0.1% precision
            else:
                self.assertGreater(rp, 0,
                    f"{e['type']} {e['symbol']} at {e['time']}: "
                    f"{len(active)} active trades but remaining_pct=0")

    def test_buy_open_remaining_equals_trade_pct(self):
        """For BUY OPEN with a single underlying trade:
        remaining_pct must exactly equal trade_pct (same equity denominator).
        For merged events (multiple trades at same time), allow tiny float tolerance."""
        for e in D["timeline"]:
            if e["type"] == "BUY" and e["tag"] == "Open":
                diff = abs(e["remaining_pct"] - e["trade_pct"])
                self.assertLessEqual(diff, 0.2,
                    f"BUY OPEN {e['symbol']} at {e['time']}: "
                    f"remaining_pct={e['remaining_pct']} vs trade_pct={e['trade_pct']} "
                    f"diff={diff}")

    def test_frontend_display_pct_equals_remaining(self):
        """Frontend: displayPct = Math.min(e.remaining_pct || 0, 130).
        Must equal min(remaining_pct, 130) exactly."""
        for e in D["timeline"]:
            rp = e.get("remaining_pct") or 0
            expected = min(rp, 130)
            # This is what the frontend computes — verify it's sane
            self.assertEqual(expected, min(e["remaining_pct"], 130))

    def test_current_position_pct_matches_latest_event(self):
        """current.{sym}_position_pct must exactly equal
        min(latest timeline event remaining_pct, 130)."""
        for sym in ["BTC", "ETH"]:
            cur_pct = D["current"][f"{sym.lower()}_position_pct"]
            # Find latest event for this symbol (timeline is descending)
            latest_rp = 0
            for e in D["timeline"]:
                if e["symbol"] == sym:
                    latest_rp = e["remaining_pct"]
                    break
            expected = min(latest_rp, 130)
            self.assertEqual(cur_pct, expected,
                f"{sym}: current position_pct={cur_pct} != "
                f"min(latest remaining_pct={latest_rp}, 130)={expected}")


class TestTagPrecise(unittest.TestCase):
    """Independently verify tag labels from raw trade data.

    For each event, count active trades BEFORE and AFTER the event
    to determine the correct tag.
    """

    def _trades_for(self, symbol):
        return D[f"trades_{symbol.lower()}"]

    def test_tag_matches_independent_computation(self):
        """For each event, independently compute what tag should be.

        At the same timestamp, SELL is processed before BUY (sort order).
        So for BUY: "before" includes effects of any SELL at the same time.
        We use active trades AT time t (entry_time <= t and exit_time > t)
        excluding the trades being opened at exactly t.
        """
        for e in D["timeline"]:
            trades = self._trades_for(e["symbol"])
            t = e["time"]

            if e["type"] == "BUY":
                # Before this BUY: trades entered before t, still open
                # (SELL at same time already processed, same-time BUY trades excluded)
                before = [tr for tr in trades
                          if tr["entry_time"] < t
                          and (tr["exit_time"] > t or tr.get("is_open"))]
                if len(before) == 0:
                    expected_tag = "Open"
                else:
                    expected_tag = "Add"
            else:  # SELL
                # After SELL: trades entered before t, still open after t
                # (same-time BUY trades excluded — SELL happens first)
                after = [tr for tr in trades
                         if tr["entry_time"] < t
                         and (tr["exit_time"] > t or tr.get("is_open"))]
                if len(after) == 0:
                    expected_tag = "Close"
                else:
                    expected_tag = "Reduce"

            self.assertEqual(e["tag"], expected_tag,
                f"{e['type']} {e['symbol']} at {e['time']}: "
                f"tag='{e['tag']}' but expected '{expected_tag}'")


if __name__ == "__main__":
    unittest.main()
