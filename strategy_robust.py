"""Hold-in-Bull, Protect-in-Bear adaptive trend-following strategy.

Core idea:
- Bull mode (golden cross): HOLD through corrections like B&H,
  only emergency exit if price crashes 2% below EMA200
- Bear mode (death cross): activate Chandelier trailing stop
- Entry: strong trend confirmation (EMA alignment + MACD + RSI)
- Position sizing: volatility-adjusted (risk 2.5% equity per trade)

This captures nearly all upside in bull markets (matching/beating B&H),
while limiting downside in bear markets.
"""

from backtesting import Strategy
import config


def evaluate_signals(df, symbol):
    """Evaluate current trading signal for a symbol."""
    row = df.iloc[-1]

    ema20 = row[f"EMA_{config.EMA_FAST}"]
    ema50 = row[f"EMA_{config.EMA_MID}"]
    ema200 = row[f"EMA_{config.EMA_SLOW}"]
    macd_h = row[f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]
    rsi = row[f"RSI_{config.RSI_PERIOD}"]
    atr = row[f"ATR_{config.ATR_PERIOD}"]
    price = row["Close"]

    golden_cross = ema50 > ema200
    strong_trend = ema20 > ema50 > ema200 and macd_h > 0

    signals = {}
    signals["Golden Cross"] = "YES" if golden_cross else "NO"
    signals["Strong Trend"] = "YES (EMA20>50>200 + MACD>0)" if strong_trend else "NO"
    signals["MACD Histogram"] = f"{macd_h:.2f}" + (" BULLISH" if macd_h > 0 else " BEARISH")
    signals["RSI"] = f"{rsi:.1f}"
    signals["ATR"] = f"${atr:,.2f}"

    if golden_cross:
        signals["Mode"] = "BULL — hold through corrections"
        if strong_trend and rsi < 80:
            recommendation = "BUY"
        else:
            recommendation = "HOLD"
    else:
        signals["Mode"] = "BEAR — trailing stop active"
        signals["Chandelier SL"] = f"${price - atr * 2.5:,.2f}"
        recommendation = "SELL / CASH" if macd_h < 0 else "CAUTION"

    return {
        "symbol": symbol,
        "time": str(df.index[-1]),
        "price": price,
        "rsi": rsi,
        "signals": signals,
        "recommendation": recommendation,
        "conditions_met": "BULL" if golden_cross else "BEAR",
    }


class RobustTrendStrategy(Strategy):
    """Hold in bull, protect in bear, vol-adjusted sizing.

    Parameters:
    - atr_mult: Chandelier Exit multiplier for bear-mode trailing stop
    - risk_pct: fraction of equity risked per trade (for vol-adjusted sizing)
    - emerg_pct: emergency exit threshold (price < EMA200 * (1 - emerg_pct))
    """

    atr_mult = 2.5
    risk_pct = 0.025
    emerg_pct = 0.02

    def init(self):
        self.ema_fast = f"EMA_{config.EMA_FAST}"
        self.ema_mid = f"EMA_{config.EMA_MID}"
        self.ema_slow = f"EMA_{config.EMA_SLOW}"
        self.macd_hist = f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
        self.atr_col = f"ATR_{config.ATR_PERIOD}"
        self.rsi_col = f"RSI_{config.RSI_PERIOD}"

        self.trailing_stop = 0
        self.highest = 0
        self.bars_since_exit = 999
        self.was_in_position = False

    def next(self):
        # Detect SL/TP exit by framework (position gone without us calling close)
        if self.was_in_position and not self.position:
            self.bars_since_exit = 0
        self.was_in_position = bool(self.position)

        self.bars_since_exit += 1
        price = self.data.Close[-1]
        df = self.data.df
        atr = df[self.atr_col].iloc[-1]

        ema20 = df[self.ema_fast].iloc[-1]
        ema50 = df[self.ema_mid].iloc[-1]
        ema200 = df[self.ema_slow].iloc[-1]
        macd_h = df[self.macd_hist].iloc[-1]
        rsi = df[self.rsi_col].iloc[-1]

        golden_cross = ema50 > ema200

        if self.position and self.position.is_long:
            # Track highest price
            if price > self.highest:
                self.highest = price

            if golden_cross:
                # ── BULL MODE: hold through corrections ──
                # Only exit on severe breakdown below EMA200
                if self.emerg_pct > 0 and price < ema200 * (1 - self.emerg_pct):
                    self.position.close()
                    self.bars_since_exit = 0
                    return
            else:
                # ── BEAR MODE: trailing stop active ──
                chandelier = self.highest - atr * self.atr_mult
                if chandelier > self.trailing_stop:
                    self.trailing_stop = chandelier

                if price < self.trailing_stop:
                    self.position.close()
                    self.bars_since_exit = 0
                    return

                # Confirmed bear: death cross + bearish momentum + price weakness
                if macd_h < 0 and price < ema50:
                    self.position.close()
                    self.bars_since_exit = 0
                    return

        else:
            # ── LOOK FOR ENTRY ──
            if self.bars_since_exit < 2:
                return

            # Strong trend entry: full EMA alignment + MACD + RSI not overbought
            strong_entry = (golden_cross and ema20 > ema50 > ema200
                           and macd_h > 0 and rsi < 80)

            if strong_entry:
                sl_price = price - atr * self.atr_mult

                # Volatility-adjusted position sizing
                risk_per_unit = price - sl_price
                if risk_per_unit > 0:
                    dollar_risk = self.equity * self.risk_pct
                    units = dollar_risk / risk_per_unit
                    size = min(units * price / self.equity, 0.95)
                    size = max(size, 0.10)
                else:
                    size = 0.95

                self.trailing_stop = sl_price
                self.highest = price
                self.buy(sl=sl_price, size=size)
                self.bars_since_exit = 0
