"""BTC strategy: EMA trend filter + adaptive vol sizing + drawdown control.

Logic:
- Entry: price > EMA200 * (1 + entry_buffer), EMA200 not falling
- Exit: price < EMA200 * (1 - exit_buffer) AND EMA200 is falling
- Sizing: volatility-adaptive (low vol = bigger, high vol = smaller), scaled by leverage
- Drawdown control: equity drops 12% from peak → reduce to 25% position
- Recovery: equity recovers to 95% of peak → restore full position
"""

from backtesting import Strategy
from core import config


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
    signals["Strong Trend"] = "YES" if strong_trend else "NO"
    signals["MACD Histogram"] = f"{macd_h:.2f}" + (" BULLISH" if macd_h > 0 else " BEARISH")
    signals["RSI"] = f"{rsi:.1f}"
    signals["ATR"] = f"${atr:,.2f}"

    if price > ema200:
        signals["Mode"] = "BULL — hold through corrections"
        recommendation = "BUY" if strong_trend and rsi < 80 else "HOLD"
    else:
        signals["Mode"] = "BEAR — trailing stop active"
        signals["Chandelier SL"] = f"${price - atr * 2.5:,.2f}"
        recommendation = "SELL / CASH" if macd_h < 0 else "CAUTION"

    return {
        "symbol": symbol, "time": str(df.index[-1]), "price": price, "rsi": rsi,
        "signals": signals, "recommendation": recommendation,
        "conditions_met": "BULL" if price > ema200 else "BEAR",
    }


class RobustTrendStrategy(Strategy):
    """EMA trend filter with adaptive sizing and drawdown control."""

    ema_col = "EMA_200"
    slope_bars = 3
    exit_buffer = 0.01
    entry_buffer = 0.005
    max_size = 1.0
    min_size = 0.30
    vol_lookback = 80
    dd_reduce = 0.12
    reduce_size = 0.25
    cooldown = 3
    leverage = 1.3

    def init(self):
        self.highest_equity = self.equity
        self.reduced = False
        self.exit_bar = -999
        self.was_in_position = False
        self.entry_log = []

    def _vol_size(self, df):
        """Adaptive position size: smaller in high vol, bigger in low vol."""
        cap = 0.99  # Framework requires size < 1; margin handles leverage
        if len(df) < self.vol_lookback:
            return cap
        vol = df["Close"].pct_change().iloc[-self.vol_lookback:].std()
        median_vol = 0.015
        ratio = median_vol / max(vol, 0.005)
        base = self.min_size + (self.max_size - self.min_size) * min(ratio, 1.5) / 1.5
        base = min(max(base, self.min_size), self.max_size)
        return min(round(base, 2), cap)

    def next(self):
        cur_bar = len(self.data) - 1
        # Detect framework SL exit
        if self.was_in_position and not self.position:
            self.exit_bar = cur_bar
        self.was_in_position = bool(self.position)

        price = self.data.Close[-1]
        df = self.data.df
        ema = df[self.ema_col].iloc[-1]

        if self.position:
            if self.equity > self.highest_equity:
                self.highest_equity = self.equity

            # Check exit first — don't modify position if we're about to exit
            below = price < ema * (1 - self.exit_buffer)
            ema_prev = df[self.ema_col].iloc[-self.slope_bars] if len(df) > self.slope_bars else ema
            if below and ema < ema_prev:
                sl_at_exit = ema * (1 - self.exit_buffer)
                self.position.close()
                self.reduced = False
                self.exit_bar = cur_bar
                self.entry_log.append({
                    "time": self.data.index[-1], "price": price,
                    "sl": sl_at_exit, "size": 0, "action": "close",
                })
                return

            # Drawdown control (only if not exiting)
            if self.dd_reduce < 1.0:
                dd = (self.highest_equity - self.equity) / self.highest_equity
                if dd > self.dd_reduce and not self.reduced:
                    cur = self.position.size * price / self.equity
                    rs = self.reduce_size
                    if cur > rs + 0.1:
                        self.sell(size=(cur - rs) / cur)
                        self.reduced = True
                if self.reduced and self.equity >= self.highest_equity * 0.95:
                    cur = self.position.size * price / self.equity
                    add = self._vol_size(df) - cur
                    if add > 0.05:
                        self.buy(size=add)
                    self.reduced = False
        else:
            # Entry: price above EMA (with cooldown)
            if cur_bar - self.exit_bar < self.cooldown:
                return
            if price > ema * (1 + self.entry_buffer):
                size = self._vol_size(df)
                self.buy(size=size)
                self.highest_equity = self.equity
                self.reduced = False
                self.entry_log.append({
                    "time": self.data.index[-1], "price": price,
                    "sl": ema * (1 - self.exit_buffer), "size": size,
                })
