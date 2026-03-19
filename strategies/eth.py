"""ETH strategy: EMA150 trend filter + adaptive vol sizing + drawdown control.

Same logic as BTC but with:
- EMA150 instead of EMA200 (ETH has shorter cycles)
- Wider exit buffer (0.015 vs 0.01)
- Longer slope check (5 bars vs 3)
- Higher median vol reference (0.02 vs 0.015)
"""

from backtesting import Strategy
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from core import config

ETH_EMA_FAST = 15
ETH_EMA_MID = 40
ETH_EMA_SLOW = 100


def add_eth_indicators(df):
    """Add ETH-specific indicators."""
    df = df.copy()
    df[f"EMA_{ETH_EMA_FAST}"] = EMAIndicator(df["Close"], window=ETH_EMA_FAST).ema_indicator()
    df[f"EMA_{ETH_EMA_MID}"] = EMAIndicator(df["Close"], window=ETH_EMA_MID).ema_indicator()
    df[f"EMA_{ETH_EMA_SLOW}"] = EMAIndicator(df["Close"], window=ETH_EMA_SLOW).ema_indicator()
    df["EMA_150"] = EMAIndicator(df["Close"], window=150).ema_indicator()
    df[f"RSI_{config.RSI_PERIOD}"] = RSIIndicator(df["Close"], window=config.RSI_PERIOD).rsi()
    macd = MACD(df["Close"], window_slow=config.MACD_SLOW, window_fast=config.MACD_FAST, window_sign=config.MACD_SIGNAL)
    df[f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"] = macd.macd_diff()
    df[f"ATR_{config.ATR_PERIOD}"] = AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=config.ATR_PERIOD
    ).average_true_range()
    df.dropna(inplace=True)
    return df


def evaluate_signals(df, symbol):
    """Evaluate current ETH signal."""
    row = df.iloc[-1]
    ema40 = row[f"EMA_{ETH_EMA_MID}"]
    ema100 = row[f"EMA_{ETH_EMA_SLOW}"]
    ema150 = row["EMA_150"]
    macd_h = row[f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]
    rsi = row[f"RSI_{config.RSI_PERIOD}"]
    atr = row[f"ATR_{config.ATR_PERIOD}"]
    price = row["Close"]

    above_ema = price > ema150
    strong = row[f"EMA_{ETH_EMA_FAST}"] > ema40 > ema100 and macd_h > 0

    signals = {}
    signals["Price vs EMA150"] = "ABOVE" if above_ema else "BELOW"
    signals["Strong Trend"] = "YES (EMA15>40>100 + MACD>0)" if strong else "NO"
    signals["MACD Histogram"] = f"{macd_h:.2f}" + (" BULLISH" if macd_h > 0 else " BEARISH")
    signals["RSI"] = f"{rsi:.1f}"

    if above_ema:
        signals["Mode"] = "BULL — hold through corrections"
        recommendation = "BUY" if strong and rsi < 80 else "HOLD"
    else:
        signals["Mode"] = "BEAR — trailing stop active"
        recommendation = "SELL / CASH" if macd_h < 0 else "CAUTION"

    return {
        "symbol": symbol, "time": str(df.index[-1]), "price": price, "rsi": rsi,
        "signals": signals, "recommendation": recommendation,
        "conditions_met": "BULL" if above_ema else "BEAR",
    }


class ETHTrendStrategy(Strategy):
    """ETH: EMA150 trend filter with adaptive sizing and drawdown control."""

    ema_col = "EMA_150"
    slope_bars = 5
    exit_buffer = 0.015
    entry_buffer = 0.005
    max_size = 0.50
    min_size = 0.20
    vol_lookback = 80
    dd_reduce = 0.12
    reduce_size = 0.20
    cooldown = 3
    leverage = 1.3

    def init(self):
        self.highest_equity = self.equity
        self.reduced = False
        self.bars_since_exit = 999
        self.was_in_position = False
        self.entry_log = []

    def _vol_size(self, df):
        if len(df) < self.vol_lookback:
            return min(self.max_size * self.leverage, 0.99)
        vol = df["Close"].pct_change().iloc[-self.vol_lookback:].std()
        median_vol = 0.02  # ETH is more volatile
        ratio = median_vol / max(vol, 0.005)
        base = self.min_size + (self.max_size - self.min_size) * min(ratio, 1.5) / 1.5
        base = min(max(base, self.min_size), self.max_size)
        return min(round(base * self.leverage, 2), 0.99)

    def next(self):
        if self.was_in_position and not self.position:
            self.bars_since_exit = 0
        self.was_in_position = bool(self.position)
        self.bars_since_exit += 1

        price = self.data.Close[-1]
        df = self.data.df
        ema = df[self.ema_col].iloc[-1]

        if self.position:
            if self.equity > self.highest_equity:
                self.highest_equity = self.equity

            if self.dd_reduce < 1.0:
                dd = (self.highest_equity - self.equity) / self.highest_equity
                if dd > self.dd_reduce and not self.reduced:
                    cur = self.position.size * price / self.equity
                    rs = self.reduce_size * self.leverage
                    if cur > rs + 0.1:
                        self.sell(size=(cur - rs) / cur)
                        self.reduced = True
                if self.reduced and self.equity >= self.highest_equity * 0.95:
                    cur = self.position.size * price / self.equity
                    add = self._vol_size(df) - cur
                    if add > 0.05:
                        self.buy(size=add)
                    self.reduced = False

            below = price < ema * (1 - self.exit_buffer)
            ema_prev = df[self.ema_col].iloc[-self.slope_bars] if len(df) > self.slope_bars else ema
            if below and ema < ema_prev:
                self.position.close()
                self.reduced = False
                self.entry_log.append({
                    "time": self.data.index[-1], "price": price,
                    "sl": ema * (1 - self.exit_buffer), "size": 0, "action": "close",
                })
                return
        else:
            if self.bars_since_exit < self.cooldown:
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
