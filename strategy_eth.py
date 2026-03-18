"""ETH-specific strategy with shorter EMA cycles tuned for ETH's faster trends.

ETH trends are shorter and more volatile than BTC, so this uses:
- Shorter EMAs: 15/40/100 instead of 20/50/200
- Tighter position sizing: 1.2% risk per trade
- Same hold-in-bull, protect-in-bear core logic
"""

from backtesting import Strategy
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import config


# ETH-specific EMA periods
ETH_EMA_FAST = 15
ETH_EMA_MID = 40
ETH_EMA_SLOW = 100


def add_eth_indicators(df):
    """Add ETH-specific indicators (shorter EMAs)."""
    df = df.copy()
    df[f"EMA_{ETH_EMA_FAST}"] = EMAIndicator(df["Close"], window=ETH_EMA_FAST).ema_indicator()
    df[f"EMA_{ETH_EMA_MID}"] = EMAIndicator(df["Close"], window=ETH_EMA_MID).ema_indicator()
    df[f"EMA_{ETH_EMA_SLOW}"] = EMAIndicator(df["Close"], window=ETH_EMA_SLOW).ema_indicator()
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
    ema15 = row[f"EMA_{ETH_EMA_FAST}"]
    ema40 = row[f"EMA_{ETH_EMA_MID}"]
    ema100 = row[f"EMA_{ETH_EMA_SLOW}"]
    macd_h = row[f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]
    rsi = row[f"RSI_{config.RSI_PERIOD}"]
    atr = row[f"ATR_{config.ATR_PERIOD}"]
    price = row["Close"]

    golden_cross = ema40 > ema100
    strong = ema15 > ema40 > ema100 and macd_h > 0

    signals = {}
    signals["Golden Cross (40/100)"] = "YES" if golden_cross else "NO"
    signals["Strong Trend"] = "YES (EMA15>40>100 + MACD>0)" if strong else "NO"
    signals["MACD Histogram"] = f"{macd_h:.2f}" + (" BULLISH" if macd_h > 0 else " BEARISH")
    signals["RSI"] = f"{rsi:.1f}"

    if golden_cross:
        signals["Mode"] = "BULL — hold through corrections"
        recommendation = "BUY" if strong and rsi < 80 else "HOLD"
    else:
        signals["Mode"] = "BEAR — trailing stop active"
        signals["Chandelier SL"] = f"${price - atr * 2.0:,.2f}"
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


class ETHTrendStrategy(Strategy):
    """ETH-specific: shorter EMAs, tighter sizing."""

    atr_mult = 2.0
    risk_pct = 0.012
    emerg_pct = 0.02

    def init(self):
        self.ema_fast = f"EMA_{ETH_EMA_FAST}"
        self.ema_mid = f"EMA_{ETH_EMA_MID}"
        self.ema_slow = f"EMA_{ETH_EMA_SLOW}"
        self.macd_hist = f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
        self.atr_col = f"ATR_{config.ATR_PERIOD}"
        self.rsi_col = f"RSI_{config.RSI_PERIOD}"
        self.trailing_stop = 0
        self.highest = 0
        self.bars_since_exit = 999
        self.was_in_position = False

    def next(self):
        if self.was_in_position and not self.position:
            self.bars_since_exit = 0
        self.was_in_position = bool(self.position)

        self.bars_since_exit += 1
        price = self.data.Close[-1]
        df = self.data.df
        atr = df[self.atr_col].iloc[-1]
        ema_f = df[self.ema_fast].iloc[-1]
        ema_m = df[self.ema_mid].iloc[-1]
        ema_s = df[self.ema_slow].iloc[-1]
        macd_h = df[self.macd_hist].iloc[-1]
        rsi = df[self.rsi_col].iloc[-1]
        golden_cross = ema_m > ema_s

        if self.position and self.position.is_long:
            if price > self.highest:
                self.highest = price
            if golden_cross:
                if self.emerg_pct > 0 and price < ema_s * (1 - self.emerg_pct):
                    self.position.close()
                    self.bars_since_exit = 0
                    return
            else:
                chandelier = self.highest - atr * self.atr_mult
                if chandelier > self.trailing_stop:
                    self.trailing_stop = chandelier
                if price < self.trailing_stop:
                    self.position.close()
                    self.bars_since_exit = 0
                    return
                if macd_h < 0 and price < ema_m:
                    self.position.close()
                    self.bars_since_exit = 0
                    return
        else:
            if self.bars_since_exit < 2:
                return
            strong = ema_f > ema_m > ema_s and macd_h > 0 and rsi < 80
            if golden_cross and strong:
                sl_price = price - atr * self.atr_mult
                risk_per = price - sl_price
                if risk_per > 0:
                    size = min(self.equity * self.risk_pct / risk_per * price / self.equity, 0.95)
                    size = max(size, 0.05)
                else:
                    size = 0.95
                self.trailing_stop = sl_price
                self.highest = price
                self.buy(sl=sl_price, size=size)
                self.bars_since_exit = 0
