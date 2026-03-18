"""Multi-indicator confluence strategy for signal generation and backtesting.

Core idea: trend-following. Stay in the market during uptrends,
exit only when multiple indicators confirm a trend reversal.
Re-enter quickly when conditions improve.
"""

import pandas as pd
import numpy as np
from backtesting import Strategy

from core import config

# Ensure config has all needed attrs
if not hasattr(config, "USE_TRAILING_STOP"):
    config.USE_TRAILING_STOP = True
if not hasattr(config, "EXIT_BEAR_THRESHOLD"):
    config.EXIT_BEAR_THRESHOLD = 3
if not hasattr(config, "ENTRY_BULL_THRESHOLD"):
    config.ENTRY_BULL_THRESHOLD = 3


def _score_bar(row):
    """Score a single bar for bull/bear confluence. Returns (bull, bear, signals)."""
    ema_fast = f"EMA_{config.EMA_FAST}"
    ema_mid = f"EMA_{config.EMA_MID}"
    ema_slow = f"EMA_{config.EMA_SLOW}"
    rsi_col = f"RSI_{config.RSI_PERIOD}"
    macd_hist = f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
    bbl_col = f"BBL_{config.BB_PERIOD}_{config.BB_STD}"
    bbu_col = f"BBU_{config.BB_PERIOD}_{config.BB_STD}"
    vol_ma = f"VOL_MA_{config.VOLUME_MA_PERIOD}"

    signals = {}
    bull = 0
    bear = 0

    # 1. EMA Trend alignment
    if row[ema_fast] > row[ema_mid] > row[ema_slow]:
        bull += 1
        signals["EMA"] = "BULLISH"
    elif row[ema_fast] < row[ema_mid] < row[ema_slow]:
        bear += 1
        signals["EMA"] = "BEARISH"
    else:
        signals["EMA"] = "NEUTRAL"

    # 2. RSI
    if row[rsi_col] < config.RSI_OVERSOLD:
        bull += 1
        signals["RSI"] = "OVERSOLD (BULLISH)"
    elif row[rsi_col] > config.RSI_OVERBOUGHT:
        bear += 1
        signals["RSI"] = "OVERBOUGHT (BEARISH)"
    elif row[rsi_col] < 45:
        bull += 0.5
        signals["RSI"] = "LEANING BULLISH"
    elif row[rsi_col] > 55:
        bear += 0.5
        signals["RSI"] = "LEANING BEARISH"
    else:
        signals["RSI"] = "NEUTRAL"

    # 3. MACD histogram
    if row[macd_hist] > 0:
        bull += 1
        signals["MACD"] = "BULLISH"
    else:
        bear += 1
        signals["MACD"] = "BEARISH"

    # 4. Bollinger Bands position
    bb_range = row[bbu_col] - row[bbl_col]
    if bb_range > 0:
        bb_pos = (row["Close"] - row[bbl_col]) / bb_range
        if bb_pos < 0.2:
            bull += 1
            signals["BB"] = "NEAR LOWER (BULLISH)"
        elif bb_pos > 0.8:
            bear += 1
            signals["BB"] = "NEAR UPPER (BEARISH)"
        else:
            signals["BB"] = "NEUTRAL"
    else:
        signals["BB"] = "NEUTRAL"

    # 5. Volume confirmation
    if row["Volume"] > row[vol_ma] * 1.2:
        if bull > bear:
            bull += 1
            signals["Volume"] = "HIGH (CONFIRMS BULL)"
        elif bear > bull:
            bear += 1
            signals["Volume"] = "HIGH (CONFIRMS BEAR)"
        else:
            signals["Volume"] = "HIGH (NO DIRECTION)"
    else:
        signals["Volume"] = "LOW"

    # 6. Price vs EMA200
    ema_slow_val = f"EMA_{config.EMA_SLOW}"
    if row["Close"] > row[ema_slow_val]:
        bull += 1
        signals["Trend"] = "ABOVE EMA200 (BULLISH)"
    else:
        bear += 1
        signals["Trend"] = "BELOW EMA200 (BEARISH)"

    return bull, bear, signals


def score_signals(df: pd.DataFrame, idx: int = -1) -> dict:
    """Score the confluence of indicators at a given bar index."""
    row = df.iloc[idx]
    bull, bear, signals = _score_bar(row)

    entry_thresh = getattr(config, "ENTRY_BULL_THRESHOLD", config.BUY_THRESHOLD)
    exit_thresh = getattr(config, "EXIT_BEAR_THRESHOLD", config.SELL_THRESHOLD)

    if bull >= entry_thresh:
        recommendation = "BUY"
    elif bear >= exit_thresh:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    return {
        "bull_score": bull,
        "bear_score": bear,
        "signals": signals,
        "recommendation": recommendation,
        "price": row["Close"],
        "rsi": row[f"RSI_{config.RSI_PERIOD}"],
    }


def evaluate_signals(df: pd.DataFrame, symbol: str) -> dict:
    """Evaluate current signals for a symbol."""
    result = score_signals(df, idx=-1)
    result["symbol"] = symbol
    result["time"] = str(df.index[-1])
    return result


class ConfluenceStrategy(Strategy):
    """Trend-following strategy: stay long in uptrends, exit on confirmed reversals.

    Key difference from typical strategies:
    - Enters aggressively when trend is up (low entry threshold)
    - Exits conservatively (needs strong bearish confirmation)
    - Uses trailing stop to protect gains without capping upside
    - Stays in the market as much as possible during uptrends
    """

    # Entry/exit thresholds (asymmetric: easy to enter, hard to exit)
    entry_bull_threshold = config.ENTRY_BULL_THRESHOLD
    exit_bear_threshold = config.EXIT_BEAR_THRESHOLD
    long_only = config.LONG_ONLY
    cooldown = config.TRADE_COOLDOWN
    use_trailing = config.USE_TRAILING_STOP
    atr_sl_mult = config.ATR_SL_MULTIPLIER
    max_risk_pct = config.MAX_RISK_PER_TRADE

    def init(self):
        self.ema_fast = f"EMA_{config.EMA_FAST}"
        self.ema_mid = f"EMA_{config.EMA_MID}"
        self.ema_slow = f"EMA_{config.EMA_SLOW}"
        self.rsi_col = f"RSI_{config.RSI_PERIOD}"
        self.macd_hist = f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
        self.bbl_col = f"BBL_{config.BB_PERIOD}_{config.BB_STD}"
        self.bbu_col = f"BBU_{config.BB_PERIOD}_{config.BB_STD}"
        self.vol_ma = f"VOL_MA_{config.VOLUME_MA_PERIOD}"
        self.atr_col = f"ATR_{config.ATR_PERIOD}"
        self.bars_since_trade = 999
        self.trailing_stop = 0
        self.highest_since_entry = 0

    def _score(self):
        price = self.data.Close[-1]
        df = self.data.df
        bull, bear = 0, 0

        if df[self.ema_fast].iloc[-1] > df[self.ema_mid].iloc[-1] > df[self.ema_slow].iloc[-1]:
            bull += 1
        elif df[self.ema_fast].iloc[-1] < df[self.ema_mid].iloc[-1] < df[self.ema_slow].iloc[-1]:
            bear += 1

        rsi = df[self.rsi_col].iloc[-1]
        if rsi < config.RSI_OVERSOLD:
            bull += 1
        elif rsi > config.RSI_OVERBOUGHT:
            bear += 1
        elif rsi < 45:
            bull += 0.5
        elif rsi > 55:
            bear += 0.5

        if df[self.macd_hist].iloc[-1] > 0:
            bull += 1
        else:
            bear += 1

        bbl = df[self.bbl_col].iloc[-1]
        bbu = df[self.bbu_col].iloc[-1]
        bb_range = bbu - bbl
        if bb_range > 0:
            bb_pos = (price - bbl) / bb_range
            if bb_pos < 0.2:
                bull += 1
            elif bb_pos > 0.8:
                bear += 1

        if self.data.Volume[-1] > df[self.vol_ma].iloc[-1] * 1.2:
            if bull > bear:
                bull += 1
            elif bear > bull:
                bear += 1

        ema200 = df[self.ema_slow].iloc[-1]
        trend_bullish = price > ema200
        if trend_bullish:
            bull += 1
        else:
            bear += 1

        atr = df[self.atr_col].iloc[-1]
        return bull, bear, trend_bullish, atr

    def next(self):
        self.bars_since_trade += 1
        price = self.data.Close[-1]
        bull, bear, trend_bullish, atr = self._score()

        if self.position and self.position.is_long:
            # Track highest price for trailing stop
            if price > self.highest_since_entry:
                self.highest_since_entry = price

            # Update trailing stop
            if self.use_trailing:
                new_trail = self.highest_since_entry - atr * self.atr_sl_mult
                if new_trail > self.trailing_stop:
                    self.trailing_stop = new_trail
                if price < self.trailing_stop:
                    self.position.close()
                    self.bars_since_trade = 0
                    return

            # Exit on strong bearish reversal + trend break
            if bear >= self.exit_bear_threshold and not trend_bullish:
                self.position.close()
                self.bars_since_trade = 0

        elif not self.position:
            if self.bars_since_trade < self.cooldown:
                return

            # Enter long when bullish confluence
            if bull >= self.entry_bull_threshold and trend_bullish:
                sl_price = price - atr * self.atr_sl_mult

                # Volatility-adjusted position sizing:
                # risk_per_share = distance to stop loss
                # size = (equity * max_risk%) / risk_per_share
                # This reduces position size when volatility is high
                risk_per_share = price - sl_price
                if risk_per_share > 0:
                    max_risk_pct = self.max_risk_pct
                    equity = self.equity
                    dollar_risk = equity * max_risk_pct
                    shares = dollar_risk / risk_per_share
                    size = min(shares * price / equity, config.BACKTEST_TRADE_SIZE)
                    size = max(size, 0.05)  # Minimum 5% position
                else:
                    size = config.BACKTEST_TRADE_SIZE

                self.trailing_stop = sl_price
                self.highest_since_entry = price
                self.buy(sl=sl_price, size=size)
                self.bars_since_trade = 0
