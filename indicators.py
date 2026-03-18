"""Compute technical indicators using the ta library."""

import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

import config


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the DataFrame.

    Expects columns: Open, High, Low, Close, Volume with DatetimeIndex.
    Returns a copy with indicator columns added and warmup rows dropped.
    """
    df = df.copy()

    # EMAs
    df[f"EMA_{config.EMA_FAST}"] = EMAIndicator(df["Close"], window=config.EMA_FAST).ema_indicator()
    df[f"EMA_{config.EMA_MID}"] = EMAIndicator(df["Close"], window=config.EMA_MID).ema_indicator()
    df[f"EMA_{config.EMA_SLOW}"] = EMAIndicator(df["Close"], window=config.EMA_SLOW).ema_indicator()

    # RSI
    df[f"RSI_{config.RSI_PERIOD}"] = RSIIndicator(df["Close"], window=config.RSI_PERIOD).rsi()

    # MACD
    macd = MACD(df["Close"], window_slow=config.MACD_SLOW, window_fast=config.MACD_FAST, window_sign=config.MACD_SIGNAL)
    df[f"MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"] = macd.macd()
    df[f"MACDs_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"] = macd.macd_signal()
    df[f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(df["Close"], window=config.BB_PERIOD, window_dev=config.BB_STD)
    df[f"BBL_{config.BB_PERIOD}_{config.BB_STD}"] = bb.bollinger_lband()
    df[f"BBM_{config.BB_PERIOD}_{config.BB_STD}"] = bb.bollinger_mavg()
    df[f"BBU_{config.BB_PERIOD}_{config.BB_STD}"] = bb.bollinger_hband()

    # ATR for stop loss
    atr = AverageTrueRange(df["High"], df["Low"], df["Close"], window=config.ATR_PERIOD)
    df[f"ATR_{config.ATR_PERIOD}"] = atr.average_true_range()

    # Volume MA (simple moving average)
    df[f"VOL_MA_{config.VOLUME_MA_PERIOD}"] = df["Volume"].rolling(window=config.VOLUME_MA_PERIOD).mean()

    # Drop warmup rows where indicators are NaN
    df.dropna(inplace=True)

    return df
