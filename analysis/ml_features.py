"""Feature engineering for ML strategy.

Builds features from:
1. Technical indicators (EMA, RSI, MACD, BB, ATR)
2. Price action (returns, volatility, momentum)
3. External data (Fear & Greed, Funding Rate)
4. Cross-asset (BTC trend for ETH)
"""

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange


def build_features(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Build ML feature matrix from OHLCV + external data.

    Returns DataFrame with feature columns and optionally target column.
    """
    feat = pd.DataFrame(index=df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ── Price action features ──
    for period in [6, 12, 24, 48, 72]:  # 1d, 2d, 4d, 8d, 12d in 4H bars
        feat[f"ret_{period}"] = close.pct_change(period)
    feat["ret_1"] = close.pct_change(1)

    # Volatility (rolling std of returns)
    for period in [12, 24, 48]:
        feat[f"vol_{period}"] = close.pct_change().rolling(period).std()

    # High-low range relative to close
    feat["hl_ratio"] = (high - low) / close

    # Volume features
    vol_ma = volume.rolling(20).mean()
    feat["vol_ratio"] = volume / vol_ma
    feat["vol_change"] = volume.pct_change(6)

    # ── Trend indicators ──
    for period in [10, 20, 50, 100, 200]:
        ema = EMAIndicator(close, window=period).ema_indicator()
        feat[f"ema_dist_{period}"] = (close - ema) / ema  # Distance from EMA as %

    # EMA alignment score: how many short EMAs are above long ones
    ema10 = EMAIndicator(close, window=10).ema_indicator()
    ema20 = EMAIndicator(close, window=20).ema_indicator()
    ema50 = EMAIndicator(close, window=50).ema_indicator()
    ema100 = EMAIndicator(close, window=100).ema_indicator()
    ema200 = EMAIndicator(close, window=200).ema_indicator()
    feat["ema_align"] = (
        (ema10 > ema20).astype(int) +
        (ema20 > ema50).astype(int) +
        (ema50 > ema100).astype(int) +
        (ema100 > ema200).astype(int)
    )
    feat["golden_cross"] = (ema50 > ema200).astype(int)

    # ADX (trend strength)
    adx = ADXIndicator(high, low, close, window=14)
    feat["adx"] = adx.adx()
    feat["di_plus"] = adx.adx_pos()
    feat["di_minus"] = adx.adx_neg()
    feat["di_diff"] = feat["di_plus"] - feat["di_minus"]

    # ── Momentum indicators ──
    rsi = RSIIndicator(close, window=14).rsi()
    feat["rsi"] = rsi
    feat["rsi_ma"] = rsi.rolling(12).mean()
    feat["rsi_change"] = rsi.diff(6)

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    feat["macd_hist"] = macd.macd_diff()
    feat["macd_hist_change"] = feat["macd_hist"].diff(3)

    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    feat["stoch_k"] = stoch.stoch()
    feat["stoch_d"] = stoch.stoch_signal()

    # ── Volatility indicators ──
    bb = BollingerBands(close, window=20, window_dev=2)
    feat["bb_position"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    feat["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

    atr = AverageTrueRange(high, low, close, window=14).average_true_range()
    feat["atr_pct"] = atr / close  # Relative ATR

    # ── External data ──
    if "FNG" in df.columns:
        feat["fng"] = df["FNG"]
        feat["fng_ma"] = df["FNG"].rolling(42, min_periods=1).mean()  # 7 days
        feat["fng_change"] = df["FNG"].diff(6)
    if "FundingRate" in df.columns:
        feat["funding"] = df["FundingRate"]
        feat["funding_ma"] = df["FundingRate"].rolling(21, min_periods=1).mean()
    if "BTC_Trend" in df.columns:
        feat["btc_trend"] = df["BTC_Trend"]

    # ── Target: forward return (next 6 bars = 1 day) ──
    if include_target:
        feat["target_ret"] = close.pct_change(6).shift(-6)
        # Binary classification: positive return = 1
        feat["target"] = (feat["target_ret"] > 0).astype(int)

    feat.dropna(inplace=True)
    return feat
