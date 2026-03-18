"""Tunable parameters for the crypto day trading signal system.

Optimized via 10-round grid search on 3 years of BTC/ETH data (2023-2026).
"""

# Trading pairs
SYMBOLS = ["BTC/USDT", "ETH/USDT"]

# Exchange
EXCHANGE_ID = "binanceus"

# Timeframes
TREND_TIMEFRAME = "1d"    # Higher timeframe for trend direction
SIGNAL_TIMEFRAME = "4h"   # Trade on 4H to reduce noise

# EMA periods
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

# RSI — widened thresholds to reduce false signals
RSI_PERIOD = 14
RSI_OVERBOUGHT = 80
RSI_OVERSOLD = 20

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2.0

# Volume
VOLUME_MA_PERIOD = 20

# ATR for trailing stop
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 2.85

# Confluence scoring — asymmetric: easy to enter, trailing stop handles exit
ENTRY_BULL_THRESHOLD = 3.15
EXIT_BEAR_THRESHOLD = 3.5
BUY_THRESHOLD = 3.15   # Alias for signal display
SELL_THRESHOLD = 3.5

# Trading mode
LONG_ONLY = True
TRADE_COOLDOWN = 4        # Minimum bars between trades (16h on 4H chart)
USE_TREND_FILTER = True
USE_TRAILING_STOP = True

# Risk management
RISK_REWARD_RATIO = 0     # 0 = no fixed TP, rely on trailing stop
MAX_RISK_PER_TRADE = 0.04 # Risk 4% of equity per trade (vol-adjusted sizing)

# Backtesting
BACKTEST_CASH = 1_000_000
BACKTEST_COMMISSION = 0.0015  # 0.1% fee + 0.05% slippage
BACKTEST_TRADE_SIZE = 0.95

# Data caching
CACHE_DIR = "cache"
