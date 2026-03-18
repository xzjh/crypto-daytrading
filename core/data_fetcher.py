"""Fetch OHLCV data via ccxt with pagination and CSV caching."""

import os
import time
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd

from core import config


def _get_exchange():
    exchange_class = getattr(ccxt, config.EXCHANGE_ID)
    return exchange_class({"enableRateLimit": True})


def _cache_path(symbol: str, timeframe: str) -> str:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    safe_symbol = symbol.replace("/", "_")
    return os.path.join(config.CACHE_DIR, f"{safe_symbol}_{timeframe}.csv")


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    days: int = 200,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data, using CSV cache when available and fresh."""
    cache_file = _cache_path(symbol, timeframe)

    # Use cache if it exists and is less than 1 hour old
    if use_cache and os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < 3600:
            df = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
            if len(df) > 0:
                return df

    exchange = _get_exchange()
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_ohlcv = []

    while True:
        for attempt in range(3):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  Retry {attempt + 1}/3: {e}")
                time.sleep(2 ** (attempt + 1))
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # Next ms after last candle
        if len(ohlcv) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_ohlcv, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], unit="ms", utc=True)
    df.set_index("Date", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    # Cache to CSV
    df.to_csv(cache_file)
    return df
