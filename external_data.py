"""Fetch external data sources: Fear & Greed Index, Funding Rates."""

import os
import json
import time
import urllib.request
from datetime import datetime, timezone

import ccxt
import pandas as pd

import config

CACHE_DIR = config.CACHE_DIR


def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.csv")


def fetch_fear_greed(use_cache=True) -> pd.DataFrame:
    """Fetch Crypto Fear & Greed Index (daily, from alternative.me)."""
    cache = _cache_path("fear_greed")
    if use_cache and os.path.exists(cache):
        age = time.time() - os.path.getmtime(cache)
        if age < 3600 * 12:
            df = pd.read_csv(cache, index_col="Date", parse_dates=True)
            if len(df) > 0:
                return df

    url = "https://api.alternative.me/fng/?limit=0&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=30).read())

    rows = []
    for d in data["data"]:
        dt = datetime.fromtimestamp(int(d["timestamp"]), tz=timezone.utc)
        rows.append({"Date": dt, "FNG": int(d["value"])})

    df = pd.DataFrame(rows)
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(cache)
    return df


def fetch_funding_rates(symbol="BTC/USDT", days=1095, use_cache=True) -> pd.DataFrame:
    """Fetch historical funding rates from Binance USDM futures."""
    safe = symbol.replace("/", "_").replace(":", "_")
    cache = _cache_path(f"funding_{safe}")
    if use_cache and os.path.exists(cache):
        age = time.time() - os.path.getmtime(cache)
        if age < 3600 * 6:
            df = pd.read_csv(cache, index_col="Date", parse_dates=True)
            if len(df) > 0:
                return df

    perp_symbol = symbol if ":" in symbol else f"{symbol}:USDT"

    # Try binanceusdm first, fall back to bybit
    for exchange_id in ["binanceusdm", "bybit"]:
        try:
            ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
            since = int((datetime.now(timezone.utc).timestamp() - days * 86400) * 1000)
            all_rates = []

            while True:
                for attempt in range(3):
                    try:
                        rates = ex.fetch_funding_rate_history(perp_symbol, since=since, limit=1000)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(2 ** (attempt + 1))
                if not rates:
                    break
                all_rates.extend(rates)
                since = rates[-1]["timestamp"] + 1
                if len(rates) < 1000:
                    break
                time.sleep(ex.rateLimit / 1000)

            if all_rates:
                df = pd.DataFrame([{
                    "Date": pd.Timestamp(r["datetime"]),
                    "FundingRate": r["fundingRate"],
                } for r in all_rates])
                df.set_index("Date", inplace=True)
                df = df[~df.index.duplicated(keep="last")]
                df.sort_index(inplace=True)
                df.to_csv(cache)
                print(f"  Funding rates: {len(df)} entries from {exchange_id}")
                return df
        except Exception as e:
            print(f"  {exchange_id} funding failed: {e}")
            continue

    return pd.DataFrame(columns=["FundingRate"])


def merge_external_data(df_price: pd.DataFrame, symbol: str = "BTC/USDT") -> pd.DataFrame:
    """Merge external data into price DataFrame.

    Adds columns:
    - FNG: Fear & Greed Index (0-100, forward-filled to 4H bars)
    - FNG_MA: 7-day moving average of FNG
    - FundingRate: latest funding rate (forward-filled)
    - FR_MA: 7-day MA of cumulative funding rate
    """
    df = df_price.copy()

    # Fear & Greed
    try:
        fng = fetch_fear_greed()
        fng.index = fng.index.tz_localize("UTC") if fng.index.tz is None else fng.index
        df = df.merge(fng, left_index=True, right_index=True, how="left")
        df["FNG"] = df["FNG"].ffill()
        df["FNG_MA"] = df["FNG"].rolling(7 * 6, min_periods=1).mean()  # 7 days of 4H bars
    except Exception as e:
        print(f"  Warning: F&G fetch failed: {e}")
        df["FNG"] = 50
        df["FNG_MA"] = 50

    # Funding Rates
    try:
        fr = fetch_funding_rates(symbol)
        if len(fr) > 0:
            fr.index = fr.index.tz_localize("UTC") if fr.index.tz is None else fr.index
            df = df.merge(fr, left_index=True, right_index=True, how="left")
            df["FundingRate"] = df["FundingRate"].ffill().fillna(0)
            # Cumulative funding over rolling 7 days
            df["FR_MA"] = df["FundingRate"].rolling(7 * 3, min_periods=1).mean()  # 3 per day
        else:
            df["FundingRate"] = 0
            df["FR_MA"] = 0
    except Exception as e:
        print(f"  Warning: Funding rate fetch failed: {e}")
        df["FundingRate"] = 0
        df["FR_MA"] = 0

    df.dropna(subset=["Close"], inplace=True)
    return df
