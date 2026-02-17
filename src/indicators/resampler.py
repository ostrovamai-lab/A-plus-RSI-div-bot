"""Resample 1-minute OHLCV data to arbitrary intervals (e.g. 8 minutes).

Bybit doesn't natively support 8-minute candles, so we fetch 1-minute bars
and aggregate them here.
"""

from __future__ import annotations

import pandas as pd


def resample_ohlcv(df_1m: pd.DataFrame, interval_minutes: int = 8) -> pd.DataFrame:
    """Resample 1-minute OHLCV DataFrame to a larger interval.

    Args:
        df_1m: DataFrame with columns: timestamp, open, high, low, close, volume.
               timestamp should be in milliseconds (epoch).
        interval_minutes: Target candle size in minutes.

    Returns:
        Resampled DataFrame with the same columns, indexed by datetime.
        Only complete candles are returned (incomplete trailing candle is dropped).
    """
    df = df_1m.copy()

    # Convert millisecond timestamp to datetime index
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    elif df.index.dtype == "int64":
        df["datetime"] = pd.to_datetime(df.index, unit="ms", utc=True)
    else:
        df["datetime"] = df.index

    df = df.set_index("datetime")

    rule = f"{interval_minutes}min"

    resampled = df.resample(rule, origin="epoch").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])

    # Restore millisecond timestamp
    resampled["timestamp"] = (resampled.index.astype("int64") // 10**6).astype(int)

    # Drop last candle if it's incomplete (fewer than interval_minutes source bars)
    if len(df) > 0 and len(resampled) > 0:
        last_candle_start = resampled.index[-1]
        bars_in_last = df.loc[df.index >= last_candle_start].shape[0]
        if bars_in_last < interval_minutes:
            resampled = resampled.iloc[:-1]

    return resampled.reset_index(drop=True)


def klines_to_df(klines: list[dict]) -> pd.DataFrame:
    """Convert list-of-dicts klines to a DataFrame.

    Accepts dicts with keys: timestamp, open, high, low, close, volume.
    """
    if not klines:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(klines)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    return df
