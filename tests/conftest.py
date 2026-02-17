"""Pytest fixtures — sample OHLCV data for indicator tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_1m():
    """Generate 1000 bars of synthetic 1-minute OHLCV data.

    Creates a sine-wave price pattern with random noise,
    suitable for testing resampling and basic indicators.
    """
    np.random.seed(42)
    n = 1000
    base_price = 100.0
    # Align start to an 8-minute (480,000 ms) boundary from epoch
    # so resampler tests produce clean candle counts
    start_ms = (1_700_000_000_000 // 480_000 + 1) * 480_000
    timestamps = np.arange(n) * 60_000 + start_ms

    # Sine wave + noise
    t = np.arange(n, dtype=float)
    trend = base_price + 10 * np.sin(2 * np.pi * t / 200) + 0.01 * t
    noise = np.random.normal(0, 0.5, n)
    close = trend + noise

    # Build OHLCV
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(np.random.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(np.random.normal(0, 0.3, n))
    volume = np.random.uniform(100, 1000, n)

    return pd.DataFrame({
        "timestamp": timestamps.astype(int),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def sample_ohlcv_8m():
    """Generate 200 bars of synthetic 8-minute OHLCV data.

    Creates a pattern with clear swings for A+ signal testing.
    """
    np.random.seed(123)
    n = 200
    timestamps = np.arange(n) * 8 * 60_000 + 1_700_000_000_000

    t = np.arange(n, dtype=float)
    # Create clear swing pattern
    base = 100.0
    swing1 = 5.0 * np.sin(2 * np.pi * t / 40)   # ~40 bar period
    swing2 = 2.0 * np.sin(2 * np.pi * t / 15)   # faster oscillation
    noise = np.random.normal(0, 0.2, n)
    close = base + swing1 + swing2 + noise

    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(np.random.normal(0, 0.15, n))
    low = np.minimum(open_, close) - np.abs(np.random.normal(0, 0.15, n))
    volume = np.random.uniform(50, 500, n)

    return pd.DataFrame({
        "timestamp": timestamps.astype(int),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def sample_klines_1m(sample_ohlcv_1m):
    """Convert sample DataFrame to list-of-dicts format (like Bybit API returns)."""
    return sample_ohlcv_1m.to_dict("records")


@pytest.fixture
def trending_up_data():
    """Generate data with a clear uptrend followed by reversal.

    Good for testing divergence detection (price HH, RSI LH = bearish div).
    """
    np.random.seed(77)
    n = 300

    timestamps = np.arange(n) * 8 * 60_000 + 1_700_000_000_000
    t = np.arange(n, dtype=float)

    # Uptrend for first 200 bars, then reversal
    close = np.where(
        t < 200,
        100.0 + 0.1 * t + 2 * np.sin(2 * np.pi * t / 30),
        120.0 - 0.15 * (t - 200) + 2 * np.sin(2 * np.pi * t / 30),
    )
    noise = np.random.normal(0, 0.2, n)
    close = close + noise

    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(np.random.normal(0, 0.15, n))
    low = np.minimum(open_, close) - np.abs(np.random.normal(0, 0.15, n))
    volume = np.random.uniform(100, 800, n)

    return pd.DataFrame({
        "timestamp": timestamps.astype(int),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
