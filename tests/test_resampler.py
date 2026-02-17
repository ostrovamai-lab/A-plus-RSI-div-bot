"""Tests for the 1min → 8min resampler."""

import numpy as np
import pandas as pd
import pytest

from indicators.resampler import klines_to_df, resample_ohlcv


class TestResampleOhlcv:
    """Test OHLCV resampling from 1-minute to 8-minute bars."""

    def test_basic_resample_bar_count(self, sample_ohlcv_1m):
        """16 one-minute bars → exactly 2 complete 8-min candles."""
        df_16 = sample_ohlcv_1m.head(16).copy()
        result = resample_ohlcv(df_16, interval_minutes=8)
        assert len(result) == 2

    def test_incomplete_candle_dropped(self, sample_ohlcv_1m):
        """13 one-minute bars → 1 complete 8-min candle (5 leftover dropped)."""
        df_13 = sample_ohlcv_1m.head(13).copy()
        result = resample_ohlcv(df_13, interval_minutes=8)
        assert len(result) == 1

    def test_ohlcv_aggregation(self, sample_ohlcv_1m):
        """Verify open=first, high=max, low=min, close=last, volume=sum."""
        df_8 = sample_ohlcv_1m.head(8).copy()
        result = resample_ohlcv(df_8, interval_minutes=8)
        assert len(result) == 1
        row = result.iloc[0]

        assert row["open"] == df_8["open"].iloc[0]
        assert row["high"] == df_8["high"].max()
        assert row["low"] == df_8["low"].min()
        assert row["close"] == df_8["close"].iloc[7]
        assert abs(row["volume"] - df_8["volume"].sum()) < 0.01

    def test_timestamp_preserved(self, sample_ohlcv_1m):
        """Resampled bars have millisecond timestamps."""
        result = resample_ohlcv(sample_ohlcv_1m.head(80), interval_minutes=8)
        assert "timestamp" in result.columns
        assert result["timestamp"].dtype in (np.int64, int)

    def test_1000_bars(self, sample_ohlcv_1m):
        """1000 one-minute bars → ~125 complete 8-min candles."""
        result = resample_ohlcv(sample_ohlcv_1m, interval_minutes=8)
        # 1000 / 8 = 125, but last candle may be incomplete
        assert 120 <= len(result) <= 125

    def test_output_columns(self, sample_ohlcv_1m):
        """Output has all required OHLCV columns."""
        result = resample_ohlcv(sample_ohlcv_1m.head(80), interval_minutes=8)
        for col in ("open", "high", "low", "close", "volume", "timestamp"):
            assert col in result.columns


class TestKlinesToDf:
    """Test klines list-of-dicts to DataFrame conversion."""

    def test_basic_conversion(self):
        klines = [
            {"timestamp": 1000, "open": 100, "high": 105, "low": 95, "close": 102, "volume": 50},
            {"timestamp": 2000, "open": 102, "high": 108, "low": 100, "close": 107, "volume": 60},
        ]
        df = klines_to_df(klines)
        assert len(df) == 2
        assert df["close"].dtype == float

    def test_empty_input(self):
        df = klines_to_df([])
        assert df.empty
