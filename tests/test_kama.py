"""Tests for KAMA indicator."""

import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest

from indicators.kama import compute_kama, compute_kama_full, kama_slope


class TestComputeKama:
    """Test KAMA computation against pandas_ta reference."""

    def test_matches_pandas_ta(self, sample_ohlcv_8m):
        """Our KAMA output should match pandas_ta.kama() directly."""
        close = sample_ohlcv_8m["close"]
        our_kama = compute_kama(close, er_length=10, fast_length=2, slow_length=30)
        ref_kama = ta.kama(close, length=10, fast=2, slow=30)

        # Skip NaN warmup bars
        valid = ~our_kama.isna() & ~ref_kama.isna()
        assert valid.sum() > 100  # enough valid bars
        np.testing.assert_allclose(
            our_kama[valid].values, ref_kama[valid].values, atol=1e-6,
        )

    def test_output_length(self, sample_ohlcv_8m):
        """Output length matches input."""
        close = sample_ohlcv_8m["close"]
        result = compute_kama(close)
        assert len(result) == len(close)

    def test_nan_at_start(self):
        """First few values should be NaN (warmup period)."""
        close = pd.Series([100.0] * 5, dtype=float)
        result = compute_kama(close, er_length=10)
        assert result.isna().all()  # Not enough data


class TestKamaSlope:
    """Test KAMA slope (rising/falling) detection."""

    def test_rising_kama(self):
        """KAMA that increases should be bullish."""
        kama_vals = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        slope = kama_slope(kama_vals, bars=3)
        # At index 3: 13 > 10 = True
        assert slope.iloc[3] == True
        assert slope.iloc[4] == True

    def test_falling_kama(self):
        """KAMA that decreases should be bearish."""
        kama_vals = pd.Series([15.0, 14.0, 13.0, 12.0, 11.0, 10.0])
        slope = kama_slope(kama_vals, bars=3)
        assert slope.iloc[3] == False
        assert slope.iloc[4] == False


class TestComputeKamaFull:
    """Test the combined KAMA + slope computation."""

    def test_returns_both(self, sample_ohlcv_8m):
        close = sample_ohlcv_8m["close"]
        result = compute_kama_full(close)
        assert "kama" in result
        assert "bullish" in result
        assert len(result["kama"]) == len(close)
        assert len(result["bullish"]) == len(close)
