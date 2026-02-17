"""Tests for TP RSI v2 indicator."""

import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest

from indicators.tp_rsi_v2 import (
    compute_tp_rsi,
    detect_divergences,
    build_divergence_lookup,
)
from models import DivergenceType


class TestComputeTPRsi:
    """Test RSI + BB + Ribbon computation."""

    def test_rsi_matches_pandas_ta(self, sample_ohlcv_8m):
        """RSI output should match pandas_ta.rsi()."""
        close = sample_ohlcv_8m["close"]
        high = sample_ohlcv_8m["high"]
        low = sample_ohlcv_8m["low"]

        result = compute_tp_rsi(close, high, low, rsi_period=50)
        ref_rsi = ta.rsi(close, length=50)

        valid = ~result["rsi"].isna() & ~ref_rsi.isna()
        assert valid.sum() > 50
        np.testing.assert_allclose(
            result["rsi"][valid].values, ref_rsi[valid].values, atol=1e-6,
        )

    def test_output_keys(self, sample_ohlcv_8m):
        """Verify all expected keys are in the output."""
        result = compute_tp_rsi(
            sample_ohlcv_8m["close"],
            sample_ohlcv_8m["high"],
            sample_ohlcv_8m["low"],
        )
        expected = {"rsi", "basis", "upper_bb", "lower_bb", "disp_up", "disp_down",
                    "bullish_cross", "bearish_cross", "ribbon_score"}
        assert expected.issubset(result.keys())

    def test_ribbon_score_range(self, sample_ohlcv_8m):
        """Ribbon score should be between 0 and 5."""
        result = compute_tp_rsi(
            sample_ohlcv_8m["close"],
            sample_ohlcv_8m["high"],
            sample_ohlcv_8m["low"],
        )
        ribbon = result["ribbon_score"]
        valid = ~ribbon.isna()
        assert ribbon[valid].min() >= 0
        assert ribbon[valid].max() <= 5

    def test_bb_relationship(self, sample_ohlcv_8m):
        """Upper BB should always be >= basis >= lower BB."""
        result = compute_tp_rsi(
            sample_ohlcv_8m["close"],
            sample_ohlcv_8m["high"],
            sample_ohlcv_8m["low"],
        )
        valid = (~result["upper_bb"].isna() & ~result["lower_bb"].isna()
                 & ~result["basis"].isna())
        assert (result["upper_bb"][valid] >= result["basis"][valid]).all()
        assert (result["basis"][valid] >= result["lower_bb"][valid]).all()


class TestDivergenceDetection:
    """Test RSI divergence detection."""

    def test_synthetic_regular_bullish(self):
        """Synthetic data: price LL + RSI HL → regular bullish divergence."""
        n = 100
        # Create price with lower lows
        close = pd.Series([50.0] * n)
        close[20] = 40.0   # First low
        close[60] = 35.0   # Lower low

        high = close + 1
        low = close - 1

        # RSI with higher lows (fake RSI)
        rsi = pd.Series([50.0] * n)
        rsi[20] = 25.0     # First RSI low
        rsi[60] = 30.0     # Higher RSI low

        divs = detect_divergences(close, high, low, rsi, pivot_lookback=5, max_range=60)

        bull_reg = [d for d in divs if d.div_type == DivergenceType.REGULAR_BULLISH]
        assert len(bull_reg) > 0

    def test_no_divergence_in_flat_data(self, sample_ohlcv_8m):
        """Random data may or may not have divergences, but shouldn't crash."""
        close = sample_ohlcv_8m["close"]
        high = sample_ohlcv_8m["high"]
        low = sample_ohlcv_8m["low"]
        rsi = ta.rsi(close, length=50)
        if rsi is None:
            rsi = pd.Series(50.0, index=close.index)

        divs = detect_divergences(close, high, low, rsi)
        assert isinstance(divs, list)

    def test_divergence_lookup_window(self):
        """Build lookup should make divergences accessible within window."""
        from models import Divergence
        divs = [
            Divergence(DivergenceType.REGULAR_BULLISH, bar_index=50,
                       rsi_current=30, rsi_previous=25,
                       price_current=40, price_previous=45),
        ]
        lookup = build_divergence_lookup(divs, window=5)
        assert 50 in lookup
        assert 55 in lookup
        assert 56 not in lookup
