"""Tests for A+ Signal state machine."""

import numpy as np
import pandas as pd
import pytest

from indicators.aplus_signal import (
    StreamingAPlusSignal,
    _compute_pivots_high,
    _compute_pivots_low,
    compute_aplus_signals,
)
from models import SignalDirection


class TestPivotComputation:
    """Test pivot high/low detection."""

    def test_simple_pivot_high(self):
        """Clear peak at index 5 should be detected."""
        arr = np.array([10, 11, 12, 13, 14, 20, 14, 13, 12, 11, 10], dtype=float)
        result = _compute_pivots_high(arr, lookback=3)
        # Pivot at index 5, confirmed at index 5+3=8
        assert result[8] == True

    def test_simple_pivot_low(self):
        """Clear trough at index 5 should be detected."""
        arr = np.array([20, 19, 18, 17, 16, 10, 16, 17, 18, 19, 20], dtype=float)
        result = _compute_pivots_low(arr, lookback=3)
        assert result[8] == True

    def test_no_pivot_in_flat(self):
        """Flat data should have no pivots."""
        arr = np.array([10.0] * 20)
        result_h = _compute_pivots_high(arr, lookback=3)
        result_l = _compute_pivots_low(arr, lookback=3)
        assert not result_h.any()
        assert not result_l.any()


class TestComputeAPlusSignals:
    """Test vectorized A+ signal computation."""

    def test_output_shape(self, sample_ohlcv_8m):
        """Output arrays should match input length."""
        result = compute_aplus_signals(sample_ohlcv_8m)
        n = len(sample_ohlcv_8m)
        assert len(result["aplus_long"]) == n
        assert len(result["aplus_short"]) == n
        assert len(result["fractal_price"]) == n
        assert len(result["ema_fast"]) == n
        assert len(result["ema_slow"]) == n

    def test_output_dtypes(self, sample_ohlcv_8m):
        """Bool arrays are bool, float arrays are float."""
        result = compute_aplus_signals(sample_ohlcv_8m)
        assert result["aplus_long"].dtype == bool
        assert result["aplus_short"].dtype == bool
        assert result["fractal_price"].dtype == float

    def test_triangle_detection(self, sample_ohlcv_8m):
        """Should detect at least some triangles in swing data."""
        result = compute_aplus_signals(sample_ohlcv_8m)
        # Triangles are intermediate steps; at least some should fire
        total_triangles = result["triangles_long"].sum() + result["triangles_short"].sum()
        assert total_triangles > 0, "No triangles detected in swing data"

    def test_fractal_price_on_signal(self, sample_ohlcv_8m):
        """Fractal price should be set on signal bars and NaN elsewhere."""
        result = compute_aplus_signals(sample_ohlcv_8m)
        signal_bars = result["aplus_long"] | result["aplus_short"]
        non_signal_bars = ~signal_bars

        # On non-signal bars, fractal_price should be NaN
        assert np.isnan(result["fractal_price"][non_signal_bars]).all()

        # If any signal fired, fractal_price should be a valid number
        if signal_bars.any():
            assert not np.isnan(result["fractal_price"][signal_bars]).any()

    def test_no_simultaneous_long_short(self, sample_ohlcv_8m):
        """A+ long and short should never fire on the same bar."""
        result = compute_aplus_signals(sample_ohlcv_8m)
        both = result["aplus_long"] & result["aplus_short"]
        assert not both.any()


class TestStreamingAPlusSignal:
    """Test bar-by-bar streaming A+ signal detector."""

    def test_basic_operation(self, sample_ohlcv_8m):
        """Streaming detector should not crash and produce valid output."""
        detector = StreamingAPlusSignal()
        signals = []
        for _, row in sample_ohlcv_8m.iterrows():
            sig = detector.update(
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
            )
            if sig is not None:
                signals.append(sig)

        # May or may not detect signals in synthetic data
        for sig in signals:
            assert sig.direction in (SignalDirection.LONG, SignalDirection.SHORT)
            assert sig.price > 0
            assert sig.fractal_price > 0

    def test_insufficient_data_returns_none(self):
        """Should return None until enough data accumulated."""
        detector = StreamingAPlusSignal()
        for i in range(10):
            result = detector.update(100.0, 101.0, 99.0, 100.5)
            assert result is None
