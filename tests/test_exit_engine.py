"""Tests for breakeven and trailing stop functions in exit_engine."""

import pytest

from models import SignalDirection
from strategy.exit_engine import compute_breakeven_sl, compute_trailing_sl


class TestBreakevenSL:
    """Tests for compute_breakeven_sl."""

    def test_long_triggered(self):
        """LONG: peak moved 1×ATR above entry → breakeven SL returned."""
        sl = compute_breakeven_sl(
            direction=SignalDirection.LONG,
            avg_entry=100.0,
            peak_favorable=102.0,  # moved 2.0 above entry
            entry_atr=1.5,         # threshold = 1.0 * 1.5 = 1.5
            be_atr_mult=1.0,
            be_buffer_pct=0.001,
        )
        assert sl is not None
        assert sl == pytest.approx(100.0 * 1.001)  # entry + 0.1% buffer

    def test_long_not_triggered(self):
        """LONG: peak hasn't moved enough → None."""
        sl = compute_breakeven_sl(
            direction=SignalDirection.LONG,
            avg_entry=100.0,
            peak_favorable=100.5,  # moved only 0.5
            entry_atr=1.5,         # threshold = 1.5
            be_atr_mult=1.0,
            be_buffer_pct=0.001,
        )
        assert sl is None

    def test_short_triggered(self):
        """SHORT: peak moved 1×ATR below entry → breakeven SL returned."""
        sl = compute_breakeven_sl(
            direction=SignalDirection.SHORT,
            avg_entry=100.0,
            peak_favorable=98.0,   # moved 2.0 below entry
            entry_atr=1.5,         # threshold = 1.5
            be_atr_mult=1.0,
            be_buffer_pct=0.001,
        )
        assert sl is not None
        assert sl == pytest.approx(100.0 * 0.999)  # entry - 0.1% buffer

    def test_short_not_triggered(self):
        """SHORT: peak hasn't moved enough → None."""
        sl = compute_breakeven_sl(
            direction=SignalDirection.SHORT,
            avg_entry=100.0,
            peak_favorable=99.5,   # moved only 0.5
            entry_atr=1.5,         # threshold = 1.5
            be_atr_mult=1.0,
            be_buffer_pct=0.001,
        )
        assert sl is None

    def test_disabled_when_mult_zero(self):
        """Feature disabled when be_atr_mult=0."""
        sl = compute_breakeven_sl(
            direction=SignalDirection.LONG,
            avg_entry=100.0,
            peak_favorable=200.0,  # huge move — would trigger if enabled
            entry_atr=1.0,
            be_atr_mult=0.0,
            be_buffer_pct=0.001,
        )
        assert sl is None

    def test_disabled_when_atr_zero(self):
        """Returns None when entry_atr is zero (no valid ATR)."""
        sl = compute_breakeven_sl(
            direction=SignalDirection.LONG,
            avg_entry=100.0,
            peak_favorable=200.0,
            entry_atr=0.0,
            be_atr_mult=1.0,
            be_buffer_pct=0.001,
        )
        assert sl is None

    def test_exact_threshold_triggers(self):
        """LONG: peak at exactly threshold should trigger."""
        sl = compute_breakeven_sl(
            direction=SignalDirection.LONG,
            avg_entry=100.0,
            peak_favorable=101.5,  # exactly 1.5 = 1.0 * 1.5
            entry_atr=1.5,
            be_atr_mult=1.0,
            be_buffer_pct=0.001,
        )
        assert sl is not None


class TestTrailingSL:
    """Tests for compute_trailing_sl."""

    def test_long_triggered(self):
        """LONG: peak moved 2×ATR → trailing SL = peak - 1.5×ATR."""
        sl = compute_trailing_sl(
            direction=SignalDirection.LONG,
            avg_entry=100.0,
            peak_favorable=104.0,  # moved 4.0 above entry
            entry_atr=1.5,         # activation = 2.0 * 1.5 = 3.0
            trail_activation_atr=2.0,
            trail_distance_atr=1.5,
        )
        assert sl is not None
        expected = 104.0 - 1.5 * 1.5  # peak - distance
        assert sl == pytest.approx(expected)

    def test_long_not_triggered(self):
        """LONG: peak hasn't moved enough → None."""
        sl = compute_trailing_sl(
            direction=SignalDirection.LONG,
            avg_entry=100.0,
            peak_favorable=102.0,  # moved 2.0, needs 3.0
            entry_atr=1.5,
            trail_activation_atr=2.0,
            trail_distance_atr=1.5,
        )
        assert sl is None

    def test_short_triggered(self):
        """SHORT: peak moved 2×ATR below → trailing SL = peak + 1.5×ATR."""
        sl = compute_trailing_sl(
            direction=SignalDirection.SHORT,
            avg_entry=100.0,
            peak_favorable=96.0,   # moved 4.0 below entry
            entry_atr=1.5,         # activation = 3.0
            trail_activation_atr=2.0,
            trail_distance_atr=1.5,
        )
        assert sl is not None
        expected = 96.0 + 1.5 * 1.5  # peak + distance
        assert sl == pytest.approx(expected)

    def test_short_not_triggered(self):
        """SHORT: peak hasn't moved enough → None."""
        sl = compute_trailing_sl(
            direction=SignalDirection.SHORT,
            avg_entry=100.0,
            peak_favorable=98.5,   # moved 1.5, needs 3.0
            entry_atr=1.5,
            trail_activation_atr=2.0,
            trail_distance_atr=1.5,
        )
        assert sl is None

    def test_disabled_when_activation_zero(self):
        """Feature disabled when trail_activation_atr=0."""
        sl = compute_trailing_sl(
            direction=SignalDirection.LONG,
            avg_entry=100.0,
            peak_favorable=200.0,
            entry_atr=1.0,
            trail_activation_atr=0.0,
            trail_distance_atr=1.5,
        )
        assert sl is None

    def test_disabled_when_atr_zero(self):
        """Returns None when entry_atr is zero."""
        sl = compute_trailing_sl(
            direction=SignalDirection.LONG,
            avg_entry=100.0,
            peak_favorable=200.0,
            entry_atr=0.0,
            trail_activation_atr=2.0,
            trail_distance_atr=1.5,
        )
        assert sl is None

    def test_trail_moves_up_with_peak(self):
        """LONG: higher peak → higher trailing SL."""
        sl1 = compute_trailing_sl(
            direction=SignalDirection.LONG,
            avg_entry=100.0,
            peak_favorable=104.0,
            entry_atr=1.5,
            trail_activation_atr=2.0,
            trail_distance_atr=1.5,
        )
        sl2 = compute_trailing_sl(
            direction=SignalDirection.LONG,
            avg_entry=100.0,
            peak_favorable=106.0,  # higher peak
            entry_atr=1.5,
            trail_activation_atr=2.0,
            trail_distance_atr=1.5,
        )
        assert sl1 is not None
        assert sl2 is not None
        assert sl2 > sl1  # trailing SL should be higher with higher peak
