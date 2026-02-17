"""Tests for Fibonacci DCA grid entry engine."""

from decimal import Decimal

import pytest

from models import SignalDirection, SignalScore
from strategy.entry_engine import compute_entry_grid, compute_stop_loss


class TestComputeEntryGrid:
    """Test Fibonacci grid level computation."""

    def test_long_grid_basic(self):
        """Long grid: 3 levels below signal price toward fractal."""
        score = SignalScore(total=80, tier="A+")
        levels = compute_entry_grid(
            SignalDirection.LONG,
            signal_price=100.0,
            fractal_price=95.0,
            atr_value=3.0,
            score=score,
            capital_usd=1000.0,
            leverage=10,
        )
        assert len(levels) == 3

        # All should be Buy orders
        for lv in levels:
            assert lv.side == "Buy"
            assert lv.is_entry

        # Prices should be below signal price and above fractal
        for lv in levels:
            assert lv.price < 100.0
            assert lv.price >= 95.0

        # Fib ordering: prices descend (further from entry = better price)
        assert levels[0].price > levels[1].price > levels[2].price

    def test_short_grid_basic(self):
        """Short grid: 3 levels above signal price toward fractal."""
        score = SignalScore(total=80, tier="A+")
        levels = compute_entry_grid(
            SignalDirection.SHORT,
            signal_price=100.0,
            fractal_price=105.0,
            atr_value=3.0,
            score=score,
            capital_usd=1000.0,
            leverage=10,
        )
        assert len(levels) == 3

        for lv in levels:
            assert lv.side == "Sell"
            assert lv.is_entry
            assert lv.price > 100.0
            assert lv.price <= 105.0

    def test_dca_volume_scaling(self):
        """Biggest order should be at best price (furthest from entry)."""
        score = SignalScore(total=80, tier="A+")
        levels = compute_entry_grid(
            SignalDirection.LONG,
            signal_price=100.0,
            fractal_price=90.0,
            atr_value=5.0,
            score=score,
            capital_usd=1000.0,
            leverage=10,
        )
        # Last level (furthest from entry) should have largest qty
        assert levels[2].qty > levels[0].qty

    def test_min_range_enforcement(self):
        """Grid range should not be smaller than min_range_atr_mult * ATR."""
        score = SignalScore(total=80, tier="A+")
        levels = compute_entry_grid(
            SignalDirection.LONG,
            signal_price=100.0,
            fractal_price=99.9,       # Very close fractal
            atr_value=5.0,            # But ATR is 5.0
            score=score,
            capital_usd=1000.0,
            leverage=10,
        )
        # Range should be at least 0.5 * 5.0 = 2.5
        assert len(levels) == 3
        lowest_price = min(lv.price for lv in levels)
        assert 100.0 - lowest_price >= 0.5 * 5.0 * 0.236  # At minimum

    def test_score_scaling(self):
        """Lower score tier should produce smaller positions."""
        score_high = SignalScore(total=85, tier="A+")
        score_low = SignalScore(total=55, tier="B")

        levels_high = compute_entry_grid(
            SignalDirection.LONG, 100.0, 95.0, 3.0,
            score_high, 1000.0, 10,
        )
        levels_low = compute_entry_grid(
            SignalDirection.LONG, 100.0, 95.0, 3.0,
            score_low, 1000.0, 10,
        )

        total_qty_high = sum(float(lv.qty) for lv in levels_high)
        total_qty_low = sum(float(lv.qty) for lv in levels_low)
        assert total_qty_high > total_qty_low

    def test_reject_score_empty_grid(self):
        """Score below 50 (REJECT) → no grid levels."""
        score = SignalScore(total=30, tier="REJECT")
        levels = compute_entry_grid(
            SignalDirection.LONG, 100.0, 95.0, 3.0,
            score, 1000.0, 10,
        )
        assert len(levels) == 0


class TestComputeStopLoss:
    """Test stop loss calculation."""

    def test_long_sl(self):
        """Long SL should be below fractal with buffer."""
        sl = compute_stop_loss(SignalDirection.LONG, 95.0, 0.002)
        assert sl < 95.0
        assert sl == pytest.approx(95.0 - 95.0 * 0.002)

    def test_short_sl(self):
        """Short SL should be above fractal with buffer."""
        sl = compute_stop_loss(SignalDirection.SHORT, 105.0, 0.002)
        assert sl > 105.0
        assert sl == pytest.approx(105.0 + 105.0 * 0.002)
