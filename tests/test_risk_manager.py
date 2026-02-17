"""Tests for risk manager."""

import math

import pytest

from strategy.risk_manager import RiskManager


class TestRiskManager:
    """Test position sizing, drawdown tracking, reinvestment."""

    def test_initial_state(self):
        rm = RiskManager(initial_capital=1000, leverage=10)
        assert rm.current_equity == 1000
        assert rm.peak_equity == 1000
        assert rm.is_halted == False

    def test_equity_update(self):
        rm = RiskManager(initial_capital=1000)
        rm.update_equity(50)   # Won $50
        assert rm.current_equity == 1050
        assert rm.peak_equity == 1050

        rm.update_equity(-30)  # Lost $30
        assert rm.current_equity == 1020
        assert rm.peak_equity == 1050  # Peak unchanged

    def test_drawdown_check(self):
        rm = RiskManager(initial_capital=1000, max_drawdown_pct=10)
        rm.update_equity(-100)  # Lost $100 → 10% drawdown
        assert rm.check_drawdown() == True
        assert rm.is_halted == True

    def test_no_drawdown_halt_within_limit(self):
        rm = RiskManager(initial_capital=1000, max_drawdown_pct=10)
        rm.update_equity(-50)
        assert rm.check_drawdown() == False
        assert rm.is_halted == False

    def test_reinvestment_scaling(self):
        """With reinvestment, position size scales with sqrt of equity ratio."""
        rm = RiskManager(initial_capital=1000, reinvest=True)

        # At initial equity → scale = 1.0
        base = rm.compute_base_size_usd(1.0)
        assert base == pytest.approx(1000.0)

        # After 50% profit → scale = sqrt(1.5) ≈ 1.22
        rm.update_equity(500)
        base_after = rm.compute_base_size_usd(1.0)
        expected = 1000 * math.sqrt(1.5)
        assert base_after == pytest.approx(expected, rel=0.01)

    def test_reinvestment_clamping(self):
        """Reinvestment scale should be clamped to [min, max]."""
        rm = RiskManager(initial_capital=1000, reinvest=True, min_scale=0.5, max_scale=2.0)

        # After big profit: equity 5x → sqrt(5)=2.24 → clamped to 2.0
        rm.update_equity(4000)
        base = rm.compute_base_size_usd(1.0)
        assert base == pytest.approx(1000 * 2.0)

        # After big loss: equity 0.1x → sqrt(0.1)=0.32 → clamped to 0.5
        # But equity clamp limits to current_equity (100), not 1000 * 0.5 = 500
        rm2 = RiskManager(initial_capital=1000, reinvest=True, min_scale=0.5, max_scale=2.0)
        rm2.update_equity(-900)
        base2 = rm2.compute_base_size_usd(1.0)
        assert base2 == pytest.approx(100.0)  # clamped to available equity

    def test_halted_returns_zero(self):
        """When halted, position size should be zero."""
        rm = RiskManager(initial_capital=1000, max_drawdown_pct=10)
        rm.update_equity(-100)
        rm.check_drawdown()
        assert rm.compute_base_size_usd(1.0) == 0.0

    def test_score_scaling(self):
        """Position scale from score tier should multiply base size."""
        rm = RiskManager(initial_capital=1000, reinvest=False)
        full = rm.compute_base_size_usd(1.0)
        half = rm.compute_base_size_usd(0.5)
        assert half == pytest.approx(full * 0.5)

    def test_drawdown_pct(self):
        """Drawdown percentage calculation."""
        rm = RiskManager(initial_capital=1000)
        rm.update_equity(200)  # Peak = 1200
        rm.update_equity(-100)  # Now = 1100
        dd = rm.drawdown_pct()
        assert dd == pytest.approx(100 / 1200 * 100, rel=0.01)
