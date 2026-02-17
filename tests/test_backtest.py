"""Tests for backtest engine."""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import run_backtest, _compute_stats
from config import load_config
from indicators.resampler import klines_to_df
from models import BacktestResult, Trade


class TestBacktestEngine:
    """Test the backtest engine on synthetic data."""

    def _make_klines_1m(self, n=2000):
        """Generate synthetic 1-minute klines."""
        np.random.seed(42)
        timestamps = np.arange(n) * 60_000 + 1_700_000_000_000
        t = np.arange(n, dtype=float)
        close = 100.0 + 10 * np.sin(2 * np.pi * t / 400) + 0.005 * t
        noise = np.random.normal(0, 0.3, n)
        close = close + noise
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        high = np.maximum(open_, close) + np.abs(np.random.normal(0, 0.2, n))
        low = np.minimum(open_, close) - np.abs(np.random.normal(0, 0.2, n))
        volume = np.random.uniform(100, 500, n)

        return [
            {
                "timestamp": int(timestamps[i]),
                "open": float(open_[i]),
                "high": float(high[i]),
                "low": float(low[i]),
                "close": float(close[i]),
                "volume": float(volume[i]),
            }
            for i in range(n)
        ]

    def test_backtest_runs_without_error(self):
        """Backtest should complete without crashing on synthetic data."""
        klines = self._make_klines_1m(2000)
        result = run_backtest(klines)
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_backtest_too_few_bars(self):
        """Backtest with too few bars should return empty result."""
        klines = self._make_klines_1m(50)
        result = run_backtest(klines)
        assert result.total_trades == 0

    def test_equity_curve_length(self):
        """Equity curve length should roughly match number of 8min bars."""
        klines = self._make_klines_1m(2000)
        result = run_backtest(klines)
        # 2000 1m bars → ~250 8m bars, minus warmup
        assert 100 < len(result.equity_curve) < 300


class TestComputeStats:
    """Test statistics computation on trade lists."""

    def test_basic_stats(self):
        result = BacktestResult()
        result.trades = [
            Trade(pnl=10, fee=1, reason="grid_fill"),
            Trade(pnl=20, fee=1, reason="grid_fill"),
            Trade(pnl=-5, fee=1, reason="sl"),
        ]
        _compute_stats(result)

        assert result.total_trades == 3
        assert result.winning_trades == 2
        assert result.losing_trades == 1
        assert result.win_rate == pytest.approx(66.67, rel=0.01)
        assert result.net_pnl == pytest.approx(25 - 3)

    def test_empty_trades(self):
        result = BacktestResult()
        _compute_stats(result)
        assert result.total_trades == 0
        assert result.win_rate == 0

    def test_profit_factor(self):
        result = BacktestResult()
        result.trades = [
            Trade(pnl=100, fee=0),
            Trade(pnl=-50, fee=0),
        ]
        _compute_stats(result)
        assert result.profit_factor == pytest.approx(2.0)
