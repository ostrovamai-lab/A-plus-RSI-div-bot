"""Tests for V2 trend-following backtest engine."""

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is on path (conftest.py sibling doesn't add it by default)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtest.engine_v2 import (
    PyramidLevel,
    TrendSession,
    _choch_is_counter,
    _compute_base_qty,
    _regime_allows,
    _regime_flipped,
    _tighten_sl_to_entry,
    run_backtest_v2,
)
from models import SignalDirection
from strategy.regime import Bias
from strategy.risk_manager import RiskManager
from strategy.sunday_open import compute_sunday_open, sunday_open_confirms


# ======================================================================
# TrendSession dataclass tests
# ======================================================================


class TestTrendSession:
    """Verify avg_entry, total_qty, unrealized_pnl."""

    def _make_session(self, levels: list[tuple[float, float]]) -> TrendSession:
        """Helper: build session from (price, qty) pairs."""
        return TrendSession(
            session_id=1,
            direction=SignalDirection.LONG,
            symbol="TEST",
            regime_bias=int(Bias.BULLISH),
            open_bar=0,
            levels=[
                PyramidLevel(bar_index=i, entry_price=p, qty=q, score=70, fractal_price=p - 1)
                for i, (p, q) in enumerate(levels)
            ],
            sl_price=90.0,
            initial_sl=90.0,
            peak_price=100.0,
        )

    def test_single_level(self):
        sess = self._make_session([(100.0, 0.5)])
        assert sess.avg_entry == pytest.approx(100.0)
        assert sess.total_qty == pytest.approx(0.5)
        assert sess.num_levels == 1

    def test_multi_level_avg_entry(self):
        sess = self._make_session([(100.0, 1.0), (110.0, 1.0)])
        assert sess.avg_entry == pytest.approx(105.0)
        assert sess.total_qty == pytest.approx(2.0)

    def test_weighted_avg_entry(self):
        sess = self._make_session([(100.0, 3.0), (120.0, 1.0)])
        expected = (100.0 * 3.0 + 120.0 * 1.0) / 4.0
        assert sess.avg_entry == pytest.approx(expected)

    def test_unrealized_pnl_long(self):
        sess = self._make_session([(100.0, 1.0)])
        # price 110 → +10
        assert sess.unrealized_pnl(110.0) == pytest.approx(10.0)
        # price 90 → -10
        assert sess.unrealized_pnl(90.0) == pytest.approx(-10.0)

    def test_unrealized_pnl_short(self):
        sess = self._make_session([(100.0, 1.0)])
        sess.direction = SignalDirection.SHORT
        # price 90 → +10 (short profit)
        assert sess.unrealized_pnl(90.0) == pytest.approx(10.0)

    def test_empty_levels(self):
        sess = self._make_session([])
        assert sess.avg_entry == 0.0
        assert sess.total_qty == 0.0
        assert sess.num_levels == 0


# ======================================================================
# Regime logic tests
# ======================================================================


class TestRegimeLogic:
    """_regime_allows, _regime_flipped, _choch_is_counter."""

    def test_regime_allows_bullish_long(self):
        assert _regime_allows(Bias.BULLISH, SignalDirection.LONG) is True

    def test_regime_allows_bullish_short(self):
        assert _regime_allows(Bias.BULLISH, SignalDirection.SHORT) is False

    def test_regime_allows_bearish_short(self):
        assert _regime_allows(Bias.BEARISH, SignalDirection.SHORT) is True

    def test_regime_allows_bearish_long(self):
        assert _regime_allows(Bias.BEARISH, SignalDirection.LONG) is False

    def test_regime_allows_neutral(self):
        assert _regime_allows(Bias.NEUTRAL, SignalDirection.LONG) is False
        assert _regime_allows(Bias.NEUTRAL, SignalDirection.SHORT) is False

    def test_regime_flipped_long_to_bearish(self):
        sess = TrendSession(
            session_id=1, direction=SignalDirection.LONG, symbol="TEST",
            regime_bias=int(Bias.BULLISH), open_bar=0,
        )
        assert _regime_flipped(sess, Bias.BEARISH) is True
        assert _regime_flipped(sess, Bias.BULLISH) is False
        assert _regime_flipped(sess, Bias.NEUTRAL) is False

    def test_regime_flipped_short_to_bullish(self):
        sess = TrendSession(
            session_id=1, direction=SignalDirection.SHORT, symbol="TEST",
            regime_bias=int(Bias.BEARISH), open_bar=0,
        )
        assert _regime_flipped(sess, Bias.BULLISH) is True
        assert _regime_flipped(sess, Bias.BEARISH) is False

    def test_choch_counter_long(self):
        sess = TrendSession(
            session_id=1, direction=SignalDirection.LONG, symbol="TEST",
            regime_bias=int(Bias.BULLISH), open_bar=0,
        )
        # Bear CHoCH is counter to LONG
        assert _choch_is_counter(sess, bull_choch=False, bear_choch=True) is True
        assert _choch_is_counter(sess, bull_choch=True, bear_choch=False) is False
        assert _choch_is_counter(sess, bull_choch=False, bear_choch=False) is False

    def test_choch_counter_short(self):
        sess = TrendSession(
            session_id=1, direction=SignalDirection.SHORT, symbol="TEST",
            regime_bias=int(Bias.BEARISH), open_bar=0,
        )
        # Bull CHoCH is counter to SHORT
        assert _choch_is_counter(sess, bull_choch=True, bear_choch=False) is True
        assert _choch_is_counter(sess, bull_choch=False, bear_choch=True) is False


# ======================================================================
# SL helpers
# ======================================================================


class TestTightenSL:
    """_tighten_sl_to_entry only tightens."""

    def test_long_tightens_up(self):
        sess = TrendSession(
            session_id=1, direction=SignalDirection.LONG, symbol="TEST",
            regime_bias=int(Bias.BULLISH), open_bar=0,
            levels=[PyramidLevel(0, 100.0, 1.0, 70, 99.0)],
            sl_price=95.0,
        )
        _tighten_sl_to_entry(sess)
        assert sess.sl_price == pytest.approx(100.0)

    def test_long_does_not_loosen(self):
        sess = TrendSession(
            session_id=1, direction=SignalDirection.LONG, symbol="TEST",
            regime_bias=int(Bias.BULLISH), open_bar=0,
            levels=[PyramidLevel(0, 100.0, 1.0, 70, 99.0)],
            sl_price=105.0,  # already above entry
        )
        _tighten_sl_to_entry(sess)
        assert sess.sl_price == pytest.approx(105.0)

    def test_short_tightens_down(self):
        sess = TrendSession(
            session_id=1, direction=SignalDirection.SHORT, symbol="TEST",
            regime_bias=int(Bias.BEARISH), open_bar=0,
            levels=[PyramidLevel(0, 100.0, 1.0, 70, 101.0)],
            sl_price=105.0,
        )
        _tighten_sl_to_entry(sess)
        assert sess.sl_price == pytest.approx(100.0)


# ======================================================================
# Sunday Open tests
# ======================================================================


class TestSundayOpen:
    """compute_sunday_open and sunday_open_confirms."""

    def _make_df_with_dates(self, start: str, n: int) -> pd.DataFrame:
        """Create an 8m DataFrame starting at *start* UTC."""
        ts = pd.date_range(start, periods=n, freq="8min", tz="UTC")
        ms = (ts.astype("int64") // 10**6).astype(int)
        np.random.seed(0)
        return pd.DataFrame({
            "timestamp": ms.values,
            "open": np.random.uniform(99, 101, n),
            "high": np.random.uniform(101, 103, n),
            "low": np.random.uniform(97, 99, n),
            "close": np.random.uniform(99, 101, n),
            "volume": np.random.uniform(100, 200, n),
        })

    def test_nan_before_first_sunday(self):
        # Start on a Monday — no Sunday in first bars
        df = self._make_df_with_dates("2025-01-06 00:00", n=100)  # Monday
        so = compute_sunday_open(df)
        # First bar should be NaN (no Sunday yet)
        assert np.isnan(so[0])

    def test_fills_after_sunday(self):
        # Start on a Saturday, so Sunday is next day
        df = self._make_df_with_dates("2025-01-04 20:00", n=200)  # Saturday evening
        so = compute_sunday_open(df)
        # Should eventually have non-NaN values
        non_nan = so[~np.isnan(so)]
        assert len(non_nan) > 0

    def test_confirms_long_above_so(self):
        assert sunday_open_confirms(True, 105.0, 100.0) is True
        assert sunday_open_confirms(True, 95.0, 100.0) is False

    def test_confirms_short_below_so(self):
        assert sunday_open_confirms(False, 95.0, 100.0) is True
        assert sunday_open_confirms(False, 105.0, 100.0) is False

    def test_nan_so_always_passes(self):
        assert sunday_open_confirms(True, 50.0, float("nan")) is True
        assert sunday_open_confirms(False, 50.0, float("nan")) is True


# ======================================================================
# Integration test
# ======================================================================


class TestV2Integration:
    """End-to-end run_backtest_v2 on synthetic data."""

    def test_completes_on_synthetic_data(self, sample_klines_1m):
        """V2 engine runs without error on synthetic 1m data (no 1h data)."""
        result = run_backtest_v2(sample_klines_1m, klines_1h=None, symbol="TESTUSDT")
        # With no 1h data, regime is NEUTRAL → no entries allowed → 0 trades
        assert result.total_trades == 0
        assert len(result.equity_curve) > 0

    def test_with_1h_data(self, sample_klines_1m):
        """V2 engine completes when 1h data is provided."""
        # Generate synthetic 1h bars spanning the same period
        n_1h = len(sample_klines_1m) // 60 + 1
        np.random.seed(99)
        ts_start = sample_klines_1m[0]["timestamp"]
        klines_1h = []
        for j in range(n_1h):
            t = ts_start + j * 3_600_000
            p = 100.0 + 5 * np.sin(j / 20.0) + np.random.normal(0, 0.5)
            klines_1h.append({
                "timestamp": int(t),
                "open": p,
                "high": p + abs(np.random.normal(0, 0.3)),
                "low": p - abs(np.random.normal(0, 0.3)),
                "close": p + np.random.normal(0, 0.2),
                "volume": float(np.random.uniform(500, 2000)),
            })

        result = run_backtest_v2(sample_klines_1m, klines_1h=klines_1h, symbol="TESTUSDT")
        # Just verify it completes — trade count depends on regime alignment
        assert isinstance(result.total_trades, int)
        assert len(result.equity_curve) > 0

    def test_empty_data_returns_empty(self):
        """Empty input → empty result."""
        result = run_backtest_v2([], klines_1h=None, symbol="TESTUSDT")
        assert result.total_trades == 0
        assert result.trades == []
