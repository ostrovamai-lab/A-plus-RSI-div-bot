"""Tests for signal scoring system."""

import pytest

from models import Divergence, DivergenceType, SignalDirection
from scoring.signal_scorer import score_signal


class TestSignalScorer:
    """Test composite score computation."""

    def test_perfect_long_score(self):
        """All conditions met for a perfect long signal → score = 100."""
        score = score_signal(
            SignalDirection.LONG,
            aplus_fired=True,
            recent_divergences=[
                Divergence(DivergenceType.REGULAR_BULLISH, 100, 25, 20, 40, 45),
            ],
            ribbon_score=5,             # All 5 layers bullish
            kama_bullish=True,
            rsi_value=25.0,             # Oversold
            htf_1h_kama_bullish=True,
            htf_4h_kama_bullish=True,
            rsi_bullish_cross=True,
            volume=300.0,
            volume_ema20=100.0,         # 3x average
        )
        assert score.total == 100
        assert score.tier == "A+"

    def test_minimal_long_score(self):
        """Only A+ signal fires, nothing else → 25 points (reject)."""
        score = score_signal(
            SignalDirection.LONG,
            aplus_fired=True,
        )
        assert score.aplus_signal == 25
        assert score.total < 50
        assert score.tier == "REJECT"

    def test_no_signal_zero(self):
        """No A+ signal → 0 for A+ component."""
        score = score_signal(
            SignalDirection.LONG,
            aplus_fired=False,
        )
        assert score.aplus_signal == 0

    def test_short_signal_scoring(self):
        """Short signal: high RSI is favorable, ribbon should be bearish."""
        score = score_signal(
            SignalDirection.SHORT,
            aplus_fired=True,
            ribbon_score=0,             # All bearish
            rsi_value=75.0,             # Overbought (good for short)
            kama_bullish=False,         # Falling KAMA
        )
        assert score.ema_ribbon == 15   # Full ribbon points (weight=15)
        assert score.rsi_position == 10  # Full RSI position points (weight=10)
        assert score.kama_trend == 10    # Full KAMA points (weight=10)

    def test_hidden_divergence_partial_score(self):
        """Hidden divergence gives 60% of the divergence weight (20)."""
        score = score_signal(
            SignalDirection.LONG,
            aplus_fired=True,
            recent_divergences=[
                Divergence(DivergenceType.HIDDEN_BULLISH, 100, 30, 35, 45, 40),
            ],
        )
        assert score.rsi_divergence == pytest.approx(20 * 0.6)

    def test_tier_boundaries(self):
        """Test tier classification at boundaries."""
        # A+ tier (>= 80)
        score = score_signal(
            SignalDirection.LONG,
            aplus_fired=True,
            recent_divergences=[Divergence(DivergenceType.REGULAR_BULLISH, 100, 25, 20, 40, 45)],
            ribbon_score=5,
            kama_bullish=True,
            rsi_value=25.0,
            htf_1h_kama_bullish=True,
            htf_4h_kama_bullish=True,
        )
        assert score.tier == "A+"

    def test_position_scale_by_tier(self):
        """Position scale should decrease with lower tiers."""
        # A+ tier
        score_aplus = score_signal(
            SignalDirection.LONG, aplus_fired=True,
            recent_divergences=[Divergence(DivergenceType.REGULAR_BULLISH, 100, 25, 20, 40, 45)],
            ribbon_score=5, kama_bullish=True, rsi_value=25.0,
            htf_1h_kama_bullish=True, htf_4h_kama_bullish=True,
            rsi_bullish_cross=True, volume=300, volume_ema20=100,
        )
        # Reject tier
        score_reject = score_signal(
            SignalDirection.LONG, aplus_fired=False,
        )

        assert score_aplus.position_scale > score_reject.position_scale
        assert score_reject.position_scale == 0.0

    def test_adx_trend_context(self):
        """ADX trend: reversal-aware scoring. Default weight=0, test with custom."""
        from config import ScoringWeights
        w = ScoringWeights(adx_trend=10.0)  # Enable for testing

        # Aligned bullish trend (ADX>=25, DI+ > DI-)
        s_aligned = score_signal(
            SignalDirection.LONG, aplus_fired=True,
            adx_value=30.0, di_plus=30.0, di_minus=15.0, weights=w,
        )
        assert s_aligned.adx_trend == 10.0

        # Counter-trend reversal (ADX>=25, DI- > DI+, going LONG)
        s_counter = score_signal(
            SignalDirection.LONG, aplus_fired=True,
            adx_value=30.0, di_plus=10.0, di_minus=30.0, weights=w,
        )
        assert s_counter.adx_trend == pytest.approx(10 * 0.5)

        # Ranging (ADX < 15) — mean-reversion favorable
        s_range = score_signal(
            SignalDirection.LONG, aplus_fired=True,
            adx_value=12.0, di_plus=20.0, di_minus=18.0, weights=w,
        )
        assert s_range.adx_trend == pytest.approx(10 * 0.6)

    def test_ema200_position(self):
        """EMA200: price above for longs = full, below = 0. Default weight=0, test with custom."""
        from config import ScoringWeights
        w = ScoringWeights(ema200_position=5.0)  # Enable for testing

        s_above = score_signal(
            SignalDirection.LONG, aplus_fired=True,
            price=100.0, ema200_value=95.0, weights=w,
        )
        assert s_above.ema200_position == 5.0

        s_below = score_signal(
            SignalDirection.LONG, aplus_fired=True,
            price=90.0, ema200_value=95.0, weights=w,
        )
        assert s_below.ema200_position == 0.0

        # Short: price below EMA200 = full
        s_short = score_signal(
            SignalDirection.SHORT, aplus_fired=True,
            price=90.0, ema200_value=95.0, weights=w,
        )
        assert s_short.ema200_position == 5.0

    def test_volume_gradient(self):
        """Volume scoring: 2x = full, 1.5x = 75%, 1x = 25%, 0.5x = 0."""
        # High volume
        s1 = score_signal(SignalDirection.LONG, aplus_fired=True,
                          volume=200, volume_ema20=100)
        # Low volume
        s2 = score_signal(SignalDirection.LONG, aplus_fired=True,
                          volume=50, volume_ema20=100)
        assert s1.volume > s2.volume
