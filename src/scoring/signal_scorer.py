"""Composite signal scorer — combines all indicator outputs into a 0-100 score.

Scoring rubric (LONG example — SHORT is mirrored):
    A+ Signal:       25 pts — A+ long fires this bar
    RSI Divergence:  15 pts — Regular bull divergence within 15 bars
    EMA Ribbon:      10 pts — All 5 layers bullish (short > long)
    ADX Trend:       15 pts — ADX confirms trend alignment
    KAMA Trend:       5 pts — KAMA rising (3-bar slope)
    EMA200 Position:  5 pts — Price above EMA200 for longs
    RSI Position:     5 pts — RSI < 30 (oversold for long)
    HTF Alignment:   10 pts — 1h + 4h KAMA both aligned
    BB Position:      5 pts — RSI crossing above lower BB
    Volume:           5 pts — Volume > 1.5x EMA(20) vol
"""

from __future__ import annotations

import numpy as np

from config import ScoringWeights
from models import Divergence, DivergenceType, SignalDirection, SignalScore


def score_signal(
    direction: SignalDirection,
    *,
    # A+ signal
    aplus_fired: bool = False,
    # RSI divergence
    recent_divergences: list[Divergence] | None = None,
    # Ribbon
    ribbon_score: int = 0,
    # ADX
    adx_value: float = 0.0,
    di_plus: float = 0.0,
    di_minus: float = 0.0,
    # KAMA
    kama_bullish: bool | None = None,
    # EMA200
    price: float = 0.0,
    ema200_value: float = 0.0,
    # RSI
    rsi_value: float = 50.0,
    # HTF
    htf_1h_kama_bullish: bool | None = None,
    htf_4h_kama_bullish: bool | None = None,
    # BB
    rsi_bullish_cross: bool = False,
    rsi_bearish_cross: bool = False,
    # Volume
    volume: float = 0.0,
    volume_ema20: float = 0.0,
    # Weights
    weights: ScoringWeights | None = None,
) -> SignalScore:
    """Compute composite score for a signal.

    Args:
        direction: LONG or SHORT.
        All other args are current-bar indicator values.
        weights: Custom scoring weights (uses defaults if None).

    Returns:
        SignalScore with total, component breakdown, and tier.
    """
    w = weights or ScoringWeights()
    recent_divergences = recent_divergences or []
    is_long = direction == SignalDirection.LONG

    score = SignalScore()

    # 1. A+ Signal (25 pts)
    if aplus_fired:
        score.aplus_signal = w.aplus_signal

    # 2. RSI Divergence (15 pts)
    if recent_divergences:
        best_div_score = 0.0
        for div in recent_divergences:
            if is_long:
                if div.div_type == DivergenceType.REGULAR_BULLISH:
                    best_div_score = max(best_div_score, 1.0)
                elif div.div_type == DivergenceType.HIDDEN_BULLISH:
                    best_div_score = max(best_div_score, 0.6)
            else:
                if div.div_type == DivergenceType.REGULAR_BEARISH:
                    best_div_score = max(best_div_score, 1.0)
                elif div.div_type == DivergenceType.HIDDEN_BEARISH:
                    best_div_score = max(best_div_score, 0.6)
        score.rsi_divergence = w.rsi_divergence * best_div_score

    # 3. EMA Ribbon (10 pts) — 5 layers, 0-5 score
    clamped_ribbon = max(0, min(5, ribbon_score))
    if is_long:
        ribbon_pct = clamped_ribbon / 5.0
    else:
        ribbon_pct = (5 - clamped_ribbon) / 5.0
    score.ema_ribbon = w.ema_ribbon * ribbon_pct

    # 4. ADX Trend Context (5 pts) — reversal-aware
    # A+ signals are reversal signals, so counter-trend entries can be valid.
    # ADX measures trend strength: high ADX = strong trend (either direction).
    # Reward: aligned trends (continuation) OR strong counter-trend (reversal).
    # Penalize: weak/ambiguous trends (ADX 15-20) where direction is unclear.
    if not np.isnan(adx_value) and not np.isnan(di_plus) and not np.isnan(di_minus):
        trend_bullish = di_plus > di_minus
        aligned = (is_long and trend_bullish) or (not is_long and not trend_bullish)

        if adx_value >= 25:
            # Strong trend — reward both alignment and strong reversals
            if aligned:
                score.adx_trend = w.adx_trend
            else:
                # Counter-trend reversal into strong trend — partial credit
                score.adx_trend = w.adx_trend * 0.5
        elif adx_value >= 15:
            # Moderate trend — partial credit either way
            score.adx_trend = w.adx_trend * 0.4
        else:
            # Ranging (ADX < 15) — mean-reversion favorable
            score.adx_trend = w.adx_trend * 0.6

    # 5. KAMA Trend (5 pts)
    if kama_bullish is not None:
        if (is_long and kama_bullish) or (not is_long and not kama_bullish):
            score.kama_trend = w.kama_trend

    # 6. EMA200 Position (5 pts) — structural trend
    if price > 0 and not np.isnan(ema200_value):
        if is_long:
            if price > ema200_value:
                score.ema200_position = w.ema200_position
            elif price > ema200_value * 0.99:  # Within 1%
                score.ema200_position = w.ema200_position * 0.3
        else:
            if price < ema200_value:
                score.ema200_position = w.ema200_position
            elif price < ema200_value * 1.01:
                score.ema200_position = w.ema200_position * 0.3

    # 7. RSI Position (5 pts) — gradient
    if is_long:
        if rsi_value <= 30:
            score.rsi_position = w.rsi_position
        elif rsi_value <= 40:
            score.rsi_position = w.rsi_position * 0.5
        elif rsi_value >= 60:
            score.rsi_position = 0.0
        else:
            score.rsi_position = w.rsi_position * max(0, (50 - rsi_value)) / 20.0
    else:
        if rsi_value >= 70:
            score.rsi_position = w.rsi_position
        elif rsi_value >= 60:
            score.rsi_position = w.rsi_position * 0.5
        elif rsi_value <= 40:
            score.rsi_position = 0.0
        else:
            score.rsi_position = w.rsi_position * max(0, (rsi_value - 50)) / 20.0

    # 8. HTF Alignment (10 pts) — 5 pts each for 1h and 4h
    htf_pts = 0.0
    if htf_1h_kama_bullish is not None:
        if (is_long and htf_1h_kama_bullish) or (not is_long and not htf_1h_kama_bullish):
            htf_pts += 0.5
    if htf_4h_kama_bullish is not None:
        if (is_long and htf_4h_kama_bullish) or (not is_long and not htf_4h_kama_bullish):
            htf_pts += 0.5
    score.htf_alignment = w.htf_alignment * htf_pts

    # 9. BB Position (5 pts) — RSI crossing favorable BB boundary
    if is_long and rsi_bullish_cross:
        score.bb_position = w.bb_position
    elif not is_long and rsi_bearish_cross:
        score.bb_position = w.bb_position

    # 10. Volume (5 pts) — gradient based on volume ratio
    if volume_ema20 > 0 and not np.isnan(volume_ema20) and not np.isnan(volume):
        vol_ratio = volume / volume_ema20
        if vol_ratio >= 2.0:
            score.volume = w.volume
        elif vol_ratio >= 1.5:
            score.volume = w.volume * 0.75
        elif vol_ratio >= 1.0:
            score.volume = w.volume * 0.25
        else:
            score.volume = 0.0

    # Total (clamped to 0-100)
    score.total = min(100.0, (
        score.aplus_signal + score.rsi_divergence + score.ema_ribbon
        + score.adx_trend + score.kama_trend + score.ema200_position
        + score.rsi_position + score.htf_alignment
        + score.bb_position + score.volume
    ))

    # Tier
    if score.total >= 80:
        score.tier = "A+"
    elif score.total >= 65:
        score.tier = "A"
    elif score.total >= 50:
        score.tier = "B"
    elif score.total >= 35:
        score.tier = "C"
    else:
        score.tier = "REJECT"

    return score
