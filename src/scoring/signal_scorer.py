"""Composite signal scorer — combines all indicator outputs into a 0-100 score.

Scoring rubric (LONG example — SHORT is mirrored):
    A+ Signal:       25 pts — A+ long fires this bar
    RSI Divergence:  20 pts — Regular bull divergence within 5 bars
    EMA Ribbon:      15 pts — All 5 layers bullish (short > long)
    KAMA Trend:      10 pts — KAMA rising (3-bar slope)
    RSI Position:    10 pts — RSI < 30 (oversold for long)
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
    # KAMA
    kama_bullish: bool | None = None,
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

    # 2. RSI Divergence (20 pts)
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

    # 3. EMA Ribbon (15 pts) — 5 layers, 0-5 score
    clamped_ribbon = max(0, min(5, ribbon_score))
    if is_long:
        ribbon_pct = clamped_ribbon / 5.0
    else:
        ribbon_pct = (5 - clamped_ribbon) / 5.0
    score.ema_ribbon = w.ema_ribbon * ribbon_pct

    # 4. KAMA Trend (10 pts)
    if kama_bullish is not None:
        if (is_long and kama_bullish) or (not is_long and not kama_bullish):
            score.kama_trend = w.kama_trend

    # 5. RSI Position (10 pts) — gradient
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

    # 6. HTF Alignment (10 pts) — 5 pts each for 1h and 4h
    htf_pts = 0.0
    if htf_1h_kama_bullish is not None:
        if (is_long and htf_1h_kama_bullish) or (not is_long and not htf_1h_kama_bullish):
            htf_pts += 0.5
    if htf_4h_kama_bullish is not None:
        if (is_long and htf_4h_kama_bullish) or (not is_long and not htf_4h_kama_bullish):
            htf_pts += 0.5
    score.htf_alignment = w.htf_alignment * htf_pts

    # 7. BB Position (5 pts) — RSI crossing favorable BB boundary
    if is_long and rsi_bullish_cross:
        score.bb_position = w.bb_position
    elif not is_long and rsi_bearish_cross:
        score.bb_position = w.bb_position

    # 8. Volume (5 pts) — gradient based on volume ratio
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
        + score.kama_trend + score.rsi_position + score.htf_alignment
        + score.bb_position + score.volume
    ))

    # Tier
    if score.total >= 80:
        score.tier = "A+"
    elif score.total >= 65:
        score.tier = "A"
    elif score.total >= 50:
        score.tier = "B"
    else:
        score.tier = "REJECT"

    return score
