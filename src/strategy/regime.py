"""SMC regime detection — wraps the VectorizedSMCEngine for 1h/8m analysis.

Provides:
- 1h swing structure (regime bias + CHoCH/BOS)
- 8m structure (FVGs, OBs, equal levels) for SL placement
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pandas as pd

# SMC library lives outside the project tree
_SMC_PATH = "/home/user/Documents/INDICATORS/smc"
if _SMC_PATH not in sys.path:
    sys.path.insert(0, _SMC_PATH)

from smc import Bias, SMCConfig, VectorizedSMCEngine  # noqa: E402
from smc.engine.vectorized import SMCResult  # noqa: E402
from smc.types import FairValueGap, OrderBlock  # noqa: E402

if TYPE_CHECKING:
    from smc.signals import BarSignals

__all__ = [
    "Bias",
    "SMCResult",
    "build_smc_1h",
    "build_smc_8m",
    "get_regime_at",
    "check_choch_at",
    "find_sl_from_fvg",
    "find_sl_from_ob",
]


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def build_smc_1h(df_1h: pd.DataFrame, swing_length: int = 10) -> SMCResult:
    """Run VectorizedSMCEngine on 1h data for regime detection.

    Only swing structure + trailing HL are enabled (no FVG/OB on 1h).
    """
    config = SMCConfig(
        swing_length=swing_length,
        detect_swing_structure=True,
        detect_trailing_hl=True,
        detect_internal_structure=False,
        detect_internal_ob=False,
        detect_swing_ob=False,
        detect_fvg=False,
        detect_equal_hl=False,
        timeframe="60",
    )
    engine = VectorizedSMCEngine(config)
    return engine.analyze(df_1h)


def build_smc_8m(
    df_8m: pd.DataFrame,
    swing_length: int = 20,
    internal_length: int = 5,
) -> SMCResult:
    """Run VectorizedSMCEngine on 8m data with full feature set.

    FVGs, OBs (swing + internal), equal levels, and trailing HL are all
    enabled — used for SL placement and premium/discount filtering.
    """
    config = SMCConfig(
        swing_length=swing_length,
        internal_length=internal_length,
        detect_swing_structure=True,
        detect_internal_structure=True,
        detect_fvg=True,
        detect_swing_ob=True,
        detect_internal_ob=True,
        detect_equal_hl=True,
        detect_trailing_hl=True,
    )
    engine = VectorizedSMCEngine(config)
    return engine.analyze(df_8m)


# ---------------------------------------------------------------------------
# Per-bar accessors
# ---------------------------------------------------------------------------

def get_regime_at(smc_1h: SMCResult, h1_idx: int | None) -> Bias:
    """Return the 1h swing bias at *h1_idx*, or ``Bias.NEUTRAL`` if unavailable."""
    if h1_idx is None or h1_idx < 0 or h1_idx >= len(smc_1h.swing_bias):
        return Bias.NEUTRAL
    return smc_1h.swing_bias[h1_idx]


def check_choch_at(smc_1h: SMCResult, h1_idx: int | None) -> tuple[bool, bool]:
    """Return ``(bullish_choch, bearish_choch)`` flags at 1h bar *h1_idx*.

    Both are ``False`` when *h1_idx* is ``None`` or out of range.
    """
    if h1_idx is None or h1_idx < 0 or h1_idx >= len(smc_1h.signals):
        return False, False
    sig: BarSignals = smc_1h.signals[h1_idx]
    return sig.swing_bullish_choch, sig.swing_bearish_choch


# ---------------------------------------------------------------------------
# Structure-based SL finders
# ---------------------------------------------------------------------------

def find_sl_from_fvg(
    direction_is_long: bool,
    entry_price: float,
    smc_8m: SMCResult,
    atr_buffer: float,
    min_dist: float,
    max_dist: float,
) -> float | None:
    """Find an SL level from the nearest unfilled FVG.

    For LONG: finds the nearest **bullish** FVG whose bottom is below
    *entry_price*, then sets SL = ``fvg.bottom - atr_buffer``.

    For SHORT: nearest **bearish** FVG above entry, SL = ``fvg.top + atr_buffer``.

    Returns ``None`` if no suitable FVG exists or the distance is outside
    [*min_dist*, *max_dist*] (expressed as absolute price distance).
    """
    best: float | None = None

    for fvg in smc_8m.fvgs:
        if fvg.mitigated:
            continue

        if direction_is_long:
            if fvg.bias != Bias.BULLISH:
                continue
            level = float(fvg.bottom) - atr_buffer
            if level >= entry_price:
                continue
            dist = entry_price - level
        else:
            if fvg.bias != Bias.BEARISH:
                continue
            level = float(fvg.top) + atr_buffer
            if level <= entry_price:
                continue
            dist = level - entry_price

        if dist < min_dist or dist > max_dist:
            continue

        # Pick tightest (closest to entry)
        if direction_is_long:
            if best is None or level > best:
                best = level
        else:
            if best is None or level < best:
                best = level

    return best


def find_sl_from_ob(
    direction_is_long: bool,
    entry_price: float,
    smc_8m: SMCResult,
    atr_buffer: float,
    min_dist: float,
    max_dist: float,
) -> float | None:
    """Find an SL level from the nearest active order block.

    For LONG: nearest **bullish** OB below entry, SL = ``ob.bar_low - atr_buffer``.
    For SHORT: nearest **bearish** OB above entry, SL = ``ob.bar_high + atr_buffer``.

    Returns ``None`` when no valid OB exists within distance bounds.
    """
    best: float | None = None

    for ob in smc_8m.order_blocks:
        if ob.mitigated:
            continue

        if direction_is_long:
            if ob.bias != Bias.BULLISH:
                continue
            level = float(ob.bar_low) - atr_buffer
            if level >= entry_price:
                continue
            dist = entry_price - level
        else:
            if ob.bias != Bias.BEARISH:
                continue
            level = float(ob.bar_high) + atr_buffer
            if level <= entry_price:
                continue
            dist = level - entry_price

        if dist < min_dist or dist > max_dist:
            continue

        if direction_is_long:
            if best is None or level > best:
                best = level
        else:
            if best is None or level < best:
                best = level

    return best
