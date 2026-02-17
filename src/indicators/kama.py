"""Kaufman's Adaptive Moving Average (KAMA) indicator.

Translated from Pine Script: /KAMA
Uses pandas_ta for the core calculation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


def compute_kama(
    close: pd.Series,
    er_length: int = 10,
    fast_length: int = 2,
    slow_length: int = 30,
) -> pd.Series:
    """Compute KAMA using pandas_ta.

    Args:
        close: Close price series.
        er_length: Efficiency Ratio lookback period.
        fast_length: Fast smoothing constant period.
        slow_length: Slow smoothing constant period.

    Returns:
        Series of KAMA values (same index as input).
    """
    result = ta.kama(close, length=er_length, fast=fast_length, slow=slow_length)
    if result is None:
        return pd.Series(np.nan, index=close.index)
    return result


def kama_slope(kama_values: pd.Series, bars: int = 3) -> pd.Series:
    """Compute KAMA slope: whether KAMA is rising or falling.

    Args:
        kama_values: KAMA series.
        bars: Number of bars to look back for slope comparison.

    Returns:
        Boolean series: True = bullish (KAMA rising), False = bearish.
    """
    return kama_values > kama_values.shift(bars)


def compute_kama_full(
    close: pd.Series,
    er_length: int = 10,
    fast_length: int = 2,
    slow_length: int = 30,
    slope_bars: int = 3,
) -> dict[str, pd.Series]:
    """Compute KAMA and its slope in one call.

    Returns:
        Dict with keys: 'kama', 'bullish' (bool series).
    """
    kama_vals = compute_kama(close, er_length, fast_length, slow_length)
    bullish = kama_slope(kama_vals, slope_bars)
    return {"kama": kama_vals, "bullish": bullish}
