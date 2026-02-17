"""TP RSI v2.0 — RSI with Bollinger Bands, 5-layer EMA ribbon, and 4-type divergence detection.

Translated from Pine Script: /TP_RSI_V2_code

Key outputs per bar:
- RSI value, BB bands on RSI, displacement thresholds
- 5-layer ribbon score (0-5): count of layers where short EMA > long EMA (all computed on RSI)
- Divergence detection: regular bullish/bearish, hidden bullish/bearish
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_ta as ta

from models import Divergence, DivergenceType


# ── Vectorized (for backtest) ──────────────────────────────


def compute_tp_rsi(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    rsi_period: int = 50,
    bb_period: int = 50,
    bb_mult: float = 1.0,
    bb_sigma: float = 0.1,
    ribbon_ma_type: str = "EMA",
    ribbon_pairs: list[list[int]] | None = None,
) -> dict[str, pd.Series]:
    """Compute all TP RSI v2 components (vectorized).

    Returns dict with keys:
        rsi, basis, upper_bb, lower_bb, disp_up, disp_down,
        bullish_cross, bearish_cross, ribbon_score,
        mashort_1..5, malong_1..5
    """
    if ribbon_pairs is None:
        ribbon_pairs = [[21, 48], [51, 98], [103, 149], [155, 199], [206, 499]]

    # RSI
    rsi = ta.rsi(close, length=rsi_period)
    if rsi is None:
        rsi = pd.Series(np.nan, index=close.index)

    # Bollinger Bands on RSI
    basis = ta.ema(rsi, length=bb_period)
    if basis is None:
        basis = pd.Series(np.nan, index=close.index)
    dev = bb_mult * rsi.rolling(bb_period).std()
    upper_bb = basis + dev
    lower_bb = basis - dev

    # Displacement thresholds
    band_width = upper_bb - lower_bb
    disp_up = basis + band_width * bb_sigma
    disp_down = basis - band_width * bb_sigma

    # Crossover/crossunder
    rsi_prev = rsi.shift(1)
    disp_up_prev = disp_up.shift(1)
    disp_down_prev = disp_down.shift(1)
    bullish_cross = (rsi > disp_up) & (rsi_prev <= disp_up_prev)
    bearish_cross = (rsi < disp_down) & (rsi_prev >= disp_down_prev)

    # 5-layer EMA ribbon on RSI
    ribbon_score = pd.Series(0, index=close.index, dtype=int)
    ribbon_data = {}

    for i, (short_len, long_len) in enumerate(ribbon_pairs, 1):
        ma_short = _compute_ma(rsi, short_len, ribbon_ma_type)
        ma_long = _compute_ma(rsi, long_len, ribbon_ma_type)
        ribbon_data[f"mashort_{i}"] = ma_short
        ribbon_data[f"malong_{i}"] = ma_long
        ribbon_score = ribbon_score + (ma_short > ma_long).astype(int)

    result = {
        "rsi": rsi,
        "basis": basis,
        "upper_bb": upper_bb,
        "lower_bb": lower_bb,
        "disp_up": disp_up,
        "disp_down": disp_down,
        "bullish_cross": bullish_cross,
        "bearish_cross": bearish_cross,
        "ribbon_score": ribbon_score,
    }
    result.update(ribbon_data)
    return result


def _compute_ma(series: pd.Series, length: int, ma_type: str = "EMA") -> pd.Series:
    """Compute moving average of given type on a series."""
    fallback = pd.Series(np.nan, index=series.index)
    func = {
        "SMA": ta.sma, "WMA": ta.wma, "HMA": ta.hma, "RMA": ta.rma,
    }.get(ma_type, ta.ema)
    result = func(series, length=length)
    return result if result is not None else fallback


# ── Divergence Detection (vectorized) ─────────────────────


def detect_divergences(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    rsi: pd.Series,
    pivot_lookback: int = 5,
    max_range: int = 60,
) -> list[Divergence]:
    """Detect all 4 types of RSI divergences.

    Mirrors the Pine Script logic:
    - Detect RSI pivot highs/lows and price pivot highs/lows
    - Compare consecutive pivot pairs within max_range bars
    - Regular Bullish: price LL + RSI HL
    - Regular Bearish: price HH + RSI LH
    - Hidden Bullish: price HL + RSI LL
    - Hidden Bearish: price LH + RSI HH

    Returns list of Divergence objects sorted by bar_index.
    """
    n = len(close)
    lb = pivot_lookback

    # Find RSI pivots
    rsi_vals = rsi.values
    high_vals = high.values
    low_vals = low.values

    rsi_highs: list[tuple[int, float]] = []  # (bar_index, rsi_value)
    rsi_lows: list[tuple[int, float]] = []
    price_highs: list[tuple[int, float]] = []
    price_lows: list[tuple[int, float]] = []

    for i in range(lb, n - lb):
        # RSI pivot high
        if _is_pivot_high(rsi_vals, i, lb):
            rsi_highs.append((i, rsi_vals[i]))
        # RSI pivot low
        if _is_pivot_low(rsi_vals, i, lb):
            rsi_lows.append((i, rsi_vals[i]))
        # Price pivot high
        if _is_pivot_high(high_vals, i, lb):
            price_highs.append((i, high_vals[i]))
        # Price pivot low
        if _is_pivot_low(low_vals, i, lb):
            price_lows.append((i, low_vals[i]))

    divergences: list[Divergence] = []

    # Check bullish divergences (at each RSI low pivot)
    for ri in range(1, len(rsi_lows)):
        curr_bar, curr_rsi = rsi_lows[ri]
        prev_bar, prev_rsi = rsi_lows[ri - 1]

        if curr_bar - prev_bar > max_range or curr_bar - prev_bar <= lb:
            continue

        # Find matching price lows
        curr_price_low = _find_nearest_pivot(price_lows, curr_bar, lb)
        prev_price_low = _find_nearest_pivot(price_lows, prev_bar, lb)
        if curr_price_low is None or prev_price_low is None:
            continue

        # Regular Bullish: price LL + RSI HL
        if curr_price_low < prev_price_low and curr_rsi > prev_rsi:
            divergences.append(Divergence(
                div_type=DivergenceType.REGULAR_BULLISH,
                bar_index=curr_bar + lb,  # confirmed lb bars after pivot
                rsi_current=curr_rsi,
                rsi_previous=prev_rsi,
                price_current=curr_price_low,
                price_previous=prev_price_low,
            ))

        # Hidden Bullish: price HL + RSI LL
        if curr_price_low > prev_price_low and curr_rsi < prev_rsi:
            divergences.append(Divergence(
                div_type=DivergenceType.HIDDEN_BULLISH,
                bar_index=curr_bar + lb,
                rsi_current=curr_rsi,
                rsi_previous=prev_rsi,
                price_current=curr_price_low,
                price_previous=prev_price_low,
            ))

    # Check bearish divergences (at each RSI high pivot)
    for ri in range(1, len(rsi_highs)):
        curr_bar, curr_rsi = rsi_highs[ri]
        prev_bar, prev_rsi = rsi_highs[ri - 1]

        if curr_bar - prev_bar > max_range or curr_bar - prev_bar <= lb:
            continue

        curr_price_high = _find_nearest_pivot(price_highs, curr_bar, lb)
        prev_price_high = _find_nearest_pivot(price_highs, prev_bar, lb)
        if curr_price_high is None or prev_price_high is None:
            continue

        # Regular Bearish: price HH + RSI LH
        if curr_price_high > prev_price_high and curr_rsi < prev_rsi:
            divergences.append(Divergence(
                div_type=DivergenceType.REGULAR_BEARISH,
                bar_index=curr_bar + lb,
                rsi_current=curr_rsi,
                rsi_previous=prev_rsi,
                price_current=curr_price_high,
                price_previous=prev_price_high,
            ))

        # Hidden Bearish: price LH + RSI HH
        if curr_price_high < prev_price_high and curr_rsi > prev_rsi:
            divergences.append(Divergence(
                div_type=DivergenceType.HIDDEN_BEARISH,
                bar_index=curr_bar + lb,
                rsi_current=curr_rsi,
                rsi_previous=prev_rsi,
                price_current=curr_price_high,
                price_previous=prev_price_high,
            ))

    divergences.sort(key=lambda d: d.bar_index)
    return divergences


def _is_pivot_high(arr: np.ndarray, idx: int, lookback: int) -> bool:
    """Check if arr[idx] is a pivot high within lookback window."""
    val = arr[idx]
    if np.isnan(val):
        return False
    start = max(0, idx - lookback)
    end = min(len(arr), idx + lookback + 1)
    for i in range(start, end):
        if i == idx:
            continue
        if arr[i] >= val:
            return False
    return True


def _is_pivot_low(arr: np.ndarray, idx: int, lookback: int) -> bool:
    """Check if arr[idx] is a pivot low within lookback window."""
    val = arr[idx]
    if np.isnan(val):
        return False
    start = max(0, idx - lookback)
    end = min(len(arr), idx + lookback + 1)
    for i in range(start, end):
        if i == idx:
            continue
        if arr[i] <= val:
            return False
    return True


def _find_nearest_pivot(
    pivots: list[tuple[int, float]], target_bar: int, max_dist: int,
) -> float | None:
    """Find the pivot value nearest to target_bar within max_dist."""
    best_val = None
    best_dist = max_dist + 1
    for bar, val in pivots:
        dist = abs(bar - target_bar)
        if dist <= max_dist and dist < best_dist:
            best_dist = dist
            best_val = val
    return best_val


# ── Build divergence lookup for backtest ───────────────────


def build_divergence_lookup(
    divergences: list[Divergence],
    window: int = 5,
) -> dict[int, list[Divergence]]:
    """Create a bar_index → divergence list mapping.

    Each divergence is accessible within `window` bars after its detection.
    This allows the scorer to check: "any divergence within last N bars?"
    """
    lookup: dict[int, list[Divergence]] = {}
    for div in divergences:
        for offset in range(window + 1):
            bar = div.bar_index + offset
            lookup.setdefault(bar, []).append(div)
    return lookup


# ── Streaming RSI (for live) ──────────────────────────────


@dataclass
class StreamingTPRSI:
    """Bar-by-bar TP RSI calculator for live trading.

    Maintains rolling buffers and computes on each new bar.
    """

    rsi_period: int = 50
    bb_period: int = 50
    bb_mult: float = 1.0
    bb_sigma: float = 0.1
    ribbon_pairs: list[list[int]] | None = None

    def __post_init__(self):
        if self.ribbon_pairs is None:
            self.ribbon_pairs = [[21, 48], [51, 98], [103, 149], [155, 199], [206, 499]]
        max_len = max(p[1] for p in self.ribbon_pairs) + self.rsi_period + 50
        self._close_buf: deque[float] = deque(maxlen=max_len)
        self._high_buf: deque[float] = deque(maxlen=max_len)
        self._low_buf: deque[float] = deque(maxlen=max_len)

    def update(self, close: float, high: float, low: float) -> dict | None:
        """Push a new bar and return computed values, or None if not enough data."""
        self._close_buf.append(close)
        self._high_buf.append(high)
        self._low_buf.append(low)

        min_needed = max(p[1] for p in self.ribbon_pairs) + self.rsi_period + 10
        if len(self._close_buf) < min_needed:
            return None

        close_s = pd.Series(list(self._close_buf))
        high_s = pd.Series(list(self._high_buf))
        low_s = pd.Series(list(self._low_buf))

        result = compute_tp_rsi(
            close_s, high_s, low_s,
            rsi_period=self.rsi_period,
            bb_period=self.bb_period,
            bb_mult=self.bb_mult,
            bb_sigma=self.bb_sigma,
            ribbon_pairs=self.ribbon_pairs,
        )

        # Return last bar values
        idx = len(close_s) - 1
        return {
            "rsi": float(result["rsi"].iloc[idx]),
            "basis": float(result["basis"].iloc[idx]),
            "upper_bb": float(result["upper_bb"].iloc[idx]),
            "lower_bb": float(result["lower_bb"].iloc[idx]),
            "disp_up": float(result["disp_up"].iloc[idx]),
            "disp_down": float(result["disp_down"].iloc[idx]),
            "bullish_cross": bool(result["bullish_cross"].iloc[idx]),
            "bearish_cross": bool(result["bearish_cross"].iloc[idx]),
            "ribbon_score": int(result["ribbon_score"].iloc[idx]),
        }
