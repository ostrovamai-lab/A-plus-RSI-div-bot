"""Shared pivot detection — single source of truth for pivot highs/lows.

Three interfaces for different contexts:
- Vectorized (numpy array → boolean array): for backtest pre-computation
- Point-check (numpy array, index → bool): for divergence detection
- Streaming (list, index → bool): for live bar-by-bar processing
"""

from __future__ import annotations

import numpy as np


def compute_pivots_high(arr: np.ndarray, lookback: int) -> np.ndarray:
    """Compute pivot highs over an array (vectorized).

    Result[i] is True if arr[i-lookback] is a pivot high, confirmed at bar i.
    Mirrors Pine Script: ta.pivothigh(series, length, length).
    """
    n = len(arr)
    result = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        if _is_extremum(arr, i, lookback, high=True):
            confirm_bar = i + lookback
            if confirm_bar < n:
                result[confirm_bar] = True
    return result


def compute_pivots_low(arr: np.ndarray, lookback: int) -> np.ndarray:
    """Compute pivot lows over an array (vectorized).

    Result[i] is True if arr[i-lookback] is a pivot low, confirmed at bar i.
    """
    n = len(arr)
    result = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        if _is_extremum(arr, i, lookback, high=False):
            confirm_bar = i + lookback
            if confirm_bar < n:
                result[confirm_bar] = True
    return result


def is_pivot_high(arr, idx: int, lookback: int) -> bool:
    """Check if arr[idx] is a pivot high (works with numpy arrays or lists)."""
    n = len(arr)
    if idx < lookback or idx + lookback >= n:
        return False
    val = arr[idx]
    if isinstance(val, float) and np.isnan(val):
        return False
    start = max(0, idx - lookback)
    end = min(n, idx + lookback + 1)
    for i in range(start, end):
        if i == idx:
            continue
        if arr[i] >= val:
            return False
    return True


def is_pivot_low(arr, idx: int, lookback: int) -> bool:
    """Check if arr[idx] is a pivot low (works with numpy arrays or lists)."""
    n = len(arr)
    if idx < lookback or idx + lookback >= n:
        return False
    val = arr[idx]
    if isinstance(val, float) and np.isnan(val):
        return False
    start = max(0, idx - lookback)
    end = min(n, idx + lookback + 1)
    for i in range(start, end):
        if i == idx:
            continue
        if arr[i] <= val:
            return False
    return True


def _is_extremum(arr: np.ndarray, idx: int, lookback: int, *, high: bool) -> bool:
    """Core extremum check (shared by vectorized functions)."""
    val = arr[idx]
    for j in range(idx - lookback, idx + lookback + 1):
        if j == idx:
            continue
        if j < 0 or j >= len(arr):
            continue
        if high and arr[j] >= val:
            return False
        if not high and arr[j] <= val:
            return False
    return True
