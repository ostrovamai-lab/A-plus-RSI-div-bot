"""A+ Signal State Machine — translated from Pine Script.

Source: /A+_pine_code (Asignal_reverse_Ostrovamoi [A+ Circles])

Three-state machine:
    STATE 0 (IDLE)         → Detect swing pivot, build zone, detect first close crossing zone
                             → Green triangle (support broken) → STATE 1, dir=LONG
                             → Red triangle (resistance broken) → STATE 1, dir=SHORT

    STATE 1 (FRACTAL_HUNT) → Find fractal (pivotlow for LONG, pivothigh for SHORT) within stepWindow
                             → Track early EMA crosses
                             → Timeout → STATE 0

    STATE 2 (GATE_ACTIVE)  → Fractal found, gate = fractal price
                             → Wait for gate BREAK (price through fractal)
                             → After break, wait for EMA cross in signal direction
                             → Optional: require slow EMA retest
                             → If all conditions met → EMIT A+ SIGNAL
                             → Gate expires → STATE 0

Two implementations:
    - Vectorized: for backtest (pre-compute EMAs, run state machine as loop)
    - Streaming: for live (bar-by-bar with deque buffers)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pandas_ta as ta

from models import APlusSignal, SignalDirection


# ── Vectorized Implementation (backtest) ──────────────────


def compute_aplus_signals(
    df: pd.DataFrame,
    pivot_lookback: int = 14,
    fractal_len: int = 3,
    step_window: int = 50,
    gate_max_age: int = 20,
    ema_fast_len: int = 3,
    ema_slow_len: int = 21,
    break_mode: str = "Wick+Buffer",
    buffer_ticks: float = 1.0,
    need_retest: bool = True,
    retest_lookback: int = 20,
    allow_pre_cross: bool = True,
    tick_size: float = 0.01,
) -> dict[str, np.ndarray]:
    """Run the A+ state machine on a DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close, volume.
        All other params match Pine Script defaults.
        tick_size: Instrument minimum tick for buffer calculation.

    Returns dict with arrays (same length as df):
        aplus_long:     bool array, True on bars where A+ long signal fires
        aplus_short:    bool array, True on bars where A+ short signal fires
        fractal_price:  float array, gate fractal price at signal time
        ema_fast:       float array
        ema_slow:       float array
        triangles_long: bool array, green triangle bars
        triangles_short: bool array, red triangle bars
    """
    n = len(df)
    high_arr = df["high"].values.astype(float)
    low_arr = df["low"].values.astype(float)
    close_arr = df["close"].values.astype(float)

    # Pre-compute EMAs
    ema_fast = ta.ema(pd.Series(close_arr), length=ema_fast_len)
    ema_slow = ta.ema(pd.Series(close_arr), length=ema_slow_len)
    if ema_fast is None:
        ema_fast = pd.Series(np.nan, index=range(n))
    if ema_slow is None:
        ema_slow = pd.Series(np.nan, index=range(n))
    ema_fast_arr = ema_fast.values.astype(float)
    ema_slow_arr = ema_slow.values.astype(float)

    # Pre-compute pivot highs and lows (main swing detection)
    pivot_highs = _compute_pivots_high(high_arr, pivot_lookback)
    pivot_lows = _compute_pivots_low(low_arr, pivot_lookback)

    # Pre-compute fractal pivots (current TF)
    fractal_highs = _compute_pivots_high(high_arr, fractal_len)
    fractal_lows = _compute_pivots_low(low_arr, fractal_len)

    # Output arrays
    aplus_long = np.zeros(n, dtype=bool)
    aplus_short = np.zeros(n, dtype=bool)
    fractal_price_out = np.full(n, np.nan)
    triangles_long = np.zeros(n, dtype=bool)
    triangles_short = np.zeros(n, dtype=bool)

    # ── Swing zone tracking (for triangle detection) ──
    # High swing zone
    ph_top = np.nan
    ph_crossed = False

    # Low swing zone
    pl_btm = np.nan
    pl_crossed = False

    # ── State machine variables ──
    state = 0       # 0=idle, 1=fractal_hunt, 2=gate_active
    direction = 0   # +1=long, -1=short
    tri_bar = -1
    fract_bar = -1
    fract_price = np.nan
    pre_cross_bar_long = -1
    pre_cross_bar_short = -1
    gate_active = False
    broke_at = -1
    # retest tracking removed — retest uses loop-based check in state 2

    lb = pivot_lookback

    for i in range(lb, n):
        # ── 1. Update swing zones ──
        if pivot_highs[i]:
            ph_top = high_arr[i - lb]
            ph_crossed = False

        if not np.isnan(ph_top) and close_arr[i] > ph_top:
            if not ph_crossed:
                ph_crossed = True
                # Red triangle: resistance broken → short signal start
                triangles_short[i] = True

        if pivot_lows[i]:
            pl_btm = low_arr[i - lb]
            pl_crossed = False

        if not np.isnan(pl_btm) and close_arr[i] < pl_btm:
            if not pl_crossed:
                pl_crossed = True
                # Green triangle: support broken → long signal start
                triangles_long[i] = True

        # ── 2. Triangle detection → enter STATE 1 ──
        if triangles_long[i]:
            state = 1
            direction = 1
            tri_bar = i
            pre_cross_bar_long = -1
            pre_cross_bar_short = -1
            fract_bar = -1
            fract_price = np.nan
            gate_active = False
            broke_at = -1

        if triangles_short[i]:
            state = 1
            direction = -1
            tri_bar = i
            pre_cross_bar_long = -1
            pre_cross_bar_short = -1
            fract_bar = -1
            fract_price = np.nan
            gate_active = False
            broke_at = -1

        # ── 3. STATE 1: Early EMA cross tracking ──
        if state == 1:
            if (not np.isnan(ema_fast_arr[i]) and not np.isnan(ema_slow_arr[i])
                    and not np.isnan(ema_fast_arr[i - 1]) and not np.isnan(ema_slow_arr[i - 1])):
                if ema_fast_arr[i] > ema_slow_arr[i] and ema_fast_arr[i - 1] <= ema_slow_arr[i - 1]:
                    pre_cross_bar_long = i
                if ema_fast_arr[i] < ema_slow_arr[i] and ema_fast_arr[i - 1] >= ema_slow_arr[i - 1]:
                    pre_cross_bar_short = i

        # ── 4. STATE 1: Timeout ──
        if state == 1 and i - tri_bar > step_window:
            state = 0
            direction = 0
            tri_bar = -1
            pre_cross_bar_long = -1
            pre_cross_bar_short = -1
            fract_bar = -1
            fract_price = np.nan
            gate_active = False
            broke_at = -1

        # ── 5. STATE 1 → STATE 2: Fractal detection ──
        fl = fractal_len
        if state == 1 and i >= fl:
            # Check for new fractal at position i (confirmed fl bars ago)
            if direction == 1 and fractal_lows[i]:
                actual_bar = i - fl
                if actual_bar - tri_bar <= step_window:
                    state = 2
                    fract_bar = actual_bar
                    fract_price = low_arr[actual_bar]
                    gate_active = True
                    broke_at = -1

            if direction == -1 and fractal_highs[i]:
                actual_bar = i - fl
                if actual_bar - tri_bar <= step_window:
                    state = 2
                    fract_bar = actual_bar
                    fract_price = high_arr[actual_bar]
                    gate_active = True
                    broke_at = -1

        # ── 6. STATE 2: Gate aging ──
        if state == 2 and fract_bar >= 0 and i - fract_bar > gate_max_age:
            state = 0
            direction = 0
            tri_bar = -1
            fract_bar = -1
            fract_price = np.nan
            gate_active = False
            broke_at = -1
            continue

        # ── 7. STATE 2: Gate break detection ──
        if state == 2 and gate_active and not np.isnan(fract_price):
            buf = buffer_ticks * tick_size
            up_thresh = fract_price + buf
            down_thresh = fract_price - buf

            broke_now = False
            if direction == 1:  # Long: price must break below fractal
                if break_mode == "Wick":
                    broke_now = low_arr[i] < fract_price
                elif break_mode == "Close":
                    broke_now = close_arr[i] < fract_price
                else:  # Wick+Buffer
                    broke_now = low_arr[i] < down_thresh
            else:  # Short: price must break above fractal
                if break_mode == "Wick":
                    broke_now = high_arr[i] > fract_price
                elif break_mode == "Close":
                    broke_now = close_arr[i] > fract_price
                else:  # Wick+Buffer
                    broke_now = high_arr[i] > up_thresh

            if broke_now:
                gate_active = False
                broke_at = i

        # ── 8. STATE 2: Check for A+ signal emission ──
        if state == 2 and broke_at >= 0 and not np.isnan(fract_price):
            # Check EMA cross condition
            has_cross = False

            if not np.isnan(ema_fast_arr[i]) and not np.isnan(ema_slow_arr[i]):
                if direction == 1:
                    # Need bullish cross (fast > slow)
                    if (not np.isnan(ema_fast_arr[i - 1]) and not np.isnan(ema_slow_arr[i - 1])
                            and ema_fast_arr[i] > ema_slow_arr[i]
                            and ema_fast_arr[i - 1] <= ema_slow_arr[i - 1]):
                        has_cross = True
                    # Allow pre-cross from state 1
                    if allow_pre_cross and pre_cross_bar_long >= tri_bar:
                        has_cross = True
                    # Already crossed (fast above slow)
                    if ema_fast_arr[i] > ema_slow_arr[i] and broke_at == i:
                        has_cross = True
                else:
                    # Need bearish cross (fast < slow)
                    if (not np.isnan(ema_fast_arr[i - 1]) and not np.isnan(ema_slow_arr[i - 1])
                            and ema_fast_arr[i] < ema_slow_arr[i]
                            and ema_fast_arr[i - 1] >= ema_slow_arr[i - 1]):
                        has_cross = True
                    if allow_pre_cross and pre_cross_bar_short >= tri_bar:
                        has_cross = True
                    if ema_fast_arr[i] < ema_slow_arr[i] and broke_at == i:
                        has_cross = True

            # Check retest condition
            retest_ok = True
            if need_retest and has_cross:
                retest_ok = False
                if direction == 1:
                    # Check if price touched/retested slow EMA from above
                    for j in range(max(0, i - retest_lookback), i + 1):
                        if low_arr[j] <= ema_slow_arr[j] * 1.002:
                            retest_ok = True
                            break
                else:
                    # Check if price touched/retested slow EMA from below
                    for j in range(max(0, i - retest_lookback), i + 1):
                        if high_arr[j] >= ema_slow_arr[j] * 0.998:
                            retest_ok = True
                            break

            if has_cross and retest_ok:
                if direction == 1:
                    aplus_long[i] = True
                else:
                    aplus_short[i] = True
                fractal_price_out[i] = fract_price

                # Reset state
                state = 0
                direction = 0
                tri_bar = -1
                fract_bar = -1
                fract_price = np.nan
                gate_active = False
                broke_at = -1

    return {
        "aplus_long": aplus_long,
        "aplus_short": aplus_short,
        "fractal_price": fractal_price_out,
        "ema_fast": ema_fast_arr,
        "ema_slow": ema_slow_arr,
        "triangles_long": triangles_long,
        "triangles_short": triangles_short,
    }


def _compute_pivots_high(arr: np.ndarray, lookback: int) -> np.ndarray:
    """Compute pivot highs. Result[i] is True if arr[i-lookback] is a pivot high.

    Pine Script: ta.pivothigh(high, length, length) confirms at bar_index,
    the pivot is at bar_index - length.
    """
    n = len(arr)
    result = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        val = arr[i]
        is_pivot = True
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if j < 0 or j >= n:
                continue
            if arr[j] >= val:
                is_pivot = False
                break
        if is_pivot:
            # Confirmed at i + lookback
            confirm_bar = i + lookback
            if confirm_bar < n:
                result[confirm_bar] = True
    return result


def _compute_pivots_low(arr: np.ndarray, lookback: int) -> np.ndarray:
    """Compute pivot lows. Result[i] is True if arr[i-lookback] is a pivot low."""
    n = len(arr)
    result = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        val = arr[i]
        is_pivot = True
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if j < 0 or j >= n:
                continue
            if arr[j] <= val:
                is_pivot = False
                break
        if is_pivot:
            confirm_bar = i + lookback
            if confirm_bar < n:
                result[confirm_bar] = True
    return result


# ── Streaming Implementation (live) ──────────────────────


@dataclass
class StreamingAPlusSignal:
    """Bar-by-bar A+ signal detector for live trading.

    Push bars one at a time via update(). Returns APlusSignal when detected, else None.
    """

    pivot_lookback: int = 14
    fractal_len: int = 3
    step_window: int = 50
    gate_max_age: int = 20
    ema_fast_len: int = 3
    ema_slow_len: int = 21
    break_mode: str = "Wick+Buffer"
    buffer_ticks: float = 1.0
    need_retest: bool = True
    retest_lookback: int = 20
    allow_pre_cross: bool = True
    tick_size: float = 0.01

    # Internal state
    _bar_count: int = field(default=0, init=False)
    _state: int = field(default=0, init=False)
    _direction: int = field(default=0, init=False)
    _tri_bar: int = field(default=-1, init=False)
    _fract_bar: int = field(default=-1, init=False)
    _fract_price: float = field(default=float("nan"), init=False)
    _pre_cross_long: int = field(default=-1, init=False)
    _pre_cross_short: int = field(default=-1, init=False)
    _gate_active: bool = field(default=False, init=False)
    _broke_at: int = field(default=-1, init=False)

    # Swing zone state
    _ph_top: float = field(default=float("nan"), init=False)
    _ph_btm: float = field(default=float("nan"), init=False)
    _ph_crossed: bool = field(default=False, init=False)
    _pl_top: float = field(default=float("nan"), init=False)
    _pl_btm: float = field(default=float("nan"), init=False)
    _pl_crossed: bool = field(default=False, init=False)

    def __post_init__(self):
        max_buf = max(self.pivot_lookback, self.ema_slow_len, self.retest_lookback) * 3 + 50
        self._open_buf: deque[float] = deque(maxlen=max_buf)
        self._high_buf: deque[float] = deque(maxlen=max_buf)
        self._low_buf: deque[float] = deque(maxlen=max_buf)
        self._close_buf: deque[float] = deque(maxlen=max_buf)
        self._ema_fast_buf: deque[float] = deque(maxlen=max_buf)
        self._ema_slow_buf: deque[float] = deque(maxlen=max_buf)

    def update(
        self, open_: float, high: float, low: float, close: float, timestamp: int = 0,
    ) -> APlusSignal | None:
        """Process one new bar. Returns APlusSignal if triggered, else None."""
        self._open_buf.append(open_)
        self._high_buf.append(high)
        self._low_buf.append(low)
        self._close_buf.append(close)
        self._bar_count += 1

        i = self._bar_count - 1

        # Compute EMAs incrementally via pandas_ta on buffer
        if len(self._close_buf) >= self.ema_slow_len + 5:
            close_s = pd.Series(list(self._close_buf))
            ef = ta.ema(close_s, length=self.ema_fast_len)
            es = ta.ema(close_s, length=self.ema_slow_len)
            ema_f = float(ef.iloc[-1]) if ef is not None else float("nan")
            ema_s = float(es.iloc[-1]) if es is not None else float("nan")
        else:
            ema_f = float("nan")
            ema_s = float("nan")

        self._ema_fast_buf.append(ema_f)
        self._ema_slow_buf.append(ema_s)

        if len(self._close_buf) < self.pivot_lookback + self.fractal_len + 5:
            return None

        # Check for swing pivot highs/lows
        lb = self.pivot_lookback
        buf_len = len(self._high_buf)
        if buf_len > 2 * lb + 1:
            arr_h = list(self._high_buf)
            arr_l = list(self._low_buf)
            arr_o = list(self._open_buf)
            arr_c = list(self._close_buf)
            pivot_idx = buf_len - 1 - lb  # The candidate pivot position

            # Pivot high check
            is_ph = self._check_pivot_high(arr_h, pivot_idx, lb)
            if is_ph:
                self._ph_top = arr_h[pivot_idx]
                self._ph_btm = max(arr_c[pivot_idx], arr_o[pivot_idx])
                self._ph_crossed = False

            # Pivot low check
            is_pl = self._check_pivot_low(arr_l, pivot_idx, lb)
            if is_pl:
                self._pl_top = min(arr_c[pivot_idx], arr_o[pivot_idx])
                self._pl_btm = arr_l[pivot_idx]
                self._pl_crossed = False

        # Triangle detection
        triangle_long = False
        triangle_short = False

        if not np.isnan(self._pl_btm) and close < self._pl_btm and not self._pl_crossed:
            self._pl_crossed = True
            triangle_long = True

        if not np.isnan(self._ph_top) and close > self._ph_top and not self._ph_crossed:
            self._ph_crossed = True
            triangle_short = True

        # Enter state 1
        if triangle_long:
            self._reset_state(1, 1, i)
        if triangle_short:
            self._reset_state(1, -1, i)

        # State 1: early cross tracking
        if self._state == 1 and len(self._ema_fast_buf) >= 2:
            ef_curr = self._ema_fast_buf[-1]
            ef_prev = self._ema_fast_buf[-2]
            es_curr = self._ema_slow_buf[-1]
            es_prev = self._ema_slow_buf[-2]
            if not np.isnan(ef_curr) and not np.isnan(es_curr):
                if ef_curr > es_curr and ef_prev <= es_prev:
                    self._pre_cross_long = i
                if ef_curr < es_curr and ef_prev >= es_prev:
                    self._pre_cross_short = i

        # State 1: timeout
        if self._state == 1 and i - self._tri_bar > self.step_window:
            self._reset_state(0, 0, -1)

        # State 1 → 2: fractal detection
        fl = self.fractal_len
        if self._state == 1 and len(self._close_buf) > 2 * fl + 1:
            buf_l = list(self._low_buf)
            buf_h = list(self._high_buf)
            frac_idx = len(buf_l) - 1 - fl

            if self._direction == 1:
                if self._check_pivot_low(buf_l, frac_idx, fl):
                    actual_bar = i - fl
                    if actual_bar - self._tri_bar <= self.step_window:
                        self._state = 2
                        self._fract_bar = actual_bar
                        self._fract_price = buf_l[frac_idx]
                        self._gate_active = True
                        self._broke_at = -1

            if self._direction == -1:
                if self._check_pivot_high(buf_h, frac_idx, fl):
                    actual_bar = i - fl
                    if actual_bar - self._tri_bar <= self.step_window:
                        self._state = 2
                        self._fract_bar = actual_bar
                        self._fract_price = buf_h[frac_idx]
                        self._gate_active = True
                        self._broke_at = -1

        # State 2: gate aging
        if self._state == 2 and self._fract_bar >= 0 and i - self._fract_bar > self.gate_max_age:
            self._reset_state(0, 0, -1)
            return None

        # State 2: gate break
        if self._state == 2 and self._gate_active and not np.isnan(self._fract_price):
            buf = self.buffer_ticks * self.tick_size
            broke = False
            if self._direction == 1:
                thresh = self._fract_price - buf if self.break_mode == "Wick+Buffer" else self._fract_price
                if self.break_mode == "Close":
                    broke = close < self._fract_price
                else:
                    broke = low < thresh
            else:
                thresh = self._fract_price + buf if self.break_mode == "Wick+Buffer" else self._fract_price
                if self.break_mode == "Close":
                    broke = close > self._fract_price
                else:
                    broke = high > thresh

            if broke:
                self._gate_active = False
                self._broke_at = i

        # State 2: check for A+ signal
        if self._state == 2 and self._broke_at >= 0 and not np.isnan(self._fract_price):
            has_cross = self._check_ema_cross(i, ema_f, ema_s)

            retest_ok = True
            if self.need_retest and has_cross:
                retest_ok = self._check_retest(low, high)

            if has_cross and retest_ok:
                sig_dir = SignalDirection.LONG if self._direction == 1 else SignalDirection.SHORT
                signal = APlusSignal(
                    bar_index=i,
                    direction=sig_dir,
                    price=close,
                    fractal_price=self._fract_price,
                    ema_fast=ema_f,
                    ema_slow=ema_s,
                    timestamp=timestamp,
                )
                self._reset_state(0, 0, -1)
                return signal

        return None

    def _reset_state(self, state: int, direction: int, tri_bar: int) -> None:
        self._state = state
        self._direction = direction
        self._tri_bar = tri_bar
        self._pre_cross_long = -1
        self._pre_cross_short = -1
        self._fract_bar = -1
        self._fract_price = float("nan")
        self._gate_active = False
        self._broke_at = -1

    def _check_ema_cross(self, i: int, ema_f: float, ema_s: float) -> bool:
        if np.isnan(ema_f) or np.isnan(ema_s) or len(self._ema_fast_buf) < 2:
            return False

        ef_prev = self._ema_fast_buf[-2]
        es_prev = self._ema_slow_buf[-2]

        if self._direction == 1:
            if ema_f > ema_s and ef_prev <= es_prev:
                return True
            if self.allow_pre_cross and self._pre_cross_long >= self._tri_bar:
                return True
            if ema_f > ema_s and self._broke_at == i:
                return True
        else:
            if ema_f < ema_s and ef_prev >= es_prev:
                return True
            if self.allow_pre_cross and self._pre_cross_short >= self._tri_bar:
                return True
            if ema_f < ema_s and self._broke_at == i:
                return True
        return False

    def _check_retest(self, low: float, high: float) -> bool:
        ema_slow_list = list(self._ema_slow_buf)
        low_list = list(self._low_buf)
        high_list = list(self._high_buf)
        lookback = min(self.retest_lookback, len(ema_slow_list))

        for j in range(len(ema_slow_list) - lookback, len(ema_slow_list)):
            if j < 0 or np.isnan(ema_slow_list[j]):
                continue
            if self._direction == 1:
                if low_list[j] <= ema_slow_list[j] * 1.002:
                    return True
            else:
                if high_list[j] >= ema_slow_list[j] * 0.998:
                    return True
        return False

    @staticmethod
    def _check_pivot_high(arr: list, idx: int, lb: int) -> bool:
        if idx < lb or idx + lb >= len(arr):
            return False
        val = arr[idx]
        for j in range(idx - lb, idx + lb + 1):
            if j == idx:
                continue
            if arr[j] >= val:
                return False
        return True

    @staticmethod
    def _check_pivot_low(arr: list, idx: int, lb: int) -> bool:
        if idx < lb or idx + lb >= len(arr):
            return False
        val = arr[idx]
        for j in range(idx - lb, idx + lb + 1):
            if j == idx:
                continue
            if arr[j] <= val:
                return False
        return True
