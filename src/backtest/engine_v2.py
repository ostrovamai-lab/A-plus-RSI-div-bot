"""V2 Backtest engine — trend-following SMC + A+ strategy.

Paradigm shift from V1:
- 1h SMC structure defines the regime (BULLISH / BEARISH / NEUTRAL)
- 8m A+ signals provide entry timing *within* the trend
- Pyramiding amplifies winners
- Structure-based SL (FVG / OB awareness) protects against liquidity sweeps
- Sessions replace isolated positions: one trend = one session with N levels

V1 engine is untouched for backward compatibility.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import BotConfig, load_config
from indicators.resampler import klines_to_df, resample_ohlcv
from indicators.suite import compute_indicator_suite
from models import BacktestResult, SignalDirection, SignalScore, Trade
from scoring.signal_scorer import score_signal
from strategy.exit_engine import check_stop_loss
from strategy.pnl import compute_fee, compute_pnl
from strategy.regime import (
    Bias,
    SMCResult,
    build_smc_1h,
    build_smc_8m,
    check_choch_at,
    find_sl_from_fvg,
    find_sl_from_ob,
    get_regime_at,
)
from strategy.risk_manager import RiskManager
from strategy.sunday_open import compute_sunday_open, sunday_open_confirms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PyramidLevel:
    """Single entry within a trend session."""

    bar_index: int
    entry_price: float
    qty: float  # leveraged qty
    score: float
    fractal_price: float


@dataclass
class TrendSession:
    """Active trend-following session (can hold multiple pyramid levels)."""

    session_id: int
    direction: SignalDirection
    symbol: str
    regime_bias: int  # Bias.BULLISH or Bias.BEARISH at open
    open_bar: int
    levels: list[PyramidLevel] = field(default_factory=list)
    sl_price: float = 0.0
    initial_sl: float = 0.0
    peak_price: float = 0.0
    choch_partial_done: bool = False

    @property
    def avg_entry(self) -> float:
        total_qty = sum(lv.qty for lv in self.levels)
        if total_qty == 0:
            return 0.0
        return sum(lv.entry_price * lv.qty for lv in self.levels) / total_qty

    @property
    def total_qty(self) -> float:
        return sum(lv.qty for lv in self.levels)

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def unrealized_pnl(self, price: float) -> float:
        """Compute mark-to-market PnL at *price*."""
        return compute_pnl(self.direction, self.avg_entry, price, self.total_qty)


# ---------------------------------------------------------------------------
# Main backtest function
# ---------------------------------------------------------------------------

def run_backtest_v2(
    klines_1m: list[dict],
    klines_1h: list[dict] | None = None,
    config: BotConfig | None = None,
    symbol: str = "BTCUSDT",
) -> BacktestResult:
    """Run V2 trend-following backtest.

    Args:
        klines_1m: 1-minute OHLCV bars (resampled to 8m internally).
        klines_1h: 1-hour bars for SMC regime detection.
        config:    Bot configuration (uses ``cfg.v2`` for V2-specific params).
        symbol:    Symbol being tested.

    Returns:
        BacktestResult with trades, equity curve, and statistics.
    """
    cfg = config or load_config()
    v2 = cfg.v2
    result = BacktestResult()

    # ── 1. Resample 1m → 8m ──────────────────────────────────────────────
    df_1m = klines_to_df(klines_1m)
    if df_1m.empty:
        logger.warning("Empty 1m data — cannot backtest")
        return result

    df_8m = resample_ohlcv(df_1m, interval_minutes=8)
    if len(df_8m) < 100:
        logger.warning("Not enough 8m bars (%d) — need at least 100", len(df_8m))
        return result

    # ── 2. Pre-compute indicator suite (A+ signals, ATR, vol_ema) ────────
    ind = compute_indicator_suite(df_8m, cfg)

    close_arr = df_8m["close"].values.astype(float)
    high_arr = df_8m["high"].values.astype(float)
    low_arr = df_8m["low"].values.astype(float)
    vol_arr = df_8m["volume"].values.astype(float)
    ts_8m = df_8m["timestamp"].values.astype(int)
    n = len(df_8m)

    # ── 3. SMC analysis ──────────────────────────────────────────────────
    # 1h regime
    if klines_1h:
        df_1h = klines_to_df(klines_1h)
        smc_1h = build_smc_1h(df_1h, swing_length=v2.smc_1h_swing_length)
        htf_1h_map = _build_htf_map(ts_8m, klines_1h)
    else:
        smc_1h = None
        htf_1h_map = {}

    # 8m structure (FVGs, OBs)
    smc_8m = build_smc_8m(
        df_8m,
        swing_length=v2.smc_8m_swing_length,
        internal_length=v2.smc_8m_internal_length,
    )

    # ── 4. Sunday Open ───────────────────────────────────────────────────
    so_prices = compute_sunday_open(df_8m)

    # ── 5. Simulation state ──────────────────────────────────────────────
    risk = RiskManager(
        initial_capital=cfg.capital_usd,
        leverage=cfg.leverage,
        max_drawdown_pct=cfg.max_drawdown_pct,
        reinvest=cfg.reinvestment.enabled,
        min_scale=cfg.reinvestment.min_scale,
        max_scale=cfg.reinvestment.max_scale,
    )

    sessions: list[TrendSession] = []
    _next_session_id = 0
    start_bar = max(cfg.aplus.pivot_lookback + 10, 50)

    # ── 6. Bar loop ──────────────────────────────────────────────────────
    for i in range(start_bar, n):
        ts = int(ts_8m[i])
        price = close_arr[i]
        hi = high_arr[i]
        lo = low_arr[i]

        if risk.is_halted:
            result.equity_curve.append(risk.current_equity)
            result.timestamps.append(ts)
            continue

        # ── 6.1 REGIME ────────────────────────────────────────────────
        h1_idx = htf_1h_map.get(ts)
        if smc_1h is not None:
            current_bias = get_regime_at(smc_1h, h1_idx)
            bull_choch, bear_choch = check_choch_at(smc_1h, h1_idx)
        else:
            current_bias = Bias.NEUTRAL
            bull_choch, bear_choch = False, False

        # ── 6.2 EXIT CHECKS ──────────────────────────────────────────
        for sess in list(sessions):
            # a. SL hit?
            if sess.sl_price > 0:
                sl_hit = check_stop_loss(sess.direction, sess.sl_price, hi, lo)
                if sl_hit:
                    trade = _close_session_full(sess, sl_hit.exit_price, i, "sl", cfg)
                    result.trades.append(trade)
                    risk.update_equity(trade.pnl - trade.fee)
                    sessions.remove(sess)
                    continue

            # b. Regime flipped?
            if smc_1h is not None and _regime_flipped(sess, current_bias):
                trade = _close_session_full(sess, price, i, "regime_flip", cfg)
                result.trades.append(trade)
                risk.update_equity(trade.pnl - trade.fee)
                sessions.remove(sess)
                continue

            # c. CHoCH counter → partial close
            if not sess.choch_partial_done and _choch_is_counter(sess, bull_choch, bear_choch):
                partial_trade = _close_session_partial(
                    sess, v2.choch_partial_pct, price, i, "choch_partial", cfg,
                )
                if partial_trade is not None:
                    result.trades.append(partial_trade)
                    risk.update_equity(partial_trade.pnl - partial_trade.fee)
                    sess.choch_partial_done = True
                    # Tighten SL to avg_entry (breakeven)
                    _tighten_sl_to_entry(sess)

            # d. Trailing SL update
            _update_trailing_sl(sess, hi, lo, i, low_arr, high_arr, v2)

        # ── 6.3 ENTRY / PYRAMID ──────────────────────────────────────
        aplus_long = bool(ind.aplus["aplus_long"][i])
        aplus_short = bool(ind.aplus["aplus_short"][i])

        for sig_dir, sig_fired in [
            (SignalDirection.LONG, aplus_long),
            (SignalDirection.SHORT, aplus_short),
        ]:
            if not sig_fired:
                continue

            fractal_p = float(ind.aplus["fractal_price"][i])
            if np.isnan(fractal_p):
                continue

            # a. Regime allows direction?
            if smc_1h is not None and not _regime_allows(current_bias, sig_dir):
                continue

            # b. Volume confirmed?
            vol_ema_val = float(ind.vol_ema.iloc[i]) if not np.isnan(ind.vol_ema.iloc[i]) else 0.0
            if vol_ema_val > 0 and vol_arr[i] < v2.volume_mult * vol_ema_val:
                continue

            # c. Sunday Open?
            if v2.require_sunday_open:
                is_long = sig_dir == SignalDirection.LONG
                if not sunday_open_confirms(is_long, price, so_prices[i]):
                    continue

            # d. Score
            entry_score = _compute_entry_score(
                sig_dir, i, ind, cfg, ts, htf_1h_map, smc_1h,
                vol_arr, close_arr,
            )
            if entry_score.total < cfg.min_score:
                continue

            # e. Existing session in same direction? → pyramid
            existing = _find_session(sessions, sig_dir, symbol)
            if existing is not None:
                if existing.num_levels >= v2.max_pyramid_levels:
                    continue
                # Pyramid sizing
                qty = _compute_pyramid_qty(existing, price, risk, cfg)
                if qty <= 0:
                    continue
                existing.levels.append(PyramidLevel(
                    bar_index=i,
                    entry_price=price,
                    qty=qty,
                    score=entry_score.total,
                    fractal_price=fractal_p,
                ))
                continue

            # f. New session
            if len(sessions) >= v2.max_sessions:
                continue

            atr_val = float(ind.atr.iloc[i]) if not np.isnan(ind.atr.iloc[i]) else 0.0
            sl = _find_sl_from_structure(
                sig_dir, price, i, smc_8m, atr_val, fractal_p, low_arr, high_arr, v2,
            )
            base_qty = _compute_base_qty(price, risk, entry_score, cfg.leverage)
            if base_qty <= 0:
                continue

            _next_session_id += 1
            sess = TrendSession(
                session_id=_next_session_id,
                direction=sig_dir,
                symbol=symbol,
                regime_bias=int(current_bias),
                open_bar=i,
                levels=[PyramidLevel(
                    bar_index=i,
                    entry_price=price,
                    qty=base_qty,
                    score=entry_score.total,
                    fractal_price=fractal_p,
                )],
                sl_price=sl,
                initial_sl=sl,
                peak_price=price,
            )
            sessions.append(sess)

        # ── 6.4 DRAWDOWN ─────────────────────────────────────────────
        if risk.check_drawdown():
            for sess in list(sessions):
                trade = _close_session_full(sess, price, i, "drawdown", cfg)
                result.trades.append(trade)
                risk.update_equity(trade.pnl - trade.fee)
            sessions.clear()

        # ── 6.5 EQUITY TRACKING ──────────────────────────────────────
        unrealized = sum(s.unrealized_pnl(price) for s in sessions)
        result.equity_curve.append(risk.current_equity + unrealized)
        result.timestamps.append(ts)

    # ── 7. Close remaining sessions ──────────────────────────────────────
    final_price = close_arr[-1]
    for sess in sessions:
        trade = _close_session_full(sess, final_price, n - 1, "backtest_end", cfg)
        result.trades.append(trade)
        risk.update_equity(trade.pnl - trade.fee)

    # ── 8. Stats (reuse V1) ──────────────────────────────────────────────
    from backtest.engine import _compute_stats
    _compute_stats(result)

    return result


# ---------------------------------------------------------------------------
# Session close helpers
# ---------------------------------------------------------------------------

def _close_session_full(
    sess: TrendSession,
    exit_price: float,
    bar_index: int,
    reason: str,
    cfg: BotConfig,
) -> Trade:
    """Close an entire session and return a single Trade."""
    avg = sess.avg_entry
    qty = sess.total_qty
    pnl = compute_pnl(sess.direction, avg, exit_price, qty)
    fee = compute_fee(avg, qty, cfg.leverage, reason,
                      cfg.v2.maker_fee, cfg.v2.taker_fee)

    avg_score = (
        sum(lv.score for lv in sess.levels) / len(sess.levels)
        if sess.levels else 0.0
    )

    return Trade(
        open_time=sess.open_bar,
        close_time=bar_index,
        symbol=sess.symbol,
        side="Buy" if sess.direction == SignalDirection.LONG else "Sell",
        entry_price=avg,
        exit_price=exit_price,
        qty=qty,
        pnl=pnl,
        fee=fee,
        reason=reason,
        score=avg_score,
        grid_fills=0,
        pyramid_level=sess.num_levels - 1,
    )


def _close_session_partial(
    sess: TrendSession,
    fraction: float,
    exit_price: float,
    bar_index: int,
    reason: str,
    cfg: BotConfig,
) -> Trade | None:
    """Close *fraction* of the session, reducing each level proportionally.

    Returns a Trade for the closed portion, or None if nothing to close.
    """
    if not sess.levels or fraction <= 0:
        return None

    close_qty = sess.total_qty * fraction
    if close_qty <= 0:
        return None

    avg = sess.avg_entry
    pnl = compute_pnl(sess.direction, avg, exit_price, close_qty)
    fee = compute_fee(avg, close_qty, cfg.leverage, reason,
                      cfg.v2.maker_fee, cfg.v2.taker_fee)

    # Reduce each level proportionally
    remaining_fraction = 1.0 - fraction
    for lv in sess.levels:
        lv.qty *= remaining_fraction

    # Remove levels with negligible qty
    sess.levels = [lv for lv in sess.levels if lv.qty > 1e-12]

    avg_score = (
        sum(lv.score for lv in sess.levels) / len(sess.levels)
        if sess.levels else 0.0
    )

    return Trade(
        open_time=sess.open_bar,
        close_time=bar_index,
        symbol=sess.symbol,
        side="Buy" if sess.direction == SignalDirection.LONG else "Sell",
        entry_price=avg,
        exit_price=exit_price,
        qty=close_qty,
        pnl=pnl,
        fee=fee,
        reason=reason,
        score=avg_score,
        grid_fills=0,
        pyramid_level=sess.num_levels,
    )


# ---------------------------------------------------------------------------
# Regime helpers
# ---------------------------------------------------------------------------

def _regime_allows(bias: Bias, direction: SignalDirection) -> bool:
    """True when *bias* permits *direction*."""
    if bias == Bias.NEUTRAL:
        return False
    if bias == Bias.BULLISH and direction == SignalDirection.LONG:
        return True
    if bias == Bias.BEARISH and direction == SignalDirection.SHORT:
        return True
    return False


def _regime_flipped(sess: TrendSession, current_bias: Bias) -> bool:
    """True when *current_bias* is the opposite of the session's opening bias."""
    if sess.direction == SignalDirection.LONG and current_bias == Bias.BEARISH:
        return True
    if sess.direction == SignalDirection.SHORT and current_bias == Bias.BULLISH:
        return True
    return False


def _choch_is_counter(
    sess: TrendSession,
    bull_choch: bool,
    bear_choch: bool,
) -> bool:
    """True when a CHoCH fires *against* the session direction."""
    if sess.direction == SignalDirection.LONG and bear_choch:
        return True
    if sess.direction == SignalDirection.SHORT and bull_choch:
        return True
    return False


# ---------------------------------------------------------------------------
# SL helpers
# ---------------------------------------------------------------------------

def _update_trailing_sl(
    sess: TrendSession,
    hi: float,
    lo: float,
    bar_idx: int,
    low_arr: np.ndarray,
    high_arr: np.ndarray,
    v2,
) -> None:
    """Update peak price and trailing SL.  SL only tightens."""
    if sess.direction == SignalDirection.LONG:
        if hi > sess.peak_price:
            sess.peak_price = hi
        # Trail: min(low) over lookback
        lookback_start = max(0, bar_idx - v2.trail_lookback + 1)
        trail_level = float(np.min(low_arr[lookback_start : bar_idx + 1]))
        if trail_level > sess.sl_price:
            sess.sl_price = trail_level
    else:
        if lo < sess.peak_price:
            sess.peak_price = lo
        lookback_start = max(0, bar_idx - v2.trail_lookback + 1)
        trail_level = float(np.max(high_arr[lookback_start : bar_idx + 1]))
        if trail_level < sess.sl_price:
            sess.sl_price = trail_level


def _tighten_sl_to_entry(sess: TrendSession) -> None:
    """Move SL to avg_entry (breakeven) — only if it tightens."""
    avg = sess.avg_entry
    if sess.direction == SignalDirection.LONG:
        if avg > sess.sl_price:
            sess.sl_price = avg
    else:
        if avg < sess.sl_price:
            sess.sl_price = avg


def _find_sl_from_structure(
    direction: SignalDirection,
    price: float,
    bar_idx: int,
    smc_8m: SMCResult,
    atr_val: float,
    fractal_price: float,
    low_arr: np.ndarray,
    high_arr: np.ndarray,
    v2,
) -> float:
    """Pick best SL from structure, with fallbacks.

    Priority (for LONG — SHORT is mirrored):
    1. Below nearest unfilled bullish FVG - ATR buffer
    2. Below nearest bullish swing OB - ATR buffer
    3. Below recent swing low (min of last 20 lows) - ATR buffer
    4. Below A+ fractal price - ATR buffer
    → Pick tightest (highest for LONG) that's still below entry, within bounds.
    """
    is_long = direction == SignalDirection.LONG
    buffer = atr_val * v2.sl_atr_buffer_mult
    min_dist = price * v2.sl_min_distance_pct
    max_dist = price * v2.sl_max_distance_pct

    candidates: list[float] = []

    # 1. FVG
    fvg_sl = find_sl_from_fvg(is_long, price, smc_8m, buffer, min_dist, max_dist)
    if fvg_sl is not None:
        candidates.append(fvg_sl)

    # 2. OB
    ob_sl = find_sl_from_ob(is_long, price, smc_8m, buffer, min_dist, max_dist)
    if ob_sl is not None:
        candidates.append(ob_sl)

    # 3. Recent swing low/high
    lookback = min(20, bar_idx + 1)
    if is_long:
        swing_sl = float(np.min(low_arr[bar_idx - lookback + 1 : bar_idx + 1])) - buffer
    else:
        swing_sl = float(np.max(high_arr[bar_idx - lookback + 1 : bar_idx + 1])) + buffer
    dist_swing = abs(price - swing_sl)
    if min_dist <= dist_swing <= max_dist:
        candidates.append(swing_sl)

    # 4. Fractal fallback
    if is_long:
        frac_sl = fractal_price - buffer
    else:
        frac_sl = fractal_price + buffer
    dist_frac = abs(price - frac_sl)
    if min_dist <= dist_frac <= max_dist:
        candidates.append(frac_sl)

    # Pick tightest
    if candidates:
        if is_long:
            return max(candidates)  # highest SL below entry
        return min(candidates)  # lowest SL above entry

    # Ultimate fallback: fractal ± buffer (ignore distance bounds)
    return frac_sl


# ---------------------------------------------------------------------------
# Sizing helpers
# ---------------------------------------------------------------------------

def _compute_base_qty(
    price: float,
    risk: RiskManager,
    score: SignalScore,
    leverage: int,
) -> float:
    """Compute qty for first entry in a new session."""
    base_usd = risk.compute_base_size_usd(score.position_scale)
    if base_usd <= 0 or price <= 0:
        return 0.0
    return base_usd * leverage / price


def _compute_pyramid_qty(
    sess: TrendSession,
    price: float,
    risk: RiskManager,
    cfg: BotConfig,
) -> float:
    """Compute qty for a pyramid addition.

    pyramid_usd = base + 50% × max(0, unrealized_pnl)
    Capped at 5 × base.
    """
    v2 = cfg.v2
    base_usd = risk.compute_base_size_usd()
    unrealized = max(0.0, sess.unrealized_pnl(price))
    pyramid_usd = base_usd + v2.pyramid_profit_fraction * unrealized
    pyramid_usd = min(pyramid_usd, v2.pyramid_max_base_mult * base_usd)
    if pyramid_usd <= 0 or price <= 0:
        return 0.0
    return pyramid_usd * cfg.leverage / price


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def _compute_entry_score(
    sig_dir: SignalDirection,
    bar_idx: int,
    ind,
    cfg: BotConfig,
    ts: int,
    htf_1h_map: dict,
    smc_1h: SMCResult | None,
    vol_arr: np.ndarray,
    close_arr: np.ndarray,
) -> SignalScore:
    """Extract indicators at *bar_idx* and delegate to score_signal."""
    tp_rsi = ind.tp_rsi
    kama = ind.kama

    rsi_val = float(tp_rsi["rsi"].iloc[bar_idx]) if not np.isnan(tp_rsi["rsi"].iloc[bar_idx]) else 50.0
    ribbon = int(tp_rsi["ribbon_score"].iloc[bar_idx])
    kama_bull = bool(kama["bullish"].iloc[bar_idx]) if not pd.isna(kama["bullish"].iloc[bar_idx]) else None
    bull_cross = bool(tp_rsi["bullish_cross"].iloc[bar_idx])
    bear_cross = bool(tp_rsi["bearish_cross"].iloc[bar_idx])

    adx_val = float(ind.adx.iloc[bar_idx]) if not np.isnan(ind.adx.iloc[bar_idx]) else 0.0
    dip_val = float(ind.di_plus.iloc[bar_idx]) if not np.isnan(ind.di_plus.iloc[bar_idx]) else 0.0
    dim_val = float(ind.di_minus.iloc[bar_idx]) if not np.isnan(ind.di_minus.iloc[bar_idx]) else 0.0
    ema200_val = float(ind.ema200.iloc[bar_idx]) if not np.isnan(ind.ema200.iloc[bar_idx]) else 0.0
    vol_ema_val = float(ind.vol_ema.iloc[bar_idx]) if not np.isnan(ind.vol_ema.iloc[bar_idx]) else 0.0

    # HTF alignment — use 1h regime bias as proxy (no separate KAMA for V2)
    h1_idx = htf_1h_map.get(ts)
    h1_bull: bool | None = None
    if smc_1h is not None and h1_idx is not None:
        bias = get_regime_at(smc_1h, h1_idx)
        if bias == Bias.BULLISH:
            h1_bull = True
        elif bias == Bias.BEARISH:
            h1_bull = False

    recent_divs = ind.div_lookup.get(bar_idx, [])

    return score_signal(
        sig_dir,
        aplus_fired=True,
        recent_divergences=recent_divs,
        ribbon_score=ribbon,
        adx_value=adx_val,
        di_plus=dip_val,
        di_minus=dim_val,
        kama_bullish=kama_bull,
        price=close_arr[bar_idx],
        ema200_value=ema200_val,
        rsi_value=rsi_val,
        htf_1h_kama_bullish=h1_bull,
        htf_4h_kama_bullish=None,
        rsi_bullish_cross=bull_cross,
        rsi_bearish_cross=bear_cross,
        volume=vol_arr[bar_idx],
        volume_ema20=vol_ema_val,
        weights=cfg.scoring,
    )


# ---------------------------------------------------------------------------
# Session lookup
# ---------------------------------------------------------------------------

def _find_session(
    sessions: list[TrendSession],
    direction: SignalDirection,
    symbol: str,
) -> TrendSession | None:
    """Return the active session matching *direction* + *symbol*, if any."""
    for s in sessions:
        if s.direction == direction and s.symbol == symbol:
            return s
    return None


# ---------------------------------------------------------------------------
# HTF map (reuse V1 pattern)
# ---------------------------------------------------------------------------

def _build_htf_map(ts_base: np.ndarray, klines_htf: list[dict]) -> dict[int, int]:
    """Map each base-TF timestamp to the latest HTF bar index."""
    if not klines_htf:
        return {}
    htf_ts = [k["timestamp"] for k in klines_htf]
    mapping: dict[int, int] = {}
    for base_ts in ts_base:
        idx = bisect.bisect_right(htf_ts, int(base_ts)) - 1
        if idx >= 0:
            mapping[int(base_ts)] = idx
    return mapping
