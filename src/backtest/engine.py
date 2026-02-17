"""Backtest engine — bar-by-bar simulation of the A+ RSI Divergence strategy.

Flow:
1. Pre-compute all indicators vectorized (fast)
2. Pre-compute HTF indicators
3. Build HTF timestamp mapping (bisect-based)
4. Bar-by-bar loop:
   a. Check SL hits (high/low)
   b. Check grid limit fills (high/low)
   c. Check A+ signals → score → enter or skip
   d. Check opposite signal → close positions
   e. Track equity
5. Generate report
"""

from __future__ import annotations

import bisect
import logging

import numpy as np
import pandas as pd

from config import BotConfig, load_config
from indicators.resampler import klines_to_df, resample_ohlcv
from indicators.suite import compute_htf_kama, compute_indicator_suite
from models import BacktestResult, SignalDirection, Trade
from scoring.signal_scorer import score_signal
from strategy.entry_engine import compute_entry_grid, compute_stop_loss
from strategy.exit_engine import check_opposite_signal, check_stop_loss, check_time_stop
from strategy.pnl import compute_fee, compute_pnl
from strategy.risk_manager import RiskManager

logger = logging.getLogger(__name__)


def run_backtest(
    klines_1m: list[dict],
    klines_1h: list[dict] | None = None,
    klines_4h: list[dict] | None = None,
    config: BotConfig | None = None,
    symbol: str = "BTCUSDT",
) -> BacktestResult:
    """Run full backtest on historical data.

    Args:
        klines_1m: 1-minute OHLCV bars (will be resampled to 8min).
        klines_1h: 1-hour bars for HTF confirmation (optional).
        klines_4h: 4-hour bars for HTF confirmation (optional).
        config: Bot configuration.
        symbol: Symbol being tested.

    Returns:
        BacktestResult with trades, equity curve, and statistics.
    """
    cfg = config or load_config()
    result = BacktestResult()

    # ── 1. Resample to 8min ──
    df_1m = klines_to_df(klines_1m)
    if df_1m.empty:
        logger.warning("Empty 1m data — cannot backtest")
        return result

    df_8m = resample_ohlcv(df_1m, interval_minutes=8)
    if len(df_8m) < 100:
        logger.warning("Not enough 8m bars (%d) — need at least 100", len(df_8m))
        return result

    close = df_8m["close"]
    high = df_8m["high"]
    low = df_8m["low"]
    volume = df_8m["volume"]
    n = len(df_8m)

    # ── 2. Pre-compute indicators ──
    ind = compute_indicator_suite(df_8m, cfg)
    aplus = ind.aplus
    tp_rsi = ind.tp_rsi
    div_lookup = ind.div_lookup
    kama = ind.kama
    atr = ind.atr
    atr_ema = ind.atr_ema
    vol_ema = ind.vol_ema
    adx = ind.adx
    di_plus = ind.di_plus
    di_minus = ind.di_minus
    ema200 = ind.ema200

    # ── 3. HTF indicators ──
    htf_1h_bullish = compute_htf_kama(klines_1h, cfg) if klines_1h else None
    htf_4h_bullish = compute_htf_kama(klines_4h, cfg) if klines_4h else None

    # Build timestamp mapping for HTF
    ts_8m = df_8m["timestamp"].values.astype(int)
    htf_1h_map = _build_htf_map(ts_8m, klines_1h) if klines_1h else {}
    htf_4h_map = _build_htf_map(ts_8m, klines_4h) if klines_4h else {}

    # ── 4. Simulation state ──
    risk = RiskManager(
        initial_capital=cfg.capital_usd,
        leverage=cfg.leverage,
        max_drawdown_pct=cfg.max_drawdown_pct,
        reinvest=cfg.reinvestment.enabled,
        min_scale=cfg.reinvestment.min_scale,
        max_scale=cfg.reinvestment.max_scale,
    )

    # Open positions: list of dicts tracking each position
    open_positions: list[dict] = []
    # Grid orders awaiting fill
    pending_grids: list[dict] = []
    # Monotonic position ID counter (avoids id() memory reuse bugs)
    _next_pos_id = 0

    close_arr = close.values.astype(float)
    high_arr = high.values.astype(float)
    low_arr = low.values.astype(float)
    vol_arr = volume.values.astype(float)

    # ── 5. Bar-by-bar loop ──
    start_bar = max(cfg.aplus.pivot_lookback + 10, 50)

    for i in range(start_bar, n):
        ts = int(ts_8m[i])
        price = close_arr[i]
        hi = high_arr[i]
        lo = low_arr[i]

        if risk.is_halted:
            result.equity_curve.append(risk.current_equity)
            result.timestamps.append(ts)
            continue

        # ── 5a. Check SL hits ──
        for pos in list(open_positions):
            if pos["sl_price"] > 0:
                sl_hit = check_stop_loss(
                    pos["direction"], pos["sl_price"], hi, lo,
                )
                if sl_hit:
                    trade = _close_position(pos, sl_hit.exit_price, i, "sl", cfg)
                    result.trades.append(trade)
                    risk.update_equity(trade.pnl - trade.fee)
                    open_positions.remove(pos)
                    # Cancel associated grid orders
                    pending_grids = [g for g in pending_grids if g.get("pos_id") != pos["pos_id"]]

        # ── 5b. Check grid limit fills ──
        for grid in list(pending_grids):
            filled = False
            if grid["side"] == "Buy" and lo <= grid["price"]:
                filled = True
            elif grid["side"] == "Sell" and hi >= grid["price"]:
                filled = True

            if filled:
                # Find associated position and add fill
                for pos in open_positions:
                    if pos["pos_id"] == grid.get("pos_id"):
                        pos["fills"].append((grid["price"], grid["qty"]))
                        pos["grid_fills"] += 1
                        break
                pending_grids.remove(grid)

        # ── 5c. Check A+ signals → score → enter ──
        aplus_long = bool(aplus["aplus_long"][i])
        aplus_short = bool(aplus["aplus_short"][i])

        for sig_dir, sig_fired in [
            (SignalDirection.LONG, aplus_long),
            (SignalDirection.SHORT, aplus_short),
        ]:
            if not sig_fired:
                continue

            fractal_p = float(aplus["fractal_price"][i])
            if np.isnan(fractal_p):
                continue

            # Check for opposite signal exit first
            for pos in list(open_positions):
                exit_sig = check_opposite_signal(
                    pos["direction"], aplus_long, aplus_short, price,
                )
                if exit_sig:
                    trade = _close_position(pos, exit_sig.exit_price, i, "opposite_signal", cfg)
                    result.trades.append(trade)
                    risk.update_equity(trade.pnl - trade.fee)
                    open_positions.remove(pos)
                    pending_grids = [g for g in pending_grids if g.get("pos_id") != pos["pos_id"]]

            # ── Extract filter indicators ──
            atr_val = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0
            atr_ema_val = float(atr_ema.iloc[i]) if not np.isnan(atr_ema.iloc[i]) else 0.0
            adx_val = float(adx.iloc[i]) if not np.isnan(adx.iloc[i]) else 0.0
            dip_val = float(di_plus.iloc[i]) if not np.isnan(di_plus.iloc[i]) else 0.0
            dim_val = float(di_minus.iloc[i]) if not np.isnan(di_minus.iloc[i]) else 0.0
            ema200_val = float(ema200.iloc[i]) if not np.isnan(ema200.iloc[i]) else 0.0

            # ── Score the signal ──
            rsi_val = float(tp_rsi["rsi"].iloc[i]) if not np.isnan(tp_rsi["rsi"].iloc[i]) else 50.0
            ribbon = int(tp_rsi["ribbon_score"].iloc[i])
            kama_bull = bool(kama["bullish"].iloc[i]) if not pd.isna(kama["bullish"].iloc[i]) else None
            bull_cross = bool(tp_rsi["bullish_cross"].iloc[i])
            bear_cross = bool(tp_rsi["bearish_cross"].iloc[i])

            # HTF
            h1_idx = htf_1h_map.get(ts)
            h4_idx = htf_4h_map.get(ts)
            h1_bull = bool(htf_1h_bullish[h1_idx]) if (htf_1h_bullish is not None and h1_idx is not None
                                                        and not np.isnan(htf_1h_bullish[h1_idx])) else None
            h4_bull = bool(htf_4h_bullish[h4_idx]) if (htf_4h_bullish is not None and h4_idx is not None
                                                        and not np.isnan(htf_4h_bullish[h4_idx])) else None

            recent_divs = div_lookup.get(i, [])

            score = score_signal(
                sig_dir,
                aplus_fired=True,
                recent_divergences=recent_divs,
                ribbon_score=ribbon,
                adx_value=adx_val,
                di_plus=dip_val,
                di_minus=dim_val,
                kama_bullish=kama_bull,
                price=price,
                ema200_value=ema200_val,
                rsi_value=rsi_val,
                htf_1h_kama_bullish=h1_bull,
                htf_4h_kama_bullish=h4_bull,
                rsi_bullish_cross=bull_cross,
                rsi_bearish_cross=bear_cross,
                volume=vol_arr[i],
                volume_ema20=float(vol_ema.iloc[i]) if not np.isnan(vol_ema.iloc[i]) else 0.0,
                weights=cfg.scoring,
            )

            if score.total < cfg.min_score:
                continue

            # Check position limits
            if len(open_positions) >= cfg.max_positions:
                continue

            # Compute entry grid
            grid_levels = compute_entry_grid(
                sig_dir, price, fractal_p, atr_val, score,
                risk.compute_base_size_usd(score.position_scale),
                cfg.leverage, cfg.entry,
            )

            if not grid_levels:
                continue

            sl = compute_stop_loss(
                sig_dir, fractal_p, cfg.entry.sl_buffer_pct,
                atr_value=atr_val,
                sl_atr_multiplier=cfg.filters.sl_atr_multiplier,
            )

            # Open position (first grid level fills immediately at signal price)
            _next_pos_id += 1
            pos = {
                "pos_id": _next_pos_id,
                "direction": sig_dir,
                "symbol": symbol,
                "open_bar": i,
                "sl_price": sl,
                "fills": [(price, float(grid_levels[0].qty))],
                "grid_fills": 1,
                "score": score.total,
                "pyramid_level": 0,
            }
            open_positions.append(pos)

            # Remaining grid levels as pending orders
            for gl in grid_levels[1:]:
                pending_grids.append({
                    "pos_id": pos["pos_id"],
                    "price": gl.price,
                    "side": gl.side,
                    "qty": float(gl.qty),
                })

        # ── 5d. Check time stops ──
        for pos in list(open_positions):
            bars_held = i - pos["open_bar"]
            ts_exit = check_time_stop(bars_held, cfg.exit.time_stop_bars, price)
            if ts_exit:
                trade = _close_position(pos, ts_exit.exit_price, i, "time_stop", cfg)
                result.trades.append(trade)
                risk.update_equity(trade.pnl - trade.fee)
                open_positions.remove(pos)
                pending_grids = [g for g in pending_grids if g.get("pos_id") != pos["pos_id"]]

        # ── 5e. Drawdown check ──
        if risk.check_drawdown():
            for pos in list(open_positions):
                trade = _close_position(pos, price, i, "drawdown", cfg)
                result.trades.append(trade)
                risk.update_equity(trade.pnl - trade.fee)
            open_positions.clear()
            pending_grids.clear()

        # Track equity
        unrealized = 0.0
        for pos in open_positions:
            unrealized += compute_pnl(
                pos["direction"], _avg_entry(pos), price,
                _total_qty(pos), cfg.leverage,
            )

        current_eq = risk.current_equity + unrealized
        result.equity_curve.append(current_eq)
        result.timestamps.append(ts)

    # ── 6. Close remaining positions ──
    final_price = close_arr[-1]
    for pos in open_positions:
        trade = _close_position(pos, final_price, n - 1, "backtest_end", cfg)
        result.trades.append(trade)
        risk.update_equity(trade.pnl - trade.fee)

    # ── 7. Compute stats ──
    _compute_stats(result)

    return result


def _close_position(pos: dict, exit_price: float, bar_index: int, reason: str, cfg: BotConfig) -> Trade:
    """Close a position dict and return a Trade."""
    avg = _avg_entry(pos)
    qty = _total_qty(pos)
    direction = pos["direction"]

    pnl = compute_pnl(direction, avg, exit_price, qty, cfg.leverage)
    fee = compute_fee(avg, qty, cfg.leverage, reason, cfg.exit.maker_fee, cfg.exit.taker_fee)

    return Trade(
        open_time=pos["open_bar"],
        close_time=bar_index,
        symbol=pos.get("symbol", ""),
        side="Buy" if direction == SignalDirection.LONG else "Sell",
        entry_price=avg,
        exit_price=exit_price,
        qty=qty,
        pnl=pnl,
        fee=fee,
        reason=reason,
        score=pos.get("score", 0),
        grid_fills=pos.get("grid_fills", 0),
        pyramid_level=pos.get("pyramid_level", 0),
    )


def _avg_entry(pos: dict) -> float:
    fills = pos["fills"]
    total_qty = sum(q for _, q in fills)
    if total_qty == 0:
        return 0.0
    return sum(p * q for p, q in fills) / total_qty


def _total_qty(pos: dict) -> float:
    return sum(q for _, q in pos["fills"])



def _build_htf_map(ts_base: np.ndarray, klines_htf: list[dict]) -> dict[int, int]:
    """Map each base-TF timestamp to the latest HTF bar index."""
    if not klines_htf:
        return {}
    htf_ts = [k["timestamp"] for k in klines_htf]
    mapping = {}
    for base_ts in ts_base:
        idx = bisect.bisect_right(htf_ts, int(base_ts)) - 1
        if idx >= 0:
            mapping[int(base_ts)] = idx
    return mapping


def _compute_stats(result: BacktestResult) -> None:
    """Compute summary statistics on the result."""
    trades = result.trades
    result.total_trades = len(trades)
    result.total_pnl = sum(t.pnl for t in trades)
    result.total_fees = sum(t.fee for t in trades)
    result.net_pnl = result.total_pnl - result.total_fees

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    result.winning_trades = len(wins)
    result.losing_trades = len(losses)
    result.win_rate = len(wins) / max(len(trades), 1) * 100

    result.avg_win = sum(t.pnl for t in wins) / max(len(wins), 1)
    result.avg_loss = sum(t.pnl for t in losses) / max(len(losses), 1)

    gross_wins = sum(t.pnl for t in wins)
    gross_losses = abs(sum(t.pnl for t in losses))
    result.profit_factor = gross_wins / max(gross_losses, 0.01)

    # Max drawdown
    if result.equity_curve:
        peak = result.equity_curve[0]
        max_dd = 0.0
        for eq in result.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / max(peak, 1) * 100
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown_pct = max_dd

    # Sharpe ratio
    if len(result.equity_curve) > 1:
        returns = []
        step = max(1, len(result.equity_curve) // 100)
        for j in range(step, len(result.equity_curve), step):
            prev = result.equity_curve[j - step]
            curr = result.equity_curve[j]
            if prev != 0:
                returns.append((curr - prev) / abs(prev))
        if returns:
            avg_r = sum(returns) / len(returns)
            std_r = (sum((r - avg_r) ** 2 for r in returns) / max(len(returns) - 1, 1)) ** 0.5
            result.sharpe_ratio = (avg_r / std_r * (365 ** 0.5)) if std_r > 0 else 0.0  # crypto: 365 days
