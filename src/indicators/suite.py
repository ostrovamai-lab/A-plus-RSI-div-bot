"""Shared indicator computation suite — computes all indicators on OHLCV data.

Eliminates duplication between backtest/engine.py and scanner/multi_coin_scanner.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_ta as ta

from config import BotConfig
from indicators.aplus_signal import compute_aplus_signals
from indicators.kama import compute_kama_full
from indicators.resampler import klines_to_df
from indicators.tp_rsi_v2 import build_divergence_lookup, compute_tp_rsi, detect_divergences
from models import Divergence


@dataclass
class IndicatorSuite:
    """All indicator outputs for a single timeframe."""
    aplus: dict
    tp_rsi: dict
    divergences: list[Divergence]
    div_lookup: dict[int, list[Divergence]]
    kama: dict
    atr: pd.Series
    vol_ema: pd.Series


def compute_indicator_suite(
    df_8m: pd.DataFrame,
    cfg: BotConfig,
    *,
    div_window: int = 5,
) -> IndicatorSuite:
    """Compute all indicators on 8-minute OHLCV data.

    Args:
        df_8m: DataFrame with columns: open, high, low, close, volume.
        cfg: Bot configuration with indicator parameters.
        div_window: Lookback window for divergence lookup table.

    Returns:
        IndicatorSuite with all computed indicators.
    """
    close = df_8m["close"]
    high = df_8m["high"]
    low = df_8m["low"]
    volume = df_8m["volume"]

    # A+ signals
    aplus = compute_aplus_signals(
        df_8m,
        pivot_lookback=cfg.aplus.pivot_lookback,
        fractal_len=cfg.aplus.fractal_len,
        step_window=cfg.aplus.step_window,
        gate_max_age=cfg.aplus.gate_max_age,
        ema_fast_len=cfg.aplus.ema_fast,
        ema_slow_len=cfg.aplus.ema_slow,
        break_mode=cfg.aplus.break_mode,
        buffer_ticks=cfg.aplus.buffer_ticks,
        need_retest=cfg.aplus.need_retest,
        retest_lookback=cfg.aplus.retest_lookback,
        allow_pre_cross=cfg.aplus.allow_pre_cross,
    )

    # TP RSI
    tp_rsi = compute_tp_rsi(
        close, high, low,
        rsi_period=cfg.tp_rsi.rsi_period,
        bb_period=cfg.tp_rsi.bb_period,
        bb_mult=cfg.tp_rsi.bb_mult,
        bb_sigma=cfg.tp_rsi.bb_sigma,
        ribbon_ma_type=cfg.tp_rsi.ribbon_ma_type,
        ribbon_pairs=cfg.tp_rsi.ribbon_pairs,
    )

    # Divergences
    divergences = detect_divergences(
        close, high, low, tp_rsi["rsi"],
        pivot_lookback=cfg.tp_rsi.divergence_lookback,
        max_range=cfg.tp_rsi.divergence_max_range,
    )
    div_lookup = build_divergence_lookup(divergences, window=div_window)

    # KAMA
    kama = compute_kama_full(
        close,
        er_length=cfg.kama.er_length,
        fast_length=cfg.kama.fast_length,
        slow_length=cfg.kama.slow_length,
        slope_bars=cfg.kama.slope_bars,
    )

    # ATR for grid sizing
    atr = ta.atr(high, low, close, length=14)
    if atr is None:
        atr = pd.Series(0.0, index=close.index)

    # Volume EMA
    vol_ema = ta.ema(volume, length=20)
    if vol_ema is None:
        vol_ema = pd.Series(0.0, index=volume.index)

    return IndicatorSuite(
        aplus=aplus,
        tp_rsi=tp_rsi,
        divergences=divergences,
        div_lookup=div_lookup,
        kama=kama,
        atr=atr,
        vol_ema=vol_ema,
    )


def compute_htf_kama(klines: list[dict], cfg: BotConfig) -> np.ndarray | None:
    """Compute KAMA bullish/bearish for higher-timeframe data.

    Used by both backtest engine and scanner for HTF alignment scoring.
    """
    df = klines_to_df(klines)
    if df.empty:
        return None
    kama_result = compute_kama_full(
        df["close"],
        er_length=cfg.kama.er_length,
        fast_length=cfg.kama.fast_length,
        slow_length=cfg.kama.slow_length,
        slope_bars=cfg.kama.slope_bars,
    )
    return kama_result["bullish"].values
