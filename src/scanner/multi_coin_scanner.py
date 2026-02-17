"""Multi-coin scanner — async parallel scan of 20+ symbols for A+ signals.

Architecture:
- Scan interval: Every 8 minutes (aligned with candle close)
- Fetch 600 × 1min bars per symbol → resample to 75 × 8min bars
- Run A+ state machine on each; only compute full indicator suite if A+ signal fires
- Rate limiting: Bybit 120 req/min → batch 4 symbols with 0.5s gaps
- Rank signals by score, enter top N
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_ta as ta

from config import BotConfig
from connectors.base import ExchangeConnector
from indicators.aplus_signal import compute_aplus_signals
from indicators.kama import compute_kama_full
from indicators.resampler import klines_to_df, resample_ohlcv
from indicators.tp_rsi_v2 import compute_tp_rsi, detect_divergences
from models import SignalDirection
from scoring.signal_scorer import score_signal

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result of scanning a single symbol."""
    symbol: str
    direction: SignalDirection | None = None
    score: float = 0.0
    fractal_price: float = 0.0
    signal_price: float = 0.0
    atr_value: float = 0.0
    rsi_value: float = 50.0
    ribbon_score: int = 0
    kama_bullish: bool | None = None
    has_signal: bool = False


class MultiCoinScanner:
    """Scans multiple symbols for A+ signals."""

    def __init__(
        self,
        connector: ExchangeConnector,
        config: BotConfig,
    ):
        self._connector = connector
        self._cfg = config
        self._symbols = config.symbols

    async def scan_all(self) -> list[ScanResult]:
        """Scan all configured symbols. Returns results sorted by score (descending)."""
        results: list[ScanResult] = []
        batch_size = self._cfg.scanner.batch_size
        delay = self._cfg.scanner.batch_delay_sec

        for batch_start in range(0, len(self._symbols), batch_size):
            batch = self._symbols[batch_start:batch_start + batch_size]
            tasks = [self._scan_symbol(sym) for sym in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for sym, res in zip(batch, batch_results):
                if isinstance(res, Exception):
                    logger.warning("Scan failed for %s: %s", sym, res)
                    continue
                if res.has_signal:
                    results.append(res)

            if batch_start + batch_size < len(self._symbols):
                await asyncio.sleep(delay)

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    async def _scan_symbol(self, symbol: str) -> ScanResult:
        """Scan a single symbol for A+ signals."""
        result = ScanResult(symbol=symbol)
        cfg = self._cfg

        # Fetch 1min klines
        klines_1m = await self._connector.get_klines(
            symbol, "1", limit=cfg.scanner.history_bars_1m,
        )
        if len(klines_1m) < 100:
            return result

        # Resample to 8min
        df_1m = klines_to_df(klines_1m)
        df_8m = resample_ohlcv(df_1m, interval_minutes=8)
        if len(df_8m) < 50:
            return result

        close = df_8m["close"]
        high = df_8m["high"]
        low = df_8m["low"]
        volume = df_8m["volume"]

        # Run A+ state machine (fast — just checks for signals)
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

        # Check last bar for signal
        last = len(df_8m) - 1
        has_long = bool(aplus["aplus_long"][last])
        has_short = bool(aplus["aplus_short"][last])

        if not has_long and not has_short:
            return result

        # Signal found — compute full indicator suite
        direction = SignalDirection.LONG if has_long else SignalDirection.SHORT
        fractal_p = float(aplus["fractal_price"][last])

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
        recent_divs = [d for d in divergences if abs(d.bar_index - last) <= 5]

        # KAMA
        kama = compute_kama_full(
            close,
            er_length=cfg.kama.er_length,
            fast_length=cfg.kama.fast_length,
            slow_length=cfg.kama.slow_length,
            slope_bars=cfg.kama.slope_bars,
        )

        # ATR
        atr = ta.atr(high, low, close, length=14)
        atr_val = float(atr.iloc[last]) if atr is not None and not np.isnan(atr.iloc[last]) else 0.0

        # Volume
        vol_ema = ta.ema(volume, length=20)
        vol_ema_val = float(vol_ema.iloc[last]) if vol_ema is not None and not np.isnan(vol_ema.iloc[last]) else 0.0

        rsi_val = float(tp_rsi["rsi"].iloc[last]) if not np.isnan(tp_rsi["rsi"].iloc[last]) else 50.0
        ribbon = int(tp_rsi["ribbon_score"].iloc[last])
        kama_bull = bool(kama["bullish"].iloc[last]) if not pd.isna(kama["bullish"].iloc[last]) else None

        # Fetch HTF data for scoring
        htf_1h_bull = None
        htf_4h_bull = None
        try:
            klines_1h = await self._connector.get_klines(symbol, "60", limit=100)
            if klines_1h:
                df_1h = klines_to_df(klines_1h)
                kama_1h = compute_kama_full(df_1h["close"], cfg.kama.er_length,
                                            cfg.kama.fast_length, cfg.kama.slow_length, cfg.kama.slope_bars)
                htf_1h_bull = bool(kama_1h["bullish"].iloc[-1]) if not pd.isna(kama_1h["bullish"].iloc[-1]) else None
        except Exception:
            logger.debug("Failed to fetch 1h data for %s", symbol)

        try:
            klines_4h = await self._connector.get_klines(symbol, "240", limit=100)
            if klines_4h:
                df_4h = klines_to_df(klines_4h)
                kama_4h = compute_kama_full(df_4h["close"], cfg.kama.er_length,
                                            cfg.kama.fast_length, cfg.kama.slow_length, cfg.kama.slope_bars)
                htf_4h_bull = bool(kama_4h["bullish"].iloc[-1]) if not pd.isna(kama_4h["bullish"].iloc[-1]) else None
        except Exception:
            logger.debug("Failed to fetch 4h data for %s", symbol)

        score = score_signal(
            direction,
            aplus_fired=True,
            recent_divergences=recent_divs,
            ribbon_score=ribbon,
            kama_bullish=kama_bull,
            rsi_value=rsi_val,
            htf_1h_kama_bullish=htf_1h_bull,
            htf_4h_kama_bullish=htf_4h_bull,
            rsi_bullish_cross=bool(tp_rsi["bullish_cross"].iloc[last]),
            rsi_bearish_cross=bool(tp_rsi["bearish_cross"].iloc[last]),
            volume=float(volume.iloc[last]),
            volume_ema20=vol_ema_val,
            weights=cfg.scoring,
        )

        result.has_signal = True
        result.direction = direction
        result.score = score.total
        result.fractal_price = fractal_p
        result.signal_price = float(close.iloc[last])
        result.atr_value = atr_val
        result.rsi_value = rsi_val
        result.ribbon_score = ribbon
        result.kama_bullish = kama_bull

        logger.info(
            "A+ signal on %s: %s score=%.1f (RSI=%.1f, ribbon=%d)",
            symbol, direction.value, score.total, rsi_val, ribbon,
        )

        return result
