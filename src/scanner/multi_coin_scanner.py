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

from config import BotConfig
from connectors.base import ExchangeConnector
from indicators.resampler import klines_to_df, resample_ohlcv
from indicators.suite import compute_htf_kama, compute_indicator_suite
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
        volume = df_8m["volume"]

        # Compute all indicators via shared suite
        ind = compute_indicator_suite(df_8m, cfg)

        # Check last bar for signal
        last = len(df_8m) - 1
        has_long = bool(ind.aplus["aplus_long"][last])
        has_short = bool(ind.aplus["aplus_short"][last])

        if not has_long and not has_short:
            return result

        # Signal found
        direction = SignalDirection.LONG if has_long else SignalDirection.SHORT
        fractal_p = float(ind.aplus["fractal_price"][last])

        recent_divs = ind.div_lookup.get(last, [])

        atr_val = float(ind.atr.iloc[last]) if not np.isnan(ind.atr.iloc[last]) else 0.0
        atr_ema_val = float(ind.atr_ema.iloc[last]) if not np.isnan(ind.atr_ema.iloc[last]) else 0.0
        vol_ema_val = float(ind.vol_ema.iloc[last]) if not np.isnan(ind.vol_ema.iloc[last]) else 0.0
        rsi_val = float(ind.tp_rsi["rsi"].iloc[last]) if not np.isnan(ind.tp_rsi["rsi"].iloc[last]) else 50.0
        ribbon = int(ind.tp_rsi["ribbon_score"].iloc[last])
        kama_bull = bool(ind.kama["bullish"].iloc[last]) if not pd.isna(ind.kama["bullish"].iloc[last]) else None
        adx_val = float(ind.adx.iloc[last]) if not np.isnan(ind.adx.iloc[last]) else 0.0
        di_plus_val = float(ind.di_plus.iloc[last]) if not np.isnan(ind.di_plus.iloc[last]) else 0.0
        di_minus_val = float(ind.di_minus.iloc[last]) if not np.isnan(ind.di_minus.iloc[last]) else 0.0
        ema200_val = float(ind.ema200.iloc[last]) if not np.isnan(ind.ema200.iloc[last]) else 0.0
        price = float(close.iloc[last])

        # Fetch HTF data for scoring
        htf_1h_bull = None
        htf_4h_bull = None
        try:
            klines_1h = await self._connector.get_klines(symbol, "60", limit=100)
            if klines_1h:
                htf_arr = compute_htf_kama(klines_1h, cfg)
                if htf_arr is not None and not np.isnan(htf_arr[-1]):
                    htf_1h_bull = bool(htf_arr[-1])
        except Exception:
            logger.debug("Failed to fetch 1h data for %s", symbol)

        try:
            klines_4h = await self._connector.get_klines(symbol, "240", limit=100)
            if klines_4h:
                htf_arr = compute_htf_kama(klines_4h, cfg)
                if htf_arr is not None and not np.isnan(htf_arr[-1]):
                    htf_4h_bull = bool(htf_arr[-1])
        except Exception:
            logger.debug("Failed to fetch 4h data for %s", symbol)

        score = score_signal(
            direction,
            aplus_fired=True,
            recent_divergences=recent_divs,
            ribbon_score=ribbon,
            adx_value=adx_val,
            di_plus=di_plus_val,
            di_minus=di_minus_val,
            kama_bullish=kama_bull,
            price=price,
            ema200_value=ema200_val,
            rsi_value=rsi_val,
            htf_1h_kama_bullish=htf_1h_bull,
            htf_4h_kama_bullish=htf_4h_bull,
            rsi_bullish_cross=bool(ind.tp_rsi["bullish_cross"].iloc[last]),
            rsi_bearish_cross=bool(ind.tp_rsi["bearish_cross"].iloc[last]),
            volume=float(volume.iloc[last]),
            volume_ema20=vol_ema_val,
            weights=cfg.scoring,
        )

        result.has_signal = True
        result.direction = direction
        result.score = score.total
        result.fractal_price = fractal_p
        result.signal_price = price
        result.atr_value = atr_val
        result.rsi_value = rsi_val
        result.ribbon_score = ribbon
        result.kama_bullish = kama_bull

        logger.info(
            "A+ signal on %s: %s score=%.1f (RSI=%.1f, ribbon=%d)",
            symbol, direction.value, score.total, rsi_val, ribbon,
        )

        return result
