"""Live trading bot — async orchestrator for the A+ RSI Divergence strategy.

Loop (every 8 minutes, on candle close):
  for each symbol in scan_list:
    1. Fetch 1min klines → resample 8min
    2. Run A+ indicator
    3. If A+ signal → score → if passes → place grid
    4. If opposite signal → close position
    5. Check existing grid fills via position API
    6. Update equity, log
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal

from config import BotConfig, load_config, BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_DEMO
from connectors.bybit_connector import BybitConnector
from models import SignalDirection
from scanner.multi_coin_scanner import MultiCoinScanner
from strategy.entry_engine import compute_entry_grid, compute_stop_loss
from strategy.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class LiveBot:
    """Live trading bot for A+ RSI Divergence strategy."""

    def __init__(self, config: BotConfig | None = None):
        self.cfg = config or load_config()
        self.connector = BybitConnector(
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
            demo=BYBIT_DEMO,
        )
        self.scanner = MultiCoinScanner(self.connector, self.cfg)
        self.risk = RiskManager(
            initial_capital=self.cfg.capital_usd,
            leverage=self.cfg.leverage,
            max_drawdown_pct=self.cfg.max_drawdown_pct,
            reinvest=self.cfg.reinvestment.enabled,
            min_scale=self.cfg.reinvestment.min_scale,
            max_scale=self.cfg.reinvestment.max_scale,
        )
        self._running = False
        self._active_symbols: set[str] = set()  # symbols with open positions

    async def start(self) -> None:
        """Start the live trading loop."""
        logger.info("Starting A+ RSI Divergence Bot")
        logger.info("Symbols: %s", self.cfg.symbols)
        logger.info("Capital: $%.2f, Leverage: %dx", self.cfg.capital_usd, self.cfg.leverage)

        # Set leverage for all symbols
        for symbol in self.cfg.symbols:
            try:
                await self.connector.set_leverage(symbol, self.cfg.leverage)
            except Exception as exc:
                logger.warning("Failed to set leverage for %s: %s", symbol, exc)

        # Update initial equity from exchange
        try:
            balance = await self.connector.get_wallet_balance()
            self.risk.current_equity = float(balance)
            self.risk.initial_capital = float(balance)
            self.risk.peak_equity = float(balance)
            logger.info("Wallet balance: $%.2f", balance)
        except Exception as exc:
            logger.warning("Failed to get balance: %s", exc)

        self._running = True
        poll_interval = self.cfg.scanner.poll_interval_sec

        while self._running:
            try:
                await self._scan_cycle()
            except Exception:
                logger.exception("Error in scan cycle")

            # Wait for next 8min candle
            logger.info("Sleeping %ds until next scan...", poll_interval)
            await asyncio.sleep(poll_interval)

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        self._running = False
        logger.info("Bot stopped")

    async def _scan_cycle(self) -> None:
        """Run one scan cycle."""
        if self.risk.is_halted:
            logger.warning("Trading halted due to drawdown limit")
            return

        # Check existing positions for exit signals
        positions = await self.connector.get_positions()
        self._active_symbols = {p.symbol for p in positions}

        # Scan for new signals
        results = await self.scanner.scan_all()

        for scan in results:
            if scan.score < self.cfg.min_score:
                continue

            if not scan.direction or not scan.has_signal:
                continue

            symbol = scan.symbol

            # Check if we already have a position on this symbol
            if symbol in self._active_symbols:
                logger.info("Skipping %s — already have position", symbol)
                continue

            # Check position limits
            if len(self._active_symbols) >= self.cfg.max_positions:
                logger.info("Max positions reached (%d), skipping %s",
                            self.cfg.max_positions, symbol)
                break

            # Place entry grid
            try:
                await self._enter_position(scan)
                self._active_symbols.add(symbol)
            except Exception:
                logger.exception("Failed to enter %s", symbol)

    async def _enter_position(self, scan) -> None:
        """Place entry grid orders for a signal."""
        from models import SignalScore

        cfg = self.cfg
        symbol = scan.symbol
        direction = scan.direction

        # Build a minimal score object
        score = SignalScore(total=scan.score)

        # Compute grid
        grid_levels = compute_entry_grid(
            direction, scan.signal_price, scan.fractal_price,
            scan.atr_value, score,
            self.risk.compute_base_size_usd(score.position_scale),
            cfg.leverage, cfg.entry,
        )

        if not grid_levels:
            logger.warning("No grid levels computed for %s", symbol)
            return

        # Compute SL
        sl_price = compute_stop_loss(direction, scan.fractal_price, cfg.entry.sl_buffer_pct)

        # Place first order as market (immediate entry)
        side = "Buy" if direction == SignalDirection.LONG else "Sell"
        first_level = grid_levels[0]

        logger.info(
            "Entering %s %s: price=%.4f, fractal=%.4f, score=%.1f, SL=%.4f",
            symbol, direction.value, scan.signal_price, scan.fractal_price,
            scan.score, sl_price,
        )

        await self.connector.place_market_order(
            symbol=symbol,
            side=side,
            qty=first_level.qty,
            sl=Decimal(str(round(sl_price, 8))),
        )

        # Place remaining levels as limit orders
        for level in grid_levels[1:]:
            try:
                await self.connector.place_limit_order(
                    symbol=symbol,
                    side=level.side,
                    qty=level.qty,
                    price=Decimal(str(round(level.price, 8))),
                )
            except Exception:
                logger.warning("Failed to place grid level %d for %s",
                               level.level_index, symbol, exc_info=True)

        logger.info("Grid placed for %s: %d levels", symbol, len(grid_levels))
