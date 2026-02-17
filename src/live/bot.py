"""Live trading bot — async orchestrator for the A+ RSI Divergence strategy.

Loop (every 8 minutes, on candle close):
  for each symbol in scan_list:
    1. Fetch 1min klines → resample 8min
    2. Run A+ indicator
    3. If A+ signal fires on a symbol with an open position in the opposite
       direction → close that position (opposite-signal exit)
    4. If A+ signal → score → if passes → place grid
    5. Update equity from exchange balance
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal

from config import BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_DEMO, BotConfig, load_config
from connectors.bybit_connector import BybitConnector
from models import Position, Side, SignalDirection
from scanner.multi_coin_scanner import MultiCoinScanner, ScanResult
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
        self._stop_event = asyncio.Event()
        self._active_positions: dict[str, Position] = {}  # symbol → Position

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
        await self._sync_equity()

        self._stop_event.clear()
        poll_interval = self.cfg.scanner.poll_interval_sec

        while not self._stop_event.is_set():
            try:
                await self._scan_cycle()
            except Exception:
                logger.exception("Error in scan cycle")

            # Wait for next candle — interruptible via stop_event
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=poll_interval,
                )
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass  # Normal: timeout means it's time for next cycle

        logger.info("Bot stopped")

    async def stop(self) -> None:
        """Signal the bot to stop gracefully after the current cycle."""
        self._stop_event.set()

    # ──────────────────────────────────────────────

    async def _sync_equity(self) -> None:
        """Sync equity from exchange wallet balance."""
        try:
            balance = await self.connector.get_wallet_balance()
            bal = float(balance)
            if bal > 0:
                self.risk.current_equity = bal
                if self.risk.initial_capital == 0:
                    self.risk.initial_capital = bal
                self.risk.peak_equity = max(self.risk.peak_equity, bal)
                logger.info("Wallet balance: $%.2f", bal)
        except Exception as exc:
            logger.warning("Failed to get balance: %s", exc)

    async def _sync_positions(self) -> None:
        """Fetch open positions from exchange and update local state."""
        positions = await self.connector.get_positions()
        self._active_positions = {p.symbol: p for p in positions}

    async def _scan_cycle(self) -> None:
        """Run one scan cycle: check exits, then enter new positions."""
        # 1. Sync state from exchange
        await self._sync_positions()
        await self._sync_equity()

        # 2. Check drawdown halt
        if self.risk.check_drawdown():
            logger.warning("Trading HALTED — drawdown limit reached (%.1f%%)",
                           self.risk.drawdown_pct())
            await self._close_all_positions("drawdown")
            return

        if self.risk.is_halted:
            logger.warning("Trading halted due to drawdown limit")
            return

        # 3. Scan for signals
        results = await self.scanner.scan_all()

        # 4. Process exits first (opposite-signal close)
        for scan in results:
            if not scan.has_signal or not scan.direction:
                continue
            await self._check_opposite_exit(scan)

        # 5. Re-sync positions after any exits
        if results:
            await self._sync_positions()

        # 6. Process entries
        for scan in results:
            if scan.score < self.cfg.min_score:
                continue
            if not scan.direction or not scan.has_signal:
                continue

            symbol = scan.symbol

            # Skip if we already have a position on this symbol
            if symbol in self._active_positions:
                logger.debug("Skipping %s — already have position", symbol)
                continue

            # Check position limits
            if len(self._active_positions) >= self.cfg.max_positions:
                logger.info("Max positions reached (%d), stopping entries",
                            self.cfg.max_positions)
                break

            # Place entry grid
            try:
                await self._enter_position(scan)
            except Exception:
                logger.exception("Failed to enter %s", symbol)

    async def _check_opposite_exit(self, scan: ScanResult) -> None:
        """Close an existing position if an opposite A+ signal fires."""
        symbol = scan.symbol
        pos = self._active_positions.get(symbol)
        if pos is None:
            return

        # Determine if signal is opposite to existing position
        is_opposite = (
            (pos.side == Side.LONG and scan.direction == SignalDirection.SHORT)
            or (pos.side == Side.SHORT and scan.direction == SignalDirection.LONG)
        )
        if not is_opposite:
            return

        logger.info(
            "Opposite signal on %s: closing %s position (new signal: %s, score=%.1f)",
            symbol, pos.side.value, scan.direction.value, scan.score,
        )

        try:
            # Cancel any open grid orders
            await self.connector.cancel_all_orders(symbol)

            # Close position with market order
            close_side = pos.side.opposite_bybit
            await self.connector.place_market_order(
                symbol=symbol,
                side=close_side,
                qty=pos.size,
            )

            # Update equity
            await self._sync_equity()
            del self._active_positions[symbol]
            logger.info("Closed %s position on %s", pos.side.value, symbol)
        except Exception:
            logger.exception("Failed to close %s on opposite signal", symbol)

    async def _close_all_positions(self, reason: str) -> None:
        """Close all positions (drawdown halt)."""
        for symbol, pos in list(self._active_positions.items()):
            try:
                await self.connector.cancel_all_orders(symbol)
                close_side = pos.side.opposite_bybit
                await self.connector.place_market_order(
                    symbol=symbol,
                    side=close_side,
                    qty=pos.size,
                )
                logger.info("Closed %s %s (reason: %s)", symbol, pos.side.value, reason)
            except Exception:
                logger.exception("Failed to close %s during %s", symbol, reason)
        self._active_positions.clear()
        await self._sync_equity()

    async def _enter_position(self, scan: ScanResult) -> None:
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

        # Track locally
        self._active_positions[symbol] = Position(
            symbol=symbol,
            side=Side.LONG if direction == SignalDirection.LONG else Side.SHORT,
            size=first_level.qty,
            entry_price=Decimal(str(scan.signal_price)),
            leverage=cfg.leverage,
        )

        logger.info("Grid placed for %s: %d levels", symbol, len(grid_levels))
