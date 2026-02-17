"""Position manager — tracks open positions, handles pyramiding.

Pyramiding rules:
- Max 3 same-direction positions per symbol
- New A+ signal in same direction while position profitable
- Pyramid requires score >= 70
- Size scaling: 100%, 75%, 50% for 1st, 2nd, 3rd pyramid
- Move existing SL to breakeven when adding pyramid
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models import SignalDirection, Trade
from strategy.pnl import compute_fee, compute_pnl


@dataclass
class ManagedPosition:
    """A position being tracked by the position manager."""
    symbol: str
    direction: SignalDirection
    entries: list[PositionEntry] = field(default_factory=list)
    sl_price: float = 0.0
    open_bar: int = 0           # bar index when first opened
    pyramid_count: int = 0      # 0 = initial, 1-2 = pyramids
    peak_equity: float = 0.0    # highest unrealized equity for this position

    @property
    def avg_entry_price(self) -> float:
        total_qty = sum(e.qty for e in self.entries)
        if total_qty == 0:
            return 0.0
        return sum(e.price * e.qty for e in self.entries) / total_qty

    @property
    def total_qty(self) -> float:
        return sum(e.qty for e in self.entries)

    @property
    def total_cost(self) -> float:
        return sum(e.price * e.qty for e in self.entries)

    def unrealized_pnl(self, current_price: float, leverage: int = 1) -> float:
        return compute_pnl(
            self.direction, self.avg_entry_price, current_price,
            self.total_qty, leverage,
        )

    def is_profitable(self, current_price: float) -> bool:
        return self.unrealized_pnl(current_price) > 0


@dataclass
class PositionEntry:
    """Single entry within a managed position (initial or pyramid)."""
    price: float
    qty: float
    bar_index: int
    pyramid_level: int = 0     # 0 = initial, 1-2 = pyramid
    grid_fills: int = 0        # how many grid levels actually filled


class PositionManager:
    """Manages all open positions across symbols."""

    def __init__(
        self,
        max_positions: int = 5,
        max_per_coin: int = 3,
        pyramid_size_scaling: list[float] | None = None,
        move_sl_to_be: bool = True,
    ):
        self._max_positions = max_positions
        self._max_per_coin = max_per_coin
        self._pyramid_scaling = pyramid_size_scaling or [1.0, 0.75, 0.5]
        self._move_sl_to_be = move_sl_to_be
        self.positions: dict[str, list[ManagedPosition]] = {}

    @property
    def total_positions(self) -> int:
        return sum(len(v) for v in self.positions.values())

    def can_open(self, symbol: str) -> bool:
        """Check if we can open a new position on this symbol."""
        if self.total_positions >= self._max_positions:
            return False
        if len(self.positions.get(symbol, [])) >= self._max_per_coin:
            return False
        return True

    def can_pyramid(self, symbol: str, direction: SignalDirection) -> bool:
        """Check if we can add a pyramid to an existing position."""
        for pos in self.positions.get(symbol, []):
            if pos.direction == direction and pos.pyramid_count < len(self._pyramid_scaling) - 1:
                return True
        return False

    def get_pyramid_scale(self, pyramid_level: int) -> float:
        """Get position size scaling factor for a pyramid level."""
        if pyramid_level < len(self._pyramid_scaling):
            return self._pyramid_scaling[pyramid_level]
        return self._pyramid_scaling[-1]

    def open_position(
        self,
        symbol: str,
        direction: SignalDirection,
        entry_price: float,
        qty: float,
        sl_price: float,
        bar_index: int,
    ) -> ManagedPosition:
        """Open a new position."""
        entry = PositionEntry(price=entry_price, qty=qty, bar_index=bar_index)
        pos = ManagedPosition(
            symbol=symbol,
            direction=direction,
            entries=[entry],
            sl_price=sl_price,
            open_bar=bar_index,
        )
        self.positions.setdefault(symbol, []).append(pos)
        return pos

    def add_pyramid(
        self,
        symbol: str,
        direction: SignalDirection,
        entry_price: float,
        qty: float,
        bar_index: int,
    ) -> ManagedPosition | None:
        """Add a pyramid entry to an existing position."""
        for pos in self.positions.get(symbol, []):
            if pos.direction == direction and pos.pyramid_count < len(self._pyramid_scaling) - 1:
                pos.pyramid_count += 1
                entry = PositionEntry(
                    price=entry_price, qty=qty,
                    bar_index=bar_index, pyramid_level=pos.pyramid_count,
                )
                pos.entries.append(entry)

                # Move SL to breakeven
                if self._move_sl_to_be:
                    pos.sl_price = pos.avg_entry_price

                return pos
        return None

    def close_position(
        self,
        symbol: str,
        direction: SignalDirection,
        exit_price: float,
        bar_index: int,
        reason: str,
        leverage: int = 1,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.00055,
    ) -> Trade | None:
        """Close a position and return the completed Trade."""
        positions = self.positions.get(symbol, [])
        for i, pos in enumerate(positions):
            if pos.direction == direction:
                qty = pos.total_qty
                avg_entry = pos.avg_entry_price

                pnl = compute_pnl(direction, avg_entry, exit_price, qty, leverage)
                fee = compute_fee(avg_entry, qty, leverage, reason, maker_fee, taker_fee)

                trade = Trade(
                    open_time=pos.open_bar,
                    close_time=bar_index,
                    symbol=symbol,
                    side="Buy" if direction == SignalDirection.LONG else "Sell",
                    entry_price=avg_entry,
                    exit_price=exit_price,
                    qty=qty,
                    pnl=pnl,
                    fee=fee,
                    reason=reason,
                    grid_fills=sum(e.grid_fills for e in pos.entries),
                    pyramid_level=pos.pyramid_count,
                )

                positions.pop(i)
                if not positions:
                    del self.positions[symbol]
                return trade
        return None

    def close_all(
        self,
        exit_price_map: dict[str, float],
        bar_index: int,
        reason: str,
        leverage: int = 1,
    ) -> list[Trade]:
        """Close all positions (drawdown halt)."""
        trades = []
        symbols = list(self.positions.keys())
        for symbol in symbols:
            price = exit_price_map.get(symbol)
            if price is None or price <= 0:
                continue
            positions = list(self.positions.get(symbol, []))
            for pos in positions:
                trade = self.close_position(
                    symbol, pos.direction, price, bar_index, reason, leverage,
                )
                if trade:
                    trades.append(trade)
        return trades

    def get_position(self, symbol: str, direction: SignalDirection) -> ManagedPosition | None:
        """Get an existing position for a symbol+direction."""
        for pos in self.positions.get(symbol, []):
            if pos.direction == direction:
                return pos
        return None

    def get_all_positions(self) -> list[ManagedPosition]:
        """Get all open positions."""
        result = []
        for positions in self.positions.values():
            result.extend(positions)
        return result
