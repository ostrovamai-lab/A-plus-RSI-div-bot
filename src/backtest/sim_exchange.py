"""Simulated exchange for backtesting — handles grid limit order fills.

Grid simulation:
- Limit BUY fills when bar's low touches or goes below buy price
- Limit SELL fills when bar's high touches or goes above sell price
- Uses maker fee for limit order fills (0.02%)
- Uses taker fee for market exits (0.055%)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimOrder:
    """A simulated limit order."""
    order_id: int
    price: float
    side: str           # "Buy" or "Sell"
    qty: float
    is_entry: bool = True
    filled: bool = False
    fill_bar: int = -1


@dataclass
class SimPosition:
    """A simulated open position."""
    symbol: str
    side: str           # "Buy" or "Sell"
    entries: list[tuple[float, float]] = field(default_factory=list)  # (price, qty)
    sl_price: float = 0.0
    open_bar: int = 0

    @property
    def avg_entry(self) -> float:
        total_qty = sum(q for _, q in self.entries)
        if total_qty == 0:
            return 0.0
        return sum(p * q for p, q in self.entries) / total_qty

    @property
    def total_qty(self) -> float:
        return sum(q for _, q in self.entries)


class SimExchange:
    """Simulated exchange for backtest order management."""

    def __init__(
        self,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.00055,
        leverage: int = 10,
    ):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.leverage = leverage
        self._next_id = 1
        self.open_orders: list[SimOrder] = []
        self.positions: dict[str, list[SimPosition]] = {}

    def place_limit_order(
        self, price: float, side: str, qty: float, is_entry: bool = True,
    ) -> SimOrder:
        """Place a simulated limit order."""
        order = SimOrder(
            order_id=self._next_id,
            price=price, side=side, qty=qty, is_entry=is_entry,
        )
        self._next_id += 1
        self.open_orders.append(order)
        return order

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        count = len(self.open_orders)
        self.open_orders.clear()
        return count

    def check_fills(
        self, bar_high: float, bar_low: float, bar_index: int,
    ) -> list[SimOrder]:
        """Check which limit orders would fill on this bar.

        Buy fills when low <= price, Sell fills when high >= price.
        Returns list of filled orders (removed from open_orders).
        """
        filled = []
        remaining = []
        for order in self.open_orders:
            hit = False
            if order.side == "Buy" and bar_low <= order.price:
                hit = True
            elif order.side == "Sell" and bar_high >= order.price:
                hit = True

            if hit:
                order.filled = True
                order.fill_bar = bar_index
                filled.append(order)
            else:
                remaining.append(order)

        self.open_orders = remaining
        return filled

    def check_sl(
        self, symbol: str, bar_high: float, bar_low: float,
    ) -> list[tuple[SimPosition, float]]:
        """Check if any position's stop loss was hit.

        Returns list of (position, fill_price) tuples for hit SLs.
        """
        hits = []
        positions = self.positions.get(symbol, [])
        for pos in positions:
            if pos.sl_price <= 0:
                continue
            if pos.side == "Buy" and bar_low <= pos.sl_price:
                hits.append((pos, pos.sl_price))
            elif pos.side == "Sell" and bar_high >= pos.sl_price:
                hits.append((pos, pos.sl_price))
        return hits

    def compute_pnl(
        self, side: str, entry_price: float, exit_price: float,
        qty: float, is_market_exit: bool = False,
    ) -> tuple[float, float]:
        """Compute PnL and fee for a trade.

        Returns (pnl, fee).
        """
        if side == "Buy":
            pnl = (exit_price - entry_price) * qty * self.leverage
        else:
            pnl = (entry_price - exit_price) * qty * self.leverage

        notional = qty * entry_price * self.leverage
        if is_market_exit:
            fee = notional * (self.maker_fee + self.taker_fee)
        else:
            fee = notional * (self.maker_fee + self.maker_fee)  # limit+limit

        return pnl, fee
