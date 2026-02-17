"""Shared PnL and fee calculation — single source of truth.

Used by:
- backtest/engine.py (_close_position)
- strategy/position_manager.py (close_position)
- ManagedPosition.unrealized_pnl
"""

from __future__ import annotations

from models import SignalDirection

# Reasons that trigger a taker (market) exit
MARKET_EXIT_REASONS = frozenset({"sl", "opposite_signal", "drawdown"})


def compute_pnl(
    direction: SignalDirection,
    entry_price: float,
    exit_price: float,
    qty: float,
    leverage: int = 1,
) -> float:
    """Compute raw PnL (before fees)."""
    if direction == SignalDirection.LONG:
        return (exit_price - entry_price) * qty * leverage
    return (entry_price - exit_price) * qty * leverage


def compute_fee(
    entry_price: float,
    qty: float,
    leverage: int,
    reason: str,
    maker_fee: float = 0.0002,
    taker_fee: float = 0.00055,
) -> float:
    """Compute total fee for a round-trip trade.

    Entry is always maker (limit). Exit depends on reason:
    - Market exit (sl, opposite_signal, drawdown): maker + taker
    - Limit exit (time_stop, backtest_end): maker + maker
    """
    notional = qty * entry_price * leverage
    if reason in MARKET_EXIT_REASONS:
        return notional * (maker_fee + taker_fee)
    return notional * (maker_fee * 2)
