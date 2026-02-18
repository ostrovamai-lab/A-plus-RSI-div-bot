"""Shared PnL and fee calculation — single source of truth.

Used by:
- backtest/engine.py (_close_position)
- strategy/position_manager.py (close_position)
- ManagedPosition.unrealized_pnl
"""

from __future__ import annotations

from models import SignalDirection

# Reasons that trigger a taker (market) exit
MARKET_EXIT_REASONS = frozenset({
    "sl", "trail_sl", "breakeven_sl", "opposite_signal", "drawdown",
    "regime_flip", "choch_partial",
})


def compute_pnl(
    direction: SignalDirection,
    entry_price: float,
    exit_price: float,
    qty: float,
    leverage: int = 1,
) -> float:
    """Compute raw PnL (before fees).

    NOTE: qty is expected to be the leveraged position size (notional / price).
    The leverage parameter is kept for backward compatibility but is NOT
    applied again — the entry engine already factors leverage into qty.
    """
    if direction == SignalDirection.LONG:
        return (exit_price - entry_price) * qty
    return (entry_price - exit_price) * qty


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

    NOTE: qty already includes leverage (notional / price), so we don't
    multiply by leverage again.
    """
    notional = qty * entry_price
    if reason in MARKET_EXIT_REASONS:
        return notional * (maker_fee + taker_fee)
    return notional * (maker_fee * 2)
