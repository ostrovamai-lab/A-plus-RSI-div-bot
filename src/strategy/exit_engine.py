"""Exit engine — manages position closing conditions.

Exit conditions (priority order):
1. Stop Loss: Hard SL behind fractal gate
2. Opposite A+ signal: Close full position at market
3. Time stop: 200 bars (~26h on 8min)
4. Max drawdown: Equity drops 10% from peak → halt all trading
"""

from __future__ import annotations

from dataclasses import dataclass

from models import SignalDirection


@dataclass
class ExitSignal:
    """Signal to close a position."""
    reason: str          # "sl", "opposite_signal", "time_stop", "drawdown"
    exit_price: float
    close_all: bool = False  # True = close ALL positions (drawdown halt)


def check_stop_loss(
    direction: SignalDirection,
    sl_price: float,
    bar_high: float,
    bar_low: float,
) -> ExitSignal | None:
    """Check if stop loss was hit during this bar.

    Uses wick extremes (high/low) for fill detection.
    """
    if direction == SignalDirection.LONG:
        if bar_low <= sl_price:
            return ExitSignal(reason="sl", exit_price=sl_price)
    else:
        if bar_high >= sl_price:
            return ExitSignal(reason="sl", exit_price=sl_price)
    return None


def check_opposite_signal(
    position_direction: SignalDirection,
    aplus_long: bool,
    aplus_short: bool,
    close_price: float,
) -> ExitSignal | None:
    """Check if an opposite A+ signal fires, requiring position close."""
    if position_direction == SignalDirection.LONG and aplus_short:
        return ExitSignal(reason="opposite_signal", exit_price=close_price)
    if position_direction == SignalDirection.SHORT and aplus_long:
        return ExitSignal(reason="opposite_signal", exit_price=close_price)
    return None


def check_time_stop(
    bars_held: int,
    max_bars: int,
    close_price: float,
) -> ExitSignal | None:
    """Check if position has exceeded maximum hold time."""
    if max_bars > 0 and bars_held >= max_bars:
        return ExitSignal(reason="time_stop", exit_price=close_price)
    return None


def check_drawdown_halt(
    current_equity: float,
    initial_capital: float,
    max_drawdown_pct: float,
    current_price: float = 0.0,
) -> ExitSignal | None:
    """Check if equity drawdown exceeds threshold — halt all trading.

    Drawdown measured from initial capital (not peak equity) to avoid
    false halts during normal fluctuations.
    """
    if max_drawdown_pct <= 0 or initial_capital <= 0:
        return None

    drawdown_pct = (initial_capital - current_equity) / initial_capital * 100
    if drawdown_pct >= max_drawdown_pct:
        return ExitSignal(reason="drawdown", exit_price=current_price, close_all=True)
    return None
