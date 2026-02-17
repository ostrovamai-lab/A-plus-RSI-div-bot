"""Fibonacci DCA grid entry engine.

When an A+ signal fires with score >= threshold:
- Compute grid range = |signal_price - fractal_price|
- Minimum range = 0.5 * ATR(14) to avoid too-tight grids
- Place 3 limit orders at Fib levels from signal_price toward fractal:
    Level 1: entry - 0.236 * range → qty = base * 1.000
    Level 2: entry - 0.382 * range → qty = base * 1.272
    Level 3: entry - 0.618 * range → qty = base * 1.618
  (biggest order at best price = furthest from entry)
- Stop Loss = fractal_price ± buffer
"""

from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

from config import EntryParams
from models import GridLevel, InstrumentInfo, SignalDirection, SignalScore


def compute_entry_grid(
    direction: SignalDirection,
    signal_price: float,
    fractal_price: float,
    atr_value: float,
    score: SignalScore,
    capital_usd: float,
    leverage: int,
    params: EntryParams | None = None,
    instrument_info: InstrumentInfo | None = None,
) -> list[GridLevel]:
    """Compute Fibonacci DCA grid levels for an entry.

    Args:
        direction: LONG or SHORT.
        signal_price: Price at A+ signal bar.
        fractal_price: Gate fractal price (used as grid boundary and SL anchor).
        atr_value: ATR(14) for minimum range enforcement.
        score: Signal score (affects position sizing via tier).
        capital_usd: Total available capital.
        leverage: Leverage multiplier.
        params: Entry parameters (uses defaults if None).
        instrument_info: Exchange instrument metadata for rounding.

    Returns:
        List of GridLevel objects (3 levels by default).
    """
    p = params or EntryParams()
    is_long = direction == SignalDirection.LONG

    # Grid range
    raw_range = abs(signal_price - fractal_price)
    min_range = p.min_range_atr_mult * atr_value
    grid_range = max(raw_range, min_range)

    if grid_range <= 0 or signal_price <= 0 or score.position_scale <= 0:
        return []

    # Base position size (adjusted by score tier)
    base_usd = capital_usd * score.position_scale / len(p.fib_levels)
    base_notional = Decimal(str(base_usd)) * leverage

    levels: list[GridLevel] = []

    for i, (fib_offset, fib_volume) in enumerate(zip(p.fib_levels, p.fib_volumes)):
        if is_long:
            price = signal_price - fib_offset * grid_range
        else:
            price = signal_price + fib_offset * grid_range

        if price <= 0:
            continue

        qty = base_notional * Decimal(str(fib_volume)) / Decimal(str(price))

        if instrument_info:
            # Round price to tick_size
            d_price = Decimal(str(price))
            d_price = (d_price / instrument_info.tick_size).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            ) * instrument_info.tick_size
            price = float(d_price)

            qty = (qty / instrument_info.qty_step).to_integral_value(
                rounding=ROUND_DOWN
            ) * instrument_info.qty_step
            if qty < instrument_info.min_qty:
                qty = instrument_info.min_qty

        side = "Buy" if is_long else "Sell"

        levels.append(GridLevel(
            level_index=i,
            price=round(price, 8),
            side=side,
            qty=qty,
            is_entry=True,
        ))

    return levels


def compute_stop_loss(
    direction: SignalDirection,
    fractal_price: float,
    sl_buffer_pct: float = 0.002,
    atr_value: float = 0.0,
    sl_atr_multiplier: float = 0.5,
) -> float:
    """Compute stop loss price behind the fractal gate.

    Uses the wider of percentage-based buffer or ATR-based buffer,
    so SL adapts to current volatility regime.

    Args:
        direction: LONG or SHORT.
        fractal_price: Gate fractal price.
        sl_buffer_pct: Buffer as fraction of fractal price (0.002 = 0.2%).
        atr_value: Current ATR(14) value.
        sl_atr_multiplier: Minimum SL distance as fraction of ATR.

    Returns:
        Stop loss price.
    """
    pct_buffer = fractal_price * sl_buffer_pct
    atr_buffer = atr_value * sl_atr_multiplier
    buffer = max(pct_buffer, atr_buffer)

    if direction == SignalDirection.LONG:
        return fractal_price - buffer
    else:
        return fractal_price + buffer
