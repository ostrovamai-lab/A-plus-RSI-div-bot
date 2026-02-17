"""Risk manager — position sizing, drawdown tracking, reinvestment scaling.

Reinvestment formula:
    scale = sqrt(current_equity / initial_equity)
    Dampened compounding: $1000 → $1500 = 1.22x, not 1.5x
    Clamped to [min_scale, max_scale] range
"""

from __future__ import annotations

import math


class RiskManager:
    """Tracks equity and computes position sizes."""

    def __init__(
        self,
        initial_capital: float = 1000.0,
        leverage: int = 10,
        max_drawdown_pct: float = 10.0,
        reinvest: bool = True,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.max_drawdown_pct = max_drawdown_pct
        self.reinvest = reinvest
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.current_equity = initial_capital
        self.peak_equity = initial_capital
        self.realized_pnl = 0.0
        self.is_halted = False

    def update_equity(self, realized_pnl_delta: float) -> None:
        """Update equity after a trade closes."""
        self.realized_pnl += realized_pnl_delta
        self.current_equity = self.initial_capital + self.realized_pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

    def check_drawdown(self) -> bool:
        """Check if max drawdown is breached. Returns True if halted."""
        if self.max_drawdown_pct <= 0:
            return False
        dd_pct = (self.initial_capital - self.current_equity) / self.initial_capital * 100
        if dd_pct >= self.max_drawdown_pct:
            self.is_halted = True
            return True
        return False

    def compute_base_size_usd(self, position_scale: float = 1.0) -> float:
        """Compute base position size in USD for a new entry.

        Args:
            position_scale: Score-based scaling (1.0 for A+, 0.75 for A, 0.5 for B).

        Returns:
            USD amount for the full grid (before Fib splitting).
        """
        if self.is_halted:
            return 0.0

        base = self.current_equity * position_scale

        if self.reinvest and self.initial_capital > 0:
            ratio = self.current_equity / self.initial_capital
            scale = math.sqrt(max(ratio, 0.01))
            scale = max(self.min_scale, min(self.max_scale, scale))
            base = self.initial_capital * scale * position_scale

        return base

    def drawdown_pct(self) -> float:
        """Current drawdown from peak as percentage."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity * 100

    def equity_scale(self) -> float:
        """Current reinvestment scale factor."""
        if not self.reinvest or self.initial_capital <= 0:
            return 1.0
        ratio = self.current_equity / self.initial_capital
        scale = math.sqrt(max(ratio, 0.01))
        return max(self.min_scale, min(self.max_scale, scale))
