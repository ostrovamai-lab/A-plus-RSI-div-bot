"""Data models for A+ RSI Divergence Trading Bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def bybit(self) -> str:
        return "Buy" if self == Side.LONG else "Sell"

    @property
    def opposite_bybit(self) -> str:
        return "Sell" if self == Side.LONG else "Buy"


class OrderStatus(str, Enum):
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    NEW = "NEW"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class DivergenceType(str, Enum):
    REGULAR_BULLISH = "REGULAR_BULLISH"
    REGULAR_BEARISH = "REGULAR_BEARISH"
    HIDDEN_BULLISH = "HIDDEN_BULLISH"
    HIDDEN_BEARISH = "HIDDEN_BEARISH"


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class APlusSignal:
    """A+ signal emitted by the state machine."""
    bar_index: int
    direction: SignalDirection
    price: float            # close price at signal bar
    fractal_price: float    # gate fractal price (used for SL)
    ema_fast: float
    ema_slow: float
    timestamp: int = 0


@dataclass
class RSIResult:
    """Full RSI indicator output for a single bar."""
    rsi: float
    basis: float            # EMA of RSI (Bollinger middle)
    upper_bb: float
    lower_bb: float
    disp_up: float          # displacement threshold up
    disp_down: float        # displacement threshold down
    ribbon_score: int       # 0-5: count of bullish ribbon layers
    bullish_cross: bool     # RSI crossed above disp_up
    bearish_cross: bool     # RSI crossed below disp_down


@dataclass
class Divergence:
    """Detected divergence between price and RSI."""
    div_type: DivergenceType
    bar_index: int          # bar where divergence confirmed
    rsi_current: float
    rsi_previous: float
    price_current: float
    price_previous: float


@dataclass
class SignalScore:
    """Composite score breakdown for a signal."""
    total: float = 0.0
    aplus_signal: float = 0.0
    rsi_divergence: float = 0.0
    ema_ribbon: float = 0.0
    kama_trend: float = 0.0
    rsi_position: float = 0.0
    htf_alignment: float = 0.0
    bb_position: float = 0.0
    volume: float = 0.0
    tier: str = "REJECT"    # A+, A, B, REJECT

    @property
    def position_scale(self) -> float:
        if self.total >= 80:
            return 1.0
        elif self.total >= 65:
            return 0.75
        elif self.total >= 50:
            return 0.5
        return 0.0


@dataclass
class GridLevel:
    """Single entry in the Fibonacci DCA grid."""
    level_index: int
    price: float
    side: str               # "Buy" or "Sell"
    qty: Decimal
    is_entry: bool = True


@dataclass
class Position:
    """Open position on exchange."""
    symbol: str
    side: Side
    size: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal = Decimal("0")
    leverage: int = 1
    sl_price: Optional[Decimal] = None
    position_idx: int = 0


@dataclass
class OrderResult:
    """Result of placing an order."""
    order_id: str
    symbol: str
    side: str
    qty: Decimal
    price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.NEW


@dataclass
class InstrumentInfo:
    """Exchange instrument metadata."""
    symbol: str
    min_qty: Decimal
    qty_step: Decimal
    tick_size: Decimal
    min_order_usd: Decimal = Decimal("5")


@dataclass
class Trade:
    """Completed trade (for backtest or live tracking)."""
    open_time: int = 0
    close_time: int = 0
    symbol: str = ""
    side: str = ""          # "Buy" or "Sell"
    entry_price: float = 0.0
    exit_price: float = 0.0
    qty: float = 0.0
    pnl: float = 0.0
    fee: float = 0.0
    reason: str = ""        # "opposite_signal", "sl", "time_stop", "drawdown"
    score: float = 0.0      # entry signal score
    grid_fills: int = 0     # how many grid levels filled
    pyramid_level: int = 0  # 0=initial, 1-2=pyramids

    @property
    def net_pnl(self) -> float:
        """PnL after fees."""
        return self.pnl - self.fee


@dataclass
class BacktestResult:
    """Backtest output."""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[int] = field(default_factory=list)

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
