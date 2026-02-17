"""Abstract base class for exchange connectors."""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional

from models import InstrumentInfo, OrderResult, Position


class ExchangeConnector(ABC):
    """Unified interface for exchange connectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Exchange name for logging."""

    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Get open positions."""

    @abstractmethod
    async def place_market_order(
        self, symbol: str, side: str, qty: Decimal,
        sl: Optional[Decimal] = None, tp: Optional[Decimal] = None,
    ) -> OrderResult:
        """Place a market order."""

    @abstractmethod
    async def place_limit_order(
        self, symbol: str, side: str, qty: Decimal,
        price: Decimal, reduce_only: bool = False,
    ) -> OrderResult:
        """Place a limit order."""

    @abstractmethod
    async def modify_sl(self, symbol: str, new_sl_price: Decimal) -> None:
        """Move stop-loss to a new price."""

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> None:
        """Cancel a single order."""

    @abstractmethod
    async def cancel_all_orders(self, symbol: str) -> None:
        """Cancel all open orders for a symbol."""

    @abstractmethod
    async def get_instrument_info(self, symbol: str) -> InstrumentInfo:
        """Get instrument metadata."""

    @abstractmethod
    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Get current last price."""

    @abstractmethod
    async def get_wallet_balance(self) -> Decimal:
        """Get USDT wallet balance."""

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol."""

    @abstractmethod
    async def get_klines(
        self, symbol: str, interval: str, limit: int = 200,
        start: Optional[int] = None, end: Optional[int] = None,
    ) -> list[dict]:
        """Fetch OHLCV klines sorted ascending by time."""

    @abstractmethod
    async def get_open_orders(self, symbol: str) -> list[dict]:
        """Get all open orders for a symbol."""

    async def set_margin_mode(
        self, symbol: str, leverage: int, cross: bool = True,
    ) -> None:
        """Set margin mode. Default no-op."""

    async def start_websocket(self, symbol: str, kline_intervals: list[str] | None = None) -> None:
        """Start WebSocket connections. Default no-op."""

    async def stop_websocket(self) -> None:
        """Stop WebSocket connections. Default no-op."""
