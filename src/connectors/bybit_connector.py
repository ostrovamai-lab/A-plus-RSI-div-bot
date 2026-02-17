"""Bybit exchange connector using pybit unified trading API.

CRITICAL: Always use demo=True, NEVER testnet=True.
Bybit Demo (api-demo.bybit.com) = mirror of real market with virtual money.
"""

import asyncio
import logging
from decimal import ROUND_DOWN, Decimal
from typing import Optional

from pybit.unified_trading import HTTP

from connectors.base import ExchangeConnector
from models import InstrumentInfo, OrderResult, OrderStatus, Position, Side

logger = logging.getLogger(__name__)


class BybitConnector(ExchangeConnector):
    """Bybit connector — category='linear' for USDT perpetuals."""

    CATEGORY = "linear"

    def __init__(self, api_key: str, api_secret: str, demo: bool = True):
        self._session = HTTP(
            testnet=False,
            demo=demo,
            api_key=api_key,
            api_secret=api_secret,
        )
        self._demo = demo
        logger.info("BybitConnector initialized (demo=%s)", demo)

    @property
    def name(self) -> str:
        return "bybit_demo" if self._demo else "bybit"

    def _run_sync(self, func, *args, **kwargs):
        return asyncio.to_thread(func, *args, **kwargs)

    @staticmethod
    def _check(resp: dict, context: str = "") -> dict:
        code = resp.get("retCode", -1)
        if code != 0:
            msg = resp.get("retMsg", "unknown error")
            raise RuntimeError(f"Bybit API error ({context}): [{code}] {msg}")
        return resp.get("result", {})

    @staticmethod
    def _qty_str(qty: Decimal, step: Decimal) -> str:
        rounded = (qty / step).to_integral_value(rounding=ROUND_DOWN) * step
        return f"{rounded:f}"

    @staticmethod
    def _price_str(price: Decimal, tick: Decimal) -> str:
        rounded = (price / tick).to_integral_value(rounding=ROUND_DOWN) * tick
        return f"{rounded:f}"

    async def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        kwargs = {"category": self.CATEGORY}
        if symbol:
            kwargs["symbol"] = symbol
        else:
            kwargs["settleCoin"] = "USDT"

        positions = []
        cursor = None
        while True:
            if cursor:
                kwargs["cursor"] = cursor
            resp = await self._run_sync(self._session.get_positions, **kwargs)
            result = self._check(resp, "get_positions")
            for p in result.get("list", []):
                size = Decimal(p.get("size", "0"))
                if size == 0:
                    continue
                side_str = p.get("side", "")
                positions.append(Position(
                    symbol=p["symbol"],
                    side=Side.LONG if side_str == "Buy" else Side.SHORT,
                    size=size,
                    entry_price=Decimal(p.get("avgPrice", "0")),
                    unrealized_pnl=Decimal(p.get("unrealisedPnl", "0")),
                    leverage=int(p.get("leverage", "1")),
                    sl_price=Decimal(p["stopLoss"]) if p.get("stopLoss") and p["stopLoss"] != "0" else None,
                    position_idx=int(p.get("positionIdx", 0)),
                ))
            cursor = result.get("nextPageCursor")
            if not cursor:
                break
        return positions

    async def place_market_order(
        self, symbol: str, side: str, qty: Decimal,
        sl: Optional[Decimal] = None, tp: Optional[Decimal] = None,
    ) -> OrderResult:
        info = await self.get_instrument_info(symbol)
        qty_s = self._qty_str(qty, info.qty_step)
        kwargs: dict = {
            "category": self.CATEGORY, "symbol": symbol,
            "side": side, "orderType": "Market",
            "qty": qty_s, "positionIdx": 0,
        }
        if sl is not None:
            kwargs["stopLoss"] = self._price_str(sl, info.tick_size)
        if tp is not None:
            kwargs["takeProfit"] = self._price_str(tp, info.tick_size)

        logger.info("place_market_order %s %s %s (sl=%s, tp=%s)", symbol, side, qty_s, sl, tp)
        resp = await self._run_sync(self._session.place_order, **kwargs)
        result = self._check(resp, "place_market_order")
        return OrderResult(
            order_id=result.get("orderId", ""), symbol=symbol,
            side=side, qty=Decimal(qty_s), status=OrderStatus.FILLED,
        )

    async def place_limit_order(
        self, symbol: str, side: str, qty: Decimal,
        price: Decimal, reduce_only: bool = False,
    ) -> OrderResult:
        info = await self.get_instrument_info(symbol)
        qty_s = self._qty_str(qty, info.qty_step)
        price_s = self._price_str(price, info.tick_size)
        kwargs: dict = {
            "category": self.CATEGORY, "symbol": symbol,
            "side": side, "orderType": "Limit",
            "qty": qty_s, "price": price_s,
            "timeInForce": "GTC", "positionIdx": 0,
        }
        if reduce_only:
            kwargs["reduceOnly"] = True

        resp = await self._run_sync(self._session.place_order, **kwargs)
        result = self._check(resp, "place_limit_order")
        return OrderResult(
            order_id=result.get("orderId", ""), symbol=symbol,
            side=side, qty=Decimal(qty_s), price=price, status=OrderStatus.NEW,
        )

    async def modify_sl(self, symbol: str, new_sl_price: Decimal) -> None:
        info = await self.get_instrument_info(symbol)
        sl_s = self._price_str(new_sl_price, info.tick_size)
        resp = await self._run_sync(
            self._session.set_trading_stop,
            category=self.CATEGORY, symbol=symbol,
            stopLoss=sl_s, slTriggerBy="MarkPrice",
            tpslMode="Full", positionIdx=0,
        )
        self._check(resp, "modify_sl")
        logger.info("SL moved to %s for %s", sl_s, symbol)

    async def cancel_order(self, symbol: str, order_id: str) -> None:
        resp = await self._run_sync(
            self._session.cancel_order,
            category=self.CATEGORY, symbol=symbol, orderId=order_id,
        )
        self._check(resp, "cancel_order")

    async def cancel_all_orders(self, symbol: str) -> None:
        resp = await self._run_sync(
            self._session.cancel_all_orders,
            category=self.CATEGORY, symbol=symbol,
        )
        self._check(resp, "cancel_all_orders")

    async def get_instrument_info(self, symbol: str) -> InstrumentInfo:
        resp = await self._run_sync(
            self._session.get_instruments_info,
            category=self.CATEGORY, symbol=symbol,
        )
        result = self._check(resp, "get_instrument_info")
        items = result.get("list", [])
        if not items:
            raise ValueError(f"Instrument not found: {symbol}")
        info = items[0]
        lot = info.get("lotSizeFilter", {})
        price_filter = info.get("priceFilter", {})
        return InstrumentInfo(
            symbol=symbol,
            min_qty=Decimal(lot.get("minOrderQty", "0.001")),
            qty_step=Decimal(lot.get("qtyStep", "0.001")),
            tick_size=Decimal(price_filter.get("tickSize", "0.01")),
        )

    async def get_ticker_price(self, symbol: str) -> Decimal:
        resp = await self._run_sync(
            self._session.get_tickers,
            category=self.CATEGORY, symbol=symbol,
        )
        result = self._check(resp, "get_ticker_price")
        items = result.get("list", [])
        if not items:
            raise ValueError(f"Ticker not found: {symbol}")
        return Decimal(items[0].get("lastPrice", "0"))

    async def get_wallet_balance(self) -> Decimal:
        resp = await self._run_sync(
            self._session.get_wallet_balance, accountType="UNIFIED",
        )
        result = self._check(resp, "get_wallet_balance")
        for account in result.get("list", []):
            for coin in account.get("coin", []):
                if coin.get("coin") == "USDT":
                    return Decimal(coin.get("walletBalance", "0"))
        return Decimal("0")

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        lev_str = str(leverage)
        try:
            resp = await self._run_sync(
                self._session.set_leverage,
                category=self.CATEGORY, symbol=symbol,
                buyLeverage=lev_str, sellLeverage=lev_str,
            )
            self._check(resp, "set_leverage")
            logger.info("Leverage set to %sx for %s", leverage, symbol)
        except Exception as exc:
            if "not modified" in str(exc).lower() or "110043" in str(exc):
                logger.debug("Leverage already %sx for %s", leverage, symbol)
            else:
                raise

    async def set_margin_mode(
        self, symbol: str, leverage: int, cross: bool = True,
    ) -> None:
        trade_mode = 0 if cross else 1
        lev_str = str(leverage)
        try:
            resp = await self._run_sync(
                self._session.switch_margin_mode,
                category=self.CATEGORY, symbol=symbol,
                tradeMode=trade_mode, buyLeverage=lev_str, sellLeverage=lev_str,
            )
            self._check(resp, "set_margin_mode")
        except Exception as exc:
            exc_str = str(exc)
            if "not modified" in exc_str.lower() or "110026" in exc_str or "10032" in exc_str:
                logger.debug("Margin mode already set for %s", symbol)
            else:
                raise

    async def get_klines(
        self, symbol: str, interval: str, limit: int = 200,
        start: Optional[int] = None, end: Optional[int] = None,
    ) -> list[dict]:
        kwargs: dict = {
            "category": self.CATEGORY, "symbol": symbol,
            "interval": interval, "limit": limit,
        }
        if start is not None:
            kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end
        resp = await self._run_sync(self._session.get_kline, **kwargs)
        result = self._check(resp, "get_klines")
        rows = result.get("list", [])
        klines = []
        for row in reversed(rows):
            klines.append({
                "timestamp": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })
        return klines

    async def get_open_orders(self, symbol: str) -> list[dict]:
        orders = []
        cursor = None
        while True:
            kwargs = {"category": self.CATEGORY, "symbol": symbol}
            if cursor:
                kwargs["cursor"] = cursor
            resp = await self._run_sync(self._session.get_open_orders, **kwargs)
            result = self._check(resp, "get_open_orders")
            for o in result.get("list", []):
                orders.append({
                    "orderId": o.get("orderId", ""),
                    "symbol": o.get("symbol", symbol),
                    "side": o.get("side", ""),
                    "price": o.get("price", "0"),
                    "qty": o.get("qty", "0"),
                    "orderType": o.get("orderType", ""),
                    "reduceOnly": o.get("reduceOnly", False),
                })
            cursor = result.get("nextPageCursor")
            if not cursor:
                break
        return orders
