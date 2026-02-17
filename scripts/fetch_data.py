#!/usr/bin/env python3
"""Download historical 1-minute OHLCV data from Bybit for backtesting.

Usage:
    python scripts/fetch_data.py --symbol BTCUSDT --days 60
    python scripts/fetch_data.py --symbol SUIUSDT --days 30
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_DEMO, DATA_DIR
from connectors.bybit_connector import BybitConnector

logger = logging.getLogger(__name__)


async def fetch_klines_paginated(
    connector: BybitConnector,
    symbol: str,
    interval: str,
    days: int,
) -> list[dict]:
    """Fetch historical klines with pagination (Bybit max 1000 per request)."""
    minutes_per_bar = {"1": 1, "3": 3, "5": 5, "15": 15, "60": 60, "240": 240}
    mpb = minutes_per_bar.get(interval, 1)
    total_needed = days * 24 * 60 // mpb

    all_klines: list[dict] = []
    end_ms = int(time.time() * 1000)
    remaining = total_needed

    while remaining > 0:
        batch_limit = min(remaining, 1000)
        klines = await connector.get_klines(
            symbol, interval, limit=batch_limit, end=end_ms,
        )
        if not klines:
            break
        all_klines = klines + all_klines
        end_ms = klines[0]["timestamp"] - 1
        remaining -= len(klines)
        logger.info("Fetched chunk: %d bars (total: %d/%d)", len(klines), len(all_klines), total_needed)
        if len(klines) < batch_limit:
            break

    # Deduplicate
    seen: set[int] = set()
    deduped = []
    for k in all_klines:
        if k["timestamp"] not in seen:
            seen.add(k["timestamp"])
            deduped.append(k)

    logger.info("Total: %d %sm klines for %s (%d days)", len(deduped), interval, symbol, days)
    return deduped


async def main():
    parser = argparse.ArgumentParser(description="Fetch historical kline data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=60, help="Days of history")
    parser.add_argument("--intervals", default="1,60,240", help="Comma-separated intervals to fetch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    connector = BybitConnector(
        api_key=BYBIT_API_KEY,
        api_secret=BYBIT_API_SECRET,
        demo=BYBIT_DEMO,
    )

    intervals = [i.strip() for i in args.intervals.split(",")]

    for interval in intervals:
        logger.info("Fetching %s %sm bars for %d days...", args.symbol, interval, args.days)
        klines = await fetch_klines_paginated(connector, args.symbol, interval, args.days)

        out_path = DATA_DIR / f"{args.symbol}_{interval}m_{args.days}d.json"
        with open(out_path, "w") as f:
            json.dump(klines, f)
        logger.info("Saved to %s (%d bars)", out_path, len(klines))


if __name__ == "__main__":
    asyncio.run(main())
