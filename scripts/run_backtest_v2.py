#!/usr/bin/env python3
"""CLI backtest runner for the V2 trend-following SMC + A+ strategy.

Usage:
    python scripts/run_backtest_v2.py --symbol BTCUSDT --days 60
    python scripts/run_backtest_v2.py --symbol SUIUSDT --days 30 --min-score 70
    python scripts/run_backtest_v2.py --symbol HYPEUSDT --sweep
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtest.engine_v2 import run_backtest_v2
from backtest.report import plot_equity_curve, print_report
from config import DATA_DIR, load_config


def load_cached_data(symbol: str, interval: str, days: int) -> list[dict]:
    """Load cached kline data from data/ directory."""
    path = DATA_DIR / f"{symbol}_{interval}m_{days}d.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


async def fetch_if_needed(symbol: str, days: int) -> tuple[list[dict], list[dict]]:
    """Load 1m and 1h data from cache."""
    klines_1m = load_cached_data(symbol, "1", days)
    klines_1h = load_cached_data(symbol, "60", days)

    if not klines_1m:
        print(f"No cached data found. Run: python scripts/fetch_data.py --symbol {symbol} --days {days}")
        sys.exit(1)

    return klines_1m, klines_1h


def main():
    parser = argparse.ArgumentParser(description="V2 Trend-Following SMC + A+ Backtester")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=60, help="Days of history")
    parser.add_argument("--min-score", type=float, default=0, help="Override min_score (0=use config)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--plot", action="store_true", help="Show equity curve plot")
    parser.add_argument("--save-plot", default=None, help="Save equity curve to file")
    parser.add_argument("--sweep", action="store_true", help="Sweep min_score from 40-90")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config(args.config)

    klines_1m, klines_1h = asyncio.run(fetch_if_needed(args.symbol, args.days))
    print(f"Data loaded: {len(klines_1m)} 1m bars, {len(klines_1h or [])} 1h bars")

    if args.sweep:
        print(f"\n{'=' * 70}")
        print(f"  V2 SCORE THRESHOLD SWEEP — {args.symbol}")
        print(f"{'=' * 70}")
        print(f"{'Min Score':>10s} | {'Trades':>7s} | {'Win Rate':>9s} | {'Net PnL':>10s} | {'Max DD':>8s} | {'Sharpe':>7s}")
        print(f"{'-' * 70}")

        for min_score in range(40, 95, 5):
            cfg.v2.min_score = float(min_score)
            result = run_backtest_v2(klines_1m, klines_1h or None, cfg, args.symbol)
            print(
                f"{min_score:>10d} | {result.total_trades:>7d} | "
                f"{result.win_rate:>8.1f}% | ${result.net_pnl:>9.2f} | "
                f"{result.max_drawdown_pct:>7.1f}% | {result.sharpe_ratio:>6.2f}"
            )
        return

    if args.min_score > 0:
        cfg.v2.min_score = args.min_score

    print(f"Running V2 backtest on {args.symbol} ({len(klines_1m)} 1m bars → 8min resampling)...")
    result = run_backtest_v2(klines_1m, klines_1h or None, cfg, args.symbol)
    print_report(result, args.symbol)

    if args.plot or args.save_plot:
        plot_equity_curve(result, args.symbol, args.save_plot)


if __name__ == "__main__":
    main()
