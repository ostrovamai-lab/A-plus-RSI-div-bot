#!/usr/bin/env python3
"""CLI live bot runner for the A+ RSI Divergence strategy.

Usage:
    python scripts/run_live.py
    python scripts/run_live.py --config config.yaml
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from live.bot import LiveBot


def main():
    parser = argparse.ArgumentParser(description="A+ RSI Divergence Live Bot")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("bot.log"),
        ],
    )

    cfg = load_config(args.config)
    bot = LiveBot(cfg)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown(sig_name):
        logging.info("Received %s — shutting down...", sig_name)
        loop.create_task(bot.stop())

    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, shutdown, sig_name.name)

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt — stopping...")
        loop.run_until_complete(bot.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
