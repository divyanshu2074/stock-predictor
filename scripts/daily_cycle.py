#!/usr/bin/env python3
"""CLI: run one daily self-learning cycle for each tracked symbol.

Usage: python scripts/daily_cycle.py AAPL MSFT ...
Grades yesterday's forecast, ingests new bars, fine-tunes, re-predicts next day,
and regenerates web/data.js.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.daily import run_cycle  # noqa: E402
from scripts.train import build_web_data  # noqa: E402


def main():
    symbols = [s.upper() for s in (sys.argv[1:] or ["AAPL"])]
    full = os.environ.get("FULL_BACKTEST", "0") == "1"
    for sym in symbols:
        print(f"::group=Daily cycle {sym}")
        run_cycle(sym, full_backtest=full)
        print("::endgroup")
    build_web_data(symbols)


if __name__ == "__main__":
    main()
