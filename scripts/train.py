#!/usr/bin/env python3
"""CLI: train the recursive-learning engine for a symbol (initial bootstrap).

Usage: python scripts/train.py AAPL [more symbols...]
Writes ai/cache/<SYM>_daily.json, ai/<SYM>_model.json, ai/<SYM>_backtest.json,
ai/<SYM>_prediction.json and updates data.js for GitHub Pages.
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.backtest import walk_forward, persist, AI_DIR  # noqa: E402


def next_trading_day(d):
    nd = d + datetime.timedelta(days=1)
    while nd.weekday() >= 5:
        nd += datetime.timedelta(days=1)
    return nd


def main():
    symbols = [s.upper() for s in (sys.argv[1:] or ["AAPL"])]
    for sym in symbols:
        print(f"::group=Recursive learning run for {sym}")
        result = walk_forward(sym, verbose=True)
        persist(sym, result)
        pred = dict(result["summary"]["next_day_prediction"])
        as_of = datetime.date.fromisoformat(pred["as_of_date"])
        pred["target_date"] = next_trading_day(as_of).isoformat()
        with open(os.path.join(AI_DIR, f"{sym}_prediction.json"), "w") as f:
            json.dump(pred, f, indent=1)
        st = {"pending": pred, "grades": [], "cycles": 0}
        with open(os.path.join(AI_DIR, f"{sym}_state.json"), "w") as f:
            json.dump(st, f, indent=1)
        print(f"Summary: {json.dumps(result['summary'], indent=1)}")
        print("::endgroup")
    build_web_data(symbols)


def build_web_data(symbols):
    """Aggregate all artifacts into data.js consumed by the GitHub Page."""
    root = os.path.join(os.path.dirname(__file__), "..")
    data = {}
    for sym in symbols:
        try:
            bt = json.load(open(os.path.join(root, AI_DIR, f"{sym}_backtest.json")))
            pred = json.load(open(os.path.join(root, AI_DIR, f"{sym}_prediction.json")))
            bars = json.load(open(os.path.join(root, "ai/cache", f"{sym}_daily.json")))
            state_p = os.path.join(root, AI_DIR, f"{sym}_state.json")
            st = json.load(open(state_p)) if os.path.exists(state_p) else {"grades": []}
            data[sym] = {"backtest": bt, "prediction": pred, "bars": bars[-260:],
                         "grades": st.get("grades", [])}
        except Exception as e:  # noqa: BLE001
            print(f"WARN skipping {sym}: {e}")
    out = os.path.join(root, "data.js")
    with open(out, "w") as f:
        f.write("window.STOCKPRED_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";")
    print(f"Wrote {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
