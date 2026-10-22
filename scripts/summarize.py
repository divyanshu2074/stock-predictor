#!/usr/bin/env python3
"""Markdown snippets for GitHub Actions step summaries.

Usage:
  summarize.py ai/AAPL_prediction.json          -> next-day forecast block
  summarize.py --backtest ai/AAPL_backtest.json -> learning-curve block
"""
import json
import sys


def pred_block(path):
    d = json.load(open(path))
    sym = path.split("/")[-1].replace("_prediction.json", "")
    print(f"**{sym}** as of {d['as_of_date']} (close ${d['as_of_close']})")
    print(f"- next-day call: **{d['direction']}** ({100*d['predicted_return']:+.2f}%) "
          f"→ target **${d['predicted_close']}** by {d.get('target_date') or 'next session'}")
    band = d["confidence_band"]
    print(f"- confidence band: ${band[0]} – ${band[1]}")
    votes = ", ".join(f"{m}: {100*r:+.2f}%" for m, r in d.get("model_votes", {}).items())
    if votes:
        print(f"- model votes → {votes}")


def bt_block(path):
    s = json.load(open(path))["summary"]
    imp = "decreased ✅" if s["improvement"]["last_half_mape"] < s["improvement"]["first_half_mape"] else "increased ⚠️"
    print(f"- **{s['symbol']}**: {s['weeks_simulated']} recursive weeks | "
          f"mean weekly MAPE {100*s['mean_mape']:.2f}% | median dir-acc {s['median_dir_acc']} | "
          f"error {imp} during learning "
          f"({100*s['improvement']['first_half_mape']:.2f}% → {100*s['improvement']['last_half_mape']:.2f}%)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--backtest":
        bt_block(args[1])
    elif args:
        pred_block(args[0])
