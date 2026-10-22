# 📈 StockPRED — a self-learning stock prediction system that runs **entirely on GitHub**

Give it a ticker → it downloads 1 year of data → builds a prediction engine →
recursively learns week by week (predict → measure deviation → fine-tune → add data → repeat)
→ and every day after market close it publishes a **next-day price prediction** on GitHub Pages.

No servers, no paid APIs, no secrets required. The repository *is* the brain:
all training data, model weights, learning curves and live predictions are committed back by GitHub Actions.

## Live dashboard

🌐 **https://divyanshu2074.github.io/stock-predictor/**

## The loop you asked for — exactly how it's implemented

| Your requirement | Where it lives |
|---|---|
| Take the name of a stock | `Add / Retrain Stock` workflow (workflow_dispatch, any Yahoo ticker) |
| Initial 1 year of data | `engine/data.py` via yfinance (+ stdlib fallback), cached in `ai/cache/` |
| Build prediction engine | `engine/models.py` — ensemble of Ridge + kNN + Decision Tree (pure Python) |
| Predict next week | `engine/backtest.py::_chained_week_predict` — 5 trading days chained day-by-day |
| Analyse deviation | weekly MAPE / RMSE / bias / direction accuracy per simulated week |
| Fine-tune or change the model | inverse-error re-weighting of ensemble members every week + recency-weighted refit |
| Add next week to training data | walk-forward window grows; live loop appends real bars in `ai/cache/` |
| Repeat month by month / recursively | full history is consumed block-by-block, then continues forever in live mode |
| Predict the actual next day | `ai/<SYM>_prediction.json` — regenerated nightly, shown on the dashboard |
| Self-learning & self-improving | see below |

### How "self-improving" works concretely
1. **Bootstrap (compressed recursion):** from 1y of history, the first ~70% trains the engine. It then predicts the next 5 trading days *chained* (each forecast feeds the next), compares with what really happened, computes each model's error share, and **re-weights the ensemble toward the models that were right**. That week is folded into training and the loop repeats until today. This is your recursive cycle, replayed at machine speed (~5s per symbol).
2. **Live daily cycle (`Daily Learn & Predict`, cron 21:30 UTC Mon–Fri):**
   grades yesterday's published forecast against the real close → appends the new bar → refits with **recency-decayed sample weights (0.98^age)** → applies a **bias correction learned from its own recent graded errors** → re-runs the adaptive weight logic periodically → emits a fresh next-day prediction → commits everything back → Pages redeploy.
3. **Memory = git.** Every commit under `ai/` is a snapshot of what the agent knew and believed. The dashboard's "learning curve" tab plots how its error evolved during recursion.

## Repo layout
```
.github/workflows/
  train-stock.yml     # bootstrap: add any ticker, runs the recursive learning
  daily-learn.yml     # nightly: grade → learn → predict → commit → publish
engine/               # data.py, features.py, models.py, backtest.py, daily.py
scripts/              # train.py, daily_cycle.py (CLIs used by workflows)
ai/                   # ← the agent's memory: cache/, *_model.json, *_state.json,
                      #    *_backtest.json, *_prediction.json (auto-committed)
web/                  # GitHub Pages site (index.html + generated data.js)
```

## Setup (once, ~2 minutes)
1. **Settings → Pages → Source: Deploy from a branch → `main` / `/web`** (the site needs no build step).
2. Workflows ship with the repo; the cron starts automatically. To add stocks: **Actions → Add / Retrain Stock → Run workflow → enter tickers**.
3. Optional repo variable `FULL_BACKTEST = true` to force the full walk-forward every day instead of every 4th cycle.

## Honest disclaimer
This is an ML/automation experiment, not financial advice. Daily returns are dominated by noise; even a genuinely self-improving ensemble will hover near coin-flip direction accuracy. The dashboard reports its own error metrics transparently — trust those numbers, not vibes.
