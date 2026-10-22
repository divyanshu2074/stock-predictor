"""Walk-forward "recursive learning" simulation.

Exactly mirrors the requested cycle, compressed into one run so the system can
bootstrap from 1 year of history in minutes:

    take initial training window --> build prediction engine
    --> predict NEXT WEEK (5 trading days, day-by-day chained)
    --> compare with what actually happened (deviation analysis)
    --> fine-tune model weights / switch model mix
    --> add that week to training data
    --> predict the week after ... repeat until end of history.

Then it trains on ALL data and emits the live next-day forecast.
"""
import json
import math
import os
import statistics

from .data import load_bars, save_bars, append_new_bars
from .features import FEATURE_NAMES, build_features, build_target, atr
from .models import Ensemble

AI_DIR = os.environ.get("STOCKPRED_AI", "ai")


def ensure_data(symbol):
    """Load bars from cache; if missing, download 1y of history."""
    try:
        return load_bars(symbol)
    except FileNotFoundError:
        from .data import fetch_history
        bars = fetch_history(symbol, range_="1y", interval="1d")
        save_bars(symbol, bars)
        return bars


def _weekly_blocks(n_train_end, total, weeks=5):
    """Split remaining bars [n_train_end .. total) into weekly blocks."""
    blocks, i = [], n_train_end
    while i < total:
        j = min(i + weeks, total)
        blocks.append((i, j))
        i = j
    return blocks


def _chained_week_predict(ens, bars, start_idx, length):
    """Predict `length` future closes by chaining 1-day-ahead predictions,
    feeding each synthetic bar back in as if it were real (pure ex-ante)."""
    work = list(bars[: start_idx + 1])
    preds = []
    for step in range(length):
        x = build_features(work, len(work) - 1)
        r_hat = ens.predict(x)
        last_close = work[-1]["close"]
        c_hat = last_close * (1.0 + r_hat)
        # synthetic bar keeps momentum state alive for the chain
        synth = {"date": f"pred+{step + 1}", "open": last_close, "high": max(last_close, c_hat),
                 "low": min(last_close, c_hat), "close": c_hat, "volume": work[-1]["volume"]}
        work.append(synth)
        preds.append(c_hat)
    return preds


def walk_forward(symbol, init_days=None, weeks=5, retrain_every_week=True, verbose=False):
    bars = ensure_data(symbol)
    total = len(bars)
    # ~70% of history as the initial training window (user's "1 year" when we have 1y),
    # remainder is consumed week-by-week by the recursive learning loop.
    if init_days is None:
        init_days = max(140, int(total * 0.7))
    if total < init_days + 10:
        raise RuntimeError(f"Not enough history for {symbol}: {total} bars")

    n_train = init_days
    blocks = _weekly_blocks(n_train, total, weeks)
    history_log, per_model_err = [], {"ridge": [], "knn": [], "tree": [], "ens": []}
    ens = None

    for bi, (s, e) in enumerate(blocks):
        train_end = s  # exclusive index of last usable bar is s-1... we use bars[0..s-1]
        X, y = [], []
        warmup = 60
        for i in range(warmup, train_end):
            t = build_target(bars, i, horizon=1)
            if t is None:
                continue
            X.append(build_features(bars, i))
            y.append(t)
        if len(X) < 50:
            break

        # ---- build / fine-tune prediction engine ----
        if ens is None or not retrain_every_week:
            ens = Ensemble().fit(X, y)
        else:
            # incremental refit: fresh fit but keep ensemble weights from deviation analysis
            prev_w = dict(ens.weights)
            ens = Ensemble(weights=prev_w).fit(X, y)

        # ---- predict next week (chained) ----
        pred_closes = _chained_week_predict(ens, bars, train_end - 1, e - s)
        actual_closes = [bars[i]["close"] for i in range(s, e)]

        # ---- deviation analysis ----
        errs = [(p - a) for p, a in zip(pred_closes, actual_closes)]
        mape = sum(abs(p - a) / a for p, a in zip(pred_closes, actual_closes)) / len(errs)
        bias = statistics.mean(errs)
        rmse = math.sqrt(sum(x * x for x in errs) / len(errs))
        dir_hits = 0
        dir_tot = 0
        for i in range(s, e):
            prev_actual = bars[i - 1]["close"]
            a_move = bars[i]["close"] - prev_actual
            p_move = pred_closes[i - s] - (actual_closes[i - s - 1] if i - s - 1 >= 0
                                           else pred_closes[i - s - 1] if i > s else prev_actual)
            if abs(a_move) > 1e-9:
                dir_tot += 1
                if (a_move > 0) == (p_move > 0):
                    dir_hits += 1

        # per-model error on this week (single-step, using true features)
        raw_err = {m: [] for m in per_model_err if m != "ens"}
        for i in range(s, e):
            x = build_features(bars, i - 1)
            truth = bars[i]["close"] / bars[i - 1]["close"] - 1.0
            rp = ens.predict_raw(x)
            for m, v in rp.items():
                raw_err[m].append(abs(v - truth))
            per_model_err["ens"].append(abs(ens.predict(x) - truth))
        for m, es in raw_err.items():
            per_model_err[m].append(sum(es) / len(es))

        # ---- self-improvement: inverse-MAD weights + recency decay ----
        avg = {m: (sum(es) / len(es) if es else 1e-6) for m, es in per_model_err.items()}
        inv = {m: 1.0 / (avg[m] + 1e-9) for m in ("ridge", "knn", "tree")}
        tot = sum(inv.values())
        new_w = {m: inv[m] / tot for m in inv}
        # blend with old weights (learning rate on meta-parameters)
        old = ens.weights
        ens.weights = {m: 0.5 * old.get(m, 1 / 3) + 0.5 * new_w[m] for m in new_w}

        hist_entry = {
            "week": bi + 1,
            "start_date": bars[s]["date"],
            "end_date": bars[e - 1]["date"],
            "mape": round(mape, 5),
            "rmse": round(rmse, 4),
            "bias": round(bias, 4),
            "dir_acc": round(dir_hits / dir_tot, 3) if dir_tot else None,
            "weights": {m: round(w, 3) for m, w in ens.weights.items()},
            "pred_first": round(pred_closes[0], 2),
            "act_first": round(actual_closes[0], 2),
        }
        history_log.append(hist_entry)
        if verbose:
            print(json.dumps(hist_entry))

    # ---- final engine trained on everything, predict the ACTUAL next day ----
    X, y = [], []
    for i in range(60, total):
        t = build_target(bars, i, horizon=1)
        if t is not None:
            X.append(build_features(bars, i))
            y.append(t)
    final = Ensemble(weights=dict(ens.weights)).fit(X, y)
    x_last = build_features(bars, total - 1)
    raw_pred = final.predict_raw(x_last)
    ret_hat = final.predict(x_last)
    last_close = bars[-1]["close"]
    band = 3.0 * atr(bars) / last_close
    ret_hat = max(-band, min(band, ret_hat))
    next_day = {
        "as_of_date": bars[-1]["date"],
        "as_of_close": last_close,
        "predicted_return": round(ret_hat, 5),
        "predicted_close": round(last_close * (1 + ret_hat), 2),
        "confidence_band": [round(last_close * (1 - abs(band)), 2),
                            round(last_close * (1 + abs(band)), 2)],
        "model_votes": {m: round(v, 5) for m, v in raw_pred.items()},
        "direction": "UP" if ret_hat > 0 else "DOWN",
    }

    summary = {
        "symbol": symbol,
        "bars": total,
        "weeks_simulated": len(history_log),
        "mean_mape": round(statistics.mean([h["mape"] for h in history_log]), 5),
        "median_dir_acc": round(statistics.median([h["dir_acc"] for h in history_log
                                                   if h["dir_acc"] is not None]), 3),
        "final_weights": {m: round(w, 3) for m, w in final.weights.items()},
        "improvement": {
            "first_half_mape": round(statistics.mean([h["mape"] for h in history_log[:len(history_log)//2]]), 5),
            "last_half_mape": round(statistics.mean([h["mape"] for h in history_log[len(history_log)//2:]]), 5),
        },
        "next_day_prediction": next_day,
    }
    return {"summary": summary, "history": history_log, "model": final}


def persist(symbol, result):
    os.makedirs(AI_DIR, exist_ok=True)
    with open(os.path.join(AI_DIR, f"{symbol}_model.json"), "w") as f:
        json.dump(result["model"].to_json(), f)
    with open(os.path.join(AI_DIR, f"{symbol}_backtest.json"), "w") as f:
        json.dump({"summary": result["summary"], "history": result["history"]}, f, indent=1)
