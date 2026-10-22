"""Daily live cycle:
 1. grade yesterday's forecast against the real close (deviation analysis)
 2. append new bars to training cache
 3. retrain/refit the ensemble with recency-weighted samples
 4. re-run a compact walk-forward so meta-weights keep adapting (self-improving)
 5. emit the NEW next-day prediction + updated metrics
"""
import json
import os
import statistics

from .data import load_bars, append_new_bars
from .features import build_features, build_target, atr
from .models import Ensemble
from .backtest import walk_forward, persist, AI_DIR


def _load_state(symbol):
    p = os.path.join(AI_DIR, f"{symbol}_state.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"pending": None, "grades": [], "cycles": 0}


def _save_state(symbol, st):
    os.makedirs(AI_DIR, exist_ok=True)
    with open(os.path.join(AI_DIR, f"{symbol}_state.json"), "w") as f:
        json.dump(st, f, indent=1)


def run_cycle(symbol, full_backtest=False):
    added = append_new_bars(symbol)
    bars = load_bars(symbol)
    st = _load_state(symbol)
    last_close = bars[-1]["close"]
    last_date = bars[-1]["date"]

    # ---- 1. grade pending prediction vs reality ----
    graded = None
    if st["pending"]:
        pend = st["pending"]
        # find the actual bar for the predicted date
        actual = next((b for b in reversed(bars) if b["date"] == pend["target_date"]), None)
        if actual is not None:
            err_abs = actual["close"] - pend["predicted_close"]
            err_pct = err_abs / pend["predicted_close"]
            dir_ok = (pend["direction"] == "UP") == (actual["close"] > pend["as_of_close"])
            graded = {
                "cycle": st["cycles"],
                "target_date": pend["target_date"],
                "predicted": pend["predicted_close"],
                "actual": actual["close"],
                "err_pct": round(err_pct, 5),
                "direction_correct": dir_ok,
            }
            st["grades"].append(graded)
            st["grades"] = st["grades"][-60:]
            st["pending"] = None

    # ---- bias correction factor learned from recent grades ----
    corr = 0.0
    recent = [g["err_pct"] for g in st["grades"][-10:]]
    if len(recent) >= 3:
        mean_err = statistics.mean(recent)
        # shrink correction: if we systematically over-predict by x%, pull back
        corr = max(-0.01, min(0.01, -mean_err))

    # ---- 2/3/4. retrain (compact or full walk-forward) ----
    if full_backtest or st["cycles"] % 4 == 0:
        result = walk_forward(symbol)
        persist(symbol, result)
        model = result["model"]
        summary = result["summary"]
    else:
        with open(os.path.join(AI_DIR, f"{symbol}_model.json")) as f:
            model = Ensemble.from_json(json.load(f))
        X, y, w = [], [], []
        for i in range(60, len(bars)):
            t = build_target(bars, i, horizon=1)
            if t is not None:
                X.append(build_features(bars, i))
                y.append(t)
                age = len(bars) - i
                w.append(0.98 ** age)  # recency weighting = online fine-tuning
        model.fit(X, y, w=w)
        bt_path = os.path.join(AI_DIR, f"{symbol}_backtest.json")
        summary = json.load(open(bt_path))["summary"] if os.path.exists(bt_path) else {}

    # ---- 5. new next-day prediction ----
    x_last = build_features(bars, len(bars) - 1)
    ret_hat = model.predict(x_last) + corr
    band = 3.0 * atr(bars) / last_close
    ret_hat = max(-band, min(band, ret_hat))
    pred = {
        "as_of_date": last_date,
        "as_of_close": last_close,
        "target_date": None,  # filled by workflow via trading-calendar guess
        "predicted_return": round(ret_hat, 5),
        "predicted_close": round(last_close * (1 + ret_hat), 2),
        "confidence_band": [round(last_close * (1 - abs(band)), 2),
                            round(last_close * (1 + abs(band)), 2)],
        "bias_correction_applied": round(corr, 5),
        "direction": "UP" if ret_hat > 0 else "DOWN",
        "model_votes": {m: round(v, 5) for m, v in model.predict_raw(x_last).items()},
    }
    st["cycles"] += 1
    st["pending"] = pred
    _save_state(symbol, st)

    # persist refreshed artifacts
    os.makedirs(AI_DIR, exist_ok=True)
    with open(os.path.join(AI_DIR, f"{symbol}_model.json"), "w") as f:
        json.dump(model.to_json(), f)
    with open(os.path.join(AI_DIR, f"{symbol}_prediction.json"), "w") as f:
        json.dump(pred, f, indent=1)

    grades_summary = {}
    if st["grades"]:
        accs = [g["direction_correct"] for g in st["grades"]]
        errs = [abs(g["err_pct"]) for g in st["grades"]]
        grades_summary = {
            "graded_cycles": len(accs),
            "live_direction_accuracy": round(sum(accs) / len(accs), 3),
            "mean_abs_error_pct": round(statistics.mean(errs), 5),
            "last_grade": st["grades"][-1],
        }
    out = {"symbol": symbol, "new_bars": added, "prediction": pred,
           "live_grades": grades_summary, "backtest_summary": summary}
    print(json.dumps(out, indent=1))
    return out
