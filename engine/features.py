"""Feature engineering shared by backtest, retrain and predict paths."""
import math


def sma(vals, n):
    vals = vals[-n:] if len(vals) >= n else vals
    return sum(vals) / len(vals)


def ema(vals, n):
    k = 2.0 / (n + 1)
    out = vals[0]
    for v in vals[1:]:
        out = v * k + out * (1 - k)
    return out


def rsi(closes, n=14):
    if len(closes) < n + 1:
        return 50.0
    gains, losses = [], []
    for i in range(len(closes) - n, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag, al = sum(gains) / n, sum(losses) / n
    if al == 0:
        return 100.0
    rs = ag / al
    return 100.0 - 100.0 / (1.0 + rs)


def _ema_series(vals, n):
    k = 2.0 / (n + 1)
    out, e = [], vals[0]
    for v in vals:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def macd_line(closes):
    if len(closes) < 26:
        return 0.0
    return ema(closes, 12) - ema(closes, 26)


def macd_signal(closes):
    if len(closes) < 34:
        return 0.0
    return ema(macd_series(closes), 9)


def macd_series(closes):
    e12, e26 = _ema_series(closes, 12), _ema_series(closes, 26)
    return [a - b for a, b in zip(e12, e26)]


def macd_histogram(closes):
    if len(closes) < 34:
        return 0.0
    ms = macd_series(closes)
    sig = _ema_series(ms, 9)
    return ms[-1] - sig[-1]


def atr(bars, n=14):
    if len(bars) < 2:
        return 0.0
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    trs = trs[-n:]
    return sum(trs) / len(trs)


FEATURE_NAMES = [
    "mom_5", "mom_10", "mom_21",          # % change over lookbacks
    "sma_ratio_5", "sma_ratio_20", "sma_ratio_50",  # close/sma - 1
    "rsi14", "macd_hist", "atr_norm",      # momentum / trend / volatility
    "vol_ratio", "gap", "intraday",        # volume behaviour
    "day_of_week_sin", "day_of_week_cos",
]


def build_features(bars, idx):
    """Features computable from bars[0..idx] (inclusive of bar idx's own OHLC)."""
    closes = [b["close"] for b in bars[: idx + 1]]
    c = closes[-1]
    feats = {}
    for n, key in [(5, "mom_5"), (10, "mom_10"), (21, "mom_21")]:
        feats[key] = (c / closes[-min(n + 1, len(closes))] - 1.0) if len(closes) > 1 else 0.0
    for n in (5, 20, 50):
        s = sma(closes, n)
        feats[f"sma_ratio_{n}"] = c / s - 1.0 if s else 0.0
    feats["rsi14"] = rsi(closes) / 100.0
    feats["macd_hist"] = macd_histogram(closes) / c if c else 0.0
    feats["atr_norm"] = atr(bars[: idx + 1]) / c if c else 0.0
    vols = [b["volume"] for b in bars[: idx + 1]]
    av = sma(vols, 20)
    feats["vol_ratio"] = vols[-1] / av - 1.0 if av else 0.0
    o = bars[idx]["open"]
    feats["gap"] = c / o - 1.0 if o else 0.0
    prev_c = closes[-2] if len(closes) > 1 else o
    feats["intraday"] = o / prev_c - 1.0 if prev_c else 0.0
    import datetime
    try:
        dow = datetime.date.fromisoformat(bars[idx]["date"]).weekday()
    except ValueError:  # synthetic forecast bar inside a chained loop
        dow = (datetime.date.fromisoformat(
            next(b["date"] for b in reversed(bars[:idx]) if not b["date"].startswith("pred"))
        ).weekday() + 1) % 5
    feats["day_of_week_sin"] = math.sin(2 * math.pi * dow / 5.0)
    feats["day_of_week_cos"] = math.cos(2 * math.pi * dow / 5.0)
    return [feats[k] for k in FEATURE_NAMES]


def build_target(bars, idx, horizon=1):
    """Return next-bar return (close[idx+horizon]/close[idx]-1) or None."""
    j = idx + horizon
    if j >= len(bars):
        return None
    return bars[j]["close"] / bars[idx]["close"] - 1.0
