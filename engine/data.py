"""Data layer: fetch stock history with yfinance (primary) or the Yahoo chart API
(fallback, no deps), store it as JSON in the repo, and serve windows for training."""
import json
import os
import time
import urllib.parse
import urllib.request

CACHE_DIR = os.environ.get("STOCKPRED_CACHE", "ai/cache")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}


def _fetch_yfinance(symbol: str, period: str, interval: str):
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    bars = []
    for idx, row in df.iterrows():
        c = row["Close"]
        if c is None or c != c:  # NaN check without pandas import cost concerns
            continue
        bars.append({
            "date": idx.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(c), 4),
            "volume": int(row["Volume"] or 0),
        })
    return bars


def fetch_history(symbol: str, range_: str = "1y", interval: str = "1d", max_retries: int = 4):
    """Return list of daily bars: [{date, open, high, low, close, volume}, ...]"""
    # 1) yfinance handles Yahoo's auth/crumb dance for us
    try:
        bars = _fetch_yfinance(symbol, period=range_, interval=interval)
        if bars:
            return bars
    except Exception as e:  # noqa: BLE001
        print(f"yfinance fetch failed ({e}); falling back to raw chart API")
    # 2) stdlib fallback
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{urllib.parse.quote(symbol)}?range={range_}&interval={interval}")
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = json.loads(r.read().decode())
            result = payload["chart"]["result"][0]
            ts = result["timestamp"]
            q = result["indicators"]["quote"][0]
            bars = []
            for i, t in enumerate(ts):
                if q["close"][i] is None:
                    continue
                bars.append({
                    "date": time.strftime("%Y-%m-%d", time.gmtime(t)),
                    "open": round(q["open"][i], 4),
                    "high": round(q["high"][i], 4),
                    "low": round(q["low"][i], 4),
                    "close": round(q["close"][i], 4),
                    "volume": int(q["volume"][i] or 0),
                })
            if not bars:
                raise RuntimeError(f"No bars returned for {symbol}")
            return bars
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt * 5, 60))
    raise RuntimeError(f"Failed to fetch {symbol}: {last_err}")


def load_bars(symbol: str):
    path = os.path.join(CACHE_DIR, f"{symbol}_daily.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached data for {symbol}. Run init first.")
    with open(path) as f:
        return json.load(f)


def save_bars(symbol: str, bars):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f"{symbol}_daily.json"), "w") as f:
        json.dump(bars, f)


def append_new_bars(symbol: str):
    """Fetch recent data and append any bars newer than what we have. Returns count added."""
    bars = load_bars(symbol)
    last = bars[-1]["date"]
    fresh = fetch_history(symbol, range_="1mo", interval="1d")
    added = [b for b in fresh if b["date"] > last]
    if added:
        save_bars(symbol, bars + added)
    return len(added)
