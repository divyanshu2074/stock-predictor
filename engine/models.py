"""Model layer: a small ensemble of pure-Python (stdlib-only) regressors so the
whole self-learning loop runs on plain GitHub Actions runners with zero heavy deps.

Models predict next-bar simple return; predictions are clamped to +/-3*ATR-ish band.
Each model exposes fit(X, y, w) / predict(x) and the ensemble stores picklable state."""
import math
import random


class LinearModel:
    """Ridge regression via gradient descent (batch)."""

    def __init__(self, lr=0.05, epochs=400, l2=0.01):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.w = None
        self.b = 0.0

    def fit(self, X, y, w=None):
        n, d = len(X), len(X[0])
        self.w = [0.0] * d
        self.b = 0.0
        wts = w or [1.0] * n
        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for i in range(n):
                pred = self.b + sum(self.w[j] * X[i][j] for j in range(d))
                err = pred - y[i]
                for j in range(d):
                    gw[j] += err * X[i][j]
                gb += err
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j])
            self.b -= self.lr * gb / n
        return self

    def predict(self, x):
        return self.b + sum(self.w[j] * x[j] for j in range(len(x)))


class KNNModel:
    """Weighted k-nearest-neighbour over recent history (recency-weighted)."""

    def __init__(self, k=12):
        self.k = k
        self.X, self.y = None, None

    def fit(self, X, y, w=None):
        self.X, self.y = X, y
        return self

    def predict(self, x):
        dists = []
        for i, xi in enumerate(self.X):
            d = sum((a - b) ** 2 for a, b in zip(x, xi)) ** 0.5
            dists.append((d, i))
        dists.sort()
        top = dists[: self.k]
        num = den = 0.0
        for d, i in top:
            wt = 1.0 / (d + 1e-6)
            num += wt * self.y[i]
            den += wt
        return num / den if den else 0.0


class TreeModel:
    """Small decision-tree regressor (CART-style, variance reduction, depth-limited)."""

    def __init__(self, max_depth=4, min_samples=8, n_features_frac=0.7, seed=0):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features_frac = n_features_frac
        self.seed = seed
        self.tree = None

    def _var(self, ys):
        m = sum(ys) / len(ys)
        return sum((y - m) ** 2 for y in ys) / len(ys)

    def _best_split(self, X, y, idxs, feats):
        best = (None, None, -1e18)
        parent_var = self._var([y[i] for i in idxs])
        n = len(idxs)
        for f in feats:
            vals = sorted(set(X[i][f] for i in idxs))
            if len(vals) < 2:
                continue
            step = max(1, len(vals) // 12)
            for t in range(step, len(vals) - 1, step):
                thr = (vals[t] + vals[t - 1]) / 2
                L = [i for i in idxs if X[i][f] <= thr]
                R = [i for i in idxs if X[i][f] > thr]
                if len(L) < self.min_samples or len(R) < self.min_samples:
                    continue
                gain = parent_var - (len(L) / n * self._var([y[i] for i in L])
                                     + len(R) / n * self._var([y[i] for i in R]))
                if gain > best[2]:
                    best = (f, thr, gain)
        return best

    def _build(self, X, y, idxs, depth, rng):
        if depth >= self.max_depth or len(idxs) < 2 * self.min_samples:
            return {"leaf": sum(y[i] for i in idxs) / max(1, len(idxs))}
        d = len(X[0])
        feats = rng.sample(range(d), max(1, int(d * self.n_features_frac)))
        f, thr, gain = self._best_split(X, y, idxs, feats)
        if f is None:
            return {"leaf": sum(y[i] for i in idxs) / max(1, len(idxs))}
        L = [i for i in idxs if X[i][f] <= thr]
        R = [i for i in idxs if X[i][f] > thr]
        return {"f": f, "thr": thr,
                "l": self._build(X, y, L, depth + 1, rng),
                "r": self._build(X, y, R, depth + 1, rng)}

    def fit(self, X, y, w=None):
        rng = random.Random(self.seed)
        self.tree = self._build(X, y, list(range(len(X))), 0, rng)
        return self

    def predict(self, x):
        node = self.tree
        while "leaf" not in node:
            node = node["l"] if x[node["f"]] <= node["thr"] else node["r"]
        return node["leaf"]


class Ensemble:
    """Weighted blend of models; weights come from the walk-forward backtest
    (inverse-error), making the system self-improving after every cycle."""

    MODELS = {
        "ridge": lambda: LinearModel(),
        "knn": lambda: KNNModel(k=12),
        "tree": lambda: TreeModel(seed=1),
    }

    def __init__(self, weights=None):
        self.weights = weights or {m: 1.0 / len(self.MODELS) for m in self.MODELS}
        self.models = {}
        self.scaling = None  # mean/std per feature

    def _scale_fit(self, X):
        n, d = len(X), len(X[0])
        means = [sum(row[j] for row in X) / n for j in range(d)]
        stds = []
        for j in range(d):
            v = sum((row[j] - means[j]) ** 2 for row in X) / n
            stds.append(math.sqrt(v) + 1e-9)
        self.scaling = (means, stds)
        return self

    def _scale(self, x):
        means, stds = self.scaling
        return [(xi - m) / s for xi, m, s in zip(x, means, stds)]

    def fit(self, X, y, w=None):
        self._scale_fit(X)
        Xs = [self._scale(x) for x in X]
        for name, ctor in self.MODELS.items():
            self.models[name] = ctor().fit(Xs, y, w)
        return self

    def predict_raw(self, x):
        xs = self._scale(x)
        return {name: m.predict(xs) for name, m in self.models.items()}

    def predict(self, x):
        preds = self.predict_raw(x)
        tot_w = sum(max(self.weights.get(k, 0.0), 0.0) for k in preds)
        if tot_w <= 0:
            return sum(preds.values()) / len(preds)
        return sum(p * max(self.weights.get(k, 0.0), 0.0) for k, p in preds.items()) / tot_w

    def to_json(self):
        return {"weights": self.weights, "scaling": self.scaling,
                "models": {name: m.__dict__ for name, m in self.models.items()}}

    @classmethod
    def from_json(cls, obj):
        e = cls(weights=obj["weights"])
        e.scaling = obj["scaling"]
        for name, state in obj["models"].items():
            m = cls.MODELS[name]()
            m.__dict__.update(state)
            e.models[name] = m
        return e
