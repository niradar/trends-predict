"""Leakage-safe preprocessing of Google Trends frames.

Every transformer has ``fit(X_train)`` / ``transform(X)`` so it can be fitted inside a
training window and applied to later data. All transformations are *causal* (only use the
current and past values of a series) so a value at date t never depends on t+1.

Djorno, Santillana & Yang (2026) pipeline is available as ``djorno_pipeline()``:
    zero repair -> log -> smoothing -> detrending -> clustering of redundant queries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Transformer:
    def fit(self, X: pd.DataFrame) -> "Transformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class Pipeline(Transformer):
    def __init__(self, steps: list[Transformer]):
        self.steps = steps

    def fit(self, X):
        cur = X
        for s in self.steps:
            cur = s.fit_transform(cur)
        return self

    def transform(self, X):
        cur = X
        for s in self.steps:
            cur = s.transform(cur)
        return cur


class ZeroRepair(Transformer):
    """Google reports 0 for volumes below its privacy threshold. Replace isolated zeros inside
    an otherwise positive series with a causal interpolation (last positive value); drop
    columns whose zero share exceeds ``max_zero_share`` (fitted on training data)."""

    def __init__(self, max_zero_share: float = 0.5):
        self.max_zero_share = max_zero_share
        self.keep_: list[str] = []

    def fit(self, X):
        share = (X.fillna(0) <= 0).mean()
        self.keep_ = [c for c in X.columns if share[c] <= self.max_zero_share]
        return self

    def transform(self, X):
        Z = X[[c for c in self.keep_ if c in X.columns]].astype(float).copy()
        Z = Z.mask(Z <= 0).ffill().bfill()
        return Z


class Log1p(Transformer):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def transform(self, X):
        return np.log(X.astype(float) + self.delta)


class Smooth(Transformer):
    """Causal exponentially weighted smoothing (a spline is non-causal at the training edge,
    so we use EWMA whose half-life plays the same role as the spline's smoothing parameter)."""

    def __init__(self, halflife: float = 2.0):
        self.halflife = halflife

    def transform(self, X):
        if not self.halflife or self.halflife <= 0:
            return X
        return X.astype(float).ewm(halflife=self.halflife, adjust=True).mean()


class Detrend(Transformer):
    """Remove slow level shifts. ``method``:
      'rolling' - subtract trailing rolling mean over ``window`` periods (anomaly vs. recent past)
      'linear'  - fit a linear trend on the training window and extrapolate it
      'diff'    - first difference
      None      - no-op
    """

    def __init__(self, method: str | None = "rolling", window: int = 52):
        self.method = method
        self.window = window
        self.coef_: dict[str, tuple[float, float]] = {}
        self.t0_: pd.Timestamp | None = None

    def fit(self, X):
        if self.method == "linear":
            t = np.arange(len(X), dtype=float)
            self.t0_ = X.index[0]
            self.n_train_ = len(X)
            for c in X.columns:
                v = X[c].astype(float).values
                ok = ~np.isnan(v)
                if ok.sum() >= 3:
                    b, a = np.polyfit(t[ok], v[ok], 1)
                    self.coef_[c] = (a, b)
                else:
                    self.coef_[c] = (float(np.nanmean(v)) if ok.any() else 0.0, 0.0)
        return self

    def transform(self, X):
        Z = X.astype(float)
        if self.method is None:
            return Z
        if self.method == "rolling":
            base = Z.rolling(self.window, min_periods=max(2, self.window // 4)).mean().shift(1)
            return (Z - base).bfill()
        if self.method == "diff":
            return Z.diff().bfill()
        if self.method == "linear":
            # position relative to training start (works for any later index)
            pos = np.arange(len(Z), dtype=float)
            if self.t0_ is not None and Z.index[0] != self.t0_:
                pos = pos + (len(Z) and (Z.index.get_indexer([self.t0_])[0] * -1 if self.t0_ in Z.index else 0))
            out = Z.copy()
            for c in Z.columns:
                a, b = self.coef_.get(c, (0.0, 0.0))
                out[c] = Z[c] - (a + b * pos)
            return out
        raise ValueError(self.method)


class Deseasonalize(Transformer):
    """Subtract seasonal position means estimated on the training window (period = 12 or 52)."""

    def __init__(self, period: int = 52):
        self.period = period
        self.means_: pd.DataFrame | None = None

    @staticmethod
    def _pos(idx: pd.Index, period: int) -> np.ndarray:
        dt = idx.to_timestamp() if isinstance(idx, pd.PeriodIndex) else pd.DatetimeIndex(idx)
        if period == 12:
            return dt.month.values - 1
        if period in (52, 53):
            return np.minimum(dt.isocalendar().week.values.astype(int) - 1, 51)
        if period == 7:
            return dt.dayofweek.values
        return np.arange(len(idx)) % period

    def fit(self, X):
        pos = self._pos(X.index, self.period)
        self.means_ = X.astype(float).groupby(pos).mean()
        self.grand_ = X.astype(float).mean()
        return self

    def transform(self, X):
        pos = self._pos(X.index, self.period)
        m = self.means_.reindex(range(self.period)).fillna(self.grand_)
        seas = m.loc[pos].set_index(X.index)
        return X.astype(float) - seas + self.grand_


class Standardize(Transformer):
    def fit(self, X):
        Z = X.astype(float)
        self.mu_ = Z.mean()
        self.sd_ = Z.std(ddof=0).replace(0, 1.0)
        return self

    def transform(self, X):
        return (X.astype(float) - self.mu_) / self.sd_


class ClusterQueries(Transformer):
    """Hierarchical clustering of queries on correlation distance (Djorno et al.); each
    cluster is replaced by the mean of its standardised members. Reduces redundancy and
    the number of coefficients the forecasting model has to estimate."""

    def __init__(self, max_clusters: int | None = None, distance_threshold: float = 0.4):
        self.max_clusters = max_clusters
        self.distance_threshold = distance_threshold
        self.labels_: dict[str, int] = {}

    def fit(self, X):
        from scipy.cluster.hierarchy import linkage, fcluster
        Z = X.astype(float)
        cols = list(Z.columns)
        if len(cols) <= 2:
            self.labels_ = {c: i for i, c in enumerate(cols)}
            self.std_ = Standardize().fit(Z)
            return self
        corr = Z.corr().fillna(0).values
        dist = 1 - corr
        iu = np.triu_indices(len(cols), 1)
        link = linkage(dist[iu], method="average")
        if self.max_clusters:
            lab = fcluster(link, t=self.max_clusters, criterion="maxclust")
        else:
            lab = fcluster(link, t=self.distance_threshold, criterion="distance")
        self.labels_ = {c: int(l) for c, l in zip(cols, lab)}
        self.std_ = Standardize().fit(Z)
        return self

    def transform(self, X):
        Z = self.std_.transform(X[[c for c in self.labels_ if c in X.columns]])
        groups: dict[int, list[str]] = {}
        for c, l in self.labels_.items():
            if c in Z.columns:
                groups.setdefault(l, []).append(c)
        out = {}
        for l, members in sorted(groups.items()):
            name = members[0] if len(members) == 1 else f"cluster{l}[{members[0]}+{len(members)-1}]"
            out[name] = Z[members].mean(axis=1)
        return pd.DataFrame(out, index=X.index)

    def membership(self) -> dict[str, list[str]]:
        g: dict[int, list[str]] = {}
        for c, l in self.labels_.items():
            g.setdefault(l, []).append(c)
        return {(m[0] if len(m) == 1 else f"cluster{l}[{m[0]}+{len(m)-1}]"): m for l, m in sorted(g.items())}


# --------------------------------------------------------------------------------------
# ready-made pipelines
# --------------------------------------------------------------------------------------

def raw_pipeline() -> Pipeline:
    """Approach 'raw GT': only drop privacy-zero series and take logs (ARGO convention)."""
    return Pipeline([ZeroRepair(), Log1p()])


def djorno_pipeline(period: int = 52, halflife: float = 2.0, detrend: str | None = "rolling",
                    cluster: bool = True, distance_threshold: float = 0.4) -> Pipeline:
    """Zero repair -> log -> smooth -> detrend -> (cluster). Fit on the training window only."""
    steps: list[Transformer] = [ZeroRepair(), Log1p(), Smooth(halflife), Detrend(detrend, window=period)]
    if cluster:
        steps.append(ClusterQueries(distance_threshold=distance_threshold))
    else:
        steps.append(Standardize())
    return Pipeline(steps)
