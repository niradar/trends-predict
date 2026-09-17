"""Forecasting models in the *direct* multi-step form used throughout the GT literature.

At forecast origin t we know y up to t (or t-1 for a nowcast) and every GT value up to t
(Google Trends is available in real time). The design row for origin t contains
    AR block      y_{t}, y_{t-1}, ..., y_{t-p+1}            (h >= 1)   or   y_{t-1..t-p} (h = 0)
    GT block      x_{k,t}, x_{k,t-1}, ..., x_{k,t-L}         for every query/cluster k
    calendar      Fourier terms / seasonal dummies for the *target* date t+h (deterministic)
and the target is y_{t+h}. Any sklearn-style regressor can then be evaluated with a rolling
origin (see evaluate.py). Models that must not see GT (the benchmark) use the AR + calendar
columns only.

Available models
    ar            AR(p)+calendar OLS benchmark (Choi & Varian's "without Google" model)
    ols           ARX OLS (Choi & Varian augmented)
    ridge         ARX ridge (RidgeCV, time-series inner CV)
    lasso         ARX LASSO (ARGO; LassoCV with time-series inner CV)
    enet          ARX elastic net (Borup & Montes Schütte)
    rf            random forest on the same design (Borup & Montes Schütte)
    arlr          greedy forward selection by likelihood ratio with AICc stop (Rangarajan et al.)
    sarimax       statsmodels SARIMAX with lagged exogenous GT (Djorno et al. ARIMAX)
Plus ``midas_weights`` / ``midas_aggregate`` for mixed-frequency features (Bangwayo-Skeete).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------------------
# design matrix
# --------------------------------------------------------------------------------------

def fourier_terms(index: pd.Index, period: int, k: int) -> pd.DataFrame:
    """Deterministic seasonal terms for the dates in ``index`` (safe: no data involved)."""
    if k <= 0:
        return pd.DataFrame(index=index)
    dt = index.to_timestamp() if isinstance(index, pd.PeriodIndex) else pd.DatetimeIndex(index)
    if period == 12:
        pos = dt.month.values - 1
    elif period in (52, 53):
        pos = dt.isocalendar().week.values.astype(int) - 1
    elif period == 7:
        pos = dt.dayofweek.values
    else:
        pos = np.arange(len(index)) % period
    cols = {}
    for j in range(1, k + 1):
        cols[f"sin{j}"] = np.sin(2 * np.pi * j * pos / period)
        cols[f"cos{j}"] = np.cos(2 * np.pi * j * pos / period)
    return pd.DataFrame(cols, index=index)


@dataclass
class Design:
    Z: pd.DataFrame           # rows = origins; columns = features
    target: pd.Series         # y_{t+h} aligned to origin index
    target_date: pd.Series    # date of the target for each origin
    ar_cols: list[str]
    gt_cols: list[str]
    cal_cols: list[str]
    h: int
    p: int
    xlags: int

    @property
    def bench_cols(self) -> list[str]:
        return self.ar_cols + self.cal_cols

    @property
    def all_cols(self) -> list[str]:
        return self.ar_cols + self.gt_cols + self.cal_cols


def make_design(y: pd.Series, X: pd.DataFrame | None, h: int = 1, p: int = 3, xlags: int = 0,
                period: int | None = None, fourier_k: int = 0, y_known_at_origin: bool = True) -> Design:
    """Build the direct-forecast design.

    ``y_known_at_origin=False`` models publication lag: at origin t the latest known target
    is y_{t-1}. For a *nowcast* use h=0 and y_known_at_origin=False.
    """
    y = y.astype(float).sort_index()
    idx = y.index
    if X is not None:
        X = X.astype(float).reindex(idx)
    ar_shift0 = 0 if (y_known_at_origin and h >= 1) else 1
    cols = {}
    ar_cols = []
    for j in range(p):
        name = f"y_lag{ar_shift0 + j}"
        cols[name] = y.shift(ar_shift0 + j)
        ar_cols.append(name)
    gt_cols = []
    if X is not None:
        for c in X.columns:
            for l in range(xlags + 1):
                name = f"{c}__l{l}"
                cols[name] = X[c].shift(l)
                gt_cols.append(name)
    Z = pd.DataFrame(cols, index=idx)
    target = y.shift(-h)
    # target date for calendar terms
    if isinstance(idx, pd.PeriodIndex):
        tdate = pd.Series(idx.shift(h), index=idx)
    else:
        step = pd.infer_freq(idx) or (idx[1] - idx[0])
        tdate = pd.Series(idx + (h * (idx[1] - idx[0]) if not isinstance(step, str) else h * pd.tseries.frequencies.to_offset(step)), index=idx)
    cal_cols: list[str] = []
    if period and fourier_k:
        F = fourier_terms(pd.Index(tdate.values), period, fourier_k)
        F.index = idx
        Z = pd.concat([Z, F], axis=1)
        cal_cols = list(F.columns)
    ok = Z[ar_cols + gt_cols].notna().all(axis=1)
    Z = Z.loc[ok]
    return Design(Z=Z, target=target.loc[Z.index], target_date=tdate.loc[Z.index],
                  ar_cols=ar_cols, gt_cols=gt_cols, cal_cols=cal_cols, h=h, p=p, xlags=xlags)


# --------------------------------------------------------------------------------------
# regressors (sklearn-style fit/predict on numpy)
# --------------------------------------------------------------------------------------

def _ts_cv(n: int, k: int = 5):
    from sklearn.model_selection import TimeSeriesSplit
    k = max(2, min(k, n // 8 if n >= 16 else 2))
    return TimeSeriesSplit(n_splits=k)


class _OLS:
    def fit(self, Z, y):
        Zc = np.column_stack([np.ones(len(Z)), Z])
        self.coef_, *_ = np.linalg.lstsq(Zc, y, rcond=None)
        return self

    def predict(self, Z):
        Zc = np.column_stack([np.ones(len(Z)), Z])
        return Zc @ self.coef_


class _Scaled:
    """Wrap a sklearn regressor with in-window standardisation."""

    def __init__(self, est):
        self.est = est

    def fit(self, Z, y):
        self.mu_ = Z.mean(axis=0)
        self.sd_ = Z.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        self.est.fit((Z - self.mu_) / self.sd_, y)
        return self

    def predict(self, Z):
        return self.est.predict((Z - self.mu_) / self.sd_)

    @property
    def coef_(self):
        return getattr(self.est, "coef_", None)


class _ARLR:
    """Greedy forward selection: start from mandatory columns, add the candidate with the
    largest likelihood-ratio gain while AICc improves (Rangarajan, Mody & Marathe 2019)."""

    def __init__(self, n_mandatory: int, max_add: int = 10):
        self.n_mandatory = n_mandatory
        self.max_add = max_add

    @staticmethod
    def _aicc(rss, n, k):
        if n - k - 1 <= 0 or rss <= 0:
            return np.inf
        return n * np.log(rss / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)

    def fit(self, Z, y):
        n, m = Z.shape
        S = list(range(self.n_mandatory))
        cand = [j for j in range(m) if j not in S]

        def fit_rss(cols):
            Zc = np.column_stack([np.ones(n), Z[:, cols]]) if cols else np.ones((n, 1))
            b, *_ = np.linalg.lstsq(Zc, y, rcond=None)
            r = y - Zc @ b
            return float(r @ r), b

        best_rss, _ = fit_rss(S)
        best = self._aicc(best_rss, n, len(S) + 1)
        for _ in range(self.max_add):
            if not cand:
                break
            scores = []
            for j in cand:
                rss, _ = fit_rss(S + [j])
                scores.append((self._aicc(rss, n, len(S) + 2), j))
            a, j = min(scores)
            if a < best - 1e-9:
                best = a
                S.append(j)
                cand.remove(j)
            else:
                break
        self.selected_ = S
        _, self.coef_ = fit_rss(S)
        return self

    def predict(self, Z):
        Zc = np.column_stack([np.ones(len(Z)), Z[:, self.selected_]])
        return Zc @ self.coef_


def make_regressor(kind: str, n_mandatory: int = 0, random_state: int = 0):
    kind = kind.lower()
    if kind in ("ar", "ols"):
        return _OLS()
    if kind == "ridge":
        from sklearn.linear_model import RidgeCV
        return _Scaled(RidgeCV(alphas=np.logspace(-3, 3, 25)))
    if kind == "lasso":
        from sklearn.linear_model import LassoCV
        return _ScaledCV(lambda n: LassoCV(cv=_ts_cv(n), alphas=40, max_iter=20000, random_state=random_state))
    if kind == "enet":
        from sklearn.linear_model import ElasticNetCV
        return _ScaledCV(lambda n: ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 0.95], cv=_ts_cv(n), alphas=30,
                                                 max_iter=20000, random_state=random_state))
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                                     random_state=random_state, n_jobs=-1)
    if kind == "arlr":
        return _ARLR(n_mandatory=n_mandatory)
    raise ValueError(f"unknown model kind {kind!r}")


class _ScaledCV(_Scaled):
    """Like _Scaled but the estimator is built lazily so inner CV can depend on n."""

    def __init__(self, factory):
        self.factory = factory
        self.est = None

    def fit(self, Z, y):
        self.est = self.factory(len(Z))
        return super().fit(Z, y)


# --------------------------------------------------------------------------------------
# SARIMAX with lagged exogenous GT (Djorno et al. ARIMAX)
# --------------------------------------------------------------------------------------

def sarimax_forecast(y_train: pd.Series, X_train: pd.DataFrame | None, X_future: pd.DataFrame | None,
                     h: int, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)) -> float:
    """Fit SARIMAX on the training window and return the h-step-ahead point forecast.
    ``X_train`` rows are exogenous values aligned to y dates *after shifting by h* (so that the
    exogenous row for target date T is x_{T-h}, known at the origin); ``X_future`` is the
    h rows of exogenous values for the forecast dates (also already shifted)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(y_train.values, exog=None if X_train is None else X_train.values,
                      order=order, seasonal_order=seasonal_order, trend="c",
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False, maxiter=200)
        fc = res.forecast(steps=h, exog=None if X_future is None else X_future.values)
    return float(np.asarray(fc)[-1])


# --------------------------------------------------------------------------------------
# MIDAS helpers (mixed frequency: weekly GT -> monthly target)
# --------------------------------------------------------------------------------------

def midas_weights(J: int, theta1: float, theta2: float) -> np.ndarray:
    j = np.arange(J + 1, dtype=float)
    w = np.exp(theta1 * j + theta2 * j ** 2)
    return w / w.sum()


def midas_aggregate(x_high: pd.Series, low_index: pd.PeriodIndex, J: int, theta1: float, theta2: float) -> pd.Series:
    """For each low-frequency period end, weight the last J+1 high-frequency observations
    (most recent first) with exponential-Almon weights. Causal: uses only observations whose
    date is <= the end of the low-frequency period."""
    w = midas_weights(J, theta1, theta2)
    xs = x_high.astype(float).sort_index()
    out = {}
    for per in low_index:
        end = per.end_time
        hist = xs.loc[:end].values
        if len(hist) < J + 1:
            out[per] = np.nan
            continue
        out[per] = float(np.dot(w, hist[-(J + 1):][::-1]))
    return pd.Series(out)


MIDAS_THETA_GRID = [(0.0, 0.0), (-0.3, 0.0), (-0.7, 0.0), (0.3, -0.05), (0.0, -0.05), (-0.1, -0.02)]
