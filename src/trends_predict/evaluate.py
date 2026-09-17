"""Rolling-origin evaluation, metrics and forecast-comparison inference.

The protocol (docs/deep-research-report.md, "Evaluation Protocols"):
  * outer rolling origins, expanding or sliding window, refit at every origin;
  * preprocessing fitted inside the training window;
  * M0 seasonal-naive, M1 target-only AR, M2 raw-GT, M3 preprocessed-GT, M4 regularised;
  * MAE, RMSE, MASE, OOS R² vs the target-only benchmark, directional accuracy;
  * Diebold–Mariano (non-nested) and Clark–West (nested) tests with HAC variance;
  * rolling conformal intervals with coverage reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .models import Design, make_design, make_regressor, sarimax_forecast
from .preprocess import Transformer


# --------------------------------------------------------------------------------------
# rolling backtest
# --------------------------------------------------------------------------------------

@dataclass
class BacktestResult:
    name: str
    preds: pd.DataFrame   # index = target date; columns y_true, y_pred, origin
    h: int
    selected: list[dict] = field(default_factory=list)  # per-origin selected features (sparse models)

    @property
    def errors(self) -> pd.Series:
        return self.preds["y_true"] - self.preds["y_pred"]


def rolling_backtest(y: pd.Series, X: pd.DataFrame | None, *, h: int, model: str, use_gt: bool,
                     p: int = 3, xlags: int = 0, period: int | None = None, fourier_k: int = 0,
                     y_known_at_origin: bool = True, min_train: int = 36, window: int | None = None,
                     preprocess: Callable[[], Transformer] | None = None, step: int = 1,
                     name: str | None = None, max_origins: int | None = None,
                     transform: str | None = None) -> BacktestResult:
    """Direct-forecast rolling-origin backtest.

    For every origin t (from ``min_train`` on, every ``step``): fit ``preprocess`` on X[:t],
    transform X[:t] (train) and the origin row, build the design, fit ``model`` on rows whose
    target date <= t, predict y_{t+h}. ``window`` = sliding-window length (None = expanding).

    ``transform='diff'`` models the *change* from the last known value (y_{t+h} − y_base, with
    AR features in first differences) and converts predictions back to levels. Use it for
    persistent series (unemployment, prices) where a random walk is the natural benchmark.
    """
    y = y.astype(float).sort_index()
    n = len(y)
    origins = list(range(min_train, n - h, step))
    if max_origins and len(origins) > max_origins:
        origins = origins[-max_origins:]
    rows = []
    selected = []
    base_shift = 0 if (y_known_at_origin and h >= 1) else 1
    for t in origins:
        lo = 0 if window is None else max(0, t + 1 - window)
        X_win = None
        if X is not None and use_gt:
            Xw = X.reindex(y.index).iloc[lo:t + 1]
            if preprocess is not None:
                pp = preprocess()
                Xw = pp.fit_transform(Xw)
            X_win = Xw
        # design on the window incl. the origin row; the origin row's target is unknown -> it is
        # the prediction row. Training rows are those with target date <= t.
        y_ext = y.iloc[lo:t + 1 + h]  # includes the true future for target alignment only
        y_model = y_ext.diff() if transform == "diff" else y_ext
        d = make_design(y_model, X_win.reindex(y_ext.index) if X_win is not None else None, h=h, p=p,
                        xlags=xlags, period=period, fourier_k=fourier_k, y_known_at_origin=y_known_at_origin)
        base = y_ext.shift(base_shift)
        if transform == "diff":
            d.target = (y_ext.shift(-h) - base).loc[d.Z.index]
        cols = d.all_cols if use_gt and X_win is not None else d.bench_cols
        origin_label = y.index[t]
        if origin_label not in d.Z.index:
            continue
        # training rows: those whose target is already *known* at the origin. With publication
        # lag (y_known_at_origin=False, which includes every h=0 nowcast) the latest known value
        # is y_{t-1}, so rows with target date == origin (incl. the origin row itself, whose
        # target is the very value being nowcast) must not be trained on.
        last_known = origin_label if (y_known_at_origin and h >= 1) else y.index[t - 1]
        known = d.target.notna() & (d.target_date <= last_known)
        Ztr = d.Z.loc[known.values, cols]
        ytr = d.target.loc[Ztr.index]
        ok = ytr.notna() & Ztr.notna().all(axis=1)
        Ztr, ytr = Ztr.loc[ok], ytr.loc[ok]
        if len(Ztr) < max(8, len(cols) // 2 if model in ("ar", "ols") else 8):
            continue
        zq = d.Z.loc[[origin_label], cols]
        if zq.isna().any(axis=None):
            continue
        reg = make_regressor(model, n_mandatory=len(d.bench_cols) if model == "arlr" else 0)
        reg.fit(Ztr.values, ytr.values)
        pred = float(np.asarray(reg.predict(zq.values))[0])
        tdate = d.target_date.loc[origin_label]
        y_true = float(y_ext.shift(-h).loc[origin_label])
        if transform == "diff":
            pred = float(base.loc[origin_label]) + pred
        rows.append({"target_date": tdate, "origin": origin_label, "y_true": y_true, "y_pred": pred})
        coef = getattr(reg, "coef_", None)
        if model == "arlr":
            selected.append({"origin": str(origin_label), "features": [cols[j] for j in reg.selected_]})
        elif coef is not None and model in ("lasso", "enet"):
            nz = [c for c, b in zip(cols, np.ravel(coef)) if abs(b) > 1e-8]
            selected.append({"origin": str(origin_label), "features": nz})
    preds = pd.DataFrame(rows)
    if len(preds):
        preds = preds.set_index("target_date")
    return BacktestResult(name or f"{model}{'+GT' if use_gt else ''}", preds, h, selected)


def forecast_next(y: pd.Series, X: pd.DataFrame | None, *, h: int, model: str, use_gt: bool,
                  p: int = 3, xlags: int = 0, period: int | None = None, fourier_k: int = 0,
                  y_known_at_origin: bool = True, window: int | None = None,
                  preprocess: Callable[[], Transformer] | None = None, transform: str | None = None) -> dict:
    """Fit on everything known and produce the *real* forward forecast.

    h>=1: origin = last date with a known y; target = h periods later.
    h=0 (nowcast): origin = the latest date for which GT exists but y does not yet (publication
    lag); requires X to extend beyond y. If X does not extend beyond y, falls back to h=1.
    Returns {'origin','target_date','point','model','n_train','features'}.
    """
    y = y.astype(float).sort_index()
    if X is not None:
        X = X.astype(float).sort_index()
    last_known = y.dropna().index.max()
    x_ahead = X is not None and X.index.max() > last_known
    if h == 0 and not x_ahead:
        return forecast_next(y, X, h=1, model=model, use_gt=use_gt, p=p, xlags=xlags, period=period,
                             fourier_k=fourier_k, y_known_at_origin=True, window=window,
                             preprocess=preprocess, transform=transform) | {"note": "no unpublished period to nowcast; returned h=1 forecast"}
    if h == 0 or (not y_known_at_origin and x_ahead):
        # publication-lag origin: the first period for which GT exists but y does not yet.
        # h=0 -> nowcast that period; h>=1 with y_known_at_origin=False -> target = origin + h,
        # AR lags start at y_{origin-1} exactly as in the backtest.
        nxt = X.index[X.index > last_known][:1]
        idx = y.index.union(nxt)
        y_full = y.reindex(idx)
        origin_pos = len(y_full) - 1
        if h >= 1:
            if isinstance(idx, pd.PeriodIndex):
                fut = pd.period_range(nxt[0] + 1, periods=h, freq=idx.freq)
            else:
                step = idx[-1] - idx[-2]
                fut = pd.DatetimeIndex([nxt[0] + step * (i + 1) for i in range(h)])
            y_full = pd.concat([y_full, pd.Series(np.nan, index=fut)])
        y_ext = y_full
    else:
        last = y.dropna().index[-1]
        if isinstance(y.index, pd.PeriodIndex):
            fut = pd.period_range(last + 1, periods=h, freq=y.index.freq)
        else:
            step = y.index[-1] - y.index[-2]
            fut = pd.DatetimeIndex([last + step * (i + 1) for i in range(h)])
        y_ext = pd.concat([y, pd.Series(np.nan, index=fut)])
        origin_pos = len(y) - 1
    lo = 0 if window is None else max(0, origin_pos + 1 - window)
    y_ext = y_ext.iloc[lo:]
    origin_pos -= lo
    X_win = None
    if X is not None and use_gt:
        Xw = X.reindex(y_ext.index)
        Xw_known = Xw.iloc[:origin_pos + 1]
        if preprocess is not None:
            pp = preprocess()
            pp.fit(Xw_known.dropna(how="all"))
            Xw = pp.transform(Xw.ffill())
        X_win = Xw
    y_model = y_ext.diff() if transform == "diff" else y_ext
    d = make_design(y_model, X_win, h=h, p=p, xlags=xlags, period=period, fourier_k=fourier_k,
                    y_known_at_origin=y_known_at_origin)
    base_shift = 0 if (y_known_at_origin and h >= 1) else 1
    base = y_ext.shift(base_shift)
    if transform == "diff":
        d.target = (y_ext.shift(-h) - base).loc[d.Z.index]
    cols = d.all_cols if (use_gt and X_win is not None) else d.bench_cols
    origin_label = y_ext.index[origin_pos]
    if origin_label not in d.Z.index:
        raise ValueError("origin row has missing features (not enough GT history at the end?)")
    known = d.target.notna() & (d.target_date <= (y.dropna().index[-1]))
    Ztr = d.Z.loc[known.values, cols]
    ytr = d.target.loc[Ztr.index]
    ok = ytr.notna() & Ztr.notna().all(axis=1)
    Ztr, ytr = Ztr.loc[ok], ytr.loc[ok]
    reg = make_regressor(model, n_mandatory=len(d.bench_cols) if model == "arlr" else 0)
    reg.fit(Ztr.values, ytr.values)
    zq = d.Z.loc[[origin_label], cols]
    pred = float(np.asarray(reg.predict(zq.values))[0])
    if transform == "diff":
        pred = float(base.loc[origin_label]) + pred
    feats = None
    coef = getattr(reg, "coef_", None)
    if model == "arlr":
        feats = [cols[j] for j in reg.selected_]
    elif coef is not None and model in ("lasso", "enet", "ols", "ridge"):
        c = np.ravel(coef)
        if len(c) == len(cols) + 1:  # OLS has intercept first
            c = c[1:]
        feats = sorted([(cols[i], float(c[i])) for i in range(len(cols)) if abs(c[i]) > 1e-8], key=lambda t: -abs(t[1]))[:12]
    return {"origin": str(origin_label), "target_date": str(d.target_date.loc[origin_label]), "point": pred,
            "model": model, "n_train": int(len(Ztr)), "features": feats}


def rolling_sarimax(y: pd.Series, X: pd.DataFrame | None, *, h: int, use_gt: bool, order=(1, 0, 0),
                    seasonal_order=(0, 0, 0, 0), min_train: int = 36, window: int | None = None,
                    preprocess: Callable[[], Transformer] | None = None, step: int = 1,
                    name: str | None = None, max_origins: int | None = None,
                    y_known_at_origin: bool = True) -> BacktestResult:
    """Rolling SARIMAX with exogenous GT shifted by h (x_{T-h} explains y_T).

    ``y_known_at_origin=False`` (publication lag, h>=1): y_t is not yet known at origin t, so the
    model is fitted through t-1 and forecasts h+1 steps; the exogenous rows for targets t..t+h
    are x_{t-h}..x_t, all known at the origin. Same information set as ``rolling_backtest``
    with ``y_known_at_origin=False``. h=0 already behaves this way."""
    y = y.astype(float).sort_index()
    n = len(y)
    origins = list(range(min_train, n - h, step))
    if max_origins and len(origins) > max_origins:
        origins = origins[-max_origins:]
    rows = []
    for t in origins:
        lo = 0 if window is None else max(0, t + 1 - window)
        y_win = y.iloc[lo:t + 1]
        Xtr = Xfu = None
        if use_gt and X is not None:
            Xa = X.reindex(y.index).astype(float)
            Xw = Xa.iloc[lo:t + 1]
            if preprocess is not None:
                Xw = preprocess().fit_transform(Xw)
            Xs = Xw.shift(h)  # exog for target date T is x_{T-h}
            Xtr = Xs.iloc[h:] if h > 0 else Xs
            y_fit = y_win.iloc[h:] if h > 0 else y_win
            if h > 0:
                # future exog rows: x at origin-h+1 .. origin -> targets t+1..t+h
                Xfu = Xw.iloc[-h:]
            else:
                Xfu = None
            if Xtr.isna().any(axis=None):
                Xtr = Xtr.bfill().ffill()
        else:
            y_fit = y_win
        try:
            if h == 0:
                # nowcast with SARIMAX = one-step forecast from t-1 using exog at t
                pred = sarimax_forecast(y_fit.iloc[:-1], None if Xtr is None else Xtr.iloc[:-1],
                                        None if Xtr is None else Xtr.iloc[[-1]], 1, order, seasonal_order)
                tdate = y.index[t]
            elif y_known_at_origin:
                pred = sarimax_forecast(y_fit, Xtr, Xfu, h, order, seasonal_order)
                tdate = y.index[t + h]
            else:
                # publication lag: fit through t-1, forecast t..t+h (h+1 steps), keep the last
                Xfu_lag = None if Xtr is None else pd.concat([Xtr.iloc[[-1]], Xfu])
                pred = sarimax_forecast(y_fit.iloc[:-1], None if Xtr is None else Xtr.iloc[:-1],
                                        Xfu_lag, h + 1, order, seasonal_order)
                tdate = y.index[t + h]
        except Exception:
            continue
        rows.append({"target_date": tdate, "origin": y.index[t], "y_true": float(y.loc[tdate]), "y_pred": pred})
    preds = pd.DataFrame(rows).set_index("target_date") if rows else pd.DataFrame(columns=["origin", "y_true", "y_pred"])
    return BacktestResult(name or f"sarimax{'+GT' if use_gt else ''}", preds, h)


def seasonal_naive(y: pd.Series, h: int, period: int, min_train: int) -> BacktestResult:
    y = y.astype(float).sort_index()
    rows = []
    for t in range(min_train, len(y) - h):
        tdate = y.index[t + h]
        lag = period if h <= period else period * int(np.ceil(h / period))
        src = t + h - lag
        if src < 0:
            continue
        rows.append({"target_date": tdate, "origin": y.index[t], "y_true": y.iloc[t + h], "y_pred": y.iloc[src]})
    return BacktestResult("seasonal_naive", pd.DataFrame(rows).set_index("target_date"), h)


def _last_known_pos(t: int, h: int, y_known_at_origin: bool) -> int:
    """Position of the latest target value known at origin t (publication lag => t-1)."""
    return t if (y_known_at_origin and h >= 1) else t - 1


def drift(y: pd.Series, h: int, min_train: int, y_known_at_origin: bool = True) -> BacktestResult:
    """Random walk with drift: last known value + steps × expanding mean change. The right
    benchmark for prices/returns (equity drift is positive) — a zero-change naive lets a GT
    model 'win' merely by having an intercept (see experiments/03-Q3-A). Under publication lag
    (y_known_at_origin=False) the last known value is y_{t-1}, as for the GT models."""
    y = y.astype(float).sort_index()
    rows = []
    for t in range(min_train, len(y) - h):
        last_pos = _last_known_pos(t, h, y_known_at_origin)
        hist = y.iloc[:last_pos + 1]
        mu = float(hist.diff().dropna().mean())
        steps = (t + h) - last_pos
        rows.append({"target_date": y.index[t + h], "origin": y.index[t], "y_true": y.iloc[t + h],
                     "y_pred": float(y.iloc[last_pos]) + steps * mu})
    return BacktestResult("drift", pd.DataFrame(rows).set_index("target_date"), h)


def naive(y: pd.Series, h: int, min_train: int, y_known_at_origin: bool = True) -> BacktestResult:
    """Last *known* value (y_t, or y_{t-1} under publication lag) carried forward."""
    y = y.astype(float).sort_index()
    rows = [{"target_date": y.index[t + h], "origin": y.index[t], "y_true": y.iloc[t + h],
             "y_pred": y.iloc[_last_known_pos(t, h, y_known_at_origin)]}
            for t in range(min_train, len(y) - h)]
    return BacktestResult("naive", pd.DataFrame(rows).set_index("target_date"), h)


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------

def mae(e): return float(np.mean(np.abs(e)))
def rmse(e): return float(np.sqrt(np.mean(np.square(e))))


def mape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.abs(y) > 1e-12
    return float(100 * np.mean(np.abs((y[ok] - yhat[ok]) / y[ok]))) if ok.any() else np.nan


def mase(y_true, y_pred, y_hist: pd.Series, period: int):
    d = np.abs(np.asarray(y_hist[period:], float) - np.asarray(y_hist[:-period], float)).mean()
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / d) if d > 0 else np.nan


def oos_r2(y_true, pred_model, pred_bench):
    y, a, b = (np.asarray(v, float) for v in (y_true, pred_model, pred_bench))
    den = np.sum((y - b) ** 2)
    return float(1 - np.sum((y - a) ** 2) / den) if den > 0 else np.nan


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series, y_prev: pd.Series) -> float:
    s_true = np.sign(np.asarray(y_true) - np.asarray(y_prev))
    s_pred = np.sign(np.round(np.asarray(y_pred) - np.asarray(y_prev), 10))
    ok = s_true != 0
    if not ok.any() or not (s_pred != 0).any():
        return np.nan  # a no-change forecast has no direction
    return float(np.mean(s_true[ok] == s_pred[ok]))


def _hac_var(d: np.ndarray, lag: int) -> float:
    n = len(d)
    d = d - d.mean()
    v = np.sum(d * d) / n
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        v += 2 * w * np.sum(d[l:] * d[:-l]) / n
    return max(v, 1e-18) / n


def diebold_mariano(e_bench: np.ndarray, e_model: np.ndarray, h: int = 1, loss: str = "sq") -> dict:
    """H0: equal accuracy. Positive statistic favours the model (lower loss than benchmark)."""
    e0, e1 = np.asarray(e_bench, float), np.asarray(e_model, float)
    L = (lambda e: e ** 2) if loss == "sq" else np.abs
    d = L(e0) - L(e1)
    n = len(d)
    if n < 8:
        return {"stat": np.nan, "p_value": np.nan, "n": n}
    var = _hac_var(d, max(h - 1, 0))
    stat = d.mean() / np.sqrt(var)
    # Harvey–Leybourne–Newbold small-sample correction
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return {"stat": float(stat), "p_value": float(p), "n": n, "mean_loss_diff": float(d.mean())}


def clark_west(y: np.ndarray, f_bench: np.ndarray, f_model: np.ndarray, h: int = 1) -> dict:
    """Clark & West (2007) for nested models. Positive => the larger (GT) model helps."""
    y, f0, f1 = (np.asarray(v, float) for v in (y, f_bench, f_model))
    d = (y - f0) ** 2 - ((y - f1) ** 2 - (f0 - f1) ** 2)
    n = len(d)
    if n < 8:
        return {"stat": np.nan, "p_value": np.nan, "n": n}
    var = _hac_var(d, max(h - 1, 0))
    stat = d.mean() / np.sqrt(var)
    p = 1 - stats.norm.cdf(stat)  # one-sided
    return {"stat": float(stat), "p_value": float(p), "n": n}


def conformal_interval(residuals: np.ndarray, alpha: float = 0.1, recent: int | None = None) -> float:
    """Half-width q such that [ŷ−q, ŷ+q] has ~(1−α) empirical coverage on past residuals."""
    r = np.abs(np.asarray(residuals, float))
    r = r[~np.isnan(r)]
    if recent:
        r = r[-recent:]
    if len(r) == 0:
        return np.nan
    k = int(np.ceil((1 - alpha) * (len(r) + 1))) - 1
    k = min(max(k, 0), len(r) - 1)
    return float(np.sort(r)[k])


def coverage(y_true, lo, hi) -> float:
    y, lo, hi = (np.asarray(v, float) for v in (y_true, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


# --------------------------------------------------------------------------------------
# comparison table
# --------------------------------------------------------------------------------------

def compare(results: dict[str, BacktestResult], bench: str, y_hist: pd.Series | None = None,
            period: int | None = None, exclude: tuple[str, str] | None = None) -> pd.DataFrame:
    """Align all results on common target dates and score them against ``bench``.

    ``exclude=('2020-03-01','2020-12-31')`` drops target dates in a range (e.g. the pandemic
    shock) for a robustness view; report both with and without.
    """
    common = None
    for r in results.values():
        idx = r.preds.index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()
    if exclude:
        lo, hi = pd.Timestamp(exclude[0]), pd.Timestamp(exclude[1])
        ts = common.to_timestamp() if isinstance(common, pd.PeriodIndex) else pd.DatetimeIndex(common)
        common = common[(ts < lo) | (ts > hi)]
    b = results[bench].preds.loc[common]
    rows = []
    for name, r in results.items():
        p = r.preds.loc[common]
        e = (p["y_true"] - p["y_pred"]).values
        eb = (b["y_true"] - b["y_pred"]).values
        row = {"model": name, "n": len(common), "MAE": mae(e), "RMSE": rmse(e), "MAPE": mape(p["y_true"], p["y_pred"]),
               "OOS_R2_vs_bench": oos_r2(p["y_true"], p["y_pred"], b["y_pred"])}
        if y_hist is not None and period:
            row["MASE"] = mase(p["y_true"], p["y_pred"], y_hist, period)
        # direction relative to the last known value at the origin
        try:
            y_prev = y_hist.shift(1).reindex(common) if (y_hist is not None and r.h == 0) else \
                y_hist.reindex(pd.Index(p["origin"].values)).values if y_hist is not None else None
            if y_prev is not None:
                row["DirAcc"] = directional_accuracy(p["y_true"].values, p["y_pred"].values, np.asarray(y_prev, float))
        except Exception:
            pass
        if name != bench:
            dm = diebold_mariano(eb, e, h=max(r.h, 1))
            cw = clark_west(p["y_true"].values, b["y_pred"].values, p["y_pred"].values, h=max(r.h, 1))
            row.update({"DM_stat": dm["stat"], "DM_p": dm["p_value"], "CW_stat": cw["stat"], "CW_p": cw["p_value"]})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def verdict(table: pd.DataFrame, bench: str, gt_models: list[str]) -> dict:
    """Plain-language decision: does GT add stable, significant value?"""
    best = None
    for m in gt_models:
        if m not in table.index:
            continue
        r2 = table.loc[m, "OOS_R2_vs_bench"]
        if best is None or r2 > best[1]:
            best = (m, r2)
    if best is None:
        return {"gt_helps": False, "reason": "no GT model evaluated"}
    m, r2 = best
    cw_p = table.loc[m].get("CW_p", np.nan)
    dm_p = table.loc[m].get("DM_p", np.nan)
    helps = bool(r2 > 0.02 and (cw_p < 0.10 or dm_p < 0.10))
    # 'strong' needs a sizeable gain AND both tests (CW alone is satisfied by models that lose
    # to the benchmark in MSE; DM alone can be driven by a single shock month)
    strong = bool(r2 > 0.15 and cw_p < 0.05 and dm_p < 0.10)
    strength = "strong" if strong else "moderate" if helps else "none"
    dm_stat = table.loc[m].get("DM_stat", np.nan)
    dm_side = ("DM favours the GT model" if dm_stat > 0 else "DM favours the benchmark") if dm_stat == dm_stat else "DM n/a"
    return {"gt_helps": helps, "best_model": m, "oos_r2": float(r2), "cw_p": float(cw_p) if cw_p == cw_p else None,
            "dm_p": float(dm_p) if dm_p == dm_p else None, "strength": strength,
            "reason": (f"{m} lowers squared error by {100*r2:.0f}% vs {bench} (Clark–West p={cw_p:.3f}; {dm_side}, p={dm_p:.2f})"
                       if helps else f"best GT model {m} has OOS R²={r2:.2f} vs {bench} (Clark–West p={cw_p:.2f}; {dm_side}, p={dm_p:.2f}); not a reliable gain")}
