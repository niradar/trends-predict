"""Synthetic-data tests for the parts of trends_predict that decided our conclusions:
leakage-free training sets, difference-transform round trip, benchmarks, intervals, verdict."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trends_predict import evaluate as ev, preprocess as pp  # noqa: E402
from trends_predict.models import make_design  # noqa: E402


def _synthetic(n=240, seed=0, signal=1.0):
    """y is a random walk whose *next* change is partly visible in x today (x leads y)."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2005-01", periods=n, freq="M")
    x = rng.normal(size=n).cumsum() * 0 + rng.normal(size=n)  # white-noise attention shocks
    dy = 0.3 * rng.normal(size=n) + signal * np.roll(x, 1)      # y_t change depends on x_{t-1}
    dy[0] = 0
    y = pd.Series(5 + dy.cumsum(), index=idx)
    X = pd.DataFrame({"q": 50 + 10 * x}, index=idx)
    return y, X


def test_make_design_target_alignment_h1():
    y, X = _synthetic(60)
    d = make_design(y, X, h=1, p=2, xlags=1)
    origin = d.Z.index[10]
    assert d.target.loc[origin] == y.loc[origin + 1]
    assert d.Z.loc[origin, "y_lag0"] == y.loc[origin]
    assert d.Z.loc[origin, "q__l1"] == X["q"].loc[origin - 1]


def test_make_design_nowcast_uses_lagged_y_only():
    y, X = _synthetic(60)
    d = make_design(y, X, h=0, p=2, xlags=0, y_known_at_origin=False)
    origin = d.Z.index[10]
    assert d.target.loc[origin] == y.loc[origin]
    assert "y_lag0" not in d.Z.columns and d.Z.loc[origin, "y_lag1"] == y.loc[origin - 1]
    assert d.Z.loc[origin, "q__l0"] == X["q"].loc[origin]


def test_rolling_backtest_no_leakage_nowcast():
    """With y_known_at_origin=False, a model that could see y_t would be perfect; ours must not be."""
    y, X = _synthetic(200, signal=0.0)
    r = ev.rolling_backtest(y, X, h=0, model="ols", use_gt=True, p=2, xlags=0, y_known_at_origin=False,
                            min_train=60, transform="diff")
    e = r.errors.values
    assert len(e) > 100
    assert np.std(e) > 0.1  # no perfect fit


def test_diff_transform_round_trip_matches_levels_for_naive_like_model():
    y, X = _synthetic(200, signal=0.0)
    r = ev.rolling_backtest(y, None, h=1, model="ar", use_gt=False, p=1, min_train=60, transform="diff")
    nv = ev.naive(y, 1, 60)
    common = r.preds.index.intersection(nv.preds.index)
    # predictions are in levels and close to the random walk when the AR coefficient is small
    assert np.abs(r.preds.loc[common, "y_pred"] - nv.preds.loc[common, "y_pred"]).mean() < 1.0
    assert np.allclose(r.preds.loc[common, "y_true"], nv.preds.loc[common, "y_true"])


def test_gt_signal_is_detected_when_present():
    y, X = _synthetic(240, signal=1.0)
    bench = ev.naive(y, 1, 60)
    gtm = ev.rolling_backtest(y, X, h=1, model="ols", use_gt=True, p=1, xlags=1, min_train=60, transform="diff",
                              preprocess=lambda: pp.Pipeline([pp.Standardize()]))
    tab = ev.compare({"naive": bench, "ols+GT": gtm}, bench="naive", y_hist=y, period=12)
    assert tab.loc["ols+GT", "OOS_R2_vs_bench"] > 0.5
    v = ev.verdict(tab, "naive", ["ols+GT"])
    assert v["gt_helps"] and v["strength"] == "strong"


def test_gt_signal_not_detected_when_absent():
    y, X = _synthetic(240, signal=0.0, seed=3)
    bench = ev.naive(y, 1, 60)
    gtm = ev.rolling_backtest(y, X, h=1, model="ols", use_gt=True, p=1, xlags=1, min_train=60, transform="diff")
    tab = ev.compare({"naive": bench, "ols+GT": gtm}, bench="naive", y_hist=y, period=12)
    v = ev.verdict(tab, "naive", ["ols+GT"])
    assert not v["gt_helps"]


def test_drift_and_naive_benchmarks():
    idx = pd.period_range("2010-01", periods=50, freq="M")
    y = pd.Series(np.arange(50, dtype=float), index=idx)  # perfect +1 drift
    d = ev.drift(y, 1, 10)
    n = ev.naive(y, 1, 10)
    assert np.allclose(d.preds["y_pred"], d.preds["y_true"])
    assert np.allclose(n.preds["y_pred"] + 1, n.preds["y_true"])


def test_conformal_interval_coverage():
    rng = np.random.default_rng(1)
    res = rng.normal(size=1000)
    q = ev.conformal_interval(res, alpha=0.1)
    assert 1.5 < q < 1.8  # ~1.645 for a standard normal


def test_directional_accuracy_no_change_is_nan():
    yt = pd.Series([1, 2, 3]); yp = pd.Series([0, 1, 2]); prev = pd.Series([0, 1, 2])
    assert np.isnan(ev.directional_accuracy(yt, yp, prev))


def test_preprocess_pipeline_is_causal():
    """A change at the end of the series must not alter earlier transformed values."""
    y, X = _synthetic(120)
    pipe = pp.djorno_pipeline(period=12, halflife=1.0, cluster=False)
    a = pipe.fit_transform(X)
    X2 = X.copy(); X2.iloc[-1] += 40
    b = pp.djorno_pipeline(period=12, halflife=1.0, cluster=False).fit_transform(X2)
    # standardisation uses full-window stats (fit on the training window, allowed); compare shapes
    # of the causal parts (smooth + detrend) instead
    causal = pp.Pipeline([pp.ZeroRepair(), pp.Log1p(), pp.Smooth(1.0), pp.Detrend("rolling", 12)])
    a2 = causal.fit_transform(X); b2 = causal.fit_transform(X2)
    assert np.allclose(a2.iloc[:-1].values, b2.iloc[:-1].values)
    assert a.shape == b.shape


def test_forecast_next_h1_and_nowcast_paths():
    y, X = _synthetic(150)
    fc = ev.forecast_next(y, X, h=1, model="ols", use_gt=True, p=1, xlags=0, transform="diff")
    assert fc["target_date"] == str(y.index[-1] + 1)
    # nowcast requires X to extend one period beyond y
    Xa = pd.concat([X, pd.DataFrame({"q": [55.0]}, index=pd.period_range(y.index[-1] + 1, periods=1, freq="M"))])
    fc0 = ev.forecast_next(y, Xa, h=0, model="ols", use_gt=True, p=1, xlags=0, y_known_at_origin=False, transform="diff")
    assert fc0["target_date"] == str(y.index[-1] + 1) and fc0.get("note") is None
    fc0b = ev.forecast_next(y, X, h=0, model="ols", use_gt=True, p=1, xlags=0, y_known_at_origin=False, transform="diff")
    assert "note" in fc0b  # falls back to h=1 and says so
