"""Experiment cell 02-Q2-C — Q2 (US ILI % in 2 and 4 weeks) x Approach C (Djorno-first ablation).

Run from the repo root:  PYTHONIOENCODING=utf-8 python research/experiments/scripts/02-Q2-C.py

Ablation: (1) target-only {naive, seasonal_naive(52), ar}; (2) ridge + raw GT; (3) ridge + Djorno
preprocessing (full / detrend=None / cluster=False); (4) SARIMAX(2,0,0) with and without Djorno GT.
Levels, p=4, xlags=1, period=52, fourier_k=2, min_train=104, publication lag everywhere
(y_known_at_origin=False): at origin week t, GT for week t is known but ILI for week t is not.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from trends_predict import gt, truth, preprocess as pp, evaluate as ev  # noqa: E402
from trends_predict.models import sarimax_forecast  # noqa: E402

ID = "02-Q2-C"
OUT = Path("research/experiments")
T0 = time.time()

KW = ["flu", "flu symptoms", "influenza", "flu a symptoms", "fever", "cough", "sore throat", "body aches",
      "chills", "tamiflu", "flu medicine", "flu test", "how long does flu last", "theraflu", "nyquil",
      "flu shot", "stomach flu"]
GEO, TF = "US", "today 5-y"
HORIZONS = [0, 2, 4]
MIN_TRAIN = 104
DESIGN = dict(p=4, xlags=1, period=52, fourier_k=2, min_train=MIN_TRAIN, y_known_at_origin=False)
EXCLUDE = ("2020-03-01", "2021-06-30")            # requested by the cell spec
# The requested window predates the 5-year GT sample (test origins start 2023-09), so it is a no-op here.
# Supplementary window that IS inside the test sample: the record 2025-26 peak (wILI 8.3 % on 2025-12-27).
EXCLUDE_SUPP = ("2025-12-01", "2026-02-28")
# Pre-fix leakage probe, run before patching ev.rolling_backtest (see report, "Surprises"):
LEAK_PROBE = {"h0_ridge_djorno_rmse_leaky": 0.2461, "note": "origin row (with its true target) was in the h=0 training set"}


def dj_full():
    return pp.djorno_pipeline(period=52, halflife=1.5, detrend="rolling", cluster=True)


def dj_nodetrend():
    return pp.djorno_pipeline(period=52, halflife=1.5, detrend=None, cluster=True)


def dj_nocluster():
    return pp.djorno_pipeline(period=52, halflife=1.5, detrend="rolling", cluster=False)


def dj_nosmooth():
    return pp.djorno_pipeline(period=52, halflife=0, detrend="rolling", cluster=True)


def smooth_only():  # ZeroRepair -> log -> EWMA(1.5) -> Standardize: isolates the smoothing step vs raw
    return pp.djorno_pipeline(period=52, halflife=1.5, detrend=None, cluster=False)


PIPES = {"ridge+raw": pp.raw_pipeline, "ridge+djorno": dj_full,
         "ridge+djorno_nodetrend": dj_nodetrend, "ridge+djorno_nocluster": dj_nocluster,
         "ridge+djorno_nosmooth": dj_nosmooth, "ridge+smooth_only": smooth_only}
GT_MODELS = list(PIPES) + ["sarimax+djorno"]


def lagged_naive(y: pd.Series, h: int, min_train: int) -> ev.BacktestResult:
    """Random walk under publication lag: at origin t the latest known value is y_{t-1}."""
    y = y.astype(float).sort_index()
    rows = [{"target_date": y.index[t + h], "origin": y.index[t], "y_true": y.iloc[t + h], "y_pred": y.iloc[t - 1]}
            for t in range(min_train, len(y) - h)]
    return ev.BacktestResult("naive_lag1", pd.DataFrame(rows).set_index("target_date"), h)


def md_table(tab: pd.DataFrame) -> str:
    cols = [c for c in ["n", "MAE", "RMSE", "MAPE", "MASE", "OOS_R2_vs_bench", "DirAcc", "CW_p", "DM_p"] if c in tab.columns]
    head = "| model | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    body = ""
    for m, r in tab.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if c == "n":
                vals.append(str(int(v)))
            elif v != v:
                vals.append("")
            elif c in ("CW_p", "DM_p"):
                vals.append(f"{v:.3f}")
            elif c == "MAPE":
                vals.append(f"{v:.1f}")
            else:
                vals.append(f"{v:.3f}")
        body += f"| {m} | " + " | ".join(vals) + " |\n"
    return head + body


def records(tab: pd.DataFrame) -> list[dict]:
    out = []
    for m, r in tab.iterrows():
        d = {"model": m}
        for k, v in r.items():
            d[k] = None if (isinstance(v, float) and v != v) else (float(v) if isinstance(v, (float, np.floating)) else int(v))
        out.append(d)
    return out


def sarimax_forward(ya: pd.Series, Xp: pd.DataFrame, h: int, use_gt: bool, preprocess=None,
                    order=(2, 0, 0)) -> dict:
    """Forward SARIMAX under publication lag: origin = first GT week without ILI (t*), fit through
    t*-1, forecast h+1 steps with exog x_{T-h} for each target T (mirrors ev.rolling_sarimax)."""
    nxt = Xp.index[Xp.index > ya.index.max()][:1]
    idx = ya.index.union(nxt)
    t_star = idx[-1]
    Xtr = Xfu = None
    y_fit = ya.copy()
    if use_gt:
        Xw = Xp.reindex(idx).astype(float)
        Xw = preprocess().fit_transform(Xw) if preprocess is not None else Xw
        Xs = Xw.shift(h)
        y_fit = ya.iloc[h:]
        Xtr = Xs.loc[y_fit.index]
        Xfu = pd.concat([Xs.loc[[t_star]]] + ([Xw.iloc[-h:]] if h > 0 else []))
    pred = sarimax_forecast(y_fit, Xtr, Xfu, h + 1, order, (0, 0, 0, 0))
    return {"origin": str(t_star), "target_date": str(t_star + h), "point": float(pred), "model": "sarimax",
            "n_train": int(len(y_fit)), "features": None}


def main() -> None:
    n_cat0 = len(gt.catalog())
    # ---------------------------------------------------------------- data
    y = truth.fluview_ili("nat", 201040)
    X, info = gt.fetch_many(KW, geo=GEO, timeframe=TF)
    n_gt_requests = len(gt.catalog()) - n_cat0
    print(f"truth: {len(y)} weeks {y.index.min().date()} -> {y.index.max().date()}")
    print(f"GT: {X.shape} {X.index.min().date()} -> {X.index.max().date()}; dropped={info['dropped']} failed={info['failed']}")
    ya, Xa = truth.align(y, X, "W")
    gaps = int((np.diff(ya.index.asi8) != 1).sum())
    print(f"aligned: {len(ya)} weeks {ya.index.min()} -> {ya.index.max()}; gaps={gaps}; NaN in X={int(Xa.isna().sum().sum())}")
    assert gaps == 0, "aligned index is not contiguous; lags would be misaligned"
    Xp = X.copy()
    Xp.index = Xp.index.to_period("W-SAT")          # NOT intersected -> keeps the GT week beyond ILI
    print(f"GT extends beyond ILI: {Xp.index.max()} > {ya.index.max()} -> {Xp.index.max() > ya.index.max()}")
    print(f"y summary: mean={ya.mean():.2f} sd={ya.std():.2f} min={ya.min():.2f} max={ya.max():.2f} (peak {ya.idxmax()})")

    # ---------------------------------------------------------------- cluster membership (full sample)
    full_pipe = dj_full()
    full_pipe.fit(Xa)
    clusters_full = full_pipe.steps[-1].membership()
    first_pipe = dj_full()
    first_pipe.fit(Xa.iloc[:MIN_TRAIN])
    clusters_first = first_pipe.steps[-1].membership()
    print("clusters (full sample):", json.dumps(clusters_full, indent=1))
    print("n clusters first window / full:", len(clusters_first), len(clusters_full))

    # ---------------------------------------------------------------- backtests
    results: dict[int, dict[str, ev.BacktestResult]] = {}
    timings: dict[str, float] = {}
    for h in HORIZONS:
        res: dict[str, ev.BacktestResult] = {}
        t = time.time()
        res["naive"] = ev.naive(ya, h=h, min_train=MIN_TRAIN)
        if h >= 1:
            res["naive_lag1"] = lagged_naive(ya, h, MIN_TRAIN)
        res["seasonal_naive"] = ev.seasonal_naive(ya, h=h, period=52, min_train=MIN_TRAIN)
        res["ar"] = ev.rolling_backtest(ya, None, h=h, model="ar", use_gt=False, **DESIGN)
        for name, pipe in PIPES.items():
            res[name] = ev.rolling_backtest(ya, Xa, h=h, model="ridge", use_gt=True, preprocess=pipe, **DESIGN)
        timings[f"h{h}_ridge_block"] = time.time() - t
        t = time.time()
        res["sarimax"] = ev.rolling_sarimax(ya, None, h=h, use_gt=False, order=(2, 0, 0), seasonal_order=(0, 0, 0, 0),
                                            min_train=MIN_TRAIN, y_known_at_origin=False)
        res["sarimax+djorno"] = ev.rolling_sarimax(ya, Xa, h=h, use_gt=True, order=(2, 0, 0), seasonal_order=(0, 0, 0, 0),
                                                   min_train=MIN_TRAIN, preprocess=dj_full, y_known_at_origin=False)
        timings[f"h{h}_sarimax_block"] = time.time() - t
        results[h] = res
        print(f"h={h}: " + ", ".join(f"{k}:n={len(v.preds)}" for k, v in res.items()) +
              f"  ({timings[f'h{h}_ridge_block']:.0f}s ridge, {timings[f'h{h}_sarimax_block']:.0f}s sarimax)")

    # ---------------------------------------------------------------- comparison tables
    tables: dict[str, list[dict]] = {}
    verdicts: dict[str, dict] = {}
    bench_used: dict[int, str] = {}
    md_out: dict[str, str] = {}
    for h in HORIZONS:
        res = results[h]
        # benchmark = best (lowest RMSE) target-only model with the SAME information set (y_t unknown)
        fair_bench = ["naive" if h == 0 else "naive_lag1", "seasonal_naive", "ar", "sarimax"]
        rm = {m: ev.rmse(res[m].errors.values) for m in fair_bench}
        bench = min(rm, key=rm.get)
        bench_used[h] = bench
        print(f"h={h} target-only RMSE: {json.dumps({k: round(v, 3) for k, v in rm.items()})} -> bench={bench}")
        for label, exc in [("full", None), ("excl", EXCLUDE), ("excl_supp", EXCLUDE_SUPP)]:
            tab = ev.compare(res, bench=bench, y_hist=ya, period=52, exclude=exc)
            key = f"h{h}_{label}"
            tables[key] = records(tab)
            md_out[key] = md_table(tab)
            verdicts[key] = ev.verdict(tab, bench=bench, gt_models=GT_MODELS)
            print(f"\n### h={h} {label} (bench={bench}, n={int(tab['n'].iloc[0])})\n" + md_out[key])
            print("verdict:", verdicts[key])

    # ---------------------------------------------------------------- season split diagnostic
    # in-season = target week in Oct..Mar; off-season = Apr..Sep. RMSE per model per regime.
    season_split: dict[str, dict] = {}
    for h in HORIZONS:
        res = results[h]
        common = None
        for r in res.values():
            common = r.preds.index if common is None else common.intersection(r.preds.index)
        month = pd.Index(common.to_timestamp()).month
        in_season = np.isin(month, [10, 11, 12, 1, 2, 3])
        d = {"n_in_season": int(in_season.sum()), "n_off_season": int((~in_season).sum())}
        for m in [bench_used[h], "ridge+raw", "ridge+djorno"]:
            e = (res[m].preds.loc[common, "y_true"] - res[m].preds.loc[common, "y_pred"]).values
            d[m] = {"rmse_in_season": round(ev.rmse(e[in_season]), 3), "rmse_off_season": round(ev.rmse(e[~in_season]), 3)}
        season_split[f"h{h}"] = d
        print(f"season split h={h}: {json.dumps(d)}")

    # ---------------------------------------------------------------- forward forecasts
    forecasts: dict[str, dict] = {}
    for h in HORIZONS:
        res = results[h]
        candidates = {m: ev.rmse(r.errors.values) for m, r in res.items() if m != "naive" or h == 0}
        best = min(candidates, key=candidates.get)
        best_gt = min((m for m in candidates if m in GT_MODELS), key=candidates.get)
        r_best = res[best]
        q52 = ev.conformal_interval(r_best.errors.values, alpha=0.1, recent=52)
        qall = ev.conformal_interval(r_best.errors.values, alpha=0.1)
        # supplementary calm-state interval: residuals from origins whose last published ILI was < 2.5 %
        last_known = ya.shift(1).reindex(pd.Index(r_best.preds["origin"].values)).values
        q_calm = ev.conformal_interval(r_best.errors.values[last_known < 2.5], alpha=0.1)
        if best in PIPES:
            fc = ev.forecast_next(ya, Xp, h=h, model="ridge", use_gt=True, p=4, xlags=1, period=52, fourier_k=2,
                                  y_known_at_origin=False, preprocess=PIPES[best])
        elif best == "ar":
            fc = ev.forecast_next(ya, Xp, h=h, model="ar", use_gt=False, p=4, xlags=0, period=52, fourier_k=2,
                                  y_known_at_origin=False)
        elif best == "sarimax":
            fc = sarimax_forward(ya, Xp, h, use_gt=False)
        elif best == "sarimax+djorno":
            fc = sarimax_forward(ya, Xp, h, use_gt=True, preprocess=dj_full)
        elif best == "seasonal_naive":
            t_star = Xp.index[Xp.index > ya.index.max()][0]
            tgt = t_star + h
            fc = {"origin": str(t_star), "target_date": str(tgt), "point": float(ya.loc[tgt - 52]), "model": "seasonal_naive",
                  "n_train": int(len(ya)), "features": None}
        else:  # naive / naive_lag1 -> last published value
            t_star = Xp.index[Xp.index > ya.index.max()][0]
            fc = {"origin": str(t_star), "target_date": str(t_star + h), "point": float(ya.iloc[-1]), "model": best,
                  "n_train": int(len(ya)), "features": None}
        # also the best GT model's forecast for reference
        fc_gt = None
        if best_gt != best:
            if best_gt in PIPES:
                fc_gt = ev.forecast_next(ya, Xp, h=h, model="ridge", use_gt=True, p=4, xlags=1, period=52, fourier_k=2,
                                         y_known_at_origin=False, preprocess=PIPES[best_gt])
            else:
                fc_gt = sarimax_forward(ya, Xp, h, use_gt=True, preprocess=dj_full)
        # benchmark's own forward forecast (the benchmark wins off-season, and the origin is off-season)
        bm = bench_used[h]
        t_star = Xp.index[Xp.index > ya.index.max()][0]
        if bm == "ar":
            fc_b = ev.forecast_next(ya, Xp, h=h, model="ar", use_gt=False, p=4, xlags=0, period=52, fourier_k=2,
                                    y_known_at_origin=False)
        elif bm == "seasonal_naive":
            fc_b = {"origin": str(t_star), "target_date": str(t_star + h), "point": float(ya.loc[t_star + h - 52]), "model": bm}
        elif bm == "sarimax":
            fc_b = sarimax_forward(ya, Xp, h, use_gt=False)
        else:
            fc_b = {"origin": str(t_star), "target_date": str(t_star + h), "point": float(ya.iloc[-1]), "model": bm}
        q_b = ev.conformal_interval(res[bm].errors.values, alpha=0.1, recent=52)
        fc_b = {**fc_b, "q90_recent52": float(q_b), "lo": float(fc_b["point"] - q_b), "hi": float(fc_b["point"] + q_b)}
        forecasts[f"h{h}"] = {"best_model": best, "backtest_rmse": float(candidates[best]), **fc,
                              "bench_forecast": fc_b,
                              "q90_recent52": float(q52), "q90_all": float(qall), "q90_calm_state": float(q_calm),
                              "n_calm_residuals": int((last_known < 2.5).sum()),
                              "lo": float(fc["point"] - q52), "hi": float(fc["point"] + q52),
                              "last_known_ili": {"date": str(ya.index[-1]), "value": float(ya.iloc[-1])},
                              "best_gt_model": best_gt, "best_gt_rmse": float(candidates[best_gt]),
                              "best_gt_forecast": fc_gt}
        print(f"\nFORECAST h={h}: best={best} -> {json.dumps({k: v for k, v in forecasts[f'h{h}'].items() if k != 'best_gt_forecast'}, default=str)}")
        if fc_gt:
            print(f"   best GT model {best_gt}: {json.dumps(fc_gt, default=str)}")

    seconds = time.time() - T0
    out = {"id": ID, "question": "How high will US influenza-like-illness activity (ILI %) be in 2 and 4 weeks?",
           "approach": "C Djorno-first ablation", "truth": {"series": "fluview wili nat", "first": str(y.index.min().date()),
                                                             "last": str(y.index.max().date()), "n": int(len(y)),
                                                             "aligned_n": int(len(ya)), "aligned_first": str(ya.index.min()),
                                                             "aligned_last": str(ya.index.max())},
           "design": {**DESIGN, "levels": True, "window": "expanding", "halflife": 1.5, "detrend_window": 52,
                      "cluster_distance_threshold": 0.4, "sarimax_order": [2, 0, 0]},
           "queries": list(X.columns), "dropped": info["dropped"], "failed": info["failed"],
           "clusters_full_sample": clusters_full, "clusters_first_window": clusters_first,
           "bench": {f"h{h}": b for h, b in bench_used.items()},
           "tables": tables, "verdicts": verdicts, "forecast": forecasts, "season_split": season_split,
           "leak_probe_prefix": LEAK_PROBE,
           "exclude": {"requested": list(EXCLUDE), "supplementary": list(EXCLUDE_SUPP)},
           "requests": {"new_interest_requests_this_run": int(n_gt_requests), "interest_requests_if_cold": len(KW),
                        "related_queries_requests": 1},
           "timings": timings, "seconds": round(seconds, 1)}
    (OUT / f"{ID}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    Path("scratch").mkdir(exist_ok=True)
    Path(f"scratch/{ID}.tables.md").write_text("\n".join(f"### {k} (bench={bench_used[int(k[1])]})\n{v}" for k, v in md_out.items()),
                                               encoding="utf-8")
    print(f"\nwrote {OUT / f'{ID}.json'}; GT interest requests this run={n_gt_requests}; wall={seconds:.0f}s")


if __name__ == "__main__":
    main()
