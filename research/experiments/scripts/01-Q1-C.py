"""Cell 01-Q1-C — US unemployment rate next month, Approach C (Djorno preprocessing first).

Run from the repo root:  PYTHONIOENCODING=utf-8 python research/experiments/scripts/01-Q1-C.py

Ablation:
  (1) no GT           : naive, seasonal_naive, ar (diff), sarimax(1,1,0) no exog
  (2) raw GT          : ridge + raw_pipeline
  (3) preprocessed GT : ridge + djorno_pipeline(period=12, halflife=1.0, detrend='rolling', cluster=True)
                        + variants halflife=0 (no smoothing) and cluster=False
  (4) SARIMAX(1,1,0)  : with djorno-preprocessed GT exog vs without
Horizons: h=0 nowcast (y_known_at_origin=False) and h=1. Full sample and 2020-03..2020-12 excluded.
"""
import sys, time, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from trends_predict import gt, truth, preprocess as pp, evaluate as ev

pd.set_option("display.width", 250)
t0 = time.time()
ID = "01-Q1-C"
OUT_JSON = f"research/experiments/{ID}.json"
EXCL = ("2020-03-01", "2020-12-31")

# ------------------------------------------------------------------ keywords
# 10 terms already used by the smoke test + 4 chosen from gt.related_queries('unemployment'|'layoffs')
# (top related for the seed, excluding state-specific and company-specific queries).
KWS = [
    "unemployment", "unemployment benefits", "file for unemployment", "unemployment office",
    "jobs hiring", "indeed jobs", "layoffs", "severance", "job openings", "resume",
    "unemployment claim", "unemployment login", "unemployment insurance", "tech layoffs",
]
RELATED_SEEDS = ["unemployment", "layoffs"]   # 2 related_queries calls, cached as JSON (not in catalog)

N_PRECACHED = 10   # the first 10 keywords were pulled by the 00b smoke test before this cell ran

def my_vintages():
    """Single-keyword vintages in the shared catalog that belong to this cell (other cells run concurrently,
    so a plain len(catalog()) diff over-counts)."""
    c = gt.catalog()
    return int(c.keywords.apply(lambda k: len(k) == 1 and k[0] in KWS).sum()) if len(c) else 0

n_cat0 = len(gt.catalog())
X, info = gt.fetch_many(KWS, geo="US", timeframe=gt.timeframe_for("M"))
n_cat1 = len(gt.catalog())
n_kw_requests = my_vintages() - N_PRECACHED   # includes any duplicate vintage written by a 429 retry
print("dropped:", info["dropped"], "failed:", info["failed"])
print("X:", X.shape, X.index.min(), "->", X.index.max(), "(current partial month dropped by gt._finish)")

y = truth.bls_series("UNRATE")
ya, Xa = truth.align(y, X, "M")
print("aligned:", ya.index.min(), "->", ya.index.max(), len(ya))
# untrimmed monthly GT frame (extends one month beyond the BLS print) — needed only for the h=0 forward nowcast
X_full = X.copy(); X_full.index = X_full.index.to_period("M"); X_full = X_full.groupby(level=0).mean().sort_index()

# ------------------------------------------------------------------ pipelines
def djorno(halflife=1.0, cluster=True):
    return lambda: pp.djorno_pipeline(period=12, halflife=halflife, detrend="rolling", cluster=cluster)

PIPES = {
    "ridge+rawGT":            (pp.raw_pipeline, "ridge"),
    "ridge+djorno":           (djorno(1.0, True), "ridge"),
    "ridge+djorno_hl0":       (djorno(0.0, True), "ridge"),
    "ridge+djorno_nocluster": (djorno(1.0, False), "ridge"),
}
GT_MODELS = list(PIPES) + ["sarimax110+djornoGT"]
TARGET_ONLY = ["naive", "seasonal_naive", "ar_diff", "sarimax110"]

# cluster membership on the full aligned sample
memb = pp.djorno_pipeline(period=12, halflife=1.0, detrend="rolling", cluster=True).fit(Xa).steps[-1].membership()
print("\nCluster membership (full sample):")
for k, v in memb.items():
    print(f"  {k}: {v}")

# ------------------------------------------------------------------ backtests
results, tables, verdicts, benches = {}, {}, {}, {}
for h, known in [(0, False), (1, True)]:
    res = {}
    kw = dict(h=h, p=3, period=12, fourier_k=1, y_known_at_origin=known, min_train=60, max_origins=120, transform="diff")
    res["naive"] = ev.naive(ya, h, 60)
    res["seasonal_naive"] = ev.seasonal_naive(ya, h, 12, 60)
    res["ar_diff"] = ev.rolling_backtest(ya, None, model="ar", use_gt=False, **kw)
    for name, (pipe, model) in PIPES.items():
        res[name] = ev.rolling_backtest(ya, Xa, model=model, use_gt=True, xlags=1, preprocess=pipe, **kw)
    sx = dict(h=h, order=(1, 1, 0), seasonal_order=(0, 0, 0, 0), min_train=60, max_origins=120)
    res["sarimax110"] = ev.rolling_sarimax(ya, None, use_gt=False, **sx)
    res["sarimax110+djornoGT"] = ev.rolling_sarimax(ya, Xa, use_gt=True, preprocess=djorno(1.0, True), **sx)
    results[h] = res
    for n_, r in res.items():
        print(f"  h={h} {n_}: n={len(r.preds)}")

    # benchmark = best target-only model by full-sample RMSE (on the common target dates)
    pre = ev.compare(res, bench="naive", y_hist=ya, period=12)
    bench = pre.loc[TARGET_ONLY, "RMSE"].idxmin()
    benches[h] = bench
    for label, excl in [("full", None), ("excl2020", EXCL)]:
        tab = ev.compare(res, bench=bench, y_hist=ya, period=12, exclude=excl)
        tables[(h, label)] = tab
        verdicts[(h, label)] = ev.verdict(tab, bench, GT_MODELS)
        print(f"\n=== h={h} (y_known={known}) bench={bench} view={label} ===")
        print(tab.round(3).to_string())
        print(verdicts[(h, label)])

# ------------------------------------------------------------------ pick winning configuration
# rule: among GT models, the one with the highest OOS R² on the shock-excluded h=0 table (calm-period robustness);
# ties/negatives -> fall back to the benchmark.
t0x = tables[(0, "excl2020")]
best_gt = t0x.loc[GT_MODELS, "OOS_R2_vs_bench"].idxmax()
print("\nbest GT config (h=0, excl2020):", best_gt, float(t0x.loc[best_gt, "OOS_R2_vs_bench"]))

# ------------------------------------------------------------------ forward forecasts
fc = {}
bench0 = benches[0]
# h=0 nowcast of the not-yet-published month (X_full extends one month beyond y)
if best_gt in PIPES:
    pipe, model = PIPES[best_gt]
    f0 = ev.forecast_next(ya, X_full, h=0, model=model, use_gt=True, p=3, xlags=1, period=12, fourier_k=1,
                          y_known_at_origin=False, preprocess=pipe, transform="diff")
    q0 = ev.conformal_interval(results[0][best_gt].errors.values, alpha=0.1, recent=36)
    f0.update({"lo": f0["point"] - q0, "hi": f0["point"] + q0, "half_width": q0, "config": best_gt})
    fc["h0_nowcast_bestGT"] = f0
    # h=1 with the same configuration (for completeness)
    f1 = ev.forecast_next(ya, Xa, h=1, model=model, use_gt=True, p=3, xlags=1, period=12, fourier_k=1,
                          y_known_at_origin=True, preprocess=pipe, transform="diff")
    q1 = ev.conformal_interval(results[1][best_gt].errors.values, alpha=0.1, recent=36)
    f1.update({"lo": f1["point"] - q1, "hi": f1["point"] + q1, "half_width": q1, "config": best_gt})
    fc["h1_bestGT"] = f1
# target-only reference forecasts
last = float(ya.iloc[-1])
qn0 = ev.conformal_interval(results[0]["naive"].errors.values, alpha=0.1, recent=36)
qn1 = ev.conformal_interval(results[1]["naive"].errors.values, alpha=0.1, recent=36)
fc["h0_naive"] = {"origin": str(ya.index[-1] + 1), "target_date": str(ya.index[-1] + 1), "point": last, "lo": last - qn0, "hi": last + qn0, "half_width": qn0}
fc["h1_naive"] = {"origin": str(ya.index[-1]), "target_date": str(ya.index[-1] + 1), "point": last, "lo": last - qn1, "hi": last + qn1, "half_width": qn1}
far = ev.forecast_next(ya, None, h=1, model="ar", use_gt=False, p=3, period=12, fourier_k=1, transform="diff")
qa1 = ev.conformal_interval(results[1]["ar_diff"].errors.values, alpha=0.1, recent=36)
far.update({"lo": far["point"] - qa1, "hi": far["point"] + qa1, "half_width": qa1})
fc["h1_ar_diff"] = far
for k, v in fc.items():
    print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "features"})
    if v.get("features"):
        print("   features:", v["features"][:8])

# ------------------------------------------------------------------ save
secs = time.time() - t0
def recs(tab):
    return json.loads(tab.reset_index().to_json(orient="records"))
out = {
    "id": ID,
    "question": "What will the US unemployment rate be next month?",
    "truth": {"series": "BLS LNS14000000 (UNRATE)", "first": str(ya.index[0]), "last": str(ya.index[-1]), "n": int(len(ya))},
    "queries": list(Xa.columns), "dropped": info["dropped"], "related_seeds": RELATED_SEEDS,
    "clusters_full_sample": memb,
    "benchmarks": {str(h): b for h, b in benches.items()},
    "tables": {f"h{h}_{lab}": recs(t) for (h, lab), t in tables.items()},
    "verdicts": {f"h{h}_{lab}": v for (h, lab), v in verdicts.items()},
    "best_gt_config": best_gt,
    "forecast": fc,
    "requests": int(n_kw_requests) + len(RELATED_SEEDS),
    "requests_note": f"{n_kw_requests} single-keyword vintages beyond the {N_PRECACHED} pre-cached by 00b "
                     f"(4 new keywords; any extra is a duplicate vintage from a 429 retry) + {len(RELATED_SEEDS)} "
                     f"related_queries calls. Shared catalog grew {n_cat0}->{n_cat1} during this run, mostly from other cells.",
    "gt_last_month": str(X_full.index.max()),
    "nowcast_note": "gt.fetch_interest drops the isPartial current month, so GT ends in the same month as the latest BLS "
                    "print; forecast_next(h=0) therefore falls back to the h=1 forecast (no complete unpublished month).",
    "seconds": round(secs, 1),
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, default=str)
print(f"\nsaved {OUT_JSON}; requests(new this run)={out['requests']}, elapsed {secs:.0f}s")
