"""Experiment cell 01-Q1-B — ARGO-style LASSO/elastic-net ARX on expanded Google Trends queries
for the US unemployment rate (BLS UNRATE), h=0 nowcast and h=1 forecast.

Run from the repo root:  PYTHONIOENCODING=utf-8 python research/experiments/scripts/01-Q1-B.py
"""
import sys, time, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from trends_predict import gt, truth, preprocess as pp, evaluate as ev
from trends_predict.gt import GTRequest

ID = "01-Q1-B"
OUT_DIR = "research/experiments"
t0 = time.time()

# ------------------------------------------------------------------------------------------
# 1. Query generation: seeds -> related_queries expansion (3 related_queries requests, cached)
# ------------------------------------------------------------------------------------------
SEEDS = ["unemployment", "unemployment benefits", "jobs"]
GEO, TF = "US", gt.timeframe_for("M")
REQUESTS = 0  # my own Google requests; the shared catalog is written by other agents too

def _rq_cached(seed):
    from trends_predict import QUERIES_DIR
    import hashlib
    from trends_predict.gt import _slugify
    key = hashlib.sha1(f"related_queries|{seed}|{GEO}|{TF}".encode("utf-8")).hexdigest()[:10]
    return (QUERIES_DIR / f"related_queries__{GEO}__{_slugify(seed)[:40]}__{key}.json").exists()

rq_new = [s for s in SEEDS if not _rq_cached(s)]
# NOTE: the 3 related_queries calls were made live earlier in this session (they were not cached
# before); count them regardless of whether this rerun hits the cache.
REQUESTS += 3
expanded = gt.expand_queries(SEEDS, geo=GEO, timeframe=TF, max_per_seed=12)

# Drop list with reasons (decided after looking at the 35 expanded terms).
DROP = {
    "texas benefits": "not unemployment-specific (Texas HHS benefits portal: SNAP/Medicaid)",
    "job": "single generic word, dominated by unrelated meanings (Steve Jobs 2011, 'job' as task)",
    "jobs indeed": "word-order duplicate of 'indeed' (kept)",
    "ny unemployment": "word-order duplicate of 'unemployment ny' (kept)",
    "nj unemployment": "word-order duplicate of 'unemployment nj' (kept)",
    "craigslist jobs": "platform-branded; secular decline of Craigslist dominates, not labour demand",
    "amazon jobs": "single-employer branded term; tracks Amazon's growth, not the labour market",
    "usa jobs": "federal hiring portal (usajobs.gov); government-hiring specific",
}
terms = [t for t in expanded if t not in DROP]
terms = list(dict.fromkeys(terms))  # dedupe, keep order
print(f"expanded {len(expanded)} terms -> kept {len(terms)} after dropping {len(DROP)}")

# ------------------------------------------------------------------------------------------
# 2. Pull individually (one request per term; cached terms are free)
# ------------------------------------------------------------------------------------------
def _cached(term):
    return (GTRequest.make([term], GEO, TF).directory() / "meta.json").exists()

new_pulls = [t for t in terms if not _cached(t)]
print(f"{len(terms) - len(new_pulls)} terms already cached, {len(new_pulls)} new pulls")
X, info = gt.fetch_many(terms, geo=GEO, timeframe=TF, mode="individual")
pulled_now = [t for t in new_pulls if _cached(t)]
REQUESTS += len(pulled_now)
# On a cached rerun nothing is pulled; keep the count recorded by the first (live) run.
import os
_prev = f"{OUT_DIR}/{ID}.json"
if os.path.exists(_prev):
    try:
        _pd = json.load(open(_prev, encoding="utf-8")).get("requests_detail", {})
        if _pd.get("interest_pulls_new", 0) > len(pulled_now):
            REQUESTS = 3 + _pd["interest_pulls_new"]
            pulled_now = [None] * _pd["interest_pulls_new"]
    except Exception:
        pass
print("fetch info: dropped", info["dropped"], "failed", info["failed"])
print("X shape", X.shape, "range", X.index.min().date(), "->", X.index.max().date())

# ------------------------------------------------------------------------------------------
# 3. Truth and alignment
# ------------------------------------------------------------------------------------------
y = truth.bls_series("UNRATE")
ya, Xa = truth.align(y, X, "M")
print("y range", y.index.min().date(), "->", y.index.max().date(), f"({len(y)} obs); aligned n={len(ya)}")
# monthly GT frame that keeps months beyond the last BLS print (for the h=0 forward nowcast)
X_ext = X.copy(); X_ext.index = X_ext.index.to_period("M"); X_ext = X_ext.groupby(level=0).mean()

# ------------------------------------------------------------------------------------------
# 4. Rolling-origin backtests
# ------------------------------------------------------------------------------------------
MIN_TRAIN, MAX_ORIGINS, EXCL = 60, 120, ("2020-03-01", "2020-12-31")
common = dict(p=3, period=12, fourier_k=1, min_train=MIN_TRAIN, max_origins=MAX_ORIGINS, transform="diff")
gt_kw = dict(use_gt=True, xlags=1, preprocess=pp.raw_pipeline)

results, tables, verdicts, selected, benches = {}, {}, {}, {}, {}
for h, known in [(0, False), (1, True)]:
    res = {}
    res["naive"] = ev.naive(ya, h, MIN_TRAIN)
    res["seasonal_naive"] = ev.seasonal_naive(ya, h, 12, MIN_TRAIN)
    res["ar_diff"] = ev.rolling_backtest(ya, None, h=h, model="ar", use_gt=False, y_known_at_origin=known, **common)
    for model in ("lasso", "enet"):
        for wname, win in (("exp", None), ("w36", 36)):
            res[f"{model}_{wname}+GT"] = ev.rolling_backtest(ya, Xa, h=h, model=model, y_known_at_origin=known,
                                                             window=win, **common, **gt_kw)
    gt_models = [m for m in res if "+GT" in m]
    # benchmark = best target-only model by RMSE on the full common sample
    prelim = ev.compare(res, bench="naive", y_hist=ya, period=12)
    bench = prelim.loc[["naive", "seasonal_naive", "ar_diff"], "RMSE"].idxmin()
    benches[h] = bench
    results[h] = res
    for tag, excl in (("all", None), ("excl2020", EXCL)):
        tab = ev.compare(res, bench=bench, y_hist=ya, period=12, exclude=excl)
        tables[(h, tag)] = tab
        verdicts[(h, tag)] = ev.verdict(tab, bench, gt_models)
        print(f"\n=== h={h} (y_known_at_origin={known}) bench={bench} exclude={excl} ===")
        print(tab.round(3).to_string())
        print(verdicts[(h, tag)])
    for m in gt_models:
        sel = res[m].selected
        selected[(h, m)] = sel[-3:] if sel else []
        print(f"{m} h={h} n_selected per origin (last 12):", [len(s["features"]) for s in sel[-12:]])
        print(f"   last origin {sel[-1]['origin'] if sel else None}: {sel[-1]['features'] if sel else None}")

# ------------------------------------------------------------------------------------------
# 5. Forward forecast with the best configuration (+ 90 % conformal interval)
#    "best" = GT model with lowest RMSE on the shock-excluded table (robust view); also report
#    the full-sample pick if different.
# ------------------------------------------------------------------------------------------
def _cfg(name):
    model, rest = name.split("_", 1)
    win = 36 if rest.startswith("w36") else None
    return model, win

forecasts = {}
for h, known in [(0, False), (1, True)]:
    tab_x = tables[(h, "excl2020")]
    gt_rows = [m for m in tab_x.index if "+GT" in m]
    best = tab_x.loc[gt_rows, "RMSE"].idxmin()
    best_all = tables[(h, "all")].loc[gt_rows, "RMSE"].idxmin()
    model, win = _cfg(best)
    Xf = X_ext if h == 0 else Xa
    try:
        fc = ev.forecast_next(ya, Xf, h=h, model=model, use_gt=True, y_known_at_origin=known, window=win,
                              p=3, xlags=1, period=12, fourier_k=1, preprocess=pp.raw_pipeline, transform="diff")
    except Exception as e:  # record, don't hide
        fc = {"error": repr(e)}
    q = ev.conformal_interval(results[h][best].errors.values, alpha=0.1, recent=36)
    q_all = ev.conformal_interval(results[h][best].errors.values, alpha=0.1)
    # benchmark forecast for context
    bfc = ev.forecast_next(ya, None, h=max(h, 1), model="ar", use_gt=False, y_known_at_origin=True,
                           p=3, period=12, fourier_k=1, transform="diff")
    fc.update({"best_model_excl2020": best, "best_model_all": best_all, "window": win,
               "q90_recent36": q, "q90_all_origins": q_all,
               "lo": (fc.get("point", np.nan) - q), "hi": (fc.get("point", np.nan) + q),
               "last_known_y": {"date": str(ya.index[-1]), "value": float(ya.iloc[-1])},
               "ar_diff_point": bfc["point"], "ar_diff_target": bfc["target_date"]})
    forecasts[h] = fc
    print(f"\n--- forward forecast h={h}: {json.dumps(fc, default=str, indent=1)}")

# ------------------------------------------------------------------------------------------
# 6. Persist JSON
# ------------------------------------------------------------------------------------------
seconds = round(time.time() - t0, 1)
def _tab_records(tab):
    return json.loads(tab.reset_index().to_json(orient="records", double_precision=4))

payload = {
    "id": ID,
    "question": "What will the US unemployment rate be next month?",
    "truth": {"source": "BLS LNS14000000 via truth.bls_series('UNRATE')", "first": str(y.index.min().date()),
              "last": str(y.index.max().date()), "n": int(len(y)), "freq": "M"},
    "gt": {"first": str(X.index.min().date()), "last": str(X.index.max().date()), "n_terms": int(X.shape[1]),
           "dropped_privacy": info["dropped"], "failed": info["failed"]},
    "queries": list(X.columns),
    "queries_expanded": expanded,
    "queries_dropped": DROP,
    "benchmarks": {str(h): b for h, b in benches.items()},
    "tables": {f"h{h}_{tag}": _tab_records(t) for (h, tag), t in tables.items()},
    "verdicts": {f"h{h}_{tag}": v for (h, tag), v in verdicts.items()},
    "selected_last_origins": {f"h{h}_{m}": s for (h, m), s in selected.items()},
    "forecast": {f"h{h}": fc for h, fc in forecasts.items()},
    "settings": {**common, "xlags": 1, "preprocess": "raw_pipeline (ZeroRepair -> log1p)", "windows": ["expanding", 36],
                 "models": ["lasso", "enet"], "exclude": EXCL},
    "requests": REQUESTS,
    "requests_detail": {"related_queries": 3, "interest_pulls_new": len(pulled_now), "interest_cached": len(terms) - len(new_pulls)},
    "seconds": seconds,
}
with open(f"{OUT_DIR}/{ID}.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=1, default=str, ensure_ascii=False)

# markdown tables for the report
print("\n\n######## MARKDOWN TABLES ########")
cols = ["n", "MAE", "RMSE", "MASE", "OOS_R2_vs_bench", "DirAcc", "CW_p", "DM_p"]

def _md(t: pd.DataFrame) -> str:  # no tabulate dependency
    t = t.copy()
    t.insert(0, "model", t.index)
    head = "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns)
    rows = ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))
                               for v in r) + " |" for r in t.itertuples(index=False)]
    return "\n".join([head] + rows)

for (h, tag), tab in tables.items():
    t = tab.reindex(columns=[c for c in cols if c in tab.columns])
    print(f"\n**h={h}, {tag}** (bench = {benches[h]})\n")
    print(_md(t))
print(f"\nrequests={REQUESTS} ({payload['requests_detail']}); seconds={seconds}")
