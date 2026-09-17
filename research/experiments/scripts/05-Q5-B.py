"""Experiment cell 05-Q5-B — Q5 Israel unemployment next month, Approach B (ARGO-style breadth, Hebrew).

Run from the repo root:  PYTHONIOENCODING=utf-8 python research/experiments/scripts/05-Q5-B.py

Question (Hebrew): "מה יהיה שיעור האבטלה בישראל בחודש הבא?"

Design (fixed by the cell):
  * seeds ["אבטלה", "דמי אבטלה", "דרושים", "פיטורים", "עבודה"] -> gt.expand_queries(geo=IL, monthly, max_per_seed=12)
    -> curated to 30 Hebrew terms (26 expansion terms dropped: duplicates / brands / off-topic, see DROP) + topic /m/07s_c
  * truth: CBS LFS unemployment rate 15+, SA, monthly (cached by cell 05-Q5-A: data/truth/cbs_unemployment_rate_il.csv)
  * individual monthly pulls, geo=IL, default privacy-zero filter (nonzero share >= 0.5)
  * preprocessing: pp.raw_pipeline and pp.djorno_pipeline(period=12, halflife=1.0, cluster=False)
  * models: lasso, enet (expanding and sliding window=36) x both pipelines; ridge + djorno-nocluster (expanding)
    vs benchmarks {naive, drift, seasonal_naive, ar_diff}
  * transform="diff", p=3, xlags=1, period=12, fourier_k=1, min_train=48
  * horizons: h=0 nowcast (y_known_at_origin=False) and h=1 (y_known_at_origin=True)
  * views: full sample and exclude=("2020-03-01","2021-06-30")
  * LASSO selected features at recent origins; forward forecast with ev.forecast_next + 90 % conformal interval
"""
import sys, time, json
from collections import Counter
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from trends_predict import gt, truth, preprocess as pp, evaluate as ev

ID = "05-Q5-B"
OUT_JSON = f"research/experiments/{ID}.json"
EXCL = ("2020-03-01", "2021-06-30")
GEO = "IL"
TF = gt.timeframe_for("M")
SEEDS = ["אבטלה", "דמי אבטלה", "דרושים", "פיטורים", "עבודה"]
TOPIC = "/m/07s_c"  # Google topic "Unemployment" (language-neutral)
MIN_TRAIN = 48
WINDOW = 36

# Expansion terms dropped after inspection (reason in the comment):
DROP = {
    # near-duplicates of a kept term (same words, other order / benefits vs. unemployment variant)
    "דמי אבטלה ביטוח לאומי": "dup of ביטוח לאומי אבטלה", "אבטלה ביטוח לאומי": "dup of ביטוח לאומי אבטלה",
    "אבטלה לשכת התעסוקה": "dup of לשכת התעסוקה", "זכאות לדמי אבטלה": "dup of זכאות דמי אבטלה",
    "חישוב דמי אבטלה": "dup of חישוב אבטלה", "מחשבון דמי אבטלה": "dup of מחשבון אבטלה",
    "פיטורין": "spelling variant of פיטורים", "פיצויי פיטורין": "spelling variant of פיצויי פיטורים",
    "חוק פיצויי פיטורים": "dup of פיצויי פיטורים / חוק פיטורים",
    # regional splits of דרושים (collinear with the seed)
    "דרושים חיפה": "regional dup of דרושים", "דרושים תל אביב": "regional dup of דרושים",
    "דרושים בצפון": "regional dup of דרושים", "דרושים ירושלים": "regional dup of דרושים",
    "דרושים באר שבע": "regional dup of דרושים",
    # brands / organisations
    "שתיל": "brand (organisation name)", "שתיל דרושים": "brand jobs page",
    # not about unemployment / ambiguous
    "ביטוח לאומי": "National Insurance in general (pensions, maternity, disability)",
    "דמי ביטוח לאומי": "NI contributions", "פיצויים": "compensation in general (accidents, damages)",
    "שימוע": "hearing in general (legal)", "שעות עבודה": "working / opening hours", "דפי עבודה": "school worksheets",
    "כלי עבודה": "tools", "דיני עבודה": "labour law (generic)", "עבודה מועדפת": "post-army preferred-work grant (niche)",
    "עבודה סוציאלית": "social work (profession)",
}
BENCH_NAMES = ["naive", "drift", "seasonal_naive", "ar_diff"]
PIPES = {"raw": pp.raw_pipeline, "djorno": lambda: pp.djorno_pipeline(period=12, halflife=1.0, cluster=False)}
GT_MODELS = {}
for pipe in PIPES:
    for model in ("lasso", "enet"):
        GT_MODELS[f"{model}+{pipe}"] = dict(model=model, pipe=pipe, window=None)
        GT_MODELS[f"{model}+{pipe}+w{WINDOW}"] = dict(model=model, pipe=pipe, window=WINDOW)
GT_MODELS["ridge+djorno"] = dict(model="ridge", pipe="djorno", window=None)

t0 = time.time()
n_cat_before = len(gt.catalog())
live_before = gt.REQUEST_COUNT

# ---------------------------------------------------------------- truth (cached by cell A)
y = truth._load("cbs_unemployment_rate_il", None)
assert y is not None, "run 05-Q5-A.py first to cache the CBS series"
print(f"truth: {y.index.min().date()} -> {y.index.max().date()} ({len(y)} obs); last values:\n{y.tail(6).round(2).to_string()}")

# ---------------------------------------------------------------- query expansion (Hebrew, IL)
expanded = gt.expand_queries(SEEDS, geo=GEO, timeframe=TF, max_per_seed=12)
req_after_expand = gt.REQUEST_COUNT
print(f"expanded terms: {len(expanded)}")
unknown_drop = [d for d in DROP if d not in expanded]
if unknown_drop:
    print("WARNING: DROP entries not in the expansion (Google changed related queries?):", unknown_drop)
terms = [t for t in expanded if t not in DROP] + [TOPIC]
dropped_curation = [{"term": t, "reason": DROP[t]} for t in expanded if t in DROP]
print(f"kept after curation: {len(terms)} (incl. topic) ; dropped by curation: {len(dropped_curation)}")
for t in terms:
    print("  ", t)

# related_queries detail for the report (cached JSON, no extra requests)
related_detail = {}
for s in SEEDS:
    rq = gt.related_queries(s, GEO, TF)
    related_detail[s] = {"n_top": len(rq.get("top") or []), "n_rising": len(rq.get("rising") or []),
                         "top12": [r.get("query") for r in (rq.get("top") or [])[:12]]}

# ---------------------------------------------------------------- Google Trends pulls (individual, monthly)
X, info = gt.fetch_many(terms, geo=GEO, timeframe=TF)  # default min_nonzero_share=0.5
req_after_pulls = gt.REQUEST_COUNT
print(f"\nGT columns kept: {len(X.columns)}; dropped (privacy zeros): {info.get('dropped')}; failed: {info.get('failed')}")
print(f"X range {X.index.min().date()} -> {X.index.max().date()} ({len(X)} rows)")
nonzero_all = (X > 0).mean().round(3).to_dict()
nonzero_2012 = (X.loc["2012":] > 0).mean().round(3).to_dict()
renamed = [c for c in X.columns if c not in terms]
if renamed:
    print("WARNING: Google renamed columns:", renamed)

ya, Xa = truth.align(y, X, "M")
Xp = X.copy(); Xp.index = Xp.index.to_period("M")
Xp = Xp.loc[Xp.index >= ya.index.min()]
print(f"aligned: {len(ya)} months {ya.index.min()} -> {ya.index.max()}; {Xa.shape[1]} GT features; X extended to {Xp.index.max()}")


# ---------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame, floatfmt=3) -> str:
    cols = list(df.columns)
    out = ["| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if (v != v) else (f"{v:.{floatfmt}f}" if c not in ("n",) else f"{int(v)}"))
            else:
                cells.append(str(v))
        out.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def records(df: pd.DataFrame) -> list[dict]:
    recs = []
    for idx, row in df.iterrows():
        d = {"model": idx}
        for c, v in row.items():
            if isinstance(v, (float, np.floating)):
                d[c] = None if v != v else float(v)
            elif isinstance(v, (int, np.integer)):
                d[c] = int(v)
            else:
                d[c] = v
        recs.append(d)
    return recs


# ---------------------------------------------------------------- backtests
tables, verdicts, benches, md_tables, results_all, selected_log, timing = {}, {}, {}, {}, {}, {}, {}
for h, known in [(0, False), (1, True)]:
    kw = dict(h=h, p=3, period=12, fourier_k=1, y_known_at_origin=known, min_train=MIN_TRAIN, transform="diff")
    res = {}
    res["naive"] = ev.naive(ya, h, MIN_TRAIN)
    res["drift"] = ev.drift(ya, h, MIN_TRAIN)
    res["seasonal_naive"] = ev.seasonal_naive(ya, h, 12, MIN_TRAIN)
    res["ar_diff"] = ev.rolling_backtest(ya, None, model="ar", use_gt=False, **kw)
    for name, spec in GT_MODELS.items():
        t1 = time.time()
        res[name] = ev.rolling_backtest(ya, Xa, model=spec["model"], use_gt=True, xlags=1, window=spec["window"],
                                        preprocess=PIPES[spec["pipe"]], **kw)
        timing[f"h{h}_{name}"] = round(time.time() - t1, 1)
        sel = res[name].selected
        if sel:
            n_per_origin = [len(s["features"]) for s in sel]
            freq = Counter(f for s in sel[-24:] for f in s["features"])
            selected_log[f"h{h}_{name}"] = {
                "n_selected_last12": n_per_origin[-12:],
                "mean_n_selected_all": round(float(np.mean(n_per_origin)), 2),
                "share_origins_empty": round(float(np.mean([n == 0 for n in n_per_origin])), 3),
                "last3": sel[-3:],
                "freq_last24": [{"feature": f, "count": c} for f, c in freq.most_common(12)],
            }
            print(f"{name} h={h}: n_selected last 12 origins {n_per_origin[-12:]} ; last origin -> {sel[-1]['features']}")
    results_all[h] = res
    for view, excl in [("full", None), ("excl", EXCL)]:
        key = f"h{h}_{view}"
        pre = ev.compare({k: res[k] for k in BENCH_NAMES}, bench="naive", exclude=excl)
        bench = pre["RMSE"].idxmin()
        tab = ev.compare(res, bench=bench, y_hist=ya, period=12, exclude=excl)
        v = ev.verdict(tab, bench, list(GT_MODELS))
        tables[key], verdicts[key], benches[key] = records(tab), v, bench
        md_tables[key] = md_table(tab)
        print(f"\n=== h={h} y_known={known} view={view} bench={bench} (n={int(tab['n'].iloc[0])}) ===")
        print(tab.round(3).to_string())
        print("verdict:", v)


# ---------------------------------------------------------------- forward forecast
def winner(h: int) -> str:
    """GT model only if the verdict says GT helps in BOTH views; pick the GT model with the best
    full-sample OOS R²; otherwise the target-only bench of the full-sample view."""
    if verdicts[f"h{h}_full"]["gt_helps"] and verdicts[f"h{h}_excl"]["gt_helps"]:
        full = {r["model"]: r["OOS_R2_vs_bench"] for r in tables[f"h{h}_full"] if r["model"] in GT_MODELS}
        return max(full, key=full.get)
    return benches[f"h{h}_full"]


def target_only_forecast(name: str, h: int) -> dict:
    last = ya.dropna().index[-1]
    tgt = last + max(h, 1)
    if name == "ar_diff":
        return ev.forecast_next(ya, None, h=h, model="ar", use_gt=False, p=3, period=12, fourier_k=1,
                                y_known_at_origin=True, transform="diff")
    if name == "seasonal_naive":
        return {"origin": str(last), "target_date": str(tgt), "point": float(ya.loc[tgt - 12]),
                "model": "seasonal_naive", "n_train": int(len(ya)), "features": None}
    if name == "drift":
        yv = ya.dropna().astype(float)
        return {"origin": str(last), "target_date": str(tgt), "point": float(yv.iloc[-1] + max(h, 1) * yv.diff().mean()),
                "model": "drift", "n_train": int(len(ya)), "features": None}
    return {"origin": str(last), "target_date": str(tgt), "point": float(ya.loc[last]),
            "model": "naive", "n_train": int(len(ya)), "features": None}


def best_gt(h: int) -> str:
    full = {r["model"]: r["RMSE"] for r in tables[f"h{h}_full"] if r["model"] in GT_MODELS}
    return min(full, key=full.get)


forecast = {}
last_y_date = ya.dropna().index.max()
print(f"\nX extends beyond y? {Xp.index.max() > last_y_date}  (X last {Xp.index.max()}, y last {last_y_date})")
for h, known in [(0, False), (1, True)]:
    win = winner(h)
    res = results_all[h]
    q90 = ev.conformal_interval(res[win].errors.values, alpha=0.1, recent=36)
    q90_all = ev.conformal_interval(res[win].errors.values, alpha=0.1)
    common = dict(h=h, p=3, xlags=1, period=12, fourier_k=1, y_known_at_origin=known, transform="diff")
    if win in GT_MODELS:
        spec = GT_MODELS[win]
        fc = ev.forecast_next(ya, Xp, model=spec["model"], use_gt=True, window=spec["window"],
                              preprocess=PIPES[spec["pipe"]], **common)
    else:
        fc = target_only_forecast(win, h)
    refs = {}
    for name in [best_gt(h), "lasso+raw", "lasso+djorno", "ridge+djorno"]:
        if name in refs:
            continue
        spec = GT_MODELS[name]
        fc_gt = ev.forecast_next(ya, Xp, model=spec["model"], use_gt=True, window=spec["window"],
                                 preprocess=PIPES[spec["pipe"]], **common)
        q = ev.conformal_interval(res[name].errors.values, alpha=0.1, recent=36)
        feats = fc_gt.get("features")
        if isinstance(feats, dict):
            feats = dict(sorted(feats.items(), key=lambda kv: -abs(kv[1]))[:8])
        refs[name] = {**fc_gt, "features": feats, "q90_recent36": float(q),
                      "lo": float(fc_gt["point"] - q), "hi": float(fc_gt["point"] + q)}
    forecast[f"h{h}"] = {"winner": win, **fc,
                         "q90_recent36": float(q90), "lo": float(fc["point"] - q90), "hi": float(fc["point"] + q90),
                         "q90_all_origins": float(q90_all), "best_gt_model_by_rmse": best_gt(h), "gt_reference": refs}
    print(f"\n--- forward forecast h={h}: winner={win} ---")
    print(json.dumps(forecast[f"h{h}"], indent=1, default=str, ensure_ascii=False))


def rolling_coverage(errs: pd.Series, alpha=0.1, recent=36):
    e = errs.values; hits = []
    for i in range(recent, len(e)):
        q = ev.conformal_interval(e[:i], alpha=alpha, recent=recent)
        hits.append(abs(e[i]) <= q)
    return float(np.mean(hits)) if hits else float("nan")


cov = {f"h{h}": rolling_coverage(results_all[h][forecast[f"h{h}"]["winner"]].errors) for h in (0, 1)}
print("rolling empirical coverage of the 90% conformal rule:", cov)

# ---------------------------------------------------------------- output
seconds = time.time() - t0
n_cat_after = len(gt.catalog())
out = {"id": ID, "question": "מה יהיה שיעור האבטלה בישראל בחודש הבא? (What will Israel's unemployment rate be next month?)",
       "approach": "B (ARGO-style breadth: Hebrew related-query expansion, geo=IL)",
       "truth": {"series": "CBS LFS unemployment rate 15+, SA, % (sid 491094 spliced with 41097; cached by 05-Q5-A)",
                 "first": str(y.index.min().date()), "last": str(y.index.max().date()), "n": int(len(y)), "freq": "M"},
       "seeds": SEEDS, "expanded_all": expanded, "n_expanded": len(expanded), "related_detail": related_detail,
       "dropped_curation": dropped_curation, "queries": terms, "queries_kept": list(X.columns),
       "queries_dropped_privacy": info.get("dropped", []), "queries_failed": info.get("failed", []),
       "renamed_columns": renamed, "nonzero_share_2004": nonzero_all, "nonzero_share_2012": nonzero_2012,
       "gt_range": {"first": str(X.index.min().date()), "last": str(X.index.max().date()), "n": int(len(X))},
       "design": {"p": 3, "xlags": 1, "period": 12, "fourier_k": 1, "min_train": MIN_TRAIN, "window": WINDOW,
                  "transform": "diff", "gt_models": {k: f"{v['model']} + {v['pipe']}" + (f" + sliding w={v['window']}" if v['window'] else " + expanding")
                                                    for k, v in GT_MODELS.items()},
                  "pipelines": {"raw": "raw_pipeline", "djorno": "djorno_pipeline(period=12, halflife=1.0, cluster=False)"},
                  "benchmarks": BENCH_NAMES, "exclude": list(EXCL)},
       "bench_per_view": benches, "tables": tables, "verdicts": verdicts, "selected": selected_log,
       "forecast": forecast, "interval_coverage_backtest": cov, "backtest_seconds": timing,
       "requests": int(gt.REQUEST_COUNT - live_before),
       "requests_detail": {"expansion_incl_retries": int(req_after_expand - live_before),
                           "interest_pulls_incl_retries": int(req_after_pulls - req_after_expand)},
       "catalog_delta": int(n_cat_after - n_cat_before), "catalog_size": int(n_cat_after), "seconds": round(seconds, 1)}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False, default=str)
print("\n\n# markdown tables for the report")
for k, t in md_tables.items():
    print(f"\n### {k} (bench = {benches[k]})\n\n{t}\n")
print(f"\nlive GT calls this process: {gt.REQUEST_COUNT - live_before}; catalog {n_cat_before} -> {n_cat_after}; elapsed {seconds:.0f}s")
print(f"wrote {OUT_JSON}")
