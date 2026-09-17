"""Experiment cell 05-Q5-A — Q5 Israel unemployment next month, Approach A (Choi–Varian minimal, Hebrew).

Run from the repo root:  PYTHONIOENCODING=utf-8 python research/experiments/scripts/05-Q5-A.py

Question (Hebrew): "מה יהיה שיעור האבטלה בישראל בחודש הבא?"

Design (fixed by the cell):
  * 5–7 hand-picked Hebrew keywords, geo=IL, monthly (gt.timeframe_for("M")), individual pulls
    + the language-neutral Google topic id for "Unemployment" (from gt.suggestions) as a candidate
  * truth: CBS Labour Force Survey unemployment rate, ages 15+, seasonally adjusted, monthly, 2012-01 ->
    (CBS series API, see cbs_unemployment_rate_il below), spliced old definition (2012-2024) + new definition (2025-)
  * GT models: OLS ARX + pp.raw_pipeline, ridge + djorno_pipeline(period=12, halflife=1.0)
    vs benchmarks {naive, seasonal_naive, ar_diff}
  * expanding window, transform="diff", p=3, xlags=1, period=12, fourier_k=1, min_train=48
  * horizons: h=0 nowcast (y_known_at_origin=False) and h=1 (y_known_at_origin=True)
  * views: full sample and exclude=("2020-03-01","2021-06-30") (COVID furlough / חל"ת period)
  * bench per view = lowest-RMSE target-only model in that view
  * forward forecast with ev.forecast_next using the winning configuration + 90 % conformal interval
"""
import sys, time, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import requests
from trends_predict import gt, truth, preprocess as pp, evaluate as ev

ID = "05-Q5-A"
OUT_JSON = f"research/experiments/{ID}.json"
EXCL = ("2020-03-01", "2021-06-30")
# Hand-picked Hebrew candidates (Choi–Varian "Welfare & Unemployment" flavour, Israeli institutions):
#   דמי אבטלה = unemployment benefits; לשכת התעסוקה / שירות התעסוקה = the Employment Service (old/official name);
#   ביטוח לאומי אבטלה = National Insurance + unemployment; דרושים = "wanted" (job ads); פיטורים = layoffs;
#   חל"ת = unpaid leave / furlough (COVID scheme). Google strips the gershayim (") so we also try the
#   quote-less spelling חלת. /m/07s_c is the language-neutral topic "Unemployment" (gt.suggestions).
KWS = ["דמי אבטלה", "לשכת התעסוקה", "שירות התעסוקה", "ביטוח לאומי אבטלה", "דרושים", "פיטורים",
       'חל"ת', "חלת", "/m/07s_c"]
BENCH_NAMES = ["naive", "seasonal_naive", "ar_diff"]
GT_MODELS = {
    "ols+rawGT": dict(model="ols", preprocess=pp.raw_pipeline),
    "ridge+djorno": dict(model="ridge", preprocess=lambda: pp.djorno_pipeline(period=12, halflife=1.0)),
}
MIN_TRAIN = 48

t0 = time.time()
n_req_before = len(gt.catalog())
live_before = gt.REQUEST_COUNT

# ---------------------------------------------------------------- truth: CBS series API (keyless)
CBS_LIST = "https://apis.cbs.gov.il/series/data/list?id={sid}&format=json&download=false&PageSize=1000"
CBS_SERIES = {
    # path 11,5,1,5,2226 : LFS, "data from Jan-2012 to Dec-2025, old definition, monthly" -> unemployed, % of
    # labour force, ages 15+, total population, seasonally adjusted ("מנוכי עונתיות")
    "old_sa": 491094,
    # path 11,6,1,5,2226 : LFS, "from Jan-2025, new definition (2022 census estimates)" -> same variable, SA
    "new_sa": 41097,
}


def cbs_series(sid: int) -> tuple[pd.Series, dict]:
    r = requests.get(CBS_LIST.format(sid=sid), timeout=90)
    r.raise_for_status()
    s = r.json()["DataSet"]["Series"][0]
    obs = {pd.Timestamp(o["TimePeriod"] + "-01"): float(o["Value"]) for o in s["obs"] if o.get("Value") is not None}
    meta = {"sid": s["id"], "update": s.get("update"), "data": s["data"]["name"], "unit": s["unit"]["name"],
            "path": {k: (v["name"] if isinstance(v, dict) else v) for k, v in s["path"].items()}}
    return pd.Series(obs).sort_index(), meta


def cbs_unemployment_rate_il(max_age_days: float | None = 1.0) -> pd.Series:
    name = "cbs_unemployment_rate_il"
    cached = truth._load(name, max_age_days)
    if cached is not None:
        return cached
    old, m_old = cbs_series(CBS_SERIES["old_sa"])
    new, m_new = cbs_series(CBS_SERIES["new_sa"])
    overlap = new.index.intersection(old.index)
    diff = (new - old).loc[overlap]
    spliced = pd.concat([old.loc[old.index < new.index.min()], new]).sort_index()
    meta = {"source": "CBS Israel series API (apis.cbs.gov.il/series/data/list)", "freq": "M",
            "description": "Unemployment rate, ages 15+, total population, seasonally adjusted, % of civilian labour force "
                           "(Labour Force Survey, monthly). Spliced: old definition (2012-01..2024-12) + new definition (2025-01..).",
            "endpoint": CBS_LIST, "series": {"old_sa": m_old, "new_sa": m_new},
            "splice_date": str(new.index.min().date()),
            "overlap_months": int(len(overlap)),
            "overlap_new_minus_old_mean": round(float(diff.mean()), 3) if len(overlap) else None,
            "overlap_new_minus_old_max_abs": round(float(diff.abs().max()), 3) if len(overlap) else None}
    return truth._save(name, spliced, meta)


y = cbs_unemployment_rate_il()
print(f"truth: {y.index.min().date()} -> {y.index.max().date()} ({len(y)} obs); last values:\n{y.tail(6).round(2).to_string()}")

# ---------------------------------------------------------------- Google Trends (Hebrew, IL)
sugg = {}
for term in ["Unemployment", "אבטלה"]:
    try:
        sugg[term] = gt.suggestions(term)[:5]
    except Exception as e:  # noqa: BLE001
        sugg[term] = f"FAILED: {e!r}"[:200]
print("suggestions:", json.dumps(sugg, ensure_ascii=False))

X, info = gt.fetch_many(KWS, geo="IL", timeframe=gt.timeframe_for("M"))  # default min_nonzero_share=0.5
print(f"GT columns kept: {list(X.columns)}\n dropped: {info.get('dropped')}\n failed: {info.get('failed')}")
print(f"X range {X.index.min().date()} -> {X.index.max().date()} ({len(X)} rows)")
nonzero_all = (X > 0).mean().round(3).to_dict()
nonzero_2012 = (X.loc["2012":] > 0).mean().round(3).to_dict()
print("non-zero share (2004-):", nonzero_all)
print("non-zero share (2012-):", nonzero_2012)

# monthly period frames: intersected for the backtest, extended (X beyond y) for the h=0 nowcast
ya, Xa = truth.align(y, X, "M")
Xp = X.copy(); Xp.index = Xp.index.to_period("M")
Xp = Xp.loc[Xp.index >= ya.index.min()]
print(f"aligned: {len(ya)} months {ya.index.min()} -> {ya.index.max()};  X extended to {Xp.index.max()}")
n_req_after = len(gt.catalog())


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
tables, verdicts, benches, md_tables, results_all = {}, {}, {}, {}, {}
for h, known in [(0, False), (1, True)]:
    kw = dict(h=h, p=3, period=12, fourier_k=1, y_known_at_origin=known, min_train=MIN_TRAIN, transform="diff")
    res = {}
    res["naive"] = ev.naive(ya, h, MIN_TRAIN)
    res["seasonal_naive"] = ev.seasonal_naive(ya, h, 12, MIN_TRAIN)
    res["ar_diff"] = ev.rolling_backtest(ya, None, model="ar", use_gt=False, **kw)
    for name, spec in GT_MODELS.items():
        res[name] = ev.rolling_backtest(ya, Xa, model=spec["model"], use_gt=True, xlags=1,
                                        preprocess=spec["preprocess"], **kw)
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
    return {"origin": str(last), "target_date": str(tgt), "point": float(ya.loc[last]),
            "model": "naive", "n_train": int(len(ya)), "features": None}


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
        fc = ev.forecast_next(ya, Xp, model=GT_MODELS[win]["model"], use_gt=True,
                              preprocess=GT_MODELS[win]["preprocess"], **common)
    else:
        fc = target_only_forecast(win, h)
    refs = {}
    for name, spec in GT_MODELS.items():
        fc_gt = ev.forecast_next(ya, Xp, model=spec["model"], use_gt=True, preprocess=spec["preprocess"], **common)
        q = ev.conformal_interval(res[name].errors.values, alpha=0.1, recent=36)
        refs[name] = {**fc_gt, "q90_recent36": float(q), "lo": float(fc_gt["point"] - q), "hi": float(fc_gt["point"] + q)}
    forecast[f"h{h}"] = {"winner": win, **fc,
                         "q90_recent36": float(q90), "lo": float(fc["point"] - q90), "hi": float(fc["point"] + q90),
                         "q90_all_origins": float(q90_all), "gt_reference": refs}
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
truth_meta = json.loads((truth.TRUTH_DIR / "cbs_unemployment_rate_il.meta.json").read_text(encoding="utf-8"))
out = {"id": ID, "question": "מה יהיה שיעור האבטלה בישראל בחודש הבא? (What will Israel's unemployment rate be next month?)",
       "approach": "A (Choi–Varian minimal, Hebrew keywords, geo=IL)",
       "truth": {"series": "CBS LFS unemployment rate 15+, SA, % (sid 491094 spliced with 41097)",
                 "first": str(y.index.min().date()), "last": str(y.index.max().date()), "n": int(len(y)), "freq": "M",
                 "meta": truth_meta},
       "queries": KWS, "queries_kept": list(X.columns), "queries_dropped": info.get("dropped", []),
       "queries_failed": info.get("failed", []), "nonzero_share_2004": nonzero_all, "nonzero_share_2012": nonzero_2012,
       "suggestions": sugg,
       "gt_range": {"first": str(X.index.min().date()), "last": str(X.index.max().date()), "n": int(len(X))},
       "design": {"p": 3, "xlags": 1, "period": 12, "fourier_k": 1, "min_train": MIN_TRAIN, "max_origins": None,
                  "transform": "diff", "window": "expanding",
                  "gt_models": {"ols+rawGT": "ols + raw_pipeline", "ridge+djorno": "ridge + djorno_pipeline(period=12, halflife=1.0)"},
                  "benchmarks": BENCH_NAMES, "exclude": list(EXCL)},
       "bench_per_view": benches, "tables": tables, "verdicts": verdicts, "forecast": forecast,
       "interval_coverage_backtest": cov,
       "requests": int(gt.REQUEST_COUNT - live_before), "catalog_delta": int(n_req_after - n_req_before),
       "catalog_size": int(n_req_after), "seconds": round(seconds, 1)}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False, default=str)
print("\n\n# markdown tables for the report")
for k, t in md_tables.items():
    print(f"\n### {k} (bench = {benches[k]})\n\n{t}\n")
print(f"\nlive GT calls this process: {gt.REQUEST_COUNT - live_before}; catalog {n_req_before} -> {n_req_after}; elapsed {seconds:.0f}s")
print(f"wrote {OUT_JSON}")
