"""Experiment cell 03-Q3-B — Q3 "Will the S&P 500 be higher one month from now?", Approach B
(ARGO-style breadth at weekly frequency with related-query expansion).

Run from the repo root:  PYTHONIOENCODING=utf-8 python research/experiments/scripts/03-Q3-B.py

Design (fixed by the cell):
  * seeds -> gt.expand_queries(max_per_seed=10) -> hand-filtered drop list (documented below)
  * individual weekly pulls, geo=US, timeframe="today 5-y"
  * truth: y = log(weekly last close of ^GSPC, Yahoo); the current partial week is dropped
  * target = 4-week-ahead log change: h=4, transform="diff", y_known_at_origin=True
  * p=2 AR lags (weekly log returns), xlags=1, fourier_k=0, period=52, min_train=104
  * benchmarks: naive (zero change), drift (random walk + expanding-mean drift), ar_diff
  * GT models: lasso / enet  x  {raw (ZeroRepair->log1p), diffGT (ZeroRepair->log1p->first diff)}
                             x  {expanding, sliding window=104}
  * bench = lowest-MAE target-only model (expected: drift)
  * views: full sample, exclude=("2020-02-01","2020-06-30") (vacuous: sample starts 2021-09),
    plus an extra shock exclusion for the April-2025 tariff crash ("2025-03-01","2025-05-31")
  * direction: hit-rate vs 0.5 and vs the always-up base rate (binomial tests; caveat: the 4-week
    targets overlap, so consecutive errors are MA(3)-autocorrelated -> DM/CW use HAC lag h-1=3,
    the binomial tests do NOT correct for it and are anti-conservative)
  * LASSO selected features at recent origins + selection frequency over all origins
  * forward forecast (winning config) 4 weeks ahead + 90% conformal interval (recent 52 residuals)
  * alignment note: truth.yahoo(freq="W") labels the Friday close with the FOLLOWING Sunday and
    truth.align("W") maps a Sunday to the Sat-ending week that STARTS on it, so as-specified the
    origin GT week would be the week AFTER the close (one-week look-ahead). We shift the price
    labels back one day (Sunday -> Saturday) so the close sits in its own Sun-Sat week; the
    as-specified alignment is run once as a leakage-sensitivity check.
"""
import sys, time, json, math
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy import stats
from trends_predict import gt, truth, preprocess as pp, evaluate as ev

ID = "03-Q3-B"
OUT_JSON = f"research/experiments/{ID}.json"
GEO, TF = "US", "today 5-y"
EXCL = ("2020-02-01", "2020-06-30")           # mandated by the cell (vacuous for a 5-y weekly sample)
EXCL2 = ("2025-03-01", "2025-05-31")          # extra: April-2025 tariff crash + rebound
H, P, XLAGS, PERIOD, FK, MIN_TRAIN, WINDOW = 4, 2, 1, 52, 0, 104, 104
SEEDS = ["stock market", "recession", "stock market crash", "S&P 500", "sell stocks"]

# Drop list for the expansion output (52 candidates), with the reason. Kept terms must be
# contemporaneous market-attention / fear / trading-intent terms.
DROP = {
    # dental, not economics
    "gum recession": "irrelevant (dental)",
    # historical / homework terms: driven by the school calendar, not by current markets
    "stock market crash 1929": "historical/homework", "great stock market crash": "historical/homework",
    "great depression stock market crash": "historical/homework", "great depression": "historical/homework",
    "stock market crash of 1929": "historical/homework", "the great depression stock market crash": "historical/homework",
    "the great depression": "historical/homework", "great recession": "historical/homework",
    # pure definitional / educational
    "what is the stock market": "educational", "what is recession": "educational (dup of 'what is a recession')",
    "what is s&p 500": "educational", "what are stocks": "educational",
    # single-word / generic tokens dominated by unrelated meanings or too broad
    "stock": "generic token", "etf": "generic token", "dow": "ambiguous (Dow Inc.) / dup of 'dow jones'",
    # near-duplicates of a seed (article / word-order variants)
    "the stock market": "dup of seed", "today stock market": "dup of 'the stock market today'",
    "the recession": "dup of seed", "the stock market crash": "dup of seed",
    "stock market dow": "dup (stock market x dow jones)", "s&p 500 stock": "malformed dup of seed",
    "s&p 500 price": "dup of 's&p 500 today'",
}
BENCH_NAMES = ["naive", "drift", "ar_diff"]
PIPES = {
    "raw": pp.raw_pipeline,
    "diffGT": lambda: pp.Pipeline([pp.ZeroRepair(), pp.Log1p(), pp.Detrend("diff")]),
}
GT_MODELS = {}
for m in ("lasso", "enet"):
    for pn in PIPES:
        for wn, w in (("exp", None), ("w104", WINDOW)):
            GT_MODELS[f"{m}+{pn}+{wn}"] = (m, pn, w)

t0 = time.time()

# ---------------------------------------------------------------- queries
expanded = gt.expand_queries(SEEDS, geo=GEO, timeframe=TF, max_per_seed=10)   # 5 cached related_queries calls
KWS = [k for k in expanded if k not in DROP]
dropped = [{"query": k, "reason": DROP[k]} for k in expanded if k in DROP]
unused_drop = sorted(set(DROP) - set(expanded))
print(f"expansion: {len(expanded)} candidates -> {len(KWS)} kept, {len(dropped)} dropped; drop-list entries not in expansion: {unused_drop}")
print("kept:", KWS)

def own_requests() -> int:
    c = gt.catalog()
    if c.empty:
        return 0
    mine = {str([k]) for k in KWS}
    return int((c["keywords"].astype(str).isin(mine) & (c["timeframe"].astype(str) == TF)).sum())

n_req_before, n_cat_before = own_requests(), len(gt.catalog())

# ---------------------------------------------------------------- data
X, info = gt.fetch_many(KWS, geo=GEO, timeframe=TF)          # individual weekly pulls
n_req_after, n_cat_after = own_requests(), len(gt.catalog())
print(f"GT frame: {X.shape}, {X.index.min().date()} -> {X.index.max().date()}; dropped(privacy)={info.get('dropped')} failed={info.get('failed')}")

px = truth.yahoo("^GSPC", freq="W")                           # Sunday-labelled weekly last close (Friday close)
px = px.astype(float).sort_index()
# belt and braces: drop the current partial week if truth.yahoo did not
while px.index[-1].to_period("W-SUN").end_time > pd.Timestamp.now():
    px = px.iloc[:-1]
y_sun = np.log(px)
print(f"price: {len(y_sun)} weeks, {y_sun.index.min().date()} -> {y_sun.index.max().date()} (last close {px.iloc[-1]:.2f})")

# alignment fix: Sunday label -> Saturday label so the Friday close maps to ITS OWN Sun-Sat week
y_sat = y_sun.copy(); y_sat.index = y_sat.index - pd.Timedelta(days=1)
ya, Xa = truth.align(y_sat, X, "W")
ya_spec, Xa_spec = truth.align(y_sun, X, "W")                 # as specified (GT week AFTER the close) - leakage check
print(f"aligned (fixed): {len(ya)} weeks {ya.index.min()} -> {ya.index.max()};  as-specified: {len(ya_spec)} weeks {ya_spec.index.min()} -> {ya_spec.index.max()}")
nonzero = (X > 0).mean().round(3).to_dict()
print("min non-zero share:", min(nonzero.values()))

# ---------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame, floatfmt=3) -> str:
    cols = list(df.columns)
    out = ["| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if (v != v) else (f"{v:.{floatfmt}f}" if c not in ("n", "k") else f"{int(v)}"))
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

def common_index(res: dict, excl):
    common = None
    for r in res.values():
        common = r.preds.index if common is None else common.intersection(r.preds.index)
    common = common.sort_values()
    if excl:
        ts = common.to_timestamp()
        common = common[(ts < pd.Timestamp(excl[0])) | (ts > pd.Timestamp(excl[1]))]
    return common

def direction_table(res: dict, y: pd.Series, common: pd.Index) -> pd.DataFrame:
    """Hit-rate of sign(predicted 4-week change) vs sign(realised) on the common target dates,
    binomial tests vs 0.5 and vs the always-up base rate. NB: overlapping targets -> the binomial
    n overstates the effective sample (errors are MA(h-1)); read p-values as anti-conservative."""
    rows = []
    ref = res["naive"].preds.loc[common]
    y_origin = y.reindex(pd.Index(ref["origin"].values)).values
    r_true = ref["y_true"].values - y_origin
    up = r_true > 0
    base = float(up.mean()); n_all = int(len(r_true))
    rows.append({"model": "always_up", "n": n_all, "k": int(up.sum()), "hit_rate": base,
                 "p_vs_0.5": stats.binomtest(int(up.sum()), n_all, 0.5).pvalue, "p_vs_always_up": np.nan, "share_pred_up": 1.0})
    for name, r in res.items():
        if name == "naive":
            continue
        p = r.preds.loc[common]
        yo = y.reindex(pd.Index(p["origin"].values)).values
        r_pred = p["y_pred"].values - yo
        ok = (r_true != 0) & (np.round(r_pred, 10) != 0)
        k = int(np.sum(np.sign(r_pred[ok]) == np.sign(r_true[ok]))); n = int(ok.sum())
        rows.append({"model": name, "n": n, "k": k, "hit_rate": k / n if n else np.nan,
                     "p_vs_0.5": stats.binomtest(k, n, 0.5).pvalue if n else np.nan,
                     "p_vs_always_up": stats.binomtest(k, n, base).pvalue if n else np.nan,
                     "share_pred_up": float(np.mean(r_pred[ok] > 0)) if n else np.nan})
    return pd.DataFrame(rows).set_index("model")

# ---------------------------------------------------------------- backtests
kw = dict(h=H, p=P, period=PERIOD, fourier_k=FK, y_known_at_origin=True, min_train=MIN_TRAIN, transform="diff")

def run_all(y, Xf, tag=""):
    res = {}
    res["naive"] = ev.naive(y, H, MIN_TRAIN)
    res["drift"] = ev.drift(y, H, MIN_TRAIN)
    res["ar_diff"] = ev.rolling_backtest(y, None, model="ar", use_gt=False, **kw)
    for name, (m, pn, w) in GT_MODELS.items():
        t1 = time.time()
        res[name] = ev.rolling_backtest(y, Xf, model=m, use_gt=True, xlags=XLAGS, preprocess=PIPES[pn], window=w, name=name, **kw)
        print(f"  backtest{tag} {name}: {len(res[name].preds)} origins ({time.time()-t1:.0f}s)")
    return res

res = run_all(ya, Xa)

VIEWS = [("full", None), ("excl2020", EXCL), ("excl_tariff2025", EXCL2)]
tables, verdicts, benches, md_tables, dir_tables, dir_md, view_meta = {}, {}, {}, {}, {}, {}, {}
for view, excl in VIEWS:
    key = f"h{H}_{view}"
    pre = ev.compare({k: res[k] for k in BENCH_NAMES}, bench="naive", exclude=excl)
    bench = pre["MAE"].idxmin()
    tab = ev.compare(res, bench=bench, y_hist=ya, period=PERIOD, exclude=excl)
    v = ev.verdict(tab, bench, list(GT_MODELS))
    common = common_index(res, excl)
    dtab = direction_table(res, ya, common)
    tables[key], verdicts[key], benches[key] = records(tab), v, bench
    md_tables[key], dir_tables[key], dir_md[key] = md_table(tab), records(dtab), md_table(dtab, 3)
    view_meta[key] = {"n": int(len(common)), "first_target": str(common.min()), "last_target": str(common.max()),
                      "exclude": list(excl) if excl else None}
    print(f"\n=== h={H} view={view} bench={bench} (n={len(common)}) targets {common.min()} -> {common.max()} ===")
    print(tab.round(3).to_string()); print("verdict:", v); print(dtab.round(3).to_string())

# ---------------------------------------------------------------- LASSO selections
def selection_summary(r: ev.BacktestResult, recent=6):
    sel = r.selected
    freq = {}
    for s in sel:
        for f in s["features"]:
            freq[f] = freq.get(f, 0) + 1
    n = len(sel)
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:15]
    return {"n_origins": n, "mean_n_selected": float(np.mean([len(s["features"]) for s in sel])) if n else None,
            "share_origins_empty": float(np.mean([len(s["features"]) == 0 for s in sel])) if n else None,
            "top_features_share": [(f, round(c / n, 3)) for f, c in top],
            "recent": sel[-recent:]}
selections = {name: selection_summary(res[name]) for name in GT_MODELS if name.startswith("lasso")}
for name, s in selections.items():
    print(f"\n{name}: mean #selected={s['mean_n_selected']:.2f}, empty share={s['share_origins_empty']:.2f}, top={s['top_features_share'][:8]}")
    for o in s["recent"]:
        print("   ", o["origin"], o["features"])

# ---------------------------------------------------------------- leakage-sensitivity check (as-specified alignment)
best_full = verdicts[f"h{H}_full"]["best_model"]
m_b, pn_b, w_b = GT_MODELS[best_full]
res_spec = {"naive": ev.naive(ya_spec, H, MIN_TRAIN), "drift": ev.drift(ya_spec, H, MIN_TRAIN),
            "ar_diff": ev.rolling_backtest(ya_spec, None, model="ar", use_gt=False, **kw),
            "lasso+raw+exp": ev.rolling_backtest(ya_spec, Xa_spec, model="lasso", use_gt=True, xlags=XLAGS, preprocess=PIPES["raw"], window=None, name="lasso+raw+exp", **kw)}
if best_full != "lasso+raw+exp":
    res_spec[best_full] = ev.rolling_backtest(ya_spec, Xa_spec, model=m_b, use_gt=True, xlags=XLAGS, preprocess=PIPES[pn_b], window=w_b, name=best_full, **kw)
pre_s = ev.compare({k: res_spec[k] for k in BENCH_NAMES}, bench="naive")
bench_s = pre_s["MAE"].idxmin()
tab_spec = ev.compare(res_spec, bench=bench_s, y_hist=ya_spec, period=PERIOD)
print(f"\n=== leakage check: as-specified alignment (GT week after the close), bench={bench_s} ===")
print(tab_spec.round(3).to_string())

# ---------------------------------------------------------------- forward forecast
def winner():
    if verdicts[f"h{H}_full"]["gt_helps"] and verdicts[f"h{H}_excl2020"]["gt_helps"]:
        return verdicts[f"h{H}_full"]["best_model"]
    return benches[f"h{H}_full"]

win = winner()
last = ya.dropna().index[-1]; tgt = last + H
common_fc = dict(h=H, p=P, xlags=XLAGS, period=PERIOD, fourier_k=FK, y_known_at_origin=True, transform="diff")
if win in GT_MODELS:
    m, pn, w = GT_MODELS[win]
    fc = ev.forecast_next(ya, Xa, model=m, use_gt=True, preprocess=PIPES[pn], window=w, **common_fc)
elif win == "ar_diff":
    fc = ev.forecast_next(ya, None, model="ar", use_gt=False, **common_fc)
elif win == "drift":
    mu = float(ya.diff().dropna().mean())
    fc = {"origin": str(last), "target_date": str(tgt), "point": float(ya.loc[last]) + H * mu, "model": "drift",
          "n_train": int(len(ya)), "features": [("drift_per_week", mu)]}
else:
    fc = {"origin": str(last), "target_date": str(tgt), "point": float(ya.loc[last]), "model": "naive", "n_train": int(len(ya)), "features": None}

def wrap(fc: dict, errs: pd.Series) -> dict:
    q90 = ev.conformal_interval(errs.values, alpha=0.1, recent=52)
    q90_all = ev.conformal_interval(errs.values, alpha=0.1)
    origin_level = float(ya.loc[last]); ret = fc["point"] - origin_level
    return {**fc, "origin_close": float(math.exp(origin_level)), "point_close": float(math.exp(fc["point"])),
            "pred_log_change_4w": float(ret), "pred_change_pct": float(100 * (math.exp(ret) - 1)),
            "q90_recent52_log": float(q90), "lo_close": float(math.exp(fc["point"] - q90)),
            "hi_close": float(math.exp(fc["point"] + q90)), "q90_all_origins_log": float(q90_all)}

forecast = {"winner": win, "origin_week_sat": str(last), "origin_close_date": str(px.index[-1].date() - pd.Timedelta(days=2)),
            "target_week_sat": str(tgt), **wrap(fc, res[win].errors)}
gt_ref = {}
for view, _ in VIEWS:
    bm = verdicts[f"h{H}_{view}"]["best_model"]
    if bm in gt_ref:
        continue
    m, pn, w = GT_MODELS[bm]
    fcg = ev.forecast_next(ya, Xa, model=m, use_gt=True, preprocess=PIPES[pn], window=w, **common_fc)
    feats = fcg.get("features")
    if isinstance(feats, (list, tuple)):
        feats = sorted(feats, key=lambda t: -abs(t[1]))[:12] if feats and isinstance(feats[0], (list, tuple)) else feats
    gt_ref[bm] = {**wrap(fcg, res[bm].errors), "features": feats}
forecast["gt_reference"] = gt_ref
r_all = ya.diff(H).dropna()
forecast["context"] = {"share_up_4w_windows_sample": float((r_all > 0).mean()), "mean_4w_log_change": float(r_all.mean()),
                       "sd_4w_log_change": float(r_all.std()), "n_4w_windows": int(len(r_all))}
print(f"\n--- forward forecast h={H} weeks: winner={win} ---")
print(json.dumps(forecast, indent=1, default=str))

def rolling_coverage(errs: pd.Series, alpha=0.1, recent=52):
    e = errs.values; hits = []
    for i in range(recent, len(e)):
        hits.append(abs(e[i]) <= ev.conformal_interval(e[:i], alpha=alpha, recent=recent))
    return float(np.mean(hits)) if hits else float("nan")
cov = {win: rolling_coverage(res[win].errors)}
for bm in gt_ref:
    cov[bm] = rolling_coverage(res[bm].errors)
print("rolling empirical coverage of the 90% conformal rule (recent-52):", cov)

# ---------------------------------------------------------------- outputs
seconds = time.time() - t0
out = {"id": ID, "question": "Will the S&P 500 be higher one month from now?",
       "approach": "B (ARGO-style breadth, weekly, related-query expansion, LASSO/ENet, expanding + sliding-104)",
       "truth": {"series": "Yahoo Finance ^GSPC weekly last close (Friday), log; labels shifted Sunday->Saturday so the close maps to its own Sun-Sat week",
                 "first": str(px.index.min().date()), "last_sunday_label": str(px.index.max().date()), "last_close": float(px.iloc[-1]),
                 "n": int(len(px)), "freq": "W"},
       "seeds": SEEDS, "expanded_candidates": expanded, "queries": KWS, "queries_dropped_by_hand": dropped,
       "queries_dropped_privacy": info.get("dropped", []), "queries_failed": info.get("failed", []), "nonzero_share": nonzero,
       "gt_range": {"first": str(X.index.min().date()), "last": str(X.index.max().date()), "n": int(len(X))},
       "aligned": {"n": int(len(ya)), "first": str(ya.index.min()), "last": str(ya.index.max())},
       "design": {"h": H, "p": P, "xlags": XLAGS, "period": PERIOD, "fourier_k": FK, "min_train": MIN_TRAIN, "window_sliding": WINDOW,
                  "transform": "diff (4-week log change)", "y_known_at_origin": True, "preprocess": list(PIPES),
                  "models": list(GT_MODELS), "benchmarks": BENCH_NAMES, "views": {k: v for k, v in view_meta.items()}},
       "bench_per_view": benches, "tables": tables, "direction_tables": dir_tables, "verdicts": verdicts,
       "lasso_selections": selections,
       "leakage_check_as_specified_alignment": {"bench": bench_s, "table": records(tab_spec)},
       "forecast": forecast, "interval_coverage_backtest": cov,
       "requests": int(n_req_after - n_req_before) + 5, "requests_note": "own-keyword weekly catalog rows + 5 related_queries calls (JSON-cached, not catalog rows)",
       "own_keyword_vintages_total": int(n_req_after), "catalog_size": int(n_cat_after), "seconds": round(seconds, 1)}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False, default=str)
with open(f"research/experiments/scripts/{ID}-tables.md", "w", encoding="utf-8") as f:
    for k in md_tables:
        f.write(f"\n### {k} (bench = {benches[k]}, {view_meta[k]})\n\n{md_tables[k]}\n\nverdict: `{json.dumps(verdicts[k], default=str)}`\n")
        f.write(f"\n#### direction {k}\n\n{dir_md[k]}\n")
    f.write(f"\n### leakage check (as-specified alignment, bench = {bench_s})\n\n{md_table(tab_spec)}\n")
print(f"\nGT requests this run (own weekly keywords): {n_req_after - n_req_before} (+5 related_queries); own vintages {n_req_before} -> {n_req_after}; "
      f"catalog {n_cat_before} -> {n_cat_after}; elapsed {seconds:.0f}s")
print(f"wrote {OUT_JSON}")
