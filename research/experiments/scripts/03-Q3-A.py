"""Experiment cell 03-Q3-A — Q3 "Will the S&P 500 be higher one month from now?", Approach A.

Run from the repo root:  PYTHONIOENCODING=utf-8 python research/experiments/scripts/03-Q3-A.py

Design (fixed by the cell):
  * 8 hand-picked English keywords, geo=US, monthly (gt.timeframe_for("M")), individual pulls
  * truth: y = log(monthly last close of ^GSPC, Yahoo); the current, incomplete month is dropped
    from BOTH y and X (its close and its GT value are partial-period values)
  * target = one-month log return (transform="diff" on log levels); naive = zero expected return
  * benchmarks: naive (random walk), seasonal_naive, AR(2) in differences
  * GT models: ols / ridge  x  {raw (log1p), Djorno(period=12, halflife=1), diff (log1p -> first diff)}
  * h=1, y_known_at_origin=True, p=2, xlags=1, period=12, fourier_k=0, min_train=60, expanding
  * views: full sample and exclude=("2020-02-01","2020-06-30")
  * extra: hit-rate (DirAcc) with a binomial test vs 0.5 and vs the "always up" base rate
  * forward forecast with ev.forecast_next (winning config) + 90% conformal interval on the log return
"""
import sys, time, json, math
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy import stats
from trends_predict import gt, truth, preprocess as pp, evaluate as ev

ID = "03-Q3-A"
OUT_JSON = f"research/experiments/{ID}.json"
EXCL = ("2020-02-01", "2020-06-30")
# Attention / fear terms (Preis-Moat-Stanley 2013; Da-Engelberg-Gao FEARS), explicit trading-
# intent terms, macro-risk terms and a safe-haven proxy. "unemployment" is already cached (Q1).
KWS = ["stock market", "recession", "market crash", "buy stocks", "sell stocks",
       "inflation", "gold price", "unemployment"]
BENCH_NAMES = ["naive", "rw_drift", "seasonal_naive", "ar_diff"]
H, P, XLAGS, PERIOD, FK, MIN_TRAIN = 1, 2, 1, 12, 0, 60


def rw_drift(y: pd.Series, h: int, min_train: int) -> ev.BacktestResult:
    """Random walk with drift: predict y_t + h * mean(diff(y)) estimated on data up to the origin.
    Added to the target-only set because the equity index has a positive unconditional drift;
    the plain naive (zero return) is too weak a benchmark for a *drifting* series (a GT model can
    'beat' it simply by learning the intercept)."""
    y = y.astype(float).sort_index()
    rows = []
    for t in range(min_train, len(y) - h):
        drift = float(y.iloc[:t + 1].diff().dropna().mean())
        rows.append({"target_date": y.index[t + h], "origin": y.index[t], "y_true": y.iloc[t + h],
                     "y_pred": y.iloc[t] + h * drift})
    return ev.BacktestResult("rw_drift", pd.DataFrame(rows).set_index("target_date"), h)

PIPES = {
    "raw": pp.raw_pipeline,
    "djorno": lambda: pp.djorno_pipeline(period=12, halflife=1.0),
    "diffGT": lambda: pp.Pipeline([pp.ZeroRepair(), pp.Log1p(), pp.Detrend("diff")]),
}
GT_MODELS = {f"{m}+{pn}": (m, pn) for pn in PIPES for m in ("ols", "ridge")}

t0 = time.time()

def own_requests() -> int:
    """Catalog rows that belong to THIS cell's keywords (other cells write to the same catalog
    concurrently, so a plain len(catalog) delta over-counts)."""
    c = gt.catalog()
    if c.empty:
        return 0
    mine = {str([k]) for k in KWS}
    return int(c["keywords"].astype(str).isin(mine).sum())

n_req_before = own_requests()
n_cat_before = len(gt.catalog())

# ---------------------------------------------------------------- data
X, info = gt.fetch_many(KWS, geo="US", timeframe=gt.timeframe_for("M"))
px = truth.yahoo("^GSPC", freq="M")               # month-start index, value = last close of that month
y = np.log(px.astype(float))
n_req_after = own_requests()
n_cat_after = len(gt.catalog())

cur_month = pd.Timestamp.today().to_period("M")
y_full_last, x_full_last = y.index.max(), X.index.max()
y = y[y.index.to_period("M") < cur_month]          # drop the incomplete current month
X = X[X.index.to_period("M") < cur_month]
ya, Xa = truth.align(y, X, "M")
print(f"GT columns: {list(X.columns)}  dropped: {info.get('dropped')}  failed: {info.get('failed')}")
print(f"raw last dates: y {y_full_last.date()}  X {x_full_last.date()}  -> current month {cur_month} dropped")
print(f"X range {X.index.min().date()} -> {X.index.max().date()} ({len(X)} rows); "
      f"y range {y.index.min().date()} -> {y.index.max().date()} ({len(y)} obs)")
print(f"aligned: {len(ya)} months {ya.index.min()} -> {ya.index.max()}")
nonzero = (X > 0).mean().round(3).to_dict()
print("non-zero share per query:", nonzero)

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

def direction_table(res: dict, common: pd.Index) -> pd.DataFrame:
    """Hit-rate of sign(predicted return) vs sign(realised return) on the common target dates,
    plus binomial tests vs 0.5 and vs the 'always up' base rate."""
    rows = []
    ref = res["naive"].preds.loc[common]
    y_origin = ya.reindex(pd.Index(ref["origin"].values)).values
    r_true = ref["y_true"].values - y_origin
    up = r_true > 0
    base = float(up.mean())
    n_all = int(len(r_true))
    rows.append({"model": "always_up", "n": n_all, "k": int(up.sum()), "hit_rate": base,
                 "p_vs_0.5": stats.binomtest(int(up.sum()), n_all, 0.5).pvalue, "p_vs_always_up": np.nan,
                 "share_pred_up": 1.0})
    for name, r in res.items():
        if name == "naive":
            continue
        p = r.preds.loc[common]
        yo = ya.reindex(pd.Index(p["origin"].values)).values
        r_pred = p["y_pred"].values - yo
        ok = (r_true != 0) & (np.round(r_pred, 10) != 0)
        k = int(np.sum(np.sign(r_pred[ok]) == np.sign(r_true[ok])))
        n = int(ok.sum())
        rows.append({"model": name, "n": n, "k": k, "hit_rate": k / n if n else np.nan,
                     "p_vs_0.5": stats.binomtest(k, n, 0.5).pvalue if n else np.nan,
                     "p_vs_always_up": stats.binomtest(k, n, base).pvalue if n else np.nan,
                     "share_pred_up": float(np.mean(r_pred[ok] > 0)) if n else np.nan})
    return pd.DataFrame(rows).set_index("model")

# ---------------------------------------------------------------- backtests
kw = dict(h=H, p=P, period=PERIOD, fourier_k=FK, y_known_at_origin=True, min_train=MIN_TRAIN, transform="diff")
res = {}
res["naive"] = ev.naive(ya, H, MIN_TRAIN)
res["rw_drift"] = rw_drift(ya, H, MIN_TRAIN)
res["seasonal_naive"] = ev.seasonal_naive(ya, H, PERIOD, MIN_TRAIN)
res["ar_diff"] = ev.rolling_backtest(ya, None, model="ar", use_gt=False, **kw)
for name, (m, pn) in GT_MODELS.items():
    res[name] = ev.rolling_backtest(ya, Xa, model=m, use_gt=True, xlags=XLAGS, preprocess=PIPES[pn], name=name, **kw)
    print(f"  backtest {name}: {len(res[name].preds)} origins")

tables, verdicts, benches, md_tables, dir_tables, dir_md = {}, {}, {}, {}, {}, {}
for view, excl in [("full", None), ("excl", EXCL)]:
    key = f"h{H}_{view}"
    pre = ev.compare({k: res[k] for k in BENCH_NAMES}, bench="naive", exclude=excl)
    bench = pre["RMSE"].idxmin()
    tab = ev.compare(res, bench=bench, y_hist=ya, period=PERIOD, exclude=excl)
    v = ev.verdict(tab, bench, list(GT_MODELS))
    # direction stats on the same common dates the compare table used
    common = None
    for r in res.values():
        common = r.preds.index if common is None else common.intersection(r.preds.index)
    common = common.sort_values()
    if excl:
        ts = common.to_timestamp()
        common = common[(ts < pd.Timestamp(excl[0])) | (ts > pd.Timestamp(excl[1]))]
    dtab = direction_table(res, common)
    tables[key], verdicts[key], benches[key] = records(tab), v, bench
    md_tables[key], dir_tables[key], dir_md[key] = md_table(tab), records(dtab), md_table(dtab, 3)
    print(f"\n=== h={H} view={view} bench={bench} (n={int(tab['n'].iloc[0])}) "
          f"targets {common.min()} -> {common.max()} ===")
    print(tab.round(3).to_string())
    print("verdict:", v)
    print("direction table:")
    print(dtab.round(3).to_string())

# ---------------------------------------------------------------- forward forecast
def winner():
    if verdicts[f"h{H}_full"]["gt_helps"] and verdicts[f"h{H}_excl"]["gt_helps"]:
        return verdicts[f"h{H}_full"]["best_model"]
    return benches[f"h{H}_full"]

win = winner()
last = ya.dropna().index[-1]
tgt = last + H
common_fc = dict(h=H, p=P, xlags=XLAGS, period=PERIOD, fourier_k=FK, y_known_at_origin=True, transform="diff")
if win in GT_MODELS:
    m, pn = GT_MODELS[win]
    fc = ev.forecast_next(ya, Xa, model=m, use_gt=True, preprocess=PIPES[pn], **common_fc)
elif win == "ar_diff":
    fc = ev.forecast_next(ya, None, model="ar", use_gt=False, **common_fc)
elif win == "seasonal_naive":
    fc = {"origin": str(last), "target_date": str(tgt), "point": float(ya.loc[tgt - 12]), "model": "seasonal_naive",
          "n_train": int(len(ya)), "features": None}
elif win == "rw_drift":
    drift = float(ya.diff().dropna().mean())
    fc = {"origin": str(last), "target_date": str(tgt), "point": float(ya.loc[last]) + H * drift, "model": "rw_drift",
          "n_train": int(len(ya)), "features": [("drift_per_month", drift)]}
else:
    fc = {"origin": str(last), "target_date": str(tgt), "point": float(ya.loc[last]), "model": "naive",
          "n_train": int(len(ya)), "features": None}

def wrap(fc: dict, errs: pd.Series) -> dict:
    q90 = ev.conformal_interval(errs.values, alpha=0.1, recent=36)
    q90_all = ev.conformal_interval(errs.values, alpha=0.1)
    origin_level = float(ya.loc[last])
    ret = fc["point"] - origin_level
    return {**fc, "origin_close": float(math.exp(origin_level)), "point_close": float(math.exp(fc["point"])),
            "pred_log_return": float(ret), "pred_return_pct": float(100 * (math.exp(ret) - 1)),
            "q90_recent36_logret": float(q90), "lo_close": float(math.exp(fc["point"] - q90)),
            "hi_close": float(math.exp(fc["point"] + q90)), "q90_all_origins_logret": float(q90_all)}

forecast = {"winner": win, **wrap(fc, res[win].errors)}
# GT reference forecasts (not the headline unless they won) — the best GT model in each view
gt_ref = {}
for view in ("full", "excl"):
    bm = verdicts[f"h{H}_{view}"]["best_model"]
    m, pn = GT_MODELS[bm]
    fcg = ev.forecast_next(ya, Xa, model=m, use_gt=True, preprocess=PIPES[pn], **common_fc)
    gt_ref[view] = {"best_gt_model": bm, **wrap(fcg, res[bm].errors)}
forecast["gt_reference"] = gt_ref
# unconditional context: share of up-months over the evaluation window and in the full history
r_all = ya.diff().dropna()
forecast["context"] = {"share_up_months_full_history": float((r_all > 0).mean()),
                       "share_up_months_last_120": float((r_all.tail(120) > 0).mean()),
                       "mean_monthly_log_return_full_history": float(r_all.mean()),
                       "sd_monthly_log_return_full_history": float(r_all.std())}
print(f"\n--- forward forecast h={H}: winner={win} ---")
print(json.dumps(forecast, indent=1, default=str))

def rolling_coverage(errs: pd.Series, alpha=0.1, recent=36):
    e = errs.values; hits = []
    for i in range(recent, len(e)):
        q = ev.conformal_interval(e[:i], alpha=alpha, recent=recent)
        hits.append(abs(e[i]) <= q)
    return float(np.mean(hits)) if hits else float("nan")
cov = {win: rolling_coverage(res[win].errors)}
for view, g in gt_ref.items():
    cov[g["best_gt_model"]] = rolling_coverage(res[g["best_gt_model"]].errors)
print("rolling empirical coverage of the 90% conformal rule:", cov)

seconds = time.time() - t0
out = {"id": ID, "question": "Will the S&P 500 be higher one month from now?", "approach": "A (minimal predictive regression, direction)",
       "truth": {"series": "Yahoo Finance ^GSPC monthly last close, log", "first": str(y.index.min().date()),
                 "last": str(y.index.max().date()), "n": int(len(y)), "freq": "M",
                 "dropped_partial_month": str(cur_month)},
       "queries": KWS, "queries_dropped": info.get("dropped", []), "queries_failed": info.get("failed", []),
       "nonzero_share": nonzero,
       "gt_range": {"first": str(X.index.min().date()), "last": str(X.index.max().date()), "n": int(len(X))},
       "design": {"h": H, "p": P, "xlags": XLAGS, "period": PERIOD, "fourier_k": FK, "min_train": MIN_TRAIN,
                  "transform": "diff (log return)", "window": "expanding", "y_known_at_origin": True,
                  "preprocess": list(PIPES), "models": list(GT_MODELS), "benchmarks": BENCH_NAMES, "exclude": list(EXCL)},
       "bench_per_view": benches, "tables": tables, "direction_tables": dir_tables, "verdicts": verdicts,
       "forecast": forecast, "interval_coverage_backtest": cov,
       "requests": int(n_req_after - n_req_before), "requests_note": "own-keyword catalog rows; 'unemployment' was already cached by cell 01",
       "own_keyword_vintages_total": int(n_req_after), "catalog_size": int(n_cat_after), "seconds": round(seconds, 1)}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False, default=str)
with open(f"research/experiments/scripts/{ID}-tables.md", "w", encoding="utf-8") as f:
    for k in md_tables:
        f.write(f"\n### {k} (bench = {benches[k]})\n\n{md_tables[k]}\n\nverdict: `{json.dumps(verdicts[k], default=str)}`\n")
        f.write(f"\n#### direction {k}\n\n{dir_md[k]}\n")
print(f"\nGT requests this run (own keywords): {n_req_after - n_req_before}  (own vintages {n_req_before} -> {n_req_after}; "
      f"whole catalog {n_cat_before} -> {n_cat_after});  elapsed {seconds:.0f}s")
print(f"wrote {OUT_JSON}")
