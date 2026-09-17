"""Experiment cell 01-Q1-A — Q1 US unemployment next month, Approach A (Choi–Varian minimal).

Run from the repo root:  PYTHONIOENCODING=utf-8 python research/experiments/scripts/01-Q1-A.py

Design (fixed by the cell):
  * 5 hand-picked English keywords, geo=US, monthly (gt.timeframe_for("M")), individual pulls
  * preprocessing pp.raw_pipeline (ZeroRepair -> log1p), fitted inside each training window
  * GT model: OLS ARX  vs  benchmarks {naive, seasonal_naive, ar} (all in first differences)
  * expanding window, transform="diff", p=3, xlags=1, period=12, fourier_k=1, min_train=60, max_origins=120
  * horizons: h=0 nowcast (y_known_at_origin=False) and h=1 (y_known_at_origin=True)
  * views: full sample and exclude=("2020-03-01","2020-12-31")
  * bench per view = lowest-RMSE target-only model in that view (brief: "the best target-only model")
  * forward forecast with ev.forecast_next using the winning configuration + 90% conformal interval
"""
import sys, time, json, math
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from trends_predict import gt, truth, preprocess as pp, evaluate as ev

ID = "01-Q1-A"
OUT_JSON = f"research/experiments/{ID}.json"
EXCL = ("2020-03-01", "2020-12-31")
KWS = ["unemployment", "unemployment benefits", "file for unemployment",
       "unemployment office", "unemployment claims"]  # Choi–Varian "Welfare & Unemployment" flavour
BENCH_NAMES = ["naive", "seasonal_naive", "ar_diff"]
GT_NAME = "ols+rawGT"

t0 = time.time()
n_req_before = len(gt.catalog())

# ---------------------------------------------------------------- data
X, info = gt.fetch_many(KWS, geo="US", timeframe=gt.timeframe_for("M"))
y = truth.bls_series("UNRATE")
ya, Xa = truth.align(y, X, "M")
n_req_after = len(gt.catalog())
print(f"GT columns: {list(X.columns)}  dropped: {info.get('dropped')}")
print(f"X range {X.index.min().date()} -> {X.index.max().date()} ({len(X)} rows); "
      f"y range {y.index.min().date()} -> {y.index.max().date()} ({len(y)} obs)")
print(f"aligned: {len(ya)} months {ya.index.min()} -> {ya.index.max()}")
nonzero = (X > 0).mean().round(3).to_dict()
print("non-zero share per query:", nonzero)

# ---------------------------------------------------------------- backtests
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

tables, verdicts, benches, md_tables, results_all = {}, {}, {}, {}, {}
for h, known in [(0, False), (1, True)]:
    kw = dict(h=h, p=3, period=12, fourier_k=1, y_known_at_origin=known,
              min_train=60, max_origins=120, transform="diff")
    res = {}
    res["naive"] = ev.naive(ya, h, 60)
    res["seasonal_naive"] = ev.seasonal_naive(ya, h, 12, 60)
    res["ar_diff"] = ev.rolling_backtest(ya, None, model="ar", use_gt=False, **kw)
    res[GT_NAME] = ev.rolling_backtest(ya, Xa, model="ols", use_gt=True, xlags=1,
                                       preprocess=pp.raw_pipeline, **kw)
    results_all[h] = res
    for view, excl in [("full", None), ("excl", EXCL)]:
        key = f"h{h}_{view}"
        # pick the best target-only model in this view as the bench
        pre = ev.compare({k: res[k] for k in BENCH_NAMES}, bench="naive", exclude=excl)
        bench = pre["RMSE"].idxmin()
        tab = ev.compare(res, bench=bench, y_hist=ya, period=12, exclude=excl)
        v = ev.verdict(tab, bench, [GT_NAME])
        tables[key], verdicts[key], benches[key] = records(tab), v, bench
        md_tables[key] = md_table(tab)
        print(f"\n=== h={h} y_known={known} view={view} bench={bench} (n={int(tab['n'].iloc[0])}) ===")
        print(tab.round(3).to_string())
        print("verdict:", v)

# ---------------------------------------------------------------- forward forecast
# Winning configuration: GT model only if the verdict says it helps in BOTH views for that horizon;
# otherwise the target-only bench of the full-sample view.
def winner(h):
    if verdicts[f"h{h}_full"]["gt_helps"] and verdicts[f"h{h}_excl"]["gt_helps"]:
        return GT_NAME
    return benches[f"h{h}_full"]

forecast = {}
last_y_date = ya.dropna().index.max()
x_beyond_y = Xa.index.max() > last_y_date
print(f"\nX extends beyond y? {x_beyond_y}  (X last {Xa.index.max()}, y last {last_y_date})")
for h, known in [(0, False), (1, True)]:
    win = winner(h)
    res = results_all[h]
    # errors used for the conformal interval: the winning model's own backtest residuals
    q90 = ev.conformal_interval(res[win].errors.values, alpha=0.1, recent=36)
    q90_all = ev.conformal_interval(res[win].errors.values, alpha=0.1)
    common = dict(h=h, p=3, xlags=1, period=12, fourier_k=1, y_known_at_origin=known, transform="diff")
    if win == GT_NAME:
        fc = ev.forecast_next(ya, Xa, model="ols", use_gt=True, preprocess=pp.raw_pipeline, **common)
    elif win == "ar_diff":
        fc = ev.forecast_next(ya, None, model="ar", use_gt=False, **common)
    elif win == "seasonal_naive":
        last = ya.dropna().index[-1]
        tgt = last + max(h, 1)
        fc = {"origin": str(last), "target_date": str(tgt), "point": float(ya.loc[tgt - 12]),
              "model": "seasonal_naive", "n_train": int(len(ya)), "features": None}
    else:  # naive
        last = ya.dropna().index[-1]
        tgt = last + max(h, 1)
        fc = {"origin": str(last), "target_date": str(tgt), "point": float(ya.loc[last]),
              "model": "naive", "n_train": int(len(ya)), "features": None}
    # Also always compute the GT model's forward number for reference (not the headline unless it won).
    fc_gt = ev.forecast_next(ya, Xa, model="ols", use_gt=True, preprocess=pp.raw_pipeline, **common)
    q90_gt = ev.conformal_interval(res[GT_NAME].errors.values, alpha=0.1, recent=36)
    forecast[f"h{h}"] = {
        "winner": win, **fc,
        "q90_recent36": float(q90), "lo": float(fc["point"] - q90), "hi": float(fc["point"] + q90),
        "q90_all_origins": float(q90_all),
        "gt_reference": {**fc_gt, "q90_recent36": float(q90_gt),
                         "lo": float(fc_gt["point"] - q90_gt), "hi": float(fc_gt["point"] + q90_gt)},
    }
    print(f"\n--- forward forecast h={h}: winner={win} ---")
    print(json.dumps(forecast[f"h{h}"], indent=1, default=str))

# backtest coverage check of the 90% interval (recent-36 rule applied rolling) for the h=1 winner
def rolling_coverage(errs: pd.Series, alpha=0.1, recent=36):
    e = errs.values; hits = []
    for i in range(recent, len(e)):
        q = ev.conformal_interval(e[:i], alpha=alpha, recent=recent)
        hits.append(abs(e[i]) <= q)
    return float(np.mean(hits)) if hits else float("nan")
cov = {f"h{h}": rolling_coverage(results_all[h][forecast[f'h{h}']['winner']].errors) for h in (0, 1)}
print("rolling empirical coverage of the 90% conformal rule:", cov)

seconds = time.time() - t0
out = {"id": ID, "question": "What will the US unemployment rate be next month?", "approach": "A (Choi–Varian minimal)",
       "truth": {"series": "BLS LNS14000000 (UNRATE)", "first": str(y.index.min().date()), "last": str(y.index.max().date()),
                 "n": int(len(y)), "freq": "M"},
       "queries": KWS, "queries_dropped": info.get("dropped", []), "nonzero_share": nonzero,
       "gt_range": {"first": str(X.index.min().date()), "last": str(X.index.max().date()), "n": int(len(X))},
       "design": {"p": 3, "xlags": 1, "period": 12, "fourier_k": 1, "min_train": 60, "max_origins": 120,
                  "transform": "diff", "window": "expanding", "preprocess": "raw_pipeline", "model": "ols",
                  "benchmarks": BENCH_NAMES, "exclude": list(EXCL)},
       "bench_per_view": benches, "tables": tables, "verdicts": verdicts, "forecast": forecast,
       "interval_coverage_backtest": cov,
       "requests": int(n_req_after - n_req_before), "catalog_size": int(n_req_after), "seconds": round(seconds, 1)}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False, default=str)
print("\n\n# markdown tables for the report")
for k, t in md_tables.items():
    print(f"\n### {k} (bench = {benches[k]})\n\n{t}\n")
print(f"\nGT requests this run: {n_req_after - n_req_before}  (catalog {n_req_before} -> {n_req_after});  elapsed {seconds:.0f}s")
print(f"wrote {OUT_JSON}")
