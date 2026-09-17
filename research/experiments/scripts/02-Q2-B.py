"""Experiment cell 02-Q2-B — US ILI %, ARGO-style (Yang, Santillana & Kou 2015).

Run from the repo root:  PYTHONIOENCODING=utf-8 python research/experiments/scripts/02-Q2-B.py
Writes research/experiments/02-Q2-B.json and prints the markdown tables pasted into 02-Q2-B.md.
"""
import sys, time, json, os
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from trends_predict import gt, truth, preprocess as pp, evaluate as ev

T0 = time.time()
ID = "02-Q2-B"
GEO, TF = "US", "today 5-y"
SEEDS = ["flu", "flu symptoms", "influenza", "fever"]
# Clearly irrelevant to influenza-like illness (WNBA team, song titles, travel vaccine, idiom):
DROP = ["indiana fever", "fever game", "night fever", "fever up", "yellow fever", "baby fever"]
P, XLAGS, PERIOD, FK, MIN_TRAIN, WIN = 8, 1, 52, 2, 104, 104
HORIZONS = [0, 2, 4]
EXCLUDE = ("2020-03-01", "2021-06-30")
# The mandated window above predates the aligned sample (GT 'today 5-y' starts 2021-09) and the
# first origin is 2023-09, so it is a no-op here. Extra robustness view that IS inside the
# evaluation window: the severe 2024-25 peak (highest ILI since 2009-10).
EXCLUDE2 = ("2024-12-01", "2025-02-28")

log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

# ------------------------------------------------------------------ queries
n_req = 0
qdir = "data/queries"
before_q = set(os.listdir(qdir)) if os.path.isdir(qdir) else set()
terms_all = gt.expand_queries(SEEDS, geo=GEO, timeframe=TF, max_per_seed=12)
after_q = set(os.listdir(qdir))
n_req += len([f for f in after_q - before_q if f.startswith("related_queries")])
# related_queries calls that were cached before this run still count towards THIS cell's budget
# only if this cell created them (they were: first run of this cell). Record both numbers.
n_related_files = len([f for f in after_q if f.startswith("related_queries") and any(
    f"__{gt._slugify(s)[:40]}__" in f for s in SEEDS)])
terms = [t for t in terms_all if t.lower() not in DROP]
say(f"expanded terms: {len(terms_all)}, dropped {len(DROP)}: {DROP}, kept {len(terms)}")

def _cached(term):
    return (gt.GTRequest.make([term], GEO, TF).directory() / "meta.json").exists()
uncached = [t for t in terms if not _cached(t)]
X, info = gt.fetch_many(terms, geo=GEO, timeframe=TF, mode="individual")
n_pulls = len(uncached)
say(f"GT pulls this run: {n_pulls} live (of {len(terms)} terms; rest cached); failed={info['failed']}; "
    f"zero-filter dropped={info['dropped']}")
say(f"X raw: {X.shape}, {X.index.min().date()} -> {X.index.max().date()}")

# ------------------------------------------------------------------ truth + align
y = truth.fluview_ili("nat", 201040)
say(f"ILI wili nat: {len(y)} weeks {y.index.min().date()} -> {y.index.max().date()}, "
    f"missing weeks={len(pd.date_range(y.index.min(), y.index.max(), freq='W-SAT').difference(y.index))}, NaN={int(y.isna().sum())}")
ya, Xa = truth.align(y, X, "W")
full_idx = pd.period_range(ya.index.min(), ya.index.max(), freq="W-SAT")
say(f"aligned: {len(ya)} weeks {ya.index.min()} -> {ya.index.max()}; period gaps={len(full_idx.difference(ya.index))}; "
    f"y NaN={int(ya.isna().sum())}; X NaN cells={int(Xa.isna().sum().sum())}; "
    f"X zero share max={float((Xa <= 0).mean().max()):.3f} ({(Xa <= 0).mean().idxmax()})")
# GT weeks beyond the last published ILI week (publication lag) -> used by the nowcast
Xp = X.copy(); Xp.index = Xp.index.to_period("W-SAT"); Xp = Xp.groupby(level=0).mean()
ahead = Xp.index[Xp.index > ya.index.max()]
say(f"GT periods beyond last ILI week: {list(map(str, ahead))}")
if len(ahead) == 0:
    say("NOTE: no GT week beyond ILI -> h=0 forward forecast will fall back to h=1")
# X for forecasting: aligned weeks + the unpublished week(s)
Xf = Xp.loc[Xp.index >= ya.index.min()]

# ------------------------------------------------------------------ backtests
def gt_bt(h, model, window, p=P, name=None):
    return ev.rolling_backtest(ya, Xa, h=h, model=model, use_gt=True, p=p, xlags=XLAGS, period=PERIOD,
                               fourier_k=FK, y_known_at_origin=False, min_train=MIN_TRAIN, window=window,
                               preprocess=pp.raw_pipeline, name=name)

results, tables, verdicts, selected, forecasts, benches = {}, {}, {}, {}, {}, {}
timing = {}
for h in HORIZONS:
    t_h = time.time()
    res = {}
    # Information set at origin t: y known up to t-1 (FluView publication lag), GT up to t.
    # naive = last KNOWN value -> y_{t-1}; for h>=1 that is ev.naive at horizon h+1 (target t+h from y_{t-1}).
    res["naive"] = ev.naive(ya, h if h == 0 else h + 1, MIN_TRAIN); res["naive"].name = "naive"
    res["seasonal_naive"] = ev.seasonal_naive(ya, h, PERIOD, MIN_TRAIN)
    res["ar"] = ev.rolling_backtest(ya, None, h=h, model="ar", use_gt=False, p=P, period=PERIOD, fourier_k=FK,
                                    y_known_at_origin=False, min_train=MIN_TRAIN, name="ar")
    for model in ("lasso", "enet"):
        res[f"{model}_exp"] = gt_bt(h, model, None, name=f"{model}_exp")
        res[f"{model}_w104"] = gt_bt(h, model, WIN, name=f"{model}_w104")
    # ARGO's p=52 AR lags: only ~260 aligned rows; try expanding and sliding
    for window, tag in ((None, "exp"), (WIN, "w104")):
        try:
            r = gt_bt(h, "lasso", window, p=52, name=f"lasso_p52_{tag}")
            if len(r.preds) >= 30:
                res[f"lasso_p52_{tag}"] = r
            else:
                say(f"h={h}: lasso_p52_{tag} produced only {len(r.preds)} predictions -> skipped")
        except Exception as e:  # noqa
            say(f"h={h}: lasso_p52_{tag} failed: {e}")
    # benchmark = best target-only model by RMSE on the common full-sample dates
    common = None
    for r in res.values():
        common = r.preds.index if common is None else common.intersection(r.preds.index)
    bench_rmse = {m: ev.rmse((res[m].preds.loc[common, "y_true"] - res[m].preds.loc[common, "y_pred"]).values)
                  for m in ("naive", "seasonal_naive", "ar")}
    bench = min(bench_rmse, key=bench_rmse.get)
    benches[h] = {"bench": bench, "rmse": bench_rmse}
    gtm = [m for m in res if m not in ("naive", "seasonal_naive", "ar")]
    tables[h], verdicts[h] = {}, {}
    say(f"h={h} evaluation targets: {common.min()} -> {common.max()} ({len(common)} weeks)")
    for key, excl in (("full", None), ("excl_2020_03_2021_06", EXCLUDE), ("excl_2024_25_peak", EXCLUDE2)):
        tab = ev.compare(res, bench=bench, y_hist=ya, period=PERIOD, exclude=excl)
        tables[h][key] = tab
        verdicts[h][key] = ev.verdict(tab, bench, gtm)
        say(f"\n=== h={h} view={key} bench={bench} n={int(tab['n'].iloc[0])} ===")
        say(tab.round(3).to_string())
        say("verdict:", verdicts[h][key])
    sel = res["lasso_exp"].selected
    selected[h] = [{"origin": s["origin"], "features": s["features"]} for s in sel[-4:]]
    say(f"h={h} lasso_exp selected at recent origins:")
    for s in selected[h]:
        say(f"  {s['origin']}: {s['features']}")
    # frequency of selection over the last 26 origins
    freq = pd.Series([f for s in sel[-26:] for f in s["features"]]).value_counts()
    selected[h + 100] = freq.head(15).to_dict()
    say(f"h={h} most-selected features over last 26 origins: {freq.head(10).to_dict()}")
    results[h] = res
    timing[h] = time.time() - t_h
    say(f"h={h} done in {timing[h]:.0f}s")

# ------------------------------------------------------------------ forward forecasts
for h in HORIZONS:
    res = results[h]; tab = tables[h]["full"]; bench = benches[h]["bench"]
    gtm = [m for m in res if m not in ("naive", "seasonal_naive", "ar")]
    best_gt = max(gtm, key=lambda m: tab.loc[m, "OOS_R2_vs_bench"])
    model = best_gt.split("_")[0]
    window = WIN if best_gt.endswith("w104") else None
    p = 52 if "p52" in best_gt else P
    fc = ev.forecast_next(ya, Xf, h=h, model=model, use_gt=True, p=p, xlags=XLAGS, period=PERIOD, fourier_k=FK,
                          y_known_at_origin=False, window=window, preprocess=pp.raw_pipeline)
    q = ev.conformal_interval(res[best_gt].errors.values, alpha=0.1, recent=52)
    # benchmark forward value for reference
    fb = ev.forecast_next(ya, Xf, h=h, model="ar", use_gt=False, p=P, period=PERIOD, fourier_k=FK,
                          y_known_at_origin=False) if bench == "ar" else None
    forecasts[h] = {"config": best_gt, "origin": fc["origin"], "target": fc["target_date"], "point": round(fc["point"], 3),
                    "lo90": round(max(0.0, fc["point"] - q), 3), "hi90": round(fc["point"] + q, 3), "q90_halfwidth": round(q, 3),
                    "n_train": fc["n_train"], "features": fc["features"], "note": fc.get("note"),
                    "bench": bench, "bench_point": (round(fb["point"], 3) if fb else None),
                    "last_known_ili": {"date": str(ya.index[-1]), "value": round(float(ya.iloc[-1]), 3)},
                    "oos_r2_full": round(float(tab.loc[best_gt, "OOS_R2_vs_bench"]), 3)}
    say(f"\nforecast h={h}: {forecasts[h]}")

# ------------------------------------------------------------------ save
def recs(df):
    d = df.reset_index().astype(object)
    return d.where(d.notna(), None).to_dict("records")

secs = time.time() - T0
# Budget accounting. The first execution of this cell (2026-09-17 16:46-16:48Z, killed and rerun after a
# library fix) pulled 31 of the 38 terms live; the other 7 (flu, flu symptoms, influenza, fever, flu shot,
# stomach flu, flu test) were already cached by cell 02-Q2-A. Later runs pull 0 live. Verified in
# data/trends/catalog.jsonl (retrieved_utc column). Total for this cell = 31 pulls + 4 related_queries.
PULLS_FIRST_RUN = 31
n_req_total = max(n_pulls, PULLS_FIRST_RUN) + n_related_files
out = {"id": ID, "question": "How high will US ILI % be in 2 and 4 weeks? (plus h=0 nowcast)",
       "truth": {"source": "Delphi Epidata fluview wili nat", "first": str(y.index.min().date()), "last": str(y.index.max().date()),
                 "n": int(len(y)), "aligned_first": str(ya.index.min()), "aligned_last": str(ya.index.max()), "aligned_n": int(len(ya))},
       "queries": terms, "queries_dropped": DROP, "queries_expanded_total": len(terms_all),
       "queries_zero_filtered": info["dropped"], "queries_failed": info["failed"],
       "design": {"p": P, "p_alt": 52, "xlags": XLAGS, "period": PERIOD, "fourier_k": FK, "min_train": MIN_TRAIN, "window": WIN,
                  "preprocess": "raw_pipeline (ZeroRepair->log1p)", "transform": None, "y_known_at_origin": False},
       "benches": benches,
       "tables": {f"h{h}_{k}": recs(t) for h, d in tables.items() for k, t in d.items()},
       "verdicts": {f"h{h}_{k}": v for h, d in verdicts.items() for k, v in d.items()},
       "lasso_selected_recent": {f"h{h}": selected[h] for h in HORIZONS},
       "lasso_selection_freq_last26": {f"h{h}": selected[h + 100] for h in HORIZONS},
       "forecast": {f"h{h}": forecasts[h] for h in HORIZONS},
       "requests": n_req_total, "requests_detail": {"interest_pulls_live_this_run": n_pulls, "interest_pulls_first_run": PULLS_FIRST_RUN,
                                                     "terms_from_shared_cache": 7, "terms_pulled": len(terms),
                                                     "related_queries_calls": n_related_files,
                                                     "upper_bound_if_shared_terms_counted": len(terms) + n_related_files},
       "seconds": round(secs, 1), "timing_per_h": {str(k): round(v, 1) for k, v in timing.items()}}
with open(f"research/experiments/{ID}.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
# persist per-model predictions so further robustness views do not need a 25-minute rerun
import pickle
os.makedirs("outputs", exist_ok=True)
with open(f"outputs/{ID}_preds.pkl", "wb") as fh:
    pickle.dump({h: {m: r.preds for m, r in res.items()} for h, res in results.items()}, fh)
with open(f"research/experiments/scripts/{ID}.log", "w", encoding="utf-8") as fh:
    fh.write("\n".join(log))
say(f"\nrequests (this cell): {n_req_total}  elapsed {secs:.0f}s")
