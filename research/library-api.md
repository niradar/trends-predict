# `trends_predict` library — API cheat-sheet

Location: `src/trends_predict/`. Use with `import sys; sys.path.insert(0, "src")`.
Run scripts with `PYTHONIOENCODING=utf-8 python ...` (Hebrew output on Windows).

## gt — Google Trends (cached, vintage-stamped)

```python
from trends_predict import gt
gt.timeframe_for("M")            # '2004-01-01 <today>'  -> monthly series
gt.timeframe_for("W")            # 'today 5-y'            -> weekly
gt.timeframe_for("D")            # 'today 3-m'            -> daily
gt.timeframe_for("W", "2016-01-01", "2020-12-31")   # explicit window (≤5y -> weekly)

df = gt.fetch_interest(["flu symptoms","cough"], geo="US", timeframe="today 5-y")
#   ≤5 keywords, jointly normalised (0–100 over the whole request). isPartial row dropped.
#   Cached under data/trends/<slug>/v_<utc>.csv; force=True adds a new vintage.

X, info = gt.fetch_many(keywords, geo="US", timeframe=gt.timeframe_for("M"))
#   mode='individual' (default): one request per keyword, full resolution each -> use for models.
#   mode='anchored', anchor='...': batches of 4 + anchor, rescaled to one common scale
#   -> use for shares/composites (e.g. candidate A vs B). info['dropped'] = privacy-zero terms.

gt.related_queries("unemployment", geo="US")   # {'top':[{'query','value'}], 'rising':[...]}
gt.related_topics("unemployment", geo="US")    # {'top':[{'mid','title','type','value'}], ...}
gt.suggestions("influenza")                    # [{'mid':'/m/0cycc','title':'Influenza','type':'Disease'}]
gt.expand_queries(["unemployment","jobs"], geo="US", max_per_seed=15)  # seeds + top related
gt.fetch_stitched("flu symptoms", "2017-09-17", "2026-09-17", geo="US")   # >5y weekly: chained windows rescaled on overlap
gt.fetch_many_stitched(keywords, start, end, geo="US")                    # same for many keywords
gt.load_vintages(...) / gt.vintage_dispersion(list)   # retrieval-noise analysis (Rivera)
gt.REQUEST_COUNT                                # live calls made by this process (cache hits excluded)
gt.catalog()                                    # DataFrame of everything ever pulled
gt.import_manual_csv(path, keywords, geo)       # register a trends.google.com CSV export
```
Topic ids (`/m/...`, `/g/...`) can be used as keywords; they are language-neutral.
Errors: `gt.TrendsUnavailable` after 5 backoff attempts — report it, do not swallow it.

## truth — ground-truth series (cached under data/truth/)

```python
from trends_predict import truth
y = truth.bls_series("UNRATE")                 # monthly, index = month start. Keys: UNRATE, UNRATE_NSA,
                                               # PAYEMS, CPIAUCNS, CPIAUCSL, UNEMPLOYED, JOBLESS_27W, or raw BLS id
y = truth.fluview_ili("nat", 201040)           # weekly wILI %, index = Saturday week end
y = truth.fluview_clinical("nat")              # weekly % positive
y = truth.yahoo("^GSPC", freq="M")             # monthly last close (freq None/'D','W','M')
y = truth.wikipedia_pageviews("Influenza", freq="W")
y = truth.csv_series("path.csv", date_col=0, value_col=1, name="cbs_unemp")
ya, Xa = truth.align(y, X, "M")                # common PeriodIndex ('M', 'W' = W-SAT, 'D'); GT averaged per period
```

## preprocess — fit inside the training window (the backtest does this for you)

```python
from trends_predict import preprocess as pp
pp.raw_pipeline()                                   # ZeroRepair -> log(x+1)            ("raw GT")
pp.djorno_pipeline(period=52, halflife=2.0,         # ZeroRepair -> log -> EWMA smooth -> rolling detrend
                   detrend="rolling", cluster=True)  # -> hierarchical clustering of redundant queries
pp.Pipeline([pp.ZeroRepair(), pp.Log1p(), pp.Deseasonalize(12), pp.Standardize()])
```
Pass a **factory** (callable returning a fresh pipeline) to the backtest, e.g. `preprocess=pp.raw_pipeline`
or `preprocess=lambda: pp.djorno_pipeline(period=12)`.

## evaluate — rolling-origin backtests and comparison

```python
from trends_predict import evaluate as ev
r_ar   = ev.rolling_backtest(ya, None, h=1, model="ar",    use_gt=False, p=3, period=12, fourier_k=1, min_train=60)
r_gt   = ev.rolling_backtest(ya, Xa,   h=1, model="lasso", use_gt=True,  p=3, xlags=1, period=12, fourier_k=1,
                             min_train=60, preprocess=pp.raw_pipeline, window=None, transform="diff")
# model: 'ar' | 'ols' | 'ridge' | 'lasso' | 'enet' | 'rf' | 'arlr'
# h=0 & y_known_at_origin=False  -> nowcast (latest target unknown, current-period GT known)
# h>=1 & y_known_at_origin=True  -> forecast from a known latest value
# window=104 -> sliding window (ARGO); None -> expanding
# transform='diff' -> model changes from the last known level (use for persistent series)
r_sx   = ev.rolling_sarimax(ya, Xa, h=1, use_gt=True, order=(1,0,0), seasonal_order=(0,1,1,12), min_train=60,
                            preprocess=lambda: pp.djorno_pipeline(period=12))
r_nv   = ev.naive(ya, h=1, min_train=60, y_known_at_origin=True)   # last *known* value (y_{t-1} under publication lag)
r_dr   = ev.drift(ya, h=1, min_train=60)                            # random walk with drift — mandatory for prices
r_sn   = ev.seasonal_naive(ya, h=1, period=12, min_train=60)

tab = ev.compare({"naive": r_nv, "ar": r_ar, "lasso+GT": r_gt}, bench="naive", y_hist=ya, period=12)
tab2 = ev.compare(..., exclude=("2020-03-01", "2020-12-31"))      # robustness without the pandemic
ev.verdict(tab, bench="naive", gt_models=["lasso+GT"])             # {'gt_helps': bool, 'strength': ..., 'reason': ...}
q = ev.conformal_interval(r_gt.errors.values, alpha=0.1, recent=36) # half-width for a 90% interval
fc = ev.forecast_next(ya, Xa, h=1, model="ridge", use_gt=True, p=3, xlags=1, period=12, fourier_k=1,
                      preprocess=lambda: pp.djorno_pipeline(period=12), transform="diff")
# -> {'origin','target_date','point','model','n_train','features'}; use the SAME settings as the winning backtest.
# h=0 nowcast works only if X extends one period beyond y (publication lag); otherwise it returns the h=1 forecast.
r_gt.selected[-1]["features"]                                       # what LASSO/ARLR picked at the last origin
```
**Benchmark rule:** the benchmark is the *best* target-only model among naive, seasonal-naive and
AR (levels or differences). Claim "GT helps" only if OOS R² > 0 against that benchmark **and**
Clark–West or Diebold–Mariano p < 0.10.

## pipeline — the whole thing from a JSON spec (what the skill calls)

```python
from trends_predict.pipeline import run_spec_file, run, write_outputs
res = run_spec_file("specs/us-unemployment.json")   # prints a JSON summary; writes outputs/<id>.html + .json
# or programmatically: res = run(spec_dict); paths = write_outputs(res, headline=..., answer_text=..., confidence=...)
```
Spec keys are documented at the top of `src/trends_predict/pipeline.py` (truth kinds: bls, fluview,
fluview_clinical, yahoo, wikipedia, csv, cached; extras `clip_min`, `history_years`, `answer_type`).
Defaults per frequency: M → diff, p=3, period 12, min_train 60; W → levels, p=4, period 52,
min_train 104. Benchmark = best of naive / drift / seasonal-naive / AR by MAE (same information set as
the GT models); the forecast uses the best GT model only if the verdict says it helps, otherwise the
benchmark; `benchmark_forecast`, `regime_split`, `exclusion_note` and the nowcast `note` are in the output.

```python
from trends_predict.pipeline import salience, rerender
salience({"id": ..., "question": ..., "entities": [{"label": "Republican Party", "query": "/m/07wbk"}, ...], "geo": "US", "timeframe": "today 12-m"})
#   discrete contests: attention share (last 4 wks, previous 4, excl. loudest of last 4) -> HTML/JSON; NOT a forecast
rerender("us-flu-ili", headline="...", answer_text="...", confidence="medium")   # rewrite the artifact from the pickled run
```
CLI for all of this: `.claude/skills/trends-forecast/scripts/run_spec.py` (`--truth`, `--probe`, run,
`--salience`, `--rerender`, `--suggest`, `--related`, `--open`).

## report — HTML artifact

```python
from trends_predict.report import render_report
render_report("outputs/q1.html", title=..., question=..., answer_headline=..., answer_text=..., confidence="medium",
              forecast={"date":"2026-10-01","point":4.2,"lo":3.9,"hi":4.5,"alpha":0.1,"label":"Oct 2026, %"},
              history=ya.tail(120), history_label="Unemployment rate (%)",
              backtest={"naive": r_nv.preds["y_pred"], "LASSO + GT": r_gt.preds["y_pred"]}, backtest_truth=r_gt.preds["y_true"],
              metrics=tab, gt_series=Xa, queries_info=[{"query": k, "nonzero_share": ...}, ...],
              method_notes=[...], caveats=[...], provenance={"truth": "BLS LNS14000000", "trends_cache": "data/trends/..."})
```
