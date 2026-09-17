# Brief for experiment sub-agents

You are running **one cell** of the experiment matrix in `research/experiments/plan.md`: one
question × one approach. You are context-isolated on purpose — everything you need is in the
repository at `C:\projects\trends-predict`. Work in English. Do not ask the user questions.

## Read first
1. `research/library-api.md` — the `trends_predict` library you must use (do not re-implement
   fetching, caching or backtesting; extend only if something is missing, and say so).
2. `research/experiments/00b-smoke-unemployment.md` — what a finished cell looks like and the
   benchmark lesson (naive random walk must be in the benchmark set).
3. Your cell's parameters (in the prompt you received).

## Rules
- Python: `PYTHONIOENCODING=utf-8 python your_script.py` from the repo root; `sys.path.insert(0,"src")`.
- All Google Trends pulls go through `trends_predict.gt` (cached under `data/trends/`). Budget:
  ≤ 60 requests. Count them (`len(gt.catalog())` before/after).
- Ground truth via `trends_predict.truth` (cached under `data/truth/`). If your question needs a
  source the library lacks, fetch it keylessly (requests / pandas.read_html) and save it with
  `truth.csv_series(...)` or `truth._save(name, series, meta)`; document the URL.
- Evaluation protocol: rolling origin, `min_train` ≥ 60 monthly / 104 weekly, benchmark set
  {naive, seasonal_naive, ar (transform='diff' for persistent series, else levels)}, GT models per
  your approach, horizons per your cell. Use `ev.compare` with `bench=` the **best** target-only
  model, and report both the full sample and an exclusion window for shock periods
  (e.g. `("2020-03-01","2020-12-31")` for macro, none for flu unless justified).
- Then produce the actual forward forecast with `ev.forecast_next(...)` using the winning
  configuration and a 90 % conformal interval from the backtest residuals.
- Put your script in `research/experiments/scripts/<ID>.py` so the run is reproducible.
- Never fabricate data or results. If something fails, record the failure and what you tried.

## Deliverables (mandatory)
`research/experiments/<ID>.md` with sections:
1. **Setup** — question, truth series (source, range, freq), queries (how generated, how many,
   how many dropped), horizons, window type, preprocessing, models.
2. **Results** — `ev.compare` tables per horizon (full and excluded), pasted as markdown.
3. **Verdict** — `ev.verdict` output per horizon + your one-sentence honest reading.
4. **Forward forecast** — origin, target date, point, 90 % interval, which features drove it.
5. **Surprises / problems** — anything that broke, was slow, or looked suspicious (leakage!).
6. **Budget** — GT requests used, wall time.
7. **Recommendation for the skill** — one paragraph: what should the skill do for questions
   like this, and what should it refuse or caveat.

`research/experiments/<ID>.json` — `{"id":..., "tables": {horizon: compare-table-as-records},
"verdicts": {...}, "forecast": {...}, "queries": [...], "requests": n, "seconds": n}`.

Finish by returning a ≤ 15-line summary of the verdict, the forecast, and your recommendation.
