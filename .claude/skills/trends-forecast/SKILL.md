---
name: trends-forecast
description: Answer forecasting / nowcasting / prediction questions ("what will X be next month", "how bad will the flu season get", "will the S&P be higher", "who will win", in English or Hebrew) with Google Trends search data validated against real historical outcomes, using the methods of the Google-Trends forecasting literature (Choi–Varian ARX, ARGO LASSO, Djorno preprocessing, rolling-origin backtests, Clark–West / Diebold–Mariano tests). Produces a written answer with honest uncertainty and a self-contained HTML evidence artifact; refuses with reasons when the question cannot be validated (private metrics such as "my sales / my traffic" get a refusal with a CSV offer; "who will win" gets a descriptive attention report, never a probability).
---

# trends-forecast

You turn a natural-language prediction question into a **validated** Google-Trends forecast — or
into a clear, reasoned refusal. The LLM (you) does the reasoning: what the target is, where its
ground truth lives, which searches plausibly lead it, what horizon. A deterministic pipeline does
the numbers identically every time. Never estimate a forecast in your head; never skip the backtest.

Repository root: the directory containing `src/trends_predict/` (this skill lives in
`.claude/skills/trends-forecast/`). Run everything from the root with
`PYTHONIOENCODING=utf-8 python .claude/skills/trends-forecast/scripts/run_spec.py …`.

## Procedure

### 1. Parse the question
Write down: **target quantity** (what number answers the question), **geography** (`geo`: `US`,
`IL`, `GB`, `US-CA`, `` = worldwide), **frequency** of the truth series (M/W/D), **horizon**
(nowcast = the latest period not yet published; or h periods ahead), **answer type**
(`quantity`, `direction`, `winner`, `yes/no`), **language** of the question (answer in English;
Hebrew/other terms go inside query strings). If the question is vague ("will the economy do
well?") pick the closest measurable target and say so in the answer.

### 2. Feasibility gate — is there ground truth?
Look up `references/truth-sources.md`. A forecast is only possible if a historical series of the
actual outcome exists (keyless public source, or a CSV the user provides). "Checked" means: the
registry in `truth-sources.md`, the files already in `data/truth/`, and — if you have web access —
one search for an official public series.
- Series exists → continue with step 3.
- Discrete outcome (election, award, match) → skip to `references/answer-rules.md` § Discrete
  outcomes: run the salience mode, never a win probability.
- Nothing → **refuse** with the template in `answer-rules.md`. A refusal is a complete answer: no
  spec, no pipeline run, no artifact; steps 3–6 are skipped; the options (CSV / description /
  proxy) are stated as standing offers, not as questions that block. Still do step 7 (log the
  case in one line under "Refused" in the playbook).

### 3. Design the query set (8–15 terms)
Read `references/playbook.md` first (proven sets and failures). Then:
- Prefer **behavioural** searches tied to the mechanism (symptoms, "file for unemployment",
  "flights to X"), not the name of the statistic ("unemployment rate").
- Mix intent levels: information ("flu symptoms"), action ("flu test near me"), consequence
  ("tamiflu"). Include 1–2 broad anchors so at least some series clear the privacy threshold.
- Entities and non-English targets: use **topic ids** (`run_spec.py --suggest "<name>"` →
  `/m/…`), which are language-neutral, alongside local-language keywords.
- Optional expansion: `run_spec.py --related "<seed>" --geo <geo>` as a *source of candidates*
  only — keep the total at 8–15. In every experiment, 25–60 expanded terms did worse than a
  hand-picked dozen (see playbook).
- Screen out terms whose seasonality does not match the target (e.g. `fever` peaks in the US
  summer, not the flu season) — `--probe` shows the correlations with target changes.
- Then probe before the full run: `run_spec.py --probe specs/<id>.json` → drop terms with
  `nonzero_share < 0.5`, keep the rest even if correlations look weak (the model selects).

### 4. Write the spec and run
Create `specs/<id>.json` (schema and defaults are documented at the top of
`src/trends_predict/pipeline.py`; examples in `specs/`; the much smaller *salience* spec for
discrete contests is described in `answer-rules.md` § Discrete outcomes). Key choices:
- `freq` M → defaults to `transform: "diff"` (persistent series, random-walk benchmark);
  W → levels with `fourier_k 2`, `min_train 104`. Prices: `truth.log: true`, `answer_type:
  "direction"`, `transform: "diff"`, `fourier_k 0`.
- `horizons` — read this carefully, it is where "next month" goes wrong:
  * `y_known: false` = publication lag: at the origin the latest *published* value is one
    period old. `{"h":0,"y_known":false}` nowcasts the unpublished period (needs GT to extend
    past the last print, e.g. flu); `{"h":k,"y_known":false}` targets k periods after that.
  * `y_known: true` = the last value is known; `{"h":1,"y_known":true}` targets the period after
    the last print — the **same calendar period** as the h=0 nowcast when there is a lag.
  * With a lag, "next month" means the *next print* (h=0 nowcast) — say so — and add the calendar
    next month (h=1 with `y_known:false`). Cover every period the user asked about; use one
    `y_known` convention per spec so horizons line up on the same origin.
- `clip_min: 0` only for quantities that cannot be negative (ILI %, counts, unemployment rate);
  **not** for inflation or returns (June 2026 CPI was −0.42 % m/m).
- Rate/flow targets (inflation, growth, returns): `transform: null` plus the AR benchmark — a
  random walk is the wrong bar for something that is already a difference.
- `models`: keep both preprocessing families — `ridge+raw`, `lasso+raw`, `ridge+djorno`,
  `ridge+djorno_nocluster`, `lasso+djorno` — the backtest decides (detrending helped
  unemployment, every Djorno step hurt flu). Omit `models` to get the defaults.
- `windows`: `[null]`. Sliding windows (`36` monthly / `104` weekly) lost to the expanding
  window in every validated run so far; add one only as a deliberate ARGO-style experiment.
- `clip_min: 0` for rates/counts; `history_years: 9` for weekly targets that need more than
  Google's 5-year weekly cap (stitched windows).
- `exclude`: the shock window relevant to the series (COVID: `2020-03-01..2020-12-31` macro,
  `..2021-06-30` health/Israel; `2020-02-01..2020-06-30` markets).
Check the truth source first (`run_spec.py --truth specs/<id>.json`), then probe the queries
(`--probe`), then run: `run_spec.py specs/<id>.json`. It prints a JSON summary and writes `outputs/<id>.json`
and `outputs/<id>.html`. Budget: ≤ 60 live Trends requests, ≤ 10 minutes. If it raises
`TrendsUnavailable`, wait 60 s and rerun (the cache resumes); if it persists, tell the user and
offer the manual-CSV import (`gt.import_manual_csv`) — never fabricate.

### 5. Read the verdicts and decide
For each horizon the summary gives `bench`, `verdict` (`strength` strong / moderate / none,
`oos_r2`, `cw_p`, `dm_p`), `verdict_excluded` (or `exclusion_note`), `chosen_model`,
`regime_split` (benchmark vs GT error when the target is high vs low — e.g. flu gains exist only
in-season) and `forecast` (point, `lo`, `hi`, `top_features`, and `note` if a nowcast fell back to
h=1 because Google Trends does not yet extend past the last published value). Rules (from
`answer-rules.md`):
- **strong** in both views → confidence `high`, forecast from the GT model. If only one view
  exists (`exclusion_note`: the shock window is outside the evaluated sample) treat it as both.
- moderate, or strong in one view → `medium`; explain which view.
- none → the forecast is the benchmark's; confidence `low`; say GT added nothing measurable.
- **Precedence:** the full-sample verdict and `chosen_model` govern. The shock-excluded view can
  only *lower* confidence (strong in full, none excluded → `medium`), never raise it: if the
  benchmark was chosen, confidence is `low` even when the excluded view is moderate — mention the
  excluded-view gain as "suggestive, not validated".
- **Requests budget** counts live attempts (retries included, `gt.REQUEST_COUNT`); if `--truth`
  shows a partial current period, delete that file in `data/truth/` and rerun (`max_age_days=1`
  cache may predate the partial-period fix).
- **Regime downgrade:** look at `regime_split` (`split_on`: level for rates/counts, absolute change for prices).
  It applies only when a GT model was chosen; a benchmark answer is already `low` and never
  becomes `none` (which means "no validation was possible").
- `gt_reference_forecast` gives the best GT model's own point even when the benchmark answers —
  quote it as "the search-augmented model would say X" so the reader sees the (non-)difference. If the *current* level of the target is in the
  regime (high/low vs `threshold`) where the benchmark's MAE is lower than the GT model's, drop
  one confidence class and say why (e.g. flu in September: the search signal is strong in-season,
  weak off-season). The pipeline does not do this for you.
- Quote the benchmark's own point (`benchmark_forecast`) next to the GT point so the reader sees
  how much the search data moved the answer.
- If **all** queries were dropped (privacy zeros) or the truth series is too short, revise the
  spec **once** (broader terms / topic ids / longer history) and rerun; then stop iterating.

### 6. Answer and deliver the artifact
Compose the answer with **every** mandatory element in `references/answer-rules.md` (answer,
source, evidence line with numbers, confidence, drivers, case-specific caveats, artifact path).
Open or point to `outputs/<id>.html` (self-contained: history + forecast + interval, backtest vs
benchmark, metrics tables, query small-multiples, robustness table, provenance).
Then make the artifact say what you say: `run_spec.py --rerender <id> --headline "…" --answer "…"
--confidence <class>` rewrites `outputs/<id>.html` from the pickled results in seconds — the
artifact must not contradict the written answer. Pass plain text (it is HTML-escaped for you).
If a Python traceback's line numbers do not match the source, a stale `__pycache__` is being
executed: delete `src/trends_predict/__pycache__` and rerun (the runner now does this itself).

### 7. Learn
Append a dated entry to `references/playbook.md` for **every** run: for a completed run — domain,
query set, what the backtest chose, verdict numbers, one lesson; for a refusal — one line under
"Refused" (question class, why, the reframe offered). This is how query design improves.

## Hard rules
- Every Google Trends pull goes through `trends_predict.gt` (cached, vintage-stamped in
  `data/trends/`); never call the API elsewhere.
- Validation is always out-of-sample rolling-origin against the best target-only benchmark
  (naive / drift / seasonal-naive / AR). No benchmark, no claim.
- Clark–West alone is never evidence; require OOS R² > 0 and DM/CW support (the pipeline's
  verdict already encodes this — do not overrule it upward).
- Report both the full-sample and shock-excluded views when they exist.
- Google Trends values are relative attention, not counts; say so once per answer.
- English output; questions may be in any language.
