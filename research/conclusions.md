# Conclusions — how the trends-forecast skill should work

Status: consolidated 2026-09-17 after 11 experiment cells (`research/experiments/`), the library
smoke tests and the pipeline runs in `specs/` → `outputs/`. Evidence pointers `[01-Q1-A]` etc.
refer to files in `research/experiments/`. Cells still pending when this was written: 02-Q2-B
(flu, ARGO expansion), 03-Q3-B (S&P weekly expansion), 05-Q5-B (Hebrew expansion) — their results
are appended in §9 when they arrive.

## 1. Data access (settled)

- Free access works reliably via `trendspy`: 30 rapid calls, 0 throttling `[00]`; ~200 requests
  across 11 concurrent cells produced a single 429 that the backoff absorbed. `pytrends` is dead.
- **Pull each keyword individually** for modelling; joint (≤5) pulls only for shares/composites.
- Frequency follows the timeframe length (≤9 m daily, ≤5 y weekly, longer monthly). Weekly history
  beyond 5 years needs stitched windows rescaled on their overlap (`gt.fetch_stitched`; overlap
  correlation ≥ 0.99 in `[02-Q2-A]`).
- Every pull is a dated vintage; same-day re-pulls are byte-identical, so Rivera-style retrieval
  noise can only be measured across days (deferred; the cache is collecting vintages).
- Hebrew works: 8/9 Hebrew terms ≥ 90 % non-zero with `geo=IL`; Google strips gershayim, so
  `חל"ת` must be sent as `חלת` (now automatic) `[05-Q5-A]`.
- Keyless truth: BLS, Delphi/CDC FluView, Yahoo, Wikipedia pageviews, **Israel CBS series API**
  (`apis.cbs.gov.il/series/data/list?id=…`, spliced old/new LFS definitions) `[05-Q5-A]`.

## 2. Evaluation discipline (settled — each rule reversed at least one conclusion)

1. **Benchmark = best target-only model among naive, drift, seasonal-naive, AR (by MAE)** and it
   must use the same information set as the GT models (under publication lag the last known
   value is y_{t−1}). Against AR-in-levels every GT model "won" by 50–65 %; against the random
   walk they lost `[00b][01-Q1-A]`. Against a zero-return naive, GT "beat" the S&P by having an
   intercept; against random-walk-with-drift the gain vanished `[03-Q3-A]`.
2. **Publication lag must be modelled exactly.** The origin row's own target was leaking into
   h=0 training; after the fix `[02-Q2-C]` the unemployment nowcast gain disappeared and the
   "GT caught April 2020" story died (nowcast 4.7 % vs actual 14.8 %) `[01-Q1-B]`.
3. **Two views: full sample and shock-excluded**, and say when the exclusion window is outside
   the sample instead of showing an identical table `[02-Q2-A]`.
4. **Clark–West alone is never evidence** (p<0.01 for models with OOS R² −4). Verdict rule:
   OOS R² > 0.02 and (CW p<0.10 or DM p<0.10); *strong* = OOS R² > 0.15 and both tests.
   A "≥80 % hits and r>0.5" rule would have validated the midterm share on n=5 `[04-Q4-F]` —
   pre-register n ≥ 8 for discrete outcomes.
5. **MAE next to RMSE**; RMSE is dominated by single shock months.
6. **Intervals: rolling conformal** on recent out-of-sample residuals (coverage 91–96 %).
7. **Regime split**: report error in high vs low target regimes; flu gains exist only in-season
   and reverse off-season (target-only ~2× better off-season) `[02-Q2-A][02-Q2-C]`.
8. Floor non-negative targets (`clip_min`); levels-OLS produced negative ILI at h=4 `[02-Q2-A]`.

## 3. Where Google Trends helps — the evidence table

| Question | Freq | Best approach | Nowcast h=0 | h=1–2 | h=4+ | Notes |
|---|---|---|---|---|---|---|
| US unemployment rate | M | none beats naive | −0.10…−0.25 vs naive (all cells); calm-view +13…21 % with ridge+detrended GT, DM n.s. | −0.02…−0.16 | — | Expansion to 27–59 features is 2× worse than naive `[01-Q1-B]`; DirAcc 0.58–0.67 is the only GT signal |
| US flu ILI % | W | **ridge/OLS + raw log GT** | **+0.54…0.59 vs AR**, DM 0.01–0.03, CW <0.001 | **+0.36…0.39** at h=2, DM 0.01–0.02 | −0.13…+0.23 vs seasonal-naive, DM n.s. | Djorno steps all hurt (clustering collapsed 15/17 queries); gains in-season only |
| S&P 500 direction | M / W | none | — | OOS R² −0.01 vs drift; hit-rate 64.5 % vs always-up 66.4 % | — | Answer = drift + base rate |
| US election winner | — | none | — | — | — | Share hit-rate 21/36 pooled; presidential 2/5; modifiers negatively correlated with margin; 2026 reading flips on one spike week |
| Israel unemployment (HE) | M | none | −0.01 vs naive | −0.01 | — | Hebrew pipeline works end-to-end; naive 3.1 % ± 0.4 |

**Pattern (matches the literature).** GT is a **nowcasting** instrument for quantities with (a) a
publication lag, (b) a direct behavioural link between the phenomenon and a search act
(symptoms → "flu test", not "unemployment rate"), and (c) high-frequency variation. It does not
beat persistence for slow monthly aggregates, prices, or discrete outcomes. Djorno-style
preprocessing helps when the signal is a slow level shift (unemployment: detrending) and hurts
when the signal is the spike itself (flu). Breadth (ARGO expansion) helped nowhere in our cells
with T ≈ 120–260 observations; hand-picked 8–15 behavioural terms did as well or better.

## 4. Query generation (decided)

- 8–15 **behavioural** terms per question, mixing information / action / consequence intents;
  individual pulls; keep everything with non-zero share ≥ 0.5; let the model select.
- Screen for target-matching seasonality: `fever` is a US *summer* term `[02-Q2-A]`; `flu shot`
  enters with a negative sign `[02-Q2-C]`.
- Related-query expansion only as a source of *candidates*, capped at ~15 kept terms; sliding
  windows with more columns than rows collapse to the random walk `[01-Q1-B]`.
- Topic ids (`/m/…`) for entities and for Hebrew/English parity; Hebrew keywords plus `geo=IL`.

## 5. Discrete outcomes (decided) — see `[04-Q4-F]`
Refuse to forecast winners. Provide the **salience** report (share, trend, spike sensitivity,
validated hit-rates with n) and redirect to a continuous target (poll margin) if the user wants
a real forecast.

## 6. Hebrew (decided) — see `[05-Q5-A]`
Fully supported: Hebrew queries, `geo=IL`, CBS truth, English answer. Bonus requirement met.

## 7. Skill architecture (implemented)

- **LLM reasons, pipeline computes.** The skill writes `specs/<id>.json` and runs
  `.claude/skills/trends-forecast/scripts/run_spec.py` (modes: `--truth`, `--probe`, full,
  `--salience`, `--suggest`, `--related`). Outputs: `outputs/<id>.json` + self-contained
  `outputs/<id>.html` (history + forecast + interval, backtest vs benchmark, metrics, query
  small-multiples, robustness table, regime split, provenance).
- Defaults by frequency: M → differences, naive/drift benchmark, models {ridge+djorno,
  ridge+djorno_nocluster, lasso+djorno, lasso+raw, ridge+raw}; W → levels, Fourier(2),
  min_train 104, `clip_min 0` for rates. The backtest picks; the forecast uses the GT model only
  if the verdict says it helps, otherwise the benchmark — and the answer says which.
- Honesty rules are in `.claude/skills/trends-forecast/references/answer-rules.md`; learned
  query sets accumulate in `references/playbook.md`.

## 8. What we did not do (and why)
- Multi-vintage DLM (Rivera): needs downloads on different days; vintages are being recorded.
- MIDAS: no cell had a mixed-frequency win worth the extra machinery; helper functions exist.
- Paid APIs: unnecessary — free access held at ~200 requests/hour across parallel agents.

## 10. Skill validation — 16 isolated runs, Fable 5.1 vs Opus 5 (2026-09-17)

Protocol: `research/validation/BRIEF.md`; scores: `research/validation/results.json`.
Eight questions (five from the experiments, three unseen: inflation, New York flu, private
website traffic; two in Hebrew) × two models, each a fresh agent with only the repo and the skill.

- **All 16 answers were correct and honest.** Every forecast came from the pipeline (spec +
  outputs exist; numbers match `outputs/<id>.json`); no probability was ever derived from a
  search share; both Hebrew questions were answered in English; the two refusals (website
  traffic) and the two election answers (salience only, hit-rates with n) followed the rules.
- **Both models converge on the same numbers** because the pipeline decides: unemployment
  4.1 % ± 0.2 (naive), flu nowcast 1.69 % [1.36, 2.03] (ols+raw, strong), S&P ≈ 7,741 from
  drift with P(up) ≈ base rate, Israel 3.1 % ± 0.4 (naive), CPI +0.27…0.31 % m/m vs +0.40 %
  (AR; P(accelerate) 31–37 % vs 51 % base rate), New York flu 2.3 % for w/e 3 Oct.
- **Model differences are in depth, not correctness.** Opus wrote longer, sharper case-specific
  caveats (NY↔Region-2 correlation, both inflation definitions, CBS definition splice) and
  found more defects; Fable was ~2× faster with the same discipline (mean 10/10 vs 9.75/10,
  the gap being two runs over the 10-minute budget).
- **Unseen questions were handled**: V6 required deriving an inflation rate from the CPI index
  (both models did; the `derive` option now exists), V7 required noticing that the state-level
  CDC series died in 2025 and substituting the HHS region (both did, and said so).
- **17 defects surfaced by the validators, all fixed** before the final skill version: leaked
  origin row in nowcast training; nowcast note missing on the benchmark branch; confidence
  precedence between views; regime rule for prices and for benchmark answers; DM sign in the
  verdict text; salience spike-week diagnostic (global vs last-4 max); topic-id labels;
  duplicate `class` on the confidence tile; "GT gain" tile shown when the benchmark answers;
  `top_features` key; stale `__pycache__`; stale Yahoo cache with a partial month; `--probe`
  ignoring `history_years` and differencing rate targets; `clip_min` guidance for inflation;
  horizon semantics under publication lag; FluView state series staleness; `cbs` truth kind,
  `--truth-check`, `--rerender --answer-file/--caveats-file`, `lo_unclipped`.

## 9. Late cells (appended when they finish)

- **05-Q5-B (Hebrew, ARGO expansion):** `gt.expand_queries` in `geo=IL` returned 56 Hebrew terms
  from 5 seeds covering the whole benefits-claim funnel; 29/31 curated terms passed the zero
  filter (≥95 % non-zero since 2012). Best model ridge+Djorno OOS R² +0.03 vs naive, CW p≈0.01 but
  DM p 0.6–0.7, gain vanishes ex-COVID; LASSO selected nothing at the last origin. Forecast = naive
  3.1 % [2.7, 3.5]. Confirms: Hebrew expansion is technically fine, monthly macro persistence wins.
- **03-Q3-B (S&P, weekly, ARGO expansion):** 52 candidates → 29 weekly terms, 153 origins, bench
  drift. Best GT model OOS R² −0.001 (CW 0.57, DM 0.84) and it selected zero search terms at every
  origin; always-up 71 % beats every model. Post-hoc exclusion of the April-2025 tariff crash makes
  sliding-window LASSO "moderate" only because it lost during the crash — not an edge. **Library
  bug found and fixed:** weekly Yahoo/Wikipedia series were labelled Sunday-ending, so `align("W")`
  put a Friday close in the following Sun–Sat bucket (one-week GT look-ahead); now `W-SAT`.
- **02-Q2-B (flu, ARGO expansion, 38 terms; partial — final write-up pending):** h=0 strong as with
  the hand-picked set, but the h=2 "moderate" verdict came entirely from the 13 peak weeks of the
  2024-25 season; excluding that season every expanded-set model is negative at h=2. Breadth adds
  noise; gains concentrate in peaks.
