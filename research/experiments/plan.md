# Experiment plan — comparing approaches on representative questions

Date: 2026-09-17. Library: `src/trends_predict` (gt, truth, preprocess, models, evaluate).
Each cell (question × approach) is run by a **context-isolated sub-agent** that receives only:
the question, the approach description, the library API, and the output contract. Results
land in `research/experiments/<id>.md` (+ `<id>.json` machine-readable) and are compared in
`research/conclusions.md`.

## Representative questions

| ID | Question | Answer type | Ground truth (keyless) | Freq | Language |
|---|---|---|---|---|---|
| Q1 | What will the US unemployment rate be next month? | quantity | BLS `LNS14000000` | M | EN |
| Q2 | How high will US flu activity (ILI %) be in 2 and 4 weeks? | quantity | Delphi FluView `wili` nat | W | EN |
| Q3 | Will the S&P 500 be higher one month from now? | direction | Yahoo `^GSPC` monthly close | M | EN |
| Q4 | Who will win the next US presidential election? | winner | past results 2004–2024 + national polls (Wikipedia) | W | EN |
| Q5 | מה יהיה שיעור האבטלה בישראל בחודש הבא? (Israel unemployment next month) | quantity | CBS via data.gov.il or CBS CSV; fallback: Wikipedia he pageviews proxy | M | HE |

Q1/Q2 are the literature's home turf (Choi–Varian, D'Amuri; ARGO, Djorno). Q3 is where the
literature says GT is weak except for direction (Bulut). Q4 is outside the literature and
tests the honesty path. Q5 tests Hebrew queries + `geo=IL` + a non-US truth source.

## Approaches under test

| Code | Name | Query generation | Preprocessing | Model(s) | Window |
|---|---|---|---|---|---|
| A | Choi–Varian minimal | 3–5 hand-picked keywords | ZeroRepair + log | OLS ARX vs AR | expanding |
| B | ARGO-style | seeds → `related_queries` expansion → 20–40 terms, individual pulls | ZeroRepair + log | LASSO / elastic-net ARX vs AR | sliding (104 w / 36 m) and expanding |
| C | Djorno-first | 10–20 terms | ZeroRepair + log + EWMA smooth + rolling detrend + clustering | ridge ARX and SARIMAX-exog vs AR; ablation raw vs preprocessed | expanding |
| F | Share proxy (winner) | topic IDs per contender, anchored joint pull | smoothing | share → margin regression validated on past contests | — |

Approach D (multi-vintage DLM) is **deferred**: vintage dispersion needs downloads on
different days; the cache is already recording vintages so it can be measured later.

## Matrix

| | A | B | C | F |
|---|---|---|---|---|
| Q1 unemployment | 01-Q1-A | 01-Q1-B | 01-Q1-C | |
| Q2 flu | 02-Q2-A | 02-Q2-B | 02-Q2-C | |
| Q3 S&P direction | 03-Q3-A | 03-Q3-B | | |
| Q4 election | | | | 04-Q4-F |
| Q5 Israel (HE) | 05-Q5-A | 05-Q5-B | | |

## Output contract for every cell

`research/experiments/<id>.md` with sections: Setup (queries, truth, freq, horizon(s)),
Results table (`evaluate.compare` output for each horizon), Verdict (`evaluate.verdict`),
What went wrong / surprises, Time & pull budget (number of GT requests, wall time), and
**one-paragraph recommendation** for the skill. `<id>.json` holds the metrics tables.

## Success criteria for the skill (decided before seeing results)

1. On Q1/Q2 at least one GT approach has OOS R² > 0 vs AR with Clark–West p < 0.10 for h ≤ 1.
2. On Q3 the skill reports honestly (expected: little or no gain; directional accuracy near 0.5).
3. On Q4 the skill produces a *validated* proxy statement with an explicit hit-rate on past
   elections and refuses to over-claim.
4. On Q5 Hebrew queries return usable series (non-zero share ≥ 50 %) and the pipeline runs.
5. Total pull budget per question ≤ 60 requests; runtime ≤ 10 minutes.
