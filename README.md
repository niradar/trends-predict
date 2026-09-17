# trends-forecast

A Claude Code skill that turns a plain-language prediction question, in English or Hebrew, into a validated Google Trends forecast, or into a reasoned refusal. It applies the methods of the Google Trends forecasting literature (Choi & Varian, ARGO, MIDAS, Rivera, Borup & Montes Schütte, Djorno et al.) with rolling-origin backtests against the best target-only benchmark, and delivers a written answer plus a self-contained evidence artifact.

The question decides the target; the data decides the answer. If search data does not beat the benchmark out of sample, the skill says so and answers with the benchmark. If no historical outcome series exists to validate against, it refuses and explains what would make the question answerable.

Web summary with showcase examples: https://claude.ai/artifact/FmFMJyL5jKSizNkx5vpkiR (source in `summary/`).

## What it answers

| Question type | Example | What you get |
|---|---|---|
| Quantity with a public series | "What will the US unemployment rate be next month?" | Point forecast, 90 % conformal interval, which model produced it and why |
| Nowcast of an unpublished value | "How high will US flu activity be over the next two weeks?" | The week the CDC has not published yet, plus the weeks after |
| Direction of a price | "Will the S&P 500 be higher a month from now?" | Probability against the base rate, with the honest note that search data adds no edge |
| Discrete outcome | "Who will win the next US election?" | A refusal to forecast, plus a descriptive attention-share report with the validated hit-rate (21 of 36 past contests) |
| Private metric | "How many people will visit my website next month?" | A refusal with an offer: supply a CSV of the history and the full pipeline runs on it |

## Quick start

Requirements: Python 3.12+ (developed on 3.14), Windows or POSIX, internet access. No API keys.

```
pip install trendspy pandas numpy scipy statsmodels scikit-learn requests yfinance epiweeks pytest
python -m pytest tests -q
```

Inside Claude Code, in this repository, ask a prediction question and invoke `/trends-forecast`. The skill reads `.claude/skills/trends-forecast/SKILL.md` and its references, writes a spec, runs the pipeline and returns the answer with the artifact path.

To run the pipeline directly:

```
set PYTHONIOENCODING=utf-8          # Windows; needed for Hebrew output
python .claude/skills/trends-forecast/scripts/run_spec.py --truth-check "{\"kind\":\"fluview\",\"region\":\"nat\"}"
python .claude/skills/trends-forecast/scripts/run_spec.py --probe specs/us-flu-ili.json
python .claude/skills/trends-forecast/scripts/run_spec.py specs/us-flu-ili.json
python .claude/skills/trends-forecast/scripts/run_spec.py --open outputs/us-flu-ili.html
```

Other runner modes: `--salience` (attention share for contests, not a forecast), `--rerender` (rewrite the artifact with your own headline, answer text and confidence without recomputing), `--suggest` (topic ids for an entity), `--related` (related queries for a seed term).

## How the skill works

1. **Parse** the question: target quantity, geography, frequency, horizon, answer type, language.
2. **Feasibility gate**: is there a keyless public series of the actual outcome? If not, refuse with options. Discrete outcomes go to salience mode.
3. **Query design**: 8 to 15 behavioural search terms or topic ids, probed for privacy zeros. Breadth did not help in any experiment; a hand-picked dozen beat 25 to 60 expanded terms every time.
4. **Backtest**: rolling origin, publication-lag aware, against the best of naive, drift, seasonal-naive and AR. Preprocessing (Djorno-style zero repair, smoothing, detrending, clustering) is fitted inside each training window and chosen by the backtest, not assumed.
5. **Verdict**: Google Trends "helps" only if out-of-sample R² is positive against the benchmark and Clark–West or Diebold–Mariano supports it. Full-sample and shock-excluded views are both reported.
6. **Forecast**: from the search-augmented model only when it helps, otherwise from the benchmark. Rolling conformal interval. Confidence class follows the verdict, the robustness view and the current regime.
7. **Artifact**: a self-contained HTML page with history and forecast, backtest versus benchmark, metrics, query small-multiples, robustness table and data provenance.
8. **Learn**: the validated query set and lesson are appended to the playbook.

## What the experiments found

Fourteen experiment cells, each run by a context-isolated agent with one question and one approach, plus sixteen validation runs of the finished skill (eight questions, Fable 5.1 and Opus 5). Full write-ups in `research/experiments/`, synthesis in `research/conclusions.md`.

- Google Trends is a **nowcasting** instrument for series that are published with a lag, move at high frequency, and have a direct behavioural link to a search act. US influenza-like illness is the success case: 54 to 63 % lower squared error than the AR benchmark at the nowcast horizon, still 36 to 39 % two weeks out, in season only.
- It does **not** beat persistence for slow monthly aggregates (US and Israeli unemployment), for prices (S&P 500 against random walk with drift), or for discrete outcomes (attention share picks the election winner 58 % of the time).
- The benchmark decides the story. Against a weak AR-in-levels benchmark every search model "won"; against the random walk they lost. Against a zero-return naive the S&P model "won" by having an intercept; against drift it did not.
- Publication lag is where leakage hides. One row per origin was leaking the very value being nowcast; fixing it erased an apparent "Google caught April 2020" result.
- Clark–West alone is not evidence. It was significant for models with out-of-sample R² of minus four.
- Preprocessing is domain-specific. Detrending helped unemployment; every Djorno step hurt flu, where the spikes are the signal.
- Hebrew works end to end: Hebrew terms with `geo=IL`, Hebrew related-query expansion, Israel CBS ground truth, English answers.

## Repository layout

```
CLAUDE.md                          project goal, decisions, working rules
README.md
.claude/skills/trends-forecast/    the skill: SKILL.md, references/, scripts/run_spec.py
src/trends_predict/                library: gt, truth, preprocess, models, evaluate, report, pipeline
specs/                             question specs (JSON) used by the pipeline
outputs/                           per-question results (.json) and evidence artifacts (.html)
data/trends/                       cached Google Trends pulls, one dated vintage per download, catalog.jsonl
data/truth/                        cached ground-truth series with metadata sidecars
research/                          literature digest, approaches, experiment cells, conclusions, validation
docs/                              the literature review this project started from
summary/                           web summary generator and output
tests/                             synthetic tests for leakage, transforms, benchmarks, intervals, verdicts
```

## Data

- **Google Trends** via `trendspy`, free, no key. Every pull is stored as a dated vintage under `data/trends/` and never re-downloaded within a run; repeated pulls on later days are kept for retrieval-noise analysis. Values are relative search interest on a 0 to 100 scale per request, not counts. Weekly history beyond Google's five-year cap is stitched from overlapping windows rescaled on the overlap.
- **Ground truth**, all keyless: BLS (labour, CPI), CDC FluView via Delphi Epidata, Yahoo Finance, Wikimedia pageviews, the Israel CBS series API, and any user CSV. Registry and caveats in `.claude/skills/trends-forecast/references/truth-sources.md`.

## Development

```
python -m pytest tests -q                    # 11 synthetic tests
python summary/build_summary.py              # rebuild summary/index.html and summary/artifact.html
```

The library is edited often; the runner disables bytecode caching and removes `src/trends_predict/__pycache__` on start so a stale `.pyc` is never executed.

All documentation, code comments and commit messages are in English; questions to the skill may be in any language.
