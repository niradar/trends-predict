# trends-predict

## Goal

Build a Claude Code **skill** (`/trends-forecast`) that lets a user ask a forecasting or
nowcasting question in natural language (e.g. *"Who will win the upcoming US election?"*,
*"Will US unemployment rise next month?"*, *"How bad will this flu season be?"*) and have the
agent:

1. Decide whether the question is answerable with Google Trends search data at all. If not,
   say so plainly and explain why (no fake confidence).
2. Propose candidate search queries that plausibly lead or track the target, then **validate
   them on history**: pull past Google Trends series, pull a ground-truth target series, and
   measure how well the queries actually predicted the target out-of-sample.
3. Apply methods from the peer-reviewed Google-Trends forecasting literature (Choi & Varian
   ARX augmentation, ARGO-style LASSO/ARX, MIDAS, state-space/DLM, elastic net / random
   forest panels, Djorno et al. preprocessing) — chosen per question — to produce a forecast
   with honest uncertainty.
4. Return a written, mathematically justified answer **and** a self-contained visual
   artifact (HTML) showing the evidence: query fit, backtest, forecast, uncertainty.
5. Persist every Google Trends download in a reusable, human-inspectable format so that
   experiments never re-download the same series and the user can audit the raw data.

Reference literature and method analysis: `docs/deep-research-report.md`.

## Direction (decisions made with the user, 2026-09-17)

- **Data access: free-first, paid fallback.** Use unofficial/free Google Trends access
  (pytrends / trendspy / hardened direct requests, manual CSV import as last resort). Only if
  free access proves genuinely unreliable, *stop and ask the user for a paid API key*
  (SerpApi / DataForSEO / Glimpse / official Trends API). Never silently degrade.
- **Ground truth: keyless public sources only** (BLS v1, CDC / Delphi Epidata, Yahoo Finance,
  Wikipedia pageviews & result tables, FiveThirtyEight archives, etc.). Tell the user when a
  free key (e.g. FRED) would unlock something better; do not require one.
- **Scope: broad — try everything, be honest.** Continuous quantities (unemployment, illness,
  tourism, box office, demand, prices) are the sweet spot. Discrete outcomes (elections,
  sports, "will X happen") are attempted via proxy quantities with loud, explicit uncertainty,
  and refused when the data cannot support an answer.
- **Language: English queries are the minimum requirement. Hebrew questions/queries are a
  bonus target** — design the query-generation step so it can translate the question, generate
  localized queries and use the right `geo` (e.g. `IL`).

## Working rules for the agent

- **Do not stop before the skill exists, is tested, and meets the requirements above.**
  This is an explicit instruction from the user: work continuously through research →
  data-access → experiments → conclusions → skill → sub-agent validation → web summary.
  Ask the user only when blocked on something only they can decide (e.g. a paid key).
- **The user may write in any language (often Hebrew). All documentation, code comments,
  commit messages and replies to the user are in English only.**
- Every Google Trends pull goes through the cached data layer (`data/`) — never call the API
  ad-hoc from a notebook or one-off script without writing the result to the cache.
- Every experiment writes its own conclusions file under `research/experiments/`; the
  aggregated conclusions live in `research/conclusions.md`.
- Sub-agents used for experiments and skill validation must be **context-isolated**
  (fresh `general-purpose` agents, not forks) so their results reflect what the skill alone
  can do.
- Prefer keyless, reproducible, scriptable steps. Windows host (PowerShell primary, Git Bash
  available); Python 3.14 at `C:\Python314`.

## Repository layout (target)

```
CLAUDE.md
docs/                      # literature review (input) + generated summaries
research/                  # conclusions, approach candidates, experiment logs
  conclusions.md
  approaches.md
  experiments/
data/                      # cached Google Trends + ground-truth series (CSV + JSON meta)
  trends/
  truth/
src/trends_predict/        # data layer, methods, evaluation, artifact rendering
.claude/skills/trends-forecast/   # the skill itself (SKILL.md + scripts + references)
summary/                   # final web summary with showcase examples
```

## Status log

- 2026-09-17: project started; requirements and decisions captured above.
- 2026-09-17: data access solved (`trendspy`, free); library `src/trends_predict` built and tested
  (`tests/`); 11 experiment cells run by isolated agents (`research/experiments/`); conclusions in
  `research/conclusions.md`; skill written at `.claude/skills/trends-forecast/`; pipeline specs in
  `specs/`, artifacts in `outputs/`; validation protocol in `research/validation/`; web summary
  generator `summary/build_summary.py`.

## How to run things

```
PYTHONIOENCODING=utf-8 python .claude/skills/trends-forecast/scripts/run_spec.py specs/<id>.json
PYTHONIOENCODING=utf-8 python -m pytest tests -q
PYTHONIOENCODING=utf-8 python summary/build_summary.py
```
