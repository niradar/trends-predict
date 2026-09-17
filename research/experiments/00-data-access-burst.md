# Experiment 00 — Google Trends access & rate-limit burst

Date: 2026-09-17 · Tool: `trendspy` 0.1.6 · Host: Windows 11, residential connection, no proxy

## Setup
30 distinct single-keyword `interest_over_time` calls (`geo='US'`, `timeframe='today 5-y'`),
0.5 s sleep between calls, immediately after ~15 feature-probe calls.

## Result
- **30/30 succeeded, 0 errors, 37.1 s total (~1.2 s/call).**
- Earlier feature probe: multi-keyword, monthly/weekly/daily, topic ids, Hebrew + `geo=IL`,
  categories, related queries/topics, suggestions — all succeeded.
- Two back-to-back downloads of the same request were byte-identical (Google serves a cached
  normalisation for a repeated request); retrieval noise must be measured across days.

## Implications for the skill
- A pull budget of ~30–60 requests per question is safe at ~1 s spacing; the data layer
  spaces calls ≥0.6 s and backs off exponentially on any error.
- Because everything is cached as vintages, re-running an experiment costs zero requests.
- `pytrends` is not viable (urllib3 incompatibility); do not fall back to it.
