# Experiment 00b — library smoke test on US unemployment (all approaches, one script)

Date: 2026-09-17 · Script: `scratch/smoke_model2.py` · Runtime 36 s · GT requests: 10 (cached after)

## Setup
- Truth: BLS `LNS14000000` monthly unemployment rate, 2004-01 → 2026-08 (271 obs).
- GT: 10 keywords pulled individually, monthly, `geo=US`: unemployment, unemployment benefits,
  file for unemployment, unemployment office, jobs hiring, indeed jobs, layoffs, severance,
  job openings, resume. None dropped by the privacy-zero filter (individual pulls).
- Design: 3 AR lags, 1 Fourier pair, GT lags 0–1, expanding window, 120 rolling origins
  (2016-09 → 2026-08), **target in differences** (`transform='diff'`).
- Horizons: h=0 nowcast (latest month's rate unknown, its GT known) and h=1.

## Results (bench = naive random walk)

| view | model | MAE | RMSE | OOS R² vs naive | CW p | DM p |
|---|---|---|---|---|---|---|
| h=0, all | naive | 0.256 | 1.015 | 0 | | |
| h=0, all | ar_diff | 0.241 | 0.838 | 0.32 | 0.11 | 0.27 |
| h=0, all | lasso + raw GT | 0.205 | 0.442 | 0.81 | 0.14 | 0.35 |
| h=0, all | **lasso + Djorno** | **0.171** | **0.394** | **0.85** | 0.13 | 0.29 |
| h=0, all | ridge + Djorno | 0.238 | 0.777 | 0.41 | 0.17 | 0.35 |
| h=0, excl. 2020-03..12 | naive | 0.103 | 0.139 | 0 | | |
| h=0, excl. | ar_diff | 0.126 | 0.159 | −0.31 | 0.70 | 0.005 (worse) |
| h=0, excl. | lasso + raw GT | 0.155 | 0.243 | −2.05 | 0.007 | 0.01 (worse) |
| h=0, excl. | lasso + Djorno | 0.114 | 0.154 | −0.22 | 0.003 | 0.45 |
| h=0, excl. | **ridge + Djorno** | **0.102** | **0.126** | **+0.18** | **0.003** | **0.07** |
| h=1, all | naive | 0.257 | 1.019 | 0 | | |
| h=1, all | best GT (ridge + Djorno) | 0.296 | 1.094 | −0.15 | 0.71 | 0.43 |
| h=1, excl. | naive | 0.103 | 0.139 | 0 | | |
| h=1, excl. | best GT (ridge + Djorno) | 0.115 | 0.148 | −0.13 | 0.83 | 0.02 (worse) |

Earlier run in **levels** (no `diff`): AR(3)+Fourier benchmark had RMSE 2.06 — worse than
naive (1.02) — so every GT model "won" by 50–65 % OOS R² against it. That gain was an
artefact of a weak benchmark.

## Findings

1. **Benchmark choice decides the story.** Against AR-in-levels everything looked great; against
   the random walk only specific configurations survive. The skill must benchmark against the
   best of {naive, seasonal naive, AR-diff}.
2. **GT value concentrates in nowcasting turning points.** With 2020 in the sample, GT models
   cut nowcast RMSE by ~60 % (the pandemic layoff wave was in the search data weeks before the
   BLS print). In calm periods the gain is small (18 %) and only with preprocessing.
3. **Raw GT hurts in calm periods; Djorno preprocessing rescues it.** Exactly the 2026 paper's
   result: raw log-GT models have OOS R² of −2 to −3.7 excluding the shock; smoothed + detrended
   + clustered GT with ridge is the only model that beats naive there.
4. **h ≥ 1 forecasting of unemployment from GT does not beat a random walk** at monthly
   frequency with these queries. The skill should say so instead of forcing an answer.
5. **Clark–West ≠ "beats the benchmark".** CW was significant (p<0.01) for models with strongly
   negative OOS R² — it detects *some* signal in the larger model after adjusting for estimation
   noise. The verdict rule must require OOS R² > 0 **and** CW/DM support (it does).
6. LASSO selected nothing at the final origin (pure random walk) — a sparse model can honestly
   say "no signal right now".

## Recommendation for the skill
Default pipeline for persistent monthly series: `transform='diff'`, benchmark set
{naive, ar_diff}, models {ridge+Djorno, lasso+Djorno, lasso+raw}, report the full-sample and
shock-excluded tables side by side, verdict per horizon, and prefer h=0 nowcast framing
("what is the number that has not been published yet") over h≥1 when GT only helps there.
