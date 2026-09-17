# Literature digest — what each paper teaches the skill

Source: `docs/deep-research-report.md` (methods review of the 12 papers) plus the paper
abstracts. This file is *my* reading of what transfers into an agent-driven skill, not a
restatement of the report. Each entry ends with a **Skill rule** — a concrete behaviour the
skill must have because of that paper.

| # | Paper | Core idea | Skill rule |
|---|---|---|---|
| 1 | Vosen & Schmidt 2011 — consumption, factor models | Many related search categories → a few latent factors beat single queries and beat survey indicators. | Generate a *family* of related queries, never a single keyword. Reduce with PCA/factor **inside the training fold** when K is large relative to T. |
| 2 | Choi & Varian 2012 — "Predicting the Present" | Simple AR(p) + seasonal + contemporaneous Google index. Nowcasting, not long-range. | Default first model is ARX: `y_t = c + Σφ y_{t-i} + β x_t + seasonal`. Always compare to the same model *without* x. Distinguish nowcast (h=0, contemporaneous x allowed) from forecast (h>0, only x observed at origin). |
| 3 | Fondeur & Karamé 2013 — French youth unemployment, state-space | Search index enters an unobserved-components model with Kalman filtering. | Provide a state-space/local-level option (statsmodels `UnobservedComponents` with exog) for targets with slowly drifting relationship. |
| 4 | Bangwayo-Skeete & Skeete 2015 — tourism, AR-MIDAS | Weekly GT vs monthly target: don't average to monthly, weight high-frequency lags with a parsimonious Almon polynomial. | When GT frequency > target frequency, build a MIDAS-style feature (exp-Almon weighted within-period lags) as an alternative to the monthly mean. |
| 5 | Yang, Santillana & Kou 2015 — ARGO (PNAS) | ARX with 52 target lags + up to 100 log-transformed queries, L1 penalty, **rolling 104-week refit**. Beat every benchmark. Robust to target revisions. | Strongest default for weekly targets with many queries: `log(x+δ)`, LASSO, sliding window, refit at every origin. Report RMSE/MAE/MAPE **and** correlation of increments. |
| 6 | Rivera 2016 — Puerto Rico hotels, DLM | Same series downloaded 11 times differs → treat GT as noisy measurement of a latent search process. Simpler models win short horizon, DLM wins >6 months. | Cache **every** download as a vintage; if ≥2 vintages exist, average them and estimate retrieval variance; surface the dispersion to the user as part of the uncertainty story. |
| 7 | D'Amuri & Marcucci 2017 — US unemployment | A "jobs" search index is a genuine leading indicator; gains largest at turning points; rigorous multi-horizon OOS. | Evaluate multi-horizon (h = 0..H) separately; report directional/turning-point accuracy, not just RMSE. |
| 8 | Bulut 2018 — exchange rates, Clark–West | GT beats random walk mainly on **direction**, heterogeneous across pairs. Nested-model inference needs Clark–West. | For price/return/binary-ish targets, score directional accuracy and use Clark–West (nested) or Diebold–Mariano (non-nested) to test if the GT gain is significant. Be candid when it is not. |
| 9 | Rangarajan, Mody & Marathe 2019 — ARLR sparse selection | Greedy likelihood-ratio forward selection with AICc stopping; deseasoning; backfilled data inflates apparent skill. | Offer ARLR-style forward selection as an interpretable alternative to LASSO; deseason GT on training data only; warn when ground truth is likely revised/backfilled. |
| 10 | Borup & Montes Schütte 2022 — employment growth, ~172-query panel | Large panel + elastic net / random forest; heterogeneity across queries is the source of gain; OOS R² vs benchmark is the metric. | Query generation should aim for breadth (dozens of terms via related-queries expansion), then let elastic net select. Primary headline metric: **OOS R² vs target-only benchmark**. |
| 11 | Eichenauer, Indergand, Martínez & Sax 2022 — consistent GT series | Daily/weekly/monthly GT windows are independently normalised and mutually inconsistent; need a rescaling/stitching procedure for long series. | Fetch one frequency per experiment at the target's frequency where possible; when stitching windows is unavoidable, rescale on the overlap; store the stitching metadata. |
| 12 | Djorno, Santillana & Yang 2026 — statistical preprocessing | Modern GT has zeros, sampling noise, level shifts; **raw GT can make forecasts worse**. Fix: hierarchical clustering of redundant queries → smoothing splines → detrending → ARIMAX. Gains 58% national / 24% state. | Always run the ablation `no-GT → raw-GT → preprocessed-GT`. Preprocess: replace privacy zeros, smooth (spline/EWMA), detrend, cluster near-duplicate queries. Only claim GT helps if the preprocessed model beats the no-GT model out of sample. |

## Cross-cutting lessons the skill must encode

1. **GT is auxiliary.** Every model keeps the target's own history. A search-only model is a
   diagnostic, never the answer.
2. **Leakage is the main way to fool yourself.** Query selection, standardisation, deseasoning,
   PCA, λ tuning — all inside the training window, rolling origin, never random K-fold.
3. **Benchmarks are the product.** The deliverable is *"GT model vs. best target-only model"*.
   If the delta is ≤ 0 the honest answer is "search data adds nothing measurable here".
4. **Preprocess before you model.** Zeros, spikes, level shifts and sampling noise in current
   GT are real; raw GT under-performs.
5. **Uncertainty is mandatory.** Residual/block bootstrap or conformal intervals; report
   coverage; say when intervals are wide.
6. **Frequency matters.** Match GT frequency to target frequency, or use MIDAS weights.
7. **Vintage awareness.** Cache every pull with timestamp; repeated pulls are information,
   not waste.

## Where the literature is silent (gaps the skill has to handle itself)

- **Discrete outcomes (elections, sports, yes/no events).** No paper forecasts a binary event
  from GT. The closest analogues are Bulut's directional accuracy and D'Amuri's turning
  points. Approach: convert to a proxy quantity (e.g. relative search share between
  candidates, or a poll-margin series) whose history can be validated against real
  outcomes/polls, then be explicit that the mapping from proxy → outcome is the weak link.
- **Query discovery.** Papers start from hand-chosen concepts. An agent must generate the
  candidate universe itself: LLM brainstorming + Google Trends *related queries* expansion +
  Google Trends *topics* (entity IDs, language-independent) + semantic families
  (symptoms / purchase intent / job loss / destination terms).
- **Non-English.** Only Fondeur & Karamé (French) is non-English. For Hebrew: use topic IDs
  where possible (language-neutral), `geo=IL`, and Hebrew keyword variants.
- **Ground truth acquisition.** Not discussed; the skill needs keyless fetchers (BLS, Delphi
  Epidata/CDC, Yahoo Finance, Wikipedia, election result tables) and a clear "no ground
  truth obtainable → cannot validate → decline or downgrade" path.
