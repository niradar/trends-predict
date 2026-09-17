# Candidate approaches for the trends-forecast skill

Status: **draft before experiments** (2026-09-17). Each approach is a complete pipeline from
question → answer; experiments will run each approach on the same representative questions
so they can be compared. After the experiments, `conclusions.md` records which parts survive.

## Shared skeleton (every approach uses this)

```
Q  question in natural language (any language)
1  Parse   → target concept, geography, horizon, answer type (quantity / direction / winner / yes-no)
2  Truth   → find a keyless ground-truth series for the target (or a proxy); if none → decline
3  Queries → candidate Google Trends queries/topics for the target (approach-specific)
4  Fetch   → cached, vintage-stamped GT pulls at the target's frequency
5  Validate→ rolling-origin backtest: no-GT benchmark vs GT model(s)
6  Forecast→ fit on all data, produce point + interval for the asked horizon
7  Answer  → plain-language answer with confidence class; HTML artifact with evidence
```

Answer types and how each is scored:

| Answer type | Example | Proxy quantity | Headline metric |
|---|---|---|---|
| Quantity | "What will US unemployment be next month?" | the series itself | OOS R² vs AR benchmark, MAE |
| Direction | "Will gold rise next month?" | return / change | directional accuracy, Clark–West |
| Winner (k-way) | "Who wins the election?" | relative search share per candidate vs. poll margin / historical results | hit-rate on past contests, Brier |
| Yes/No event | "Will there be a recession this year?" | continuous indicator + threshold | calibration on history if enough events, else *decline* |

## Approach A — "Choi–Varian minimal": hand-picked ARX

- Queries: 3–5 LLM-chosen keywords, one GT pull (jointly normalised).
- Features: level + one lag, log(x+1).
- Model: OLS ARX with AR(p) chosen by AIC, seasonal dummies; benchmark AR(p).
- Validation: expanding-window rolling origin, h=0 (nowcast) and h=1.
- Pros: fast, interpretable, hard to overfit. Cons: keyword choice is a guess; no preprocessing.

## Approach B — "ARGO-style": query expansion + LASSO, rolling window

- Queries: seed concepts → `related_queries` / `related_topics` expansion → 20–60 terms
  fetched in batches of 5 with a common anchor term for cross-batch rescaling.
- Preprocessing: log(x+1), standardise in-window, drop >50 % zero series.
- Model: LASSO (or elastic net) on [52 or 12 target lags + all GT terms], sliding window
  (104 weeks / 36 months), refit every origin; λ by inner time-series CV.
- Benchmarks: AR(p), seasonal naive.
- Pros: the literature's strongest default for weekly targets. Cons: many pulls, rate-limit
  exposure, unstable selection with correlated terms.

## Approach C — "Djorno preprocessing first": repair GT, then ARIMAX

- Queries: as in B but smaller (10–20).
- Preprocessing: privacy-zero imputation, hierarchical clustering of near-duplicate series
  (correlation distance) → cluster means, smoothing (spline/EWMA), detrending (rolling
  regression or first differences), all fit inside the training window.
- Model: SARIMAX with exogenous cluster signals; ablation no-GT → raw-GT → preprocessed-GT.
- Pros: directly addresses today's GT data quality. Cons: preprocessing choices multiply.

## Approach D — "Measurement-error aware": multi-vintage DLM

- Queries: 5–9 terms combined into one index (Rivera).
- Data: pull the same series R≥3 times (different sessions), store each vintage; use mean as
  signal and dispersion as measurement variance.
- Model: local-linear-trend + seasonal state-space with GT as regressor with time-varying
  coefficient (statsmodels `UnobservedComponents` / `MLEModel`).
- Pros: honest uncertainty, good at longer horizons. Cons: slower, more pulls.

## Approach E — "Large panel ML": elastic net + random forest

- Queries: 50–150 via aggressive expansion (Borup & Montes Schütte).
- Model: elastic net and random forest on lags of target + all GT features; OOS R².
- Pros: exploits heterogeneity. Cons: needs long history; most pulls; RF extrapolates badly.

## Approach F — "Discrete outcome via share proxy" (elections / winners)

- Queries: each contender as a **topic ID** (language-neutral) pulled jointly so shares are
  comparable; plus intent modifiers ("vote for X", "X policies").
- Proxy: search share `s_i = x_i / Σx`, smoothed.
- Validation: backtest on past contests (e.g. US 2008–2024 presidential, recent gubernatorial /
  Israeli Knesset elections): does the share (or its trend) match poll margins and the actual
  winner? Report hit-rate and calibration honestly (small n!).
- Answer: probability with wide interval, explicit statement of the proxy assumption and known
  failure mode (attention ≠ support — negative news drives searches too).

## What the experiments must decide

1. Does query *expansion* (B/E) beat hand-picking (A) out of sample, or just add noise?
2. Does Djorno-style preprocessing (C) reliably help versus raw GT?
3. Is multi-vintage dispersion (D) large enough to matter given today's GT?
4. How many GT pulls can we make per session before throttling, and what is the minimal
   pull budget that still gives a good answer?
5. For discrete outcomes (F): is the share proxy better than a coin flip on history?
6. Does the pipeline work with Hebrew queries and `geo=IL`?
