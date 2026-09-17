# Answer rules — what every trends-forecast answer must contain

These rules come from the experiments in `research/experiments/` (leakage, weak benchmarks and
Clark–West-only claims each produced false positives). Follow them literally.

## Mandatory elements (in this order)

1. **The answer**: point forecast, units, target period, and the interval (nominal level + that it
   is a rolling conformal interval calibrated on out-of-sample residuals).
2. **Where it comes from**: "from the Google-Trends model `<name>`" or "from the target-only
   benchmark `<name>` because Google Trends did not add measurable value".
3. **The evidence line**: benchmark name; best GT model; OOS R² vs benchmark; Clark–West p; DM p;
   number of rolling origins; and the same for the shock-excluded view when it exists. One
   sentence each, numbers included.
4. **The confidence class** (`high` / `medium` / `low` / `none`) and the reason: `high` only when
   the verdict is *strong* in both views; `medium` when moderate or strong in one view only; `low`
   when the answer is the benchmark; `none` when no validation was possible.
5. **What drove the forecast**: the top GT features / clusters with sign, in words ("searches for
   'unemployment benefits' rose 30 % above their trailing-year level").
6. **Caveats specific to the case** (publication lag, holiday effects, regime shift, small n, proxy
   assumptions). Never generic filler.
7. **The artifact path** (`outputs/<id>.html`) and the data provenance (truth source, number of GT
   series, cache location). *Refusals are exempt from 1, 5 and 7*: they contain the reason, what
   was checked, the confidence class `none`, and the standing options.

## Wording rules
- Say "search interest" or "attention", never "demand", "support" or "cases" when describing GT.
- Never quote Clark–West alone as evidence of a gain; pair it with OOS R² (and DM).
- If the full-sample and shock-excluded verdicts disagree, say which way and why (e.g. "the gain
  comes from April 2020 when benefit searches led the BLS print by a month").
- A nowcast (h=0) is only a nowcast if the GT period is *after* the last published truth value;
  otherwise call it a one-period forecast.
- Direction questions: report the base rate of "up" and the model's hit-rate side by side; if the
  model's hit-rate does not beat the base rate, say the answer is the base rate.
- Reply in English regardless of the question's language (repo rule) unless the user asks otherwise;
  keep Hebrew inside query strings.

## Discrete outcomes (who wins / will X happen)
Validated finding (`research/experiments/04-Q4-F.md`, 56 Trends requests, US 2006–2024):
attention share does **not** predict winners. Pooled hit-rate 21/36 (58 %); presidential raw name
share 2/5 vs popular vote (Pearson share↔margin −0.21); intent modifiers ("vote for X", "X rally",
"X policies") 1–3/5 each, all negatively correlated with the margin; swing states 12/21 (p=0.33);
midterm party names 4/5 but n=5, p=0.07, shares within 0.45–0.52 and the 2026 reading flips from
R-leaning to D-leaning when one news-spike week is removed. Therefore:
- Do **not** turn a search share into a win probability, ever.
- Run the pipeline's **salience mode** (`run_spec.py --salience specs/<id>.json`) to give the user
  the current attention share, its weekly history and trend, with the "attention ≠ support" caveat
  and the hit-rates above quoted with n. This is a description, and the answer must call it that.
  Salience spec: `{"id","question","title","entities":[{"label":"Republican Party","query":"/m/07wbk"},…]
  or plain strings,"geo","timeframe","context"}` — no truth/freq/queries. In a salience answer the
  mandatory elements 1 and 3 are *replaced* by: leader and share (last 4 weeks vs previous 4, and
  excluding the loudest of the last 4 weeks), the validated hit-rates with n, and confidence `none`;
  element 7 (artifact path) still applies because salience produces an artifact.
- Redirect to the right target: polling averages or prediction-market prices as a *continuous*
  truth series (the user can supply a CSV; the pipeline can then validate a GT nowcast of the
  poll margin — that is a legitimate question, "who wins" is not).
- If the user insists on a number, give the historical hit-rate (58 %, n=36) as the calibration
  and nothing more.

## Refusal template (a complete answer — no spec, no run, no artifact)
"I can't produce a validated forecast for this. Google Trends measures relative search attention,
and to turn it into a forecast I need a historical series of the actual outcome to test the
search signal against. For `<target>` there is no keyless public series (checked: the skill's
truth registry — BLS, CDC FluView, Yahoo Finance, Wikipedia pageviews, Israel CBS — the cached
series in `data/truth/`, `<and any web search you did>`). Confidence: none. Three ways forward,
whenever you want them: (a) a CSV of the historical values (`date,value`, ≥ 60 monthly or ≥ 104
weekly points) and I will run the full validated pipeline; (b) a descriptive analysis of search
interest for `<brand/topic>` (attention, not a forecast); (c) the closest measurable proxy:
`<proxy>`, with the caveat that it measures the category, not `<target>`."
State the options as standing offers; do not end with a question when you cannot wait for a reply.
