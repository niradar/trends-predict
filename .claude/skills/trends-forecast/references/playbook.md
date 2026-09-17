# Query & configuration playbook (learned; append after every validated run)

Format: domain → what worked, what failed, recommended spec fragment. The skill reads this before
designing queries and **appends a dated entry after every run** — full entry for completed runs
(any verdict), one line under "Refused" for feasibility-gate refusals — so query design and
scoping improve over time.

## US unemployment rate (monthly, BLS UNRATE) — 2026-09-17, cells 01-Q1-A/B/C + specs/us-unemployment.json
- Verdict: **none** at h=0 and h=1 in the full sample (OOS R² −0.10…−0.25 vs naive); +13–21 % only
  in the 2020-excluded view with ridge + rolling-detrended GT (DM n.s.). GT did *not* catch the
  April-2020 jump once leakage was fixed (nowcast 4.7 % vs 14.8 %).
- Query expansion hurt: 27 terms → LASSO picked 34–43 of 59 features, 2× worse than naive; sliding
  36-month windows selected nothing (random walk). Keep ≤ 10–15 terms.
- Ablation: detrending helps, EWMA neutral, **clustering hurt** (`ridge+djorno_nocluster`).
- Spec: `specs/us-unemployment.json` (freq M, diff, windows [null, 36], exclude 2020-03→12).
- Answer template: "unchanged at 4.1 % ± 0.2 pp (random walk); Google Trends adds no measurable
  value at monthly frequency; its only signal is directional (DirAcc 0.58–0.67)."

## US unemployment rate — 12-term replication — 2026-09-17, specs/us-unemployment-opus.json (run `-opus`)
- Repeat of the run above with two extra action-intent terms (`unemployment insurance`,
  `unemployment claims`; 12 total). All terms already cached → **0 live Trends requests, 76 s**.
- Same conclusion: verdict **none** full-sample at h=0 (`ridge+raw` OOS R² −0.14, CW p=0.52,
  DM p=0.57, n=120 origins) and h=1 (`ridge+djorno@w36` −0.15, CW p=0.15, DM p=0.25);
  **moderate** in the 2020-excluded view at h=0 (`ridge+djorno_nocluster` +0.22, CW p=0.003 but
  DM p=0.17, n=110) and h=1 (`ridge+raw` +0.07, CW p=0.006, DM p=0.57). Answer = naive: 4.1 %
  for 2026-09 (90 % interval 3.9–4.3). Confirms detrending helps, clustering hurts, and every
  `@w36` sliding-window variant is worse than its expanding twin in all four tables.
- The two claims-style terms added nothing → the proven 10-term set is the recommended one; the
  open lever is **frequency, not query count** (weekly FRED ICSA, needs a free key).
- Truth now ends 2026-08 (4.1 %) and monthly GT also ends 2026-08, so **h=0 is not a nowcast**
  here: `forecast_next` returns "no unpublished period to nowcast; returned h=1 forecast", but
  that note is attached to the GT forecast only — when the benchmark wins (`chosen_model: naive`)
  the h=0 `note` is `null` and h=0/h=1 print the identical 4.1 %. Call it a one-period-ahead
  forecast for unemployment regardless of what the h=0 row says.
- Probe correlations mislead here: `file for unemployment` has corr(ΔY, lag 1) = 0.79, almost
  entirely COVID; in the fitted ridge its lag-1 coefficient is *negative*. Don't read the probe
  table as evidence of signal for this target.
- Regime split (median 5.0 %): naive MAE 0.10 pp when the rate is low vs 1.19 pp when high, and
  GT is worse in both (0.11 / 1.54) — the honest caveat is that neither model sees turning points.

## US influenza-like illness (weekly, FluView wILI) — 2026-09-17, cells 02-Q2-A/C + specs/us-flu-ili.json
- Verdict: **strong** at h=0 (OOS R² +0.54…0.59 vs AR, DM 0.01–0.03, CW <0.001) and h=2 (+0.36…0.39);
  h=4 not reliable. True nowcast possible (GT extends one week past the CDC print).
- Queries: flu symptoms, flu, influenza, cough, tamiflu, flu test, sore throat, body aches, flu shot
  (negative sign), how long does flu last. Drop `fever` (US summer term). All ≥ 0.5 non-zero.
- Preprocessing: **raw log GT + ridge/OLS wins; every Djorno step hurt** (clustering collapsed 15/17
  queries). Keep `ridge+raw`, `lasso+raw`, `ols+raw` first in `models`.
- Gains are in-season only; off-season the target-only model is ~2× better — state the season.
- Spec: `specs/us-flu-ili.json` (freq W, levels, p 4, fourier_k 2, min_train 104, clip_min 0,
  history_years 9 → stitched windows).

## US influenza-like illness (weekly, FluView wILI) — 2026-09-17, specs/us-flu-fable.json (early-season rerun, h=0/1/2)
- Verdict: **strong** at every horizon vs AR, n=150 origins (Oct-2023 → Sep-2026): h=0 OOS R² 0.63 (CW 0.001, DM 0.011),
  h=1 0.57 (CW 0.001, DM 0.018), h=2 0.39 (CW <0.001, DM 0.037). `ols+raw` won all three; `ridge+raw` within 0.01 R²;
  `lasso+djorno` negative at h=2. Confirms "raw beats Djorno" for flu.
- Queries: the playbook's 10-term set without `fever` (+ `how long does flu last`, the single best term, lag-0 corr 0.77
  with ILI changes). All nonzero share 1.0; 0 live requests (cache from the earlier run covered the 9-year stitch).
- Regime split matters for the *answer*: below 1.92% wILI the GT model's MAE is 1.6–1.9× AR's; above it, GT is 30–35%
  better. In September (last print 1.51%) report `medium`, not `high`, and say the edge arrives in-season.
- With `history_years 9`, `min_train 104`, `max_origins 150` the evaluated origins start ~Oct-2023, so the COVID
  `exclude` window is outside the sample and the robustness view is omitted — either accept the single view or raise
  `max_origins` (≈ 300) to reach back into 2021.
- Bug fixed on this run: `pipeline.write_outputs` crashed with `UnboundLocalError: _fmt` whenever `table_excluded` was
  absent (a function-local `from .report import _fmt` shadowed the module-level `_fmt`); the import line was removed.

## US influenza-like illness — rerun with 9y stitched history — 2026-09-17, specs/us-flu-opus.json
- Verdict **strong** at h=0/1/2 with the *AR* benchmark winning the target-only race at all three:
  OOS R² +0.63 / +0.57 / +0.39, CW p ≤ 0.0006, DM p 0.011 / 0.018 / 0.037, 150 origins, n=468.
- The playbook query set works as advertised: dropping `fever` and adding **`how long does flu last`**
  (nonzero 1.0, corr with ΔwILI 0.77 — the single best term) improved h=0 OOS R² from +0.54 (10-term
  set with `fever`) to +0.63. `flu shot` stays (corr 0.12, useful as a negative-signed seasonal control).
- `ols+raw` edged out `ridge+raw` at every horizon once the sample was 468 weeks (9-year stitched)
  instead of 260; Djorno again worse (clustering worst), and **every `@w104` sliding window was worse
  than expanding** — confirms "windows: [null]" for weekly flu. Adding 104 costs nothing but noise.
- h=1 (`y_known: false`) is valid and useful, not just h=0/h=2: the origin is the GT week past the
  CDC print, so h=0/1/2 = nowcast + the two genuinely future weeks. Use these three for "next two weeks".
- Regime split matters for the *answer*: above the 1.92 % median GT beats AR (MAE 0.17 vs 0.27 at h=0),
  below it GT is worse (0.117 vs 0.073; 0.35 vs 0.20 at h=2). In September the level is below median →
  report confidence **medium** and quote the AR benchmark's flat path as the alternative.
- With `history_years: 9` the `exclude` COVID window falls outside the 150 evaluated origins, so no
  shock-excluded view is produced (pipeline says so explicitly) — "strong in both views" is then
  unreachable; treat the single shock-free view as evidence for medium, not high.
- Gotcha: `--probe` ignores `history_years` (it always uses `fetch_many`/`today 5-y`), so probe
  diagnostics are computed on 260 weeks while the run uses 468.

## S&P 500 direction (monthly & weekly) — 2026-09-17
- Verdict: none. Any "gain" vs zero-return naive was the drift intercept; vs `drift` OOS R² ≈ −0.01.
- Always include `drift` in the benchmark set for prices; report base rate of up months (~66 %).
- Answer: drift ± conformal interval, probability ≈ base rate, "Google Trends adds no edge".

## S&P 500 direction (monthly, Yahoo ^GSPC log close) — 2026-09-17, specs/sp500-direction-fable.json (run `-fable`, h=1 & h=2)
- Queries (12, all kept, nonzero ≥ 0.99): stock market, dow jones, market crash, bear market, recession, buy stocks,
  stocks to buy, sell stocks, inflation, gold price, layoffs, unemployment. 3 new terms (dow jones, stocks to buy,
  bear market) cost 6 live requests in the probe (429 retries count against `REQUEST_COUNT`); full run 0 requests, 47 s.
- Verdict: **none** at both horizons in both views, n=120 origins (115 excl. Feb–Jun 2020). h=1: best `lasso+raw`
  OOS R² −0.004 vs `drift` (CW p 0.70, DM p 0.30); excluded view `lasso+djorno` −0.005 (CW 0.79, DM 0.16). h=2:
  `lasso+raw` −0.006 (CW 0.86, DM 0.25); excluded −0.008. Every GT model's DirAcc (0.64–0.73) is within ±0.02 of
  drift's own (0.675 h=1, 0.717 h=2). `ols+raw` is the only model *significantly worse* than drift (DM p 0.005/0.03).
- Probe correlations are all lag-0 and negative (attention spikes *with* falls: dow jones −0.40, bear market −0.40,
  stock market −0.34) — contemporaneous, not leading; lag-1 |r| ≤ 0.13. Adding intent variants did not change that.
- Answer = drift: from the Aug-2026 close 7686, +0.7 %/month → 7741 for Sep (90 % conformal 7146–8385), 7796 for Oct
  (7058–8611); P(up) 0.57–0.60 vs base rate of up months 0.66 (h=1) / 0.73 (2-month). Confidence low.
- Data quirk: `data/truth/yahoo_GSPC_Close_M.csv` (retrieved 2026-09-17 16:46 UTC) contains a **partial September row**
  (7636.91) although `truth.yahoo` now drops the incomplete period; alignment with GT (ends 2026-08) removed it, so the
  backtest is clean, but `--truth` reports `last 2026-09-01`. Delete the cache or wait for `max_age_days=1` to refresh.
- Lesson: third run on this target, third `none`; the negative lag-0 correlations are the whole story (news follows
  price). Stop spending runs on monthly S&P query design; the only untested lever is weekly frequency with `drift`.

## S&P 500, "higher a month from now?" (monthly, Yahoo ^GSPC, log) — 2026-09-17, specs/sp500-1m-opus.json (run `-opus`)
- Verdict **none** at h=1 and h=2, full sample *and* 2020-excluded → answer = `drift`, confidence **low**.
  h=1 best `ridge+diffgt` OOS R² −0.010 (CW 0.35, DM 0.72, n=120); h=2 `ridge+diffgt` +0.010 (CW 0.13, DM 0.49,
  n=120); excluded view +0.002 / +0.009, CW 0.18. Third independent confirmation of the "S&P: none" entry above.
- 14 terms, all kept (nonzero ≥ 0.99): stock market, market crash, bear market, recession, buy stocks, sell stocks,
  stocks to buy, vix, 401k, buy gold, gold price, debt, inflation, layoffs. Adding the Preis–Moat–Stanley `debt`
  term, `vix`, `bear market`, `401k` to the older 8-term set changed nothing → stop expanding the query set here.
- **The diagnostic worth keeping:** GT attention to markets is *coincident, not leading*. corr with Δlog price at
  lag 0 is −0.40 (`bear market`), −0.34 (`stock market`), −0.34 (`vix`), −0.30 (`recession`); at lag 1 it collapses
  to −0.00 / −0.11 / −0.09 / −0.03. That is the whole story of why monthly GT cannot beat drift on equities.
- Direction questions: quote DirAcc side by side — drift 0.675 (h=1) / 0.717 (h=2); **no** GT model beat it
  (best 0.667 / 0.717). `forecast.prob_up` came out 0.57 / 0.60 vs base rates 0.66 / 0.73.
- `windows [null, 36]` with 14 queries: every `@w36` variant was far worse (ols+raw@w36 OOS R² −71 at h=1, −84 at
  h=2). Consistent with unemployment. Just use `[null]` for price targets.
- `regime_split` is **meaningless for a trending price level**: threshold 7626.28 (index units) against a log-price
  series gave n_high=120 / n_low=0, so the regime downgrade rule cannot fire. Do not read it for prices.
- Anchor caveat that matters for the user-facing answer: monthly truth ends at the last *complete* month
  (2026-08 close 7,686) while the index traded ~7,640–7,660 mid-September, so the h=2 point (7,796) is only ~+2 %
  above the live level. Always restate the forecast relative to the current quote, not just the anchor.
- Cost: 7 live Trends requests (6 new terms, 1 re-pulled after a 429 backoff), pipeline run 101 s, 0 requests.

## Elections / winners — 2026-09-17
- Raw attention share: 2/4 recent presidential contests (Romney 0.67 lost, Harris 0.56 lost).
- See `research/experiments/04-Q4-F.md` for modifiers, state-level and midterm tests.

## Own-website traffic ("how many people will visit my website next month") — 2026-09-17 (run `-opus`)
- **Refused at the feasibility gate; 0 Trends requests, no spec, no artifact.** Site-internal
  analytics (sessions/users) have no keyless public series, and the question does not even name a
  domain, so neither the truth series nor the brand/topic query set can be built.
- `truth.py` kinds are bls / fluview / fluview_clinical / yahoo / wikipedia_pageviews / csv_series —
  no web-analytics source exists or can exist keylessly (third parties cannot read a private GA4).
- Correct reframe to offer: user exports monthly sessions from GA4/Plausible/Cloudflare →
  `{"kind":"csv",...}`, `freq M`, `transform diff`, `horizons [{"h":1,"y_known":true}]`,
  queries = brand/domain name + the 6–10 behavioural searches the site actually ranks for
  (from Search Console), `clip_min 0`. Lesson: brand-name GT series are usually mostly privacy
  zeros for small sites — probe `nonzero_share` before promising anything.

## US midterms 2026, salience mode — 2026-09-17, specs/us-midterms-2026-salience-fable(.json / -topics.json)
- Hebrew question "מי ינצח בבחירות הקרובות בארצות הברית?" → discrete outcome → `--salience`, no forecast.
- Keywords ("Democratic Party"/"Republican Party", geo US, 2026-06-01..09-17): R 0.54 / D 0.46 last 4 weeks,
  flipped from D 0.58 / R 0.42 in the previous 4. Topic ids (`/m/0d075m`, `/m/07wbk`): R 0.56 / D 0.44 vs
  R 0.51 / D 0.49. Both agree on the leader; the gap size depends on the query form (0.08 vs 0.12) — always
  run both and quote the range, never a single share.
- Cost: keyword pull cached (0 requests); topic pull 4 live calls (three 429 retries, 62 s). `--suggest`
  also hit 429s. The trendspy client is rate-limited today; salience is cheap enough to tolerate it.
- Pipeline quirk: the "excl. spike week" column drops the *global* max-total week (here 2026-06-06), so it
  was identical to the last-4 share and missed the real spike: the keyword R-lead is one week (w/e 2026-09-05,
  R share 0.71); the other three recent weeks were 0.43 / 0.51 / 0.51. Always print the weekly share table
  (cached, 0 requests) and check the spike *within* the last-4 window. Topic-id entities render as raw
  `/m/…` labels in the HTML — keep a keyword-labelled artifact as the user-facing one.
- Lesson: the descriptive answer must lead with the 21/36 hit-rate and the redirect (generic-ballot poll
  average / prediction-market CSV as a continuous truth series); the share itself is a footnote.

## US 2026 midterms, "who will win" (Hebrew question) — 2026-09-17, specs/us-election-2026-opus.json
- No forecast (correct outcome): discrete winner, no keyless truth series → salience mode only.
- Party **topic ids** work and are the right entity choice: `/m/0d075m` Democratic Party (US),
  `/m/07wbk` Republican Party (US). Joint pull, 12-month timeframe → 52 weekly points, 1 request.
- Reading: R share 56.3 % of joint attention in the last 4 weeks (50.0 % in the previous 4),
  52.0 % mean over 52 weeks, range 44.1–61.3 %, R above 50 % in 40/52 weeks. The recent R lean is
  one 4-week drift (58.1 %, 61.3 % in the last two weeks); dropping the highest-attention week
  inside the last four leaves 54.7 %, so it is not a single-week artefact — but it is still a
  ±6 pp band around 50 %, i.e. inside the noise of a 58 %-hit-rate indicator.
- Lesson / caveat for salience mode: `share_last4_ex_spike_week` drops the **global** max-total
  week, which for a 12-month window is usually an old election/news week (here 2025-11-02, the
  off-year elections, when D led attention). It then equals `share_last4` and the spike diagnostic
  is vacuous. Recompute the "excluding the loudest of the last four weeks" number by hand, or pass
  a shorter timeframe (`today 3-m`) if you want the built-in column to mean something.

## US CPI inflation — "will inflation accelerate next month" — 2026-09-17, specs/us-inflation-opus.json (+ -yoy-opus)
- Target design is the whole job here: the BLS registry gives the CPI *index*, but the question is about the
  *rate*. Derived two truth series with `truth._save` and referenced them as `{"kind":"cached"}` (the same
  pattern the registry documents for Israeli CBS): `bls_cpi_mom_sa_pct` = 100·Δlog(CUSR0000SA0) and
  `bls_cpi_yoy_nsa_pct` = y/y % of CUUR0000SA0. **Reindex the index to a full monthly range and interpolate
  before differencing** — BLS is missing Oct-2025 (shutdown), and `.diff()` / `.shift(12)` on a gapped index
  silently produce a 2-month change labelled 1-month and a misaligned y/y. Also: no `clip_min` (m/m can be
  negative). With the rate as the target, `transform: "diff"` + `answer_type: "direction"` makes "up" mean
  exactly "acceleration", so the direction block (base_rate_up / prob_up / DirAcc) answers the question.
- Queries (12, all kept, nonzero ≥ 0.58): inflation, gas prices, gas prices near me, cheap gas, cost of living,
  grocery prices, food prices, coupons, used car prices, rent increase, price increase, egg prices. Probe
  correlations with Δ(m/m rate) are tiny (|r| ≤ 0.27, best = `gas prices` lag 0 = 0.27, i.e. contemporaneous
  energy only); nothing at lag 1–2.
- Verdict: **none** everywhere. m/m, 120 origins, bench = **ar** (naive/drift are 15–30 % worse — for a rate that
  is already a difference the random walk is a weak benchmark, always let AR into the set): h=0 lasso+raw
  OOS R² −0.00 (CW 0.07, DM 0.98), h=1 lasso+djorno −0.05, h=2 lasso+raw −0.03; 2020-excluded no better
  (−0.02…−0.09). y/y: lasso+raw −0.04 (h=1) / −0.06 (h=2). Djorno detrending hurt at every horizon.
- Answer = AR: Sep-2026 +0.31 % m/m vs +0.40 % in Aug (90 % −0.11…+0.74), P(accel) ≈ 0.37 vs a 0.51 base rate;
  y/y 3.43 % vs 3.40 %, P(up) ≈ 0.54. Confidence low. Regime split (median 0.22 % m/m): GT and AR are within
  0.001 pp of each other in both regimes — no regime downgrade, just no signal.
- Cost: 5 live requests for all 12 queries (fetch_many batches ~3 keywords/request), 60 s; the y/y rerun and
  both rerenders were 0 requests. GT monthly data end at the last *full* month (2026-08), so h=0 is not a
  nowcast (the pipeline says so) — call it a one-month-ahead forecast for the September print.
- Lesson: consumer-price search attention is a *level* story (it tracked the 2021–22 surge) but says nothing
  about the *month-to-month acceleration* that the question asks about; don't retry with more price terms.
  The plausible upgrade is a different target (weekly gasoline prices as an energy nowcast) or inflation
  expectations, not a bigger query set.

## Hebrew / Israel — 2026-09-17
- Hebrew keywords with `geo=IL` return series; check nonzero share (privacy zeros).
- Truth: CBS series API (`apis.cbs.gov.il/series/...`). See `05-Q5-A`.

## Israel unemployment rate (monthly, CBS LFS SA) — 2026-09-17, specs/il-unemployment-opus.json (run `-opus`)
- Hebrew question "מה יהיה שיעור האבטלה בישראל בחודש הבא?" → truth `{"kind":"cached","name":"cbs_unemployment_rate_il"}`
  (already in `data/truth/`, 175 obs 2012-01..2026-07, SA, spliced old/new census definition at 2025-01).
  **1 live Trends request** (11 of 12 terms already cached), ~6 min end to end; run itself 128 s.
- Queries (12, geo IL, all kept, nonzero share 1.00 since 2012): the 05-Q5-A set (דמי אבטלה, לשכת התעסוקה,
  שירות התעסוקה, ביטוח לאומי אבטלה, דרושים, פיטורים, חלת, `/m/07s_c`) + אבטלה, משרות, חיפוש עבודה, פיצויי פיטורים.
  The four added terms changed nothing — the 8-term set is enough. Probe correlations are all |r| ≤ 0.17 at every lag,
  i.e. the probe table is honestly uninformative here (unlike the US case where 2020 inflates it).
- Verdict: **none** in the full sample at h=0/1/2 — best `ridge+djorno_nocluster` OOS R² +0.008 (CW 0.078, DM 0.89,
  n=115) at h=0, +0.018 (CW 0.041, DM 0.78, n=114) at h=1, `lasso+raw` −0.030 at h=2 (n=113). Shock-excluded
  (2020-03..2021-06): **moderate** at all three (+0.031 / +0.041 / +0.086, CW 0.035 / 0.023 / 0.019, DM 0.45–0.55).
  Benchmark `naive` wins everywhere → answer 3.1 % for 2026-08 [2.7, 3.5] and 2026-09 [2.6, 3.6], confidence **low**.
- `ridge+djorno_nocluster` (rolling detrend, no clustering) is again the best GT variant, as for US unemployment —
  the detrend-yes/cluster-no combination is now 3 for 3 on monthly labour-market targets.
  Every `@w36` sliding window was far worse (OOS R² −0.05 … −10.4); with 12 queries do not bother adding 36.
- h=0 **is** a genuine nowcast here (monthly GT covers 2026-08, CBS is published ~6 weeks late and ends 2026-07) —
  unlike BLS, where GT and truth end in the same month. Check `note` on the h=0 forecast: null = real nowcast.
- Reporting hook: Israeli layoff attention (`פיטורים`) tripled in 2026 (31 in Jan → 100 in Jun → 77 in Aug vs a 46
  trailing-12-month mean) while benefit-claim searches stayed flat and the SA rate moved 2.7→3.1. Describe it, never
  forecast from it — exactly the 2020 furlough pattern (search spikes the headline rate cannot show).
- Regime split (median 4.2 %): naive beats GT in both regimes (low 0.164 vs 0.171, high 0.171 vs 0.179) — no regime
  downgrade is available beyond `low`; the honest line is that neither model sees turning points.
- Artifact nit (report.py:251): the confidence tile renders `<div class="tile-value" class="tile-value conf-…">`,
  a duplicate `class` attribute, so the colour class is dropped by browsers. Text is correct.

## New York flu, "how bad in two weeks" — 2026-09-17, specs/ny-flu-fable.json (Fable; truth = HHS Region 2)
- **State-level FluView is stale in Delphi Epidata**: `region: ny` and `jfk` (NYC) both end 2025-09-27 (782 rows), so a
  state nowcast for 2026 is impossible. `hhs2` (NY, NJ, PR, VI) is current to 2026-09-05 (831 rows, same as `nat`) — use
  it as the New York truth and say so; keep the searches on `geo: US-NY`. Check the `--truth` tail *before* pulling GT.
- Horizons: "two weeks from now" is **h=4** from the last CDC print (print 09-05, today 09-17, target w/e 10-03); the
  GT-model horizons h=0..2 run from the GT origin (w/e 09-12) while the AR benchmark's h=3/h=4 run from the truth origin,
  so h=2 (GT) and h=3 (AR) land on the *same* target week — quote both as a range.
- Queries: the proven 10-term flu set, all kept (nonzero ≥ 0.76 in US-NY; `flu symptoms` lag-0 r 0.51). 20 live requests
  (10 terms × 2 stitched windows), probe 84 s, full 5-horizon run 127 s.
- Verdict vs AR, n=150 origins (Oct-2023 → Sep-2026), single view (COVID window outside sample): h=0 **moderate**
  `ols+raw` OOS R² +0.14 (CW 0.0002, DM 0.29); h=1 moderate `ridge+djorno` +0.07 (CW 0.006, DM 0.44); h=2 moderate
  `ridge+djorno` +0.02 (CW 0.018, DM 0.80); h=3/h=4 **none** (−0.02 / −0.01, AR chosen). Weaker than national
  (+0.63 at h=0): regional ILI is noisier and NY-only searches cover ~2/3 of the region.
- Regime split (median 2.60 %): GT beats AR only in the high regime (h=0 MAE 0.36 vs 0.38); in the low regime AR is
  1.5–2× better at every horizon. Current level 1.50 % → downgrade medium → **low** everywhere. Answer: 2.3 % for w/e
  2026-10-03 (AR), nowcast 1.8 % (GT) vs 1.7 % (AR) for w/e 09-12.
- Lesson: here Djorno (`ridge+djorno`) won h=1/h=2, unlike national flu where raw won everything — keep both families.
  The 90 % conformal band at h=4 (0–6.6 %) is calibrated over all seasons and is useless off-season; quote the
  low-regime MAE (0.23 pp) next to it.

## New York flu (weekly ILI) — 2026-09-17, specs/ny-flu-ili-opus.json (run `-opus`, 20 requests, 210 s)
- **Truth trap: Delphi `fluview` region `ny` is dead after epiweek 202539 (w/e 2025-10-04)** while `nat`,
  `nj`, `pa`, `hhs2`, `cen2` all run to 202635. Always `--truth` the *state* series before promising a
  state forecast. Substitute used: `hhs2` (NY+NJ+PR+VI), which vs the NY-only series over 2010–2025 has
  level corr 0.913, weekly-change corr 0.873, mean ratio NY/Region2 = 0.987 — a defensible NY proxy.
  `fluview_clinical` has no state/`ny` region at all (`result: -2`). NY-DOH lab-confirmed counts
  (Socrata `health.data.ny.gov`, dataset `jr8b-6gh6`) exist but were last updated 2026-05-29 → 15 weeks
  stale in September, useless for a 2-week-ahead target; catalog search endpoint
  `api.us.socrata.com/api/catalog/v1?q=influenza&domains=health.data.ny.gov` is the quick way to check.
- Queries: the proven 10-term flu set (no `fever`) at `geo US-NY`. All 10 kept (nonzero 0.76–1.0; the
  sparsest are `how long does flu last` 0.76, `tamiflu` 0.84, `flu test` 0.88 — state geo thins the tail
  but does not kill it). 20 live requests = 10 queries × 2 stitched 5-year windows (`history_years 9`).
- Horizon arithmetic matters: with `y_known false` and GT one week ahead of the CDC print, h counts from
  the *nowcast* week, so "in two weeks" from the ask date is **h=3**, not h=2. Ran h=0..3.
- Verdicts (bench `ar`, n=300 origins, Dec-2020→Sep-2026): h=0 strong/strong (+0.18 / +0.16, DM 0.05 /
  0.08, `ols+raw`), h=1 strong/moderate (+0.16 / +0.12), h=2 moderate/moderate (+0.18 / +0.08), h=3
  strong/moderate (+0.22 / +0.08, CW 2e-4, DM 0.09, `ridge+raw`). `max_origins 300` (playbook's earlier
  suggestion) does reach back into the COVID window and produces the second view — it works.
- Regime split (threshold 2.60% wILI) forced the downgrade: GT beats AR only *above* threshold
  (0.71 vs 0.82 pp MAE below it at h=3). Current level 1.50% → below → medium dropped to **low**.
  Same story as the national September run: state-level flu in September is an off-season question.
- Answer: 2.31% wILI for w/e 2026-10-03 (AR benchmark 2.07%), 90% conformal 0.0–5.1% — the interval is
  ~±2.8 pp because the residual pool includes season peaks; say plainly that it does not constrain much.
  Useful framing for "how bad": percentile of the point in the whole history (53rd) plus the same
  calendar week in past seasons (2.4–3.2% in 2021–25) plus when the season actually peaks (Dec–Feb).

## US unemployment rate (monthly, BLS UNRATE) — 2026-09-17, specs/us-unemployment-fable.json (fresh run, Fable)
- Queries (13, all kept, nonzero ≥ 0.59): file for unemployment, how to apply for unemployment, unemployment
  claim, unemployment benefits, unemployment login, unemployment office, layoffs, severance, jobs hiring,
  jobs near me, part time jobs, indeed, unemployment. Adding claimant *action* terms (claim / login / how to
  apply) raised probe correlations with ΔUNRATE (lag-1 r 0.5–0.8, driven by 2020) but did not change the verdict.
- Verdict: **none** again in the full sample (h=1 best lasso+djorno@w36 OOS R² −0.10, CW 0.77; h=0 ridge+raw
  −0.24). 2020-excluded: **moderate** — ridge+raw +13 % at h=1 (CW 0.002, DM 0.36), ridge+djorno_nocluster
  +26 % at h=0 (CW 0.001, DM 0.11). Sliding 36-month windows: all worse than expanding. Forecast = naive
  4.1 % ± 0.2 pp for 2026-09, confidence low. 4 Trends requests (12/13 cached), ~3 min.
- Lesson: at monthly frequency the calm-period edge is ~0.01–0.03 pp MAE — below BLS rounding. Do not spend
  more runs on query design here; the only upgrade is a weekly truth (FRED ICSA needs a key) or the NSA rate
  with seasonal GT. Also: GT monthly data stop at the last full month, so h=0 silently equals h=1 — call it a
  one-month forecast.

## Israel unemployment rate (monthly, CBS LFS 15+ SA) — 2026-09-17, specs/il-unemployment-fable.json (Hebrew question, Fable)
- Truth: `{"kind":"cached","name":"cbs_unemployment_rate_il"}` refreshed first from the CBS series API (sids 491094 + 41097,
  script `scratch/refresh_cbs_il_fable.py`, same splice as 05-Q5-A). Latest print still 2026-07 = 3.1 % (published 2026-08-16);
  monthly GT ends 2026-08 → a genuine h=0 nowcast of August exists. 175 months 2012-01..2026-07.
- Queries (13 → 12 kept): דמי אבטלה, ביטוח לאומי אבטלה, שירות התעסוקה, לשכת התעסוקה, חלת, פיטורים, פיצויי פיטורים, דרושים,
  חיפוש עבודה, קורות חיים, alljobs, /m/07s_c. Dropped by probe: `תביעת אבטלה` (nonzero share 0.48). All kept series are
  100 % non-zero since 2012; probe correlations with ΔY are all |r| ≤ 0.17 (no COVID-driven inflation here because the SA
  target never spiked). 6 live requests (probe; 8 terms cached from 05-Q5-A), 0 in the full run; ~7 min end to end.
- Verdict: **none** full-sample at every horizon (h=0/1/2/3, n=115/114/113/112 vs naive): best `ridge+djorno_nocluster`
  OOS R² +0.01 (CW 0.07, DM 0.85) at h=0, +0.02 (CW 0.04, DM 0.76) at h=1; `lasso+djorno` −0.03 / −0.04 at h=2/3.
  COVID-furlough-excluded view: **moderate** at all four horizons (+0.03 / +0.04 / +0.09 / +0.13, CW 0.01–0.03) but DM
  p ≈ 0.5 and MAE gap 0.01 pp → "suggestive, not validated". Every `@w36` variant worse than expanding (R² down to −9.8).
  Regime split (median 4.2 %): naive beats GT in both regimes. Answer = naive 3.1 % for 2026-08 [2.7, 3.5], 2026-09
  [2.6, 3.6], 2026-10 [2.5, 3.7]; confidence low.
- Horizon semantics to remember: with `y_known` true, h=k targets last-published + k, so **h=0 and h=1 both target the first
  unpublished month** (h=0 uses that month's GT, h=1 does not); "next month" in the calendar sense from a mid-month question
  is h=3 here (Jul print → Oct). Cover h=0..3 to be safe; it costs nothing once the pulls are cached.
- Lesson: same as the US — at monthly frequency the search data ties the random walk (detrending helps, clustering and
  sliding windows hurt). The one thing the August search data says that the model cannot use: `פיטורים` (layoffs) +67 %
  vs trailing year in Jul–Aug 2026 while benefit-claim terms are flat — report it as attention, not as a forecast input.
  Israel-specific caveat stays: the SA rate does not count furloughed workers, so a real shock can appear in searches first.

## US CPI inflation, "will it accelerate next month?" — 2026-09-17, specs/us-inflation-accel-fable.json
- Target framing matters: `answer_type: direction` compares the forecast with the *last level* of the truth, so
  feeding log CPI would answer "will prices rise?" (always yes). Used the **m/m SA inflation rate itself** as
  truth (100·Δlog CUSR0000SA0, derived with `truth.bls_series('CPIAUCSL')`, saved to
  `data/truth/us_cpi_mm_sa_src.csv`, `{"kind":"csv"}`), `transform: null` (already stationary), so "up" =
  acceleration and naive = "same m/m as last month". Interpolate the Oct-2025 shutdown gap *before* differencing
  (271 months, 2004-02 → 2026-08). Base rate of acceleration ≈ 51 %.
- Queries (14, all kept, nonzero ≥ 0.58): gas prices, gas prices near me, grocery prices, food prices, egg
  prices, rent increase, price increase, cost of living, coupons, used car prices, electric bill, car insurance,
  tariffs, inflation. Probe correlations with Δ(m/m) all |r| ≤ 0.27 (`gas prices` lag 0 only).
- Verdict: **none** in both views vs AR(3)+Fourier (bench MAE 0.18 pp). Full: `lasso+raw` OOS R² −0.02, CW 0.15,
  DM 0.79, n=120; excl. 2020: `lasso+djorno` +0.02, CW 0.06, DM 0.78, n=110. Every `@w36` sliding window
  worse than expanding (−0.27…−0.81). Forecast = AR: Sep-2026 +0.27 % m/m (90 % −0.31…+0.85) vs Aug +0.40 %
  → P(accelerate) 0.31; GT models' own points 0.28–0.32 %, same direction. Confidence low.
- Lesson: headline m/m CPI is too noisy (MAE 0.18 pp ≈ the whole Aug→Sep gap) for monthly GT to help; the only
  hint of signal is **directional** — `lasso+raw` DirAcc 0.66 vs AR 0.55 / base 0.51 with zero MAE gain — so report
  it as suggestive, never as the answer. Next lever would be a *component* truth (gasoline CPI or energy index,
  BLS ids) where `gas prices` has a real mechanism, not more headline queries. Cost: 16 requests (13 pulls + 3
  429 retries) in the probe, 0 in the run; ~2.5 min of pipeline time.
