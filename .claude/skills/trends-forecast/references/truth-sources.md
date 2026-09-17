# Keyless ground-truth sources (what the pipeline can validate against)

A forecast can only be *validated* if a historical target series exists. This is the registry the
skill consults in step 2. Every source here works without an API key and is wrapped in
`trends_predict.truth`; the spec `truth` block selects it.

| Domain | Series | Spec `truth` block | Freq | Notes |
|---|---|---|---|---|
| US unemployment rate (SA) | BLS LNS14000000 | `{"kind":"bls","id":"UNRATE"}` | M | Published ~1st Friday after month end. Oct-2025 missing (shutdown) — pipeline interpolates. |
| US unemployment rate (NSA) | BLS LNU04000000 | `{"kind":"bls","id":"UNRATE_NSA"}` | M | Use with seasonal GT if you want raw seasonality. |
| US nonfarm payrolls | BLS CES0000000001 | `{"kind":"bls","id":"PAYEMS"}` | M | Level in thousands; model in differences. |
| US CPI (all items) | BLS CUUR0000SA0 / CUSR0000SA0 | `{"kind":"bls","id":"CPIAUCSL","derive":"pct_change"}` | M | "Will inflation accelerate?" = will the m/m rate rise: derive the rate with `derive` (pct_change / logdiff, gaps interpolated first), then `transform: null`, `answer_type: "direction"` compares next month's rate with the last one. |
| Any BLS series | raw id | `{"kind":"bls","id":"<BLS id>"}` | M | 25 requests/day without a key. **All** BLS series miss Oct-2025 (shutdown); the pipeline interpolates the gap before any `derive` differencing. |
| US influenza-like illness % | CDC FluView via Delphi | `{"kind":"fluview","region":"nat","start_epiweek":201040}` | W | Regions `hhs1..hhs10` are current. **State/city ILINet series (`ny`, `jfk`, …) stopped in Sep-2025** — for a state question use its HHS region (NY → `hhs2`) with `geo` = the state (`US-NY`) and say the truth is regional. Published ~1 week late → true nowcast possible. |
| US flu lab positivity % | FluView clinical | `{"kind":"fluview_clinical","region":"nat"}` | W | since 2015-40. |
| Stock indices, FX, crypto, commodities | Yahoo Finance | `{"kind":"yahoo","ticker":"^GSPC","freq":"M","log":true}` | D/W/M | `freq` resamples to last close; `log` models log prices (returns). Partial current period dropped. |
| Attention to any topic | Wikipedia pageviews | `{"kind":"wikipedia","article":"Influenza","project":"en.wikipedia","freq":"W"}` | D/W/M | Since 2015-07. Also `he.wikipedia`. Proxy, not an outcome. |
| Israel unemployment rate (LFS, 15+, SA) | CBS series API | `{"kind":"cached","name":"cbs_unemployment_rate_il"}` — **already cached** in `data/truth/` (2012-01 →, monthly) | M | Spliced at 2025-01 from the old to the new LFS definition (mean gap 0.04 pp) — carry that caveat. Publication lag ~6 weeks → "next month" = next print (h=0). To refresh or fetch other CBS series: `https://apis.cbs.gov.il/series/data/list?id=<series id>&format=json` (ids 491094 old / 41097 new), save with `truth._save`. |
| Anything the user has | CSV | `{"kind":"csv","path":"data/truth/my.csv","date_col":0,"value_col":1}` | any | Ask the user for a CSV when no public series exists. |

## Sources that need a free key (tell the user, do not require)
- FRED (`fred.stlouisfed.org`) — thousands of series incl. initial jobless claims (ICSA, weekly),
  retail sales, housing starts. A free key would unlock weekly labour nowcasting.

## No ground truth → what to do
- **Discrete outcomes (elections, awards, sports):** the outcome series is too short to validate a
  model (a handful of contests). See `answer-rules.md` § Discrete outcomes.
- **Company-internal quantities (my sales, my signups):** ask for a CSV; without it, refuse to
  forecast — you can only *describe* search interest.
- **"Will X happen" with no measurable X:** refuse, explain that Google Trends measures attention,
  and offer the closest measurable proxy if one exists.
