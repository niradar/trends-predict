# Google Trends data access — findings (2026-09-17)

## Verdict

**Free access works today via `trendspy` (v0.1.6).** No key, no proxy, no cookies. Every
capability the skill needs was verified live from this Windows host:

| Capability | Call | Result |
|---|---|---|
| Single keyword, weekly (5 y) | `interest_over_time(['unemployment benefits'], geo='US', timeframe='today 5-y')` | 262 rows, 0.3–2.6 s |
| Up to 5 keywords jointly normalised | 5 labour-market terms | 262 × 5 + `isPartial` |
| Monthly since 2004 | `timeframe='2004-01-01 2026-09-01'` | 273 rows |
| Daily (≤ 9 months) | `timeframe='today 3-m'` | 93 rows |
| Topic IDs (language-neutral) | `['/m/07s_c']` (Unemployment) | works |
| Hebrew keywords, Israel | `['דמי אבטלה','לשכת התעסוקה'], geo='IL'` | works |
| Category filter | `cat=45` (Health) | works |
| Related queries / topics | `related_queries('unemployment', geo='US')` | dicts of `top` / `rising` DataFrames |
| Entity lookup | `suggestions('influenza')` | mid + title + type |
| Worldwide (no geo) | `geo=''` | works |

`pytrends` (the classic library) is **broken** on current urllib3
(`Retry.__init__() got an unexpected keyword argument 'method_whitelist'`); it is not used.

## Google Trends semantics that the data layer must respect

1. **Values are relative, 0–100, normalised per request.** The max across all keywords *and*
   dates in a request is 100. Series from different requests are not comparable without an
   anchor. → `fetch_many()` batches keywords in groups of four plus a shared anchor keyword
   and rescales each batch by the anchor ratio (Eichenauer-style overlap rescaling).
2. **Frequency depends on timeframe length:** ≤ 9 months → daily; ≤ 5 years → weekly;
   longer → monthly. Different frequencies of the same query are *independently* normalised
   and mutually inconsistent. → Pull at the target's frequency; never stitch daily windows into
   weekly unless rescaled on overlap.
3. **`isPartial`** flags the current, incomplete period. → Drop it from modelling.
4. **Privacy zeros:** low-volume terms return 0 (below threshold), not "small". → Treat runs
   of zeros as missing/censored; drop series with > 50 % zeros; choose broader terms or topics.
5. **Repeat downloads:** identical within a session (server-side cache keyed on the request).
   Rivera-style dispersion appears across *days*. → Every pull is stored as a dated vintage;
   re-pull with `force=True` on later days to build a vintage set.
6. **Topics vs keywords:** topic IDs (`/m/…`, `/g/…`) aggregate all languages and spellings —
   the right choice for entities (candidates, diseases, products) and for Hebrew/English parity.

## Rate limiting

See the burst-test result recorded in `research/experiments/00-data-access-burst.md`.
Policy in the data layer regardless of result: exponential backoff (5, 10, 20, 40, 80 s),
max 5 attempts, then raise `TrendsUnavailable` so the skill can tell the user rather than
silently returning partial data. All pulls are cached, so retries never repeat successful work.

## Fallbacks (in order)

1. `trendspy` with backoff (primary).
2. Wait-and-resume: cache is per-series, so a throttled run resumes where it stopped.
3. Manual CSV import: the skill accepts a Google Trends "Download CSV" export dropped into
   `data/trends/manual/` and registers it in the cache with the same metadata schema.
4. Paid API (SerpApi / DataForSEO / official Trends API) — **only after asking the user**.

## Ground-truth sources verified keyless

| Domain | Source | Endpoint | Notes |
|---|---|---|---|
| US labour (unemployment rate, payrolls, CPI…) | BLS Public API v1 | `POST api.bls.gov/publicAPI/v1/timeseries/data/` | 25 req/day, ≤10 years per request, ≤25 series |
| US influenza (ILI %, clinical positivity) | Delphi Epidata (CMU) | `api.delphi.cmu.edu/epidata/fluview/`, `/fluview_clinical/` | weekly by epiweek, includes `release_date` for vintage-aware evaluation |
| Attention proxy / any Wikipedia topic | Wikimedia Pageviews REST | `wikimedia.org/api/rest_v1/metrics/pageviews/per-article/…` | daily since 2015-07, needs a User-Agent |
| Prices, indices, FX, crypto | Yahoo Finance via `yfinance` | — | daily |
| Box office | Box Office Mojo | HTML weekend tables | scrape |
| Polls / election results | Wikipedia tables | HTML | scrape with pandas.read_html |
| Israel official statistics | data.gov.il CKAN | `/api/3/action/package_search` | Hebrew datasets discoverable |
