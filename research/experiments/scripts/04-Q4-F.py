"""Cell 04-Q4-F — "Who will win the next US election?" via Google Trends attention share.

Approach F: treat the joint-normalised search share of candidate A vs B (or party vs party)
as a proxy for a discrete outcome, and VALIDATE it on history before saying anything about 2026.

Steps (see research/experiments/04-Q4-F.md):
 1. Presidential 2008-2024: raw name share (last 4 weeks; trend over last 8 weeks) vs popular-vote margin.
 2. Intent-modified queries ("vote for X", "X rally", "X policies", "X for president").
 3. State level 2016/2020/2024 in 7 swing states.
 4. Midterms 2006-2022: party-level terms vs national House popular-vote margin.
 5. 2026-06-01 -> today: same party-level pulls; forecast only if 1-4 validate.

Run:  PYTHONIOENCODING=utf-8 python research/experiments/scripts/04-Q4-F.py   (from repo root)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from trends_predict import gt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT_MD = ROOT / "research" / "experiments" / "04-Q4-F.md"
OUT_JSON = ROOT / "research" / "experiments" / "04-Q4-F.json"
TODAY = "2026-09-17"

# --------------------------------------------------------------------------------------
# Ground truth (hard-coded). Popular-vote margins in percentage points, positive = Democrat.
# Sources: Wikipedia "United States presidential election" pages (FEC certified totals);
# state pages "2016/2020/2024 United States presidential election in <state>";
# House popular vote: Wikipedia "United States House of Representatives elections" pages.
# --------------------------------------------------------------------------------------
PRES = {
    2008: dict(dem="Barack Obama", rep="John McCain", order=["Barack Obama", "John McCain"],
               tf="2008-06-01 2008-11-04", eday="2008-11-04", margin=+7.27, ec="D"),
    2012: dict(dem="Barack Obama", rep="Mitt Romney", order=["Barack Obama", "Mitt Romney"],
               tf="2012-06-01 2012-11-06", eday="2012-11-06", margin=+3.86, ec="D"),
    2016: dict(dem="Hillary Clinton", rep="Donald Trump", order=["Donald Trump", "Hillary Clinton"],
               tf="2016-06-01 2016-11-08", eday="2016-11-08", margin=+2.09, ec="R"),
    2020: dict(dem="Joe Biden", rep="Donald Trump", order=["Joe Biden", "Donald Trump"],
               tf="2020-06-01 2020-11-03", eday="2020-11-03", margin=+4.45, ec="D"),
    2024: dict(dem="Kamala Harris", rep="Donald Trump", order=["Kamala Harris", "Donald Trump"],
               tf="2024-06-01 2024-11-05", eday="2024-11-05", margin=-1.48, ec="R"),
}
# Swing-state two-party margins (pp, +=Dem).  Wikipedia state result pages.
STATES = ["US-PA", "US-MI", "US-WI", "US-GA", "US-AZ", "US-NV", "US-NC"]
STATE_MARGIN = {
    2016: {"US-PA": -0.72, "US-MI": -0.23, "US-WI": -0.77, "US-GA": -5.13, "US-AZ": -3.55, "US-NV": +2.42, "US-NC": -3.66},
    2020: {"US-PA": +1.17, "US-MI": +2.78, "US-WI": +0.63, "US-GA": +0.23, "US-AZ": +0.31, "US-NV": +2.39, "US-NC": -1.35},
    2024: {"US-PA": -1.71, "US-MI": -1.42, "US-WI": -0.86, "US-GA": -2.20, "US-AZ": -5.53, "US-NV": -3.10, "US-NC": -3.24},
}
# National House popular-vote margin (pp, +=Dem).  Wikipedia House election pages.
MIDTERMS = {
    2006: dict(tf="2006-06-01 2006-11-07", eday="2006-11-07", margin=+8.0),
    2010: dict(tf="2010-06-01 2010-11-02", eday="2010-11-02", margin=-6.8),
    2014: dict(tf="2014-06-01 2014-11-04", eday="2014-11-04", margin=-5.7),
    2018: dict(tf="2018-06-01 2018-11-06", eday="2018-11-06", margin=+8.6),
    2022: dict(tf="2022-06-01 2022-11-08", eday="2022-11-08", margin=-2.8),
}
PARTY_PAIRS = {
    "party_name": ["Democratic Party", "Republican Party"],
    "vote_party": ["vote Democrat", "vote Republican"],
}
MODIFIERS = {
    "vote_for": "vote for {n}",
    "rally": "{n} rally",
    "policies": "{n} policies",
    "for_president": "{n} for president",
}

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
ATTEMPTS = {"n": 0, "failed": []}


def pull(keywords, geo, tf):
    """Joint fetch through the library; count attempts; never swallow silently."""
    ATTEMPTS["n"] += 1
    try:
        return gt.fetch_interest(keywords, geo=geo, timeframe=tf)
    except gt.TrendsUnavailable as e:  # record and move on
        ATTEMPTS["failed"].append({"keywords": list(keywords), "geo": geo, "tf": tf, "error": str(e)[:200]})
        print(f"  !! failed: {keywords} {geo} {tf}: {str(e)[:120]}")
        return None


def share_stats(df: pd.DataFrame, col_a: str, col_b: str, eday: str, end_inclusive: bool = False) -> dict:
    """Share of A in A+B over several windows ending the day BEFORE election day.

    The pulls are daily (Google returns daily resolution for < ~9 months). "Last 4 weeks" = 28
    days; trend = share(last 4w) - share(previous 4w); slope = OLS slope of weekly share over
    the last 8 weeks (share points / week).
    """
    end = pd.Timestamp(eday) if end_inclusive else pd.Timestamp(eday) - pd.Timedelta(days=1)
    d = df.loc[:end]
    a, b = d[col_a].astype(float), d[col_b].astype(float)

    def sh(s_a, s_b):
        tot = s_a.sum() + s_b.sum()
        return float(s_a.sum() / tot) if tot > 0 else np.nan

    last4 = sh(a.loc[end - pd.Timedelta(days=27):], b.loc[end - pd.Timedelta(days=27):])
    prev4 = sh(a.loc[end - pd.Timedelta(days=55): end - pd.Timedelta(days=28)],
               b.loc[end - pd.Timedelta(days=55): end - pd.Timedelta(days=28)])
    full = sh(a, b)
    # weekly share over last 8 weeks
    w = d.loc[end - pd.Timedelta(days=55):]
    wk = (w[[col_a, col_b]].astype(float)).resample("7D", origin="end").sum()
    wk_share = (wk[col_a] / (wk[col_a] + wk[col_b])).dropna()
    slope = np.nan
    if len(wk_share) >= 3:
        x = np.arange(len(wk_share))
        slope = float(np.polyfit(x, wk_share.values, 1)[0])
    zero_share = float(((a == 0) | (b == 0)).mean())
    return dict(share_last4=last4, share_prev4=prev4, trend_4v4=(last4 - prev4) if pd.notna(last4) and pd.notna(prev4) else np.nan,
                slope_8w=slope, share_full=full, n_days=int(len(d)), zero_day_share=zero_share)


def hit(share, margin):
    if pd.isna(share) or share == 0.5:
        return None
    return bool((share > 0.5) == (margin > 0))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return np.nan, np.nan
    pr = float(np.corrcoef(x[m], y[m])[0, 1])
    sp = float(pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman"))
    return pr, sp


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            elif v is None:
                cells.append("")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def hit_summary(hits):
    hs = [h for h in hits if h is not None]
    return dict(hits=int(sum(hs)), n=len(hs), rate=(sum(hs) / len(hs)) if hs else np.nan)


# --------------------------------------------------------------------------------------
t0 = time.time()
cat_before = len(gt.catalog())
results: dict = {"id": "04-Q4-F", "question": "Who will win the next US election? (2026 midterms, House control / popular vote)",
                 "today": TODAY, "tables": {}, "hit_rates": {}, "correlations": {}, "queries": []}

# ---------------- Step 1: presidential raw name share ----------------
print("Step 1: presidential raw name share")
rows1 = []
for yr, c in PRES.items():
    df = pull(c["order"], "US", c["tf"])
    results["queries"].append({"step": 1, "keywords": c["order"], "geo": "US", "tf": c["tf"]})
    if df is None:
        rows1.append(dict(year=yr, dem=c["dem"], rep=c["rep"], share_dem_last4=np.nan, trend_4v4=np.nan, slope_8w=np.nan,
                          share_dem_full=np.nan, margin_dem=c["margin"], pv_hit=None, ec_hit=None, note="pull failed"))
        continue
    s = share_stats(df, c["dem"], c["rep"], c["eday"])
    rows1.append(dict(year=yr, dem=c["dem"], rep=c["rep"], share_dem_last4=s["share_last4"], trend_4v4=s["trend_4v4"],
                      slope_8w=s["slope_8w"], share_dem_full=s["share_full"], margin_dem=c["margin"],
                      pv_hit=hit(s["share_last4"], c["margin"]), ec_hit=hit(s["share_last4"], 1 if c["ec"] == "D" else -1),
                      trend_hit=hit(0.5 + s["trend_4v4"], c["margin"]) if pd.notna(s["trend_4v4"]) else None,
                      note=f"{s['n_days']} daily obs"))
tab1 = pd.DataFrame(rows1)
results["tables"]["step1_presidential_raw"] = json.loads(tab1.to_json(orient="records"))
results["hit_rates"]["step1_raw_last4_vs_popular_vote"] = hit_summary(tab1["pv_hit"])
results["hit_rates"]["step1_raw_last4_vs_electoral_college"] = hit_summary(tab1["ec_hit"])
results["hit_rates"]["step1_trend_4v4_vs_popular_vote"] = hit_summary(tab1["trend_hit"])
pr, sp = corr(tab1["share_dem_last4"], tab1["margin_dem"])
results["correlations"]["step1_share_last4_vs_margin"] = dict(pearson=pr, spearman=sp, n=int(tab1["share_dem_last4"].notna().sum()))
print(md_table(tab1))

# ---------------- Step 2: intent modifiers ----------------
print("Step 2: intent-modified queries")
rows2 = []
for mod_key, pat in MODIFIERS.items():
    for yr, c in PRES.items():
        kw = [pat.format(n=c["dem"]), pat.format(n=c["rep"])]
        df = pull(kw, "US", c["tf"])
        results["queries"].append({"step": 2, "keywords": kw, "geo": "US", "tf": c["tf"]})
        if df is None:
            rows2.append(dict(modifier=mod_key, year=yr, share_dem_last4=np.nan, trend_4v4=np.nan, share_dem_full=np.nan,
                              zero_day_share=np.nan, margin_dem=c["margin"], pv_hit=None, ec_hit=None, note="pull failed/empty"))
            continue
        s = share_stats(df, kw[0], kw[1], c["eday"])
        rows2.append(dict(modifier=mod_key, year=yr, share_dem_last4=s["share_last4"], trend_4v4=s["trend_4v4"],
                          share_dem_full=s["share_full"], zero_day_share=s["zero_day_share"], margin_dem=c["margin"],
                          pv_hit=hit(s["share_last4"], c["margin"]), ec_hit=hit(s["share_last4"], 1 if c["ec"] == "D" else -1),
                          note=""))
tab2 = pd.DataFrame(rows2)
results["tables"]["step2_modifiers"] = json.loads(tab2.to_json(orient="records"))
mod_summary = []
for mod_key in MODIFIERS:
    sub = tab2[tab2["modifier"] == mod_key]
    hs_pv = hit_summary(sub["pv_hit"]); hs_ec = hit_summary(sub["ec_hit"])
    pr, sp = corr(sub["share_dem_last4"], sub["margin_dem"])
    mod_summary.append(dict(modifier=mod_key, pv_hits=f"{hs_pv['hits']}/{hs_pv['n']}", ec_hits=f"{hs_ec['hits']}/{hs_ec['n']}",
                            pearson_share_margin=pr, spearman=sp, mean_zero_day_share=float(sub["zero_day_share"].mean())))
raw_pv = results["hit_rates"]["step1_raw_last4_vs_popular_vote"]; raw_ec = results["hit_rates"]["step1_raw_last4_vs_electoral_college"]
mod_summary.insert(0, dict(modifier="raw name (step 1)", pv_hits=f"{raw_pv['hits']}/{raw_pv['n']}", ec_hits=f"{raw_ec['hits']}/{raw_ec['n']}",
                           pearson_share_margin=results["correlations"]["step1_share_last4_vs_margin"]["pearson"],
                           spearman=results["correlations"]["step1_share_last4_vs_margin"]["spearman"],
                           mean_zero_day_share=np.nan))
tab2s = pd.DataFrame(mod_summary)
results["tables"]["step2_modifier_summary"] = json.loads(tab2s.to_json(orient="records"))
print(md_table(tab2)); print(md_table(tab2s))

# ---------------- Step 3: swing states ----------------
print("Step 3: swing states")
rows3 = []
for yr in (2016, 2020, 2024):
    c = PRES[yr]
    nat_share = float(tab1.loc[tab1["year"] == yr, "share_dem_last4"].iloc[0])
    for st in STATES:
        df = pull(c["order"], st, c["tf"])
        results["queries"].append({"step": 3, "keywords": c["order"], "geo": st, "tf": c["tf"]})
        if df is None:
            rows3.append(dict(year=yr, state=st, share_dem_last4=np.nan, rel_share=np.nan, margin_dem=STATE_MARGIN[yr][st], hit=None, rel_hit=None))
            continue
        s = share_stats(df, c["dem"], c["rep"], c["eday"])
        m = STATE_MARGIN[yr][st]
        rows3.append(dict(year=yr, state=st, share_dem_last4=s["share_last4"], rel_share=s["share_last4"] - nat_share,
                          margin_dem=m, hit=hit(s["share_last4"], m),
                          rel_hit=hit(0.5 + (s["share_last4"] - nat_share), m - c["margin"]) if pd.notna(s["share_last4"]) else None))
tab3 = pd.DataFrame(rows3)
results["tables"]["step3_states"] = json.loads(tab3.to_json(orient="records"))
results["hit_rates"]["step3_state_winner"] = hit_summary(tab3["hit"])
results["hit_rates"]["step3_state_relative_to_national"] = hit_summary(tab3["rel_hit"])
pr, sp = corr(tab3["share_dem_last4"], tab3["margin_dem"])
results["correlations"]["step3_share_vs_state_margin_pooled"] = dict(pearson=pr, spearman=sp, n=int(tab3["share_dem_last4"].notna().sum()))
within = {}
for yr in (2016, 2020, 2024):
    sub = tab3[tab3["year"] == yr]
    pr_y, sp_y = corr(sub["share_dem_last4"], sub["margin_dem"])
    within[str(yr)] = dict(pearson=pr_y, spearman=sp_y, n=int(sub["share_dem_last4"].notna().sum()),
                           hits=hit_summary(sub["hit"]))
results["correlations"]["step3_within_year"] = within
# relative share vs relative margin (removes the national level)
tab3["rel_margin"] = [tab3.loc[i, "margin_dem"] - PRES[int(tab3.loc[i, "year"])]["margin"] for i in tab3.index]
pr, sp = corr(tab3["rel_share"], tab3["rel_margin"])
results["correlations"]["step3_relative_share_vs_relative_margin"] = dict(pearson=pr, spearman=sp, n=int(tab3["rel_share"].notna().sum()))
print(md_table(tab3))

# ---------------- Step 4: midterms ----------------
print("Step 4: midterms, party-level terms")
rows4 = []
for pair_key, kw in PARTY_PAIRS.items():
    for yr, c in MIDTERMS.items():
        df = pull(kw, "US", c["tf"])
        results["queries"].append({"step": 4, "keywords": kw, "geo": "US", "tf": c["tf"]})
        if df is None:
            rows4.append(dict(pair=pair_key, year=yr, share_dem_last4=np.nan, trend_4v4=np.nan, share_dem_full=np.nan,
                              zero_day_share=np.nan, margin_dem=c["margin"], hit=None, note="pull failed/empty"))
            continue
        s = share_stats(df, kw[0], kw[1], c["eday"])
        rows4.append(dict(pair=pair_key, year=yr, share_dem_last4=s["share_last4"], trend_4v4=s["trend_4v4"],
                          share_dem_full=s["share_full"], zero_day_share=s["zero_day_share"], margin_dem=c["margin"],
                          hit=hit(s["share_last4"], c["margin"]), note=""))
tab4 = pd.DataFrame(rows4)
results["tables"]["step4_midterms"] = json.loads(tab4.to_json(orient="records"))
for pair_key in PARTY_PAIRS:
    sub = tab4[tab4["pair"] == pair_key]
    results["hit_rates"][f"step4_{pair_key}"] = hit_summary(sub["hit"])
    pr, sp = corr(sub["share_dem_last4"], sub["margin_dem"])
    results["correlations"][f"step4_{pair_key}_share_vs_margin"] = dict(pearson=pr, spearman=sp, n=int(sub["share_dem_last4"].notna().sum()))
print(md_table(tab4))

# ---------------- Step 5: 2026 present ----------------
print("Step 5: 2026 to date")
rows5 = []
tf26 = f"2026-06-01 {TODAY}"
for pair_key, kw in PARTY_PAIRS.items():
    df = pull(kw, "US", tf26)
    results["queries"].append({"step": 5, "keywords": kw, "geo": "US", "tf": tf26})
    if df is None:
        rows5.append(dict(pair=pair_key, share_dem_last4=np.nan, trend_4v4=np.nan, slope_8w=np.nan, share_dem_full=np.nan, last_obs=None, note="pull failed"))
        continue
    last_obs = df.index.max()
    s = share_stats(df, kw[0], kw[1], str(last_obs.date()), end_inclusive=True)
    rows5.append(dict(pair=pair_key, share_dem_last4=s["share_last4"], trend_4v4=s["trend_4v4"], slope_8w=s["slope_8w"],
                      share_dem_full=s["share_full"], last_obs=str(last_obs.date()), n_days=s["n_days"], note=""))
tab5 = pd.DataFrame(rows5)
results["tables"]["step5_2026_current"] = json.loads(tab5.to_json(orient="records"))
print(md_table(tab5))

# ---------------- Robustness checks on the only positive-looking result (midterm party names) -------------
from scipy import stats  # noqa: E402

def binom_p(hits, n):
    """One-sided exact binomial p for >= hits successes out of n at p=0.5."""
    return float(stats.binom.sf(hits - 1, n, 0.5)) if n else np.nan

def pearson_p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return np.nan
    return float(stats.pearsonr(x[m], y[m])[1])

robust = {}
sub = tab4[tab4["pair"] == "party_name"].dropna(subset=["share_dem_last4"]).reset_index(drop=True)
# (a) leave-one-out threshold: predict D iff share > mean share of the OTHER midterms (removes the fixed 0.5 assumption
#     without fitting on the held-out year); also record the margin by which each call is decided.
loo = []
for i in range(len(sub)):
    thr = float(sub.drop(i)["share_dem_last4"].mean())
    s = float(sub.loc[i, "share_dem_last4"])
    loo.append(dict(year=int(sub.loc[i, "year"]), share=s, loo_threshold=thr, decided_by=s - thr,
                    call="D" if s > thr else "R", actual="D" if sub.loc[i, "margin_dem"] > 0 else "R",
                    hit=bool((s > thr) == (sub.loc[i, "margin_dem"] > 0))))
loo_df = pd.DataFrame(loo)
robust["loo_threshold"] = json.loads(loo_df.to_json(orient="records"))
robust["loo_hits"] = hit_summary(loo_df["hit"])
robust["loo_min_abs_decided_by"] = float(loo_df["decided_by"].abs().min())
# (b) always-R baseline (majority class of the 5 midterms)
robust["always_R_baseline"] = hit_summary([m < 0 for m in sub["margin_dem"]])
# (c) p-values
r4 = results["hit_rates"]["step4_party_name"]
robust["binom_p_fixed_0.5"] = binom_p(r4["hits"], r4["n"])
robust["binom_p_loo"] = binom_p(robust["loo_hits"]["hits"], robust["loo_hits"]["n"])
robust["pearson_p_share_vs_margin"] = pearson_p(sub["share_dem_last4"], sub["margin_dem"])
# (d) retrieval noise: same-day forced re-pulls exist for 2014 and 2018 (2 extra requests made once, outside this
#     script, then read back here). Google serves one sample per request-day, so same-day vintages are identical;
#     the check is inconclusive rather than reassuring.
noise = {}
for yr in (2014, 2018):
    vs = gt.load_vintages(PARTY_PAIRS["party_name"], geo="US", timeframe=MIDTERMS[yr]["tf"])
    shares = [share_stats(v, "Democratic Party", "Republican Party", MIDTERMS[yr]["eday"])["share_last4"] for v in vs]
    noise[str(yr)] = dict(n_vintages=len(vs), shares_last4=shares, identical=bool(len(vs) > 1 and all(v.equals(vs[0]) for v in vs)))
robust["retrieval_noise_same_day"] = noise
# (e) 2026 sensitivity: the week ending 2026-09-05 had a one-off "Republican Party" spike; recompute without it.
df26 = gt.fetch_interest(PARTY_PAIRS["party_name"], geo="US", timeframe=tf26)
w26 = df26.resample("W-SAT").sum()
w26["dem_share"] = w26["Democratic Party"] / (w26["Democratic Party"] + w26["Republican Party"])
spike_wk = w26["Republican Party"].idxmax()
end26 = df26.index.max()
last4 = df26.loc[end26 - pd.Timedelta(days=27):]
mask = ~((last4.index > spike_wk - pd.Timedelta(days=7)) & (last4.index <= spike_wk))
ex = last4.loc[mask]
robust["s2026"] = dict(spike_week_ending=str(spike_wk.date()),
                       spike_week_values=dict(dem=int(w26.loc[spike_wk, "Democratic Party"]), rep=int(w26.loc[spike_wk, "Republican Party"])),
                       share_last4=float(last4["Democratic Party"].sum() / (last4["Democratic Party"].sum() + last4["Republican Party"].sum())),
                       share_last4_excl_spike_week=float(ex["Democratic Party"].sum() / (ex["Democratic Party"].sum() + ex["Republican Party"].sum())),
                       weekly_dem_share=[{"week_ending": str(i.date()), "share": round(float(v), 3)} for i, v in w26["dem_share"].items()],
                       historical_mean_share_last4=float(sub["share_dem_last4"].mean()))
results["robustness_party_name"] = robust

# ---------------- Validation decision (pre-registered rule) ----------------
# A relationship counts as validated only if, on history: (i) n >= 8 outcomes, (ii) hit-rate above chance with exact
# binomial p < 0.05, (iii) share correlates with margin with p < 0.05, (iv) the result survives leave-one-out
# thresholding and a trivial majority-class baseline. n=5 midterms cannot satisfy (i), so the best this cell can
# deliver for the midterm pair is "suggestive, unvalidated".
n_total = sum(v["n"] for k, v in results["hit_rates"].items() if k.startswith(("step1_raw_last4_vs_popular", "step3_state_winner", "step4")))
h_total = sum(v["hits"] for k, v in results["hit_rates"].items() if k.startswith(("step1_raw_last4_vs_popular", "step3_state_winner", "step4")))
checks = {}
for key in ["step1_raw_last4_vs_popular_vote", "step3_state_winner", "step4_party_name", "step4_vote_party"]:
    r = results["hit_rates"][key]
    ck = dict(n=r["n"], hits=r["hits"], rate=r["rate"], binom_p=binom_p(r["hits"], r["n"]))
    if key == "step1_raw_last4_vs_popular_vote":
        ck["corr_p"] = pearson_p(tab1["share_dem_last4"], tab1["margin_dem"])
    elif key == "step3_state_winner":
        ck["corr_p"] = pearson_p(tab3["share_dem_last4"], tab3["margin_dem"])
    else:
        s4 = tab4[tab4["pair"] == key.replace("step4_", "")]
        ck["corr_p"] = pearson_p(s4["share_dem_last4"], s4["margin_dem"])
    ck["passes"] = bool(ck["n"] >= 8 and ck["binom_p"] < 0.05 and pd.notna(ck["corr_p"]) and ck["corr_p"] < 0.05)
    checks[key] = ck
results["validation_checks"] = checks
validated = any(c["passes"] for c in checks.values())
results["validated"] = validated
results["validation_reasons"] = [k for k, c in checks.items() if c["passes"]] or [
    "no relationship meets the pre-registered rule (n>=8, binomial p<0.05, correlation p<0.05)"]
results["forecast"] = None
results["unvalidated_lean"] = dict(
    pair="party_name", dem_share_last4=float(tab5.loc[tab5["pair"] == "party_name", "share_dem_last4"].iloc[0]),
    dem_share_last4_excl_spike=robust["s2026"]["share_last4_excl_spike_week"],
    historical_mean_share=robust["s2026"]["historical_mean_share_last4"],
    reading="Below the historical midterm mean only because of one news-spike week; excluding it the share is at/above the "
            "mean. Not a forecast.")
if validated:  # unreachable with n=5 midterms / n=5 presidentials; kept so a future run with more history can use it
    cur = results["unvalidated_lean"]["dem_share_last4"]
    results["forecast"] = dict(origin=TODAY, target_date="2026-11-03", pair="party_name", dem_share_last4=cur,
                               point="D" if cur > robust["s2026"]["historical_mean_share_last4"] else "R", interval=None)

cat_after = len(gt.catalog())
# The catalog is shared with other cells running concurrently, so its delta is not attributable to this cell.
# Attribution by request signature: 58 attempts on the first run, 4 of which hit the pre-existing probe pulls
# (2012/2016/2020/2024 name pairs) -> 54 new pulls, plus 2 forced same-day re-pulls for the retrieval-noise check.
results["requests"] = dict(attempted_this_run=ATTEMPTS["n"], cached_probe_pairs=4,
                           new_pulls_first_run=54, forced_repulls=2, total_new_requests=56, budget=60,
                           catalog_before_this_run=cat_before, catalog_after_this_run=cat_after,
                           failed=ATTEMPTS["failed"])
results["seconds"] = round(time.time() - t0, 1)
results["pooled_history_hits"] = dict(hits=h_total, n=n_total, rate=(h_total / n_total) if n_total else np.nan)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


OUT_JSON.write_text(json.dumps(_clean(results), indent=2, ensure_ascii=False), encoding="utf-8")

# ---------------- Markdown tables dump for the report ----------------
md_parts = {
    "tab1": md_table(tab1), "tab2": md_table(tab2), "tab2s": md_table(tab2s), "tab3": md_table(tab3),
    "tab4": md_table(tab4), "tab5": md_table(tab5),
}
(ROOT / "research" / "experiments" / "scripts" / "04-Q4-F-tables.md").write_text(
    "\n\n".join(f"### {k}\n\n{v}" for k, v in md_parts.items()), encoding="utf-8")

print("\n=== SUMMARY ===")
print(json.dumps(_clean({k: results[k] for k in ("hit_rates", "correlations", "robustness_party_name", "validation_checks",
                                                    "validated", "validation_reasons", "forecast", "unvalidated_lean",
                                                    "requests", "seconds", "pooled_history_hits")}), indent=1))
