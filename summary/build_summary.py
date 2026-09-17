"""Build summary/index.html — the web summary of the project with showcase examples.

    PYTHONIOENCODING=utf-8 python summary/build_summary.py

Reads outputs/<id>.json (pipeline results), copies the matching HTML artifacts into
summary/artifacts/, reads research/validation/results.json if present (model comparison), and
writes a single self-contained page (same palette/chrome as the artifacts).
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trends_predict.report import LIGHT, DARK  # noqa: E402

OUT = ROOT / "summary"
ART = OUT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

# curated showcase: (output id, question shown, one-line takeaway written by the author)
SHOWCASE = [
    ("us-flu-ili", "How high will US flu activity be in the coming weeks?",
     "The success case: Google searches for symptoms and treatments nowcast the CDC's influenza-like-illness rate a week before it is published, cutting error by more than half versus the best target-only model."),
    ("us-unemployment", "What will the US unemployment rate be next month?",
     "The honest null: after fixing publication-lag leakage, no search-augmented model beats the random walk at monthly frequency. The skill answers with the random walk and says so."),
    ("sp500-direction", "Will the S&P 500 be higher a month from now?",
     "Prices: any apparent 'gain' was the equity drift. The skill benchmarks against random-walk-with-drift and answers with the base rate."),
    ("us-midterms-2026-salience", "Who will win the next US election? (asked in Hebrew)",
     "Discrete outcomes: search attention picked the winner in 21 of 36 past US contests — a coin flip. The skill refuses to forecast and shows attention share as description only."),
    ("il-unemployment", "מה יהיה שיעור האבטלה בישראל בחודש הבא? (Israel unemployment, in Hebrew)",
     "Hebrew end to end: Hebrew search terms with geo=IL, ground truth from the Israel CBS series API, answer in English. The next CBS print is a genuine nowcast; the search-augmented model's gain is marginal, so the answer stays with the random walk."),
    ("us-inflation-accel", "Will US inflation accelerate next month?",
     "An unseen question: the skill derives the month-over-month CPI rate from the BLS index, benchmarks against AR (not a random walk — it is already a rate), and answers with a probability against the base rate."),
]


def esc(s):
    return html.escape(str(s))


def fmt(v, nd=2):
    if v is None:
        return "—"
    try:
        v = float(v)
    except Exception:
        return esc(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{nd}f}"


def load(id_):
    p = ROOT / "outputs" / f"{id_}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def copy_artifact(id_):
    src = ROOT / "outputs" / f"{id_}.html"
    if src.exists():
        shutil.copy(src, ART / f"{id_}.html")
        return f"artifacts/{id_}.html"
    return None


def showcase_card(id_, question, takeaway):
    r = load(id_)
    link = copy_artifact(id_)
    if not r:
        return ""
    if r.get("is_forecast") is False:  # salience report
        lead = r.get("leader_by_attention")
        share = r.get("share_last4", {}).get(lead)
        head = "No forecast — attention share is not a predictor (validated hit-rate 21/36)"
        stats = [("Attention leader (last 4 wks)", f"{esc(lead)} {share:.0%}" if share is not None else "—"),
                 ("Excl. spike week", f"{r.get('share_last4_ex_spike_week', {}).get(lead, float('nan')):.0%}"),
                 ("Historical hit-rate", "58 % (n=36)")]
        conf = "none"
    else:
        hz_key = next((k for k, v in r["horizons"].items() if v.get("forecast")), None)
        hz = r["horizons"][hz_key]
        fc = hz["forecast"]
        vd = hz["verdict"]
        units = r["spec"].get("units", "")
        log_ = isinstance(r["spec"].get("truth"), dict) and r["spec"]["truth"].get("log")
        import math
        pt, lo, hi = (math.exp(fc["point"]), math.exp(fc["lo"]), math.exp(fc["hi"])) if log_ else (fc["point"], fc["lo"], fc["hi"])
        head = f"{fmt(pt)}{'' if log_ else units} for {esc(fc['target_date'])} · 90 % interval {fmt(lo)}–{fmt(hi)}"
        stats = [("Benchmark", esc(hz["bench"])), ("Model used", esc(hz.get("chosen_model"))),
                 ("Best GT model OOS R²", f"{100 * vd.get('oos_r2', 0):+.0f} %"),
                 ("Clark–West p / DM p", f"{fmt(vd.get('cw_p'), 3)} / {fmt(vd.get('dm_p'), 3)}"),
                 ("Verdict", esc(vd.get("strength")))]
        if fc.get("base_rate_up") is not None:
            stats.append(("Base rate 'up'", f"{fc['base_rate_up']:.0%}"))
            if fc.get("prob_up") is not None:
                stats.append(("Model P(up)", f"{fc['prob_up']:.0%}"))
        if hz.get("benchmark_forecast") and hz["benchmark_forecast"].get("point") is not None and hz.get("chosen_model") != hz["bench"]:
            bp = hz["benchmark_forecast"]["point"]
            stats.append(("Benchmark alone", fmt(math.exp(bp) if log_ else bp)))
        conf = {"strong": "high", "moderate": "medium", "none": "low"}.get(vd.get("strength"), "low")
    rows = "".join(f"<div class='kv'><span>{k}</span><b>{v}</b></div>" for k, v in stats)
    link_html = f"<a class='btn' href='{link}'>Open the evidence artifact →</a>" if link else ""
    return f"""<article class="card show">
  <div class="q">“{esc(question)}”</div>
  <div class="head conf-{conf}">{head}</div>
  <p class="take">{esc(takeaway)}</p>
  <div class="kvs">{rows}</div>
  {link_html}
</article>"""


def evidence_table():
    rows = [
        ("US unemployment rate", "monthly", "none beats the random walk; +13–21 % only in the 2020-excluded view (DM n.s.); query expansion to 27–59 features is 2× worse", "01-Q1-A/B/C"),
        ("US flu ILI %", "weekly", "<b>strong</b>: OOS R² +0.54–0.63 at h=0, +0.36–0.39 at h=2 vs AR (DM ≤ 0.03); in-season only; raw log GT beats Djorno preprocessing", "02-Q2-A/B/C"),
        ("S&P 500 direction", "monthly/weekly", "none vs random walk with drift; hit-rate 64.5 % vs always-up 66.4 %; attention is coincident, not leading", "03-Q3-A/B"),
        ("US election winner", "—", "attention share hit-rate 21/36 pooled, 2/5 presidential, modifiers negatively correlated with margins; refused", "04-Q4-F"),
        ("Israel unemployment (Hebrew)", "monthly", "none vs naive (marginal +3–4 % with detrended GT, DM n.s.); Hebrew queries, Hebrew related-query expansion (56 terms) and CBS truth all work", "05-Q5-A/B"),
        ("US CPI inflation (validation)", "monthly", "none vs AR; P(accelerate) 31–37 % vs 51 % base rate", "V6"),
        ("New York flu (validation)", "weekly", "moderate at h≤2 in-season only; state ILINet dead since 2025 → HHS Region 2", "V7"),
    ]
    tr = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td><code>{d}</code></td></tr>" for a, b, c, d in rows)
    return f"<table><thead><tr><th>Question</th><th>Frequency</th><th>Does Google Trends help?</th><th>Cells</th></tr></thead><tbody>{tr}</tbody></table>"


def validation_section():
    p = ROOT / "research" / "validation" / "results.json"
    if not p.exists():
        return "<p class='muted'>Validation results pending.</p>"
    v = json.loads(p.read_text(encoding="utf-8"))
    head = "<tr><th>Question</th><th>Model</th><th>Target/truth</th><th>Ran pipeline</th><th>Answer elements</th><th>No over-claim</th><th>Efficiency</th><th>Total /10</th><th>Note</th></tr>"
    rows = ""
    for r in v.get("rows", []):
        rows += "<tr>" + "".join(f"<td>{esc(r.get(k, ''))}</td>" for k in ("question", "model", "s1", "s2", "s3", "s4", "s5", "total", "note")) + "</tr>"
    summ = v.get("summary", "")
    return f"<p>{esc(summ)}</p><div class='scroll'><table><thead>{head}</thead><tbody>{rows}</tbody></table></div>"


def build():
    cards = "".join(showcase_card(*s) for s in SHOWCASE)
    css = f"""
:root{{--surface:{LIGHT['surface']};--page:{LIGHT['page']};--ink:{LIGHT['ink']};--ink2:{LIGHT['ink2']};--muted:{LIGHT['muted']};--grid:{LIGHT['grid']};--border:{LIGHT['border']};--s1:{LIGHT['s'][0]};--s2:{LIGHT['s'][1]};--s3:{LIGHT['s'][2]};--good:{LIGHT['good']};--serious:{LIGHT['serious']};--critical:{LIGHT['critical']};color-scheme:light}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--surface:{DARK['surface']};--page:{DARK['page']};--ink:{DARK['ink']};--ink2:{DARK['ink2']};--muted:{DARK['muted']};--grid:{DARK['grid']};--border:{DARK['border']};--s1:{DARK['s'][0]};--s2:{DARK['s'][1]};--s3:{DARK['s'][2]};color-scheme:dark}}}}
:root[data-theme="dark"]{{--surface:{DARK['surface']};--page:{DARK['page']};--ink:{DARK['ink']};--ink2:{DARK['ink2']};--muted:{DARK['muted']};--grid:{DARK['grid']};--border:{DARK['border']};--s1:{DARK['s'][0]};--s2:{DARK['s'][1]};--s3:{DARK['s'][2]};color-scheme:dark}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--page);color:var(--ink);font:15px/1.6 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:36px 20px 80px}}h1{{font:600 40px/1.1 "Newsreader",Georgia,"Times New Roman",serif;letter-spacing:-.01em;margin:0 0 10px;text-wrap:balance}}h2{{font:600 24px/1.2 "Newsreader",Georgia,serif;margin:48px 0 14px;text-wrap:balance}}
.lede{{font-size:17px;color:var(--ink2);max-width:72ch;margin:0 0 26px}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 22px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
.show .q{{font-size:17px;font-weight:600;margin-bottom:8px}}.show .head{{font-size:15px;font-weight:600;margin-bottom:8px}}.show .take{{color:var(--ink2);margin:0 0 12px}}
.kvs{{display:grid;grid-template-columns:1fr 1fr;gap:6px 14px;font-size:13px;margin-bottom:14px}}.kv span{{color:var(--muted);display:block;font-size:11px;text-transform:uppercase;letter-spacing:.05em}}.kv b{{font-weight:600}}
.btn{{display:inline-block;font-size:13px;color:var(--s1);text-decoration:none;border:1px solid var(--border);border-radius:8px;padding:6px 10px}}.btn:hover{{background:var(--grid)}}
.conf-high{{color:var(--good)}}.conf-medium{{color:#a86f00}}.conf-low{{color:var(--serious)}}.conf-none{{color:var(--critical)}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--grid);vertical-align:top}}th{{color:var(--ink2);font-weight:600}}
.steps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;counter-reset:s}}.step{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px;font-size:13.5px}}.step b{{display:block;margin-bottom:4px}}.step::before{{counter-increment:s;content:counter(s);display:inline-block;width:22px;height:22px;border-radius:11px;background:var(--s1);color:#fff;font-size:12px;text-align:center;line-height:22px;margin-bottom:8px}}
ul{{margin:6px 0;padding-left:22px}}li{{margin:5px 0}}code{{font-size:12.5px;background:var(--grid);padding:1px 5px;border-radius:4px}}.muted{{color:var(--muted)}}.scroll{{overflow:auto}}
footer{{margin-top:50px;font-size:12.5px;color:var(--muted)}}
"""
    fonts = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
    head = f'<title>trends-forecast</title>{fonts}<style>{css}</style>'
    body = f"""<main>
<h1>trends-forecast</h1>
<p class="lede">A Claude Code skill that turns a plain-language prediction question — in English or Hebrew — into a <b>validated</b> Google Trends forecast, or into a reasoned refusal. It applies the methods of the Google-Trends forecasting literature (Choi &amp; Varian, ARGO, MIDAS, Rivera, Borup &amp; Montes Schütte, Djorno et al.) with rolling-origin backtests against the best target-only benchmark, and delivers a written answer plus a self-contained evidence artifact.</p>

<h2>Showcase</h2>
<div class="grid">{cards}</div>

<h2>What the experiments found</h2>
<div class="card">{evidence_table()}
<p class="muted" style="margin-top:10px">11 experiment cells, each run by a context-isolated agent with one question × one approach (hand-picked ARX · ARGO-style expansion + LASSO · Djorno preprocessing · share proxy). Full write-ups in <code>research/experiments/</code>; synthesis in <code>research/conclusions.md</code>.</p></div>

<h2>How the skill works</h2>
<div class="steps">
<div class="step"><b>Parse</b>target quantity, geography, frequency, horizon, answer type, language</div>
<div class="step"><b>Feasibility</b>is there a keyless ground-truth series? If not → refuse with options</div>
<div class="step"><b>Queries</b>8–15 behavioural search terms / topic ids; probe for privacy zeros</div>
<div class="step"><b>Backtest</b>rolling origin, publication-lag aware, vs naive · drift · seasonal · AR</div>
<div class="step"><b>Verdict</b>OOS R² &gt; 0 <i>and</i> Clark–West / Diebold–Mariano; full &amp; shock-excluded views</div>
<div class="step"><b>Forecast</b>GT model only if it helps, else the benchmark; conformal interval</div>
<div class="step"><b>Artifact</b>self-contained HTML: history, backtest, metrics, queries, provenance</div>
<div class="step"><b>Learn</b>append the validated query set to the playbook</div>
</div>

<h2>Rules that changed our answers</h2>
<div class="card"><ul>
<li><b>The benchmark decides the story.</b> Against AR-in-levels every Google model "won" by 50–65 %; against the random walk they lost. Against a zero-return naive the S&amp;P model "won" by having an intercept; against random-walk-with-drift it did not.</li>
<li><b>Publication lag is where leakage hides.</b> One row per origin was leaking the very value being nowcast; fixing it erased the "Google caught April 2020" result for unemployment.</li>
<li><b>Clark–West alone is not evidence</b> — it was significant for models with OOS R² of −4. The verdict requires a positive out-of-sample gain <i>and</i> a test.</li>
<li><b>Preprocessing is domain-specific.</b> Detrending helped unemployment; every Djorno step hurt flu, where the spikes <i>are</i> the signal. The backtest chooses.</li>
<li><b>Breadth did not help.</b> 25–60 expanded terms lost to a hand-picked dozen in every cell; sliding windows with more features than rows collapse to the random walk.</li>
<li><b>Attention ≠ support.</b> Search share picked the winner in 21 of 36 US contests. The skill refuses "who will win" and shows salience only.</li>
</ul></div>

<h2>Skill validation — Fable 5.1 vs Opus 5</h2>
<div class="card">{validation_section()}</div>

<h2>Data</h2>
<div class="card"><ul>
<li>Google Trends via <code>trendspy</code> (free, no key); every pull cached as a dated vintage under <code>data/trends/</code> with a browsable <code>catalog.jsonl</code>.</li>
<li>Ground truth, keyless: BLS (labour, CPI), CDC FluView via Delphi Epidata, Yahoo Finance, Wikipedia pageviews, Israel CBS series API, user CSV — cached under <code>data/truth/</code>.</li>
<li>Library: <code>src/trends_predict/</code> (gt · truth · preprocess · models · evaluate · report · pipeline), 11 synthetic tests.</li>
</ul></div>
<footer>trends-predict · 2026-09-17 · Google Trends values are relative search interest, not counts.</footer>
</main>"""
    (OUT / "index.html").write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{head}</head><body>{body}</body></html>', encoding="utf-8")
    (OUT / "artifact.html").write_text(head + body, encoding="utf-8")  # body-only version for the Artifact tool
    print("wrote", OUT / "index.html")


if __name__ == "__main__":
    build()
