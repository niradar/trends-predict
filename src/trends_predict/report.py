"""Self-contained HTML artifact for a forecast: answer, evidence, backtest, queries, data.

No external assets (works offline, safe to share). Charts are drawn client-side in SVG
from embedded JSON. Design follows the dataviz skill: single y-axis per chart, fixed
categorical slot order, hairline grid, legend for >=2 series, selective direct labels,
crosshair tooltip, table view for every chart, dark mode selected (not auto-inverted).
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# reference palette (dataviz skill references/palette.md)
LIGHT = {"surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
         "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "rgba(11,11,11,0.10)",
         "s": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
         "good": "#0ca30c", "warn": "#fab219", "serious": "#ec835a", "critical": "#d03b3b", "band": "rgba(42,120,214,0.14)"}
DARK = {"surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "border": "rgba(255,255,255,0.10)",
        "s": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "good": "#0ca30c", "warn": "#fab219", "serious": "#ec835a", "critical": "#d03b3b", "band": "rgba(57,135,229,0.18)"}


def _ser(s: pd.Series | None) -> list[list[Any]]:
    if s is None:
        return []
    out = []
    for k, v in s.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ts = k.to_timestamp() if isinstance(k, pd.Period) else pd.Timestamp(k)
        out.append([ts.strftime("%Y-%m-%d"), float(v)])
    return out


def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    a = abs(float(v))
    if a >= 1000:
        return f"{float(v):,.0f}"
    if a >= 100:
        return f"{float(v):.1f}"
    return f"{float(v):.{nd}f}"


def render_report(path: str | Path, *, title: str, question: str, answer_headline: str, answer_text: str,
                  confidence: str, forecast: dict | None, history: pd.Series, history_label: str,
                  backtest: dict[str, pd.Series] | None, backtest_truth: pd.Series | None,
                  metrics: pd.DataFrame | None, gt_series: pd.DataFrame | None, queries_info: list[dict] | None,
                  method_notes: list[str], caveats: list[str], provenance: dict, extra_sections: list[dict] | None = None,
                  lang_dir: str = "ltr", chosen_model: str | None = None) -> Path:
    """Write the HTML file and return its path.

    forecast: {"date": "2026-10-01", "point": 4.2, "lo": 3.9, "hi": 4.5, "alpha": 0.1, "label": "..."}
    backtest: {"AR benchmark": Series(pred by date), "LASSO + Google Trends": Series(...)}  (<=4)
    confidence: one of "high" | "medium" | "low" | "none"
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hist_pts = _ser(history)
    fc = forecast or {}
    bt_truth = _ser(backtest_truth)
    _truth_dates = {p[0] for p in bt_truth}
    # every backtest series is restricted to the common backtest dates so lines share one span
    bt_series = {k: [p for p in _ser(v) if not _truth_dates or p[0] in _truth_dates] for k, v in (backtest or {}).items()}
    gt_cols = list(gt_series.columns)[:8] if gt_series is not None else []
    gt_pts = {c: _ser(gt_series[c]) for c in gt_cols}
    metrics_rows = []
    if metrics is not None and len(metrics):
        m = metrics.reset_index()
        for _, r in m.iterrows():
            metrics_rows.append({k: (None if (isinstance(v, float) and np.isnan(v)) else (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else str(v))) for k, v in r.items()})
    data = {"history": hist_pts, "historyLabel": history_label, "forecast": fc, "backtest": bt_series,
            "backtestTruth": bt_truth, "gt": gt_pts, "metrics": metrics_rows}
    conf_color = {"high": "good", "medium": "warn", "low": "serious", "none": "critical"}.get(confidence, "muted")
    conf_icon = {"high": "●", "medium": "◐", "low": "○", "none": "✕"}.get(confidence, "?")

    tiles = []
    if fc:
        tiles.append(("Forecast", f"{_fmt(fc.get('point'))}", fc.get("label", "")))
        if fc.get("lo") is not None:
            tiles.append((f"{int(round((1 - fc.get('alpha', 0.1)) * 100))}% interval", f"{_fmt(fc.get('lo'))} – {_fmt(fc.get('hi'))}", "rolling conformal"))
    if metrics is not None and "OOS_R2_vs_bench" in metrics.columns:
        best = metrics["OOS_R2_vs_bench"].drop(index=[i for i in metrics.index if "bench" in str(i).lower() or str(i) in ("ar", "naive", "drift", "seasonal_naive")], errors="ignore")
        if len(best):
            b = best.idxmax()
            used = (chosen_model is not None and str(chosen_model) == str(b))
            tiles.append(("Google Trends gain" if used else "Best Google Trends model",
                          f"{100 * float(best.max()):+.0f}%",
                          f"OOS R² vs target-only benchmark · {html.escape(str(b))}" + ("" if used else " · <b>not used</b>: benchmark answers")))
    tiles.append(("Confidence", f"{conf_icon} {confidence}", "evidence-based class", f"conf-{conf_color}"))

    def tile_html(t):
        lab, val, sub = t[0], t[1], t[2]
        cls = f"tile-value {t[3]}" if len(t) > 3 else "tile-value"
        return f'<div class="tile"><div class="tile-label">{html.escape(lab)}</div><div class="{cls}">{html.escape(val)}</div><div class="tile-sub">{sub}</div></div>'

    def table_html(df: pd.DataFrame | None, caption: str) -> str:
        if df is None or not len(df):
            return ""
        d = df.copy()
        if d.index.name or not isinstance(d.index, pd.RangeIndex):
            d = d.reset_index()
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
        body = ""
        for _, r in d.iterrows():
            cells = []
            for c in d.columns:
                v = r[c]
                cells.append(f"<td>{html.escape(_fmt(v, 3)) if isinstance(v, (float, np.floating)) else html.escape(str(v))}</td>")
            body += "<tr>" + "".join(cells) + "</tr>"
        return f'<details class="tbl"><summary>{html.escape(caption)}</summary><div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></details>'

    q_rows = ""
    if queries_info:
        for q in queries_info:
            q_rows += "<tr>" + "".join(f"<td>{html.escape(str(q.get(k, '')))}</td>" for k in q.keys()) + "</tr>"
        q_head = "".join(f"<th>{html.escape(k)}</th>" for k in queries_info[0].keys())
        q_table = f'<div class="scroll"><table><thead><tr>{q_head}</tr></thead><tbody>{q_rows}</tbody></table></div>'
    else:
        q_table = "<p class='muted'>No query table.</p>"

    def li(items):
        return "".join(f"<li>{html.escape(x) if not x.startswith('<') else x}</li>" for x in items)

    prov_rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in provenance.items())
    extra_html = ""
    for sec in (extra_sections or []):
        extra_html += f"<section><h2>{html.escape(sec.get('title', ''))}</h2>{sec.get('html', '')}</section>"

    hist_table = table_html(pd.DataFrame({"date": [p[0] for p in hist_pts], history_label: [p[1] for p in hist_pts]}), "Table view — history")
    bt_df = None
    if bt_truth:
        bt_df = pd.DataFrame({"date": [p[0] for p in bt_truth], "actual": [p[1] for p in bt_truth]})
        for k, pts in bt_series.items():
            mp = dict(pts)
            bt_df[k] = [mp.get(d) for d in bt_df["date"]]
    bt_table = table_html(bt_df, "Table view — backtest")
    gt_df = None
    if gt_pts:
        dates = sorted({p[0] for pts in gt_pts.values() for p in pts})
        gt_df = pd.DataFrame({"date": dates})
        for c, pts in gt_pts.items():
            mp = dict(pts)
            gt_df[c] = [mp.get(d) for d in dates]
    gt_table = table_html(gt_df, "Table view — Google Trends series")

    css = f"""
:root{{--surface:{LIGHT['surface']};--page:{LIGHT['page']};--ink:{LIGHT['ink']};--ink2:{LIGHT['ink2']};--muted:{LIGHT['muted']};--grid:{LIGHT['grid']};--axis:{LIGHT['axis']};--border:{LIGHT['border']};--band:{LIGHT['band']};
{''.join(f'--s{i+1}:{c};' for i, c in enumerate(LIGHT['s']))}--good:{LIGHT['good']};--warn:{LIGHT['warn']};--serious:{LIGHT['serious']};--critical:{LIGHT['critical']};color-scheme:light}}
@media (prefers-color-scheme: dark){{:root:where(:not([data-theme="light"])){{--surface:{DARK['surface']};--page:{DARK['page']};--ink:{DARK['ink']};--ink2:{DARK['ink2']};--muted:{DARK['muted']};--grid:{DARK['grid']};--axis:{DARK['axis']};--border:{DARK['border']};--band:{DARK['band']};{''.join(f'--s{i+1}:{c};' for i, c in enumerate(DARK['s']))}color-scheme:dark}}}}
:root[data-theme="dark"]{{--surface:{DARK['surface']};--page:{DARK['page']};--ink:{DARK['ink']};--ink2:{DARK['ink2']};--muted:{DARK['muted']};--grid:{DARK['grid']};--axis:{DARK['axis']};--border:{DARK['border']};--band:{DARK['band']};{''.join(f'--s{i+1}:{c};' for i, c in enumerate(DARK['s']))}color-scheme:dark}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:28px 20px 60px}}
header{{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:8px}}
h1{{font-size:26px;margin:0 0 6px;font-weight:650;letter-spacing:-.01em}}h2{{font-size:17px;margin:34px 0 10px;font-weight:650}}
.q{{color:var(--ink2);font-size:16px;margin:0 0 18px}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 20px}}
.answer{{font-size:21px;font-weight:600;margin:0 0 8px;line-height:1.35}}
.answer-text{{color:var(--ink2);margin:0}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0 6px}}
.tile{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}}
.tile-label{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}}
.tile-value{{font-size:30px;font-weight:600;margin:2px 0}}.tile-sub{{font-size:12px;color:var(--ink2)}}
.conf-good{{color:var(--good)}}.conf-warn{{color:#a86f00}}.conf-serious{{color:var(--serious)}}.conf-critical{{color:var(--critical)}}
.chart{{position:relative;width:100%}}.chart svg{{display:block;width:100%;height:auto;overflow:visible}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--ink2);margin:6px 0 2px}}.legend span::before{{content:"";display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle;background:var(--c)}}
.tip{{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:12px;color:var(--ink);box-shadow:0 4px 14px rgba(0,0,0,.08);display:none;white-space:nowrap;z-index:2}}
.tip b{{font-weight:600}}.tip .row{{display:flex;gap:10px;justify-content:space-between}}
.tbl{{margin-top:10px;font-size:13px}}.tbl summary{{cursor:pointer;color:var(--ink2)}}.scroll{{overflow:auto;max-height:340px;margin-top:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums;white-space:nowrap}}th{{color:var(--ink2);font-weight:600;position:sticky;top:0;background:var(--surface)}}
ul{{margin:6px 0;padding-left:22px}}li{{margin:4px 0}}.muted{{color:var(--muted)}}
.toggle{{font-size:12px;color:var(--ink2);background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:5px 9px;cursor:pointer}}
.small-multiples{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}.sm h3{{font-size:13px;margin:0 0 4px;font-weight:600;color:var(--ink2)}}
footer{{margin-top:40px;font-size:12px;color:var(--muted)}}
"""
    js = r"""
const D = window.__DATA__;
const css = getComputedStyle(document.documentElement);
const v = n => css.getPropertyValue(n).trim();
const S = () => [1,2,3,4,5,6,7,8].map(i => v('--s'+i));
const parse = d => new Date(d + 'T00:00:00Z');
function fmt(x){ if(x==null||isNaN(x)) return '—'; const a=Math.abs(x); return a>=1000? x.toLocaleString(undefined,{maximumFractionDigits:0}) : a>=100? x.toFixed(1) : x.toFixed(2); }
function lineChart(el, series, opts){
  // series: [{name, pts:[[date,val]], color, dash?, marker?}], opts: {band:{pts:[[date,lo,hi]]}, height, forecastPoint}
  opts = opts||{}; const W=960, H=opts.height||300, m={t:14,r:70,b:34,l:52};
  const all=[]; series.forEach(s=>s.pts.forEach(p=>all.push(p)));
  if(opts.band) opts.band.pts.forEach(p=>{all.push([p[0],p[1]]);all.push([p[0],p[2]]);});
  if(opts.point) all.push([opts.point.date, opts.point.point]);
  if(!all.length){ el.innerHTML='<p class="muted">no data</p>'; return; }
  const xs=all.map(p=>parse(p[0]).getTime()), ys=all.map(p=>p[1]);
  let x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys), y1=Math.max(...ys);
  if(opts.zero && y0>0) y0=0; const pad=(y1-y0||1)*0.08; y0-=pad; y1+=pad; if(x1===x0) x1=x0+864e5;
  const X=t=>m.l+(t-x0)/(x1-x0)*(W-m.l-m.r), Y=y=>m.t+(y1-y)/(y1-y0)*(H-m.t-m.b);
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${opts.label||''}">`;
  // grid
  const ny=5; for(let i=0;i<=ny;i++){const yy=y0+(y1-y0)*i/ny; svg+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(yy)}" y2="${Y(yy)}" stroke="${v('--grid')}" stroke-width="1"/><text x="${m.l-8}" y="${Y(yy)+4}" text-anchor="end" font-size="11" fill="${v('--muted')}" style="font-variant-numeric:tabular-nums">${fmt(yy)}</text>`;}
  // x ticks at calendar boundaries (years if span > 3y, else months), thinned to <= 8
  const span=x1-x0; const ticks=[]; const d0=new Date(x0);
  if(span>3*365*864e5){ for(let yr=d0.getUTCFullYear()+1; yr<=new Date(x1).getUTCFullYear(); yr++) ticks.push([Date.UTC(yr,0,1), String(yr)]); }
  else { let d=new Date(Date.UTC(d0.getUTCFullYear(), d0.getUTCMonth()+1, 1)); while(d.getTime()<=x1){ ticks.push([d.getTime(), d.toISOString().slice(0,7)]); d=new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth()+1, 1)); } }
  const every=Math.max(1, Math.ceil(ticks.length/8)); ticks.filter((_,i)=>i%every===0).forEach(([t,lab])=>{ svg+=`<text x="${X(t)}" y="${H-10}" text-anchor="middle" font-size="11" fill="${v('--muted')}">${lab}</text>`; });
  svg+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(y0)}" y2="${Y(y0)}" stroke="${v('--axis')}" stroke-width="1"/>`;
  if(opts.band){ const b=opts.band.pts; let up=b.map(p=>`${X(parse(p[0]).getTime())},${Y(p[2])}`).join(' '); let lo=b.slice().reverse().map(p=>`${X(parse(p[0]).getTime())},${Y(p[1])}`).join(' '); svg+=`<polygon points="${up} ${lo}" fill="${v('--band')}" stroke="none"/>`; }
  const placed=[]; // end-label y positions already used (avoid collisions)
  series.forEach((s,i)=>{ if(!s.pts.length) return; const pts=s.pts.map(p=>`${X(parse(p[0]).getTime())},${Y(p[1])}`).join(' '); svg+=`<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" ${s.dash?'stroke-dasharray="5 4"':''}/>`;
    if(s.pts.length===1||s.marker){ s.pts.forEach(p=>{svg+=`<circle cx="${X(parse(p[0]).getTime())}" cy="${Y(p[1])}" r="4.5" fill="${s.color}" stroke="${v('--surface')}" stroke-width="2"/>`;}); }
    const last=s.pts[s.pts.length-1]; const ly=Y(last[1]);
    if(opts.endLabels!==false && !opts.point && !s.dash && series.length<=4 && !placed.some(p=>Math.abs(p-ly)<13)){ placed.push(ly); svg+=`<text x="${X(parse(last[0]).getTime())+8}" y="${ly+4}" font-size="11" fill="${v('--ink2')}">${fmt(last[1])}</text>`; } });
  if(opts.point){ const p=opts.point; const cx=X(parse(p.date).getTime()); if(p.lo!=null){ svg+=`<line x1="${cx}" x2="${cx}" y1="${Y(p.lo)}" y2="${Y(p.hi)}" stroke="${v('--s1')}" stroke-width="2"/><line x1="${cx-6}" x2="${cx+6}" y1="${Y(p.lo)}" y2="${Y(p.lo)}" stroke="${v('--s1')}" stroke-width="2"/><line x1="${cx-6}" x2="${cx+6}" y1="${Y(p.hi)}" y2="${Y(p.hi)}" stroke="${v('--s1')}" stroke-width="2"/>`; }
    svg+=`<circle cx="${cx}" cy="${Y(p.point)}" r="6" fill="${v('--s1')}" stroke="${v('--surface')}" stroke-width="2"/><text x="${cx+10}" y="${Y(p.point)+4}" font-size="12" font-weight="600" fill="${v('--ink')}">${fmt(p.point)}</text>`; }
  svg+=`<line class="xh" x1="0" x2="0" y1="${m.t}" y2="${H-m.b}" stroke="${v('--axis')}" stroke-width="1" style="display:none"/>`;
  svg+='</svg>'; el.innerHTML=svg+'<div class="tip"></div>';
  // legend
  if(series.length>=2){ const lg=document.createElement('div'); lg.className='legend'; series.forEach(s=>{const sp=document.createElement('span'); sp.style.setProperty('--c',s.color); sp.textContent=s.name; lg.appendChild(sp);}); el.appendChild(lg); }
  // hover
  const svgEl=el.querySelector('svg'), tip=el.querySelector('.tip'), xh=el.querySelector('.xh');
  const dates=[...new Set(series.flatMap(s=>s.pts.map(p=>p[0])))].sort();
  const lookup=series.map(s=>Object.fromEntries(s.pts));
  svgEl.addEventListener('mousemove',e=>{ const r=svgEl.getBoundingClientRect(); const px=(e.clientX-r.left)/r.width*W; if(px<m.l||px>W-m.r){tip.style.display='none';xh.style.display='none';return;}
    const t=x0+(px-m.l)/(W-m.l-m.r)*(x1-x0); let best=null,bd=Infinity; dates.forEach(d=>{const dd=Math.abs(parse(d).getTime()-t); if(dd<bd){bd=dd;best=d;}}); if(!best) return;
    xh.setAttribute('x1',X(parse(best).getTime())); xh.setAttribute('x2',X(parse(best).getTime())); xh.style.display='';
    let h=`<b>${best}</b>`; series.forEach((s,i)=>{const val=lookup[i][best]; if(val!=null) h+=`<div class="row"><span style="color:${s.color}">■</span><span>${s.name}</span><b>${fmt(val)}</b></div>`;}); tip.innerHTML=h; tip.style.display='block';
    const left=(e.clientX-r.left)+14, top=(e.clientY-r.top)-10; tip.style.left=Math.min(left, r.width-tip.offsetWidth-8)+'px'; tip.style.top=top+'px'; });
  svgEl.addEventListener('mouseleave',()=>{tip.style.display='none';xh.style.display='none';});
}
const C=S();
// 1 history + forecast
(function(){ const el=document.getElementById('c-hist'); const hist=D.history; const fc=D.forecast||{}; const tail=hist.slice(-Math.min(hist.length, 96));
  const ser=[{name:D.historyLabel, pts:tail, color:C[0]}]; const opts={label:'history and forecast'};
  if(fc.point!=null){ opts.point=fc; if(tail.length){ ser.push({name:'forecast path', pts:[tail[tail.length-1],[fc.date,fc.point]], color:C[0], dash:true}); } }
  lineChart(el, ser, opts); })();
// 2 backtest
(function(){ const el=document.getElementById('c-bt'); if(!D.backtestTruth.length){ el.innerHTML='<p class="muted">No backtest available.</p>'; return; }
  const ser=[{name:'actual', pts:D.backtestTruth, color:C[0]}]; let i=1; for(const k in D.backtest){ ser.push({name:k, pts:D.backtest[k], color:C[i%8], dash:false}); i++; } lineChart(el, ser, {label:'rolling-origin backtest'}); })();
// 3 GT small multiples
(function(){ const wrap=document.getElementById('c-gt'); const keys=Object.keys(D.gt); if(!keys.length){ wrap.innerHTML='<p class="muted">No Google Trends series.</p>'; return; }
  keys.forEach((k,i)=>{ const d=document.createElement('div'); d.className='sm card'; d.innerHTML=`<h3>${k}</h3><div class="chart"></div>`; wrap.appendChild(d); lineChart(d.querySelector('.chart'), [{name:k, pts:D.gt[k], color:C[0]}], {height:150, zero:true, endLabels:false}); }); })();
// theme toggle
document.getElementById('theme').addEventListener('click',()=>{ const r=document.documentElement; const cur=r.getAttribute('data-theme'); const next = cur==='dark'?'light': cur==='light'?'dark': (matchMedia('(prefers-color-scheme: dark)').matches?'light':'dark'); r.setAttribute('data-theme',next); location.hash='t'; setTimeout(()=>location.reload(),0); });
"""
    doc = f"""<!doctype html><html lang="en" dir="{lang_dir}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{css}</style></head>
<body><main>
<header><div><h1>{html.escape(title)}</h1><p class="q">{html.escape(question)}</p></div><button class="toggle" id="theme">Toggle theme</button></header>
<div class="card"><p class="answer">{html.escape(answer_headline)}</p><p class="answer-text">{html.escape(answer_text)}</p></div>
<div class="tiles">{''.join(tile_html(t) for t in tiles)}</div>
<section><h2>History and forecast</h2><div class="card"><div class="chart" id="c-hist"></div>{hist_table}</div></section>
<section><h2>Backtest — does Google Trends beat the target-only benchmark?</h2><div class="card"><div class="chart" id="c-bt"></div>{table_html(metrics, 'Metrics by model (rolling origin, out of sample)')}{bt_table}</div></section>
<section><h2>Google Trends queries used</h2><div class="card">{q_table}</div><div class="small-multiples" style="margin-top:12px" id="c-gt"></div>{gt_table}</section>
{extra_html}
<section><h2>Method</h2><div class="card"><ul>{li(method_notes)}</ul></div></section>
<section><h2>Caveats</h2><div class="card"><ul>{li(caveats)}</ul></div></section>
<section><h2>Data provenance</h2><div class="card"><table>{prov_rows}</table></div></section>
<footer>Generated by the trends-forecast skill · Google Trends values are relative search interest (0–100, normalised per request), not counts. Self-contained file; no external requests.</footer>
</main>
<script>window.__DATA__ = {json.dumps(data, ensure_ascii=False)};</script>
<script>{js}</script></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path
