"""End-to-end deterministic pipeline: a JSON *spec* in, results + HTML artifact out.

The LLM (skill) does the thinking — target, truth source, queries, geo, language, horizon —
and writes a spec. This module does the numbers the same way every time:

    spec -> truth series -> GT pulls (cached) -> align -> benchmark set -> GT model grid
         -> comparison tables (full + shock-excluded) -> verdict -> forward forecast + interval
         -> results.json + report.html

Spec schema (all keys optional unless marked *):
{
  "id": "us-unemployment",                      * slug used for output files
  "question": "...",                            * original question (any language)
  "title": "US unemployment rate — next month",
  "lang_dir": "ltr",                            # "rtl" for Hebrew UI text
  "truth": {"kind": "bls", "id": "UNRATE"} |    * see truth_from_spec(); any kind accepts
           {"kind": "bls", "id": "CPIAUCSL", "derive": "pct_change"}   #  "derive": pct_change|logdiff|diff|log
           {"kind": "fluview", "region": "nat", "start_epiweek": 201040} |
           {"kind": "yahoo", "ticker": "^GSPC", "field": "Close", "log": true} |
           {"kind": "wikipedia", "article": "Influenza", "project": "en.wikipedia"} |
           {"kind": "cbs", "id": "UNEMP_IL_SA"}  (Israel CBS; or raw "id"/"new_id" series ids) |
           {"kind": "csv", "path": "data/truth/x.csv", "date_col": 0, "value_col": 1} |
           {"kind": "cached", "name": "cbs_unemployment_rate_il"},
  "freq": "M" | "W" | "D",                      * target frequency
  "period": 12 | 52 | 7,                        # seasonal period (defaults from freq)
  "geo": "US", "gprop": "", "cat": 0,
  "queries": ["...", "..."],                    * GT keywords / topic ids (individual pulls)
  "timeframe": null,                            # override GT timeframe (default from freq)
  "horizons": [{"h": 0, "y_known": false}, {"h": 1, "y_known": true}],
  "transform": "diff" | null,                   # default: 'diff' for M/D persistent series
  "p": 3, "xlags": 1, "fourier_k": 1, "min_train": 60, "max_origins": 120,
  "models": ["ridge+djorno", "lasso+djorno", "lasso+raw", "ols+raw", "enet+raw", "arlr+raw", "ridge+raw"],
  "windows": [null, 36],                        # expanding and/or sliding lengths
  "exclude": ["2020-03-01", "2020-12-31"],       # shock window for the robustness table
  "alpha": 0.1,                                 # interval level
  "clip_min": 0,                                # floor for predictions/intervals (rates, counts)
  "history_years": 9,                           # weekly targets: stitch >5y of weekly GT windows
  "answer_type": "quantity" | "direction",      # ("winner"/"yes-no" questions use salience(), below)
  "units": "%", "label": "Unemployment rate (%)",
  "out_dir": "outputs"
}
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import REPO_ROOT, gt, truth, preprocess as pp, evaluate as ev
from .report import render_report

# both preprocessing families, with and without clustering: the backtest picks (01-Q1-C: detrending
# helps unemployment, clustering hurts; flu: raw beats everything)
DEFAULT_MODELS = ["ridge+raw", "ols+raw", "lasso+raw", "ridge+djorno", "ridge+djorno_nocluster", "lasso+djorno"]


# --------------------------------------------------------------------------------------
# spec helpers
# --------------------------------------------------------------------------------------

def truth_from_spec(t: dict) -> tuple[pd.Series, str]:
    kind = t["kind"].lower()
    if kind == "bls":
        s = truth.bls_series(t["id"], start_year=int(t.get("start_year", 2004)))
        desc = f"BLS {truth.BLS_SERIES.get(t['id'], (t['id'],))[0]} via api.bls.gov v1"
        s, desc = _derive(s, desc, t)
        return s, desc
    if kind == "fluview":
        s = truth.fluview_ili(t.get("region", "nat"), int(t.get("start_epiweek", 201040)), field=t.get("field", "wili"))
        return s, f"CDC FluView {t.get('field', 'wili')} ({t.get('region', 'nat')}) via Delphi Epidata"
    if kind == "fluview_clinical":
        s = truth.fluview_clinical(t.get("region", "nat"), int(t.get("start_epiweek", 201540)), field=t.get("field", "percent_positive"))
        return s, "CDC FluView clinical via Delphi Epidata"
    if kind == "yahoo":
        s = truth.yahoo(t["ticker"], field=t.get("field", "Close"), freq=t.get("freq"))
        if t.get("log"):
            s = np.log(s)
        return s, f"Yahoo Finance {t['ticker']} {t.get('field', 'Close')}"
    if kind == "wikipedia":
        s = truth.wikipedia_pageviews(t["article"], project=t.get("project", "en.wikipedia"), freq=t.get("freq", "W"))
        return s, f"Wikimedia pageviews {t.get('project', 'en.wikipedia')}/{t['article']}"
    if kind == "csv":
        s = truth.csv_series(t["path"], t.get("date_col", 0), t.get("value_col", 1), name=t.get("name"), date_format=t.get("date_format"))
        return _derive(s, f"CSV {t['path']}", t)
    if kind == "cbs":
        s = truth.cbs_series(t.get("id", "UNEMP_IL_SA"), t.get("new_id"), name=t.get("name"))
        return _derive(s, f"Israel CBS series API {t.get('id', 'UNEMP_IL_SA')}", t)
    if kind == "cached":
        s = truth._load(t["name"], None)
        if s is None:
            raise FileNotFoundError(f"no cached truth series named {t['name']!r} in data/truth/")
        return _derive(s, f"cached truth series {t['name']}", t)
    raise ValueError(f"unknown truth kind {kind!r}")


def _derive(s: pd.Series, desc: str, t: dict) -> tuple[pd.Series, str]:
    """Optional transforms of a truth series: "derive": "pct_change" | "logdiff" | "diff" | "log"
    (× "scale", default 100 for pct_change/logdiff). Gaps are interpolated *before* differencing
    so a missing month (BLS shutdown) does not become a two-period change (validation V6)."""
    d = t.get("derive")
    if not d:
        return s, desc
    idx = pd.date_range(s.index.min(), s.index.max(), freq=pd.infer_freq(s.index) or "MS")
    s = s.reindex(idx).interpolate(limit=3, limit_area="inside")
    scale = float(t.get("scale", 100 if d in ("pct_change", "logdiff") else 1))
    if d == "pct_change":
        s = s.pct_change() * scale
    elif d == "logdiff":
        s = np.log(s).diff() * scale
    elif d == "diff":
        s = s.diff() * scale
    elif d == "log":
        s = np.log(s)
    else:
        raise ValueError(f"unknown derive {d!r}")
    return s.dropna(), f"{desc}, {d}" + (f"×{scale:g}" if scale != 1 else "")


def _pipeline_factory(kind: str, period: int):
    if kind == "raw":
        return pp.raw_pipeline
    if kind == "djorno":
        return lambda: pp.djorno_pipeline(period=period, halflife=1.0 if period == 12 else 1.5, detrend="rolling", cluster=True)
    if kind == "djorno_nocluster":
        return lambda: pp.djorno_pipeline(period=period, halflife=1.0 if period == 12 else 1.5, detrend="rolling", cluster=False)
    if kind == "diffgt":
        return lambda: pp.Pipeline([pp.ZeroRepair(), pp.Log1p(), pp.Detrend("diff")])
    raise ValueError(kind)


def _defaults(spec: dict) -> dict:
    s = dict(spec)
    freq = s["freq"].upper()[0]
    s.setdefault("period", {"M": 12, "W": 52, "D": 7}[freq])
    s.setdefault("timeframe", gt.timeframe_for(freq))
    s.setdefault("geo", "")
    s.setdefault("gprop", "")
    s.setdefault("cat", 0)
    s.setdefault("horizons", [{"h": 0, "y_known": False}, {"h": 1, "y_known": True}])
    s.setdefault("transform", "diff" if freq == "M" else None)
    s.setdefault("p", 3 if freq == "M" else 4)
    s.setdefault("xlags", 1)
    s.setdefault("fourier_k", 1 if freq == "M" else 2)
    s.setdefault("min_train", 60 if freq == "M" else 104)
    s.setdefault("max_origins", 120 if freq == "M" else 160)
    s.setdefault("models", DEFAULT_MODELS)
    s.setdefault("windows", [None])
    s.setdefault("exclude", None)
    s.setdefault("alpha", 0.1)
    s.setdefault("answer_type", "quantity")
    s.setdefault("units", "")
    s.setdefault("label", s.get("title", s["id"]))
    s.setdefault("out_dir", "outputs")
    s.setdefault("lang_dir", "ltr")
    return s


# --------------------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------------------

def run(spec: dict, verbose: bool = True) -> dict:
    t0 = time.time()
    s = _defaults(spec)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    n_req0 = gt.REQUEST_COUNT

    # 1 truth
    y_raw, truth_desc = truth_from_spec(s["truth"])
    log(f"[truth] {truth_desc}: {len(y_raw)} obs {y_raw.index.min().date()} → {y_raw.index.max().date()}")

    # 2 GT  (weekly targets needing >5 years of history use stitched windows)
    if s["freq"].upper()[0] == "W" and s.get("history_years", 5) > 5:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        start = (pd.Timestamp.today() - pd.DateOffset(years=int(s["history_years"]))).strftime("%Y-%m-%d")
        X_raw, info = gt.fetch_many_stitched(s["queries"], start, end, geo=s["geo"], cat=s["cat"], gprop=s["gprop"])
        log(f"[gt] stitched {s['history_years']}-year weekly windows ({start} → {end})")
    else:
        X_raw, info = gt.fetch_many(s["queries"], geo=s["geo"], timeframe=s["timeframe"], cat=s["cat"], gprop=s["gprop"])
    log(f"[gt] {X_raw.shape[1]} series kept, dropped {[d['keyword'] for d in info['dropped']]}, failed {[f['keyword'] for f in info['failed']]}")
    if X_raw.shape[1] == 0:
        raise gt.TrendsUnavailable("no usable Google Trends series (all below privacy threshold)")

    # 3 align
    y, X = truth.align(y_raw, X_raw, s["freq"].upper()[0], how="mean")
    y = y.dropna()
    # enforce a contiguous PeriodIndex (e.g. BLS skipped Oct-2025 during the shutdown): fill
    # isolated gaps by linear interpolation and record them; calendar-based target dates in the
    # design must line up with positional ones in the naive benchmarks.
    full = pd.period_range(y.index[0], y.index[-1], freq=y.index.freq)
    gaps = [str(p) for p in full.difference(y.index)]
    if gaps:
        y = y.reindex(full).interpolate(limit=3, limit_area="inside")
        y = y.dropna()
        log(f"[align] filled {len(gaps)} gap(s) in the target by interpolation: {gaps[:6]}")
    X = X.reindex(y.index).ffill().bfill()
    log(f"[align] {len(y)} common periods {y.index[0]} → {y.index[-1]}")
    period = s["period"]

    results = {"id": s["id"], "question": s["question"], "truth": truth_desc, "freq": s["freq"], "n": int(len(y)), "gaps_filled": gaps,
               "range": [str(y.index[0]), str(y.index[-1])], "queries_kept": list(X.columns), "queries_dropped": info["dropped"],
               "horizons": {}, "spec": s}

    # 4 per horizon
    for hz in s["horizons"]:
        h = int(hz["h"])
        known = bool(hz.get("y_known", h >= 1))
        common = dict(h=h, p=s["p"], period=period, fourier_k=s["fourier_k"], y_known_at_origin=known,
                      min_train=s["min_train"], max_origins=s["max_origins"], transform=s["transform"])
        res: dict[str, ev.BacktestResult] = {}
        res["naive"] = ev.naive(y, h, s["min_train"], y_known_at_origin=known)
        res["drift"] = ev.drift(y, h, s["min_train"], y_known_at_origin=known)
        try:
            res["seasonal_naive"] = ev.seasonal_naive(y, h, period, s["min_train"])
        except Exception:
            pass
        clip_min = s.get("clip_min")  # e.g. 0 for rates/counts that cannot be negative
        res["ar"] = ev.rolling_backtest(y, None, model="ar", use_gt=False, **common)
        gt_models = []
        for m in s["models"]:
            model_kind, prep_kind = m.split("+")
            for w in s["windows"]:
                name = m if w is None else f"{m}@w{w}"
                try:
                    res[name] = ev.rolling_backtest(y, X, model=model_kind, use_gt=True, xlags=s["xlags"],
                                                    preprocess=_pipeline_factory(prep_kind, period), window=w, **common)
                    if len(res[name].preds):
                        if clip_min is not None:
                            res[name].preds["y_pred"] = res[name].preds["y_pred"].clip(lower=clip_min)
                        gt_models.append(name)
                    else:
                        del res[name]
                except Exception as e:  # keep going; record the failure
                    log(f"[warn] {name} h={h} failed: {type(e).__name__}: {e}")
        # benchmark = best target-only model by MAE on the common index
        bench_cands = [k for k in ("naive", "drift", "seasonal_naive", "ar") if k in res and len(res[k].preds)]
        tab0 = ev.compare({k: res[k] for k in bench_cands}, bench=bench_cands[0], y_hist=y, period=period)
        bench = tab0["MAE"].idxmin()
        tab = ev.compare(res, bench=bench, y_hist=y, period=period)
        verdict = ev.verdict(tab, bench, gt_models)
        out_h = {"bench": bench, "table": tab.reset_index().to_dict(orient="records"), "verdict": verdict}
        if s["exclude"]:
            tab_ex = ev.compare(res, bench=bench, y_hist=y, period=period, exclude=tuple(s["exclude"]))
            if int(tab_ex["n"].iloc[0]) == int(tab["n"].iloc[0]):
                out_h["exclusion_note"] = f"exclusion window {s['exclude']} is outside the evaluated sample; robustness view identical to the full sample and therefore omitted"
                log(f"[h={h}] {out_h['exclusion_note']}")
            else:
                out_h["table_excluded"] = tab_ex.reset_index().to_dict(orient="records")
                out_h["verdict_excluded"] = ev.verdict(tab_ex, bench, gt_models)
        # choose the model to forecast with: best GT model if it helps, else the benchmark
        chosen = verdict["best_model"] if verdict.get("gt_helps") else bench
        if s["exclude"] and out_h.get("verdict_excluded") and not out_h["verdict_excluded"].get("gt_helps") and verdict.get("gt_helps"):
            # helps only thanks to the shock window -> still use GT but flag it
            out_h["note"] = "GT gain comes mainly from the shock period; calm-period gain not significant"
        out_h["chosen_model"] = chosen
        # regime split: error when the target is high (in-season / stressed) vs low. Flu cells
        # showed GT gains are entirely in-season and reverse off-season (02-Q2-A/C).
        try:
            best_gt = verdict.get("best_model")
            if best_gt in res and bench in res:
                b, g = res[bench].preds, res[best_gt].preds
                common_i = b.index.intersection(g.index)
                if s["answer_type"] == "direction" or s["transform"] == "diff" and float(y.diff().abs().mean()) < 0.05 * float(y.abs().mean()):
                    # trending series (prices): split on the size of the realised change, not the level
                    chg = (b.loc[common_i, "y_true"].values - y.reindex(pd.Index(b.loc[common_i, "origin"].values)).values)
                    med = float(np.nanmedian(np.abs(chg)))
                    hi = pd.Series(np.abs(chg) > med, index=common_i)
                    split_on = "abs_change"
                else:
                    med = float(y.median())
                    hi = b.loc[common_i, "y_true"] > med
                    split_on = "level"
                def _mae(p, m): return float(np.mean(np.abs(p.loc[common_i, "y_true"][m] - p.loc[common_i, "y_pred"][m]))) if m.any() else None
                out_h["regime_split"] = {"split_on": split_on, "threshold": med, "n_high": int(hi.sum()), "n_low": int((~hi).sum()),
                                         "bench_mae_high": _mae(b, hi), "gt_mae_high": _mae(g, hi),
                                         "bench_mae_low": _mae(b, ~hi), "gt_mae_low": _mae(g, ~hi)}
        except Exception:
            pass
        # 5 forward forecast (benchmark's own point is always computed for the answer text)
        def _bench_point(name: str):
            last = float(y.iloc[-1])
            steps = h if h >= 1 else 1
            if name == "naive":
                return last
            if name == "drift":
                return last + steps * float(y.diff().dropna().mean())
            if name == "seasonal_naive":
                return float(y.iloc[-period + (steps - 1)])
            try:
                return ev.forecast_next(y, None, model="ar", use_gt=False, p=s["p"], period=period, fourier_k=s["fourier_k"],
                                        y_known_at_origin=known, transform=s["transform"], h=max(h, 1) if not known else h)["point"]
            except Exception:
                return None
        try:
            out_h["benchmark_forecast"] = {"model": bench, "point": _bench_point(bench)}
        except Exception:
            pass
        # the best GT model's own point is always computed too, so the answer can show how much the
        # search data would have moved the number even when the benchmark is chosen (validation V6)
        best_gt = verdict.get("best_model")
        if best_gt and best_gt in res and best_gt != chosen:
            try:
                base_ = best_gt.split("@")[0]
                mk, pk = base_.split("+")
                w_ = int(best_gt.split("@w")[1]) if "@w" in best_gt else None
                Xf = X_raw.copy()
                Xf.index = Xf.index.to_period({"M": "M", "W": "W-SAT", "D": "D"}[s["freq"].upper()[0]])
                Xf = Xf.groupby(level=0).mean()
                gfc = ev.forecast_next(y, Xf, h=h, model=mk, use_gt=True, p=s["p"], xlags=s["xlags"], period=period,
                                       fourier_k=s["fourier_k"], y_known_at_origin=known, window=w_,
                                       preprocess=_pipeline_factory(pk, period), transform=s["transform"])
                out_h["gt_reference_forecast"] = {"model": best_gt, "point": gfc["point"],
                                                  "top_features": [x[0] if isinstance(x, (list, tuple)) else x for x in (gfc.get("features") or [])[:8]]}
            except Exception as e:
                out_h["gt_reference_forecast"] = {"model": best_gt, "error": f"{type(e).__name__}: {e}"}
        # nowcast feasibility note, independent of which model is chosen (validation V1: the note
        # was only attached to GT forecasts, so a benchmark answer silently duplicated h=1)
        if h == 0:
            x_last = X_raw.index.max().to_period({"M": "M", "W": "W-SAT", "D": "D"}[s["freq"].upper()[0]])
            if x_last <= y.index[-1]:
                out_h["note"] = ("no unpublished period to nowcast: Google Trends ends in the same period as the last "
                                 "published value, so h=0 is a one-period-ahead forecast identical to h=1")
        try:
            if chosen in ("naive", "drift", "seasonal_naive"):
                last = float(y.iloc[-1])
                steps = h if h >= 1 else 1
                nxt = y.index[-1] + steps
                if chosen == "naive":
                    point = last
                elif chosen == "drift":
                    point = last + steps * float(y.diff().dropna().mean())
                else:
                    point = float(y.iloc[-period + (steps - 1)])
                fc = {"origin": str(y.index[-1]), "target_date": str(nxt), "point": point, "model": chosen, "features": None}
            elif chosen == "ar":
                fc = ev.forecast_next(y, None, model="ar", use_gt=False, p=s["p"], period=period, fourier_k=s["fourier_k"],
                                      y_known_at_origin=known, transform=s["transform"], h=max(h, 1) if not known else h)
            else:
                base = chosen.split("@")[0]
                model_kind, prep_kind = base.split("+")
                w = int(chosen.split("@w")[1]) if "@w" in chosen else None
                X_for_fc = truth.align(y_raw, X_raw, s["freq"].upper()[0])[1]  # keep GT rows beyond y for nowcasts
                X_for_fc = X_raw.copy()
                X_for_fc.index = X_for_fc.index.to_period({"M": "M", "W": "W-SAT", "D": "D"}[s["freq"].upper()[0]])
                X_for_fc = X_for_fc.groupby(level=0).mean()
                fc = ev.forecast_next(y, X_for_fc, h=h, model=model_kind, use_gt=True, p=s["p"], xlags=s["xlags"], period=period,
                                      fourier_k=s["fourier_k"], y_known_at_origin=known, window=w,
                                      preprocess=_pipeline_factory(prep_kind, period), transform=s["transform"])
            errs = res[chosen].errors.values if chosen in res else res[bench].errors.values
            q = ev.conformal_interval(errs, alpha=s["alpha"], recent=min(60, len(errs)))
            if clip_min is not None:
                fc["point"] = max(float(fc["point"]), float(clip_min))
            fc["lo"], fc["hi"], fc["alpha"] = fc["point"] - q, fc["point"] + q, s["alpha"]
            if clip_min is not None and fc["lo"] < float(clip_min):
                fc["lo_unclipped"] = fc["lo"]  # report the clip instead of hiding it (validation V7)
                fc["lo"] = float(clip_min)
            fc["interval_halfwidth"] = q
            if fc.get("features"):
                fc["top_features"] = [x[0] if isinstance(x, (list, tuple)) else x for x in fc["features"][:8]]
            if s["answer_type"] == "direction":
                # base rate of "up" over the backtest window and hit-rates vs that base rate
                bp = res[bench].preds
                if chosen in res and chosen != bench:
                    bp = bp.loc[bp.index.intersection(res[chosen].preds.index)]  # same n as the metrics table
                y_prev = y.reindex(pd.Index(bp["origin"].values)).values if h >= 1 else y.shift(1).reindex(bp.index).values
                ups = (bp["y_true"].values > y_prev)
                base_rate = float(np.nanmean(ups))
                fc["base_rate_up"] = base_rate
                if chosen in res and chosen != bench:
                    cp = res[chosen].preds.reindex(bp.index)
                    hits = float(np.nanmean(np.sign(cp["y_pred"].values - y_prev) == np.sign(cp["y_true"].values - y_prev)))
                    fc["hit_rate_chosen"] = hits
                fc["hit_rate_always_up"] = base_rate
                fc["direction_n"] = int(len(bp))
                last_known = float(y.iloc[-1])
                resid = res[chosen].errors.values
                sd = float(np.std(resid)) if len(resid) > 5 else float("nan")
                from scipy.stats import norm
                fc["prob_up"] = float(1 - norm.cdf(0, loc=fc["point"] - last_known, scale=sd)) if sd == sd and sd > 0 else None
                if not verdict.get("gt_helps"):
                    fc["prob_up_note"] = "Google Trends adds no validated edge; prob_up is the benchmark's drift + residual spread, close to the unconditional base rate"
            out_h["forecast"] = fc
        except Exception as e:
            out_h["forecast_error"] = f"{type(e).__name__}: {e}"
            log(f"[warn] forecast h={h} failed: {e}")
        # keep preds for the report
        out_h["_preds"] = {k: v.preds for k, v in res.items()}
        results["horizons"][str(h)] = out_h
        log(f"[h={h}] bench={bench} verdict={verdict['strength']} ({verdict['reason']})")

    results["requests_used"] = gt.REQUEST_COUNT - n_req0
    results["seconds"] = round(time.time() - t0, 1)
    results["_y"], results["_X"] = y, X
    return results


# --------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------

def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and (np.isnan(o) or np.isinf(o))) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Period, pd.Timestamp)):
        return str(o)
    if isinstance(o, (pd.Series, pd.DataFrame)):
        return None
    return o


def write_outputs(results: dict, headline: str | None = None, answer_text: str | None = None,
                  confidence: str | None = None, method_notes: list[str] | None = None,
                  caveats: list[str] | None = None) -> dict:
    """Write results.json and report.html; return paths. Text arguments let the skill supply
    the plain-language answer; defaults are generated from the verdicts."""
    s = results["spec"]
    out_dir = REPO_ROOT / s["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    y, X = results["_y"], results["_X"]
    # primary horizon: first one with a forecast
    hz_key = next((k for k, v in results["horizons"].items() if "forecast" in v), None)
    hz = results["horizons"][hz_key] if hz_key else None
    fc = hz["forecast"] if hz else None
    vd = hz["verdict"] if hz else {"strength": "none", "reason": "no horizon evaluated"}
    strength = vd.get("strength", "none")
    conf = confidence
    if conf is None:
        # default class follows the skill's rules: full-sample verdict governs; the excluded view can
        # only lower it; the current regime can lower it once more when the GT model was chosen
        ladder = ["none", "low", "medium", "high"]
        conf = {"strong": "high", "moderate": "medium", "none": "low"}.get(strength, "low")
        vex = hz.get("verdict_excluded") if hz else None
        if vex and vex.get("strength") == "none" and conf == "high":
            conf = "medium"
        rs = hz.get("regime_split") if hz else None
        if rs and hz.get("chosen_model") not in (None, hz.get("bench")) and conf != "low":
            cur_high = float(y.iloc[-1]) > rs["threshold"] if rs.get("split_on") == "level" else None
            if cur_high is not None:
                b_mae, g_mae = (rs["bench_mae_high"], rs["gt_mae_high"]) if cur_high else (rs["bench_mae_low"], rs["gt_mae_low"])
                if b_mae is not None and g_mae is not None and b_mae < g_mae:
                    conf = ladder[max(1, ladder.index(conf) - 1)]
                    hz["confidence_note"] = ("downgraded one class: the target is currently in the regime "
                                             f"({'high' if cur_high else 'low'} vs {rs['threshold']:.2f}) where the benchmark's error is lower")
    if fc and headline is None and not (s.get("display_exp") or (isinstance(s.get("truth"), dict) and s["truth"].get("log"))):
        headline = f"{fc['point']:.2f}{s['units']} for {fc['target_date']} ({int((1 - s['alpha']) * 100)}% interval {fc['lo']:.2f}–{fc['hi']:.2f})"
    if answer_text is None:
        answer_text = vd.get("reason", "")
        if hz and hz.get("verdict_excluded"):
            answer_text += f" Excluding the shock window: {hz['verdict_excluded'].get('reason', '')}"
        if hz and hz.get("confidence_note"):
            answer_text += f" Confidence {conf}: {hz['confidence_note']}."
    metrics = pd.DataFrame(hz["table"]).set_index("model") if hz else None
    preds = hz["_preds"] if hz else {}
    # log targets (spec truth.log=true or display_exp=true) are shown in original units
    if s.get("display_exp") or (isinstance(s.get("truth"), dict) and s["truth"].get("log")):
        y = np.exp(y)
        preds = {k: v.assign(y_true=np.exp(v["y_true"]), y_pred=np.exp(v["y_pred"])) for k, v in preds.items()}
        if fc:
            fc = dict(fc)
            for key in ("point", "lo", "hi"):
                if fc.get(key) is not None:
                    fc[key] = float(np.exp(fc[key]))
            if headline is None:
                headline = f"{fc['point']:,.0f} for {fc['target_date']} ({int((1 - s['alpha']) * 100)}% interval {fc['lo']:,.0f}–{fc['hi']:,.0f})"
    bench = hz["bench"] if hz else None
    chosen = hz.get("chosen_model") if hz else None
    bt = {}
    if bench in preds:
        bt[f"{bench} (benchmark)"] = preds[bench]["y_pred"]
    if chosen and chosen != bench and chosen in preds:
        bt[chosen] = preds[chosen]["y_pred"]
    elif hz and vd.get("best_model") in preds and vd.get("best_model") != bench:
        bt[vd["best_model"]] = preds[vd["best_model"]]["y_pred"]
    bt_truth = preds[bench]["y_true"] if bench in preds else None
    if bt_truth is not None and bt:
        common_idx = bt_truth.index
        for v in bt.values():
            common_idx = common_idx.intersection(v.index)
        bt_truth = bt_truth.loc[common_idx]
        bt = {k: v.loc[common_idx] for k, v in bt.items()}
    fc_date = None
    if fc:
        try:
            fc_date = pd.Period(fc["target_date"]).to_timestamp().strftime("%Y-%m-%d")
        except Exception:
            fc_date = str(fc["target_date"])[:10]
    extra = []
    if hz and hz.get("table_excluded"):
        tab_ex = pd.DataFrame(hz["table_excluded"]).set_index("model")
        rows ="".join("<tr><td>" + html_escape(str(i)) + "</td>" + "".join(f"<td>{_fmt(r[c], 3) if isinstance(r[c], float) else r[c]}</td>" for c in ("n", "MAE", "RMSE", "OOS_R2_vs_bench", "DirAcc", "DM_p", "CW_p") if c in tab_ex.columns) + "</tr>" for i, r in tab_ex.iterrows())
        head = "".join(f"<th>{c}</th>" for c in ["model"] + [c for c in ("n", "MAE", "RMSE", "OOS_R2_vs_bench", "DirAcc", "DM_p", "CW_p") if c in tab_ex.columns])
        extra.append({"title": f"Robustness — excluding {s['exclude'][0]} → {s['exclude'][1]}",
                      "html": f"<div class='card'><p class='muted'>{html_escape(hz['verdict_excluded'].get('reason', ''))}</p><div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div></div>"})
    if len(results["horizons"]) > 1:
        rows = ""
        for k, v in results["horizons"].items():
            f = v.get("forecast", {})
            rows += f"<tr><td>h={k}</td><td>{html_escape(v['bench'])}</td><td>{html_escape(v['verdict'].get('best_model', '—'))}</td><td>{_fmt(v['verdict'].get('oos_r2'))}</td><td>{v['verdict'].get('strength')}</td><td>{html_escape(v.get('chosen_model', ''))}</td><td>{_fmt(f.get('point')) if f else '—'}</td><td>{(_fmt(f.get('lo')) + ' – ' + _fmt(f.get('hi'))) if f and f.get('lo') is not None else '—'}</td></tr>"
        extra.append({"title": "All horizons", "html": f"<div class='card'><div class='scroll'><table><thead><tr><th>horizon</th><th>benchmark</th><th>best GT model</th><th>OOS R²</th><th>strength</th><th>used for forecast</th><th>point</th><th>interval</th></tr></thead><tbody>{rows}</tbody></table></div></div>"})
    queries_info = [{"query": c, "nonzero_share": round(float((X[c] > 0).mean()), 2), "corr_with_target_changes": round(float(np.corrcoef(X[c].diff().dropna(), y.diff().dropna().reindex(X[c].diff().dropna().index).fillna(0))[0, 1]), 2) if len(X) > 10 else None} for c in X.columns]
    method_notes = method_notes or [
        f"Target frequency {s['freq']}, {results['n']} periods {results['range'][0]} → {results['range'][1]}; Google Trends pulled individually per query (geo={s['geo'] or 'worldwide'}), cached as dated vintages.",
        f"Rolling-origin backtest ({s['max_origins']} origins max, min_train={s['min_train']}), direct multi-step design, {'differences' if s['transform'] == 'diff' else 'levels'}; preprocessing fitted inside each training window.",
        f"Benchmark = best target-only model among naive / seasonal naive / AR: {bench}. GT models: {', '.join(s['models'])}.",
        "Verdict rule: Google Trends 'helps' only if OOS R² > 0.02 vs the benchmark and Clark–West or Diebold–Mariano p < 0.10.",
        f"Interval: rolling conformal from the last ≤60 out-of-sample residuals of the chosen model (nominal {int((1 - s['alpha']) * 100)}%).",
    ]
    caveats = caveats or [
        "Google Trends values are relative search interest (0–100 per request), sampled and revised by Google; not counts.",
        "Search attention reacts to news as well as to the underlying quantity; relationships drift (ARGO-style rolling refits mitigate, not eliminate).",
        "Out-of-sample gains concentrated in shock periods do not guarantee gains in calm periods — see the robustness table.",
    ]
    html_path = out_dir / f"{s['id']}.html"
    render_report(html_path, title=s.get("title", s["id"]), question=s["question"], answer_headline=headline or "No forecast produced",
                  answer_text=answer_text, confidence=conf,
                  forecast={"date": fc_date, "point": fc["point"], "lo": fc.get("lo"), "hi": fc.get("hi"), "alpha": s["alpha"], "label": s["label"]} if fc else None,
                  history=y.tail(160), history_label=s["label"], backtest=bt, backtest_truth=bt_truth, metrics=metrics,
                  gt_series=X.tail(160), queries_info=queries_info, method_notes=method_notes, caveats=caveats,
                  provenance={"truth": results["truth"], "google_trends": f"data/trends/ ({len(X.columns)} series, geo={s['geo'] or 'worldwide'}, timeframe={s['timeframe']})",
                              "requests_this_run": results["requests_used"], "runtime_s": results["seconds"], "spec": json.dumps(_json_safe(s), ensure_ascii=False)[:400]},
                  extra_sections=extra, lang_dir=s.get("lang_dir", "ltr"), chosen_model=chosen)
    json_path = out_dir / f"{s['id']}.json"
    json_path.write_text(json.dumps(_json_safe(results), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return {"html": str(html_path), "json": str(json_path)}


def html_escape(s: str) -> str:
    import html as _h
    return _h.escape(str(s))


def _fmt(v, nd=2):
    from .report import _fmt as f
    return f(v, nd)


def salience(spec: dict) -> dict:
    """Descriptive (NOT predictive) attention-share report for discrete contests.

    spec: {"id","question","title",
           "entities": ["Democratic Party","Republican Party"]  |  [{"label":"Republican Party","query":"/m/07wbk"}, ...],
           "geo":"US","timeframe":"2026-06-01 2026-11-03", "context": "...optional text..."}
    No truth/freq/queries keys are needed; this mode bypasses the forecasting pipeline entirely.
    Pulls the entities jointly (one scale), computes each entity's share of total attention, the
    4-week mean share, its trend, and a spike diagnostic (share with the single largest week
    removed). Writes outputs/<id>.html + .json. The artifact says explicitly that attention share
    picked winners in 21/36 past US contests (experiments/04-Q4-F) and is not a forecast.
    """
    # entities: plain strings, or {"label": "Republican Party", "query": "/m/07wbk"} for topic ids
    raw_ents = list(spec["entities"])[:5]
    queries = [e["query"] if isinstance(e, dict) else e for e in raw_ents]
    labels = [e.get("label", e["query"]) if isinstance(e, dict) else e for e in raw_ents]
    geo = spec.get("geo", "")
    tf = spec.get("timeframe", "today 3-m")
    df = gt.fetch_interest(queries, geo=geo, timeframe=tf)
    df.columns = [labels[queries.index(c)] if c in queries else c for c in df.columns]
    ents = list(df.columns)
    # short timeframes come back daily -> aggregate to weeks so "last 4" means 4 weeks
    if len(df) > 1 and (df.index[1] - df.index[0]).days < 7:
        df = df.resample("W-SAT").mean()
    tot = df.sum(axis=1).replace(0, np.nan)
    share = df.div(tot, axis=0)
    last4 = share.tail(4).mean()
    prev4 = share.iloc[-8:-4].mean() if len(share) >= 8 else share.head(4).mean()
    # spike diagnostic: drop the loudest week *within the last four* (not the global max, which
    # is usually an old news week and makes the check vacuous — validation V4-opus)
    spike_week = tot.tail(4).idxmax()
    share_ex_spike = share.tail(4).drop(index=spike_week).mean()
    lead = last4.idxmax()
    out_dir = REPO_ROOT / spec.get("out_dir", "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"id": spec["id"], "question": spec["question"], "entities": ents, "geo": geo, "timeframe": tf,
           "share_last4": {k: round(float(v), 3) for k, v in last4.items()},
           "share_prev4": {k: round(float(v), 3) for k, v in prev4.items()},
           "share_last4_ex_spike_week": {k: round(float(v), 3) for k, v in share_ex_spike.items()},
           "spike_week": str(spike_week.date()), "leader_by_attention": lead,
           "validated_hit_rate": "21/36 pooled US contests 2006–2024 (58%); presidential raw share 2/5; swing states 12/21",
           "is_forecast": False}
    (out_dir / f"{spec['id']}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = "".join(f"<tr><td>{html_escape(e)}</td><td>{last4[e]:.3f}</td><td>{prev4[e]:.3f}</td><td>{share_ex_spike[e]:.3f}</td></tr>" for e in ents)
    extra = [{"title": "Attention share (descriptive)", "html": f"<div class='card'><div class='scroll'><table><thead><tr><th>entity</th><th>share, last 4 weeks</th><th>share, previous 4 weeks</th><th>last 4 weeks excl. spike week {spike_week.date()}</th></tr></thead><tbody>{rows}</tbody></table></div></div>"}]
    render_report(out_dir / f"{spec['id']}.html", title=spec.get("title", spec["id"]), question=spec["question"],
                  answer_headline=f"No forecast: search attention does not predict winners (validated hit-rate 21/36).",
                  answer_text=f"Current attention leader: {lead} with {last4[lead]:.0%} of joint search interest over the last 4 weeks "
                              f"({prev4[lead]:.0%} in the previous 4; {share_ex_spike[lead]:.0%} excluding the spike week). "
                              f"Attention tracks news and controversy, not support — in 2012 and 2024 the more-searched presidential candidate lost. " + spec.get("context", ""),
                  confidence="none", forecast=None, history=share[lead].rename(f"{lead} share"), history_label=f"{lead} share of attention",
                  backtest=None, backtest_truth=None, metrics=None, gt_series=df, queries_info=[{"entity": e, "mean_interest": round(float(df[e].mean()), 1)} for e in ents],
                  method_notes=["Joint Google Trends pull (one normalisation scale) for all entities; share = entity / sum per week.",
                                "Validation on history (research/experiments/04-Q4-F.md): presidential 2008–2024 raw share 2/5, intent-modified queries 1–3/5 with negative correlation to margins, swing states 12/21, midterms 4/5 (n=5)."],
                  caveats=["This is a description of attention, not a forecast. Do not read a share above 50% as a probability of winning.",
                           "A single news week can flip the share (see the excl.-spike column).",
                           "For a real forecast use polling averages or prediction markets as the target series."],
                  provenance={"google_trends": f"data/trends/ joint pull {ents} geo={geo or 'worldwide'} {tf}"}, extra_sections=extra,
                  lang_dir=spec.get("lang_dir", "ltr"))
    res["outputs"] = {"html": str(out_dir / f"{spec['id']}.html"), "json": str(out_dir / f"{spec['id']}.json")}
    return res


def _pickle_path(res: dict) -> Path:
    return REPO_ROOT / res["spec"]["out_dir"] / f"{res['spec']['id']}.pkl"


def rerender(id_: str, out_dir: str = "outputs", **text) -> dict:
    """Re-write the artifact with the skill's own headline/answer_text/confidence without
    re-running the backtests (results are pickled by run_spec_file).

    Security note: the pickle is produced by this same process family into the local outputs
    directory and only ever read back from there; it is not an untrusted input."""
    import pickle
    p = REPO_ROOT / out_dir / f"{id_}.pkl"
    res = pickle.loads(p.read_bytes())
    return write_outputs(res, **{k: v for k, v in text.items() if v})


def run_spec_file(path: str | Path, verbose: bool = True) -> dict:
    import pickle
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    res = run(spec, verbose=verbose)
    paths = write_outputs(res)
    try:
        _pickle_path(res).write_bytes(pickle.dumps(res))
    except Exception:
        pass
    if verbose:
        def _fc(f):
            if not f:
                return None
            g = {k: v for k, v in f.items() if k != "features"}
            if f.get("features"):
                g["top_features"] = [x[0] if isinstance(x, (list, tuple)) else x for x in f["features"][:6]]
            return g
        print(json.dumps({"outputs": paths, "requests_used": res["requests_used"], "seconds": res["seconds"], "gaps_filled": res.get("gaps_filled"),
                          "horizons": {k: {"bench": v["bench"], "verdict": v["verdict"], "verdict_excluded": v.get("verdict_excluded"),
                                           "chosen_model": v.get("chosen_model"), "benchmark_forecast": v.get("benchmark_forecast"),
                                           "gt_reference_forecast": v.get("gt_reference_forecast"),
                                           "note": v.get("note"), "exclusion_note": v.get("exclusion_note"),
                                           "regime_split": v.get("regime_split"), "forecast": _fc(v.get("forecast")),
                                           "forecast_error": v.get("forecast_error")}
                                       for k, v in res["horizons"].items()}}, ensure_ascii=False, indent=1, default=str))
    return res


if __name__ == "__main__":
    import sys
    run_spec_file(sys.argv[1])
