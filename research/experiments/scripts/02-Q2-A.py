"""Cell 02-Q2-A — Q2 (US ILI %, 2 and 4 weeks ahead) × Approach A (Choi–Varian minimal).

Run from the repo root:  PYTHONIOENCODING=utf-8 python research/experiments/scripts/02-Q2-A.py

Design (from the cell prompt): 5 hand-picked keywords, weekly GT via individual pulls over
`today 5-y`, raw pipeline (ZeroRepair -> log1p), OLS ARX vs {naive, seasonal_naive(52), ar},
expanding window, levels AND transform='diff', p=4, xlags=1, period=52, fourier_k=2,
min_train=104. Horizons h=0 (nowcast), 2, 4 — all with y_known_at_origin=False because the
latest ILI week is not yet published at the origin.

Two small local helpers extend the library (documented in the .md, not patched into src/ because
other cells run concurrently against the same package):
  * naive_lag  — random walk that uses y_{t-1} (the last *published* value) for h>=1;
                 ev.naive uses y_t, which is not known at the origin under publication lag.
  * forecast_pub_lag — ev.forecast_next with the origin at the latest GT week (the h=0 origin
                 logic) for any h when y_known_at_origin=False.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from trends_predict import gt, truth, preprocess as pp, evaluate as ev
from trends_predict.evaluate import BacktestResult
from trends_predict.models import make_design, make_regressor

ID = "02-Q2-A"
T0 = time.time()
RUN_START_UTC = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
KWS = ["flu symptoms", "flu", "tamiflu", "fever", "cough"]
GEO = "US"
TF_PRIMARY = "today 5-y"
TF_EXT = gt.timeframe_for("W", "2017-01-01", "2021-12-31")   # chained extension (secondary)
EXCL = ("2020-03-01", "2021-06-30")
HORIZONS = (0, 2, 4)
DESIGN = dict(p=4, xlags=1, period=52, fourier_k=2, min_train=104, y_known_at_origin=False, window=None)
OUT_JSON = f"research/experiments/{ID}.json"


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def naive_lag(y: pd.Series, h: int, min_train: int) -> BacktestResult:
    """Random walk from the last *published* value y_{t-1} (publication lag of one week)."""
    y = y.astype(float).sort_index()
    rows = [{"target_date": y.index[t + h], "origin": y.index[t], "y_true": y.iloc[t + h], "y_pred": y.iloc[t - 1]}
            for t in range(min_train, len(y) - h)]
    return BacktestResult("naive_lag", pd.DataFrame(rows).set_index("target_date"), h)


def forecast_pub_lag(y: pd.Series, X: pd.DataFrame, *, h: int, model: str, use_gt: bool, p: int, xlags: int,
                     period: int, fourier_k: int, preprocess, transform: str | None) -> dict:
    """Forward forecast with the origin at the latest GT week (y unknown there, GT known).
    Mirrors ev.forecast_next's h=0 branch, generalised to any h with y_known_at_origin=False."""
    y = y.astype(float).sort_index()
    X = X.astype(float).sort_index()
    last_y = y.dropna().index.max()
    nxt = X.index[X.index > last_y][:1]
    if len(nxt) == 0:
        raise ValueError("GT does not extend beyond the last published y")
    idx = y.index.union(nxt)
    y_ext = y.reindex(idx)
    origin_pos = len(y_ext) - 1
    Xw = X.reindex(idx)
    if use_gt and preprocess is not None:
        tr = preprocess()
        tr.fit(Xw.iloc[:origin_pos + 1].dropna(how="all"))
        Xw = tr.transform(Xw.ffill())
    y_model = y_ext.diff() if transform == "diff" else y_ext
    d = make_design(y_model, Xw if use_gt else None, h=h, p=p, xlags=xlags, period=period, fourier_k=fourier_k,
                    y_known_at_origin=False)
    base = y_ext.shift(1)
    if transform == "diff":
        d.target = (y_ext.shift(-h) - base).loc[d.Z.index]
    cols = d.all_cols if use_gt else d.bench_cols
    origin_label = idx[origin_pos]
    known = d.target.notna() & (d.target_date <= last_y)
    Ztr = d.Z.loc[known.values, cols]
    ytr = d.target.loc[Ztr.index]
    ok = ytr.notna() & Ztr.notna().all(axis=1)
    Ztr, ytr = Ztr.loc[ok], ytr.loc[ok]
    reg = make_regressor(model)
    reg.fit(Ztr.values, ytr.values)
    zq = d.Z.loc[[origin_label], cols]
    pred = float(np.asarray(reg.predict(zq.values))[0])
    if transform == "diff":
        pred = float(base.loc[origin_label]) + pred
    c = np.ravel(reg.coef_)[1:]  # drop intercept
    feats = sorted([(cols[i], float(c[i])) for i in range(len(cols))], key=lambda t: -abs(t[1]))
    return {"origin": str(origin_label), "target_date": str(d.target_date.loc[origin_label]), "point": pred,
            "model": model, "transform": transform, "use_gt": use_gt, "n_train": int(len(Ztr)),
            "last_published_y": {"date": str(last_y), "value": float(y.loc[last_y])},
            "features": feats, "origin_row": {k: float(v) for k, v in zq.iloc[0].items()}}


def md_table(tab: pd.DataFrame, bench: str) -> str:
    cols = [c for c in ["n", "MAE", "RMSE", "MAPE", "OOS_R2_vs_bench", "MASE", "DirAcc", "DM_p", "CW_p"] if c in tab.columns]
    head = "| model | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    body = ""
    for m, r in tab.iterrows():
        name = f"**{m}** (bench)" if m == bench else m
        cells = []
        for c in cols:
            v = r[c]
            if c == "n":
                cells.append(str(int(v)))
            elif pd.isna(v):
                cells.append("")
            elif c in ("DM_p", "CW_p"):
                cells.append(f"{v:.3f}")
            else:
                cells.append(f"{v:.3f}")
        body += f"| {name} | " + " | ".join(cells) + " |\n"
    return head + body


def records(tab: pd.DataFrame) -> list[dict]:
    out = []
    for m, r in tab.iterrows():
        d = {"model": m}
        for k, v in r.items():
            d[k] = None if (isinstance(v, float) and np.isnan(v)) else (int(v) if k == "n" else float(v))
        out.append(d)
    return out


def my_requests(kws, timeframe) -> int:
    cat = gt.catalog()
    if cat.empty:
        return 0
    kwcol = cat["keywords"].apply(lambda k: k[0] if isinstance(k, list) and len(k) == 1 else str(k))
    # total vintages ever pulled for this cell's keyword/timeframe set (cache-hits on re-runs add none);
    # len(gt.catalog()) deltas are unusable because other cells pull concurrently.
    m = kwcol.isin(kws) & (cat["timeframe"] == timeframe) & (cat["geo"] == GEO)
    return int(m.sum())


# ----------------------------------------------------------------------------------------------
# backtest battery for one aligned (y, X)
# ----------------------------------------------------------------------------------------------

def run_battery(ya: pd.Series, Xa: pd.DataFrame, label: str) -> dict:
    out = {}
    for h in HORIZONS:
        kw = dict(h=h, **DESIGN)
        res = {}
        if h >= 1:
            res["naive_y_t (not known at origin)"] = ev.naive(ya, h, DESIGN["min_train"])
        res["naive_lag"] = naive_lag(ya, h, DESIGN["min_train"]) if h >= 1 else ev.naive(ya, 0, DESIGN["min_train"])
        res["seasonal_naive"] = ev.seasonal_naive(ya, h, 52, DESIGN["min_train"])
        res["ar_levels"] = ev.rolling_backtest(ya, None, model="ar", use_gt=False, **{k: v for k, v in kw.items() if k != "xlags"})
        res["ar_diff"] = ev.rolling_backtest(ya, None, model="ar", use_gt=False, transform="diff", **{k: v for k, v in kw.items() if k != "xlags"})
        res["ols+GT_levels"] = ev.rolling_backtest(ya, Xa, model="ols", use_gt=True, preprocess=pp.raw_pipeline, **kw)
        res["ols+GT_diff"] = ev.rolling_backtest(ya, Xa, model="ols", use_gt=True, preprocess=pp.raw_pipeline, transform="diff", **kw)
        gt_models = ["ols+GT_levels", "ols+GT_diff"]
        fair_bench_cands = ["naive_lag", "seasonal_naive", "ar_levels", "ar_diff"]
        # pick the benchmark = best information-consistent target-only model by RMSE on the common sample
        tab0 = ev.compare(res, bench="naive_lag", y_hist=ya, period=52)
        bench = tab0.loc[fair_bench_cands, "RMSE"].idxmin()
        tables = {}
        verdicts = {}
        for view, excl in (("full", None), ("excl_covid", EXCL)):
            tab = ev.compare(res, bench=bench, y_hist=ya, period=52, exclude=excl)
            tables[view] = tab
            verdicts[view] = ev.verdict(tab, bench=bench, gt_models=gt_models)
            print(f"\n=== [{label}] h={h}  view={view}  bench={bench}  n={int(tab['n'].iloc[0])} ===")
            print(tab.round(3).to_string())
            print("verdict:", verdicts[view])
        out[h] = {"results": res, "bench": bench, "tables": tables, "verdicts": verdicts, "gt_models": gt_models}
    return out


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------

def main():
    # --- truth ---------------------------------------------------------------------------------
    y = truth.fluview_ili("nat", 201040)
    full = pd.date_range(y.index.min(), y.index.max(), freq="W-SAT")
    truth_info = {"source": "CDC FluView ILINet via Delphi Epidata (fluview, wili, nat)", "first": str(y.index.min().date()),
                  "last": str(y.index.max().date()), "n": int(len(y)), "missing_saturdays": int(len(full.difference(y.index))),
                  "nan": int(y.isna().sum())}
    print("truth:", truth_info)

    # --- GT primary -----------------------------------------------------------------------------
    X, info = gt.fetch_many(KWS, geo=GEO, timeframe=TF_PRIMARY)
    gt_info = {"timeframe": TF_PRIMARY, "first": str(X.index.min().date()), "last": str(X.index.max().date()), "n": int(len(X)),
               "cols": list(X.columns), "dropped": info["dropped"], "failed": info["failed"],
               "zeros": {c: int((X[c] == 0).sum()) for c in X.columns}, "nan": int(X.isna().sum().sum())}
    print("GT primary:", gt_info)

    ya, Xa = truth.align(y, X, "W")
    fullp = pd.period_range(ya.index.min(), ya.index.max(), freq="W-SAT")
    align_info = {"first": str(ya.index.min()), "last": str(ya.index.max()), "n": int(len(ya)),
                  "gaps": int(len(fullp.difference(ya.index))), "nan_y": int(ya.isna().sum()), "nan_X": int(Xa.isna().sum().sum()),
                  "gt_weeks_beyond_last_y": [str(d.date()) for d in X.index[X.index > y.index.max()]]}
    print("aligned primary:", align_info)
    assert align_info["gaps"] == 0 and align_info["nan_y"] == 0 and align_info["nan_X"] == 0

    primary = run_battery(ya, Xa, "primary today 5-y")

    # --- GT chained extension (secondary, makes the COVID exclusion non-vacuous) ------------------
    X_old, info_old = gt.fetch_many(KWS, geo=GEO, timeframe=TF_EXT)
    ov = X_old.index.intersection(X.index)
    scale = (X.loc[ov].mean() / X_old.loc[ov].mean())
    X_chain = pd.concat([X_old.loc[X_old.index < X.index.min()].mul(scale, axis=1), X]).sort_index()
    ext_info = {"timeframe": TF_EXT, "first": str(X_old.index.min().date()), "last": str(X_old.index.max().date()),
                "n_old": int(len(X_old)), "overlap_weeks": int(len(ov)), "scale": {c: round(float(scale[c]), 3) for c in X.columns},
                "overlap_corr": {c: round(float(np.corrcoef(X.loc[ov, c], X_old.loc[ov, c])[0, 1]), 3) for c in X.columns},
                "zeros_old": {c: int((X_old[c] == 0).sum()) for c in X_old.columns}, "dropped": info_old["dropped"],
                "failed": info_old["failed"]}
    print("GT extension:", ext_info)
    ya2, Xa2 = truth.align(y, X_chain, "W")
    fullp2 = pd.period_range(ya2.index.min(), ya2.index.max(), freq="W-SAT")
    align2_info = {"first": str(ya2.index.min()), "last": str(ya2.index.max()), "n": int(len(ya2)),
                   "gaps": int(len(fullp2.difference(ya2.index))), "nan_y": int(ya2.isna().sum()), "nan_X": int(Xa2.isna().sum().sum())}
    print("aligned chained:", align2_info)
    assert align2_info["gaps"] == 0
    extended = run_battery(ya2, Xa2, "chained 2017-2026")

    # --- forward forecasts (primary sample, winning configuration per horizon) -------------------
    # The aligned Xa is intersected with y, so it stops at the last published ILI week. For the
    # forward step we need the full GT frame on the same W-SAT period index (one week beyond y).
    Xp = X.astype(float).copy()
    Xp.index = Xp.index.to_period("W-SAT")
    Xp = Xp.groupby(level=0).mean().sort_index()
    forecasts = {}
    for h in HORIZONS:
        cell = primary[h]
        tab = cell["tables"]["full"]
        winner = tab["RMSE"].drop(index=[m for m in tab.index if "not known" in m]).idxmin()
        best_gt = tab.loc[cell["gt_models"], "RMSE"].idxmin()
        specs = {
            "ar_levels": dict(model="ar", use_gt=False, transform=None),
            "ar_diff": dict(model="ar", use_gt=False, transform="diff"),
            "ols+GT_levels": dict(model="ols", use_gt=True, transform=None),
            "ols+GT_diff": dict(model="ols", use_gt=True, transform="diff"),
        }
        fc = {}
        # winner + best GT model, plus the target-only references so the GT point can be read against them
        for name in dict.fromkeys([winner, best_gt, "ar_levels", "naive_lag", "seasonal_naive"]):
            r = cell["results"][name]
            errs = r.errors.values
            q_all = ev.conformal_interval(errs, alpha=0.1)
            q_recent = ev.conformal_interval(errs, alpha=0.1, recent=52)
            if name in specs:
                f = forecast_pub_lag(ya, Xp, h=h, preprocess=pp.raw_pipeline, **specs[name],
                                     **{k: DESIGN[k] for k in ("p", "xlags", "period", "fourier_k")})
            else:
                # naive_lag / seasonal_naive: closed-form from the aligned truth
                last_y = ya.index.max()
                origin = last_y + 1
                target = origin + h
                if name == "naive_lag":
                    point = float(ya.loc[last_y])
                else:
                    point = float(ya.loc[target - 52])
                f = {"origin": str(origin), "target_date": str(target), "point": point, "model": name, "n_train": None,
                     "last_published_y": {"date": str(last_y), "value": float(ya.loc[last_y])}, "features": None}
            f.update({"q90_all": q_all, "q90_recent52": q_recent,
                      "lo90_all": max(0.0, f["point"] - q_all), "hi90_all": f["point"] + q_all,
                      "lo90_recent52": max(0.0, f["point"] - q_recent), "hi90_recent52": f["point"] + q_recent,
                      "n_backtest_errors": int(len(errs))})
            fc[name] = f
        # library forecast_next for reference (origin at last published y, so target is one week earlier)
        try:
            spec = specs[best_gt]
            lib = ev.forecast_next(ya, Xp, h=h, preprocess=pp.raw_pipeline, y_known_at_origin=False,
                                   **{k: DESIGN[k] for k in ("p", "xlags", "period", "fourier_k")}, **spec)
            lib_ref = {"origin": lib["origin"], "target_date": lib["target_date"], "point": lib["point"], "note": lib.get("note")}
        except Exception as e:  # noqa
            lib_ref = {"error": str(e)[:200]}
        forecasts[h] = {"winner": winner, "best_gt": best_gt, "forecasts": fc, "library_forecast_next_ref": lib_ref}
        print(f"\n--- forward h={h}: winner={winner} best_gt={best_gt}")
        for name, f in fc.items():
            print(f"  {name}: origin {f['origin']} -> target {f['target_date']}  point={f['point']:.3f}  "
                  f"90% all=[{f['lo90_all']:.2f},{f['hi90_all']:.2f}] recent52=[{f['lo90_recent52']:.2f},{f['hi90_recent52']:.2f}]")
            if f.get("features"):
                print("    top features:", [(k, round(v, 3)) for k, v in f["features"][:6]])
        print("  library forecast_next ref:", lib_ref)

    # --- markdown tables to stdout for pasting ---------------------------------------------------
    print("\n\n##### MARKDOWN TABLES #####")
    for label, batt in (("primary (today 5-y, 2021-09 → 2026-09)", primary), ("chained (2017-01 → 2026-09)", extended)):
        for h in HORIZONS:
            for view in ("full", "excl_covid"):
                tab = batt[h]["tables"][view]
                print(f"\n**{label} — h={h}, {view}** (bench = {batt[h]['bench']}, n = {int(tab['n'].iloc[0])})\n")
                print(md_table(tab, batt[h]["bench"]))
                print("verdict:", json.dumps(batt[h]["verdicts"][view]))

    # --- JSON ----------------------------------------------------------------------------------
    reqs = my_requests(KWS, TF_PRIMARY) + my_requests(KWS, TF_EXT)
    secs = time.time() - T0
    payload = {
        "id": ID, "question": "How high will US influenza-like-illness activity (ILI %) be in 2 and 4 weeks?",
        "approach": "A — Choi–Varian minimal (5 hand-picked keywords, OLS ARX, raw pipeline)",
        "truth": truth_info, "gt": gt_info, "aligned": align_info, "design": DESIGN, "horizons": list(HORIZONS),
        "exclude": list(EXCL),
        "tables": {str(h): {view: records(primary[h]["tables"][view]) for view in ("full", "excl_covid")} for h in HORIZONS},
        "bench": {str(h): primary[h]["bench"] for h in HORIZONS},
        "verdicts": {str(h): primary[h]["verdicts"] for h in HORIZONS},
        "forecast": {str(h): forecasts[h] for h in HORIZONS},
        "extended": {"gt": ext_info, "aligned": align2_info,
                     "tables": {str(h): {view: records(extended[h]["tables"][view]) for view in ("full", "excl_covid")} for h in HORIZONS},
                     "bench": {str(h): extended[h]["bench"] for h in HORIZONS},
                     "verdicts": {str(h): extended[h]["verdicts"] for h in HORIZONS}},
        "queries": KWS, "requests": reqs, "requests_note": "total gt.catalog() vintages for these 5 keywords x 2 timeframes (5 primary pulled in the probe run, 5 chained extension); cached re-runs add none",
        "seconds": round(secs, 1),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    print(f"\nwrote {OUT_JSON}; requests this run = {reqs}; elapsed {secs:.0f}s")


if __name__ == "__main__":
    main()
