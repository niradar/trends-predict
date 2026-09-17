"""Keyless ground-truth series. Every fetch is cached under ``data/truth/<name>.csv`` with a
``<name>.meta.json`` sidecar so experiments are reproducible and the user can inspect them.

Sources (all verified reachable without keys, see research/data-access.md):
  bls_series          BLS Public API v1 (monthly; 25 requests/day, ≤10 years/request)
  fluview_ili         CMU Delphi Epidata FluView (weekly ILI %, national/regions/states)
  fluview_clinical    Delphi FluView clinical lab positivity (weekly)
  yahoo               Yahoo Finance via yfinance (daily -> resampled)
  wikipedia_pageviews Wikimedia REST (daily since 2015-07)
  csv_series          any user CSV with a date column and a value column
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from . import TRUTH_DIR

_UA = {"User-Agent": "trends-predict-research/0.1 (local research tool)"}
Freq = Literal["D", "W", "M"]


def _save(name: str, s: pd.Series, meta: dict) -> pd.Series:
    s = s.sort_index()
    s.index.name = "date"
    s.name = name
    s.to_csv(TRUTH_DIR / f"{name}.csv", encoding="utf-8")
    meta = {**meta, "name": name, "rows": int(len(s)), "first": str(s.index.min()), "last": str(s.index.max()),
            "retrieved_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}
    (TRUTH_DIR / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return s


def _load(name: str, max_age_days: float | None) -> pd.Series | None:
    p = TRUTH_DIR / f"{name}.csv"
    m = TRUTH_DIR / f"{name}.meta.json"
    if not (p.exists() and m.exists()):
        return None
    if max_age_days is not None:
        meta = json.loads(m.read_text(encoding="utf-8"))
        ts = datetime.strptime(meta["retrieved_utc"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - ts > timedelta(days=max_age_days):
            return None
    df = pd.read_csv(p, index_col=0, parse_dates=True, encoding="utf-8")
    s = df.iloc[:, 0]
    s.name = name
    return s


# --------------------------------------------------------------------------------------
# BLS
# --------------------------------------------------------------------------------------

BLS_SERIES = {
    "UNRATE": ("LNS14000000", "Unemployment rate, 16+, seasonally adjusted, %"),
    "UNRATE_NSA": ("LNU04000000", "Unemployment rate, 16+, not seasonally adjusted, %"),
    "PAYEMS": ("CES0000000001", "Total nonfarm payrolls, SA, thousands"),
    "CPIAUCNS": ("CUUR0000SA0", "CPI-U all items, NSA, index 1982-84=100"),
    "CPIAUCSL": ("CUSR0000SA0", "CPI-U all items, SA"),
    "UNEMPLOYED": ("LNS13000000", "Unemployment level, SA, thousands"),
    "JOBLESS_27W": ("LNS13008636", "Unemployed 27 weeks and over, SA, thousands"),
}


def bls_series(series_id: str, start_year: int = 2004, end_year: int | None = None,
               name: str | None = None, max_age_days: float | None = 1.0) -> pd.Series:
    """Monthly BLS series as a Series indexed by month start. ``series_id`` may be a key of
    BLS_SERIES (e.g. 'UNRATE') or a raw BLS id."""
    sid, desc = BLS_SERIES.get(series_id, (series_id, series_id))
    name = name or f"bls_{series_id}"
    cached = _load(name, max_age_days)
    if cached is not None:
        return cached
    end_year = end_year or datetime.now().year
    rows = []
    y0 = start_year
    while y0 <= end_year:
        y1 = min(y0 + 9, end_year)
        r = requests.post("https://api.bls.gov/publicAPI/v1/timeseries/data/",
                          json={"seriesid": [sid], "startyear": str(y0), "endyear": str(y1)}, timeout=60)
        r.raise_for_status()
        j = r.json()
        if j.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS error: {j.get('message')}")
        for s in j["Results"]["series"]:
            for d in s["data"]:
                if d["period"].startswith("M") and d["period"] != "M13":
                    try:
                        rows.append((f"{d['year']}-{d['period'][1:]}-01", float(d["value"])))
                    except (TypeError, ValueError):
                        continue  # BLS uses '-' for unavailable months
        y0 = y1 + 1
    s = pd.Series({pd.Timestamp(k): v for k, v in rows}).sort_index()
    return _save(name, s, {"source": "BLS API v1", "series_id": sid, "description": desc, "freq": "M"})


# --------------------------------------------------------------------------------------
# Delphi Epidata (CDC FluView)
# --------------------------------------------------------------------------------------

def _epiweek_to_date(ew: int) -> pd.Timestamp:
    """MMWR epiweek (YYYYWW) -> Saturday week-ending date."""
    from epiweeks import Week
    w = Week(ew // 100, ew % 100)
    return pd.Timestamp(w.enddate())


def fluview_ili(region: str = "nat", start_epiweek: int = 200440, end_epiweek: int | None = None,
                field: str = "wili", max_age_days: float | None = 1.0) -> pd.Series:
    """Weekly ILI % (weighted by default) from CDC FluView via Delphi. Index = week-ending Saturday."""
    if end_epiweek is None:
        from epiweeks import Week
        w = Week.thisweek()
        end_epiweek = w.year * 100 + w.week
    name = f"fluview_{field}_{region}"
    cached = _load(name, max_age_days)
    if cached is not None:
        return cached
    r = requests.get("https://api.delphi.cmu.edu/epidata/fluview/",
                     params={"regions": region, "epiweeks": f"{start_epiweek}-{end_epiweek}"}, timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("result") != 1:
        raise RuntimeError(f"Delphi fluview: {j.get('message')}")
    rows = {_epiweek_to_date(e["epiweek"]): e[field] for e in j["epidata"] if e.get(field) is not None}
    s = pd.Series(rows).sort_index()
    return _save(name, s, {"source": "Delphi Epidata fluview", "region": region, "field": field, "freq": "W-SAT",
                           "note": "ILINet; off-season weeks (epiweeks 21-39) may be missing in older years"})


def fluview_clinical(region: str = "nat", start_epiweek: int = 201540, end_epiweek: int | None = None,
                     field: str = "percent_positive", max_age_days: float | None = 1.0) -> pd.Series:
    if end_epiweek is None:
        from epiweeks import Week
        w = Week.thisweek()
        end_epiweek = w.year * 100 + w.week
    name = f"fluview_clinical_{field}_{region}"
    cached = _load(name, max_age_days)
    if cached is not None:
        return cached
    r = requests.get("https://api.delphi.cmu.edu/epidata/fluview_clinical/",
                     params={"regions": region, "epiweeks": f"{start_epiweek}-{end_epiweek}"}, timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("result") != 1:
        raise RuntimeError(f"Delphi fluview_clinical: {j.get('message')}")
    rows = {_epiweek_to_date(e["epiweek"]): e[field] for e in j["epidata"] if e.get(field) is not None}
    s = pd.Series(rows).sort_index()
    return _save(name, s, {"source": "Delphi Epidata fluview_clinical", "region": region, "field": field, "freq": "W-SAT"})


# --------------------------------------------------------------------------------------
# Israel CBS series API (keyless)
# --------------------------------------------------------------------------------------

CBS_LIST = "https://apis.cbs.gov.il/series/data/list?id={sid}&format=json&download=false&PageSize=1000"
CBS_SERIES = {
    # LFS unemployment rate, ages 15+, total population, seasonally adjusted, % — old definition
    # (2012-01..2025-12, 2008-census weights) and new definition (2025-01.., 2022-census weights)
    "UNEMP_IL_SA": (491094, 41097, "Unemployment rate, 15+, SA, % (LFS monthly; old+new definition spliced)"),
}


def _cbs_one(sid: int) -> tuple[pd.Series, dict]:
    r = requests.get(CBS_LIST.format(sid=sid), timeout=90)
    r.raise_for_status()
    s = r.json()["DataSet"]["Series"][0]
    obs = {pd.Timestamp(o["TimePeriod"] + "-01"): float(o["Value"]) for o in s["obs"] if o.get("Value") is not None}
    meta = {"sid": s["id"], "update": s.get("update"), "data": s["data"]["name"], "unit": s["unit"]["name"]}
    return pd.Series(obs).sort_index(), meta


def cbs_series(key_or_sid: str | int, new_sid: int | None = None, name: str | None = None,
               max_age_days: float | None = 1.0) -> pd.Series:
    """Israel CBS monthly series. ``key_or_sid`` = a key of CBS_SERIES or a raw series id; when
    two ids are given (old/new definition) the new one overrides on the overlap and the splice
    gap is recorded in the metadata."""
    if isinstance(key_or_sid, str) and key_or_sid in CBS_SERIES:
        old_sid, new_sid, desc = CBS_SERIES[key_or_sid]
        name = name or f"cbs_{key_or_sid.lower()}"
    else:
        old_sid, desc = int(key_or_sid), f"CBS series {key_or_sid}"
        name = name or f"cbs_{old_sid}"
    cached = _load(name, max_age_days)
    if cached is not None:
        return cached
    old, m_old = _cbs_one(old_sid)
    meta = {"source": "CBS Israel series API", "endpoint": CBS_LIST, "description": desc, "freq": "M", "series": {"old": m_old}}
    s = old
    if new_sid:
        new, m_new = _cbs_one(new_sid)
        overlap = old.index.intersection(new.index)
        diff = (new.loc[overlap] - old.loc[overlap]) if len(overlap) else pd.Series(dtype=float)
        s = pd.concat([old.loc[old.index < new.index.min()], new]).sort_index()
        meta.update({"series": {"old": m_old, "new": m_new}, "splice_date": str(new.index.min().date()),
                     "overlap_months": int(len(overlap)),
                     "overlap_new_minus_old_mean": round(float(diff.mean()), 3) if len(overlap) else None,
                     "overlap_new_minus_old_max_abs": round(float(diff.abs().max()), 3) if len(overlap) else None})
    return _save(name, s, meta)


# --------------------------------------------------------------------------------------
# Yahoo Finance
# --------------------------------------------------------------------------------------

def yahoo(ticker: str, start: str = "2004-01-01", end: str | None = None, field: str = "Close",
          freq: Freq | None = None, max_age_days: float | None = 1.0) -> pd.Series:
    """Daily prices via yfinance; optionally resampled to W (Sunday-ending) or M (month start)
    using the last observation in each period."""
    name = f"yahoo_{ticker.replace('^', '').replace('=', '_').replace('-', '_')}_{field}_{freq or 'D'}"
    cached = _load(name, max_age_days)
    if cached is not None:
        return cached
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned nothing for {ticker}")
    col = df[field]
    s = col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col
    s = s.astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    if freq:
        # Weekly buckets end on Saturday so a Mon–Fri trading week lands in the same Sun–Sat
        # period as Google's Sunday-start week; a Sunday label would push the Friday close into
        # the *next* bucket and give GT a one-week look-ahead (experiments/03-Q3-B).
        rule = {"W": "W-SAT", "M": "MS", "D": "D"}[freq]
        s = s.resample(rule).last().dropna()
        # drop the current, incomplete period (its 'last close' is not the period's close)
        per = s.index[-1].to_period({"W": "W-SAT", "M": "M", "D": "D"}[freq])
        if per.end_time > pd.Timestamp.now():
            s = s.iloc[:-1]
    return _save(name, s, {"source": "Yahoo Finance (yfinance)", "ticker": ticker, "field": field, "freq": freq or "D"})


# --------------------------------------------------------------------------------------
# Wikipedia pageviews
# --------------------------------------------------------------------------------------

def wikipedia_pageviews(article: str, project: str = "en.wikipedia", start: str = "20150701",
                        end: str | None = None, freq: Freq = "W", max_age_days: float | None = 1.0) -> pd.Series:
    """Daily user pageviews aggregated to D/W/M. Useful as attention ground truth or proxy."""
    end = end or datetime.now().strftime("%Y%m%d")
    name = f"wiki_{project.split('.')[0]}_{article.replace('/', '_')}_{freq}"
    cached = _load(name, max_age_days)
    if cached is not None:
        return cached
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/user/"
           f"{requests.utils.quote(article, safe='')}/daily/{start}/{end}")
    r = requests.get(url, headers=_UA, timeout=60)
    r.raise_for_status()
    items = r.json().get("items", [])
    s = pd.Series({pd.Timestamp(i["timestamp"][:8]): i["views"] for i in items}).sort_index().astype(float)
    if freq != "D":
        s = s.resample({"W": "W-SAT", "M": "MS"}[freq]).sum()  # Sun–Sat weeks, same as Google Trends
    return _save(name, s, {"source": "Wikimedia pageviews", "article": article, "project": project, "freq": freq})


# --------------------------------------------------------------------------------------
# generic CSV
# --------------------------------------------------------------------------------------

def csv_series(path: str | Path, date_col: str | int = 0, value_col: str | int = 1, name: str | None = None,
               date_format: str | None = None) -> pd.Series:
    """Load a user-provided CSV (e.g. downloaded from a statistics office) into the truth cache."""
    p = Path(path)
    df = pd.read_csv(p, encoding="utf-8-sig")
    dc = df.columns[date_col] if isinstance(date_col, int) else date_col
    vc = df.columns[value_col] if isinstance(value_col, int) else value_col
    idx = pd.to_datetime(df[dc], format=date_format, errors="coerce")
    s = pd.Series(pd.to_numeric(df[vc], errors="coerce").values, index=idx).dropna().sort_index()
    name = name or f"csv_{p.stem}"
    return _save(name, s, {"source": f"csv:{p.name}", "date_col": str(dc), "value_col": str(vc)})


# --------------------------------------------------------------------------------------
# alignment helper
# --------------------------------------------------------------------------------------

def align(y: pd.Series, X: pd.DataFrame, freq: Freq, how: str = "mean") -> tuple[pd.Series, pd.DataFrame]:
    """Bring target and GT frame to a common period index.

    Google weekly points are Sunday-*starting* weeks; FluView weeks end Saturday; we map every
    timestamp to its period (W = week containing the date, M = month) and aggregate GT by
    ``how`` (mean by default; use 'last' for stock-like series).
    """
    if freq == "M":
        yi = y.copy(); yi.index = yi.index.to_period("M")
        Xi = X.copy(); Xi.index = Xi.index.to_period("M")
    elif freq == "W":
        # Anchor weeks on Saturday-ending (MMWR-style) so a Google Sunday-start week and a
        # CDC Saturday-end week for the same 7 days map to the same period.
        yi = y.copy(); yi.index = yi.index.to_period("W-SAT")
        Xi = X.copy(); Xi.index = Xi.index.to_period("W-SAT")
    else:
        yi = y.copy(); yi.index = yi.index.to_period("D")
        Xi = X.copy(); Xi.index = Xi.index.to_period("D")
    yi = yi.groupby(level=0).agg(how if how != "mean" else "mean")
    Xi = Xi.groupby(level=0).agg(how)
    idx = yi.index.intersection(Xi.index)
    return yi.loc[idx].sort_index(), Xi.loc[idx].sort_index()
