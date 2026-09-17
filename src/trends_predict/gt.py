"""Cached, vintage-stamped Google Trends access.

Every pull is written to ``data/trends/<slug>/`` as
    meta.json                 request parameters + list of vintages
    v_YYYYMMDDTHHMMSSZ.csv    one CSV per download (a *vintage*)
and appended to ``data/trends/catalog.jsonl`` so the user can browse what exists.

Design rules (see research/data-access.md):
* values are relative per request -> ``fetch_many`` rescales batches with a shared anchor;
* one frequency per request -> caller picks the timeframe that yields the target frequency;
* repeated downloads are vintages, never overwritten;
* rate limiting -> exponential backoff, then ``TrendsUnavailable`` (never silent partial data).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import TRENDS_DIR, QUERIES_DIR

MAX_KEYWORDS_PER_REQUEST = 5
_BACKOFF_SECONDS = (5, 10, 20, 40, 80)
_MIN_GAP_SECONDS = 0.6  # polite spacing between live calls


class TrendsUnavailable(RuntimeError):
    """Raised when Google Trends could not be reached after all retries."""


# --------------------------------------------------------------------------------------
# request description / cache addressing
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GTRequest:
    keywords: tuple[str, ...]
    geo: str = ""
    timeframe: str = "today 5-y"
    cat: int = 0
    gprop: str = ""  # '' web, 'news', 'images', 'youtube', 'froogle'

    @staticmethod
    def make(keywords: Sequence[str] | str, geo: str = "", timeframe: str = "today 5-y",
             cat: int = 0, gprop: str = "") -> "GTRequest":
        if isinstance(keywords, str):
            keywords = [keywords]
        # Google strips quote marks; Hebrew abbreviations with gershayim (חל"ת) come back as a broken
        # column, so normalise them here (experiments/05-Q5-A). Topic ids are untouched.
        kws = tuple(_sanitize(k) for k in keywords if k and k.strip())
        kws = tuple(k for k in kws if k)
        if not kws:
            raise ValueError("at least one keyword is required")
        if len(kws) > MAX_KEYWORDS_PER_REQUEST:
            raise ValueError(f"Google Trends accepts at most {MAX_KEYWORDS_PER_REQUEST} keywords per request")
        return GTRequest(kws, geo.strip().upper(), timeframe.strip(), int(cat), gprop.strip())

    def key(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    def slug(self) -> str:
        first = _slugify(self.keywords[0])[:40]
        geo = self.geo or "WW"
        tf = _slugify(self.timeframe)[:24]
        extra = f"_cat{self.cat}" if self.cat else ""
        extra += f"_{self.gprop}" if self.gprop else ""
        n = f"_x{len(self.keywords)}" if len(self.keywords) > 1 else ""
        return f"{geo}__{first}{n}__{tf}{extra}__{self.key()}"

    def directory(self) -> Path:
        return TRENDS_DIR / self.slug()


def _sanitize(k: str) -> str:
    k = k.strip()
    if k.startswith("/"):
        return k
    return re.sub(r"[\"״׳']", "", k).strip()


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "q"


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------------------
# live client with backoff
# --------------------------------------------------------------------------------------

_client = None
_last_call = 0.0
REQUEST_COUNT = 0  # live Google Trends calls made by this process (cache hits excluded)


def _get_client():
    global _client
    if _client is None:
        from trendspy import Trends  # imported lazily so the cache works offline
        _client = Trends()
    return _client


def _throttle():
    global _last_call
    gap = time.time() - _last_call
    if gap < _MIN_GAP_SECONDS:
        time.sleep(_MIN_GAP_SECONDS - gap)
    _last_call = time.time()


def _with_backoff(fn, what: str):
    global REQUEST_COUNT
    last_err: Exception | None = None
    for attempt, wait in enumerate((0,) + _BACKOFF_SECONDS):
        if wait:
            time.sleep(wait)
        try:
            _throttle()
            REQUEST_COUNT += 1
            return fn()
        except Exception as e:  # noqa: BLE001 - we re-raise a typed error below
            last_err = e
            msg = str(e)
            # trendspy raises generic exceptions; look for throttling hints
            if not any(tok in msg for tok in ("429", "Too Many", "rate", "quota", "timed out", "Connection")):
                if attempt >= 1:
                    break
    raise TrendsUnavailable(f"Google Trends unavailable for {what}: {type(last_err).__name__}: {last_err}")


# --------------------------------------------------------------------------------------
# public API: interest over time
# --------------------------------------------------------------------------------------

def fetch_interest(keywords: Sequence[str] | str, geo: str = "", timeframe: str = "today 5-y",
                   cat: int = 0, gprop: str = "", *, force: bool = False,
                   drop_partial: bool = True) -> pd.DataFrame:
    """Return interest-over-time for ≤5 keywords (jointly normalised), using the cache.

    ``force=True`` downloads a new vintage even if one exists (use on a later day to study
    Google's retrieval noise). The returned frame has a DatetimeIndex and one column per
    keyword; the ``isPartial`` row(s) are dropped by default.
    """
    req = GTRequest.make(keywords, geo, timeframe, cat, gprop)
    d = req.directory()
    meta_path = d / "meta.json"
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("vintages"):
            df = _read_vintage(d / meta["vintages"][-1]["file"])
            return _finish(df, req, drop_partial)

    def _call():
        return _get_client().interest_over_time(list(req.keywords), geo=req.geo,
                                                 timeframe=req.timeframe, cat=req.cat, gprop=req.gprop)

    raw = _with_backoff(_call, f"{req.keywords} geo={req.geo!r} tf={req.timeframe!r}")
    if raw is None or len(raw) == 0:
        raise TrendsUnavailable(f"empty response for {req.keywords} (geo={req.geo!r}); try broader terms or a topic id")
    raw = raw.copy()
    raw.index.name = "date"
    _save_vintage(req, raw)
    return _finish(raw, req, drop_partial)


def _finish(df: pd.DataFrame, req: GTRequest, drop_partial: bool) -> pd.DataFrame:
    df = df.copy()
    if "isPartial" in df.columns:
        if drop_partial:
            part = df["isPartial"].astype(str).str.lower().isin(["true", "1"])
            df = df.loc[~part]
        df = df.drop(columns=["isPartial"])
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    # keep requested order, coerce numeric
    cols = [c for c in req.keywords if c in df.columns] or list(df.columns)
    df = df[cols].apply(pd.to_numeric, errors="coerce")
    df.attrs["gt_request"] = asdict(req)
    return df


def _save_vintage(req: GTRequest, raw: pd.DataFrame) -> Path:
    d = req.directory()
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
        "request": asdict(req), "slug": req.slug(), "vintages": []}
    tag = _now_tag()
    fname = f"v_{tag}.csv"
    raw.to_csv(d / fname, encoding="utf-8")
    vintage = {"file": fname, "retrieved_utc": tag, "rows": int(len(raw)),
               "first": str(raw.index.min()), "last": str(raw.index.max())}
    meta["vintages"].append(vintage)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    with (TRENDS_DIR / "catalog.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"slug": req.slug(), **asdict(req), **vintage}, ensure_ascii=False) + "\n")
    return d / fname


def _read_vintage(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True, encoding="utf-8")


def load_vintages(keywords: Sequence[str] | str, geo: str = "", timeframe: str = "today 5-y",
                  cat: int = 0, gprop: str = "") -> list[pd.DataFrame]:
    """All cached vintages of a request (oldest first). Empty list if never pulled."""
    req = GTRequest.make(keywords, geo, timeframe, cat, gprop)
    meta_path = req.directory() / "meta.json"
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return [_finish(_read_vintage(req.directory() / v["file"]), req, True) for v in meta["vintages"]]


def vintage_dispersion(vintages: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Rivera-style retrieval noise: per-date mean and std across vintages (None if <2)."""
    if len(vintages) < 2:
        return None
    stacked = pd.concat(vintages, keys=range(len(vintages)), names=["vintage", "date"])
    g = stacked.groupby(level="date")
    out = pd.concat({"mean": g.mean(), "std": g.std(ddof=1)}, axis=1)
    return out


# --------------------------------------------------------------------------------------
# many keywords: batch + anchor rescaling
# --------------------------------------------------------------------------------------

def fetch_many(keywords: Sequence[str], geo: str = "", timeframe: str = "today 5-y", cat: int = 0,
               gprop: str = "", anchor: str | None = None, *, mode: str = "individual",
               force: bool = False, min_nonzero_share: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """Fetch any number of keywords.

    mode='individual' (default, for regression features): every keyword is pulled **alone**,
        so each series keeps full 0–100 resolution. Models standardise features anyway, and a
        joint pull with one dominant term rounds the others to privacy zeros.
    mode='anchored' (for shares / composite indices): batches of 4 + ``anchor`` rescaled by the
        anchor ratio so all series are on one comparable scale (Eichenauer et al. 2022).

    Series whose non-zero share is below ``min_nonzero_share`` are dropped (privacy zeros).
    Returns (frame, info) with dropped keywords and per-batch scale factors.
    """
    kws = [k.strip() for k in keywords if k and k.strip()]
    kws = list(dict.fromkeys(kws))  # dedupe, keep order
    if not kws:
        raise ValueError("no keywords")
    info: dict = {"mode": mode, "anchor": None, "batches": [], "dropped": [], "scale": {}, "failed": []}

    if mode == "individual":
        frames = []
        for k in kws:
            try:
                frames.append(fetch_interest([k], geo, timeframe, cat, gprop, force=force))
            except TrendsUnavailable as e:
                info["failed"].append({"keyword": k, "error": str(e)[:200]})
        if not frames:
            raise TrendsUnavailable("all keyword pulls failed")
        out = pd.concat(frames, axis=1)
        out = out.loc[:, ~out.columns.duplicated()]
        return _zero_filter(out, info, min_nonzero_share, geo, timeframe, cat, gprop, None)

    anchor = (anchor or kws[0]).strip()
    info["anchor"] = anchor
    others = [k for k in kws if k != anchor]
    if not others:
        df = fetch_interest([anchor], geo, timeframe, cat, gprop, force=force)
        return df, info

    frames: list[pd.DataFrame] = []
    ref_anchor_mean: float | None = None
    for i in range(0, len(others), MAX_KEYWORDS_PER_REQUEST - 1):
        batch = [anchor] + others[i:i + MAX_KEYWORDS_PER_REQUEST - 1]
        df = fetch_interest(batch, geo, timeframe, cat, gprop, force=force)
        a = df[anchor].astype(float)
        a_mean = float(a[a > 0].mean()) if (a > 0).any() else float("nan")
        if ref_anchor_mean is None:
            ref_anchor_mean = a_mean
            scale = 1.0
        else:
            scale = ref_anchor_mean / a_mean if a_mean and a_mean == a_mean else 1.0
        info["batches"].append({"keywords": batch, "scale": scale})
        scaled = df.drop(columns=[anchor]) * scale if i > 0 else df * scale
        for c in scaled.columns:
            info["scale"][c] = scale
        frames.append(scaled)

    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return _zero_filter(out, info, min_nonzero_share, geo, timeframe, cat, gprop, anchor)


def _zero_filter(out: pd.DataFrame, info: dict, min_nonzero_share: float, geo, timeframe, cat, gprop, anchor):
    keep = []
    for c in out.columns:
        share = float((out[c].fillna(0) > 0).mean())
        if share >= min_nonzero_share:
            keep.append(c)
        else:
            info["dropped"].append({"keyword": c, "nonzero_share": round(share, 3)})
    out = out[keep]
    out.attrs["gt_many"] = {"geo": geo, "timeframe": timeframe, "cat": cat, "gprop": gprop, "anchor": anchor,
                            "mode": info.get("mode")}
    return out, info


# --------------------------------------------------------------------------------------
# long weekly history: chained 5-year windows rescaled on their overlap (Eichenauer et al.)
# --------------------------------------------------------------------------------------

def fetch_stitched(keyword: str, start: str, end: str, geo: str = "", cat: int = 0, gprop: str = "",
                   window_years: int = 5, overlap_weeks: int = 26, *, force: bool = False) -> pd.DataFrame:
    """Weekly series longer than Google's 5-year weekly cap.

    Pulls consecutive ≤5-year windows (each weekly) that overlap by ``overlap_weeks`` and rescales
    every window so that its mean over the overlap equals the previous window's mean there. The
    result is on the scale of the *last* (most recent) window. Metadata about the stitching is in
    ``df.attrs['stitch']``. Use only when >5 years of weekly data are genuinely needed.
    """
    s0, e0 = pd.Timestamp(start), pd.Timestamp(end)
    windows = []
    cur_end = e0
    while cur_end > s0:
        cur_start = max(s0, cur_end - pd.DateOffset(years=window_years) + pd.Timedelta(days=1))
        windows.append((cur_start, cur_end))
        if cur_start <= s0:
            break
        cur_end = cur_start + pd.Timedelta(weeks=overlap_weeks)
    windows = windows[::-1]  # oldest first
    frames = []
    for ws, we in windows:
        frames.append(fetch_interest([keyword], geo, f"{ws:%Y-%m-%d} {we:%Y-%m-%d}", cat, gprop, force=force))
    out = frames[-1].astype(float)  # most recent window keeps Google's scale
    scales = []
    for f in reversed(frames[:-1]):
        f = f.astype(float)
        ov = out.index.intersection(f.index)
        a, b = out.loc[ov, keyword], f.loc[ov, keyword]
        ratio = float(a[b > 0].mean() / b[b > 0].mean()) if (b > 0).any() and a[b > 0].mean() > 0 else 1.0
        scales.append({"window_end": str(f.index.max().date()), "overlap_weeks": int(len(ov)), "scale": ratio,
                       "overlap_corr": float(np.corrcoef(a, b)[0, 1]) if len(ov) > 3 else None})
        older = (f * ratio).loc[f.index < out.index.min()]
        out = pd.concat([older, out])
    out = out.sort_index()
    out.attrs["stitch"] = {"keyword": keyword, "windows": [(str(a.date()), str(b.date())) for a, b in windows], "scales": scales}
    return out


def fetch_many_stitched(keywords: Sequence[str], start: str, end: str, geo: str = "", cat: int = 0, gprop: str = "",
                        min_nonzero_share: float = 0.5, **kw) -> tuple[pd.DataFrame, dict]:
    info: dict = {"mode": "stitched", "anchor": None, "batches": [], "dropped": [], "scale": {}, "failed": [], "stitch": {}}
    frames = []
    for k in dict.fromkeys(_sanitize(k) for k in keywords if k and k.strip()):
        try:
            f = fetch_stitched(k, start, end, geo, cat, gprop, **kw)
            info["stitch"][k] = f.attrs.get("stitch")
            frames.append(f)
        except TrendsUnavailable as e:
            info["failed"].append({"keyword": k, "error": str(e)[:200]})
    if not frames:
        raise TrendsUnavailable("all stitched pulls failed")
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return _zero_filter(out, info, min_nonzero_share, geo, f"{start} {end}", cat, gprop, None)


# --------------------------------------------------------------------------------------
# query expansion (cached JSON)
# --------------------------------------------------------------------------------------

def _cached_json(kind: str, term: str, geo: str, timeframe: str, fn, force: bool = False):
    key = hashlib.sha1(f"{kind}|{term}|{geo}|{timeframe}".encode("utf-8")).hexdigest()[:10]
    path = QUERIES_DIR / f"{kind}__{(geo or 'WW')}__{_slugify(term)[:40]}__{key}.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    data = _with_backoff(fn, f"{kind}({term!r}, geo={geo!r})")
    payload = {"kind": kind, "term": term, "geo": geo, "timeframe": timeframe,
               "retrieved_utc": _now_tag(), "data": data}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return payload


def _frames_to_records(d) -> dict:
    out = {}
    if not isinstance(d, dict):
        return {"raw": str(d)}
    for k, v in d.items():
        if isinstance(v, pd.DataFrame):
            out[k] = v.to_dict(orient="records")
        else:
            out[k] = v
    return out


def related_queries(term: str, geo: str = "", timeframe: str = "today 5-y", *, force: bool = False) -> dict:
    """{'top': [{'query','value'}...], 'rising': [...]} for a seed term (cached)."""
    def _call():
        return _frames_to_records(_get_client().related_queries(term, geo=geo, timeframe=timeframe))
    return _cached_json("related_queries", term, geo, timeframe, _call, force)["data"]


def related_topics(term: str, geo: str = "", timeframe: str = "today 5-y", *, force: bool = False) -> dict:
    """{'top': [{'mid','title','type','value'}...], 'rising': [...]} (cached)."""
    def _call():
        return _frames_to_records(_get_client().related_topics(term, geo=geo, timeframe=timeframe))
    return _cached_json("related_topics", term, geo, timeframe, _call, force)["data"]


def suggestions(term: str, *, force: bool = False) -> list[dict]:
    """Entity lookup -> [{'mid','title','type'}...]. Topic ids are language-neutral."""
    def _call():
        s = _get_client().suggestions(term)
        return s.to_dict(orient="records") if isinstance(s, pd.DataFrame) else s
    return _cached_json("suggestions", term, "", "", _call, force)["data"]


def expand_queries(seeds: Iterable[str], geo: str = "", timeframe: str = "today 5-y",
                   max_per_seed: int = 15, include_rising: bool = False) -> list[str]:
    """Seed terms -> deduped list of seeds + their top related queries."""
    out: list[str] = []
    for s in seeds:
        out.append(s)
        try:
            rq = related_queries(s, geo, timeframe)
        except TrendsUnavailable:
            continue
        for row in (rq.get("top") or [])[:max_per_seed]:
            q = row.get("query")
            if q:
                out.append(q)
        if include_rising:
            for row in (rq.get("rising") or [])[:max_per_seed // 3]:
                q = row.get("query")
                if q:
                    out.append(q)
    return list(dict.fromkeys(o.strip() for o in out if o and o.strip()))


# --------------------------------------------------------------------------------------
# manual CSV import (Google Trends "Download CSV" button)
# --------------------------------------------------------------------------------------

def import_manual_csv(path: str | Path, keywords: Sequence[str], geo: str = "", timeframe: str = "manual",
                      cat: int = 0, gprop: str = "") -> pd.DataFrame:
    """Register a CSV exported from trends.google.com as a vintage in the cache."""
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    # Google exports start with a title line and a blank line before the header
    start = next(i for i, ln in enumerate(lines) if ln.lower().startswith(("week", "month", "day", "time", "date")))
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines[start:])))
    df = df.rename(columns={df.columns[0]: "date"}).set_index("date")
    df.columns = [re.sub(r":\s*\(.*\)$", "", c).strip() for c in df.columns]
    df = df.replace("<1", 0.5).apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(df.index)
    req = GTRequest.make(list(keywords), geo, timeframe, cat, gprop)
    _save_vintage(req, df)
    return _finish(df, req, True)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def timeframe_for(freq: str, start: str | None = None, end: str | None = None) -> str:
    """Pick a timeframe string that makes Google return the requested frequency.

    freq: 'M' monthly (>5y window), 'W' weekly (≤5y), 'D' daily (≤9 months).
    Explicit start/end (YYYY-MM-DD) override the presets.
    """
    if start and end:
        return f"{start} {end}"
    return {"M": "2004-01-01 " + datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "W": "today 5-y", "D": "today 3-m"}[freq.upper()[0]]


def catalog() -> pd.DataFrame:
    """Everything ever pulled, one row per vintage."""
    p = TRENDS_DIR / "catalog.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return pd.DataFrame(rows)
