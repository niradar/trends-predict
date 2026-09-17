"""Runner for the trends-forecast skill.

    python .claude/skills/trends-forecast/scripts/run_spec.py specs/<id>.json          # full run
    python .claude/skills/trends-forecast/scripts/run_spec.py --probe specs/<id>.json  # queries only
    python .claude/skills/trends-forecast/scripts/run_spec.py --truth specs/<id>.json  # truth only
    python .claude/skills/trends-forecast/scripts/run_spec.py --truth-check '{"kind":"fluview","region":"ny"}'  # no spec needed; reports staleness
    python .claude/skills/trends-forecast/scripts/run_spec.py --suggest "Kamala Harris" "אבטלה"
    python .claude/skills/trends-forecast/scripts/run_spec.py --related unemployment --geo US
    python .claude/skills/trends-forecast/scripts/run_spec.py --salience specs/<id>.json  # discrete contests: attention share, NOT a forecast
    python .claude/skills/trends-forecast/scripts/run_spec.py --rerender <id> --headline "..." --answer "..." --confidence medium
    python .claude/skills/trends-forecast/scripts/run_spec.py --rerender <id> --answer-file answer.txt   # long text from a file
                                                              # rewrite outputs/<id>.html with your answer text (no re-run)
    python .claude/skills/trends-forecast/scripts/run_spec.py --open outputs/<id>.html

The full run prints a JSON summary (verdicts, chosen model, forecast, interval, requests used)
and writes outputs/<id>.json and outputs/<id>.html. Always run from the repository root with
PYTHONIOENCODING=utf-8 (Hebrew).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
# The library is edited often; a stale .pyc with matching mtime/size once executed old bytecode
# (validation V2-opus). Never write bytecode from the runner and drop any cache that exists.
sys.dont_write_bytecode = True
import shutil  # noqa: E402
shutil.rmtree(ROOT / "src" / "trends_predict" / "__pycache__", ignore_errors=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trends_predict import gt, truth  # noqa: E402
from trends_predict.pipeline import run_spec_file, truth_from_spec, _defaults  # noqa: E402


def probe(spec_path: str) -> None:
    """Pull the spec's queries (cached), align with the truth and print per-query diagnostics so
    the query set can be fixed *before* a full run: non-zero share, correlation with target
    changes at lags 0..2, and how many queries survive the privacy-zero filter."""
    spec = _defaults(json.loads(Path(spec_path).read_text(encoding="utf-8")))
    y_raw, desc = truth_from_spec(spec["truth"])
    if spec["freq"].upper()[0] == "W" and spec.get("history_years", 5) > 5:  # same sample as the full run
        import pandas as pd
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        start = (pd.Timestamp.today() - pd.DateOffset(years=int(spec["history_years"]))).strftime("%Y-%m-%d")
        X_raw, info = gt.fetch_many_stitched(spec["queries"], start, end, geo=spec["geo"], cat=spec["cat"], gprop=spec["gprop"])
    else:
        X_raw, info = gt.fetch_many(spec["queries"], geo=spec["geo"], timeframe=spec["timeframe"], cat=spec["cat"], gprop=spec["gprop"])
    y, X = truth.align(y_raw, X_raw, spec["freq"].upper()[0])
    # correlate GT changes with target *changes* for persistent series (transform diff), and with the
    # target itself when the spec models levels/rates (transform null) — validation V6
    dy = y.diff() if spec.get("transform", "diff" if spec["freq"].upper()[0] == "M" else None) == "diff" else y
    rows = []
    for c in X.columns:
        dx = np.log1p(X[c]).diff()
        r = {"query": c, "nonzero_share": round(float((X[c] > 0).mean()), 2), "mean": round(float(X[c].mean()), 1)}
        for lag in (0, 1, 2):
            r[f"corr_dY_lag{lag}"] = round(float(dx.shift(lag).corr(dy)), 2)
        rows.append(r)
    out = {"truth": desc, "truth_range": [str(y_raw.index.min().date()), str(y_raw.index.max().date())], "aligned_periods": int(len(y)),
           "kept": len(X.columns), "dropped": info["dropped"], "failed": info["failed"], "requests_used": gt.REQUEST_COUNT,
           "note": ("lag0 matters for nowcasts, lag1+ for h>=1 forecasts. Strong correlations can be a single shock "
                    "(e.g. 2020) rather than signal — the backtest, not this table, decides."),
           "queries": sorted(rows, key=lambda r: -max(abs(r[k]) for k in ("corr_dY_lag0", "corr_dY_lag1", "corr_dY_lag2") if r[k] == r[k]) if any(r[k] == r[k] for k in ("corr_dY_lag0", "corr_dY_lag1", "corr_dY_lag2")) else 0)}
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


def check_truth(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    y, desc = truth_from_spec(spec["truth"])
    import pandas as pd
    age_days = (pd.Timestamp.today() - pd.Timestamp(y.index.max())).days
    print(json.dumps({"truth": desc, "n": int(len(y)), "first": str(y.index.min().date()), "last": str(y.index.max().date()),
                      "days_since_last": age_days, "stale": bool(age_days > 60),
                      "tail": {str(k.date()): float(v) for k, v in y.tail(6).items()}}, ensure_ascii=False, indent=1))


def main(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    if argv[0] == "--probe":
        probe(argv[1]); return
    if argv[0] == "--truth":
        check_truth(argv[1]); return
    if argv[0] == "--suggest":
        for term in argv[1:]:
            print(term, "->", json.dumps(gt.suggestions(term)[:5], ensure_ascii=False))
        return
    if argv[0] == "--related":
        geo = argv[argv.index("--geo") + 1] if "--geo" in argv else ""
        term = argv[1]
        rq = gt.related_queries(term, geo=geo)
        print(json.dumps({"top": (rq.get("top") or [])[:20], "rising": (rq.get("rising") or [])[:10]}, ensure_ascii=False, indent=1))
        return
    if argv[0] == "--salience":
        from trends_predict.pipeline import salience
        spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        print(json.dumps(salience(spec), ensure_ascii=False, indent=1, default=str))
        return
    if argv[0] == "--rerender":
        # --rerender <id> --headline "..." --answer "..." --confidence low|medium|high|none
        from trends_predict.pipeline import rerender
        id_ = argv[1]
        def _opt(flag):
            return argv[argv.index(flag) + 1] if flag in argv else None
        answer = _opt("--answer")
        if _opt("--answer-file"):  # long multi-sentence answers: read from a UTF-8 text file
            answer = Path(_opt("--answer-file")).read_text(encoding="utf-8").strip()
        caveats = None
        if _opt("--caveats-file"):  # one caveat per line
            caveats = [ln.strip() for ln in Path(_opt("--caveats-file")).read_text(encoding="utf-8").splitlines() if ln.strip()]
        print(json.dumps(rerender(id_, headline=_opt("--headline"), answer_text=answer, confidence=_opt("--confidence"),
                                  caveats=caveats), ensure_ascii=False))
        return
    if argv[0] == "--truth-check":
        # --truth-check '{"kind":"fluview","region":"ny"}'  -> is the series alive, without a spec
        y, desc = truth_from_spec(json.loads(argv[1]))
        import pandas as pd
        age_days = (pd.Timestamp.today() - pd.Timestamp(y.index.max())).days
        print(json.dumps({"truth": desc, "n": int(len(y)), "first": str(y.index.min().date()), "last": str(y.index.max().date()),
                          "days_since_last": age_days, "stale": bool(age_days > 60)}, ensure_ascii=False, indent=1))
        return
    if argv[0] == "--open":
        os.startfile(str(ROOT / argv[1]))  # Windows: open the artifact in the default browser
        return
    run_spec_file(argv[0])


if __name__ == "__main__":
    main(sys.argv[1:])
