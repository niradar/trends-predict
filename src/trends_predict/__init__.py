"""trends_predict — Google Trends forecasting toolkit behind the /trends-forecast skill.

Modules
-------
gt          cached, vintage-stamped Google Trends access (trendspy) + query expansion
truth       keyless ground-truth series (BLS, Delphi FluView, Yahoo, Wikipedia, CSV)
preprocess  leakage-safe GT repair: zeros, smoothing, detrending, deseasoning, clustering
models      AR benchmark, ARX, LASSO/elastic-net ARX (ARGO-style), SARIMAX, RF, MIDAS weights
evaluate    rolling-origin backtests, metrics, Diebold–Mariano / Clark–West, intervals
report      self-contained HTML artifact renderer
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TRENDS_DIR = DATA_DIR / "trends"
TRUTH_DIR = DATA_DIR / "truth"
QUERIES_DIR = DATA_DIR / "queries"

for _d in (TRENDS_DIR, TRUTH_DIR, QUERIES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
