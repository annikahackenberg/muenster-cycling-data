"""Read what the daily job published, for the Streamlit app.

The forecast log and the monitoring status are committed to main; the
data files are on GitHub Pages (static_page.read_published). If GitHub
cannot be reached, the local files of the last run are read.
"""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from radmuenster.forecast import FORECAST_FILE, read_forecasts
from radmuenster.monitoring import STATUS_FILE

# The daily job commits to main; reading from there gets the latest forecast
# without redeploying the app
RAW_URL = "https://raw.githubusercontent.com/annikahackenberg/muenster-cycling-data/main/"


def read_forecast_log() -> pd.DataFrame:
    """The forecast log from main on GitHub, or the local file if it is not
    reachable (e.g. before the first merge)."""
    try:
        return read_forecasts(RAW_URL + FORECAST_FILE.as_posix())
    except URLError:
        return read_forecasts(FORECAST_FILE)


def read_status() -> dict:
    """The warnings of the latest monitoring run, like read_forecast_log."""
    try:
        with urlopen(RAW_URL + STATUS_FILE.as_posix(), timeout=10) as response:
            return json.load(response)
    except URLError:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
