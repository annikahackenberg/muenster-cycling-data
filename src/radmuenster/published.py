"""Read what the daily job published, for the Streamlit app.

The forecast log and the monitoring status are committed to main; the
data files are on GitHub Pages (static_page.read_published). If GitHub
cannot be reached, the local files of the last run are read.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from radmuenster.assistant_tools import AssistantData
from radmuenster.forecast import (
    BACKTEST_FILE,
    FORECAST_FILE,
    SITES_FORECAST_FILE,
    read_forecasts,
)
from radmuenster.monitoring import STATUS_FILE
from radmuenster.static_page import DAILY_DATA, SITES_DATA, WEATHER_DATA, read_published

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


def _read_optional_log(path: Path) -> pd.DataFrame:
    """A log that may not exist yet (before the first daily run after it was
    added): from main, else the local file, else an empty table."""
    try:
        return read_forecasts(RAW_URL + path.as_posix())
    except URLError:
        # HTTPError (e.g. 404, not on main yet) is a URLError too
        if path.exists():
            return read_forecasts(path)
        return pd.DataFrame(columns=["forecast_date", "site_id", "site_name", "model", "prediction"])


def read_sites_forecast_log() -> pd.DataFrame:
    """The forecasts of all sites (SITES_FORECAST_FILE)."""
    return _read_optional_log(SITES_FORECAST_FILE)


def read_backtest() -> pd.DataFrame:
    """The backtest of all sites (BACKTEST_FILE)."""
    return _read_optional_log(BACKTEST_FILE)


def read_status() -> dict:
    """The warnings of the latest monitoring run, like read_forecast_log.

    A status from before actions and checked warnings existed gets empty
    lists for them, so that the pages can rely on all keys."""
    try:
        with urlopen(RAW_URL + STATUS_FILE.as_posix(), timeout=10) as response:
            status = json.load(response)
    except URLError:
        status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    status.setdefault("actions", [])
    status.setdefault("checked_warnings", [])
    return status


def read_assistant_data(today: date) -> AssistantData:
    """Everything the assistant's tools read, as published by the daily job."""
    status = read_status()
    return AssistantData(
        sites_daily=read_published(SITES_DATA),
        daily=read_published(DAILY_DATA),
        weather=read_published(WEATHER_DATA),
        log=read_forecast_log(),
        today=today,
        warnings=status["warnings"],
        checked_warnings=status["checked_warnings"],
        sites_log=read_sites_forecast_log(),
        backtest=read_backtest(),
    )
