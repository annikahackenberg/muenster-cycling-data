"""Daily weather in Münster from Open-Meteo.

Source: https://open-meteo.com (historical weather API, ERA5 reanalysis, and
forecast API for the coming days)
License: CC BY 4.0
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Seconds to wait before each new attempt when Open-Meteo has a server error or
# does not answer in time: both happened in the daily job (November 2026) and
# went away by themselves, but one failed request stopped the whole run
RETRY_WAITS = (10, 30)

# Answers worth a new attempt: server errors, and 429 "too many requests"
# (the runners of GitHub Actions share IP addresses)
RETRY_STATUS = {429, 500, 502, 503, 504}

# Warendorfer Straße counting site
LATITUDE = 51.963
LONGITUDE = 7.645

# Open-Meteo variable -> our column name
DAILY_VARIABLES = {
    "temperature_2m_mean": "temp_mean",
    "temperature_2m_max": "temp_max",
    "precipitation_sum": "precipitation",
    "precipitation_hours": "precipitation_hours",
    "snowfall_sum": "snowfall",
    "sunshine_duration": "sunshine",
    "wind_speed_10m_max": "wind_max",
}


def _get(url: str, params: dict, timeout: float) -> requests.Response:
    """requests.get with new attempts after server errors (RETRY_STATUS),
    timeouts and lost connections, also while the answer is downloaded. Other
    client errors (4xx, e.g. a wrong parameter) would fail again, so they are
    raised at once; so are the errors of the last attempt."""
    for wait in RETRY_WAITS:
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as error:
            problem = type(error).__name__
        else:
            if resp.status_code not in RETRY_STATUS:
                break
            problem = f"HTTP {resp.status_code}"
        logger.warning("Open-Meteo: %s for %s, new attempt in %s s", problem, url, wait)
        time.sleep(wait)
    else:
        resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


def download_weather_year(year: int, cache_dir: Path) -> str:
    """Fetch one year of daily weather as JSON text and cache it locally.

    Only past years are cached: the archive fills the most recent days with
    preliminary values that are revised later.
    """
    cache_file = cache_dir / f"{year}.json"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": f"{year}-01-01",
        "end_date": min(date(year, 12, 31), date.today()).isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Europe/Berlin",
    }
    resp = _get(ARCHIVE_URL, params, timeout=30)

    if year < date.today().year:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(resp.text, encoding="utf-8")
    return resp.text


def parse_weather_json(text: str) -> pd.DataFrame:
    """Convert an Open-Meteo response into a DataFrame with one row per day.

    Columns as in DAILY_VARIABLES; sunshine is converted from seconds to hours.
    """
    daily = json.loads(text)["daily"]
    weather = pd.DataFrame({ours: daily[theirs] for theirs, ours in DAILY_VARIABLES.items()})
    weather.index = pd.DatetimeIndex(daily["time"], name="date")
    weather["sunshine"] = weather["sunshine"] / 3600
    return weather


def load_weather(
    start: date = date(2019, 1, 1),
    end: date | None = None,
    cache_dir: Path = Path("data/raw/weather"),
) -> pd.DataFrame:
    """Daily weather from start to end (inclusive)."""
    end = end or date.today()
    frames = [
        parse_weather_json(download_weather_year(year, cache_dir))
        for year in range(start.year, end.year + 1)
    ]
    return pd.concat(frames).sort_index().loc[str(start) : str(end)]


def load_weather_forecast(days: int = 2) -> pd.DataFrame:
    """Daily weather forecast from today on, for the given number of days.

    The forecast comes from weather models (by default a mix of the best
    models for the location), the archive from the ERA5 reanalysis. The two
    differ systematically, e.g. in sunshine hours, so a model trained on the
    archive sees slightly different inputs in the forecast.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "forecast_days": days,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Europe/Berlin",
    }
    resp = _get(FORECAST_URL, params, timeout=30)
    return parse_weather_json(resp.text)


PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Hourly Open-Meteo variables of the previous runs API, forecast 24 hours ahead
PREVIOUS_DAY_VARIABLES = {
    "temperature_2m_previous_day1": "temperature",
    "precipitation_previous_day1": "precipitation",
    "snowfall_previous_day1": "snowfall",
    "sunshine_duration_previous_day1": "sunshine",
    "wind_speed_10m_previous_day1": "wind",
}


def daily_from_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily weather with the columns of DAILY_VARIABLES from hourly values
    (columns of PREVIOUS_DAY_VARIABLES, local time as index).

    The same aggregation as Open-Meteo's daily values: a rain hour is an hour
    with any precipitation; sunshine in hours.
    """
    days = hourly.groupby(hourly.index.normalize().rename("date"))
    return pd.DataFrame(
        {
            "temp_mean": days["temperature"].mean(),
            "temp_max": days["temperature"].max(),
            "precipitation": days["precipitation"].sum(),
            "precipitation_hours": days["precipitation"].apply(lambda hours: (hours > 0).sum()).astype(float),
            "snowfall": days["snowfall"].sum(),
            "sunshine": days["sunshine"].sum() / 3600,
            "wind_max": days["wind"].max(),
        }
    )


def load_weather_previous_day(start: date, end: date) -> pd.DataFrame:
    """The weather forecast for each day from start to end as it looked one
    day earlier, like the forecast the daily job had in the evening.

    From Open-Meteo's previous runs API (archived since January 2024): each
    hour as forecast 24 hours ahead. Only hourly values are archived, so the
    daily values are built with daily_from_hourly.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(PREVIOUS_DAY_VARIABLES),
        "timezone": "Europe/Berlin",
    }
    resp = _get(PREVIOUS_RUNS_URL, params, timeout=60)
    hourly = json.loads(resp.text)["hourly"]
    table = pd.DataFrame({ours: hourly[theirs] for theirs, ours in PREVIOUS_DAY_VARIABLES.items()})
    table.index = pd.DatetimeIndex(hourly["time"])
    return daily_from_hourly(table)
