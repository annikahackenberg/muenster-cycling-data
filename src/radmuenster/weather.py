"""Daily weather in Münster from Open-Meteo.

Source: https://open-meteo.com (historical weather API, ERA5 reanalysis, and
forecast API for the coming days)
License: CC BY 4.0
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

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
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()

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
    resp = requests.get(FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    return parse_weather_json(resp.text)
