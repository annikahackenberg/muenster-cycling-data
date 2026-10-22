from datetime import date

import numpy as np
import pandas as pd

from radmuenster.backtest import backtest
from radmuenster.weather import DAILY_VARIABLES, daily_from_hourly


def test_daily_from_hourly_aggregates_like_open_meteo():
    hours = pd.date_range("2026-10-13", periods=48, freq="h")
    hourly = pd.DataFrame(
        {
            "temperature": np.r_[np.full(24, 10.0), np.arange(24.0)],
            "precipitation": np.r_[np.zeros(24), [0.5, 0.2] + [0.0] * 22],
            "snowfall": 0.0,
            "sunshine": 1800.0,  # half of every hour
            "wind": np.r_[np.full(24, 5.0), np.full(24, 7.0)],
        },
        index=hours,
    )
    daily = daily_from_hourly(hourly)
    second = daily.loc["2026-10-14"]
    assert list(daily.columns) == list(DAILY_VARIABLES.values())
    assert second[["temp_mean", "temp_max", "precipitation", "precipitation_hours", "sunshine", "wind_max"]].tolist() == [
        11.5, 23.0, 0.7, 2.0, 12.0, 7.0,
    ]


def make_counts(start: str, end: str, size: float) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D", name="date")
    return pd.DataFrame({"count": size * np.where(dates.dayofweek < 5, 1.0, 0.5), "complete": True}, index=dates)


def test_backtest_uses_the_weather_forecast_and_only_the_days_known_that_evening():
    dates = pd.date_range("2025-01-01", "2026-10-20", freq="D", name="date")
    measured = pd.DataFrame({column: 1.0 for column in DAILY_VARIABLES.values()}, index=dates)
    forecast = pd.DataFrame({column: 5.0 for column in DAILY_VARIABLES.values()}, index=dates)
    daily = make_counts("2025-01-01", "2026-10-20", 6000)
    counts = {"100034983": daily, "100034980": make_counts("2025-01-01", "2026-10-20", 7000)}

    result = backtest(counts, daily, measured, forecast, [date(2026, 10, 13), date(2026, 10, 14)])
    assert result.groupby("forecast_date").size().tolist() == [3, 3]  # two sites + the own model
    assert set(result["model"]) == {"global_lightgbm", "lightgbm"}
    first = result[result["forecast_date"] == date(2026, 10, 13)]
    assert (first["made_on"] == date(2026, 10, 12)).all()
    assert (first["latest_complete_day"] == date(2026, 10, 11)).all()
    # The forecast weather of the evening before, not the measured weather
    assert (first["temp_max"] == 5.0).all()
