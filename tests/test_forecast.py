from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from radmuenster.forecast import (
    combine_weather,
    forecast_day,
    read_forecasts,
    save_forecast,
    today_in_muenster,
)
from radmuenster.weather import DAILY_VARIABLES


def test_today_in_muenster_is_ahead_of_utc_around_midnight():
    # 23:30 UTC on 12 October is 01:30 on 13 October in Münster (summer time)
    assert today_in_muenster(datetime(2026, 10, 12, 23, 30, tzinfo=timezone.utc)) == date(
        2026, 10, 13
    )


def make_weather(start: str, end: str, value: float = 1.0) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D", name="date")
    return pd.DataFrame({c: value for c in DAILY_VARIABLES.values()}, index=dates)


def test_combine_weather_prefers_archive_and_adds_forecast_days():
    archive = make_weather("2026-10-10", "2026-10-12", value=1.0)
    forecast = make_weather("2026-10-12", "2026-10-13", value=2.0)
    weather = combine_weather(archive, forecast)
    assert weather["temp_max"].tolist() == [1.0, 1.0, 1.0, 2.0]


def make_daily(start: str, end: str) -> pd.DataFrame:
    """Complete days with a weekly pattern: 6000 on workdays, 3000 at weekends."""
    dates = pd.date_range(start, end, freq="D", name="date")
    count = np.where(dates.dayofweek < 5, 6000.0, 3000.0)
    return pd.DataFrame({"count": count, "complete": True}, index=dates)


@pytest.fixture
def daily():
    return make_daily("2025-01-01", "2026-10-11")


def test_forecast_day_predicts_tomorrow(daily):
    weather = make_weather("2025-01-01", "2026-10-13")
    record = forecast_day(daily, weather, date(2026, 10, 13), made_on=date(2026, 10, 12))
    assert len(record) == 1
    row = record.iloc[0]
    # Tuesday: the model should have learned the workday level
    assert row["prediction"] == pytest.approx(6000, rel=0.05)
    assert row["latest_complete_day"] == date(2026, 10, 11)
    assert row["lag_2"] == 3000.0  # Sunday
    assert row["temp_max"] == 1.0


def test_forecast_day_ignores_days_the_forecast_could_not_know(daily):
    # Forecasting a past date: the day before must not be used
    weather = make_weather("2025-01-01", "2026-10-13")
    record = forecast_day(daily, weather, date(2026, 10, 1), made_on=date(2026, 9, 30))
    assert record.iloc[0]["latest_complete_day"] == date(2026, 9, 29)


def test_forecast_day_raises_without_weather_for_target(daily):
    weather = make_weather("2025-01-01", "2026-10-12")
    with pytest.raises(ValueError, match="2026-10-13"):
        forecast_day(daily, weather, date(2026, 10, 13), made_on=date(2026, 10, 12))


def make_record(forecast_date: date, prediction: int, model: str = "lightgbm") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_date": [forecast_date],
            "made_on": [forecast_date - pd.Timedelta(days=1)],
            "model": [model],
            "prediction": [prediction],
            "latest_complete_day": [forecast_date - pd.Timedelta(days=2)],
        }
    )


def test_save_forecast_appends_and_replaces_same_date_and_model(tmp_path):
    path = tmp_path / "forecasts.csv"
    save_forecast(make_record(date(2026, 10, 13), 5000), path)
    save_forecast(make_record(date(2026, 10, 14), 6000), path)
    save_forecast(make_record(date(2026, 10, 13), 6000, model="other"), path)
    # Rerun on the same day replaces the first forecast
    save_forecast(make_record(date(2026, 10, 13), 5500), path)

    log = read_forecasts(path)
    assert log[["forecast_date", "model", "prediction"]].values.tolist() == [
        [pd.Timestamp("2026-10-13"), "lightgbm", 5500],
        [pd.Timestamp("2026-10-13"), "other", 6000],
        [pd.Timestamp("2026-10-14"), "lightgbm", 6000],
    ]
