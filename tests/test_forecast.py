from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from radmuenster.explain import GROUPS
from radmuenster.forecast import (
    EXPLANATION_FILE,
    FORECAST_KEY,
    combine_weather,
    forecast_day,
    forecast_sites,
    read_forecasts,
    save_forecast,
    run_day,
    today_in_muenster,
)
from radmuenster.weather import DAILY_VARIABLES


def test_today_in_muenster_is_ahead_of_utc_around_midnight():
    # 23:30 UTC on 12 October is 01:30 on 13 October in Münster (summer time)
    assert today_in_muenster(datetime(2026, 10, 12, 23, 30, tzinfo=timezone.utc)) == date(
        2026, 10, 13
    )


def test_run_day_keeps_a_late_run_on_the_evening_before():
    utc = timezone.utc
    # Winter: 22:52 UTC on 17 November is 23:52 in Münster, still the 17th
    assert run_day(datetime(2026, 11, 17, 22, 52, tzinfo=utc)) == date(2026, 11, 17)
    # 23:44 UTC on 16 November is 00:44 on the 17th: a late run of the 16th
    assert run_day(datetime(2026, 11, 16, 23, 44, tzinfo=utc)) == date(2026, 11, 16)
    # Summer: 02:59 UTC on 13 October is 04:59 in Münster, still the 12th ...
    assert run_day(datetime(2026, 10, 13, 2, 59, tzinfo=utc)) == date(2026, 10, 12)
    # ... from 05:00 on, the city may have published the 12th: a run of the 13th
    assert run_day(datetime(2026, 10, 13, 3, 0, tzinfo=utc)) == date(2026, 10, 13)
    assert run_day(datetime(2026, 11, 18, 15, 17, tzinfo=utc)) == date(2026, 11, 18)


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
    record, _ = forecast_day(daily, weather, date(2026, 10, 13), made_on=date(2026, 10, 12))
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
    record, _ = forecast_day(daily, weather, date(2026, 10, 1), made_on=date(2026, 9, 30))
    assert record.iloc[0]["latest_complete_day"] == date(2026, 9, 29)


def test_forecast_day_raises_without_weather_for_target(daily):
    weather = make_weather("2025-01-01", "2026-10-12")
    with pytest.raises(ValueError, match="2026-10-13"):
        forecast_day(daily, weather, date(2026, 10, 13), made_on=date(2026, 10, 12))


def test_forecast_day_explains_its_forecast(daily):
    weather = make_weather("2025-01-01", "2026-10-13")
    record, explanation = forecast_day(daily, weather, date(2026, 10, 13), made_on=date(2026, 10, 12))
    row = explanation.iloc[0]
    assert (row["forecast_date"], row["made_on"]) == (date(2026, 10, 13), date(2026, 10, 12))
    assert (row["site_id"], row["model"]) == ("100034983", "lightgbm")
    # The own model has no site feature
    groups = [group for group in GROUPS if group != "this_site"]
    assert list(explanation.columns) == [
        *FORECAST_KEY[:1], "made_on", *FORECAST_KEY[1:], "prediction", "base", *groups
    ]
    # The logged prediction, to match the explanation to its forecast
    assert row["prediction"] == record.iloc[0]["prediction"]
    # Base times the factors gives the forecast (factors rounded to 4 decimals)
    assert row["base"] * row[groups].prod() == pytest.approx(record.iloc[0]["prediction"], rel=1e-3)


def test_failed_explanation_keeps_the_forecast(daily, monkeypatch):
    def broken(*args):
        raise ValueError("Features without a group in GROUPS: ['new_feature']")

    monkeypatch.setattr("radmuenster.forecast.explain_lightgbm", broken)
    monkeypatch.setattr("radmuenster.forecast.explain_global", broken)
    weather = make_weather("2025-01-01", "2026-10-13")
    record, explanation = forecast_day(daily, weather, date(2026, 10, 13), made_on=date(2026, 10, 12))
    assert len(record) == 1 and explanation is None
    counts = {"100034980": daily, "100034981": daily.assign(count=daily["count"] / 10)}
    record, explanation = forecast_sites(counts, weather, date(2026, 10, 13), made_on=date(2026, 10, 12))
    assert len(record) == 2 and explanation is None


def test_forecast_sites_explains_every_forecast(daily):
    weather = make_weather("2025-01-01", "2026-10-13")
    counts = {"100034980": daily, "100034981": daily.assign(count=daily["count"] / 10)}
    record, explanation = forecast_sites(counts, weather, date(2026, 10, 13), made_on=date(2026, 10, 12))
    assert explanation["site_id"].tolist() == record["site_id"].tolist()
    assert explanation["prediction"].tolist() == record["prediction"].tolist()
    assert (explanation["model"] == "global_lightgbm").all()
    factors = explanation[list(GROUPS)].prod(axis=1)
    assert (explanation["base"] * factors).tolist() == pytest.approx(record["prediction"].tolist(), rel=1e-3)


def test_explanations_are_saved_once_per_forecast(daily, tmp_path):
    weather = make_weather("2025-01-01", "2026-10-13")
    path = tmp_path / EXPLANATION_FILE.name
    _, explanation = forecast_day(daily, weather, date(2026, 10, 13), made_on=date(2026, 10, 12))
    save_forecast(explanation, path, key=FORECAST_KEY)
    # A second run on the same day replaces the row
    log = save_forecast(explanation, path, key=FORECAST_KEY)
    assert len(log) == 1
    assert read_forecasts(path)["site_id"].tolist() == ["100034983"]


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


def test_forecast_sites_predicts_every_site_on_its_own_scale(daily):
    weather = make_weather("2025-01-01", "2026-10-13")
    counts = {
        "100034980": daily,  # Hammer Straße
        "100034981": daily.assign(count=daily["count"] / 10),  # Weseler Straße, a tenth
    }
    record, _ = forecast_sites(counts, weather, date(2026, 10, 13), made_on=date(2026, 10, 12))
    assert record["site_name"].tolist() == ["Hammer Straße", "Weseler Straße"]
    assert record["prediction"].tolist() == pytest.approx([6000, 600], rel=0.05)
    assert (record["model"] == "global_lightgbm").all()
    assert (record["latest_complete_day"] == date(2026, 10, 11)).all()


def test_forecast_sites_skips_a_site_without_usable_days(daily):
    weather = make_weather("2025-01-01", "2026-10-13")
    counts = {"100034980": daily, "100034981": daily.assign(complete=False)}
    record, explanation = forecast_sites(counts, weather, date(2026, 10, 13), made_on=date(2026, 10, 12))
    assert record["site_id"].tolist() == ["100034980"]
    assert explanation["site_id"].tolist() == ["100034980"]


def test_save_forecast_keeps_one_row_per_site(tmp_path):
    path = tmp_path / "sites.csv"
    key = ("forecast_date", "site_id", "model")
    first = pd.concat([make_record(date(2026, 10, 13), 100), make_record(date(2026, 10, 13), 200)])
    first["site_id"] = ["a", "b"]
    save_forecast(first, path, key=key)
    rerun = make_record(date(2026, 10, 13), 150).assign(site_id="a")
    log = save_forecast(rerun, path, key=key)
    assert log[["site_id", "prediction"]].values.tolist() == [["a", 150], ["b", 200]]


def test_read_forecasts_keeps_site_ids_as_text(tmp_path):
    path = tmp_path / "sites.csv"
    save_forecast(make_record(date(2026, 10, 13), 100).assign(site_id="100034983"), path)
    assert read_forecasts(path)["site_id"].tolist() == ["100034983"]
