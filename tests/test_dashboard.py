from datetime import date

import numpy as np
import pandas as pd
import pytest

from radmuenster.dashboard import (
    COUNTED,
    FORECAST,
    chart_data,
    day_notes,
    forecast_accuracy,
    forecast_context,
    is_stale,
    latest_forecast,
    site_forecast_accuracy,
    sites_overview,
    weather_metrics,
)


def make_daily():
    dates = pd.date_range("2026-10-08", "2026-10-11", freq="D", name="date")
    return pd.DataFrame(
        {"count": [6000.0, 5000.0, 4000.0, 3000.0], "complete": [True, True, False, True]},
        index=dates,
    )


def make_log(dates: list[str], predictions: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"forecast_date": pd.to_datetime(dates), "model": "lightgbm", "prediction": predictions}
    )


def test_latest_forecast_and_staleness():
    log = make_log(["2026-10-13", "2026-10-12"], [7000.0, 6000.0])
    forecast = latest_forecast(log)
    assert forecast["prediction"] == 7000.0
    assert not is_stale(forecast, date(2026, 10, 12))
    # On the day itself, before the evening run
    assert not is_stale(forecast, date(2026, 10, 13))
    # The day after without a new forecast: the job did not run
    assert is_stale(forecast, date(2026, 10, 14))


def test_chart_data_breaks_line_on_incomplete_days():
    log = make_log(["2026-10-01", "2026-10-11", "2026-10-13"], [1.0, 3300.0, 7000.0])
    data = chart_data(make_daily(), log, date(2026, 10, 9))
    counted = data[data["series"] == COUNTED]["trips"].tolist()
    assert counted[0] == 5000.0 and np.isnan(counted[1]) and counted[2] == 3000.0
    # Forecasts before start are left out, tomorrow's is included
    assert data[data["series"] == FORECAST]["trips"].tolist() == [3300.0, 7000.0]


def test_forecast_accuracy_on_complete_days():
    # 9 Oct: 10 % too high; 10 Oct incomplete (skipped); 11 Oct: 10 % too low
    log = make_log(["2026-10-09", "2026-10-10", "2026-10-11"], [5500.0, 1.0, 2700.0])
    accuracy = forecast_accuracy(log, make_daily())
    assert accuracy["n"] == 2
    assert accuracy["mae"] == pytest.approx(400.0)
    assert accuracy["wape"] == pytest.approx(800.0 / 8000.0)
    assert accuracy["bias"] == pytest.approx(0.0)


def test_forecast_accuracy_without_evaluated_days():
    accuracy = forecast_accuracy(make_log(["2026-10-13"], [7000.0]), make_daily())
    assert accuracy["n"] == 0 and np.isnan(accuracy["wape"])


def test_forecast_context_uses_regular_days_only():
    # Five Wednesdays before the forecast date 2026-10-14
    dates = pd.date_range("2026-09-09", "2026-10-12", freq="D", name="date")
    daily = pd.DataFrame({"count": 5000.0, "complete": True}, index=dates)
    daily.loc["2026-10-07", "count"] = 7000.0  # last week
    daily.loc["2026-09-30", "complete"] = False  # an outage two weeks before
    forecast = pd.Series({"forecast_date": pd.Timestamp("2026-10-14")})
    context = forecast_context(forecast, daily)
    assert context["last_week"] == 7000.0
    # Mean of 7 October, 23 and 16 September; the outage is left out
    assert context["typical"] == pytest.approx((7000 + 5000 + 5000) / 3)


def test_forecast_context_after_a_holiday_is_unknown():
    # The same weekday one week before is German Unity Day, no regular day
    dates = pd.date_range("2026-09-25", "2026-10-08", freq="D", name="date")
    daily = pd.DataFrame({"count": 5000.0, "complete": True}, index=dates)
    forecast = pd.Series({"forecast_date": pd.Timestamp("2026-10-10")})
    assert forecast_context(forecast, daily)["last_week"] is None


def test_weather_metrics_show_snow_only_when_forecast():
    forecast = pd.Series(
        {"temp_max": 3.4, "precipitation": 2.3, "precipitation_hours": 5.0,
         "sunshine": 1.2, "wind_max": 18.0, "snowfall": 0.0}
    )
    metrics = weather_metrics(forecast)
    assert metrics["Rain (2.3 mm)"] == "5 h"
    assert "Snow" not in metrics
    forecast["snowfall"] = 1.5
    assert weather_metrics(forecast)["Snow"] == "1.5 cm"


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        ("2026-10-14", []),
        ("2026-10-03", ["Public holiday: German Unity Day", "No lectures at the university"]),
        ("2026-05-15", ["Bridge day between two days off, many people take it off"]),
        ("2026-08-10", ["School holidays in NRW", "No lectures at the university"]),
    ],
    ids=["regular", "holiday", "bridge_day", "summer"],
)
def test_day_notes(day, expected):
    assert day_notes(pd.Timestamp(day)) == expected


def test_site_forecast_accuracy_per_site_and_since():
    dates = pd.date_range("2026-10-01", periods=4, freq="D")
    log = pd.DataFrame(
        {
            "forecast_date": list(dates) * 2,
            "site_id": ["a"] * 4 + ["b"] * 4,
            "site_name": ["A"] * 4 + ["B"] * 4,
            "prediction": [110.0, 90.0, 110.0, 90.0, 1000.0, 1000.0, 1000.0, 1200.0],
        }
    )
    counts = {s: pd.DataFrame({"count": 100.0 if s == "a" else 1000.0, "complete": True}, index=dates) for s in "ab"}
    accuracy = site_forecast_accuracy(log, counts)
    assert accuracy.loc["a", ["n", "mae", "wape", "bias"]].tolist() == pytest.approx([4, 10.0, 0.1, 0.0])
    assert accuracy.loc["b", "site_name"] == "B"
    recent = site_forecast_accuracy(log, counts, since=pd.Timestamp("2026-10-04"))
    assert recent.loc["b", ["n", "bias"]].tolist() == pytest.approx([1, 0.2])


def make_site_rows(site_id: str, name: str, prediction: float, latest: str = "2026-10-11") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_date": [pd.Timestamp("2026-10-13")],
            "site_id": [site_id],
            "site_name": [name],
            "prediction": [prediction],
            "latest_complete_day": [pd.Timestamp(latest)],
        }
    )


def test_sites_overview_shows_the_own_model_and_notes_stale_sites():
    dates = pd.date_range("2026-09-01", "2026-10-11", freq="D")
    counts = {
        "100034983": pd.DataFrame({"count": 6000.0, "complete": True}, index=dates),
        "x": pd.DataFrame({"count": 1000.0, "complete": True}, index=dates),
        "old": pd.DataFrame({"count": 500.0, "complete": True}, index=dates),
    }
    global_log = pd.concat(
        [
            make_site_rows("100034983", "Warendorfer Straße", 5000),
            make_site_rows("x", "X", 1100),
            make_site_rows("old", "Old", 400, latest="2025-08-06"),
        ]
    )
    own_log = make_site_rows("100034983", "Warendorfer Straße", 6600).drop(columns="site_id")
    backtest = make_site_rows("x", "X", 1100).assign(forecast_date=pd.Timestamp("2026-10-01"))

    overview = sites_overview(global_log, own_log, counts, date(2026, 10, 12), global_backtest=backtest)
    rows = overview.set_index("site_id")
    assert overview["site_id"].tolist() == ["100034983", "x", "old"]  # largest forecast first
    assert rows.loc["100034983", "prediction"] == 6600  # the own model, not the global one
    assert rows.loc["x", "vs_typical"] == pytest.approx(0.1)
    assert rows.loc["x", ["error", "error_basis"]].tolist() == [pytest.approx(0.1), "backtest"]
    assert rows.loc["old", "note"].startswith("No usable counts since 06 Aug 2025")
    assert pd.isna(rows.loc["old", "vs_typical"])
