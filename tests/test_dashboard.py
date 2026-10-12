from datetime import date

import numpy as np
import pandas as pd
import pytest

from radmuenster.dashboard import (
    COUNTED,
    FORECAST,
    chart_data,
    forecast_accuracy,
    is_stale,
    latest_forecast,
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
    # Next evening without a new forecast: the job did not run
    assert is_stale(forecast, date(2026, 10, 13))


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
