import inspect
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from radmuenster.assistant_tools import (
    MAX_ROWS,
    SITE_IDS,
    TOOL_SPECS,
    AssistantData,
    ToolError,
    get_counts,
    get_flagged_periods,
    get_forecast,
    get_forecast_accuracy,
    get_forecasts_all_sites,
    get_weather,
    rank_sites,
    run_tool,
)

TODAY = date(2026, 10, 13)
FLAGS = ["missing", "partial", "status_fault", "zero", "dead_channel"]


def make_sites_daily() -> pd.DataFrame:
    """Two sites from 1 September to 12 October 2026; the Neutor counted
    nothing from 10 to 12 September."""
    dates = pd.date_range("2026-09-01", "2026-10-12", freq="D")
    frames = []
    for name, total in [("Warendorfer Straße", 5000.0), ("Neutor", 8000.0)]:
        frame = pd.DataFrame(
            {"site_id": SITE_IDS[name], "site_name": name, "date": dates, "total": total, "note": "", "ok": True}
        )
        for flag in FLAGS:
            frame[flag] = False
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    outage = (table["site_name"] == "Neutor") & table["date"].between("2026-09-10", "2026-09-12")
    table.loc[outage, ["total", "zero", "ok"]] = [0.0, True, False]
    return table


def make_daily() -> pd.DataFrame:
    """Warendorfer Straße since 2024: 6,000 trips on workdays, 3,000 at weekends."""
    dates = pd.date_range("2024-01-01", "2026-10-12", freq="D", name="date")
    count = np.where(dates.dayofweek < 5, 6000.0, 3000.0)
    return pd.DataFrame(
        {"count": count, "count_in": count / 2, "complete": True, "direction_fault": False, "n_flagged": 0},
        index=dates,
    )


def make_weather() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", "2026-10-14", freq="D", name="date")
    weather = pd.DataFrame(
        {
            "temp_mean": 10.0,
            "temp_max": 15.0,
            "precipitation": 1.0,
            "precipitation_hours": np.tile([0.0, 2.0, 5.0, 10.0], len(dates) // 4 + 1)[: len(dates)],
            "snowfall": 0.0,
            "sunshine": 5.0,
            "wind_max": 20.0,
        },
        index=dates,
    )
    return weather


def make_log() -> pd.DataFrame:
    """Forecasts for 10 to 14 October; the last one is for tomorrow."""
    dates = pd.date_range("2026-10-10", "2026-10-14", freq="D")
    log = pd.DataFrame(
        {
            "forecast_date": dates,
            "made_on": dates - pd.Timedelta(days=1),
            "model": "lightgbm",
            "prediction": [3300, 3300, 6600, 6000, 6600],
            "latest_complete_day": dates - pd.Timedelta(days=2),
        }
    )
    for column in ["temp_mean", "temp_max", "precipitation", "precipitation_hours", "snowfall", "sunshine", "wind_max"]:
        log[column] = 1.0
    return log


def make_sites_log(dates: pd.DatetimeIndex, prediction: float) -> pd.DataFrame:
    """Forecasts of the global model at the Neutor and the Warendorfer Straße."""
    frames = []
    for name in ["Neutor", "Warendorfer Straße"]:
        log = make_log().iloc[: len(dates)].assign(
            forecast_date=dates, site_id=SITE_IDS[name], site_name=name, model="global_lightgbm",
            prediction=prediction,
        )
        log["made_on"] = dates - pd.Timedelta(days=1)
        log["latest_complete_day"] = dates - pd.Timedelta(days=2)
        frames.append(log)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def data() -> AssistantData:
    return AssistantData(
        make_sites_daily(), make_daily(), make_weather(), make_log(), TODAY, ["A warning."],
        sites_log=make_sites_log(pd.date_range("2026-10-10", "2026-10-14", freq="D"), 8800.0),
        backtest=make_sites_log(pd.date_range("2026-10-01", "2026-10-05", freq="D"), 8000.0),
    )


def test_get_counts_per_week_ignores_case_of_the_site(data):
    result = get_counts(data, "warendorfer straße", "2026-09-07", "2026-09-20", "week")
    assert result["site"] == "Warendorfer Straße"
    assert result["total_trips"] == 14 * 5000
    assert [row["period"] for row in result["rows"]] == ["2026-09-07", "2026-09-14"]
    assert all(row["trips_per_day"] == 5000 and row["days"] == 7 for row in result["rows"])


def test_get_counts_reports_flagged_days(data):
    result = get_counts(data, "Neutor", "2026-09-07", "2026-09-13", "day")
    assert result["flagged_days"] == 3
    assert [row["flagged_days"] for row in result["rows"]] == [0, 0, 0, 1, 1, 1, 0]
    assert result["rows"][3]["flag_reasons"] != ""


def test_unknown_site_suggests_close_names(data):
    with pytest.raises(ToolError, match="Did you mean: Warendorfer Straße"):
        get_counts(data, "Warendorfer Str", "2026-09-01", "2026-09-30", "day")


def test_too_many_rows_ask_for_a_coarser_resolution(data):
    sites = make_sites_daily()
    longer = pd.concat([sites.assign(date=sites["date"] - pd.Timedelta(days=42 * k)) for k in range(4)])
    many = AssistantData(longer, data.daily, data.weather, data.log, TODAY)
    with pytest.raises(ToolError, match=f"at most {MAX_ROWS}"):
        get_counts(many, "Neutor", "2026-01-01", "2026-10-12", "day")


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [("1 Sept", "2026-09-30", "YYYY-MM-DD"), ("2026-09-30", "2026-09-01", "is after end")],
)
def test_invalid_periods(data, start, end, message):
    with pytest.raises(ToolError, match=message):
        get_counts(data, "Neutor", start, end, "day")


def test_rank_sites_leaves_flagged_days_out(data):
    result = rank_sites(data, "2026-09-01", "2026-09-30")
    assert [row["site"] for row in result["sites"]] == ["Neutor", "Warendorfer Straße"]
    neutor = result["sites"][0]
    # The three zero days do not pull the mean down, but lower the coverage
    assert neutor["trips_per_day"] == 8000
    assert neutor["ok_days"] == 27
    assert neutor["coverage"] == 0.9


def test_get_forecast_puts_the_number_into_context(data):
    result = get_forecast(data)
    assert result["forecast_date"] == "2026-10-14"
    assert result["weekday"] == "Wednesday"
    assert result["predicted_trips"] == 6600
    assert result["same_weekday_last_week"] == 6000
    assert result["typical_same_weekday"] == 6000
    assert not result["paused"]
    assert result["calendar"] == ["regular day"]


def test_get_forecast_at_another_site_comes_from_the_global_model(data):
    result = get_forecast(data, "neutor")
    assert (result["site"], result["predicted_trips"], result["typical_same_weekday"]) == ("Neutor", 8800, 8000)
    assert result["model"].startswith("One LightGBM for all sites")
    assert "note" not in result


def test_get_forecast_accuracy_at_another_site_with_its_backtest(data):
    result = get_forecast_accuracy(data, 7, "Neutor")
    assert result["days_evaluated"] == 3  # 10 to 12 October have counts
    assert result["mae_trips"] == 800
    assert result["backtest"]["wape"] == 0.0
    assert "cross_validation" not in result  # only for the Warendorfer Straße


def test_get_forecasts_all_sites_shows_the_own_model_of_the_warendorfer_strasse(data):
    result = get_forecasts_all_sites(data)
    assert result["forecast_date"] == "2026-10-14"
    sites = {row["site"]: row for row in result["sites"]}
    assert sites["Neutor"]["predicted_trips"] == 8800
    assert sites["Neutor"]["change_vs_typical"] == 0.1
    assert sites["Warendorfer Straße"]["predicted_trips"] == 6600  # own model, not 8800
    assert result["sites"][0]["site"] == "Neutor"  # largest first


def test_get_forecast_accuracy_over_the_last_days(data):
    result = get_forecast_accuracy(data, 7)
    # 10 to 12 October have counts; 13 and 14 October not yet
    assert result["days_evaluated"] == 3
    assert result["mae_trips"] == 400
    assert [day["date"] for day in result["days"]] == ["2026-10-10", "2026-10-11", "2026-10-12"]
    assert result["cross_validation"]["wape"] == 0.089


@pytest.mark.parametrize("last_days", [0, 400, "7"])
def test_get_forecast_accuracy_rejects_invalid_days(data, last_days):
    with pytest.raises(ToolError, match="last_days"):
        get_forecast_accuracy(data, last_days)


def test_get_weather_rows_and_limit(data):
    result = get_weather(data, "2026-10-12", "2026-10-14")
    assert [day["date"] for day in result["days"]] == ["2026-10-12", "2026-10-13", "2026-10-14"]
    assert result["days"][0]["max_temperature_c"] == 15.0
    with pytest.raises(ToolError, match="too many"):
        get_weather(data, "2025-01-01", "2026-01-01")


def test_get_flagged_periods(data):
    result = get_flagged_periods(data, "Neutor", "2026-09-01", "2026-10-12")
    assert len(result["periods"]) == 1
    period = result["periods"][0]
    assert (period["first_day"], period["last_day"], period["days"]) == ("2026-09-10", "2026-09-12", 3)


VALID_ARGUMENTS = {
    "list_sites": {},
    "get_counts": {"site": "Neutor", "start": "2026-09-01", "end": "2026-09-30", "resolution": "week"},
    "rank_sites": {"start": "2026-09-01", "end": "2026-09-30"},
    "get_forecast": {"site": "Neutor"},
    "get_forecasts_all_sites": {},
    "get_forecast_accuracy": {"last_days": 28, "site": "Neutor"},
    "get_weather": {"start": "2026-10-01", "end": "2026-10-14"},
    "get_insights": {},
    "get_flagged_periods": {"site": "Neutor", "start": "2026-09-01", "end": "2026-10-12"},
    "get_monitoring_status": {},
}


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda spec: spec.name)
def test_every_tool_returns_json(data, spec):
    # json.dumps fails on numpy numbers and timestamps, which the model
    # could not read
    json.dumps(run_tool(data, spec.name, VALID_ARGUMENTS[spec.name]))


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda spec: spec.name)
def test_schemas_match_the_functions(spec):
    parameters = list(inspect.signature(spec.function).parameters)[1:]
    assert list(spec.input_schema["properties"]) == parameters
    assert spec.input_schema["additionalProperties"] is False


def test_run_tool_rejects_unknown_tools_and_arguments(data):
    with pytest.raises(ToolError, match="Unknown tool"):
        run_tool(data, "delete_everything", {})
    with pytest.raises(ToolError, match="Invalid arguments for get_weather"):
        run_tool(data, "get_weather", {"start": "2026-10-01"})
    with pytest.raises(ToolError, match="Invalid arguments for list_sites"):
        run_tool(data, "list_sites", {"site": "Neutor"})
