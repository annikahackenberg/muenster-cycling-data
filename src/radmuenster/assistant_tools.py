"""Tools of the AI assistant: functions that answer questions from the data.

Each tool takes the data (AssistantData) and simple arguments (strings and
numbers, dates as YYYY-MM-DD) and returns a small dict that can be sent as
JSON. The language model only chooses a tool and its arguments and words the
answer; every number comes from these functions.

Nothing here depends on an LLM library: the tools can be tested on their
own, and the provider can change without touching them. TOOL_SPECS describes
the tools as JSON Schema, run_tool calls one by name.
"""

from __future__ import annotations

import difflib
import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from radmuenster.dashboard import (
    CV_RESULTS,
    day_notes,
    forecast_context,
    is_stale,
    latest_forecast,
)
from radmuenster.explore import RESOLUTIONS, aggregate, flagged_periods, ranking, select
from radmuenster.findings import key_findings
from radmuenster.monitoring import evaluate_forecasts

# The site with the forecast; its daily counts carry the direction checks
FORECAST_SITE = "Warendorfer Straße"

# A tool returns at most this many rows, so that one answer cannot fill the
# model's context; for longer periods the model has to choose a coarser
# resolution
MAX_ROWS = 100

WEATHER_COLUMNS = {
    "temp_max": "max_temperature_c",
    "precipitation": "precipitation_mm",
    "precipitation_hours": "rain_hours",
    "sunshine": "sunshine_hours",
    "wind_max": "max_wind_kmh",
    "snowfall": "snowfall_cm",
}


class ToolError(ValueError):
    """Invalid arguments. The message is meant for the model, so that it can
    correct its call."""


@dataclass(frozen=True)
class AssistantData:
    """Everything the tools read.

    sites_daily  daily counts of all sites (load_all_sites)
    daily        daily counts of the Warendorfer Straße (to_daily)
    weather      daily weather (load_weather), including forecast days
    log          the forecast log (read_forecasts)
    today        today in Münster
    warnings     the current monitoring warnings
    """

    sites_daily: pd.DataFrame
    daily: pd.DataFrame
    weather: pd.DataFrame
    log: pd.DataFrame
    today: date
    warnings: list[str] = field(default_factory=list)


def _number(value: float, digits: int = 0) -> float | int | None:
    """A rounded number for JSON; NaN becomes None (null)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(round(value)) if digits == 0 else round(float(value), digits)


def _day(value: str, name: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ToolError(f"{name} must be a date as YYYY-MM-DD, got {value!r}.") from None


def _period(start: str, end: str) -> tuple[date, date]:
    first, last = _day(start, "start"), _day(end, "end")
    if first > last:
        raise ToolError(f"start ({first}) is after end ({last}).")
    return first, last


def _site(data: AssistantData, name: str) -> str:
    """The exact site name for name, ignoring case; unknown names raise a
    ToolError with the closest matches."""
    names = sorted(data.sites_daily["site_name"].unique())
    by_folded = {n.casefold(): n for n in names}
    if str(name).casefold() in by_folded:
        return by_folded[str(name).casefold()]
    close = difflib.get_close_matches(str(name), names, n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else " Call list_sites for all names."
    raise ToolError(f"Unknown counting site {name!r}.{hint}")


def list_sites(data: AssistantData) -> dict:
    """All counting sites with the first and last day of data."""
    spans = data.sites_daily.groupby("site_name")["date"].agg(["min", "max"])
    return {
        "sites": [
            {"name": name, "first_day": f"{row['min']:%Y-%m-%d}", "last_day": f"{row['max']:%Y-%m-%d}"}
            for name, row in spans.iterrows()
        ]
    }


def get_counts(data: AssistantData, site: str, start: str, end: str, resolution: str = "day") -> dict:
    """Trips per day at one site, per day, week or month, as published."""
    name = _site(data, site)
    first, last = _period(start, end)
    key = str(resolution).capitalize()
    if key not in RESOLUTIONS:
        raise ToolError(f"resolution must be day, week or month, got {resolution!r}.")
    rows = select(data.sites_daily, [name], first, last)
    if rows.empty:
        raise ToolError(f"No data for {name} between {first} and {last}.")
    table = aggregate(rows, key, leave_out_flagged=False)
    if len(table) > MAX_ROWS:
        raise ToolError(
            f"{len(table)} rows are too many (at most {MAX_ROWS}): choose a coarser resolution "
            "or a shorter period."
        )
    return {
        "site": name,
        "resolution": key.lower(),
        "unit": "trips per day (for weeks and months the mean over the days with data)",
        "total_trips": _number(rows["total"].sum()),
        "days_with_data": len(rows),
        "flagged_days": int((~rows["ok"]).sum()),
        "rows": [
            {
                "period": f"{row.period:%Y-%m-%d}",
                "trips_per_day": _number(row.trips),
                "days": int(row.days),
                "flagged_days": int(row.flagged_days),
                "flag_reasons": row.reasons,
            }
            for row in table.itertuples()
        ],
    }


def rank_sites(data: AssistantData, start: str, end: str) -> dict:
    """All sites ranked by their mean trips per day in a period."""
    first, last = _period(start, end)
    table = ranking(data.sites_daily, first, last)
    return {
        "period": {"start": f"{first}", "end": f"{last}"},
        "note": (
            "Mean trips per day from days without flags only. coverage is the share of the period "
            "these days cover; sites with reliable = false have too few such days for a fair "
            "comparison."
        ),
        "sites": [
            {
                "rank": rank,
                "site": row.site_name,
                "trips_per_day": _number(row.trips),
                "ok_days": int(row.ok_days),
                "coverage": _number(row.coverage, 2),
                "reliable": bool(row.reliable),
            }
            for rank, row in enumerate(table.itertuples(), 1)
        ],
    }


def get_forecast(data: AssistantData) -> dict:
    """The latest forecast with the counts, weather and calendar it compares to."""
    forecast = latest_forecast(data.log)
    context = forecast_context(forecast, data.daily)
    return {
        "site": FORECAST_SITE,
        "forecast_date": f"{forecast['forecast_date']:%Y-%m-%d}",
        "weekday": f"{forecast['forecast_date']:%A}",
        "predicted_trips": _number(forecast["prediction"]),
        "made_on": f"{forecast['made_on']:%Y-%m-%d}",
        "paused": is_stale(forecast, data.today),
        "same_weekday_last_week": _number(context["last_week"]),
        "typical_same_weekday": _number(context["typical"]),
        "weather_forecast": {
            label: _number(forecast[column], 1) for column, label in WEATHER_COLUMNS.items()
        },
        "calendar": day_notes(forecast["forecast_date"]) or ["regular day"],
        "model": (
            "LightGBM, retrained every evening on all complete days since 2019, with calendar, "
            "university and school holidays, recent counts and the weather forecast."
        ),
    }


def get_forecast_accuracy(data: AssistantData, last_days: int = 28) -> dict:
    """Error of the logged daily forecasts over the last days, and the
    cross-validation result for comparison."""
    if not isinstance(last_days, int) or not 1 <= last_days <= 366:
        raise ToolError(f"last_days must be a whole number between 1 and 366, got {last_days!r}.")
    evaluated = evaluate_forecasts(data.log, data.daily)
    since = pd.Timestamp(data.today) - pd.Timedelta(days=last_days)
    evaluated = evaluated[evaluated["forecast_date"] >= since].sort_values("forecast_date")
    name, mae, wape = CV_RESULTS[-1]
    result = {
        "site": FORECAST_SITE,
        "days_evaluated": len(evaluated),
        "cross_validation": {
            "model": name,
            "mae_trips": mae,
            "wape": wape,
            "note": "2023 to 2026 with measured weather, so a lower bound for the daily forecast",
        },
    }
    if evaluated.empty:
        result["note"] = "No forecast day with complete counts in this period yet."
        return result
    absolute = evaluated["error"].abs()
    result.update(
        mae_trips=_number(absolute.mean()),
        wape=_number(absolute.sum() / evaluated["actual"].sum(), 3),
        bias=_number(evaluated["pct_error"].mean(), 3),
        bias_note="mean percentage error; positive means the forecasts were too high",
        days=[
            {
                "date": f"{row.forecast_date:%Y-%m-%d}",
                "predicted": _number(row.prediction),
                "actual": _number(row.actual),
            }
            for row in evaluated.tail(MAX_ROWS).itertuples()
        ],
    )
    return result


def get_weather(data: AssistantData, start: str, end: str) -> dict:
    """Daily weather in Münster (measured; forecast for the coming days)."""
    first, last = _period(start, end)
    rows = data.weather.loc[pd.Timestamp(first) : pd.Timestamp(last)]
    if rows.empty:
        raise ToolError(f"No weather data between {first} and {last}.")
    if len(rows) > MAX_ROWS:
        raise ToolError(f"{len(rows)} days are too many (at most {MAX_ROWS}): choose a shorter period.")
    return {
        "days": [
            {"date": f"{day:%Y-%m-%d}"}
            | {label: _number(row[column], 1) for column, label in WEATHER_COLUMNS.items()}
            for day, row in rows.iterrows()
        ]
    }


def get_insights(data: AssistantData) -> dict:
    """The key findings of the analysis of the Warendorfer Straße."""
    return {
        "site": FORECAST_SITE,
        "findings": [
            {"title": item.title, "text": item.text} for item in key_findings(data.daily, data.weather)
        ],
    }


def get_flagged_periods(data: AssistantData, site: str, start: str, end: str) -> dict:
    """Periods with suspicious days (outages, missing or incomplete data) at
    one site."""
    name = _site(data, site)
    first, last = _period(start, end)
    periods = flagged_periods(select(data.sites_daily, [name], first, last))
    return {
        "site": name,
        "note": (
            "The counts are published as they are; suspicious days are only flagged. The shortened "
            "last day of each month is routine and not listed."
        ),
        "periods": [
            {
                "first_day": f"{row.first:%Y-%m-%d}",
                "last_day": f"{row.last:%Y-%m-%d}",
                "days": int(row.days),
                "reason": row.reason,
            }
            for row in periods.tail(MAX_ROWS).itertuples()
        ],
    }


def get_monitoring_status(data: AssistantData) -> dict:
    """The warnings of this evening's monitoring checks."""
    return {
        "checked_on": f"{data.today}",
        "checks": [
            "freshness of the counts",
            "incomplete days in the last week",
            "jumps in the share of trips into the city (a sign that the site changed)",
            "forecasts off in the same direction for weeks",
        ],
        "warnings": data.warnings,
    }


_DATE = {"type": "string", "description": "Date as YYYY-MM-DD"}
_SITE = {"type": "string", "description": "Name of a counting site, as listed by list_sites"}


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    function: Callable[..., dict]


TOOL_SPECS = [
    ToolSpec(
        "list_sites",
        "List all bicycle counting sites of the City of Münster with their first and last day of data.",
        _schema({}, []),
        list_sites,
    ),
    ToolSpec(
        "get_counts",
        "Daily bicycle counts at one site between two dates, per day, week or month, as published, "
        "with the flagged (suspicious) days. Use week or month for periods longer than about three "
        f"months; at most {MAX_ROWS} rows.",
        _schema(
            {
                "site": _SITE,
                "start": _DATE,
                "end": _DATE,
                "resolution": {"type": "string", "enum": ["day", "week", "month"]},
            },
            ["site", "start", "end", "resolution"],
        ),
        get_counts,
    ),
    ToolSpec(
        "rank_sites",
        "Rank all counting sites by their mean trips per day between two dates, from days without "
        "flags only, with how reliable each mean is.",
        _schema({"start": _DATE, "end": _DATE}, ["start", "end"]),
        rank_sites,
    ),
    ToolSpec(
        "get_forecast",
        f"The latest daily forecast for the {FORECAST_SITE}, with the same weekday last week, a "
        "typical such day, the weather forecast and the calendar (holidays, lectures).",
        _schema({}, []),
        get_forecast,
    ),
    ToolSpec(
        "get_forecast_accuracy",
        "How far off the daily forecasts were over the last days (mean error, WAPE, bias, each "
        "day), and the cross-validation result for comparison.",
        _schema(
            {"last_days": {"type": "integer", "description": "Number of days back, 1 to 366"}},
            ["last_days"],
        ),
        get_forecast_accuracy,
    ),
    ToolSpec(
        "get_weather",
        f"Daily weather in Münster between two dates (at most {MAX_ROWS} days): temperature, "
        "precipitation, rain hours, sunshine, wind, snowfall.",
        _schema({"start": _DATE, "end": _DATE}, ["start", "end"]),
        get_weather,
    ),
    ToolSpec(
        "get_insights",
        f"The key findings of the data analysis of the {FORECAST_SITE}: effect of the weekday, "
        "the season and rain, and the known data problems.",
        _schema({}, []),
        get_insights,
    ),
    ToolSpec(
        "get_flagged_periods",
        "Periods with suspicious days at one site between two dates: outages the city did not "
        "flag, missing or incomplete days, dead channels.",
        _schema({"site": _SITE, "start": _DATE, "end": _DATE}, ["site", "start", "end"]),
        get_flagged_periods,
    ),
    ToolSpec(
        "get_monitoring_status",
        "The warnings of the latest monitoring run of the forecast (data freshness, incomplete "
        "days, measurement jumps, forecast bias).",
        _schema({}, []),
        get_monitoring_status,
    ),
]

_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


def run_tool(data: AssistantData, name: str, arguments: dict) -> dict:
    """Call the tool name with arguments (as sent by the model).

    Raises ToolError for an unknown tool or invalid arguments; the caller
    passes the message back to the model.
    """
    if name not in _BY_NAME:
        raise ToolError(f"Unknown tool {name!r}.")
    function = _BY_NAME[name].function
    try:
        # Check the argument names before the call, so that a TypeError from
        # a bug inside the tool is not mistaken for a wrong call
        inspect.signature(function).bind(data, **arguments)
    except TypeError as error:
        raise ToolError(f"Invalid arguments for {name}: {error}") from None
    return function(data, **arguments)
