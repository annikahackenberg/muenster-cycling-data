"""Data preparation for the Streamlit page (streamlit_app.py).

Kept apart from the page itself, so that it can be tested without Streamlit.
"""

from __future__ import annotations

from datetime import date

import altair as alt
import numpy as np
import pandas as pd

from holidays import Germany

from radmuenster.features import SUBDIVISION, academic_features, calendar_features, lag_features
from radmuenster.monitoring import evaluate_forecasts

COUNTED = "Counted"
FORECAST = "Forecast"

# Categorical slots 1 and 2 of the validated palette, per theme
SERIES_COLORS = {
    "light": {COUNTED: "#2a78d6", FORECAST: "#eb6834"},
    "dark": {COUNTED: "#3987e5", FORECAST: "#d95926"},
}


def latest_forecast(log: pd.DataFrame) -> pd.Series:
    """The forecast with the latest forecast_date."""
    return log.loc[log["forecast_date"].idxmax()]


def is_stale(forecast: pd.Series, today: date) -> bool:
    """True if the latest forecast is not for tomorrow: the daily job did not
    run (failed, or switched off after the application phase)."""
    return forecast["forecast_date"].date() <= today


def chart_data(daily: pd.DataFrame, log: pd.DataFrame, start: date) -> pd.DataFrame:
    """Counts and forecasts from start on, in long format for the chart.

    Columns: date, series (COUNTED or FORECAST), trips. Days that are not
    complete get NaN as counted value, so the line breaks there instead of
    connecting across an outage.
    """
    start = pd.Timestamp(start)
    days = daily.loc[start:]
    counted = pd.DataFrame(
        {
            "date": days.index,
            "series": COUNTED,
            "trips": days["count"].where(days["complete"]).to_numpy(),
        }
    )
    recent = log[log["forecast_date"] >= start]
    forecast = pd.DataFrame(
        {"date": recent["forecast_date"], "series": FORECAST, "trips": recent["prediction"]}
    )
    return pd.concat([counted, forecast], ignore_index=True)


def forecast_accuracy(log: pd.DataFrame, daily: pd.DataFrame) -> dict[str, float]:
    """Error of the logged forecasts on complete days.

    Unlike the cross-validation, these forecasts used the weather forecast,
    so this is the honest error of the daily forecast.

    Keys: n (number of days), mae (trips), wape, bias (mean percentage error,
    positive: forecasts too high). NaN for the errors without any day.
    """
    evaluated = evaluate_forecasts(log, daily)
    if evaluated.empty:
        return {"n": 0, "mae": np.nan, "wape": np.nan, "bias": np.nan}
    absolute = evaluated["error"].abs()
    return {
        "n": len(evaluated),
        "mae": absolute.mean(),
        "wape": absolute.sum() / evaluated["actual"].sum(),
        "bias": evaluated["pct_error"].mean(),
    }


def weather_summary(forecast: pd.Series) -> str:
    """The forecast weather of a row of the forecast log, in one line."""
    return (
        f"{forecast['temp_max']:.0f} °C max · {forecast['precipitation']:.1f} mm rain "
        f"over {forecast['precipitation_hours']:.0f} h · {forecast['sunshine']:.0f} h sun · "
        f"wind up to {forecast['wind_max']:.0f} km/h"
    )


def weather_metrics(forecast: pd.Series) -> dict[str, str]:
    """The forecast weather of a row of the forecast log, one short value
    per label (they are shown large, side by side). Rain as hours, which
    matter more for cycling than the amount; the amount goes into the label.
    Snow only appears when some is forecast."""
    metrics = {
        "Max temperature": f"{forecast['temp_max']:.0f} °C",
        f"Rain ({forecast['precipitation']:.1f} mm)": f"{forecast['precipitation_hours']:.0f} h",
        "Sunshine": f"{forecast['sunshine']:.0f} h",
        "Max wind": f"{forecast['wind_max']:.0f} km/h",
    }
    if forecast["snowfall"] > 0:
        metrics["Snow"] = f"{forecast['snowfall']:.1f} cm"
    return metrics


def forecast_context(forecast: pd.Series, daily: pd.DataFrame) -> dict[str, float | None]:
    """Counts that put the forecast into context.

    last_week  count on the same weekday one week before the forecast date
    typical    mean of the same weekday one to four weeks before

    Both come from lag_features, as in the model: incomplete days and special
    days off (public holidays, bridge days, ...) are left out, so both
    describe a regular day. None where no such day is known.
    """
    dates = pd.DatetimeIndex([forecast["forecast_date"]])
    lags = lag_features(daily, dates).iloc[0]
    columns = {"last_week": "lag_7", "typical": "same_weekday_4w"}
    return {key: None if pd.isna(lags[col]) else float(lags[col]) for key, col in columns.items()}


def day_notes(day: pd.Timestamp) -> list[str]:
    """What sets the day apart for cycling, from the calendar the model uses.

    Empty for a regular day: no public holiday, no school holidays, lectures
    at the university.
    """
    dates = pd.DatetimeIndex([day]).normalize()
    calendar = calendar_features(dates).iloc[0]
    academic = academic_features(dates).iloc[0]
    notes = []
    if calendar["is_holiday"]:
        names = Germany(subdiv=SUBDIVISION, years=day.year, language="en_US")
        notes.append(f"Public holiday: {names.get(dates[0].date())}")
    if calendar["is_bridge_day"]:
        notes.append("Bridge day between two days off, many people take it off")
    if calendar["is_christmas"]:
        notes.append("Between Christmas and New Year, many people are off work")
    if calendar["is_rosenmontag"]:
        notes.append("Rosenmontag (carnival Monday)")
    if academic["is_school_holiday"]:
        notes.append("School holidays in NRW")
    if not academic["is_lecture_period"]:
        notes.append("No lectures at the university")
    return notes


def counts_chart(data: pd.DataFrame, theme: str, surface: str) -> alt.LayerChart:
    """Counted trips as a line, forecasts as dots, with a hover rule.

    data is the output of chart_data, theme "light" or "dark", surface the
    page background, for the ring that separates the dots from the line.
    """
    colors = SERIES_COLORS[theme]
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=list(colors), range=list(colors.values())),
        legend=alt.Legend(orient="top", title=None),
    )
    base = alt.Chart(data).encode(
        x=alt.X(
            "date:T", title=None, axis=alt.Axis(format="%d %b", grid=False, labelSeparation=8)
        ),
        y=alt.Y("trips:Q", title="Trips per day", scale=alt.Scale(zero=True)),
        color=color,
    )
    # invalid="break-paths-show-domains": a missing day breaks the line instead of being bridged
    line = base.transform_filter(alt.datum.series == COUNTED).mark_line(
        strokeWidth=2, invalid="break-paths-show-domains"
    )
    dots = base.transform_filter(alt.datum.series == FORECAST).mark_point(
        filled=True, size=90, opacity=1, stroke=surface, strokeWidth=2
    )

    # Hover: the date nearest to the pointer, with both values in the tooltip
    wide = data.pivot_table(index="date", columns="series", values="trips").reset_index()
    for series in (COUNTED, FORECAST):
        # As text, so that a missing value shows as "–" instead of "null"
        values = wide[series] if series in wide else pd.Series(float("nan"), index=wide.index)
        wide[series] = values.map(lambda v: "–" if pd.isna(v) else f"{v:,.0f}")
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    rule = (
        alt.Chart(wide)
        .mark_rule(color="gray", strokeWidth=1)
        .encode(
            x="date:T",
            opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%a %d %b %Y"),
                alt.Tooltip(f"{COUNTED}:N", title=COUNTED),
                alt.Tooltip(f"{FORECAST}:N", title=FORECAST),
            ],
        )
        .add_params(hover)
    )
    return alt.layer(line, dots, rule).properties(height=320)
