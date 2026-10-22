"""Data preparation for the Streamlit page (streamlit_app.py).

Kept apart from the page itself, so that it can be tested without Streamlit.
"""

from __future__ import annotations

from datetime import date

import altair as alt
import numpy as np
import pandas as pd

from holidays import Germany

from radmuenster.data import DEFAULT_SITE
from radmuenster.features import SUBDIVISION, academic_features, calendar_features, lag_features
from radmuenster.monitoring import evaluate_forecasts, evaluate_site_forecasts

# Width of the content of the Streamlit app in pixels. The wide layout uses
# the whole window, which spreads a chart too far on a large screen; the
# centered layout (about 700 pixels) is too narrow for the explore page.
APP_CONTENT_WIDTH = 1000

COUNTED = "Counted"
FORECAST = "Forecast"

# Categorical slots 1 and 2 of the validated palette, per theme
SERIES_COLORS = {
    "light": {COUNTED: "#2a78d6", FORECAST: "#eb6834"},
    "dark": {COUNTED: "#3987e5", FORECAST: "#d95926"},
}


# Time-based cross-validation (notebook 02), 1,232 test days 2023 to 2026:
# (model, MAE in trips, WAPE). Update by hand when the model changes, as the
# table in the README.
CV_RESULTS = (
    ("Naive baseline (same weekday last week)", 960, 0.161),
    ("Holiday baseline", 848, 0.143),
    ("Linear regression on log(count)", 646, 0.109),
    ("LightGBM on log(count)", 529, 0.089),
)

# From the city's counts to the forecast, as (step, explanation), for both
# the static page and the Streamlit app
HOW_IT_WORKS = (
    (
        "Data",
        "the City of Münster publishes 15-minute counts every morning. They are summed to daily "
        "totals; outages that the city's status misses are detected by comparing the two directions.",
    ),
    (
        "Features",
        "weekday, public holidays and bridge days in NRW, lecture periods of the University of "
        "Münster, school holidays, weather, and counts from 2, 7, 14 and 364 days earlier. Lag 1 is "
        "left out on purpose: when the forecast is made, today is not over yet.",
    ),
    (
        "Model",
        "gradient-boosted trees (LightGBM) on the logarithm of the count, retrained every evening "
        "on all complete days since 2019.",
    ),
    (
        "All sites",
        "the other 22 counting sites are forecast by one global model for all sites: counts and "
        "lags are divided by each site's median, so that the model learns weekly patterns, weather "
        "and holidays from all sites together. In a cross-validation it was as good as one model per "
        "site for the sites with a long history and clearly better for the newer ones.",
    ),
    (
        "Automation",
        "a scheduled GitHub Actions workflow forecasts tomorrow every evening, runs the monitoring "
        "and stores the forecast with its weather forecast, which both the static page and the "
        "app show.",
    ),
)


def latest_forecast(log: pd.DataFrame) -> pd.Series:
    """The forecast with the latest forecast_date."""
    return log.loc[log["forecast_date"].idxmax()]


def is_stale(forecast: pd.Series, today: date) -> bool:
    """True if the latest forecast is for a day before today: the daily job
    did not run (failed, or switched off after the application phase).

    During the day, before the evening run, the latest forecast is the one
    for today, which is still current.
    """
    return forecast["forecast_date"].date() < today


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


def site_forecast_accuracy(
    log: pd.DataFrame, counts: dict[str, pd.DataFrame], since: pd.Timestamp | None = None
) -> pd.DataFrame:
    """forecast_accuracy per site, for the forecasts of all sites (counts:
    site_counts_by_id), from since on. One row per site with at least one
    evaluated day, indexed by site_id: site_name, n, mae, wape, bias.

    The bias is the sum of the errors over the sum of the actual counts (see
    check_site_bias), not the mean percentage error of forecast_accuracy.
    """
    evaluated = evaluate_site_forecasts(log, counts)
    if since is not None:
        evaluated = evaluated[evaluated["forecast_date"] >= since]
    evaluated = evaluated.assign(absolute=evaluated["error"].abs())
    by_site = evaluated.groupby("site_id")
    return pd.DataFrame(
        {
            "site_name": by_site["site_name"].first(),
            "n": by_site.size(),
            "mae": by_site["absolute"].mean(),
            "wape": by_site["absolute"].sum() / by_site["actual"].sum(),
            "bias": by_site["error"].sum() / by_site["actual"].sum(),
        }
    )


def _errors(
    global_log: pd.DataFrame | None, own_log: pd.DataFrame | None, counts: dict[str, pd.DataFrame]
) -> dict[str, tuple[int, float]]:
    """(evaluated days, WAPE) per site: the global model's forecasts, but the
    own model's for the Warendorfer Straße."""
    errors = {}
    if global_log is not None:
        accuracy = site_forecast_accuracy(global_log, counts)
        errors = {site: (row["n"], row["wape"]) for site, row in accuracy.iterrows()}
    if own_log is not None and DEFAULT_SITE in counts:
        own = forecast_accuracy(own_log, counts[DEFAULT_SITE])
        errors[DEFAULT_SITE] = (own["n"], own["wape"])
    return errors


# A site whose latest usable day lies more than this before the forecast date
# gets a note instead of comparisons: its forecast rests on old data
STALE_COUNTS_DAYS = 14
# Real forecasts are evaluated over the last ACCURACY_DAYS; with fewer than
# MIN_ACCURACY_DAYS of them, the backtest's error is shown instead
ACCURACY_DAYS = 28
MIN_ACCURACY_DAYS = 7


def sites_overview(
    global_log: pd.DataFrame,
    own_log: pd.DataFrame,
    counts: dict[str, pd.DataFrame],
    today: date,
    global_backtest: pd.DataFrame | None = None,
    own_backtest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per site for the latest forecast date of global_log, the
    largest forecast first.

    global_log holds the forecasts of all sites by the global model, own_log
    those of the Warendorfer Straße by its own model, which it shows instead;
    the backtests likewise (backtest.py). counts: site_counts_by_id.

    Columns: site_id, site_name, prediction, typical (mean of the same
    weekday over the four weeks before, see forecast_context), vs_typical,
    error (WAPE of the last ACCURACY_DAYS, or of the backtest while a site
    has fewer than MIN_ACCURACY_DAYS evaluated forecasts; NaN without
    either), error_basis ("last 4 weeks", "backtest" or ""), note.
    """
    latest = global_log["forecast_date"].max()
    rows = global_log[global_log["forecast_date"] == latest].set_index("site_id")
    own = own_log[own_log["forecast_date"] == latest]
    if DEFAULT_SITE in rows.index and not own.empty:
        rows.loc[DEFAULT_SITE, "prediction"] = own["prediction"].iloc[0]

    # Errors per site as (number of days, WAPE): of the real forecasts of the
    # last weeks, and of the backtest
    since = pd.Timestamp(today) - pd.Timedelta(days=ACCURACY_DAYS)
    live = _errors(
        global_log[global_log["forecast_date"] >= since], own_log[own_log["forecast_date"] >= since], counts
    )
    backtest = _errors(global_backtest, own_backtest, counts)

    records = []
    for site_id, row in rows.iterrows():
        typical = forecast_context(row, counts[site_id])["typical"] if site_id in counts else None
        live_n, live_wape = live.get(site_id, (0, np.nan))
        back_wape = backtest.get(site_id, (0, np.nan))[1]
        if live_n >= MIN_ACCURACY_DAYS:
            error, basis = live_wape, "last 4 weeks"
        elif not pd.isna(back_wape):
            error, basis = back_wape, "backtest"
        else:
            error, basis = np.nan, ""

        note = ""
        last_usable = pd.Timestamp(row["latest_complete_day"])
        if last_usable < latest - pd.Timedelta(days=STALE_COUNTS_DAYS):
            note = (
                f"No usable counts since {last_usable:%d %b %Y}: the forecast rests on older data "
                "and cannot be checked."
            )
            typical, error, basis = None, np.nan, ""
        records.append(
            {
                "site_id": site_id,
                "site_name": row["site_name"],
                "prediction": int(row["prediction"]),
                "typical": typical,
                "vs_typical": row["prediction"] / typical - 1 if typical else np.nan,
                "error": error,
                "error_basis": basis,
                "note": note,
            }
        )
    return pd.DataFrame(records).sort_values("prediction", ascending=False, ignore_index=True)


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
