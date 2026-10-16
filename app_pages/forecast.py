"""Forecast page: tomorrow's forecast and the recent weeks."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

from radmuenster.dashboard import (
    CV_RESULTS,
    HOW_IT_WORKS,
    chart_data,
    counts_chart,
    day_notes,
    forecast_accuracy,
    forecast_context,
    is_stale,
    latest_forecast,
    weather_metrics,
)
from radmuenster.data import DEFAULT_SITE, load_intervals, to_daily
from radmuenster.forecast import today_in_muenster
from radmuenster.published import read_forecast_log, read_status

REPO_URL = "https://github.com/annikahackenberg/muenster-cycling-data"
CHART_WEEKS = 8
# Width of a metric in pixels: two fit next to each other on a phone
METRIC_WIDTH = 160

# The app's background (dark: backgroundColor in .streamlit/config.toml), for
# the ring that separates markers from the line
SURFACE = {"light": "#ffffff", "dark": "#262624"}

# The cached data is reloaded at most once an hour
CACHE_SECONDS = 3600


@st.cache_data(ttl=CACHE_SECONDS)
def load_forecast_log() -> pd.DataFrame:
    return read_forecast_log()


@st.cache_data(ttl=CACHE_SECONDS)
def load_status() -> dict:
    return read_status()


@st.cache_data(ttl=CACHE_SECONDS)
def load_recent_counts(start) -> pd.DataFrame:
    """Daily counts from start on, from the city's monthly files."""
    return to_daily(load_intervals(DEFAULT_SITE, start=start))


theme = st.context.theme.type or "light"
today = today_in_muenster()

st.title("Bicycle traffic forecast")
st.write(
    "A daily forecast of bicycle traffic in Münster, built end to end from the city's open "
    "counting data: data cleaning, analysis, a machine learning model and an automated pipeline "
    "that updates this page every evening."
)
st.caption("Counting site Warendorfer Straße, Münster · trips per day, both directions")

log = load_forecast_log()
forecast = latest_forecast(log)
if is_stale(forecast, today):
    st.info(
        f"The daily forecast is paused. The latest forecast is from "
        f"{forecast['made_on']:%d %B %Y}."
    )

start = today - timedelta(weeks=CHART_WEEKS)
daily = load_recent_counts(start)
day = forecast["forecast_date"]
weekday = f"{day:%A}"
context = forecast_context(forecast, daily)


def trips(value: float | None) -> str:
    return "–" if value is None else f"{value:,.0f}"


def metric_row():
    """A row for metrics of fixed width that wraps when there is no space.
    st.columns would spread them over the whole screen (layout="wide") and
    put each in a line of its own on a phone."""
    return st.container(horizontal=True)


st.subheader(f"{day:%A, %d %B %Y}")
row = metric_row()
delta = None
if context["typical"] is not None:
    delta = f"{forecast['prediction'] / context['typical'] - 1:+.0%} vs. typical"
# delta_color="off": more or fewer trips is neither good nor bad
row.metric("Forecast", f"{forecast['prediction']:,}", delta=delta, delta_color="off", width=METRIC_WIDTH)
row.metric(
    f"Last {weekday}", trips(context["last_week"]),
    help="Counted trips one week earlier. Empty if that day was a public holiday or incomplete.",
    width=METRIC_WIDTH,
)
row.metric(
    f"Typical {weekday}", trips(context["typical"]),
    width=METRIC_WIDTH,
    help=(
        "Mean of the same weekday over the four weeks before, without public holidays and "
        "incomplete days."
    ),
)

st.markdown("**Weather forecast**")
metrics = weather_metrics(forecast)
row = metric_row()
for label, value in metrics.items():
    row.metric(label, value, width=METRIC_WIDTH)

notes = day_notes(day)
if notes:
    st.markdown("**Calendar:** " + " · ".join(notes))
else:
    st.markdown(
        "**Calendar:** a regular day: no public holiday, no school holidays, lectures at the "
        "university."
    )

st.subheader(f"The last {CHART_WEEKS} weeks")
data = chart_data(daily, log, start)
st.altair_chart(counts_chart(data, theme, SURFACE[theme]), width="stretch")
st.caption(
    "Days with an outage or incomplete data are left out. The city publishes each day's counts "
    "the next morning."
)
with st.expander("Table"):
    table = data.pivot_table(index="date", columns="series", values="trips").sort_index(
        ascending=False
    )
    st.dataframe(table.style.format("{:,.0f}", na_rep="–"), width="stretch")

st.subheader("How good are the forecasts?")
accuracy = forecast_accuracy(log, daily)
if accuracy["n"] < 7:
    st.write(
        f"The daily forecast started recently; {accuracy['n']} forecast day(s) can be checked "
        "so far. In the time-based cross-validation (2023 to 2026, measured weather), the model "
        f"was off by {CV_RESULTS[-1][2]:.1%} on average (WAPE)."
    )
else:
    row = metric_row()
    row.metric("Mean error", f"{accuracy['mae']:,.0f} trips", width=METRIC_WIDTH)
    row.metric("WAPE", f"{accuracy['wape']:.1%}", width=METRIC_WIDTH)
    row.metric("Bias", f"{accuracy['bias']:+.1%}", width=METRIC_WIDTH)
    st.caption(
        f"Over the last {accuracy['n']} forecast days with complete counts. WAPE: sum of errors "
        "divided by the sum of counted trips. Bias: positive means the forecasts were too high."
    )

status = load_status()
st.subheader("Monitoring")
if status["warnings"]:
    for warning in status["warnings"]:
        st.warning(warning, icon="⚠️")
else:
    st.success(f"All checks passed on {status['checked_on']}.", icon="✅")

st.subheader("How it works")
st.markdown("\n".join(f"{i}. **{step}:** {text}" for i, (step, text) in enumerate(HOW_IT_WORKS, 1)))
st.markdown(
    "**Model comparison** in a time-based cross-validation: each quarter from 2023 on is forecast "
    "by a model trained on all days before it, with the measured weather of the forecast day. The "
    "daily forecast only has a weather forecast, so its error will be somewhat higher."
)
st.dataframe(
    pd.DataFrame(CV_RESULTS, columns=["Model", "Mean error (trips)", "WAPE"]).style.format(
        {"Mean error (trips)": "{:,}", "WAPE": "{:.1%}"}
    ),
    hide_index=True,
    width="content",
)
st.caption(
    f"Code, analysis and model comparison: [GitHub]({REPO_URL}). "
    "Data: bicycle counts by the [City of Münster](https://github.com/od-ms/radverkehr-zaehlstellen), "
    "license [Datenlizenz Deutschland Namensnennung 2.0](https://www.govdata.de/dl-de/by-2-0). "
    "Weather: [Open-Meteo](https://open-meteo.com), "
    "license [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)."
)
