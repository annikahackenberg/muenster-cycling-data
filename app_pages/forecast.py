"""Forecast page: tomorrow's forecast at all sites, and the details of one."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from radmuenster.dashboard import (
    CHECKED_HEADING,
    CV_RESULTS,
    HOW_IT_WORKS,
    MIN_ACCURACY_DAYS,
    chart_data,
    counts_chart,
    day_notes,
    forecast_accuracy,
    forecast_context,
    is_stale,
    latest_forecast,
    monitoring_headline,
    sites_overview,
    weather_metrics,
)
from radmuenster.data import DEFAULT_SITE, load_intervals, to_daily
from radmuenster.forecast import GLOBAL_MODEL_NAME, MODEL_NAME, today_in_muenster
from radmuenster.multi_site import site_counts_by_id
from radmuenster.published import read_backtest, read_forecast_log, read_sites_forecast_log, read_status
from radmuenster.sites import SITES_BY_ID
from radmuenster.static_page import DAILY_DATA, SITES_DATA, read_published

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
def load_sites_log() -> pd.DataFrame:
    return read_sites_forecast_log()


@st.cache_data(ttl=CACHE_SECONDS)
def load_backtest() -> pd.DataFrame:
    return read_backtest()


@st.cache_data(ttl=CACHE_SECONDS)
def load_site_counts() -> dict[str, pd.DataFrame]:
    """Daily counts of all sites by id, as published by the daily job."""
    return site_counts_by_id(read_published(SITES_DATA), read_published(DAILY_DATA))


@st.cache_data(ttl=CACHE_SECONDS)
def load_status() -> dict:
    return read_status()


@st.cache_data(ttl=CACHE_SECONDS)
def load_recent_counts(start) -> pd.DataFrame:
    """Daily counts from start on, from the city's monthly files."""
    return to_daily(load_intervals(DEFAULT_SITE, start=start))


theme = st.context.theme.type or "light"
today = today_in_muenster()


def trips(value: float | None) -> str:
    return "–" if value is None or pd.isna(value) else f"{value:,.0f}"


def metric_row():
    """A row for metrics of fixed width that wraps when there is no space.
    st.columns would spread them over the whole screen (layout="wide") and
    put each in a line of its own on a phone."""
    return st.container(horizontal=True)


def show_details(
    site_id: str, log: pd.DataFrame, daily: pd.DataFrame, backtest_wape: float, note: str = ""
) -> None:
    """Forecast, context, weather, the last weeks and the accuracy at one site.

    log holds the forecasts of this site by the model shown, daily its counts
    (count, complete), backtest_wape the error of its backtest (NaN without),
    note the overview's note on the site, e.g. why its chart has no counts.
    """
    forecast = latest_forecast(log)
    day = forecast["forecast_date"]
    weekday = f"{day:%A}"
    context = forecast_context(forecast, daily)
    st.caption(f"Counting site {SITES_BY_ID[site_id].name} · trips per day, both directions")
    if note:
        st.info(f"{note} Its days are flagged, so the chart shows no counts.", icon="ℹ️")

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
    row = metric_row()
    for label, value in weather_metrics(forecast).items():
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
    start = today - timedelta(weeks=CHART_WEEKS)
    data = chart_data(daily, log, start)
    st.altair_chart(counts_chart(data, theme, SURFACE[theme]), width="stretch")
    st.caption(
        "Days with an outage or incomplete data are left out. The city publishes each day's "
        "counts the next morning."
    )
    with st.expander("Table"):
        table = data.pivot_table(index="date", columns="series", values="trips").sort_index(
            ascending=False
        )
        st.dataframe(table.style.format("{:,.0f}", na_rep="–"), width="stretch")

    st.subheader("How good are the forecasts?")
    accuracy = forecast_accuracy(log, daily)
    if accuracy["n"] < MIN_ACCURACY_DAYS:
        text = (
            f"The daily forecast started recently; {accuracy['n']} forecast day(s) can be checked "
            "so far."
        )
        if not pd.isna(backtest_wape):
            text += (
                " In a backtest of the last 90 days, made afterwards with the weather forecast of "
                f"each evening before, it was off by {backtest_wape:.1%} on average (WAPE)."
            )
        if site_id == DEFAULT_SITE:
            text += (
                " In the time-based cross-validation (2023 to 2026, measured weather), the model "
                f"was off by {CV_RESULTS[-1][2]:.1%}."
            )
        st.write(text)
    else:
        row = metric_row()
        row.metric("Mean error", f"{accuracy['mae']:,.0f} trips", width=METRIC_WIDTH)
        row.metric("WAPE", f"{accuracy['wape']:.1%}", width=METRIC_WIDTH)
        row.metric("Bias", f"{accuracy['bias']:+.1%}", width=METRIC_WIDTH)
        st.caption(
            f"Over the last {accuracy['n']} forecast days with complete counts. WAPE: sum of "
            "errors divided by the sum of counted trips. Bias: positive means the forecasts were "
            "too high."
        )


st.title("Bicycle traffic forecast")
st.write(
    "A daily forecast of bicycle traffic at all 23 counting sites in Münster, built end to end "
    "from the city's open counting data: data cleaning, analysis, machine learning models and an "
    "automated pipeline that updates this page every evening."
)

log = load_forecast_log()
if is_stale(latest_forecast(log), today):
    st.info(
        f"The daily forecast is paused. The latest forecast is from "
        f"{latest_forecast(log)['made_on']:%d %B %Y}."
    )

sites_log = load_sites_log()
global_log = sites_log[sites_log["model"] == GLOBAL_MODEL_NAME]
backtest = load_backtest()
global_backtest = backtest[backtest["model"] == GLOBAL_MODEL_NAME]
own_backtest = backtest[backtest["model"] == MODEL_NAME]
counts = load_site_counts()

backtest_wapes, notes = {}, {}
if global_log.empty:
    st.info("Forecasts for all sites start with the next daily run; until then, only the Warendorfer Straße.")
    choices = [DEFAULT_SITE]
else:
    overview = sites_overview(global_log, log, counts, today, global_backtest, own_backtest)
    day = global_log["forecast_date"].max()
    st.subheader(f"{day:%A, %d %B %Y} at all sites")
    st.table(
        pd.DataFrame(
            {
                "Site": overview["site_name"] + overview["note"].map(lambda note: " *" if note else ""),
                "Forecast": overview["prediction"].map("{:,}".format),
                "vs. typical": overview["vs_typical"].map(lambda v: "–" if pd.isna(v) else f"{v:+.0%}"),
                "Error": [
                    "–" if pd.isna(error) else f"{error:.0%}" + (" ¹" if basis == "backtest" else "")
                    for error, basis in zip(overview["error"], overview["error_basis"])
                ],
            }
        ),
        hide_index=True,
    )
    st.caption(
        "vs. typical: compared with the same weekday over the four weeks before. Error: how far "
        "the forecasts were off on average (WAPE) in the last 4 weeks; ¹ until a site has a week "
        "of real forecasts, from a backtest of the last 90 days with the weather forecast of each "
        "evening. The Warendorfer Straße has its own model, all other sites share one."
    )
    for name, note in zip(overview["site_name"], overview["note"]):
        if note:
            st.caption(f"\\* {name}: {note}")
    choices = overview["site_id"].tolist()
    backtest_wapes = dict(zip(overview["site_id"], overview["error"].where(overview["error_basis"] == "backtest")))
    notes = dict(zip(overview["site_id"], overview["note"]))

site_id = st.selectbox(
    "Details for", choices, index=choices.index(DEFAULT_SITE) if DEFAULT_SITE in choices else 0,
    format_func=lambda site: SITES_BY_ID[site].name,
)
if site_id == DEFAULT_SITE:
    show_details(
        DEFAULT_SITE, log, load_recent_counts(today - timedelta(weeks=CHART_WEEKS)),
        backtest_wapes.get(DEFAULT_SITE, float("nan")),
    )
else:
    show_details(
        site_id, global_log[global_log["site_id"] == site_id], counts[site_id],
        backtest_wapes.get(site_id, float("nan")), notes.get(site_id, ""),
    )

status = load_status()
st.subheader("Monitoring")
headline = monitoring_headline(
    len(status["warnings"]), len(status["checked_warnings"]), date.fromisoformat(status["checked_on"])
)
if status["warnings"]:
    st.markdown(headline)
    for warning in status["warnings"]:
        st.warning(warning, icon="⚠️")
else:
    st.success(headline, icon="✅")
if status["checked_warnings"]:
    st.markdown(CHECKED_HEADING)
    for warning in status["checked_warnings"]:
        st.info(warning, icon="🔎")

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
