"""Forecast page: tomorrow's forecast and the recent weeks."""

from __future__ import annotations

import json
from datetime import timedelta
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
import streamlit as st

from radmuenster.dashboard import (
    chart_data,
    counts_chart,
    forecast_accuracy,
    is_stale,
    latest_forecast,
    weather_summary,
)
from radmuenster.data import DEFAULT_SITE, load_intervals, to_daily
from radmuenster.forecast import FORECAST_FILE, read_forecasts, today_in_muenster
from radmuenster.monitoring import STATUS_FILE

REPO_URL = "https://github.com/annikahackenberg/muenster-cycling-data"
# The daily job commits to main; reading from there gets the latest forecast
# without redeploying the page
RAW_URL = "https://raw.githubusercontent.com/annikahackenberg/muenster-cycling-data/main/"
CHART_WEEKS = 8

# The app's background (dark: backgroundColor in .streamlit/config.toml), for
# the ring that separates markers from the line
SURFACE = {"light": "#ffffff", "dark": "#262624"}

# The cached data is reloaded at most once an hour
CACHE_SECONDS = 3600


@st.cache_data(ttl=CACHE_SECONDS)
def load_forecast_log() -> pd.DataFrame:
    """The forecast log from main on GitHub, or the local file if it is not
    reachable (e.g. before the first merge)."""
    try:
        return read_forecasts(RAW_URL + FORECAST_FILE.as_posix())
    except URLError:
        return read_forecasts(FORECAST_FILE)


@st.cache_data(ttl=CACHE_SECONDS)
def load_status() -> dict:
    """The warnings of the latest monitoring run, like load_forecast_log."""
    try:
        with urlopen(RAW_URL + STATUS_FILE.as_posix(), timeout=10) as response:
            return json.load(response)
    except URLError:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))


@st.cache_data(ttl=CACHE_SECONDS)
def load_recent_counts(start) -> pd.DataFrame:
    """Daily counts from start on, from the city's monthly files."""
    return to_daily(load_intervals(DEFAULT_SITE, start=start))


theme = st.context.theme.type or "light"
today = today_in_muenster()

st.title("Bicycle traffic forecast")
st.caption("Counting site Warendorfer Straße, Münster · trips per day, both directions")

log = load_forecast_log()
forecast = latest_forecast(log)
if is_stale(forecast, today):
    st.info(
        f"The daily forecast is paused. The latest forecast is from "
        f"{forecast['made_on']:%d %B %Y}."
    )

st.metric(f"Forecast for {forecast['forecast_date']:%A, %d %B %Y}", f"{forecast['prediction']:,}")
st.caption(f"Weather forecast: {weather_summary(forecast)}")

start = today - timedelta(weeks=CHART_WEEKS)
daily = load_recent_counts(start)
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
        "was off by 8.9 % on average (WAPE)."
    )
else:
    cols = st.columns(3)
    cols[0].metric("Mean error", f"{accuracy['mae']:,.0f} trips")
    cols[1].metric("WAPE", f"{accuracy['wape']:.1%}")
    cols[2].metric("Bias", f"{accuracy['bias']:+.1%}")
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

with st.expander("About"):
    st.markdown(
        f"""
Every evening a scheduled job trains a LightGBM model on all counts since 2019 and forecasts the
next day, using the calendar (weekday, public holidays, university lecture periods, school
holidays), recent counts and the weather forecast. It also checks the data for outages and for
changes at the counting site. Code, analysis and model comparison: [GitHub]({REPO_URL}).

Data: bicycle counts by the [City of Münster](https://github.com/od-ms/radverkehr-zaehlstellen),
license [Datenlizenz Deutschland Namensnennung 2.0](https://www.govdata.de/dl-de/by-2-0).
Weather: [Open-Meteo](https://open-meteo.com),
license [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
"""
    )
