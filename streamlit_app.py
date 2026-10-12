"""Streamlit page: tomorrow's forecast and the recent weeks.

Run locally: streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from datetime import timedelta
from urllib.error import URLError
from urllib.request import urlopen

import altair as alt
import pandas as pd
import streamlit as st

from radmuenster.dashboard import (
    COUNTED,
    FORECAST,
    chart_data,
    forecast_accuracy,
    is_stale,
    latest_forecast,
)
from radmuenster.data import DEFAULT_SITE, load_intervals, to_daily
from radmuenster.forecast import FORECAST_FILE, read_forecasts, today_in_muenster
from radmuenster.monitoring import STATUS_FILE

REPO_URL = "https://github.com/annikahackenberg/muenster-cycling-data"
# The daily job commits to main; reading from there gets the latest forecast
# without redeploying the page
RAW_URL = "https://raw.githubusercontent.com/annikahackenberg/muenster-cycling-data/main/"
CHART_WEEKS = 8

# Categorical slots 1 and 2 of the validated palette, per theme
SERIES_COLORS = {
    "light": {COUNTED: "#2a78d6", FORECAST: "#eb6834"},
    "dark": {COUNTED: "#3987e5", FORECAST: "#d95926"},
}
# Streamlit's background, for the ring that separates markers from the line
SURFACE = {"light": "#ffffff", "dark": "#0e1117"}

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


def weather_summary(forecast: pd.Series) -> str:
    return (
        f"{forecast['temp_max']:.0f} °C max · {forecast['precipitation']:.1f} mm rain "
        f"over {forecast['precipitation_hours']:.0f} h · {forecast['sunshine']:.0f} h sun · "
        f"wind up to {forecast['wind_max']:.0f} km/h"
    )


def counts_chart(data: pd.DataFrame, theme: str) -> alt.LayerChart:
    """Counted trips as a line, forecasts as dots, with a hover rule."""
    colors = SERIES_COLORS[theme]
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=list(colors), range=list(colors.values())),
        legend=alt.Legend(orient="top", title=None),
    )
    base = alt.Chart(data).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", grid=False)),
        y=alt.Y("trips:Q", title="Trips per day", scale=alt.Scale(zero=True)),
        color=color,
    )
    # invalid=None: a missing day breaks the line instead of being bridged
    line = base.transform_filter(alt.datum.series == COUNTED).mark_line(
        strokeWidth=2, invalid=None
    )
    dots = base.transform_filter(alt.datum.series == FORECAST).mark_point(
        filled=True, size=90, opacity=1, stroke=SURFACE[theme], strokeWidth=2
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


st.set_page_config(page_title="Münster bicycle forecast", page_icon="🚲")
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
st.altair_chart(counts_chart(data, theme), width="stretch")
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
