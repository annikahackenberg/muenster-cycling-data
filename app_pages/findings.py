"""Insights page: the key findings from the counts at the Warendorfer Straße."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from radmuenster.dashboard import APP_CONTENT_WIDTH
from radmuenster.findings import key_findings, site_findings, site_profiles
from radmuenster.multi_site import site_counts_by_id
from radmuenster.static_page import DAILY_DATA, SITES_DATA, WEATHER_DATA, read_published

REPO_URL = "https://github.com/annikahackenberg/muenster-cycling-data"
NOTEBOOK_URL = f"{REPO_URL}/blob/main/notebooks/01_exploratory_analysis.ipynb"

# The cached data is reloaded at most once an hour
CACHE_SECONDS = 3600


@st.cache_data(ttl=CACHE_SECONDS)
def load_site_counts(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily counts of all sites by id, as published by the daily job."""
    return site_counts_by_id(read_published(SITES_DATA), daily)


@st.cache_data(ttl=CACHE_SECONDS)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily counts of the Warendorfer Straße and the weather, published by
    the daily job: all data since 2019, without downloading it from the city."""
    return read_published(DAILY_DATA), read_published(WEATHER_DATA)


theme = st.context.theme.type or "light"
try:
    daily, weather = load_data()
except FileNotFoundError:
    # Neither on GitHub Pages nor from a local run, e.g. right after this
    # page was added and before the next daily job
    st.title("Insights")
    st.info("The findings appear after the next daily run this evening.")
    st.stop()

st.title("Insights")
st.write(
    f"Key findings from the counts at the Warendorfer Straße since {daily.index.min():%B %Y}, "
    "recomputed every evening from all data, so the numbers stay current. They explain what the "
    "forecast model has to know: the weekday, the season, the weather, and where the data "
    "itself cannot be trusted."
)

for finding in key_findings(daily, weather):
    st.subheader(finding.title)
    # A fixed width in pixels for charts that need less than the full width:
    # in a narrow window the chart gets the whole width instead of shrinking
    # with a column
    width = round(finding.width * APP_CONTENT_WIDTH) if finding.width < 1 else "stretch"
    st.altair_chart(finding.chart(theme), width=width)
    st.write(finding.text)

st.caption(
    f"Details, more plots and the reasoning behind them: [exploratory analysis]({NOTEBOOK_URL}) "
    "on GitHub. Only complete days are used."
)

st.header("All counting sites")
st.write(
    "The same methods applied to all 23 counting sites of the city, from their complete days only. "
    "They explain why the forecasts differ so much between sites on the same day."
)
for finding in site_findings(site_profiles(load_site_counts(daily), weather)):
    st.subheader(finding.title)
    st.altair_chart(finding.chart(theme), width="stretch")
    st.write(finding.text)
