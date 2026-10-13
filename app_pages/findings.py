"""Findings page: what the counts at the Warendorfer Straße show."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from radmuenster.findings import key_findings
from radmuenster.static_page import DAILY_DATA, WEATHER_DATA, read_published

REPO_URL = "https://github.com/annikahackenberg/muenster-cycling-data"
NOTEBOOK_URL = f"{REPO_URL}/blob/main/notebooks/01_exploratory_analysis.ipynb"

# The cached data is reloaded at most once an hour
CACHE_SECONDS = 3600


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
    st.title("What the data shows")
    st.info("The findings appear after the next daily run this evening.")
    st.stop()

st.title("What the data shows")
st.write(
    f"Key findings from the counts at the Warendorfer Straße since {daily.index.min():%B %Y}, "
    "recomputed every evening from all data, so the numbers stay current. They explain what the "
    "forecast model has to know: the weekday, the season, the weather, and where the data "
    "itself cannot be trusted."
)

for finding in key_findings(daily, weather):
    st.subheader(finding.title)
    # A narrower column on wide screens; on a phone the columns stack and
    # the chart gets the full width
    column = st.columns([finding.width, 1 - finding.width])[0] if finding.width < 1 else st
    column.altair_chart(finding.chart(theme), width="stretch")
    st.write(finding.text)

st.caption(
    f"Details, more plots and the reasoning behind them: [exploratory analysis]({NOTEBOOK_URL}) "
    "on GitHub. Only complete days are used."
)
