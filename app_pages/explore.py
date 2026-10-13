"""Explore page: the daily counts of all counting sites, as published."""

from __future__ import annotations

from datetime import timedelta
from io import BytesIO

import pandas as pd
import requests
import streamlit as st

from radmuenster.explore import (
    MAX_SITES,
    RESOLUTIONS,
    SITES_DATA_URL,
    aggregate,
    assign_colors,
    flagged_periods,
    ranking,
    ranking_chart,
    select,
    sites_chart,
)
from radmuenster.static_page import SITE_DIR, SITES_DATA

DEFAULT_SITES = ["Warendorfer Straße", "Neutor", "Wolbecker Straße"]

# The cached data is reloaded at most once an hour
CACHE_SECONDS = 3600


@st.cache_data(ttl=CACHE_SECONDS)
def load_sites_daily() -> pd.DataFrame:
    """Daily counts of all sites from GitHub Pages, where the daily job
    publishes them, or the local file of the last local run."""
    try:
        response = requests.get(SITES_DATA_URL, timeout=30)
        response.raise_for_status()
        return pd.read_parquet(BytesIO(response.content))
    except requests.RequestException:
        return pd.read_parquet(SITE_DIR / SITES_DATA)


theme = st.context.theme.type or "light"
sites_daily = load_sites_daily()
first_day, last_day = sites_daily["date"].min().date(), sites_daily["date"].max().date()

st.title("Explore all counting sites")
st.caption(
    f"Daily counts of the city's {sites_daily['site_id'].nunique()} counting sites as published, "
    f"{first_day:%B %Y} to {last_day:%d %B %Y}. Suspicious days are not changed, only marked."
)

all_sites = sorted(sites_daily["site_name"].unique())
selected = st.multiselect(
    "Counting sites", all_sites, default=DEFAULT_SITES, max_selections=MAX_SITES,
    help=f"Up to {MAX_SITES} sites at a time, so that each keeps a distinct color.",
)
# Two columns, the toggle below: with three, the resolution had no room for
# its three buttons side by side in a narrow window
cols = st.columns(2)
period = cols[0].date_input(
    "Period", value=(last_day - timedelta(days=365), last_day), min_value=first_day, max_value=last_day
)
resolution = cols[1].segmented_control("Resolution", list(RESOLUTIONS), default="Week")
leave_out = st.toggle(
    "Hide flagged days", value=False,
    help="Off: the counts as published. On: flagged days are left out of the means and lines.",
)

# While the second date is being picked, date_input returns only one
if not selected or len(period) != 2 or resolution is None:
    st.info("Choose at least one site, a start and an end date, and a resolution.")
    st.stop()
start, end = period

rows = select(sites_daily, selected, start, end)
data = aggregate(rows, resolution, leave_out_flagged=leave_out)
st.session_state["colors"] = assign_colors(selected, st.session_state.get("colors", {}))
st.altair_chart(sites_chart(data, st.session_state["colors"], theme, resolution), width="stretch")
st.caption(
    "Hollow rings mark periods with flagged days (apart from the shortened last day of each "
    "month); hover over them for the reason. "
    + ("Weeks and months show the mean per day. " if resolution != "Day" else "")
    + "Not flagged: changes in how a site counts, such as the jump at the Warendorfer Straße "
    "in October 2025."
)

st.subheader("Flagged periods")
periods = flagged_periods(rows)
if periods.empty:
    st.write("No flagged days for these sites in this period.")
else:
    st.dataframe(
        periods,
        hide_index=True,
        width="stretch",
        column_config={
            "site_name": "Site",
            "first": st.column_config.DateColumn("From", format="D MMM YYYY"),
            "last": st.column_config.DateColumn("To", format="D MMM YYYY"),
            "days": "Days",
            "reason": "Reason",
        },
    )
    st.caption(
        "The last day of each month is left out here: the city cuts it short every month "
        "(and left it out completely from 2021 to 2025)."
    )

st.subheader("Which sites are busiest?")
table = ranking(sites_daily, start, end)
st.altair_chart(ranking_chart(table, theme), width="stretch")
unusable = table.loc[table["trips"].isna(), "site_name"].tolist()
st.caption(
    f"Mean trips per day from {start:%d %b %Y} to {end:%d %b %Y}, from days without flags only. "
    "Grey bars and \"few days\": less than 80% of the period could be used, e.g. because the site "
    "started later or had a long outage, so the mean is less reliable."
    + (f" No usable days: {', '.join(unusable)}." if unusable else "")
)

st.caption(
    "Data: bicycle counts by the [City of Münster](https://github.com/od-ms/radverkehr-zaehlstellen), "
    "license [Datenlizenz Deutschland Namensnennung 2.0](https://www.govdata.de/dl-de/by-2-0). "
    f"Download: [sites_daily.parquet]({SITES_DATA_URL})."
)
