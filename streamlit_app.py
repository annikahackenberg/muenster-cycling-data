"""Streamlit app: tomorrow's forecast and the daily counts of all sites.

Run locally: streamlit run streamlit_app.py
The pages are in app_pages/.
"""

import streamlit as st

st.set_page_config(page_title="Münster bicycle traffic", page_icon="🚲")
page = st.navigation(
    [
        st.Page("app_pages/forecast.py", title="Forecast", default=True),
        st.Page("app_pages/explore.py", title="Explore all sites"),
    ]
)
page.run()
