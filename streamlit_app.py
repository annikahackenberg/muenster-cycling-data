"""Streamlit app: tomorrow's forecast and the daily counts of all sites.

Run locally: streamlit run streamlit_app.py
The pages are in app_pages/.
"""

import streamlit as st

# Wide: the charts and the controls of the explore page need the width; the
# default centered layout limits the content to about 730 pixels
st.set_page_config(page_title="Münster bicycle traffic", page_icon="🚲", layout="wide")
page = st.navigation(
    [
        st.Page("app_pages/forecast.py", title="Forecast", default=True),
        st.Page("app_pages/findings.py", title="What the data shows"),
        st.Page("app_pages/explore.py", title="Explore all sites"),
    ]
)
page.run()
