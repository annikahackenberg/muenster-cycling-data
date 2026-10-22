"""Streamlit app: tomorrow's forecast and the daily counts of all sites.

Run locally: streamlit run streamlit_app.py
The pages are in app_pages/.
"""

import streamlit as st

from radmuenster.dashboard import APP_CONTENT_WIDTH

# Wide: the charts and the controls of the explore page need the width; the
# default centered layout limits the content to about 730 pixels
st.set_page_config(page_title="Münster bicycle traffic", page_icon="🚲", layout="wide")
# ... but not the whole width of a large screen. The 10rem are the side
# padding of the wide layout (5rem on each side)
st.html(f"""<style>
[data-testid="stMainBlockContainer"] {{ max-width: calc({APP_CONTENT_WIDTH}px + 10rem); }}
</style>""")
page = st.navigation(
    [
        st.Page("app_pages/forecast.py", title="Forecast", default=True),
        st.Page("app_pages/findings.py", title="Insights", url_path="insights"),
        st.Page("app_pages/explore.py", title="Explore all sites"),
        st.Page("app_pages/assistant.py", title="Leeze AI", url_path="leeze-ai"),
    ]
)
page.run()
