import json
import re
from datetime import date

import pandas as pd

from radmuenster.static_page import _json_for_script, render_page, write_site
from radmuenster.weather import DAILY_VARIABLES

TODAY = date(2026, 10, 12)


def make_daily():
    dates = pd.date_range("2026-08-01", "2026-10-11", freq="D", name="date")
    return pd.DataFrame({"count": 6000.0, "complete": True}, index=dates)


def make_log(forecast_date: str = "2026-10-13") -> pd.DataFrame:
    log = pd.DataFrame(
        {
            "forecast_date": [pd.Timestamp(forecast_date)],
            "made_on": [pd.Timestamp(forecast_date) - pd.Timedelta(days=1)],
            "model": ["lightgbm"],
            "prediction": [7545],
        }
    )
    for column in DAILY_VARIABLES.values():
        log[column] = 1.0
    return log


def chart_specs(html: str) -> dict:
    match = re.search(r'<script id="chart-specs" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(match.group(1))


def test_page_shows_forecast_and_both_chart_themes():
    html = render_page(make_daily(), make_log(), [], TODAY)
    assert "Forecast for Tuesday, 13 October 2026" in html
    assert "7,545" in html
    assert "All checks passed" in html
    assert set(chart_specs(html)) == {"light", "dark"}
    # Every placeholder of the template was filled; "$schema" belongs to the
    # two Vega-Lite specs
    assert set(re.findall(r"\$[a-z_]+", html)) == {"$schema"}


def test_page_escapes_warnings():
    html = render_page(make_daily(), make_log(), ["Share <0.40> & falling"], TODAY)
    assert "Share &lt;0.40&gt; &amp; falling" in html


def test_page_says_when_the_forecast_is_old():
    html = render_page(make_daily(), make_log("2026-10-05"), [], TODAY)
    assert "The daily forecast is paused" in html


def test_json_for_script_cannot_end_the_script_element():
    text = _json_for_script({"label": "</script><script>alert(1)"})
    assert "</" not in text
    assert json.loads(text) == {"label": "</script><script>alert(1)"}


def test_write_site_creates_index(tmp_path):
    path = write_site("<html></html>", tmp_path / "site")
    assert path.read_text(encoding="utf-8") == "<html></html>"
    assert path.name == "index.html"
