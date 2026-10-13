import json
import re
from datetime import date

import numpy as np
import pandas as pd

import requests

from radmuenster import static_page
from radmuenster.static_page import (
    SITES_DATA,
    _json_for_script,
    read_published,
    render_page,
    write_data,
    write_site,
)
from radmuenster.weather import DAILY_VARIABLES

TODAY = date(2026, 10, 12)


def make_daily():
    """Two years of complete days, enough for every finding."""
    dates = pd.date_range("2024-01-01", "2026-10-11", freq="D", name="date")
    count = np.where(dates.dayofweek < 5, 6000.0, 3000.0)
    return pd.DataFrame(
        {
            "count": count,
            "count_in": count * 0.5,
            "complete": True,
            "direction_fault": False,
            "n_flagged": 0,
        },
        index=dates,
    )


def make_weather():
    dates = pd.date_range("2024-01-01", "2026-10-13", freq="D", name="date")
    hours = np.tile([0.0, 2.0, 5.0, 10.0], len(dates) // 4 + 1)[: len(dates)]
    return pd.DataFrame({"precipitation_hours": hours}, index=dates)


def render(warnings=(), forecast_date="2026-10-13"):
    return render_page(make_daily(), make_log(forecast_date), list(warnings), TODAY, make_weather())


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
    html = render()
    assert "Forecast for Tuesday, 13 October 2026" in html
    assert "7,545" in html
    assert "All checks passed" in html
    specs = chart_specs(html)
    assert set(specs) == {"counts", "weekday", "month", "rain", "share"}
    assert all(set(themes) == {"light", "dark"} for themes in specs.values())
    # Every chart has its element on the page
    assert all(f'id="chart-{name}"' in html for name in specs)
    # Numbers in the text come from the data: Sunday is half a workday here
    assert "On Sundays about 50% as many people cycle" in html
    # Every placeholder of the template was filled; "$schema" belongs to the
    # two Vega-Lite specs
    assert set(re.findall(r"\$[a-z_]+", html)) == {"$schema"}


def test_page_escapes_warnings():
    html = render(warnings=["Share <0.40> & falling"])
    assert "Share &lt;0.40&gt; &amp; falling" in html


def test_page_says_when_the_forecast_is_old():
    html = render(forecast_date="2026-10-05")
    assert "The daily forecast is paused" in html


def test_json_for_script_cannot_end_the_script_element():
    text = _json_for_script({"label": "</script><script>alert(1)"})
    assert "</" not in text
    assert json.loads(text) == {"label": "</script><script>alert(1)"}


def test_write_site_creates_index(tmp_path):
    path = write_site("<html></html>", tmp_path / "site")
    assert path.read_text(encoding="utf-8") == "<html></html>"
    assert path.name == "index.html"


def test_page_text_without_unflagged_outages():
    html = render()
    assert "outages it misses are found by comparing the two directions" in html


def test_write_data_next_to_the_page(tmp_path):
    table = pd.DataFrame({"site_id": ["1"], "date": [pd.Timestamp("2026-10-11")], "total": [10.0]})
    path = write_data(table, SITES_DATA, tmp_path)
    assert path == tmp_path / "data" / "sites_daily.parquet"
    assert pd.read_parquet(path).equals(table)


def test_write_data_keeps_the_date_index(tmp_path):
    daily = make_daily()
    path = write_data(daily, SITES_DATA, tmp_path)
    assert pd.read_parquet(path).equals(daily)


def test_read_published_falls_back_to_the_local_file(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(static_page.requests, "get", offline)
    table = pd.DataFrame({"total": [1.0, 2.0]})
    write_data(table, SITES_DATA, tmp_path)
    assert read_published(SITES_DATA, tmp_path).equals(table)
