"""Static web page for GitHub Pages, built by the daily job.

Unlike the Streamlit page, it never sleeps: it is a single HTML file, rebuilt
every evening with the new forecast. The chart is the same Altair chart,
embedded as a Vega-Lite spec and drawn in the browser.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from html import escape
from pathlib import Path
from string import Template

import pandas as pd

from radmuenster.dashboard import (
    COUNTED,
    FORECAST,
    chart_data,
    counts_chart,
    forecast_accuracy,
    is_stale,
    latest_forecast,
    weather_summary,
)

SITE_DIR = Path("site")
CHART_WEEKS = 8

REPO_URL = "https://github.com/annikahackenberg/muenster-cycling-data"
STREAMLIT_URL = "https://muenster-cycling-data.streamlit.app"

# Exact versions of the chart libraries, matching the Vega-Lite schema of Altair
VEGA_SCRIPTS = (
    "https://cdn.jsdelivr.net/npm/vega@6.4.0",
    "https://cdn.jsdelivr.net/npm/vega-lite@6.4.3",
    "https://cdn.jsdelivr.net/npm/vega-embed@7.2.0",
)

# Page colors per theme; surface and text as in the validated palette
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#3a3a37"},
}

# Time-based cross-validation (notebook 02), 1,232 test days 2023 to 2026
CV_RESULTS = (
    ("Naive baseline (same weekday last week)", 960, 0.161),
    ("Holiday baseline", 848, 0.143),
    ("Linear regression on log(count)", 646, 0.109),
    ("LightGBM on log(count)", 529, 0.089),
)

# Minimum number of checked forecast days before the page shows their error
MIN_ACCURACY_DAYS = 7


def chart_spec(data: pd.DataFrame, theme: str) -> dict:
    """The Vega-Lite spec of the counts chart in the page colors of theme."""
    colors = THEMES[theme]
    chart = (
        counts_chart(data, theme, colors["surface"])
        .properties(width="container")
        .configure(background="transparent")
        .configure_axis(
            labelColor=colors["muted"],
            titleColor=colors["muted"],
            gridColor=colors["grid"],
            domainColor=colors["grid"],
            tickColor=colors["grid"],
            labelFontSize=12,
            titleFontSize=12,
        )
        .configure_legend(labelColor=colors["text"], labelFontSize=13)
        .configure_view(stroke=None)
    )
    return chart.to_dict()


def _json_for_script(value) -> str:
    """JSON that is safe inside a <script> element: "</" would end it."""
    return json.dumps(value).replace("</", "<\\/")


def _number(value: float) -> str:
    return "–" if pd.isna(value) else f"{value:,.0f}"


def _table_rows(data: pd.DataFrame) -> str:
    """Rows of the table view, newest day first."""
    wide = data.pivot_table(index="date", columns="series", values="trips", dropna=False)
    rows = []
    for day, values in wide.sort_index(ascending=False).iterrows():
        counted = _number(values.get(COUNTED, float("nan")))
        forecast = _number(values.get(FORECAST, float("nan")))
        rows.append(f"<tr><td>{day:%a %d %b %Y}</td><td>{counted}</td><td>{forecast}</td></tr>")
    return "\n".join(rows)


def _accuracy_section(log: pd.DataFrame, daily: pd.DataFrame) -> str:
    accuracy = forecast_accuracy(log, daily)
    if accuracy["n"] < MIN_ACCURACY_DAYS:
        return (
            f"<p>The daily forecast started on 12 October 2026; {accuracy['n']} forecast day(s) "
            "can be checked so far. The table below shows the error in the time-based "
            "cross-validation, which used the measured weather.</p>"
        )
    return f"""<div class="tiles">
  <div class="tile"><div class="label">Mean error</div><div class="value">{accuracy['mae']:,.0f}</div><div class="unit">trips per day</div></div>
  <div class="tile"><div class="label">WAPE</div><div class="value">{accuracy['wape']:.1%}</div><div class="unit">of the counted trips</div></div>
  <div class="tile"><div class="label">Bias</div><div class="value">{accuracy['bias']:+.1%}</div><div class="unit">positive: too high</div></div>
</div>
<p class="note">Over {accuracy['n']} forecast days with complete counts. These forecasts used the
weather forecast, so this is the honest error of the daily forecast.</p>"""


def _monitoring_section(warnings: list[str], today: date) -> str:
    if not warnings:
        return f'<p class="status ok"><span aria-hidden="true">✓</span> All checks passed on {today:%d %B %Y}.</p>'
    items = "\n".join(f"<li>{escape(w)}</li>" for w in warnings)
    return f'<div class="status warn"><p><span aria-hidden="true">!</span> Warnings on {today:%d %B %Y}:</p><ul>{items}</ul></div>'


def render_page(daily: pd.DataFrame, log: pd.DataFrame, warnings: list[str], today: date) -> str:
    """The whole page as HTML.

    daily is the output of to_daily, log the forecast log, warnings the
    result of run_checks, today the date of the run in Münster.
    """
    forecast = latest_forecast(log)
    start = today - timedelta(weeks=CHART_WEEKS)
    data = chart_data(daily, log, start)
    stale = ""
    if is_stale(forecast, today):
        stale = (
            '<p class="status warn">The daily forecast is paused. This is the latest forecast, '
            f"made on {forecast['made_on']:%d %B %Y}.</p>"
        )
    cv_rows = "\n".join(
        f"<tr><td>{name}</td><td>{mae:,}</td><td>{wape:.1%}</td></tr>" for name, mae, wape in CV_RESULTS
    )
    return PAGE.substitute(
        forecast_day=f"{forecast['forecast_date']:%A, %d %B %Y}",
        prediction=f"{forecast['prediction']:,}",
        weather=escape(weather_summary(forecast)),
        made_on=f"{forecast['made_on']:%d %B %Y}",
        stale=stale,
        weeks=CHART_WEEKS,
        specs=_json_for_script({t: chart_spec(data, t) for t in THEMES}),
        table_rows=_table_rows(data),
        accuracy=_accuracy_section(log, daily),
        cv_rows=cv_rows,
        monitoring=_monitoring_section(warnings, today),
        scripts="\n".join(f'<script src="{url}"></script>' for url in VEGA_SCRIPTS),
        repo=REPO_URL,
        streamlit=STREAMLIT_URL,
        **{f"{t}_{k}": v for t, colors in THEMES.items() for k, v in colors.items()},
    )


def write_site(html: str, site_dir: Path = SITE_DIR) -> Path:
    """Save the page as index.html in site_dir, the folder GitHub Pages publishes."""
    site_dir.mkdir(parents=True, exist_ok=True)
    path = site_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


# $name are placeholders for Template.substitute; a literal dollar sign would be $$
PAGE = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Münster Bicycle Forecast</title>
<meta name="description" content="Daily forecast of bicycle trips at the Warendorfer Straße counting site in Münster, with a LightGBM model retrained every evening.">
<style>
:root {
  color-scheme: light;
  --surface: $light_surface; --text: $light_text; --muted: $light_muted; --grid: $light_grid;
  --ok: #1a7f37; --warn: #9a6700;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --surface: $dark_surface; --text: $dark_text; --muted: $dark_muted; --grid: $dark_grid;
    --ok: #3fb950; --warn: #d29922;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface); color: var(--text);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 760px; margin: 0 auto; padding: 32px 16px 48px; }
h1 { font-size: 2rem; line-height: 1.2; margin: 0 0 4px; }
h2 { font-size: 1.25rem; margin: 40px 0 8px; }
a { color: inherit; }
.sub, .note, .unit, .label, footer { color: var(--muted); }
.sub { margin: 0 0 28px; }
.note { font-size: 0.9rem; }
.hero .label { font-size: 0.95rem; }
.hero .value { font-size: 3rem; font-weight: 650; line-height: 1.1; font-variant-numeric: tabular-nums; }
.hero .unit { margin-bottom: 4px; }
#chart { width: 100%; min-height: 360px; }
details { margin-top: 8px; }
summary { cursor: pointer; color: var(--muted); }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 500; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.tile { border: 1px solid var(--grid); border-radius: 8px; padding: 12px 14px; }
.tile .value { font-size: 1.6rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.status { border-left: 4px solid; padding: 8px 12px; border-radius: 4px; }
.status.ok { border-color: var(--ok); }
.status.warn { border-color: var(--warn); }
.status span { font-weight: 700; margin-right: 6px; }
.status.ok span { color: var(--ok); }
.status.warn span { color: var(--warn); }
ol li { margin-bottom: 6px; }
footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--grid); font-size: 0.85rem; }
</style>
</head>
<body>
<main>
<h1>Bicycle traffic forecast</h1>
<p class="sub">Counting site Warendorfer Straße, Münster · trips per day, both directions</p>

$stale
<section class="hero">
  <div class="label">Forecast for $forecast_day</div>
  <div class="value">$prediction</div>
  <div class="unit">bicycle trips</div>
  <p class="note">Weather forecast: $weather. Made on $made_on, updated every evening.</p>
</section>

<h2>The last $weeks weeks</h2>
<div id="chart" role="img" aria-label="Line chart of the counted trips per day and dots for the forecasts; the same values are in the table below."></div>
<p class="note">Days with an outage or incomplete data are left out. The city publishes each day's counts the next morning.</p>
<details>
  <summary>Table</summary>
  <table>
    <thead><tr><th>Date</th><th>Counted</th><th>Forecast</th></tr></thead>
    <tbody>
$table_rows
    </tbody>
  </table>
</details>

<h2>How good are the forecasts?</h2>
$accuracy
<table>
  <thead><tr><th>Model (cross-validation, measured weather)</th><th>MAE</th><th>WAPE</th></tr></thead>
  <tbody>
$cv_rows
  </tbody>
</table>
<p class="note">Each quarter from 2023 on was forecast by a model trained only on the days before it
(1,232 test days). MAE: mean error in trips per day. WAPE: sum of errors divided by the sum of counted trips.</p>

<h2>Monitoring</h2>
$monitoring
<p class="note">Every evening the data is checked for outages, for jumps in the share of trips into the city
(a sign that the counting site changed) and for forecasts that are off in the same direction for weeks.</p>

<h2>How it works</h2>
<ol>
  <li><strong>Data:</strong> the City of Münster publishes 15-minute counts every morning. They are summed to
  daily totals; outages that the city's status misses are detected by comparing the two directions.</li>
  <li><strong>Features:</strong> weekday, public holidays and bridge days in NRW, lecture periods of the
  University of Münster, school holidays, weather, and counts from 2, 7, 14 and 364 days earlier. Lag 1 is
  left out on purpose: when the forecast is made, today is not over yet.</li>
  <li><strong>Model:</strong> gradient-boosted trees (LightGBM) on the logarithm of the count, retrained every
  evening on all complete days since 2019.</li>
  <li><strong>Automation:</strong> a scheduled GitHub Actions workflow forecasts tomorrow, runs the
  monitoring, stores the forecast with its weather forecast and rebuilds this page.</li>
</ol>
<p>Code, exploratory analysis and model comparison: <a href="$repo">GitHub</a>.
An interactive version of this page runs on <a href="$streamlit">Streamlit</a>
(it may take a moment to wake up).</p>

<footer>
Bicycle counts: <a href="https://github.com/od-ms/radverkehr-zaehlstellen">City of Münster</a>,
license <a href="https://www.govdata.de/dl-de/by-2-0">Datenlizenz Deutschland Namensnennung 2.0</a>.
Weather: <a href="https://open-meteo.com">Open-Meteo</a>,
license <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
</footer>
</main>

<script id="chart-specs" type="application/json">$specs</script>
$scripts
<script>
const specs = JSON.parse(document.getElementById("chart-specs").textContent);
const dark = window.matchMedia("(prefers-color-scheme: dark)");
function draw() {
  vegaEmbed("#chart", dark.matches ? specs.dark : specs.light, { actions: false });
}
dark.addEventListener("change", draw);
draw();
</script>
</body>
</html>
""")
