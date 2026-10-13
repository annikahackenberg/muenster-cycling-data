"""Static web page for GitHub Pages, built by the daily job.

Unlike the Streamlit page, it never sleeps: it is a single HTML file, rebuilt
every evening with the new forecast and the key findings of the analysis.
The charts are Altair charts, embedded as Vega-Lite specs and drawn in the
browser.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from html import escape
from io import BytesIO
from pathlib import Path
from string import Template

import altair as alt
import pandas as pd
import requests

from radmuenster import findings
from radmuenster.data import DEFAULT_SITE
from radmuenster.dashboard import (
    COUNTED,
    CV_RESULTS,
    FORECAST,
    HOW_IT_WORKS,
    chart_data,
    counts_chart,
    day_notes,
    forecast_accuracy,
    forecast_context,
    is_stale,
    latest_forecast,
    weather_metrics,
)

SITE_DIR = Path("site")
PAGES_URL = "https://annikahackenberg.github.io/muenster-cycling-data/"
# Data files published next to the page (see write_data), for the Streamlit
# app: the daily counts of all sites, and for the findings the daily counts of
# the forecast site and the weather
SITES_DATA = Path("data/sites_daily.parquet")
DAILY_DATA = Path(f"data/daily_{DEFAULT_SITE}.parquet")
WEATHER_DATA = Path("data/weather.parquet")
CHART_WEEKS = 8

REPO_URL = "https://github.com/annikahackenberg/muenster-cycling-data"
STREAMLIT_URL = "https://muenster-cycling-data.streamlit.app"

# Exact versions of the chart libraries, matching the Vega-Lite schema of Altair
VEGA_SCRIPTS = (
    "https://cdn.jsdelivr.net/npm/vega@6.4.0",
    "https://cdn.jsdelivr.net/npm/vega-lite@6.4.3",
    "https://cdn.jsdelivr.net/npm/vega-embed@7.2.0",
)

# Page colors per theme. The dark theme avoids pure white on near black
# (contrast 17:1 glares): a lighter surface with an off-white text (9.5:1,
# muted 5.6:1). The series colors keep at least 3.9:1 against it.
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"},
    "dark": {"surface": "#262624", "text": "#cfcdc6", "muted": "#9f9d95", "grid": "#44443f"},
}

# Minimum number of checked forecast days before the page shows their error
MIN_ACCURACY_DAYS = 7


def themed_spec(chart: alt.TopLevelMixin, theme: str) -> dict:
    """The Vega-Lite spec of chart, full width, in the page colors of theme."""
    colors = THEMES[theme]
    chart = (
        chart.properties(width="container")
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


def _findings_section(daily: pd.DataFrame, weather: pd.DataFrame) -> tuple[str, dict]:
    """HTML of the key findings and the specs of their charts (per theme)."""
    items = findings.key_findings(daily, weather)
    html = "\n".join(
        f"<h3>{escape(item.title)}</h3>\n"
        f'<div id="chart-{item.name}" class="chart small" style="--chart-width: {item.width:.0%}" '
        f'role="img" aria-label="{escape(item.description)}"></div>\n'
        f"<p>{escape(item.text)}</p>\n"
        for item in items
    )
    specs = {item.name: {t: themed_spec(item.chart(t), t) for t in THEMES} for item in items}
    return html, specs


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


def _tile(label: str, value: str, unit: str = "") -> str:
    unit_html = f'<div class="unit">{escape(unit)}</div>' if unit else ""
    return (
        f'  <div class="tile"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>{unit_html}</div>'
    )


def _context_section(forecast: pd.Series, daily: pd.DataFrame) -> tuple[str, str]:
    """The forecast in context, as on the forecast page of the app: the
    deviation from a typical day (after the number), and the tiles for the
    same weekday, the weather and the calendar (below it)."""
    weekday = f"{forecast['forecast_date']:%A}"
    context = forecast_context(forecast, daily)

    def trips(value: float | None) -> str:
        return "–" if value is None else f"{value:,.0f}"

    delta = ""
    if context["typical"] is not None:
        change = forecast["prediction"] / context["typical"] - 1
        delta = f" · {abs(change):.0%} {'more' if change >= 0 else 'fewer'} than a typical {weekday}"
    days = "\n".join(
        [
            _tile(f"Last {weekday}", trips(context["last_week"]), "counted trips"),
            _tile(f"Typical {weekday}", trips(context["typical"]), "mean of the four weeks before"),
        ]
    )
    weather = "\n".join(_tile(label, value) for label, value in weather_metrics(forecast).items())
    notes = day_notes(forecast["forecast_date"])
    calendar = (
        " · ".join(notes)
        if notes
        else "a regular day: no public holiday, no school holidays, lectures at the university."
    )
    html = f"""<div class="tiles">
{days}
</div>
<h3>Weather forecast</h3>
<div class="tiles">
{weather}
</div>
<p><strong>Calendar:</strong> {escape(calendar)}</p>"""
    return delta, html


def _monitoring_section(warnings: list[str], today: date) -> str:
    if not warnings:
        return f'<p class="status ok"><span aria-hidden="true">✓</span> All checks passed on {today:%d %B %Y}.</p>'
    items = "\n".join(f"<li>{escape(w)}</li>" for w in warnings)
    return f'<div class="status warn"><p><span aria-hidden="true">!</span> Warnings on {today:%d %B %Y}:</p><ul>{items}</ul></div>'


def render_page(
    daily: pd.DataFrame,
    log: pd.DataFrame,
    warnings: list[str],
    today: date,
    weather: pd.DataFrame,
) -> str:
    """The whole page as HTML.

    daily is the output of to_daily (all days since 2019, for the findings),
    log the forecast log, warnings the result of run_checks, today the date of
    the run in Münster, weather the output of load_weather.
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
    delta, context = _context_section(forecast, daily)
    findings_html, specs = _findings_section(daily, weather)
    specs["counts"] = {
        t: themed_spec(counts_chart(data, t, THEMES[t]["surface"]), t) for t in THEMES
    }
    cv_rows = "\n".join(
        f"<tr><td>{name}</td><td>{mae:,}</td><td>{wape:.1%}</td></tr>" for name, mae, wape in CV_RESULTS
    )
    return PAGE.substitute(
        forecast_day=f"{forecast['forecast_date']:%A, %d %B %Y}",
        prediction=f"{forecast['prediction']:,}",
        delta=escape(delta),
        context=context,
        made_on=f"{forecast['made_on']:%d %B %Y}",
        stale=stale,
        weeks=CHART_WEEKS,
        specs=_json_for_script(specs),
        findings=findings_html,
        table_rows=_table_rows(data),
        accuracy=_accuracy_section(log, daily),
        cv_rows=cv_rows,
        how_steps="\n".join(
            f"  <li><strong>{step}:</strong> {escape(text)}</li>" for step, text in HOW_IT_WORKS
        ),
        monitoring=_monitoring_section(warnings, today),
        scripts="\n".join(f'<script src="{url}"></script>' for url in VEGA_SCRIPTS),
        repo=REPO_URL,
        sites_data=SITES_DATA.as_posix(),
        streamlit=STREAMLIT_URL,
        **{f"{t}_{k}": v for t, colors in THEMES.items() for k, v in colors.items()},
    )


def write_data(table: pd.DataFrame, path: Path, site_dir: Path = SITE_DIR) -> Path:
    """Save table as Parquet at path inside site_dir, next to the page.

    GitHub Pages publishes it with the page, without a commit, so the
    repository history does not grow every day. The Streamlit app reads it
    from there (read_published) instead of downloading hundreds of monthly
    files from the city.
    """
    target = site_dir / path
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(target)
    return target


def read_published(path: Path, site_dir: Path = SITE_DIR) -> pd.DataFrame:
    """A data file published by write_data: from GitHub Pages or, if that
    cannot be reached, from site_dir, where the last local run wrote it."""
    try:
        response = requests.get(PAGES_URL + path.as_posix(), timeout=30)
        response.raise_for_status()
        return pd.read_parquet(BytesIO(response.content))
    except requests.RequestException:
        return pd.read_parquet(site_dir / path)


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
.sub { margin: 0 0 12px; }
.intro { margin: 0 0 12px; }
nav { margin: 0 0 32px; color: var(--muted); font-size: 0.95rem; line-height: 1.9; }
nav a { white-space: nowrap; }
.note { font-size: 0.9rem; }
.hero .label { font-size: 0.95rem; }
.hero .value { font-size: 3rem; font-weight: 650; line-height: 1.1; font-variant-numeric: tabular-nums; }
.hero > .unit { margin-bottom: 16px; }
.app-link { margin: -16px 0 32px; color: var(--muted); }
.app-link a {
  display: inline-block; margin-right: 4px; padding: 6px 14px; border: 1px solid var(--text);
  border-radius: 8px; color: var(--text); font-weight: 600; text-decoration: none;
}
.app-link a:hover { background: var(--grid); }
.chart { width: 100%; min-height: 360px; }
.chart.small { min-height: 270px; }
/* A few bars look lost across the full width (Finding.width); on a phone
   every chart keeps the full width */
@media (min-width: 600px) { .chart.small { max-width: var(--chart-width, 100%); } }
h3 { font-size: 1.05rem; margin: 28px 0 6px; }
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
<p class="intro">A daily forecast of bicycle traffic in Münster, built end to end from the city's open counting
data: data cleaning, analysis, a machine learning model and an automated pipeline that updates this page
every evening.</p>
<nav aria-label="Sections">
  <a href="#forecast">Forecast</a> · <a href="#findings">Insights</a> ·
  <a href="#accuracy">Accuracy</a> · <a href="#monitoring">Monitoring</a> · <a href="#how">How it works</a>
</nav>
<p class="app-link"><a href="$streamlit">Open the interactive app</a> to explore and compare all 23
counting sites of the city (it may take a moment to wake up).</p>

$stale
<section class="hero" id="forecast">
  <div class="label">Forecast for $forecast_day</div>
  <div class="value">$prediction</div>
  <div class="unit">bicycle trips$delta</div>
$context
  <p class="note">Made on $made_on, updated every evening.</p>
</section>

<h2>The last $weeks weeks</h2>
<div id="chart-counts" class="chart" role="img" aria-label="Line chart of the counted trips per day and dots for the forecasts; the same values are in the table below."></div>
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

<h2 id="findings">Insights</h2>
<p class="note">Key findings of the exploratory analysis, recomputed every evening from all complete days since 2019.</p>
$findings

<h2 id="accuracy">How good are the forecasts?</h2>
$accuracy
<table>
  <thead><tr><th>Model (cross-validation, measured weather)</th><th>MAE</th><th>WAPE</th></tr></thead>
  <tbody>
$cv_rows
  </tbody>
</table>
<p class="note">Each quarter from 2023 on was forecast by a model trained only on the days before it
(1,232 test days). MAE: mean error in trips per day. WAPE: sum of errors divided by the sum of counted trips.</p>

<h2 id="monitoring">Monitoring</h2>
$monitoring
<p class="note">Every evening the data is checked for outages, for jumps in the share of trips into the city
(a sign that the counting site changed) and for forecasts that are off in the same direction for weeks.</p>

<h2 id="how">How it works</h2>
<ol>
$how_steps
</ol>
<p>Daily counts of all 23 counting sites of the city, with flags for suspicious days (outages, missing and
incomplete days), are published every evening as <a href="$sites_data">$sites_data</a> (Parquet).</p>
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
  for (const [name, spec] of Object.entries(specs)) {
    vegaEmbed("#chart-" + name, dark.matches ? spec.dark : spec.light, { actions: false });
  }
}
dark.addEventListener("change", draw);
draw();
</script>
</body>
</html>
""")
