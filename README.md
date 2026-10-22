# muenster-cycling-data

Exploring Münster's bicycle traffic with open data – analysis, forecasting and an AI assistant.

First project: **daily forecast of bicycle trips at the Warendorfer Straße counting site** (data since January 2019),
now extended to **all 23 counting sites** of the city with one global model.

**Live: [annikahackenberg.github.io/muenster-cycling-data](https://annikahackenberg.github.io/muenster-cycling-data/)**:
tomorrow's forecast, updated every evening, with the last eight weeks of counts, tomorrow at all sites, the
monitoring and how it works.
An interactive version runs on [Streamlit](https://muenster-cycling-data.streamlit.app); the free hosting puts it
to sleep when nobody visits it for a while, then it takes a moment to wake up.

## Data

Source: [od-ms/radverkehr-zaehlstellen](https://github.com/od-ms/radverkehr-zaehlstellen), City of Münster,
license [Datenlizenz Deutschland Namensnennung 2.0](https://www.govdata.de/dl-de/by-2-0).
The city publishes 15-minute counts per counting site as one CSV file per month. This is raw data:
outages and construction sites are not cleaned up.

Weather: [Open-Meteo](https://open-meteo.com) historical weather API (ERA5 reanalysis) and
forecast API, license [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Build daily counts

```bash
python -m radmuenster
```

Downloads all monthly files to `data/raw/` and writes the daily totals to
`data/processed/daily_100034983.parquet`. Files of finished months are cached; the current ones are
downloaded from the latest commit of the city's repository, because a link to `main` can return an old
version of a file from GitHub's cache. The column `complete` marks days that are complete,
have no fault status, where both directions counted plausibly and that are not in the list of known
faults (`KNOWN_FAULTS` in [data.py](src/radmuenster/data.py), faults the automatic checks miss).

## Findings from the exploratory analysis

Details, plots and reasoning: [notebooks/01_exploratory_analysis.ipynb](notebooks/01_exploratory_analysis.ipynb)
(data as of 9 October 2026).

**Data quality**
- The city's status value is not enough to detect outages. The inbound channel was empty twice for about
  80 days (autumn 2021 and autumn 2023), plus several partial outages, all with status "OK".
  They are detected by the share of the two directions (`direction_fault`).
- From April 2021 to May 2025 the last day of every month is missing from the data.
- The inbound share jumps twice and stays on the new level, each time right after a disturbance. This points
  to maintenance or changes at the counting site:
  - June 2024: the inbound channel counts about 14 % less than before, so part of the lower level since then
    is probably a measurement artefact.
  - End of October 2025: the outbound channel counts about 60 % more than before, so counts since then are
    probably too high.

  Counts from different periods are not fully comparable.

**Patterns**
- **Weekday** is the strongest effect: on Sundays about 40 % as many people cycle as on a workday;
  Wednesdays and Saturdays (market days) are the busiest.
- **Season** is weaker (January and December about 13 % below, May to November about 3–8 % above the
  annual mean), with a dip in April (Easter, lecture-free period).
- **Outliers** are almost all explained: public holidays, Christmas, Rosenmontag (carnival), snow in February 2021.
  The rest are mostly Sundays with particularly good or bad weather.
- **Autocorrelation** is highest at 7 days (0.70) and high at 364 days (0.58). Lag 1 is not available for a
  forecast for tomorrow (leakage); usable lags start at 2 days.

**All counting sites** (the same methods applied to every site,
[findings.py](src/radmuenster/findings.py), recomputed every evening; numbers as of October 2026)
- **Commuter and leisure routes:** at 20 of the 23 sites, Sundays have 37–56 % of a workday's trips. The
  three sections of the Kanalpromenade along the canal reach 88–178 %: leisure routes. A site counts as a
  leisure route when its Sundays reach at least 80 % of a workday, so new sites are sorted by the data.
- **Rain** hits the leisure routes hardest: on weekends with 8+ hours of rain they keep 57–66 % of a
  typical day, the commuter routes 78–92 %.
- **Season:** most sites count 1.2–1.9 times as many trips in their busiest month as in their quietest,
  the leisure routes 2.8–5.5 times. Most peak in May or June, the Neutor (next to the university) in October.

## Features

`radmuenster.features.build_features(daily, weather, dates)` builds one row per date, for past dates
(training) and for tomorrow (forecast) alike; `training_table(daily, weather)` adds the target `count`
for all complete days.

| Group | Columns | Source |
|---|---|---|
| Calendar | weekday, month, season (sin/cos), public holiday, bridge day, Christmas, Rosenmontag | `holidays` |
| Academic | lecture period of the University of Münster, school holidays NRW | `radmuenster/calendars.py` |
| Measurement period | 0 / 1 from 12 June 2024 / 2 from 28 October 2025 | exploratory analysis |
| Lags | counts 2, 7, 14 and 364 days earlier, 7-day mean, same weekday over 4 weeks | daily counts |
| Weather | temperature, precipitation, sunshine, wind, snowfall, snow of the last 5 days | Open-Meteo |

Lags start at 2 days: a forecast for tomorrow is made while today is still running (no leakage).
The lecture and school holiday dates are maintained by hand in `calendars.py`; the features refuse
dates beyond the known calendar instead of silently treating them as normal days.

## Models

Time-based cross-validation: each quarter from 2023 on is a test period, the model is trained on all days
before it (`radmuenster.evaluation`). Results over 1,232 test days
(details: [notebooks/02_models.ipynb](notebooks/02_models.ipynb)):

| Model | MAE (trips) | WAPE | MAPE | NRMSE |
|---|---|---|---|---|
| Naive baseline (same weekday last week) | 960 | 16.1 % | 24.0 % | 23.3 % |
| Holiday baseline (naive × factor per kind of special day) | 848 | 14.3 % | 17.7 % | 18.9 % |
| Linear regression on log(count) | 646 | 10.9 % | 12.1 % | 14.4 % |
| **LightGBM on log(count)** | **529** | **8.9 %** | **10.2 %** | **12.0 %** |

WAPE is the sum of errors divided by the sum of actual counts; MAPE is dominated by quiet days (snow),
NRMSE by a few large errors. The cross-validation uses the measured weather of the forecast day, so the
error of the app, which only has a weather forecast, will be higher.

### More counting sites

Does the model work for the other counting sites, and is one **global model** for all sites better
than one model per site? The global model divides the count and the lags by the median count of
each site (from the training data only) and gets the site as a categorical feature, so that it can
learn weekly patterns, weather and holidays from all sites together (`radmuenster.multi_site`,
details: [notebooks/03_multi_site_models.ipynb](notebooks/03_multi_site_models.ipynb)).

| Sites, test period | Naive baseline | LightGBM per site | Global LightGBM |
|---|---|---|---|
| 9 sites with data since 2019/2020, quarters from 2023 | 17.9 % | 9.0 % | 9.1 % |
| 14 sites with data since 2023/2024, quarters from 2025 | 20.7 % | 11.5 % | **9.9 %** |

WAPE over all test days. With a long history both approaches are equally good; for the newer sites
the global model is clearly better, so the daily forecasts of all sites use one global model.

A **backtest** ([backtest.py](src/radmuenster/backtest.py), `python -m radmuenster.backtest --days 90`) forecasts
every day of the last 90 days as if the daily job had run on the evening before: trained on the days known
then, with the weather forecast of that evening from Open-Meteo's archive of previous runs. Its result is in
[forecasts/sites_backtest.csv](forecasts/sites_backtest.csv), never in the logs of real forecasts:

| 24 July to 21 October 2026, weather forecast of the evening before | WAPE |
|---|---|
| All 23 sites, global model | 9.2 % |
| Single sites | 7.0 % (Promenade) to 18.4 % (Kanalpromenade, Abschnitt 6) |
| Warendorfer Straße, own model / global model (same days) | 7.3 % / 7.9 % |

It is still slightly optimistic: the flags of outages use hindsight, and summer and autumn are easier to
forecast than winter and the holidays in the cross-validation.

## Daily forecast

```bash
python -m radmuenster.forecast
```

Trains LightGBM on all complete days, forecasts tomorrow with the Open-Meteo weather forecast and appends
the result to [forecasts/forecasts.csv](forecasts/forecasts.csv), together with the forecast weather it used.
That weather is replaced by measured weather in the archive later, so the log is the basis for the honest
error of the daily forecast. "Tomorrow" is determined in Münster's time zone.

The same run trains the global model on all sites and logs tomorrow's forecast of every site in
[forecasts/sites_forecasts.csv](forecasts/sites_forecasts.csv). The Warendorfer Straße keeps its own model;
its global forecast is logged too, to compare both on real forecasts. If the forecasts of all sites fail,
the forecast of the Warendorfer Straße and the page are still made, and the monitoring reports the error.

The same run checks whether the forecast can still be trusted and writes the warnings to
[forecasts/status.json](forecasts/status.json):

| Check | Warns when |
|---|---|
| Freshness | yesterday has no counts at all (the city's daily update failed) |
| Incomplete days | 3 or more of the last 7 days are incomplete or faulty (dead channel) |
| Inbound share | the median share of the last 14 days differs by more than 5 points from the 90 days before (new measurement jump) |
| Bias | the forecasts of the last 28 days are on average more than 8 % too high or too low |
| Bias of all sites | the forecasts of a site over 28 days are more than 15 % too high or too low (sum of errors over sum of counts) |
| Low counts | a site is far below its usual count compared with the other sites for three days in a row (see below) |

Run against the history, the inbound share check finds both known jumps within two weeks
(26 June 2024, 5 November 2025) and one more episode: from mid-May to early June 2026 the share
dropped from about 0.50 to 0.37–0.43 for four weeks and then recovered. The inbound channel counted
30–50 % less than a year earlier while the outbound channel was normal, a slowly failing sensor rather
than a change in traffic, so these days are now marked as a known fault.

A GitHub Actions workflow ([.github/workflows/daily-forecast.yml](.github/workflows/daily-forecast.yml))
runs every evening at 18:00 UTC, commits the forecast to `main` and opens an issue labelled `monitoring`
when there are warnings.

## Web page

```bash
pip install -e ".[app]"
streamlit run streamlit_app.py
```

The Streamlit app has four pages and can be understood on its own. [Forecast](app_pages/forecast.py) starts
with tomorrow at all sites: the forecast, the change against a typical day and the typical error of each
site (from the backtest until a site has a week of real forecasts). Below it, for a selected site, it shows
the forecast next to the same weekday last week and a typical one, the weather forecast it is based on, what sets the day apart in the calendar
(public holiday, school holidays, no lectures), the counts and forecasts of the last eight weeks, the error of the logged forecasts and the monitoring
status. It reads the forecast log from `main` on GitHub, so it shows a new forecast without redeploying,
and loads the recent counts directly from the city. If the daily job is switched off, the page says so
and keeps showing the latest forecast. [Insights](app_pages/findings.py) shows the key findings
of the exploratory analysis, the same as on the static page, and the findings across all sites (the static
page shows only the first of them). [Explore all sites](app_pages/explore.py) shows the daily counts of
all 23 counting sites of the city as published: compare sites over any period by day, week or month, see
which periods are flagged and why, and which sites are busiest. [Leeze AI](app_pages/assistant.py) answers
questions about the data in a chat (see below). On Streamlit Community Cloud the app is
installed from [requirements.txt](requirements.txt).

### AI assistant

*Leeze* is Münster dialect for bicycle. The chat answers questions about the forecasts at all sites and their accuracy, the
counts at all sites, the weather and problems in the data. A language model (Claude Haiku) only chooses
tools and words the answer; every number comes from one of ten tools
([assistant_tools.py](src/radmuenster/assistant_tools.py)), plain functions on the published data that can
be tested without a model. Below each answer, the page shows which tools were called with which arguments.
The tool-use loop ([assistant.py](src/radmuenster/assistant.py)) is written by hand on the Messages API: it
runs the tools the model asks for, sends wrong calls back as errors the model can correct, and makes the
model answer after at most six tool rounds. The client is passed in, so the tests play the model.

The API key is read from Streamlit secrets (locally `.streamlit/secrets.toml`, which is not committed):

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

Without a key, the page says that the assistant is not set up. Against cost and misuse, a question has at
most 500 characters, a conversation at most 10 questions (the whole conversation is sent with every
question) and the app at most 100 questions per day across all users. The hard limit is the spending
limit of the API account.

### All counting sites

The city's 23 counting sites label their channels differently (inbound/outbound, IN/OUT, towards a place),
some rebuilt their channels over time, and one swapped its labels. So only each site's total is used,
which is always the sum of its channels ([sites.py](src/radmuenster/sites.py)). The counts are not changed;
suspicious days are flagged instead: missing, incomplete, flagged by the city, zero, and dead channels. A
channel counts as dead when it carried at least 10% of the trips, counts nothing for whole days and the
site's total drops by at least 25% at the same time; when channels are rebuilt, the total stays the same.
This finds outages the city did not flag, for example at the Wolbecker Straße, where one direction has
counted almost nothing since August 2025. Changes in how a site counts (like the jump at the Warendorfer
Straße in October 2025) are not detected.

Partial outages, where a site counts about half of its usual trips or less, are harder: a rule cannot
tell them from building works or a leisure route in bad weather. So the daily monitoring only warns when
a site is far below its usual count compared with the other sites for three days in a row (the usual
count is the median of the same weekday in the last weeks; dividing by the median of the other sites takes
out rain, holidays and the season). Whether it is an outage is decided by hand; real outages go into
`SITE_FAULTS` in [sites.py](src/radmuenster/sites.py) and are flagged as partial outages. Only a day with
less than a tenth of the usual count is flagged automatically ("counted almost nothing"): no weather does
that to one road but not the others.

The daily job also builds a static version of the page ([static_page.py](src/radmuenster/static_page.py)) and
publishes it on GitHub Pages, without a commit. It is a single HTML file with the same chart, embedded as a
Vega-Lite spec, so it loads instantly and never sleeps. It shows the forecast in the same context as the
app, tomorrow at all sites, and the key findings of the exploratory analysis (weekly pattern, season, rain on workdays and
weekends, data quality), recomputed every evening by [findings.py](src/radmuenster/findings.py), and links
to the app. Both pages take their texts and numbers from the same functions. Next to the page, the job
publishes the daily counts of all sites, of the Warendorfer Straße and the weather as Parquet files, which
the app reads instead of downloading them from the city.

## Roadmap

1. [x] Load data, aggregate to daily counts and explore (weekly pattern, season, autocorrelation, outliers, data quality)
2. [x] Features: weather (Open-Meteo), weekday, public holidays NRW, semester periods, school holidays, lags, measurement period
3. [x] Models: naive and holiday baseline, linear regression, LightGBM with time-based cross-validation
4. [x] Daily forecast: a scheduled GitHub Actions job computes tomorrow's forecast and the monitoring
   (warn when the direction share or the forecast error drifts, a sign of a new measurement jump);
   a [Streamlit page](https://muenster-cycling-data.streamlit.app) and a
   [static page](https://annikahackenberg.github.io/muenster-cycling-data/) show the forecast and the recent weeks
5. [x] Key findings on the static page: weekly pattern, season, rain, data quality, recomputed every evening
6. [x] All 23 counting sites: load them, flag suspicious days (outages, dead channels) and explore and
   compare them in Streamlit
7. [x] AI assistant: ask questions about the data in natural language (an LLM with tools that query the
   counts, weather and forecasts)
8. [x] Forecasts for all counting sites
   - [x] Compare one model per site with one global model for all sites
   - [x] Flag partial outages that the current flags miss (a monitoring warning, decided by hand)
   - [x] Daily forecasts for all sites with the global model, a backtest, on both pages and in the assistant
9. [x] Insights for all sites: commuter and leisure routes, rain and season at each site, in the app, on the
   static page (the first finding) and in the assistant
