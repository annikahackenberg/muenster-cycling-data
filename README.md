# muenster-cycling-data

Exploring Münster's bicycle traffic with open data – analysis, forecasting and an AI assistant.

First project: **daily forecast of bicycle trips at the Warendorfer Straße counting site** (data since January 2019).

## Data

Source: [od-ms/radverkehr-zaehlstellen](https://github.com/od-ms/radverkehr-zaehlstellen), City of Münster,
license [Datenlizenz Deutschland Namensnennung 2.0](https://www.govdata.de/dl-de/by-2-0).
The city publishes 15-minute counts per counting site as one CSV file per month. This is raw data:
outages and construction sites are not cleaned up.

Weather: [Open-Meteo](https://open-meteo.com) historical weather API (ERA5 reanalysis),
license [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

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

Downloads all monthly files to `data/raw/` (cached) and writes the daily totals to
`data/processed/daily_100034983.parquet`. The column `complete` marks days that are complete,
have no fault status and where both directions counted plausibly.

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

## Roadmap

1. [x] Load data, aggregate to daily counts and explore (weekly pattern, season, autocorrelation, outliers, data quality)
2. [x] Features: weather (Open-Meteo), weekday, public holidays NRW, semester periods, school holidays, lags, measurement period
3. [x] Models: naive and holiday baseline, linear regression, LightGBM with time-based cross-validation
4. [ ] Daily forecast: a scheduled GitHub Actions job computes tomorrow's forecast and the monitoring
   (warn when the direction share or the forecast error drifts, a sign of a new measurement jump);
   a Streamlit page shows the forecast and the recent weeks
5. [ ] AI assistant: ask questions about the data in natural language (an LLM with tools that query the
   counts, weather and forecasts)
