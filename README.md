# muenster-cycling-data

Exploring Münster's bicycle traffic with open data – analysis, forecasting and an AI assistant.

First project: **daily forecast of bicycle trips at the Warendorfer Straße counting site** (data since January 2019).

## Data

Source: [od-ms/radverkehr-zaehlstellen](https://github.com/od-ms/radverkehr-zaehlstellen), City of Münster,
license [Datenlizenz Deutschland Namensnennung 2.0](https://www.govdata.de/dl-de/by-2-0).
The city publishes 15-minute counts per counting site as one CSV file per month. This is raw data:
outages and construction sites are not cleaned up.

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
- The inbound share jumps twice and stays on the new level, each time right after a disturbance (June 2024,
  October 2025). This points to maintenance or changes at the counting site. Counts before and after a jump
  are not fully comparable.

**Patterns**
- **Weekday** is the strongest effect: on Sundays about 40 % as many people cycle as on a workday;
  Wednesdays and Saturdays (market days) are the busiest.
- **Season** is weaker (January and December about 13 % below, May to November about 3–8 % above the
  annual mean), with a dip in April (Easter, lecture-free period).
- **Outliers** are almost all explained: public holidays, Christmas, Rosenmontag (carnival), snow in February 2021.
  The rest are mostly Sundays with particularly good or bad weather.
- **Autocorrelation** is highest at 7 days (0.70) and high at 364 days (0.58). Lag 1 is not available for a
  forecast for tomorrow (leakage); usable lags start at 2 days.

## Roadmap

1. [x] Load data, aggregate to daily counts and explore (weekly pattern, season, autocorrelation, outliers, data quality)
2. [ ] Features: weather (Open-Meteo), weekday, public holidays NRW, semester periods, lags
3. [ ] Models: naive baseline, linear regression, LightGBM with time-based cross-validation
4. [ ] Streamlit app with the forecast for tomorrow
