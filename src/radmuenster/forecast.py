"""Forecast tomorrow's count and keep a log of all forecasts.

Run every evening: python -m radmuenster.forecast
This also runs the monitoring checks, saves their warnings, builds the
static page for GitHub Pages and the daily counts of all sites next to it.

The model is trained anew on every run, on all complete days known so far.
That takes a few seconds and means there is no stored model that could get
out of date.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from radmuenster.data import DEFAULT_SITE, load_intervals, to_daily
from radmuenster.features import MIN_LAG, PERIOD_STARTS, build_features, training_table
from radmuenster.models import fit_lightgbm, predict_lightgbm
from radmuenster.monitoring import STATUS_FILE, run_checks, write_status
from radmuenster.multi_site import fit_global_lightgbm, site_counts, sites_table
from radmuenster.sites import SITES_BY_ID, load_all_sites
from radmuenster.static_page import (
    DAILY_DATA,
    SITES_DATA,
    WEATHER_DATA,
    render_page,
    write_data,
    write_site,
)
from radmuenster.weather import DAILY_VARIABLES, load_weather, load_weather_forecast

# The counting site's local time decides what "today" and "tomorrow" are. The
# GitHub Actions runner runs on UTC, where the date changes at 1 or 2 am
# German time.
TIMEZONE = ZoneInfo("Europe/Berlin")

# Unlike data/, this file is part of the repository: it is the forecast
# history the monitoring and the web page read.
FORECAST_FILE = Path("forecasts/forecasts.csv")

MODEL_NAME = "lightgbm"

# Forecasts for all counting sites, by one global model (notebook 03). The
# Warendorfer Straße keeps its own model in FORECAST_FILE; its global
# forecast is logged here too, to compare the two on real forecasts.
SITES_FORECAST_FILE = Path("forecasts/sites_forecasts.csv")
GLOBAL_MODEL_NAME = "global_lightgbm"


def today_in_muenster(now: datetime | None = None) -> date:
    """Today's date in Münster. now must carry a time zone; default: now."""
    now = now or datetime.now(TIMEZONE)
    return now.astimezone(TIMEZONE).date()


def combine_weather(archive: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Archive weather, extended by forecast days the archive does not have yet.

    Where both have a day, the archive wins: it is closer to the weather the
    model was trained on.
    """
    # combine_first keeps all values of archive and fills only its gaps (and
    # dates it lacks) from forecast
    return archive.combine_first(forecast)


def forecast_day(
    daily: pd.DataFrame, weather: pd.DataFrame, target: date, made_on: date
) -> pd.DataFrame:
    """Forecast the count on target, as one row for the forecast log.

    daily is the output of to_daily, weather the output of combine_weather.
    Only days at least MIN_LAG days before target are used, for training and
    for the lags, as in the cross-validation. In the evening run this changes
    nothing; it matters when forecasting a past date, which must not see the
    days the real forecast could not have seen.

    Raises ValueError if weather has no complete row for target: without
    weather, the model would silently predict from the lags alone.
    """
    target = pd.Timestamp(target)
    known = daily.loc[: target - pd.Timedelta(days=MIN_LAG)]

    target_weather = weather.reindex([target])[list(DAILY_VARIABLES.values())]
    if target_weather.isna().any(axis=None):
        raise ValueError(f"Weather for {target:%Y-%m-%d} is missing or incomplete")

    model, factors = fit_lightgbm(training_table(known, weather))
    features = build_features(known, weather, pd.DatetimeIndex([target]))
    prediction = predict_lightgbm(model, factors, features)[0]

    record = pd.DataFrame(
        {
            "forecast_date": [target.date()],
            "made_on": [made_on],
            "model": [MODEL_NAME],
            "prediction": [round(prediction)],
            # Lets the monitoring spot when the city's data stopped coming
            "latest_complete_day": [known.index[known["complete"]].max().date()],
            "lag_2": features["lag_2"].to_numpy(),
            "period": features["period"].to_numpy(),
        }
    )
    # The weather the forecast was based on. Measured weather replaces it
    # later in the archive, so this is the only record of the forecast
    # weather, and the basis for an honest error of the daily forecast.
    for column in DAILY_VARIABLES.values():
        record[column] = target_weather[column].to_numpy()
    return record


def forecast_sites(
    counts: dict[str, pd.DataFrame], weather: pd.DataFrame, target: date, made_on: date
) -> pd.DataFrame:
    """Forecast the count on target at every site with one global model, as
    rows for SITES_FORECAST_FILE.

    counts maps a site id to its daily counts (to_daily or site_counts). As in
    forecast_day, only days at least MIN_LAG days before target are used, and
    weather must have target. A site the model cannot forecast (no usable day
    in the training data) gets no row.
    """
    target = pd.Timestamp(target)
    known = {site: daily.loc[: target - pd.Timedelta(days=MIN_LAG)] for site, daily in counts.items()}
    # Without a single usable day there is nothing to train on or to scale by
    known = {site: daily for site, daily in known.items() if daily["complete"].any()}
    periods = {DEFAULT_SITE: PERIOD_STARTS}

    target_weather = weather.reindex([target])[list(DAILY_VARIABLES.values())]
    if target_weather.isna().any(axis=None):
        raise ValueError(f"Weather for {target:%Y-%m-%d} is missing or incomplete")

    model = fit_global_lightgbm(sites_table(known, weather, periods))
    features = pd.concat(
        [
            build_features(daily, weather, pd.DatetimeIndex([target]), periods.get(site, ())).assign(site=site)
            for site, daily in known.items()
        ]
    )
    record = pd.DataFrame(
        {
            "forecast_date": target.date(),
            "made_on": made_on,
            "site_id": features["site"].to_numpy(),
            "site_name": [SITES_BY_ID[site].name for site in features["site"]],
            "model": GLOBAL_MODEL_NAME,
            "prediction": model.predict(features),
            "latest_complete_day": [
                known[site].index[known[site]["complete"]].max().date() for site in features["site"]
            ],
            "lag_2": features["lag_2"].to_numpy(),
        }
    )
    for column in DAILY_VARIABLES.values():
        record[column] = target_weather[column].iloc[0]
    record = record.dropna(subset=["prediction"])
    record["prediction"] = record["prediction"].round().astype(int)
    return record.reset_index(drop=True)


DATE_COLUMNS = ["forecast_date", "made_on", "latest_complete_day"]


def read_forecasts(path: Path = FORECAST_FILE) -> pd.DataFrame:
    """The forecast log, with the date columns as timestamps."""
    return pd.read_csv(path, parse_dates=DATE_COLUMNS)


def save_forecast(
    record: pd.DataFrame, path: Path = FORECAST_FILE, key: tuple[str, ...] = ("forecast_date", "model")
) -> pd.DataFrame:
    """Append record to the forecast log at path and return the whole log.

    A forecast with the same key (date and model; for all sites also the site)
    replaces the earlier one, so that running the job twice on a day keeps one row.
    """
    key = list(key)
    record = record.copy()
    # Timestamps like the log read back from CSV, so that keys compare equal
    record[DATE_COLUMNS] = record[DATE_COLUMNS].apply(pd.to_datetime)
    if path.exists():
        log = read_forecasts(path)
        # Anti-join: keep the rows of log whose key does not occur in record
        same = log.set_index(key).index.isin(record.set_index(key).index)
        log = pd.concat([log[~same], record], ignore_index=True)
    else:
        log = record
    log = log.sort_values(key, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return log


def main() -> None:
    today = today_in_muenster()
    target = today + pd.Timedelta(days=1)
    daily = to_daily(load_intervals(DEFAULT_SITE, end=today))
    weather = combine_weather(load_weather(end=today), load_weather_forecast())
    record = forecast_day(daily, weather, target, made_on=today)
    log = save_forecast(record)
    sites_daily, failed_sites = load_all_sites(end=today)
    warnings = run_checks(daily, log, today, sites_daily) + [
        f"Counting site {name} could not be loaded ({error})." for name, error in failed_sites.items()
    ]
    # The forecast of the Warendorfer Straße above must not depend on this one
    try:
        counts = {
            site_id: daily if site_id == DEFAULT_SITE else site_counts(rows)
            for site_id, rows in sites_daily.groupby("site_id")
        }
        sites_record = forecast_sites(counts, weather, target, made_on=today)
        save_forecast(sites_record, SITES_FORECAST_FILE, key=("forecast_date", "site_id", "model"))
    # Deliberately broad, like a broken site in load_all_sites
    except Exception as error:
        sites_record = pd.DataFrame()
        warnings.append(f"The forecasts for all sites failed ({type(error).__name__}: {error}).")
    write_status(warnings, today)
    page = write_site(render_page(daily, log, warnings, today, weather))
    sites_data = write_data(sites_daily, SITES_DATA)
    write_data(daily, DAILY_DATA)
    write_data(weather, WEATHER_DATA)

    row = record.iloc[0]
    print(f"Forecast for {row['forecast_date']:%a %Y-%m-%d}: {row['prediction']} trips")
    print(f"Latest complete day: {row['latest_complete_day']:%Y-%m-%d}")
    print(f"Saved to {FORECAST_FILE} and {STATUS_FILE}, page built in {page}")
    print(f"Daily counts of {sites_daily['site_id'].nunique()} sites saved to {sites_data}")
    print(f"Forecasts for {len(sites_record)} sites saved to {SITES_FORECAST_FILE}")
    for warning in warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
