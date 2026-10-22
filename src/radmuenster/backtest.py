"""Forecasts made afterwards, as if the daily job had run on each evening.

Run once (not in the daily job): python -m radmuenster.backtest [--days 90]

For every day of the period, both models are trained on the days the daily
job would have known that evening (up to two days before the forecast day)
and forecast the day with the weather forecast of the evening before, from
Open-Meteo's archive of previous runs. That makes the error close to the
error of the real daily forecasts, which only start in October 2026.

Still slightly optimistic: the flags use hindsight. The dead channel rule
looks two weeks ahead, and SITE_FAULTS was written after the outages; on the
evening itself, some broken days still counted as usable. The training days
also have the measured weather, where the daily job, a few days after the
archive ends, has gaps.

The result goes to its own file and never into the forecast logs, which hold
only forecasts that were really made.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from radmuenster.data import DEFAULT_SITE, load_intervals, to_daily
from radmuenster.forecast import BACKTEST_FILE, MODEL_NAME, forecast_day, forecast_sites
from radmuenster.multi_site import site_counts_by_id
from radmuenster.sites import SITES_BY_ID, load_all_sites
from radmuenster.weather import load_weather, load_weather_previous_day

DEFAULT_DAYS = 90


def backtest(
    counts: dict[str, pd.DataFrame],
    daily: pd.DataFrame,
    weather: pd.DataFrame,
    forecast_weather: pd.DataFrame,
    days: list[date],
) -> pd.DataFrame:
    """Forecasts of all sites (global model) and of the Warendorfer Straße
    (its own model) for each of days, in the format of SITES_FORECAST_FILE.

    weather is the measured weather (load_weather), forecast_weather the
    forecast of the evening before (load_weather_previous_day); for each day,
    its forecast replaces the measured weather of that day.
    """
    frames = []
    for target in days:
        target = pd.Timestamp(target)
        made_on = (target - pd.Timedelta(days=1)).date()
        evening_weather = weather.copy()
        evening_weather.loc[target] = forecast_weather.loc[target, weather.columns]
        frames.append(forecast_sites(counts, evening_weather, target.date(), made_on))
        own = forecast_day(daily, evening_weather, target.date(), made_on)
        own.insert(2, "site_id", DEFAULT_SITE)
        own.insert(3, "site_name", SITES_BY_ID[DEFAULT_SITE].name)
        frames.append(own.drop(columns="period"))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="days back from the latest complete day")
    args = parser.parse_args()

    daily = to_daily(load_intervals(DEFAULT_SITE))
    sites_daily, failed = load_all_sites()
    if failed:
        print(f"Not loaded: {failed}")
    counts = site_counts_by_id(sites_daily, daily)
    # Up to the latest day with counts; there is nothing to compare later forecasts with
    last = daily.index[daily["complete"]].max().date()
    first = last - timedelta(days=args.days - 1)
    weather = load_weather()
    forecast_weather = load_weather_previous_day(first, last)

    days = [first + timedelta(days=i) for i in range(args.days)]
    result = backtest(counts, daily, weather, forecast_weather, days)
    BACKTEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(BACKTEST_FILE, index=False)
    models = result["model"].value_counts().to_dict()
    print(f"Backtest {first} to {last}: {models} forecasts saved to {BACKTEST_FILE}")
    print(f"(the Warendorfer Straße's own model is {MODEL_NAME!r})")


if __name__ == "__main__":
    main()
