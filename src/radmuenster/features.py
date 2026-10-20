"""Features for the daily forecast.

Every function takes the dates to describe and returns one row per date, so the
same code builds the training table and the row for tomorrow's forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.easter import easter
from holidays import Germany

from radmuenster.calendars import (
    LECTURE_PERIODS,
    LECTURES_KNOWN_UNTIL,
    SCHOOL_HOLIDAYS_KNOWN_UNTIL,
    SCHOOL_HOLIDAYS_NRW,
)

# North Rhine-Westphalia
SUBDIVISION = "NW"

# The forecast for tomorrow is made while today is still running, so the most
# recent complete day is yesterday: two days before the forecast date.
# Lag 1 would be leakage.
MIN_LAG = 2

# 7 and 14: same weekday one and two weeks earlier; 364: same weekday a year
# earlier (52 weeks); 2: most recent complete day
LAGS = (2, 7, 14, 364)

# Calendar columns of special days off, which are no reference for regular
# days in the following weeks (see lag_features)
SPECIAL_DAYS = ("is_holiday", "is_bridge_day", "is_christmas", "is_rosenmontag")

# First day of each measurement period. The counting site changed twice right
# after a disturbance (see notebook 01, inbound share):
# - 12 June 2024, after the partial inbound outage of 3-11 June: inbound
#   counts about 14 % lower than before
# - 28 October 2025, after the odd day of 27 October: outbound counts about
#   60 % higher than before
# Counts from different periods are not fully comparable.
# These are the periods of the Warendorfer Straße only; other sites pass
# their own period starts (or none) to build_features.
PERIOD_STARTS = (pd.Timestamp("2024-06-12"), pd.Timestamp("2025-10-28"))


def _public_holidays(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """All public holidays in NRW in the years around the given dates."""
    # One year of margin on both sides, because bridge days look at the
    # neighbouring days (e.g. 2 January needs 1 January of the same year,
    # 31 December needs 1 January of the next year)
    years = range(dates.min().year - 1, dates.max().year + 2)
    return pd.DatetimeIndex(sorted(Germany(subdiv=SUBDIVISION, years=years)))


def _is_day_off(dates: pd.DatetimeIndex, holidays: pd.DatetimeIndex) -> np.ndarray:
    """True for Saturdays, Sundays and public holidays."""
    return (dates.dayofweek >= 5) | dates.isin(holidays)


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features for each date.

    Columns:
      weekday         0 = Monday ... 6 = Sunday
      month           1 ... 12
      season_sin,     position in the year on a circle, so that 31 December
      season_cos      and 1 January are close to each other
      is_holiday      public holiday in NRW
      is_bridge_day   workday between two days off (e.g. Friday after
                      Ascension Day), many people take it off
      is_christmas    24 to 31 December: holidays, many people off work
      is_rosenmontag  carnival Monday, 48 days before Easter Sunday
    """
    dates = pd.DatetimeIndex(dates).normalize()
    holidays = _public_holidays(dates)
    one_day = pd.Timedelta(days=1)

    # Day of year as an angle: 1 January = 0, end of the year = almost 2 pi
    angle = 2 * np.pi * (dates.dayofyear - 1) / 365.25
    rosenmontag = pd.DatetimeIndex(
        [pd.Timestamp(easter(year)) - pd.Timedelta(days=48) for year in dates.year.unique()]
    )

    return pd.DataFrame(
        {
            "weekday": dates.dayofweek,
            "month": dates.month,
            "season_sin": np.sin(angle),
            "season_cos": np.cos(angle),
            "is_holiday": dates.isin(holidays),
            "is_bridge_day": ~_is_day_off(dates, holidays)
            & _is_day_off(dates - one_day, holidays)
            & _is_day_off(dates + one_day, holidays),
            "is_christmas": (dates.month == 12) & (dates.day >= 24),
            "is_rosenmontag": dates.isin(rosenmontag),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def lag_features(daily: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Past counts for each date, using only days at least MIN_LAG days earlier.

    daily is the output of to_daily. Days that are not complete count as
    missing, so a lag never carries a value from an outage. Special days off
    (public holidays, bridge days, Christmas, Rosenmontag) count as missing
    too, except in lag_364: Easter Monday is no reference for the Monday one
    week later, but last Christmas is a good reference for this Christmas.
    dates may lie beyond the end of daily (e.g. tomorrow); lags that reach
    into the unknown future are NaN.

    Columns:
      lag_<k>          count k days before the date, for k in LAGS
      mean_7d          mean of the 7 days ending MIN_LAG days before the date
      same_weekday_4w  mean of the same weekday 1 to 4 weeks earlier
    """
    dates = pd.DatetimeIndex(dates).normalize()
    # One row per calendar day, so that "k days earlier" is a date shift and
    # not a row shift (the data has gaps)
    counts = daily["count"].where(daily["complete"])
    counts = counts.reindex(pd.date_range(counts.index.min(), counts.index.max(), freq="D"))
    special = calendar_features(counts.index)[list(SPECIAL_DAYS)].any(axis=1)
    regular = counts.where(~special)

    def days_before(series: pd.Series, k: int) -> np.ndarray:
        # Look up the value k days before each date; dates outside the
        # data range give NaN
        return series.reindex(dates - pd.Timedelta(days=k)).to_numpy()

    columns = {f"lag_{k}": days_before(regular if k < 364 else counts, k) for k in LAGS}
    # rolling(7) at day t covers t-6 ... t; looked up MIN_LAG days before the
    # date it covers date-8 ... date-2
    columns["mean_7d"] = days_before(regular.rolling(7, min_periods=4).mean(), MIN_LAG)
    same_weekday = np.column_stack([days_before(regular, 7 * w) for w in range(1, 5)])
    # Mean over the weeks that have a value; all four missing gives NaN
    columns["same_weekday_4w"] = pd.DataFrame(same_weekday).mean(axis=1).to_numpy()

    return pd.DataFrame(columns, index=pd.DatetimeIndex(dates, name="date"))


def measurement_period(
    dates: pd.DatetimeIndex, period_starts: tuple[pd.Timestamp, ...] = PERIOD_STARTS
) -> pd.Series:
    """Measurement period of each date: 0 before the first entry of
    period_starts, 1 from the first, 2 from the second, and so on.
    Without period_starts every date is in period 0."""
    dates = pd.DatetimeIndex(dates).normalize()
    # searchsorted counts how many period starts lie on or before each date
    period = pd.DatetimeIndex(list(period_starts)).searchsorted(dates, side="right")
    return pd.Series(period, index=pd.DatetimeIndex(dates, name="date"), name="period")


# Snow stays on the roads for days after it fell (February 2021: 20 cm on
# the 7th, almost no snowfall afterwards, but very few cyclists until the 15th)
SNOW_MEMORY_DAYS = 5


def weather_features(weather: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Weather for each date, plus the snowfall of the last SNOW_MEMORY_DAYS days.

    weather is the output of load_weather. Using the weather of the forecast
    day itself is fine: for tomorrow a weather forecast is available. Note that
    the model is trained on measured weather but will get forecasts, which are
    less accurate.
    """
    dates = pd.DatetimeIndex(dates).normalize()
    full = weather.reindex(pd.date_range(weather.index.min(), weather.index.max(), freq="D"))
    full["snow_recent"] = full["snowfall"].rolling(SNOW_MEMORY_DAYS, min_periods=1).sum()
    return full.reindex(dates).rename_axis("date")


def _in_ranges(dates: pd.DatetimeIndex, ranges: list[tuple[str, str]]) -> np.ndarray:
    """True for dates that fall into one of the (first day, last day) ranges."""
    inside = np.zeros(len(dates), dtype=bool)
    for first, last in ranges:
        inside |= (dates >= pd.Timestamp(first)) & (dates <= pd.Timestamp(last))
    return inside


def academic_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """University and school calendar for each date.

    Columns:
      is_lecture_period  lectures at the University of Münster (without the
                         Christmas and Pentecost breaks)
      is_school_holiday  school holidays in NRW

    Raises ValueError for dates beyond the known calendar: there the features
    would silently be False. Extend the lists in calendars.py in that case.
    """
    dates = pd.DatetimeIndex(dates).normalize()
    known_until = min(pd.Timestamp(LECTURES_KNOWN_UNTIL), pd.Timestamp(SCHOOL_HOLIDAYS_KNOWN_UNTIL))
    if dates.max() > known_until:
        raise ValueError(
            f"Calendar only known until {known_until:%Y-%m-%d}, got {dates.max():%Y-%m-%d}: "
            "extend the lists in radmuenster/calendars.py"
        )
    lectures = [(start, end) for start, end, _, _ in LECTURE_PERIODS]
    breaks = [(first, last) for _, _, first, last in LECTURE_PERIODS]
    return pd.DataFrame(
        {
            "is_lecture_period": _in_ranges(dates, lectures) & ~_in_ranges(dates, breaks),
            "is_school_holiday": _in_ranges(dates, SCHOOL_HOLIDAYS_NRW),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def build_features(
    daily: pd.DataFrame,
    weather: pd.DataFrame,
    dates: pd.DatetimeIndex,
    period_starts: tuple[pd.Timestamp, ...] = PERIOD_STARTS,
) -> pd.DataFrame:
    """All features for the given dates, one row per date.

    daily is the output of to_daily (or any table with the columns count and
    complete), weather the output of load_weather. period_starts are the
    measurement periods of the site (see PERIOD_STARTS).
    Works for past dates (training) and for tomorrow (forecast) alike; the
    target count is never part of the result.
    """
    dates = pd.DatetimeIndex(dates).normalize()
    return pd.concat(
        [
            calendar_features(dates),
            academic_features(dates),
            measurement_period(dates, period_starts),
            lag_features(daily, dates),
            weather_features(weather, dates),
        ],
        axis=1,
    )


def training_table(
    daily: pd.DataFrame,
    weather: pd.DataFrame,
    period_starts: tuple[pd.Timestamp, ...] = PERIOD_STARTS,
) -> pd.DataFrame:
    """Features plus the target column count for every complete day."""
    complete = daily.index[daily["complete"]]
    return build_features(daily, weather, complete, period_starts).join(daily["count"])
