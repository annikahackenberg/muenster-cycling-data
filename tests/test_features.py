import numpy as np
import pandas as pd
import pytest

from radmuenster.calendars import LECTURE_PERIODS, SCHOOL_HOLIDAYS_NRW
from radmuenster.features import (
    academic_features,
    build_features,
    calendar_features,
    lag_features,
    measurement_period,
    training_table,
    weather_features,
)


def features_for(day: str) -> pd.Series:
    return calendar_features(pd.DatetimeIndex([day])).iloc[0]


def test_one_row_per_date_in_input_order():
    dates = pd.DatetimeIndex(["2026-03-02", "2025-01-01", "2026-03-01"])
    features = calendar_features(dates)
    assert list(features.index) == list(dates)


def test_weekday_and_month():
    row = features_for("2026-10-07")  # a Wednesday
    assert row["weekday"] == 2
    assert row["month"] == 10


def test_season_is_continuous_over_new_year():
    features = calendar_features(pd.DatetimeIndex(["2025-12-31", "2026-01-01", "2026-07-02"]))
    end, start, summer = features[["season_sin", "season_cos"]].to_numpy()
    # New Year's Eve and New Year's Day are neighbours, midsummer is far away
    assert np.linalg.norm(end - start) < 0.05
    assert np.linalg.norm(start - summer) > 1.9


@pytest.mark.parametrize(
    "day", ["2025-12-25", "2026-04-06", "2025-11-01", "2025-10-03", "2026-06-04"],
    ids=["christmas", "easter_monday", "all_saints", "unity_day", "corpus_christi"],
)
def test_public_holidays_in_nrw(day):
    assert features_for(day)["is_holiday"]


def test_regular_workday_is_no_holiday():
    row = features_for("2026-10-07")
    assert not row["is_holiday"]
    assert not row["is_bridge_day"]


@pytest.mark.parametrize(
    "day", ["2026-01-02", "2026-05-15", "2025-05-30"],
    ids=["after_new_year", "after_ascension_2026", "after_ascension_2025"],
)
def test_bridge_days(day):
    assert features_for(day)["is_bridge_day"]


def test_weekend_is_no_bridge_day():
    # Saturday 3 January 2026 lies between 2 January (bridge day) and Sunday
    assert not features_for("2026-01-03")["is_bridge_day"]


def test_christmas_period():
    assert features_for("2025-12-24")["is_christmas"]
    assert features_for("2025-12-31")["is_christmas"]
    assert not features_for("2026-01-01")["is_christmas"]
    assert not features_for("2025-12-23")["is_christmas"]


@pytest.mark.parametrize("day", ["2019-03-04", "2020-02-24", "2026-02-16"])
def test_rosenmontag(day):
    assert features_for(day)["is_rosenmontag"]
    assert features_for(day)["weekday"] == 0


def make_daily(start: str, counts: list[float], incomplete: tuple[str, ...] = ()) -> pd.DataFrame:
    """Minimal output of to_daily: count and complete for consecutive days."""
    index = pd.date_range(start, periods=len(counts), freq="D", name="date")
    daily = pd.DataFrame({"count": counts, "complete": True}, index=index)
    daily.loc[pd.DatetimeIndex(incomplete), "complete"] = False
    return daily


def test_lags_look_back_the_right_number_of_days():
    # Count = day number, so a lag of k days must give (day - k). The date
    # (15 February 2025) is chosen so that no lag falls on a public holiday.
    daily = make_daily("2024-02-01", list(range(400)))
    day = pd.Timestamp("2024-02-01") + pd.Timedelta(days=380)
    row = lag_features(daily, pd.DatetimeIndex([day])).iloc[0]
    assert row["lag_2"] == 378
    assert row["lag_7"] == 373
    assert row["lag_14"] == 366
    assert row["lag_364"] == 16
    assert row["mean_7d"] == np.mean(range(372, 379))  # days 372 ... 378
    assert row["same_weekday_4w"] == np.mean([373, 366, 359, 352])


def test_lags_ignore_the_last_day_before_the_forecast():
    # Leakage check: changing the day itself and the day before must not
    # change any feature, because those days are not known at forecast time
    daily = make_daily("2024-01-01", [100.0] * 60)
    day = pd.DatetimeIndex(["2024-02-20"])
    before = lag_features(daily, day)
    changed = daily.copy()
    changed.loc[["2024-02-19", "2024-02-20"], "count"] = 99999
    pd.testing.assert_frame_equal(lag_features(changed, day), before)


def test_lags_skip_incomplete_days():
    daily = make_daily("2024-01-01", [100.0] * 30, incomplete=("2024-01-18",))
    row = lag_features(daily, pd.DatetimeIndex(["2024-01-25"])).iloc[0]
    assert np.isnan(row["lag_7"])
    # The weekday mean uses the remaining three weeks
    assert row["same_weekday_4w"] == 100


def test_lags_for_tomorrow_beyond_the_data():
    daily = make_daily("2024-01-01", [100.0] * 30)  # last day: 30 January
    row = lag_features(daily, pd.DatetimeIndex(["2024-01-31"])).iloc[0]
    assert row["lag_2"] == 100
    assert row["lag_7"] == 100


@pytest.mark.parametrize(
    "day, expected",
    [("2019-01-01", 0), ("2024-06-11", 0), ("2024-06-12", 1), ("2025-10-27", 1),
     ("2025-10-28", 2), ("2026-10-10", 2)],
)
def test_measurement_period_boundaries(day, expected):
    assert measurement_period(pd.DatetimeIndex([day])).iloc[0] == expected


def test_weather_features_remember_recent_snow():
    days = pd.date_range("2021-02-05", periods=10, freq="D")
    weather = pd.DataFrame({"snowfall": 0.0, "temp_mean": -5.0}, index=days)
    weather.loc["2021-02-07", "snowfall"] = 20.0
    features = weather_features(weather, pd.DatetimeIndex(["2021-02-11", "2021-02-12"]))
    # 11 February: 7 February is within the last 5 days (7th ... 11th)
    assert features.loc["2021-02-11", "snow_recent"] == 20.0
    # 12 February: the window covers 8th ... 12th, the snow day has dropped out
    assert features.loc["2021-02-12", "snow_recent"] == 0.0
    assert features.loc["2021-02-11", "temp_mean"] == -5.0


def test_weather_features_missing_weather_is_nan():
    days = pd.date_range("2024-01-01", periods=3, freq="D")
    weather = pd.DataFrame({"snowfall": 0.0, "temp_mean": 3.0}, index=days)
    features = weather_features(weather, pd.DatetimeIndex(["2024-01-10"]))
    assert np.isnan(features.loc["2024-01-10", "temp_mean"])


@pytest.mark.parametrize(
    "day, expected",
    [
        ("2025-10-12", False),  # Sunday before the lectures of WiSe 2025/26 start
        ("2025-10-13", True),   # first lecture day
        ("2025-12-23", False),  # Christmas break
        ("2026-01-07", True),   # first day after the break
        ("2026-02-06", True),   # last lecture day
        ("2026-02-07", False),  # lecture-free period
        ("2020-10-15", False),  # WiSe 2020/21 started late (2 November)
    ],
)
def test_lecture_period(day, expected):
    assert academic_features(pd.DatetimeIndex([day])).iloc[0]["is_lecture_period"] == expected


@pytest.mark.parametrize(
    "day, expected",
    [("2024-07-08", True), ("2024-08-20", True), ("2024-08-21", False), ("2024-05-21", True)],
    ids=["summer_first", "summer_last", "after_summer", "pentecost"],
)
def test_school_holiday(day, expected):
    assert academic_features(pd.DatetimeIndex([day])).iloc[0]["is_school_holiday"] == expected


@pytest.mark.parametrize("ranges", [LECTURE_PERIODS, SCHOOL_HOLIDAYS_NRW], ids=["lectures", "school"])
def test_calendar_lists_are_sorted_and_do_not_overlap(ranges):
    # Catches typos when the lists are extended by hand
    starts = [pd.Timestamp(r[0]) for r in ranges]
    ends = [pd.Timestamp(r[1]) for r in ranges]
    assert all(s <= e for s, e in zip(starts, ends))
    assert all(e < s for e, s in zip(ends, starts[1:]))


def test_lecture_breaks_lie_within_their_lecture_period():
    for start, end, first, last in LECTURE_PERIODS:
        assert start <= first <= last <= end


def test_academic_features_refuse_dates_beyond_the_known_calendar():
    with pytest.raises(ValueError, match="calendars.py"):
        academic_features(pd.DatetimeIndex(["2026-10-10", "2031-01-01"]))


def make_weather(start: str, days: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=days, freq="D", name="date")
    return pd.DataFrame({"snowfall": 0.0, "temp_mean": 10.0}, index=index)


def test_build_features_combines_all_groups_without_the_target():
    daily = make_daily("2024-01-01", [100.0] * 400)
    dates = pd.DatetimeIndex(["2025-01-20", "2025-01-21"])
    features = build_features(daily, make_weather("2024-01-01", 400), dates)
    assert list(features.index) == list(dates)
    for column in ["weekday", "is_lecture_period", "period", "lag_7", "temp_mean", "snow_recent"]:
        assert column in features.columns
    assert "count" not in features.columns


def test_training_table_has_only_complete_days_and_the_target():
    daily = make_daily("2024-01-01", [100.0] * 30, incomplete=("2024-01-10",))
    table = training_table(daily, make_weather("2024-01-01", 30))
    assert len(table) == 29
    assert pd.Timestamp("2024-01-10") not in table.index
    assert (table["count"] == 100).all()


def test_lags_skip_special_days_off_except_lag_364():
    # Easter Monday 2025 (21 April) is a public holiday with few trips
    daily = make_daily("2024-01-01", [1000.0] * 500)
    daily.loc["2025-04-21", "count"] = 300
    daily.loc["2024-04-22", "count"] = 400  # 364 days before 21 April 2025
    row = lag_features(daily, pd.DatetimeIndex(["2025-04-28"])).iloc[0]
    # The Monday after: lag_7 would be Easter Monday, so it is missing ...
    assert np.isnan(row["lag_7"])
    # ... and the weekday mean uses the three regular Mondays before
    assert row["same_weekday_4w"] == 1000
    # lag_364 keeps special days: last year's holiday is a reference for this one
    holiday = lag_features(daily, pd.DatetimeIndex(["2025-04-21"])).iloc[0]
    assert holiday["lag_364"] == 400
