import numpy as np
import pandas as pd
import pytest

from radmuenster.findings import (
    month_profile,
    rain_effect,
    ratio_to_typical,
    unflagged_outages,
    weekday_profile,
)


def make_daily(start="2022-01-01", end="2023-12-31", weekend_factor=0.5):
    """Complete days: 1000 trips on workdays, weekend_factor of that at weekends."""
    dates = pd.date_range(start, end, freq="D", name="date")
    count = np.where(dates.dayofweek < 5, 1000.0, 1000.0 * weekend_factor)
    return pd.DataFrame(
        {"count": count, "complete": True, "direction_fault": False, "n_flagged": 0}, index=dates
    )


def test_weekday_profile_relative_to_workdays():
    profile = weekday_profile(make_daily()).set_index("weekday")["factor"]
    assert profile["Wed"] == pytest.approx(1.0)
    assert profile["Sun"] == pytest.approx(0.5)


def test_month_profile_removes_the_level_of_the_year():
    daily = make_daily()
    # 2023 is twice as busy, and July is 20 % busier in both years
    daily.loc["2023", "count"] *= 2
    daily.loc[daily.index.month == 7, "count"] *= 1.2
    profile = month_profile(daily)
    assert profile.attrs["years"] == [2022, 2023]
    factors = profile.set_index("month")["factor"]
    assert factors["Jul"] / factors["Jan"] == pytest.approx(1.2, rel=0.01)


def test_month_profile_skips_years_with_short_months():
    daily = make_daily()
    daily.loc["2022-03-01":"2022-03-20", "complete"] = False  # only 11 complete days left
    assert month_profile(daily).attrs["years"] == [2023]


def test_ratio_to_typical_leaves_out_special_days():
    ratio = ratio_to_typical(make_daily())
    assert pd.Timestamp("2022-12-25") not in ratio.index  # Christmas
    assert ratio.loc["2022-06-15"] == pytest.approx(1.0)


def test_rain_effect_compares_with_the_typical_day():
    daily = make_daily()
    weather = pd.DataFrame({"precipitation_hours": 0.0}, index=daily.index)
    rainy = pd.Timestamp("2022-06-15")  # a Wednesday
    weather.loc[rainy, "precipitation_hours"] = 10
    daily.loc[rainy, "count"] = 800.0
    effect = rain_effect(daily, weather).set_index(["day_type", "rain"])
    assert effect.loc[("Workday", "8+ h"), "factor"] == pytest.approx(0.8)
    assert effect.loc[("Workday", "8+ h"), "days"] == 1
    assert effect.loc[("Workday", "Dry"), "factor"] == pytest.approx(1.0)


def test_unflagged_outages_bridge_short_gaps():
    daily = make_daily()
    daily.loc["2022-09-01":"2022-10-15", "direction_fault"] = True
    # The last day of the month is missing, as in the city's data 2021-2025
    daily = daily.drop(pd.Timestamp("2022-09-30"))
    # Flagged by the city: not an unflagged outage
    daily.loc["2023-03-01":"2023-03-31", ["direction_fault", "n_flagged"]] = [True, 4]
    # Too short to count
    daily.loc["2023-05-01":"2023-05-05", "direction_fault"] = True
    outages = unflagged_outages(daily)
    assert len(outages) == 1
    assert outages.iloc[0]["start"] == pd.Timestamp("2022-09-01")
    assert outages.iloc[0]["end"] == pd.Timestamp("2022-10-15")
    assert outages.iloc[0]["days"] == 44
