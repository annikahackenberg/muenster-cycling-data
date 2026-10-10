import numpy as np
import pandas as pd
import pytest

from radmuenster.features import SPECIAL_DAYS
from radmuenster.models import (
    LINEAR_FLAGS,
    LINEAR_LAGS,
    LINEAR_NUMERIC,
    _linear_inputs,
    holiday_baseline,
    lightgbm,
    linear_regression,
    naive_baseline,
)


def test_naive_baseline_uses_lag_7_and_falls_back_in_order():
    train = pd.DataFrame({"weekday": [0, 0, 1], "count": [100.0, 300.0, 50.0]})
    test = pd.DataFrame(
        {
            "weekday": [0, 0, 0],
            "lag_7": [500.0, np.nan, np.nan],
            "same_weekday_4w": [999.0, 400.0, np.nan],
        }
    )
    # lag_7, then same_weekday_4w, then the training median of Mondays (200)
    assert naive_baseline(train, test).tolist() == [500.0, 400.0, 200.0]


def make_table(counts, lag_7, holiday, bridge=None):
    table = pd.DataFrame({"weekday": 0, "count": counts, "lag_7": lag_7, "same_weekday_4w": lag_7})
    for column in SPECIAL_DAYS:
        table[column] = False
    table["is_holiday"] = holiday
    if bridge is not None:
        table["is_bridge_day"] = bridge
    return table


def test_holiday_baseline_learns_one_factor_per_kind_of_day():
    # Training: holidays at 40 % and 50 % (median 0.45), one bridge day at 90 %
    train = make_table(
        [400.0, 500.0, 900.0, 1000.0], [1000.0] * 4,
        holiday=[True, True, False, False], bridge=[False, False, True, False],
    )
    test = make_table(
        [np.nan] * 3, [2000.0] * 3, holiday=[True, False, False], bridge=[False, True, False]
    ).drop(columns="count")
    assert holiday_baseline(train, test).tolist() == pytest.approx([900.0, 1800.0, 2000.0])


def test_holiday_beats_christmas_when_both_apply():
    train = make_table([400.0, 800.0], [1000.0] * 2, holiday=[True, False])
    train["is_christmas"] = [False, True]  # Christmas factor 0.8, holiday factor 0.4
    test = make_table([np.nan], [1000.0], holiday=[True]).drop(columns="count")
    test["is_christmas"] = True  # 25 December
    assert holiday_baseline(train, test).tolist() == pytest.approx([400.0])


def make_linear_table(n_days: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic training table: count = 5000 * 0.4 on holidays * 0.5 on Sundays."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=n_days, freq="D", name="date")
    table = pd.DataFrame(index=index)
    table["weekday"] = index.dayofweek
    table["period"] = 0
    for column in LINEAR_FLAGS:
        table[column] = False
    table["is_holiday"] = rng.random(n_days) < 0.05
    for column in LINEAR_NUMERIC:
        table[column] = rng.normal(size=n_days)
    table["snow_recent"] = 0.0
    count = 5000 * np.where(table["is_holiday"], 0.4, 1.0) * np.where(table["weekday"] == 6, 0.5, 1.0)
    table["count"] = count
    for column in LINEAR_LAGS:
        table[column] = 5000.0
    return table


def test_linear_regression_learns_multiplicative_effects():
    table = make_linear_table(400)
    train, test = table.iloc[:300], table.iloc[300:].drop(columns="count")
    predicted = linear_regression(train, test)
    np.testing.assert_allclose(predicted, table["count"].iloc[300:], rtol=1e-6)


def test_linear_regression_handles_missing_lags_and_unseen_period():
    table = make_linear_table(400)
    train, test = table.iloc[:300], table.iloc[300:].drop(columns="count").copy()
    test.loc[test.index[:5], ["lag_7", "same_weekday_4w", "lag_364"]] = np.nan
    test["period"] = 1  # never seen in training
    predicted = linear_regression(train, test)
    assert np.isfinite(predicted).all()
    assert (predicted > 0).all()


def test_lags_are_relative_to_the_median_of_their_own_weekday():
    # lag_2 of a Tuesday is a Sunday: a normal Sunday gives 0, a missing one too
    train = make_linear_table(70)
    weekday_median = train.groupby("weekday")["count"].median()
    tuesdays = train[train["weekday"] == 1].iloc[[0, 1]].drop(columns="count").copy()
    tuesdays["lag_2"] = [weekday_median[6] * 1.1, np.nan]
    inputs = _linear_inputs(tuesdays, weekday_median)
    assert inputs["rel_lag_2"].tolist() == pytest.approx([np.log(1.1), 0.0])


def test_lightgbm_learns_an_interaction():
    # Snow only matters on workdays here: an interaction a linear model
    # without an explicit product term cannot express
    rng = np.random.default_rng(0)
    n = 2000
    table = pd.DataFrame({"weekday": rng.integers(0, 7, n), "snow_recent": rng.random(n) < 0.2})
    workday = table["weekday"] < 5
    table["count"] = np.where(workday, 6000.0, 3000.0) * np.where(workday & table["snow_recent"], 0.3, 1.0)
    # Columns the special day factor needs; no special days, no lags here
    for column in SPECIAL_DAYS:
        table[column] = False
    table[["lag_7", "same_weekday_4w"]] = np.nan
    train, test = table.iloc[:1500], table.iloc[1500:]
    predicted = lightgbm(train, test.drop(columns="count"))
    np.testing.assert_allclose(predicted, test["count"], rtol=0.05)


def test_lightgbm_handles_missing_values_and_is_reproducible():
    table = make_linear_table(400)
    train, test = table.iloc[:300], table.iloc[300:].drop(columns="count").copy()
    test.loc[test.index[:5], ["lag_7", "temp_max"]] = np.nan
    first, second = lightgbm(train, test), lightgbm(train, test)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
