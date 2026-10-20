import numpy as np
import pandas as pd

from radmuenster.features import PERIOD_STARTS
from radmuenster.multi_site import global_lightgbm, per_site, site_counts, sites_table

START = "2022-01-01"
DAYS = 3 * 365


def make_daily(size: float) -> pd.DataFrame:
    """Daily counts with a weekly pattern (weekends half as busy) and a
    little noise, scaled to size."""
    index = pd.date_range(START, periods=DAYS, freq="D", name="date")
    weekend = index.dayofweek >= 5
    noise = np.random.default_rng(0).normal(1.0, 0.02, DAYS)
    return pd.DataFrame({"count": size * np.where(weekend, 0.5, 1.0) * noise, "complete": True}, index=index)


def make_weather() -> pd.DataFrame:
    index = pd.date_range(START, periods=DAYS, freq="D", name="date")
    return pd.DataFrame({"snowfall": 0.0, "temp_mean": 10.0}, index=index)


def test_site_counts_takes_the_total_and_the_ok_flag():
    site_daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "total": [1200.0, np.nan],
            "ok": [True, False],
        }
    )
    counts = site_counts(site_daily)
    assert list(counts.columns) == ["count", "complete"]
    assert counts["count"].iloc[0] == 1200
    assert counts["complete"].tolist() == [True, False]


def test_sites_table_stacks_the_sites_with_their_own_periods():
    counts = {"a": make_daily(1000), "b": make_daily(100)}
    table = sites_table(counts, make_weather(), period_starts={"a": PERIOD_STARTS})
    assert table.columns[0] == "site"
    assert table.index.is_monotonic_increasing
    assert (table["site"].value_counts() == table.index.nunique()).all()
    # Only site a has measurement periods
    after_jump = table.loc["2024-07-01"].set_index("site")["period"]
    assert after_jump.to_dict() == {"a": 1, "b": 0}


def test_per_site_fits_each_site_on_its_own_data():
    counts = {"a": make_daily(1000), "b": make_daily(100)}
    table = sites_table(counts, make_weather())
    train, test = table[table.index < "2024-01-01"], table[table.index >= "2024-01-01"]

    def mean_model(site_train, site_test):
        assert "site" not in site_train and "site" not in site_test
        return np.full(len(site_test), site_train["count"].mean())

    predicted = per_site(mean_model)(train, test.drop(columns="count"))
    expected = test["site"].map(train.groupby("site")["count"].mean())
    np.testing.assert_allclose(predicted, expected)


def test_global_lightgbm_predicts_each_site_on_its_own_scale():
    # Same pattern, but site b is ten times as big as site a
    counts = {"a": make_daily(100), "b": make_daily(1000)}
    table = sites_table(counts, make_weather())
    train, test = table[table.index < "2024-07-01"], table[table.index >= "2024-07-01"]

    predicted = global_lightgbm(train, test.drop(columns="count"))
    error = np.abs(predicted / test["count"].to_numpy() - 1)
    assert np.median(error) < 0.05
