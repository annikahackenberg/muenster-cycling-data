import numpy as np
import pandas as pd
import pytest

from radmuenster.evaluation import (
    cross_validate,
    mae,
    mape,
    nrmse,
    summarize,
    time_series_folds,
    wape,
)


def test_folds_train_strictly_before_test():
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    for train, test in time_series_folds(dates, pd.Timestamp("2023-01-01"), 3):
        assert dates[train].max() < dates[test].min()


def test_folds_cover_the_test_range_once_and_expand():
    dates = pd.date_range("2022-01-01", "2023-08-15", freq="D")
    folds = list(time_series_folds(dates, pd.Timestamp("2023-01-01"), 3))
    # Jan-Mar, Apr-Jun, Jul-Aug 15 (shorter last block)
    assert len(folds) == 3
    tests = np.vstack([test for _, test in folds])
    assert (tests.sum(axis=0) == (dates >= "2023-01-01")).all()
    train_sizes = [train.sum() for train, _ in folds]
    assert train_sizes == sorted(train_sizes)
    assert dates[folds[1][1]].min() == pd.Timestamp("2023-04-01")


def test_folds_skip_empty_test_periods():
    # No data in Q2 2023 (e.g. an outage): that fold is left out
    dates = pd.date_range("2022-01-01", "2023-03-31").append(pd.date_range("2023-07-01", "2023-07-31"))
    folds = list(time_series_folds(dates, pd.Timestamp("2023-01-01"), 3))
    assert len(folds) == 2


def test_metrics():
    assert mae([100, 200], [110, 180]) == 15
    assert mape([100, 200], [110, 180]) == pytest.approx(0.1)
    assert wape([100, 200], [110, 180]) == pytest.approx(30 / 300)
    assert nrmse([100, 200], [110, 180]) == pytest.approx(np.sqrt((100 + 400) / 2) / 150)


def test_wape_is_robust_against_quiet_days_mape_is_not():
    # A snow day with 40 trips predicted as 4040, plus 9 normal days predicted exactly
    actual = [40] + [5000] * 9
    predicted = [4040] + [5000] * 9
    assert mape(actual, predicted) == pytest.approx(10.0)  # 1000 %
    assert wape(actual, predicted) == pytest.approx(4000 / 45040)  # under 9 %


def test_cross_validate_hides_the_target_from_the_model():
    index = pd.date_range("2022-01-01", "2023-06-30", freq="D", name="date")
    table = pd.DataFrame({"lag_7": 100.0, "count": 100.0}, index=index)
    seen_test_columns = []

    def model(train, test):
        seen_test_columns.append(list(test.columns))
        assert train.index.max() < test.index.min()
        return test["lag_7"].to_numpy()

    predictions = cross_validate(model, table, first_test_start=pd.Timestamp("2023-01-01"))
    assert all("count" not in columns for columns in seen_test_columns)
    # One prediction per test day, in two folds
    assert list(predictions.index) == list(index[index >= "2023-01-01"])
    assert sorted(predictions["fold"].unique()) == [0, 1]
    assert (predictions["actual"] == predictions["predicted"]).all()


def test_cross_validate_keeps_the_site_of_each_row():
    index = pd.date_range("2022-12-30", "2023-01-02", freq="D", name="date")
    # Two sites on the same dates; site b always has ten times the count
    table = pd.concat(
        [
            pd.DataFrame({"site": "a", "lag_7": 1.0, "count": 1.0}, index=index),
            pd.DataFrame({"site": "b", "lag_7": 10.0, "count": 10.0}, index=index),
        ]
    ).sort_index()

    predictions = cross_validate(
        lambda train, test: test["lag_7"].to_numpy(),
        table,
        keep=("site",),
        first_test_start=pd.Timestamp("2023-01-01"),
    )
    assert len(predictions) == 4
    assert (predictions["actual"] == predictions["site"].map({"a": 1.0, "b": 10.0})).all()


def test_summarize_weights_every_day_equally():
    # Fold 0: 1 day with error 100, fold 1: 3 days with error 0
    predictions = pd.DataFrame(
        {"fold": [0, 1, 1, 1], "actual": [100.0] * 4, "predicted": [200.0, 100.0, 100.0, 100.0]},
        index=pd.date_range("2023-01-01", periods=4, freq="D"),
    )
    overall = summarize(predictions).iloc[0]
    assert overall["n_days"] == 4
    assert overall["mae"] == 25  # not (100 + 0) / 2 = 50
    per_fold = summarize(predictions, by="fold")
    assert per_fold["mae"].tolist() == [100, 0]
