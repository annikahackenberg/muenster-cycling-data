"""Time-based cross-validation for the daily forecast.

Every fold trains on all days before a test period and tests on that period,
so a model never sees the future it is evaluated on. A model is any function

    fit_predict(train: DataFrame, test: DataFrame) -> array of predictions

where train is a slice of training_table (features plus the target count) and
test contains the features of the test days only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd

FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]

# First test period: 2023 leaves four years of training data in the first fold
# and still gives more than three years of test folds
FIRST_TEST_START = pd.Timestamp("2023-01-01")
TEST_MONTHS = 3


def time_series_folds(
    dates: pd.DatetimeIndex,
    first_test_start: pd.Timestamp = FIRST_TEST_START,
    test_months: int = TEST_MONTHS,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_mask, test_mask) pairs over dates.

    Test periods are consecutive blocks of test_months, starting at
    first_test_start; the last block may be shorter. Each training set holds
    all dates before its test period (expanding window).
    """
    dates = pd.DatetimeIndex(dates)
    start = pd.Timestamp(first_test_start)
    while start <= dates.max():
        end = start + pd.DateOffset(months=test_months)
        test = (dates >= start) & (dates < end)
        if test.any():
            yield np.asarray(dates < start), np.asarray(test)
        start = end


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error in bicycle trips."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, as a fraction (0.1 = 10 %).

    Days with very few trips (snow) produce huge percentages and dominate this
    mean; compare it with the MAE.
    """
    actual = np.asarray(actual, dtype=float)
    return float(np.mean(np.abs(actual - np.asarray(predicted)) / actual))


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted absolute percentage error: sum of errors / sum of actual values.

    Like MAPE a fraction, but each day counts by its error in trips, so a snow
    day with 42 trips does not blow it up.
    """
    actual = np.asarray(actual, dtype=float)
    return float(np.sum(np.abs(actual - np.asarray(predicted))) / np.sum(actual))


def nrmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error divided by the mean of the actual values.

    Squaring makes large errors count more: an NRMSE well above the WAPE means
    a few days with big misses (holidays, snow, measurement jumps).
    """
    actual = np.asarray(actual, dtype=float)
    return float(np.sqrt(np.mean((actual - np.asarray(predicted)) ** 2)) / np.mean(actual))


def cross_validate(fit_predict: FitPredict, table: pd.DataFrame, **fold_args) -> pd.DataFrame:
    """Predict every test day of every fold of table (the output of training_table).

    Returns one row per test day with the columns fold, actual and predicted,
    indexed by date. Summarise it with summarize; computing the metrics over
    the individual days (instead of averaging fold metrics) keeps short folds,
    e.g. 6 test days during an outage, from weighing as much as a full quarter.
    """
    frames = []
    for fold, (train_mask, test_mask) in enumerate(time_series_folds(table.index, **fold_args)):
        train = table[train_mask]
        # The model gets the test features without the target
        test = table[test_mask].drop(columns="count")
        frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "actual": table.loc[test_mask, "count"].to_numpy(dtype=float),
                    "predicted": np.asarray(fit_predict(train, test), dtype=float),
                },
                index=test.index,
            )
        )
    return pd.concat(frames)


def summarize(predictions: pd.DataFrame, by: str | pd.Series | None = None) -> pd.DataFrame:
    """Number of days, MAE, WAPE, MAPE and NRMSE of the output of cross_validate.

    MAE (trips) and WAPE (fraction) are the main metrics; MAPE is dominated by
    days with very few trips, NRMSE by a few large errors.

    by groups the days, e.g. by="fold" or by=predictions.index.year;
    without by there is one row for all test days.
    """
    def metrics(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n_days": len(group),
                "mae": mae(group["actual"], group["predicted"]),
                "wape": wape(group["actual"], group["predicted"]),
                "mape": mape(group["actual"], group["predicted"]),
                "nrmse": nrmse(group["actual"], group["predicted"]),
            }
        )

    if by is None:
        return metrics(predictions).to_frame("all").T
    return predictions.groupby(by).apply(metrics)
