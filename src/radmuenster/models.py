"""Forecast models, each as a fit_predict function for evaluation.cross_validate."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from radmuenster.features import LAGS, SPECIAL_DAYS


def naive_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Count of the same weekday one week earlier (lag_7).

    No training at all; this is the bar every real model has to beat. If lag_7
    is missing (outage a week earlier), fall back to the mean of the same
    weekday over four weeks, and then to the median of that weekday in the
    training data.
    """
    weekday_median = train.groupby("weekday")["count"].median()
    fallback = test["weekday"].map(weekday_median)
    return test["lag_7"].fillna(test["same_weekday_4w"]).fillna(fallback).to_numpy()


def _special_day_kind(table: pd.DataFrame) -> pd.Series:
    """Kind of special day off per row (the first matching column of
    SPECIAL_DAYS, so 25 December counts as a holiday), or None."""
    kind = pd.Series(None, index=table.index, dtype=object)
    for column in reversed(SPECIAL_DAYS):
        kind[table[column].to_numpy()] = column
    return kind


def holiday_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Naive baseline, multiplied by a factor per kind of special day off.

    Each factor is the median ratio of the actual count to the naive forecast
    on that kind of day in the training data, so it is learned from the past
    only: roughly 0.4 for public holidays, but about 0.9 for bridge days,
    when most people still work. Kinds without training examples keep 1.
    """
    return naive_baseline(train, test) * _special_day_factor(test, _special_day_factors(train))


def _special_day_factors(train: pd.DataFrame) -> pd.Series:
    """Median ratio of the actual count to the naive forecast per kind of
    special day off in the training data (e.g. is_holiday: 0.39)."""
    ratio = train["count"] / naive_baseline(train, train)
    return ratio.groupby(_special_day_kind(train)).median()


def _special_day_factor(table: pd.DataFrame, factors: pd.Series) -> np.ndarray:
    """Factor of each row's kind of special day; 1 for regular days and for
    kinds without training examples."""
    return _special_day_kind(table).map(factors).fillna(1.0).to_numpy()


# Inputs of the linear regression (see _linear_inputs)
LINEAR_CATEGORICAL = ["weekday", "period"]
LINEAR_FLAGS = [
    "is_holiday", "is_bridge_day", "is_christmas", "is_rosenmontag",
    "is_lecture_period", "is_school_holiday",
]
LINEAR_NUMERIC = [
    "season_sin", "season_cos", "temp_max", "precipitation_hours", "sunshine", "wind_max",
]
LINEAR_LAGS = ["lag_2", "lag_7", "lag_14", "lag_364", "mean_7d", "same_weekday_4w"]


def _linear_inputs(table: pd.DataFrame, weekday_median: pd.Series) -> pd.DataFrame:
    """Prepare the features for the linear regression.

    - Lags enter relative to the training median of their own weekday, as
      log(lag / median): 0 means a normal day for that weekday, +0.1 about
      10 % more. A lag of k days refers to the weekday k days earlier (lag_2
      of a Tuesday is a Sunday). With absolute lags, the weekday coefficients
      had to compensate for the weekday of lag_2 and lost their meaning.
    - A linear regression cannot handle NaN: a missing lag becomes 0, i.e.
      "no known deviation". Weekly lags first fall back to same_weekday_4w.
    - Snow enters as log(1 + cm): the first centimetres already stop most
      cyclists, 20 cm are not ten times worse than 2 cm.
    """
    inputs = table[LINEAR_CATEGORICAL + LINEAR_NUMERIC].copy()
    inputs[LINEAR_FLAGS] = table[LINEAR_FLAGS].astype(float)
    own_median = table["weekday"].map(weekday_median)
    same_weekday = np.log(table["same_weekday_4w"] / own_median)
    for k in LAGS:
        # Median of the weekday k days before the forecast day
        lag_weekday_median = ((table["weekday"] - k) % 7).map(weekday_median)
        relative = np.log(table[f"lag_{k}"] / lag_weekday_median)
        fallback = same_weekday if k % 7 == 0 else 0.0
        inputs[f"rel_lag_{k}"] = relative.fillna(fallback).fillna(0.0)
    inputs["rel_mean_7d"] = np.log(table["mean_7d"] / weekday_median.mean()).fillna(0.0)
    inputs["rel_same_weekday_4w"] = same_weekday.fillna(0.0)
    inputs["log_snow_recent"] = np.log1p(table["snow_recent"])
    return inputs


def fit_linear(train: pd.DataFrame) -> tuple[Pipeline, pd.Series]:
    """Fit the linear regression on log(count).

    Returns the fitted pipeline and the weekday medians it needs to prepare
    new data. The pipeline one-hot encodes weekday and period (one 0/1 column
    per value, so each weekday gets its own effect) and standardises the
    other inputs, so that the coefficients are comparable.
    """
    weekday_median = train.groupby("weekday")["count"].median()
    inputs = _linear_inputs(train, weekday_median)
    other = [c for c in inputs.columns if c not in LINEAR_CATEGORICAL]
    pipeline = Pipeline(
        [
            (
                "prepare",
                ColumnTransformer(
                    [
                        # handle_unknown="ignore": a period not seen in training
                        # (right after a measurement jump) gets all zeros
                        ("onehot", OneHotEncoder(handle_unknown="ignore"), LINEAR_CATEGORICAL),
                        ("scale", StandardScaler(), other),
                    ]
                ),
            ),
            ("regression", LinearRegression()),
        ]
    )
    pipeline.fit(inputs, np.log(train["count"]))
    return pipeline, weekday_median


def linear_regression(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Linear regression on log(count), see fit_linear."""
    pipeline, weekday_median = fit_linear(train)
    return np.exp(pipeline.predict(_linear_inputs(test, weekday_median)))


# Fixed, moderate settings: many small trees with a low learning rate.
# Tuning them on the test folds would leak the test data into the model;
# that needs a separate validation period inside each training set.
LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 0,
    "verbose": -1,
}


def _lightgbm_inputs(table: pd.DataFrame, factors: pd.Series) -> pd.DataFrame:
    """All features as numbers, plus special_day_factor.

    LightGBM handles NaN itself, so missing lags stay missing, and trees need
    neither scaling nor one-hot encoding.

    special_day_factor is the factor of the holiday baseline, learned from the
    training data. A leaf needs at least min_child_samples days, so trees
    cannot split on rare kinds of days (about 5 Rosenmontage, 20 bridge days in
    training); the factor puts all special days on one scale, and a split like
    "factor < 0.7" has enough days behind it.
    """
    inputs = table.drop(columns="count", errors="ignore").astype(float)
    inputs["special_day_factor"] = _special_day_factor(table, factors)
    return inputs


def fit_lightgbm(train: pd.DataFrame) -> tuple[LGBMRegressor, pd.Series]:
    """Fit gradient-boosted trees on log(count), like the linear regression.

    Returns the model and the special day factors it needs to prepare new data.
    """
    factors = _special_day_factors(train)
    model = LGBMRegressor(**LIGHTGBM_PARAMS)
    model.fit(_lightgbm_inputs(train, factors), np.log(train["count"]))
    return model, factors


def lightgbm(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Gradient-boosted trees on log(count), see fit_lightgbm."""
    model, factors = fit_lightgbm(train)
    return np.exp(model.predict(_lightgbm_inputs(test, factors)))
