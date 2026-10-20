"""Forecasts for several counting sites: one model per site or one global model.

All sites are stacked into one table with a column site, so that both
approaches run through the same cross-validation with the same folds. The
models are fit_predict functions as in models.py; per_site turns a model for
one site into one that fits a separate model for each site of the table.

Only the total of a site is used (see sites.py). Measurement jumps are only
known for the Warendorfer Straße (PERIOD_STARTS); other sites get no periods,
so a jump there looks like a trend to the models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from radmuenster.evaluation import FitPredict
from radmuenster.features import LAGS, training_table
from radmuenster.models import LIGHTGBM_PARAMS, _lightgbm_inputs, _special_day_factors
from radmuenster.sites import SITES

# Sites with data since 2020 or earlier: at least three years of training
# data before the first test quarter (2023)
LONG_HISTORY_SITES = [site for site in SITES if site.first_month <= "2020-01"]
# All other sites, with data since 2023 or 2024 only
SHORT_HISTORY_SITES = [site for site in SITES if site not in LONG_HISTORY_SITES]

# Features measured in trips, which the global model divides by the scale of the site
SCALED_COLUMNS = [*(f"lag_{k}" for k in LAGS), "mean_7d", "same_weekday_4w"]


def site_counts(site_daily: pd.DataFrame) -> pd.DataFrame:
    """One site of load_all_sites as daily counts like to_daily: count is the
    total, complete means no flag (ok), indexed by date."""
    return pd.DataFrame(
        {
            "count": site_daily["total"].to_numpy(dtype=float),
            "complete": site_daily["ok"].to_numpy(dtype=bool),
        },
        index=pd.DatetimeIndex(site_daily["date"], name="date"),
    )


def sites_table(
    counts: dict[str, pd.DataFrame],
    weather: pd.DataFrame,
    period_starts: dict[str, tuple[pd.Timestamp, ...]] | None = None,
) -> pd.DataFrame:
    """training_table of every site, stacked, with the site in the first column.

    counts maps a site name to its daily counts (to_daily or site_counts),
    period_starts maps a site name to its measurement periods; sites without
    an entry get none. Sorted by date, so every date appears once per site
    that has a complete day there.
    """
    period_starts = period_starts or {}
    frames = []
    for site, daily in counts.items():
        table = training_table(daily, weather, period_starts.get(site, ()))
        table.insert(0, "site", site)
        frames.append(table)
    # stable: the sites stay in the same order within a date
    return pd.concat(frames).sort_index(kind="stable")


def per_site(fit_predict: FitPredict) -> FitPredict:
    """A model that fits fit_predict separately for each site of the table.

    The site column is dropped before fit_predict sees the data, so any model
    from models.py works unchanged.
    """

    def fit_predict_per_site(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        predicted = np.full(len(test), np.nan)
        for site in test["site"].unique():
            in_test = (test["site"] == site).to_numpy()
            site_train = train[train["site"] == site].drop(columns="site")
            predicted[in_test] = fit_predict(site_train, test[in_test].drop(columns="site"))
        return predicted

    return fit_predict_per_site


def _scaled(table: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """table with the count and the lag features divided by the scale of each row's site."""
    scaled = table.copy()
    columns = [column for column in [*SCALED_COLUMNS, "count"] if column in scaled]
    # to_numpy: the index (dates) repeats, so aligning by index would fail
    scaled[columns] = scaled[columns].to_numpy() / table["site"].map(scale).to_numpy()[:, None]
    return scaled


def global_lightgbm(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """One LightGBM for all sites, on log(count) like models.lightgbm.

    The sites differ tenfold in size, and a split like "lag_7 < 3000" means
    something else at each of them. So the count and the lags are divided by
    the median count of the site, and 1.0 is a typical day everywhere; the
    trees can then learn weekly patterns, weather and holidays from all sites
    together. The site itself is a categorical feature, for what remains
    specific to a site (e.g. how much rain matters on a commuter route).

    The median comes from the training data only: the median of all days
    would carry the level of the test period into the model. A site without
    training data gets NaN.
    """
    scale = train.groupby("site")["count"].median()
    sites = pd.CategoricalDtype(sorted(scale.index))
    scaled_train = _scaled(train, scale)
    factors = _special_day_factors(scaled_train)

    def inputs(table: pd.DataFrame) -> pd.DataFrame:
        features = _lightgbm_inputs(_scaled(table, scale).drop(columns="site"), factors)
        # LightGBM treats a pandas category as a categorical feature
        features["site"] = pd.Categorical(table["site"], dtype=sites)
        return features

    model = LGBMRegressor(**LIGHTGBM_PARAMS)
    model.fit(inputs(train), np.log(scaled_train["count"]))
    return np.exp(model.predict(inputs(test))) * test["site"].map(scale).to_numpy()
