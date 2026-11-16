"""Explain single forecasts of the LightGBM models with SHAP values.

LightGBM computes SHAP values itself (TreeSHAP, predict with
pred_contrib=True): for each forecast, one contribution per feature plus a
base value, the average prediction over the training data. They add up
exactly to the prediction. The models predict log(count), so the
contributions add up on the log scale and become factors on the count:

    prediction = base * factor of feature 1 * factor of feature 2 * ...

The contributions explain how the model weighed its inputs, not what causes
the traffic. Correlated features share their effect in a way that depends on
the trees: the mean of the last four same weekdays already contains the
weekday, the season and the measurement period, so part of "Sunday" shows up
under the recent level. The features are therefore summed into a few groups,
which are easier to read and less arbitrary than single features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from radmuenster.features import LAGS
from radmuenster.models import _lightgbm_inputs
from radmuenster.multi_site import GlobalModel
from radmuenster.weather import DAILY_VARIABLES

# Feature groups in the order they are shown. "this_site" exists only in the
# global model: what is specific to a site beyond its usual level. Not called
# "site", which is the column of the site id in the tables of the global model.
GROUPS = {
    "weekday_season": ("weekday", "month", "season_sin", "season_cos"),
    "holidays": ("is_holiday", "is_bridge_day", "is_christmas", "is_rosenmontag", "special_day_factor"),
    "uni_school": ("is_lecture_period", "is_school_holiday"),
    "weather": (*DAILY_VARIABLES.values(), "snow_recent"),
    "recent_level": (*(f"lag_{k}" for k in LAGS), "mean_7d", "same_weekday_4w", "period"),
    "this_site": ("site",),
}

GROUP_LABELS = {
    "weekday_season": "weekday and season",
    "holidays": "holidays and bridge days",
    "uni_school": "university and school holidays",
    "weather": "weather",
    "recent_level": "level of the last weeks",
    "this_site": "this site",
}

FEATURE_GROUP = {feature: group for group, features in GROUPS.items() for feature in features}

# Groups that change the forecast by less than this are left out of the text
MIN_CHANGE = 0.03


def grouped_contributions(model: LGBMRegressor, inputs: pd.DataFrame) -> pd.DataFrame:
    """SHAP values of model for the rows of inputs, summed per group.

    One row per row of inputs, one column per group the model uses (log
    scale) and the column base (log of the average prediction). Raises
    ValueError for a feature without a group, so a new feature cannot drop
    out of the explanation unnoticed.
    """
    ungrouped = sorted(set(inputs.columns) - set(FEATURE_GROUP))
    if ungrouped:
        raise ValueError(f"Features without a group in GROUPS: {ungrouped}")

    contributions = pd.DataFrame(
        model.predict(inputs, pred_contrib=True),
        index=inputs.index,
        columns=[*inputs.columns, "base"],
    )
    # Transposed, so that groupby sums the feature columns of each row
    features = contributions.drop(columns="base").T
    grouped = features.groupby(features.index.map(FEATURE_GROUP)).sum().T
    used = [group for group in GROUPS if group in grouped]
    return pd.concat([grouped[used], contributions["base"]], axis=1)


def _as_factors(grouped: pd.DataFrame, scale: np.ndarray | float = 1.0) -> pd.DataFrame:
    """From log contributions to the base count, a factor per group and the
    prediction (base times all factors). scale turns the base back into
    trips where the model works on scaled counts."""
    factors = np.exp(grouped.drop(columns="base"))
    result = pd.DataFrame({"base": np.exp(grouped["base"]) * scale}, index=grouped.index)
    result[factors.columns] = factors
    result["prediction"] = result["base"] * factors.prod(axis=1)
    return result


def explain_lightgbm(model: LGBMRegressor, factors: pd.Series, table: pd.DataFrame) -> pd.DataFrame:
    """Explanation of the forecasts of a model from models.fit_lightgbm for
    the rows of table (output of build_features).

    Columns: base (average prediction over the training data, in trips), one
    factor per group, prediction (equal to predict_lightgbm).
    """
    return _as_factors(grouped_contributions(model, _lightgbm_inputs(table, factors)))


def explain_global(fitted: GlobalModel, table: pd.DataFrame) -> pd.DataFrame:
    """Like explain_lightgbm for the global model of all sites.

    The global model works on counts divided by the median of each site, so
    the base is an average day relative to the site's usual level, times that
    level: the explanation starts from a usual day at this site.
    """
    scale = table["site"].map(fitted.scale).to_numpy()
    return _as_factors(grouped_contributions(fitted.model, fitted.inputs(table)), scale)


def main_reasons(explanation: pd.Series, top: int = 3) -> list[tuple[str, float]]:
    """The groups that move one forecast most, as (group, factor), largest
    change first; groups below MIN_CHANGE are left out.

    explanation is one row of explain_lightgbm or explain_global. Compared on
    the log scale, so that halving (0.5) counts as much as doubling (2.0).
    """
    factors = explanation.drop(["base", "prediction"]).astype(float)
    factors = factors[(factors - 1).abs() >= MIN_CHANGE]
    order = np.log(factors).abs().sort_values(ascending=False).index[:top]
    return [(group, float(factors[group])) for group in order]


def reasons_text(explanation: pd.Series, reference: str = "an average day") -> str:
    """One sentence on the main reasons of a forecast, e.g. "Compared with an
    average day (5,548 trips): weather -22 %, level of the last weeks +25 %."

    reference names what the base stands for ("a usual day at this site" for
    the global model).
    """
    start = f"Compared with {reference} ({explanation['base']:,.0f} trips)"
    reasons = main_reasons(explanation)
    if not reasons:
        return f"{start}: no input changes the forecast by more than {MIN_CHANGE:.0%}."
    parts = [f"{GROUP_LABELS[group]} {(factor - 1) * 100:+.0f} %" for group, factor in reasons]
    return f"{start}: {', '.join(parts)}."
