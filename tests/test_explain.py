import numpy as np
import pandas as pd
import pytest

from radmuenster.explain import (
    GROUPS,
    explain_global,
    explain_lightgbm,
    grouped_contributions,
    main_reasons,
    reasons_text,
)
from radmuenster.features import build_features, training_table
from radmuenster.models import _lightgbm_inputs, fit_lightgbm, predict_lightgbm
from radmuenster.multi_site import fit_global_lightgbm, sites_table
from radmuenster.weather import DAILY_VARIABLES

START, END = "2024-01-01", "2026-06-30"


def make_weather() -> pd.DataFrame:
    """Dry days, with 10 hours of rain on every fifth day."""
    dates = pd.date_range(START, END, freq="D", name="date")
    weather = pd.DataFrame({column: 1.0 for column in DAILY_VARIABLES.values()}, index=dates)
    weather["snowfall"] = 0.0
    rainy = np.arange(len(dates)) % 5 == 0
    weather["precipitation_hours"] = np.where(rainy, 10.0, 0.0)
    weather["precipitation"] = np.where(rainy, 8.0, 0.0)
    return weather


def make_daily(size: float, weather: pd.DataFrame) -> pd.DataFrame:
    """Weekends half as busy, rainy days 30 % fewer trips, a little noise."""
    dates = weather.index
    weekend = np.where(dates.dayofweek >= 5, 0.5, 1.0)
    rain = np.where(weather["precipitation_hours"] > 0, 0.7, 1.0)
    noise = np.random.default_rng(0).normal(1.0, 0.02, len(dates))
    return pd.DataFrame({"count": size * weekend * rain * noise, "complete": True}, index=dates)


@pytest.fixture(scope="module")
def weather():
    return make_weather()


@pytest.fixture(scope="module")
def own_model(weather):
    """Model from fit_lightgbm, trained before 2026-06, and the table of June."""
    table = training_table(make_daily(6000, weather), weather)
    model, factors = fit_lightgbm(table.loc[:"2026-05-31"])
    return model, factors, table.loc["2026-06-01":].drop(columns="count")


def test_explanation_adds_up_to_the_prediction(own_model):
    model, factors, june = own_model
    explanation = explain_lightgbm(model, factors, june)
    np.testing.assert_allclose(explanation["prediction"], predict_lightgbm(model, factors, june))
    groups = explanation.drop(columns=["base", "prediction"])
    np.testing.assert_allclose(explanation["base"] * groups.prod(axis=1), explanation["prediction"])


def test_own_model_has_all_groups_but_site(own_model):
    model, factors, june = own_model
    explanation = explain_lightgbm(model, factors, june)
    assert list(explanation.columns) == ["base", *(g for g in GROUPS if g != "this_site"), "prediction"]


def test_base_is_the_same_average_day_for_every_forecast(own_model):
    model, factors, june = own_model
    base = explain_lightgbm(model, factors, june)["base"]
    assert base.nunique() == 1
    # Between a weekend day in the rain and a dry workday
    assert 6000 * 0.5 * 0.7 < base.iloc[0] < 6000


def test_rain_lowers_the_forecast_through_the_weather(own_model):
    model, factors, june = own_model
    weather_factor = explain_lightgbm(model, factors, june)["weather"]
    rainy = june["precipitation_hours"] > 0
    assert weather_factor[rainy].max() < 1 < weather_factor[~rainy].min()


def test_feature_without_group_is_refused(own_model):
    model, factors, june = own_model
    inputs = _lightgbm_inputs(june, factors).rename(columns={"temp_max": "new_feature"})
    with pytest.raises(ValueError, match="new_feature"):
        grouped_contributions(model, inputs)


def test_global_explanation_starts_from_the_usual_day_of_each_site(weather):
    counts = {"small": make_daily(100, weather), "big": make_daily(1000, weather)}
    table = sites_table(counts, weather)
    fitted = fit_global_lightgbm(table[table.index < "2026-06-01"])
    june = table[table.index >= "2026-06-01"].drop(columns="count")

    explanation = explain_global(fitted, june)
    np.testing.assert_allclose(explanation["prediction"], fitted.predict(june))
    assert "this_site" in explanation
    # Same pattern at both sites, so the base differs by their size
    base = explanation["base"].groupby(june["site"].to_numpy()).first()
    assert base["big"] / base["small"] == pytest.approx(10, rel=0.05)


def test_explanation_for_tomorrow_works_without_count(own_model, weather):
    # As in the evening run: features of one future day from build_features
    model, factors, _ = own_model
    daily = make_daily(6000, weather).loc[:"2026-06-28"]
    features = build_features(daily, weather, pd.DatetimeIndex(["2026-06-30"]))
    explanation = explain_lightgbm(model, factors, features)
    assert len(explanation) == 1
    assert explanation["prediction"].iloc[0] == pytest.approx(predict_lightgbm(model, factors, features)[0])


def make_explanation(**factors: float) -> pd.Series:
    base = 5000.0
    return pd.Series({"base": base, **factors, "prediction": base * np.prod(list(factors.values()))})


def test_main_reasons_orders_by_change_on_the_log_scale_and_drops_small_ones():
    explanation = make_explanation(weather=0.5, recent_level=1.8, weekday_season=1.02, holidays=1.0)
    # Halving moves the forecast more than +80 % on the log scale
    assert main_reasons(explanation) == [("weather", 0.5), ("recent_level", 1.8)]


def test_main_reasons_keeps_the_top_ones():
    explanation = make_explanation(weather=0.8, recent_level=1.1, weekday_season=1.3, holidays=0.95)
    assert [group for group, _ in main_reasons(explanation, top=2)] == ["weekday_season", "weather"]


def test_reasons_text_names_the_reference_and_the_changes():
    explanation = make_explanation(weather=0.78, recent_level=1.25, uni_school=1.01)
    assert reasons_text(explanation) == (
        "Compared with an average day (5,000 trips): weather -22 %, level of the last weeks +25 %."
    )


def test_reasons_text_without_large_changes():
    text = reasons_text(make_explanation(weather=1.01), reference="a usual day at this site")
    assert text == (
        "Compared with a usual day at this site (5,000 trips): "
        "no input changes the forecast by more than 3%."
    )
