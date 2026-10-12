"""Data preparation for the Streamlit page (streamlit_app.py).

Kept apart from the page itself, so that it can be tested without Streamlit.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from radmuenster.monitoring import evaluate_forecasts

COUNTED = "Counted"
FORECAST = "Forecast"


def latest_forecast(log: pd.DataFrame) -> pd.Series:
    """The forecast with the latest forecast_date."""
    return log.loc[log["forecast_date"].idxmax()]


def is_stale(forecast: pd.Series, today: date) -> bool:
    """True if the latest forecast is not for tomorrow: the daily job did not
    run (failed, or switched off after the application phase)."""
    return forecast["forecast_date"].date() <= today


def chart_data(daily: pd.DataFrame, log: pd.DataFrame, start: date) -> pd.DataFrame:
    """Counts and forecasts from start on, in long format for the chart.

    Columns: date, series (COUNTED or FORECAST), trips. Days that are not
    complete get NaN as counted value, so the line breaks there instead of
    connecting across an outage.
    """
    start = pd.Timestamp(start)
    days = daily.loc[start:]
    counted = pd.DataFrame(
        {
            "date": days.index,
            "series": COUNTED,
            "trips": days["count"].where(days["complete"]).to_numpy(),
        }
    )
    recent = log[log["forecast_date"] >= start]
    forecast = pd.DataFrame(
        {"date": recent["forecast_date"], "series": FORECAST, "trips": recent["prediction"]}
    )
    return pd.concat([counted, forecast], ignore_index=True)


def forecast_accuracy(log: pd.DataFrame, daily: pd.DataFrame) -> dict[str, float]:
    """Error of the logged forecasts on complete days.

    Unlike the cross-validation, these forecasts used the weather forecast,
    so this is the honest error of the daily forecast.

    Keys: n (number of days), mae (trips), wape, bias (mean percentage error,
    positive: forecasts too high). NaN for the errors without any day.
    """
    evaluated = evaluate_forecasts(log, daily)
    if evaluated.empty:
        return {"n": 0, "mae": np.nan, "wape": np.nan, "bias": np.nan}
    absolute = evaluated["error"].abs()
    return {
        "n": len(evaluated),
        "mae": absolute.mean(),
        "wape": absolute.sum() / evaluated["actual"].sum(),
        "bias": evaluated["pct_error"].mean(),
    }
