"""Checks whether the daily forecast can still be trusted.

Each check returns a warning text, or None if everything looks fine. The
checks look for problems the forecast itself does not notice:

- the city stopped delivering data
- a channel of the counting site failed (days marked incomplete)
- the counting site changed, visible as a jump in the inbound share, like in
  June 2024 and October 2025 (see PERIOD_STARTS in features.py)
- the forecast is off in the same direction for weeks
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from radmuenster.features import PERIOD_STARTS

STATUS_FILE = Path("forecasts/status.json")

# Recent days in the incomplete-days check; warn from this many incomplete days on
RECENT_DAYS = 7
MAX_INCOMPLETE_DAYS = 3

# Inbound share: recent days against the REFERENCE_DAYS before them. The
# jumps so far were 6 and 10 percentage points. The share also drifts slowly
# (yearly median 0.575 in 2019, 0.605 in 2023), so a long reference would
# hide a jump: against all days since 2019 (0.586), the drop to 0.56 in June
# 2024 was only 2.6 points, against the 90 days before (0.618) 5.8 points.
SHARE_WINDOW_DAYS = 14
REFERENCE_DAYS = 90
MAX_SHARE_DEVIATION = 0.05
MIN_SHARE_DAYS = 7  # complete days needed in the window
MIN_REFERENCE_DAYS = 28  # complete days needed in the reference

# Forecast bias: mean percentage error of the forecasts of the last
# BIAS_WINDOW_DAYS. A single forecast is off by about 9 % (WAPE in the
# cross-validation), in both directions; the mean of a few weeks should be
# close to 0.
BIAS_WINDOW_DAYS = 28
MIN_EVALUATED_DAYS = 14
MAX_BIAS = 0.08


def check_freshness(daily: pd.DataFrame, today: date) -> str | None:
    """Warn if yesterday has no counts at all: the city's update failed.

    Counts as present: any quarter hour. The last day of a month is always
    cut short by the city, which is no reason for a warning.
    """
    latest = daily.index[daily["n_intervals"] > 0].max()
    yesterday = pd.Timestamp(today) - pd.Timedelta(days=1)
    if latest < yesterday:
        return f"No counts since {latest:%Y-%m-%d}: the city's daily update may have failed."
    return None


def check_incomplete_days(daily: pd.DataFrame, today: date) -> str | None:
    """Warn if many of the last RECENT_DAYS days are incomplete.

    A dead channel marks every day as incomplete (direction_fault); the
    forecast then runs without recent lags.
    """
    start = pd.Timestamp(today) - pd.Timedelta(days=RECENT_DAYS)
    recent = daily.loc[start:]
    n_incomplete = int((~recent["complete"]).sum())
    if n_incomplete >= MAX_INCOMPLETE_DAYS:
        return (
            f"{n_incomplete} of the last {RECENT_DAYS} days are incomplete or faulty: "
            "check the counting site's channels."
        )
    return None


def check_inbound_share(daily: pd.DataFrame, today: date) -> str | None:
    """Warn if the inbound share of the last SHARE_WINDOW_DAYS days differs
    from the REFERENCE_DAYS before by more than MAX_SHARE_DEVIATION.

    The reference never reaches back beyond the start of the current
    measurement period. Both sides use the median of the daily shares of
    complete days, so that single odd days do not matter.
    """
    complete = daily[daily["complete"]]
    share = complete["count_in"] / complete["count"]
    window_start = pd.Timestamp(today) - pd.Timedelta(days=SHARE_WINDOW_DAYS)
    recent = share.loc[window_start:]
    reference_start = max(window_start - pd.Timedelta(days=REFERENCE_DAYS), PERIOD_STARTS[-1])
    reference = share.loc[reference_start : window_start - pd.Timedelta(days=1)]
    if len(recent) < MIN_SHARE_DAYS or len(reference) < MIN_REFERENCE_DAYS:
        return None
    deviation = recent.median() - reference.median()
    if abs(deviation) > MAX_SHARE_DEVIATION:
        return (
            f"Inbound share of the last {SHARE_WINDOW_DAYS} days is {recent.median():.2f}, "
            f"against {reference.median():.2f} in the {REFERENCE_DAYS} days before: "
            "the counting site may have changed. If so, add a new entry to PERIOD_STARTS."
        )
    return None


def evaluate_forecasts(log: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """The forecast log with the actual count of each forecast day.

    Only forecasts of complete days are kept, with
      actual       the actual count
      error        prediction - actual (positive: forecast too high)
      pct_error    error / actual
    """
    actual = daily["count"].where(daily["complete"]).rename("actual")
    # join looks up each forecast_date in the index of daily
    evaluated = log.join(actual, on="forecast_date").dropna(subset=["actual"])
    evaluated["error"] = evaluated["prediction"] - evaluated["actual"]
    evaluated["pct_error"] = evaluated["error"] / evaluated["actual"]
    return evaluated


def check_bias(log: pd.DataFrame, daily: pd.DataFrame, today: date) -> str | None:
    """Warn if the forecasts of the last BIAS_WINDOW_DAYS are on average more
    than MAX_BIAS too high or too low (at least MIN_EVALUATED_DAYS forecasts)."""
    evaluated = evaluate_forecasts(log, daily)
    start = pd.Timestamp(today) - pd.Timedelta(days=BIAS_WINDOW_DAYS)
    recent = evaluated[evaluated["forecast_date"] >= start]
    if len(recent) < MIN_EVALUATED_DAYS:
        return None
    bias = recent["pct_error"].mean()
    if abs(bias) > MAX_BIAS:
        direction = "too high" if bias > 0 else "too low"
        return (
            f"Forecasts of the last {BIAS_WINDOW_DAYS} days were on average {abs(bias):.0%} "
            f"{direction} ({len(recent)} days): the level may have shifted."
        )
    return None


def run_checks(daily: pd.DataFrame, log: pd.DataFrame, today: date) -> list[str]:
    """All warnings for today; an empty list if everything looks fine."""
    warnings = [
        check_freshness(daily, today),
        check_incomplete_days(daily, today),
        check_inbound_share(daily, today),
        check_bias(log, daily, today),
    ]
    return [w for w in warnings if w is not None]


def write_status(warnings: list[str], today: date, path: Path = STATUS_FILE) -> None:
    """Save today's warnings as JSON, for the GitHub Action and the web page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    status = {"checked_on": today.isoformat(), "warnings": warnings}
    path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
