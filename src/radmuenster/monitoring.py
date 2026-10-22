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
from radmuenster.sites import low_count_days

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

# Forecasts of all sites: the same check per site, with a wider limit. Small
# sites and leisure routes depend more on the weather; in the cross-validation
# their error was up to 22 % (notebook 03).
MAX_SITE_BIAS = 0.15

# Low counts at any site (low_count_days): warn about runs that reach into
# the last LOW_COUNT_RECENT_DAYS; low days at most this far apart belong to
# the same run
LOW_COUNT_RECENT_DAYS = 7


def check_freshness(daily: pd.DataFrame, today: date) -> str | None:
    """Warn if yesterday has no counts at all: the city's update failed, or
    the download got an old version of the files.

    Counts as present: any quarter hour. The last day of a month is always
    cut short by the city, which is no reason for a warning.
    """
    latest = daily.index[daily["n_intervals"] > 0].max()
    yesterday = pd.Timestamp(today) - pd.Timedelta(days=1)
    if latest < yesterday:
        return (
            f"No counts after {latest:%Y-%m-%d} in the downloaded data: the city's daily update "
            "may have failed, or the download got an old version (compare with the city's repository)."
        )
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


def evaluate_site_forecasts(log: pd.DataFrame, counts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The log of the forecasts of all sites with the actual count of each
    forecast day, like evaluate_forecasts: only complete days, with the
    columns actual, error and pct_error. counts maps a site id to its daily
    counts (site_counts_by_id)."""
    actual = pd.concat(
        {site: daily["count"].where(daily["complete"]) for site, daily in counts.items()},
        names=["site_id", "forecast_date"],
    ).rename("actual")
    evaluated = log.join(actual, on=["site_id", "forecast_date"]).dropna(subset=["actual"])
    evaluated["error"] = evaluated["prediction"] - evaluated["actual"]
    evaluated["pct_error"] = evaluated["error"] / evaluated["actual"]
    return evaluated


def check_site_bias(log: pd.DataFrame, counts: dict[str, pd.DataFrame], today: date) -> str | None:
    """Warn about sites whose forecasts of the last BIAS_WINDOW_DAYS were on
    average more than MAX_SITE_BIAS too high or too low (at least
    MIN_EVALUATED_DAYS forecasts), in one warning for all of them.

    The bias is the sum of the errors over the sum of the actual counts, not
    the mean percentage error: a day on which a counter almost stopped (29
    trips at the Hammer Straße on 1 August 2026) would make that +24,000 %.
    """
    evaluated = evaluate_site_forecasts(log, counts)
    start = pd.Timestamp(today) - pd.Timedelta(days=BIAS_WINDOW_DAYS)
    recent = evaluated[evaluated["forecast_date"] >= start]
    sites = []
    for name, days in recent.groupby("site_name"):
        bias = days["error"].sum() / days["actual"].sum()
        if len(days) >= MIN_EVALUATED_DAYS and abs(bias) > MAX_SITE_BIAS:
            sites.append(f"{name} {bias:+.0%}")
    if not sites:
        return None
    return (
        f"Forecasts for all sites, last {BIAS_WINDOW_DAYS} days on average too high (+) or too low (-): "
        + ", ".join(sites)
        + ". The level of these sites may have shifted, or their counts are wrong."
    )


def check_low_counts(sites_daily: pd.DataFrame, today: date) -> str | None:
    """Warn about sites far below their usual count compared with the other
    sites in the last days (low_count_days): maybe a partial outage.

    Days in SITE_FAULTS are not ok and so never low: once an outage is
    entered there, the warning stops.
    """
    low = sites_daily.loc[low_count_days(sites_daily), ["site_name", "date"]]
    recent_start = pd.Timestamp(today) - pd.Timedelta(days=LOW_COUNT_RECENT_DAYS)
    sites = []
    for name, days in low.groupby("site_name")["date"]:
        days = days.sort_values()
        if days.iloc[-1] < recent_start:
            continue
        # The start of the latest run: the first day after the last gap
        gaps = days.diff() > pd.Timedelta(days=LOW_COUNT_RECENT_DAYS)
        since = days[gaps].iloc[-1] if gaps.any() else days.iloc[0]
        sites.append(f"{name} since {since:%Y-%m-%d}")
    if not sites:
        return None
    return (
        "Far below the usual count compared with the other sites, maybe a partial outage: "
        + ", ".join(sites)
        + ". Check them on the explore page; add real outages to SITE_FAULTS in sites.py."
    )


def run_checks(
    daily: pd.DataFrame, log: pd.DataFrame, today: date, sites_daily: pd.DataFrame | None = None
) -> list[str]:
    """All warnings for today; an empty list if everything looks fine.
    sites_daily (load_all_sites) adds the check of all sites."""
    warnings = [
        check_freshness(daily, today),
        check_incomplete_days(daily, today),
        check_inbound_share(daily, today),
        check_bias(log, daily, today),
        check_low_counts(sites_daily, today) if sites_daily is not None else None,
    ]
    return [w for w in warnings if w is not None]


def write_status(warnings: list[str], today: date, path: Path = STATUS_FILE) -> None:
    """Save today's warnings as JSON, for the GitHub Action and the web page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    status = {"checked_on": today.isoformat(), "warnings": warnings}
    path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
