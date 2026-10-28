"""Checks whether the daily forecast can still be trusted.

Each check returns a MonitoringWarning, or None if everything looks fine
(check_site_bias and check_low_counts: a list with one warning per site).
The checks look for problems the forecast itself does not notice:

- the city stopped delivering data
- a channel of the counting site failed (days marked incomplete)
- the counting site changed, visible as a jump in the inbound share, like in
  June 2024 and October 2025 (see PERIOD_STARTS in features.py)
- the forecast is off in the same direction for weeks, at the Warendorfer
  Straße or at any site
- a site counts far less than usual compared with the other sites (maybe a
  partial outage)

The warnings are public (web page, app, AI assistant), so each has a text for
visitors and, separately, what to do about it; only the issue of the daily
job shows the second.

A warning that was checked by hand and needs nothing done goes into
CHECKED_WARNINGS. The check then only judges the days after the check: the
pages show the result of the check instead of the warning, and a problem on
later days warns again.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import pandas as pd

from radmuenster.data import DEFAULT_SITE
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

# Warnings checked by hand that need nothing done: (check, site id, day of the
# check, result). The entry covers all days up to the day of the check; the
# check then only judges later days (the latest entry of a check and site
# counts). Checks: incomplete_days, inbound_share and bias (Warendorfer
# Straße, DEFAULT_SITE), site_bias and low_counts (any site). Missing counts
# and failed downloads cannot be checked away: they need a fix, or pass.
CHECKED_WARNINGS = [
    (
        "low_counts", "300037936", "2026-10-28",
        "no outage; rain hits this leisure route much harder than the other sites "
        "(12-17 hours of rain a day), as in October 2025",
    ),
]
CHECKS = {"incomplete_days", "inbound_share", "bias", "site_bias", "low_counts"}


@dataclass(frozen=True)
class MonitoringWarning:
    """One warning of the checks.

    text     what visitors of the pages read
    action   what to do about it, only for the issue
    checked  True if it was checked by hand (CHECKED_WARNINGS): text then
             ends with the result of the check, and there is nothing to do
    """

    text: str
    action: str = ""
    checked: bool = False


def _latest_check(
    check: str, site_id: str, checked: list[tuple[str, str, str, str]]
) -> tuple[pd.Timestamp, str] | None:
    """The latest entry of checked for this check and site: (day of the check, result)."""
    entries = [(pd.Timestamp(day), note) for c, s, day, note in checked if (c, s) == (check, site_id)]
    return max(entries, key=lambda entry: entry[0], default=None)


def _review(
    check: str,
    site_id: str,
    find: Callable[[pd.Timestamp | None], MonitoringWarning | None],
    today: date,
    checked: list[tuple[str, str, str, str]],
) -> MonitoringWarning | None:
    """The warning of a check, taking the checks by hand into account.

    find(after) is the check itself, judging only the days after `after`
    (None: all days). If the latest check by hand of this check and site
    covers the problem, the warning is shown as checked; if the days after
    it are a problem on their own, they warn again.
    """
    warning = find(None)
    entry = _latest_check(check, site_id, checked) if warning is not None else None
    if entry is None:
        later = warning
    else:
        checked_on, note = entry
        later = find(checked_on)
        if later is None:
            return replace(
                warning, text=f"{warning.text} Checked on {checked_on:%Y-%m-%d}: {note}.", action="", checked=True
            )
    if later is None:
        return None
    return replace(
        later,
        action=f"{later.action} If it needs nothing done, add "
        f'("{check}", "{site_id}", "{today:%Y-%m-%d}", "<result>") to CHECKED_WARNINGS in monitoring.py.',
    )


def _after(days: pd.Series | pd.DataFrame, after: pd.Timestamp | None):
    """The rows of days (indexed by date) after the day after."""
    return days if after is None else days.loc[after + pd.Timedelta(days=1) :]


def check_freshness(daily: pd.DataFrame, today: date) -> MonitoringWarning | None:
    """Warn if yesterday has no counts at all: the city's update failed, or
    the download got an old version of the files.

    Counts as present: any quarter hour. The last day of a month is always
    cut short by the city, which is no reason for a warning.
    """
    latest = daily.index[daily["n_intervals"] > 0].max()
    yesterday = pd.Timestamp(today) - pd.Timedelta(days=1)
    if latest < yesterday:
        return MonitoringWarning(
            f"No counts after {latest:%Y-%m-%d}: the forecast uses older counts.",
            "The city's daily update may have failed, or the download got an old version "
            "(compare with the city's repository).",
        )
    return None


def check_incomplete_days(
    daily: pd.DataFrame, today: date, checked: list[tuple[str, str, str, str]] = CHECKED_WARNINGS
) -> MonitoringWarning | None:
    """Warn if many of the last RECENT_DAYS days are incomplete.

    A dead channel marks every day as incomplete (direction_fault); the
    forecast then runs without recent lags.
    """
    start = pd.Timestamp(today) - pd.Timedelta(days=RECENT_DAYS)

    def find(after: pd.Timestamp | None) -> MonitoringWarning | None:
        recent = _after(daily.loc[start:], after)
        n_incomplete = int((~recent["complete"]).sum())
        if n_incomplete < MAX_INCOMPLETE_DAYS:
            return None
        return MonitoringWarning(
            f"{n_incomplete} of the last {RECENT_DAYS} days at the Warendorfer Straße are incomplete "
            "or faulty: the forecast may be less accurate.",
            "Check the counting site's channels on the explore page.",
        )

    return _review("incomplete_days", DEFAULT_SITE, find, today, checked)


def check_inbound_share(
    daily: pd.DataFrame, today: date, checked: list[tuple[str, str, str, str]] = CHECKED_WARNINGS
) -> MonitoringWarning | None:
    """Warn if the inbound share of the last SHARE_WINDOW_DAYS days differs
    from the REFERENCE_DAYS before by more than MAX_SHARE_DEVIATION.

    The reference never reaches back beyond the start of the current
    measurement period. Both sides use the median of the daily shares of
    complete days, so that single odd days do not matter.
    """
    complete = daily[daily["complete"]]
    share = complete["count_in"] / complete["count"]
    window_start = pd.Timestamp(today) - pd.Timedelta(days=SHARE_WINDOW_DAYS)
    reference_start = max(window_start - pd.Timedelta(days=REFERENCE_DAYS), PERIOD_STARTS[-1])
    reference = share.loc[reference_start : window_start - pd.Timedelta(days=1)]

    def find(after: pd.Timestamp | None) -> MonitoringWarning | None:
        recent = _after(share.loc[window_start:], after)
        if len(recent) < MIN_SHARE_DAYS or len(reference) < MIN_REFERENCE_DAYS:
            return None
        if abs(recent.median() - reference.median()) <= MAX_SHARE_DEVIATION:
            return None
        return MonitoringWarning(
            f"Share of trips into the city at the Warendorfer Straße in the last {SHARE_WINDOW_DAYS} days "
            f"is {recent.median():.2f}, against {reference.median():.2f} in the {REFERENCE_DAYS} days before: "
            "the counting site may have changed, so the forecast may be off.",
            "If the site changed, add a new entry to PERIOD_STARTS in features.py.",
        )

    return _review("inbound_share", DEFAULT_SITE, find, today, checked)


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


def check_bias(
    log: pd.DataFrame,
    daily: pd.DataFrame,
    today: date,
    checked: list[tuple[str, str, str, str]] = CHECKED_WARNINGS,
) -> MonitoringWarning | None:
    """Warn if the forecasts of the last BIAS_WINDOW_DAYS are on average more
    than MAX_BIAS too high or too low (at least MIN_EVALUATED_DAYS forecasts)."""
    evaluated = evaluate_forecasts(log, daily)
    start = pd.Timestamp(today) - pd.Timedelta(days=BIAS_WINDOW_DAYS)
    window = evaluated[evaluated["forecast_date"] >= start].set_index("forecast_date")

    def find(after: pd.Timestamp | None) -> MonitoringWarning | None:
        recent = _after(window, after)
        if len(recent) < MIN_EVALUATED_DAYS:
            return None
        bias = recent["pct_error"].mean()
        if abs(bias) <= MAX_BIAS:
            return None
        direction = "too high" if bias > 0 else "too low"
        return MonitoringWarning(
            f"Forecasts for the Warendorfer Straße were on average {abs(bias):.0%} {direction} "
            f"({len(recent)} days of the last {BIAS_WINDOW_DAYS}): the level may have shifted.",
            "Look for a shift in the counts on the explore page; if the site changed, add a new "
            "entry to PERIOD_STARTS in features.py.",
        )

    return _review("bias", DEFAULT_SITE, find, today, checked)


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


def check_site_bias(
    log: pd.DataFrame,
    counts: dict[str, pd.DataFrame],
    today: date,
    checked: list[tuple[str, str, str, str]] = CHECKED_WARNINGS,
) -> list[MonitoringWarning]:
    """Warn about each site whose forecasts of the last BIAS_WINDOW_DAYS were
    on average more than MAX_SITE_BIAS too high or too low (at least
    MIN_EVALUATED_DAYS forecasts).

    The bias is the sum of the errors over the sum of the actual counts, not
    the mean percentage error: a day on which a counter almost stopped (29
    trips at the Hammer Straße on 1 August 2026) would make that +24,000 %.
    """
    evaluated = evaluate_site_forecasts(log, counts)
    start = pd.Timestamp(today) - pd.Timedelta(days=BIAS_WINDOW_DAYS)
    recent = evaluated[evaluated["forecast_date"] >= start]
    warnings = []
    for (site_id, name), days in recent.groupby(["site_id", "site_name"]):
        days = days.set_index("forecast_date")

        def find(after: pd.Timestamp | None, days=days, name=name) -> MonitoringWarning | None:
            judged = _after(days, after)
            if len(judged) < MIN_EVALUATED_DAYS:
                return None
            bias = judged["error"].sum() / judged["actual"].sum()
            if abs(bias) <= MAX_SITE_BIAS:
                return None
            return MonitoringWarning(
                f"Forecasts for {name} were on average {bias:+.0%} off ({len(judged)} days of the last "
                f"{BIAS_WINDOW_DAYS}): the level of the site may have shifted, or its counts are wrong.",
                "Check the site on the explore page; add a real outage to SITE_FAULTS in sites.py.",
            )

        warning = _review("site_bias", site_id, find, today, checked)
        if warning is not None:
            warnings.append(warning)
    return warnings


def check_low_counts(
    sites_daily: pd.DataFrame, today: date, checked: list[tuple[str, str, str, str]] = CHECKED_WARNINGS
) -> list[MonitoringWarning]:
    """Warn about sites far below their usual count compared with the other
    sites in the last days (low_count_days): maybe a partial outage. One
    warning per site, so that a checked site does not hide another.

    Days in SITE_FAULTS are not ok and so never low: once an outage is
    entered there, the warning stops.
    """
    low = sites_daily.loc[low_count_days(sites_daily), ["site_id", "site_name", "date"]]
    recent_start = pd.Timestamp(today) - pd.Timedelta(days=LOW_COUNT_RECENT_DAYS)
    warnings = []
    for (site_id, name), days in low.groupby(["site_id", "site_name"])["date"]:
        days = pd.Series(days.values, index=days.values).sort_index()

        def find(after: pd.Timestamp | None, days=days, name=name) -> MonitoringWarning | None:
            judged = _after(days, after)
            if judged.empty or judged.iloc[-1] < recent_start:
                return None
            # The start of the latest run: the first day after the last gap
            gaps = judged.diff() > pd.Timedelta(days=LOW_COUNT_RECENT_DAYS)
            since = judged[gaps].iloc[-1] if gaps.any() else judged.iloc[0]
            return MonitoringWarning(
                f"{name}: far below its usual count since {since:%Y-%m-%d}, compared with the other "
                "sites, maybe a partial outage.",
                "Check it on the explore page; add a real outage to SITE_FAULTS in sites.py.",
            )

        warning = _review("low_counts", site_id, find, today, checked)
        if warning is not None:
            warnings.append(warning)
    return warnings


def run_checks(
    daily: pd.DataFrame, log: pd.DataFrame, today: date, sites_daily: pd.DataFrame | None = None
) -> list[MonitoringWarning]:
    """All warnings for today, including the checked ones; an empty list if
    everything looks fine. sites_daily (load_all_sites) adds the check of all
    sites."""
    warnings = [
        check_freshness(daily, today),
        check_incomplete_days(daily, today),
        check_inbound_share(daily, today),
        check_bias(log, daily, today),
    ]
    if sites_daily is not None:
        warnings += check_low_counts(sites_daily, today)
    return [w for w in warnings if w is not None]


def write_status(warnings: list[MonitoringWarning], today: date, path: Path = STATUS_FILE) -> None:
    """Save today's warnings as JSON, for the GitHub Action and the pages.

    warnings          the texts for visitors of the open warnings
    actions           what to do about each of them (same order), for the issue
    checked_warnings  the texts of the warnings checked by hand

    The web app reads warnings and checked_warnings; only the issue shows the
    actions. An issue is only needed while warnings is not empty.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    open_warnings = [w for w in warnings if not w.checked]
    status = {
        "checked_on": today.isoformat(),
        "warnings": [w.text for w in open_warnings],
        "actions": [w.action for w in open_warnings],
        "checked_warnings": [w.text for w in warnings if w.checked],
    }
    path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
