import json
from datetime import date

import pandas as pd
import pytest

from radmuenster.monitoring import (
    CHECKED_WARNINGS,
    CHECKS,
    MonitoringWarning,
    check_site_bias,
    used_late_faults,
    evaluate_site_forecasts,
    check_low_counts,
    check_bias,
    check_freshness,
    check_inbound_share,
    check_incomplete_days,
    evaluate_forecasts,
    run_checks,
    write_status,
)

TODAY = date(2026, 10, 12)


def make_daily(start: str = "2025-11-01", end: str = "2026-10-11", share: float = 0.47):
    """Complete days with 6000 trips and the given inbound share."""
    dates = pd.date_range(start, end, freq="D", name="date")
    return pd.DataFrame(
        {"count": 6000.0, "count_in": 6000.0 * share, "n_intervals": 96, "complete": True},
        index=dates,
    )


def make_log(dates: pd.DatetimeIndex, prediction: float) -> pd.DataFrame:
    """Forecasts made on the evening before each date."""
    dates = pd.DatetimeIndex(dates)
    return pd.DataFrame(
        {"forecast_date": dates, "made_on": dates - pd.Timedelta(days=1), "model": "lightgbm", "prediction": prediction}
    )


def test_freshness_is_fine_with_yesterday():
    assert check_freshness(make_daily(), TODAY) is None


def test_freshness_warns_without_yesterday():
    warning = check_freshness(make_daily(end="2026-10-10"), TODAY)
    assert "Warendorfer Straße after 2026-10-10" in warning.text
    assert "city's repository" in warning.action
    assert "started too late" in warning.action


def test_freshness_accepts_a_cut_short_day():
    daily = make_daily()
    daily.loc["2026-10-11", ["n_intervals", "complete"]] = [6, False]
    assert check_freshness(daily, TODAY) is None


def test_incomplete_days_warns_from_three_days_on():
    daily = make_daily()
    daily.loc["2026-10-09":"2026-10-10", "complete"] = False
    assert check_incomplete_days(daily, TODAY) is None
    daily.loc["2026-10-08", "complete"] = False
    assert "3 of the last 7 days" in check_incomplete_days(daily, TODAY, checked=[]).text


def test_incomplete_days_checked_by_hand_warn_again_about_later_days():
    daily = make_daily()
    daily.loc["2026-10-06":"2026-10-08", "complete"] = False
    checked = [("incomplete_days", "100034983", "2026-10-08", "the city repaired it")]
    warning = check_incomplete_days(daily, TODAY, checked)
    assert warning.checked and warning.action == ""
    assert warning.text.endswith("Checked on 2026-10-08: the city repaired it.")
    # Three more incomplete days after the check: a new problem
    daily.loc["2026-10-09":"2026-10-11", "complete"] = False
    warning = check_incomplete_days(daily, TODAY, checked)
    assert not warning.checked and "3 of the last 7 days" in warning.text


def test_inbound_share_is_fine_without_jump():
    assert check_inbound_share(make_daily(), TODAY) is None


def test_inbound_share_warns_after_jump():
    daily = make_daily()
    # Last 14 days: share drops from 0.47 to 0.40, like the jump of October 2025
    daily.loc["2026-09-28":, "count_in"] = 6000.0 * 0.40
    warning = check_inbound_share(daily, TODAY, checked=[])
    assert "0.40" in warning.text and "0.47" in warning.text
    # Code names only in what to do, not in the public text
    assert "PERIOD_STARTS" in warning.action and "PERIOD_STARTS" not in warning.text


def test_inbound_share_needs_enough_days_in_the_period():
    # Current period started 28 October 2025: shortly after, no reference yet
    daily = make_daily(start="2025-10-28", end="2025-11-20", share=0.40)
    assert check_inbound_share(daily, date(2025, 11, 21)) is None


def test_evaluate_forecasts_skips_incomplete_and_future_days():
    daily = make_daily()
    daily.loc["2026-10-10", "complete"] = False
    log = make_log(pd.date_range("2026-10-09", "2026-10-13"), prediction=6600.0)
    evaluated = evaluate_forecasts(log, daily)
    assert evaluated["forecast_date"].tolist() == [
        pd.Timestamp("2026-10-09"), pd.Timestamp("2026-10-11")
    ]
    assert evaluated["pct_error"].tolist() == pytest.approx([0.1, 0.1])


def test_bias_warns_when_forecasts_are_too_high_for_weeks():
    log = make_log(pd.date_range("2026-09-15", "2026-10-11"), prediction=6600.0)
    warning = check_bias(log, make_daily(), TODAY, checked=[])
    assert "10% too high" in warning.text
    # The issue says how to mark it as checked
    assert '("bias", "100034983", "2026-10-12", "<result>")' in warning.action


def test_bias_is_fine_when_errors_cancel_out():
    dates = pd.date_range("2026-09-15", "2026-10-11")
    log = make_log(dates, prediction=6600.0)
    log.loc[::2, "prediction"] = 5400.0  # every other day 10 % too low
    assert check_bias(log, make_daily(), TODAY) is None


def test_bias_needs_enough_forecasts():
    log = make_log(pd.date_range("2026-10-01", "2026-10-11"), prediction=9000.0)
    assert check_bias(log, make_daily(), TODAY) is None


def test_run_checks_and_write_status(tmp_path):
    daily = make_daily(end="2026-10-10")
    warnings = run_checks(daily, make_log(pd.DatetimeIndex([]), 0.0), TODAY)
    assert len(warnings) == 1
    path = tmp_path / "status.json"
    write_status(warnings + [MonitoringWarning("Checked.", checked=True)], TODAY, path)
    assert json.loads(path.read_text()) == {
        "checked_on": "2026-10-12",
        "warnings": [warnings[0].text],
        "actions": [warnings[0].action],
        "checked_warnings": ["Checked."],
    }


def make_low_table(sites: dict[str, list[str]]) -> pd.DataFrame:
    """Low days by site id; the site name is the id in capitals."""
    rows = [(site, site.upper(), day) for site, days in sites.items() for day in days]
    table = pd.DataFrame(rows, columns=["site_id", "site_name", "date"])
    table["date"] = pd.to_datetime(table["date"])
    return table


def test_low_counts_warn_about_recent_runs_only(monkeypatch):
    table = make_low_table(
        {"old": ["2026-09-01", "2026-09-02"], "new": ["2026-10-15", "2026-10-16", "2026-10-17"]}
    )
    monkeypatch.setattr("radmuenster.monitoring.low_count_days", lambda t: pd.Series(True, index=t.index))
    warnings = check_low_counts(table, date(2026, 10, 18), checked=[])
    assert len(warnings) == 1
    assert "NEW: far below its usual count since 2026-10-15" in warnings[0].text
    assert '("low_counts", "new", "2026-10-18", "<result>")' in warnings[0].action
    assert not warnings[0].checked


def test_low_counts_show_the_result_of_a_check(monkeypatch):
    table = make_low_table({"a": ["2026-10-15", "2026-10-16"], "b": ["2026-10-16", "2026-10-17"]})
    monkeypatch.setattr("radmuenster.monitoring.low_count_days", lambda t: pd.Series(True, index=t.index))
    checked = [("low_counts", "a", "2026-10-17", "no outage, rain")]
    a, b = check_low_counts(table, date(2026, 10, 18), checked)
    assert a == MonitoringWarning(
        "A: far below its usual count since 2026-10-15, compared with other sites, maybe a partial "
        "outage. Checked on 2026-10-17: no outage, rain.",
        checked=True,
    )
    # The check of site a does not hide site b
    assert not b.checked


def test_low_counts_after_the_check_warn_again_within_the_same_run(monkeypatch):
    # Checked on 17 October; two days later, without a gap, the site is low again
    table = make_low_table({"a": ["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"]})
    monkeypatch.setattr("radmuenster.monitoring.low_count_days", lambda t: pd.Series(True, index=t.index))
    [warning] = check_low_counts(table, date(2026, 10, 21), [("low_counts", "a", "2026-10-17", "rain")])
    assert not warning.checked and "since 2026-10-19" in warning.text


def test_low_counts_stay_checked_when_the_start_of_the_run_moves(monkeypatch):
    # Reported on 15 October and checked; later, 15 October is no longer judged low
    table = make_low_table({"a": ["2026-10-16", "2026-10-17"]})
    monkeypatch.setattr("radmuenster.monitoring.low_count_days", lambda t: pd.Series(True, index=t.index))
    [warning] = check_low_counts(table, date(2026, 10, 18), [("low_counts", "a", "2026-10-17", "rain")])
    assert warning.checked


def test_checks_by_hand_of_another_check_or_site_do_not_count(monkeypatch):
    table = make_low_table({"a": ["2026-10-16", "2026-10-17"]})
    monkeypatch.setattr("radmuenster.monitoring.low_count_days", lambda t: pd.Series(True, index=t.index))
    checked = [("site_bias", "a", "2026-10-17", "known"), ("low_counts", "b", "2026-10-17", "rain")]
    [warning] = check_low_counts(table, date(2026, 10, 18), checked)
    assert not warning.checked


def test_low_counts_are_fine_without_low_days(monkeypatch):
    table = make_low_table({"a": ["2026-10-17"]})
    monkeypatch.setattr("radmuenster.monitoring.low_count_days", lambda t: pd.Series(False, index=t.index))
    assert check_low_counts(table, date(2026, 10, 18)) == []


def test_checked_warnings_name_known_sites():
    from radmuenster.sites import SITES_BY_ID

    for check, site_id, checked_on, note in CHECKED_WARNINGS:
        assert check in CHECKS
        assert site_id in SITES_BY_ID
        assert pd.Timestamp(checked_on)
        assert note and not note.endswith(".")  # the text adds the full stop


def make_site_log(site_id: str, name: str, dates: pd.DatetimeIndex, prediction: float) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    return pd.DataFrame(
        {
            "forecast_date": dates,
            "made_on": dates - pd.Timedelta(days=1),
            "site_id": site_id,
            "site_name": name,
            "prediction": prediction,
        }
    )


def make_counts(
    dates: pd.DatetimeIndex,
    count: float,
    incomplete: tuple[str, ...] = (),
    faults: pd.DatetimeIndex | None = None,
    marked_on: str = "",
) -> pd.DataFrame:
    """Daily counts; the days of faults are incomplete, marked on marked_on,
    as in site_counts."""
    counts = pd.DataFrame({"count": count, "complete": True, "marked_on": pd.NaT}, index=dates)
    counts.loc[pd.DatetimeIndex(incomplete), "complete"] = False
    if faults is not None:
        counts.loc[faults, "complete"] = False
        counts.loc[faults, "marked_on"] = pd.Timestamp(marked_on)
    return counts


def test_used_late_faults_looks_at_the_input_days_made_before_the_mark():
    dates = pd.date_range("2025-01-01", "2027-10-31", freq="D")
    daily = make_counts(dates, 100.0, faults=pd.DatetimeIndex(["2026-10-01"]), marked_on="2026-11-15")
    forecasts = pd.DataFrame(
        [
            ("2026-10-02", "2026-10-01", False),  # 1 day after: the day before is never an input
            ("2026-10-03", "2026-10-02", True),  # 2 days: lag_2
            ("2026-10-09", "2026-10-08", True),  # 8 days: in mean_7d
            ("2026-10-10", "2026-10-09", False),  # 9 days: not an input
            ("2026-10-29", "2026-10-28", True),  # 28 days: same weekday 4 weeks earlier
            ("2026-10-29", "2026-11-15", False),  # made on the day marked: saw it as missing
            ("2027-09-30", "2027-09-29", False),  # 364 days, but made after it was marked
        ],
        columns=["forecast_date", "made_on", "used"],
    ).astype({"forecast_date": "datetime64[ns]", "made_on": "datetime64[ns]"})
    assert used_late_faults(forecasts, daily).tolist() == forecasts["used"].tolist()
    # Counts without the column have no known faults
    assert not used_late_faults(forecasts, daily[["count", "complete"]]).any()


def test_evaluate_site_forecasts_matches_site_and_day():
    dates = pd.date_range("2026-10-01", periods=3, freq="D")
    log = pd.concat([make_site_log("a", "A", dates, 110.0), make_site_log("b", "B", dates, 900.0)])
    counts = {"a": make_counts(dates, 100.0, incomplete=("2026-10-02",)), "b": make_counts(dates[:1], 1000.0)}
    evaluated = evaluate_site_forecasts(log, counts)
    # a: the incomplete day is left out; b: only the day with a count
    assert evaluated[["site_id", "actual", "error"]].values.tolist() == [
        ["a", 100.0, 10.0],
        ["a", 100.0, 10.0],
        ["b", 1000.0, -100.0],
    ]
    assert evaluated["pct_error"].round(2).tolist() == [0.1, 0.1, -0.1]


def test_site_bias_warns_only_about_biased_sites_with_enough_days():
    today = date(2026, 10, 30)
    dates = pd.date_range("2026-10-08", "2026-10-29", freq="D")  # 22 days
    log = pd.concat(
        [
            make_site_log("a", "Biased", dates, 120.0),  # 20 % too high
            make_site_log("b", "Fine", dates, 105.0),  # 5 %
            make_site_log("c", "Few days", dates[-5:], 200.0),
        ]
    )
    counts = {site: make_counts(dates, 100.0) for site in "abc"}
    [warning] = check_site_bias(log, counts, today, checked=[])
    assert "Forecasts for Biased were on average +20% off" in warning.text


def test_site_bias_checked_by_hand_comes_back_after_two_weeks():
    dates = pd.date_range("2026-10-08", "2026-11-03", freq="D")
    log = make_site_log("a", "Biased", dates, 120.0)
    checked = [("site_bias", "a", "2026-10-20", "counter is low, decided to keep it")]

    def warning_on(today: date) -> MonitoringWarning:
        # Counts up to the day before today, as in the daily run
        [warning] = check_site_bias(log, {"a": make_counts(dates[dates < pd.Timestamp(today)], 100.0)}, today, checked)
        return warning

    assert warning_on(date(2026, 10, 30)).checked
    # From 14 forecasts after the check on (21 October to 3 November), the bias warns again
    assert not warning_on(date(2026, 11, 4)).checked


def test_site_bias_leaves_out_forecasts_that_used_an_outage_marked_later():
    # Like the Grevener Straße in October 2026: the counter counted too little
    # until 20 October; forecasts that used those days were too low
    today = date(2026, 11, 10)
    outage = pd.date_range("2026-09-13", "2026-10-20", freq="D")
    dates = pd.date_range("2026-08-01", "2026-11-09", freq="D")
    after = dates[dates > outage[-1]]
    # The forecasts of the first week after it got the low counts as lag_2 and in mean_7d
    first_week = after[:8]
    log = make_site_log("a", "A", after, 100.0)
    log.loc[log["forecast_date"].isin(first_week), "prediction"] = 60.0
    # Marked on 5 November: the forecasts made before that and using it are
    # left out; 6 to 9 November are left, too few to judge
    marked = make_counts(dates, 100.0, faults=outage, marked_on="2026-11-05")
    assert check_site_bias(log, {"a": marked}, today, checked=[]) == []
    # Before the outage was marked, the same forecasts warned
    unmarked = make_counts(dates, 100.0)
    [warning] = check_site_bias(log, {"a": unmarked}, today, checked=[])
    assert "-16% off" in warning.text


def test_bias_leaves_out_forecasts_that_used_a_fault_marked_later():
    daily = make_daily()
    daily["marked_on"] = pd.NaT
    fault = slice("2026-09-20", "2026-09-30")
    daily.loc[fault, "complete"] = False
    # 17 forecasts of complete days, all 17 % too high
    log = make_log(pd.date_range("2026-09-14", "2026-10-11", freq="D"), 7000.0)
    # Marked before the forecasts: they saw the fault as missing and count
    daily.loc[fault, "marked_on"] = pd.Timestamp("2026-09-01")
    assert check_bias(log, daily, TODAY, checked=[]) is not None
    # Marked today: the 11 forecasts after the fault used it; 6 are left, too few to judge
    daily.loc[fault, "marked_on"] = pd.Timestamp("2026-10-12")
    assert check_bias(log, daily, TODAY, checked=[]) is None
    # The evaluation itself keeps them (the accuracy on the pages)
    assert len(evaluate_forecasts(log, daily)) == 17


def test_site_bias_is_fine_without_bias():
    dates = pd.date_range("2026-10-08", "2026-10-29", freq="D")
    log = make_site_log("a", "A", dates, 100.0)
    assert check_site_bias(log, {"a": make_counts(dates, 100.0)}, date(2026, 10, 30)) == []


def test_site_bias_is_not_thrown_off_by_a_day_with_almost_no_count():
    dates = pd.date_range("2026-10-08", "2026-10-29", freq="D")
    log = make_site_log("a", "A", dates, 100.0)
    counts = make_counts(dates, 100.0)
    # The counter almost stopped on one day: 1 trip instead of 100
    counts.loc["2026-10-20", "count"] = 1.0
    # Mean percentage error: +450 %; errors over counts: +4.7 %
    assert check_site_bias(log, {"a": counts}, date(2026, 10, 30)) == []
