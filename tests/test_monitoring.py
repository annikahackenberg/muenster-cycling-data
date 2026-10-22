import json
from datetime import date

import pandas as pd
import pytest

from radmuenster.monitoring import (
    check_site_bias,
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
    return pd.DataFrame({"forecast_date": dates, "model": "lightgbm", "prediction": prediction})


def test_freshness_is_fine_with_yesterday():
    assert check_freshness(make_daily(), TODAY) is None


def test_freshness_warns_without_yesterday():
    warning = check_freshness(make_daily(end="2026-10-10"), TODAY)
    assert "2026-10-10" in warning


def test_freshness_accepts_a_cut_short_day():
    daily = make_daily()
    daily.loc["2026-10-11", ["n_intervals", "complete"]] = [6, False]
    assert check_freshness(daily, TODAY) is None


def test_incomplete_days_warns_from_three_days_on():
    daily = make_daily()
    daily.loc["2026-10-09":"2026-10-10", "complete"] = False
    assert check_incomplete_days(daily, TODAY) is None
    daily.loc["2026-10-08", "complete"] = False
    assert "3 of the last 7 days" in check_incomplete_days(daily, TODAY)


def test_inbound_share_is_fine_without_jump():
    assert check_inbound_share(make_daily(), TODAY) is None


def test_inbound_share_warns_after_jump():
    daily = make_daily()
    # Last 14 days: share drops from 0.47 to 0.40, like the jump of October 2025
    daily.loc["2026-09-28":, "count_in"] = 6000.0 * 0.40
    warning = check_inbound_share(daily, TODAY)
    assert "0.40" in warning and "0.47" in warning


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
    assert "10% too high" in check_bias(log, make_daily(), TODAY)


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
    write_status(warnings, TODAY, path)
    assert json.loads(path.read_text()) == {"checked_on": "2026-10-12", "warnings": warnings}


def test_low_counts_warn_about_recent_runs_only(monkeypatch):
    days = pd.to_datetime(["2026-09-01", "2026-09-02", "2026-10-15", "2026-10-16", "2026-10-17"])
    table = pd.DataFrame({"site_name": ["Old", "Old", "New", "New", "New"], "date": days})
    monkeypatch.setattr("radmuenster.monitoring.low_count_days", lambda t: pd.Series(True, index=t.index))
    warning = check_low_counts(table, date(2026, 10, 18))
    assert "New since 2026-10-15" in warning
    assert "Old" not in warning


def test_low_counts_are_fine_without_low_days(monkeypatch):
    table = pd.DataFrame({"site_name": ["A"], "date": pd.to_datetime(["2026-10-17"])})
    monkeypatch.setattr("radmuenster.monitoring.low_count_days", lambda t: pd.Series(False, index=t.index))
    assert check_low_counts(table, date(2026, 10, 18)) is None


def make_site_log(site_id: str, name: str, dates: pd.DatetimeIndex, prediction: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"forecast_date": dates, "site_id": site_id, "site_name": name, "prediction": prediction}
    )


def make_counts(dates: pd.DatetimeIndex, count: float, incomplete: tuple[str, ...] = ()) -> pd.DataFrame:
    counts = pd.DataFrame({"count": count, "complete": True}, index=dates)
    counts.loc[pd.DatetimeIndex(incomplete), "complete"] = False
    return counts


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
    warning = check_site_bias(log, counts, today)
    assert "Biased +20%" in warning
    assert "Fine" not in warning and "Few days" not in warning


def test_site_bias_is_fine_without_bias():
    dates = pd.date_range("2026-10-08", "2026-10-29", freq="D")
    log = make_site_log("a", "A", dates, 100.0)
    assert check_site_bias(log, {"a": make_counts(dates, 100.0)}, date(2026, 10, 30)) is None


def test_site_bias_is_not_thrown_off_by_a_day_with_almost_no_count():
    dates = pd.date_range("2026-10-08", "2026-10-29", freq="D")
    log = make_site_log("a", "A", dates, 100.0)
    counts = make_counts(dates, 100.0)
    # The counter almost stopped on one day: 1 trip instead of 100
    counts.loc["2026-10-20", "count"] = 1.0
    # Mean percentage error: +450 %; errors over counts: +4.7 %
    assert check_site_bias(log, {"a": counts}, date(2026, 10, 30)) is None
