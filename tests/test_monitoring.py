import json
from datetime import date

import pandas as pd
import pytest

from radmuenster.monitoring import (
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
