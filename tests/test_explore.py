from datetime import date

import numpy as np
import pandas as pd
import pytest

from radmuenster.explore import (
    MAX_SITES,
    aggregate,
    assign_colors,
    day_reasons,
    flagged_periods,
    ranking,
    select,
)

FLAGS = ["missing", "partial", "status_fault", "zero", "dead_channel"]


def make_sites_daily(sites: dict[str, list[float]], start: str = "2026-01-05") -> pd.DataFrame:
    """Table like load_all_sites: one row per site and day, no flags."""
    frames = []
    for name, totals in sites.items():
        frame = pd.DataFrame(
            {
                "site_id": name.lower(),
                "site_name": name,
                "date": pd.date_range(start, periods=len(totals), freq="D"),
                "total": totals,
                "note": "",
            }
        )
        for flag in FLAGS:
            frame[flag] = False
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table["ok"] = True
    return table


def flag(table: pd.DataFrame, site: str, first: str, last: str, column: str, note: str = "") -> None:
    rows = (table["site_name"] == site) & table["date"].between(first, last)
    table.loc[rows, column] = True
    table.loc[rows, "ok"] = False
    if note:
        table.loc[rows, "note"] = note


def test_colors_stay_with_their_site():
    colors = assign_colors(["A", "B", "C"], {})
    assert colors == {"A": 0, "B": 1, "C": 2}
    # Removing B keeps C's color; a new site takes the free slot
    colors = assign_colors(["A", "C", "D"], colors)
    assert colors == {"A": 0, "C": 2, "D": 1}
    assert MAX_SITES == 8


def test_day_reasons_explain_the_flag():
    table = make_sites_daily({"A": [100.0] * 3})
    flag(table, "A", "2026-01-05", "2026-01-05", "missing")
    flag(table, "A", "2026-01-06", "2026-01-06", "dead_channel", note="channel 1 counted nothing")
    assert day_reasons(table).tolist() == ["no data", "channel outage: channel 1 counted nothing", ""]


def test_day_reasons_explain_a_known_fault_with_its_reason():
    table = make_sites_daily({"A": [100.0] * 2})
    table["known_fault"] = False
    flag(table, "A", "2026-01-06", "2026-01-06", "known_fault", note="half the usual count")
    assert day_reasons(table).tolist() == ["", "partial outage: half the usual count"]


def test_aggregate_weeks_with_and_without_flagged_days():
    # Monday 5 January to Sunday 11 January: one week; the outage halves two days
    table = make_sites_daily({"A": [100.0, 100.0, 50.0, 50.0, 100.0, 100.0, 100.0]})
    flag(table, "A", "2026-01-07", "2026-01-08", "dead_channel", note="channel 1 counted nothing")
    raw = aggregate(table, "Week", leave_out_flagged=False)
    assert raw["period"].tolist() == [pd.Timestamp("2026-01-05")]
    assert raw["trips"].iloc[0] == pytest.approx(600 / 7)
    assert raw["flagged_days"].iloc[0] == 2
    assert raw["reasons"].iloc[0] == "channel outage: channel 1 counted nothing"
    clean = aggregate(table, "Week", leave_out_flagged=True)
    assert clean["trips"].iloc[0] == pytest.approx(100.0)


def test_aggregate_days_shows_gaps_for_left_out_days():
    table = make_sites_daily({"A": [100.0, 0.0, 100.0]})
    flag(table, "A", "2026-01-06", "2026-01-06", "zero")
    days = aggregate(table, "Day", leave_out_flagged=True)
    assert days["trips"].iloc[0] == 100.0 and np.isnan(days["trips"].iloc[1])


def test_ranking_uses_days_without_flags_and_reports_coverage():
    table = make_sites_daily({"Busy": [1000.0] * 10, "Quiet": [100.0] * 10, "Late": [500.0] * 2})
    flag(table, "Busy", "2026-01-05", "2026-01-06", "dead_channel")
    result = ranking(table, date(2026, 1, 5), date(2026, 1, 14)).set_index("site_name")
    assert result.index.tolist() == ["Busy", "Late", "Quiet"]
    assert result.loc["Busy", "ok_days"] == 8
    assert result.loc["Late", "coverage"] == pytest.approx(0.2)
    assert not result.loc["Late", "reliable"] and result.loc["Quiet", "reliable"]


def test_flagged_periods_skip_month_ends_and_join_outages():
    table = make_sites_daily({"A": [100.0] * 10}, start="2026-01-27")
    note = "channel 1 counted nothing"
    flag(table, "A", "2026-01-28", "2026-01-30", "dead_channel", note=note)
    flag(table, "A", "2026-01-31", "2026-01-31", "partial")  # cut short by the city
    flag(table, "A", "2026-02-01", "2026-02-03", "dead_channel", note=note)
    periods = flagged_periods(table)
    assert len(periods) == 1
    assert periods.iloc[0]["first"] == pd.Timestamp("2026-01-28")
    assert periods.iloc[0]["last"] == pd.Timestamp("2026-02-03")
    assert periods.iloc[0]["days"] == 6


def test_select_by_site_and_dates():
    table = make_sites_daily({"A": [1.0] * 5, "B": [2.0] * 5})
    rows = select(table, ["B"], date(2026, 1, 6), date(2026, 1, 7))
    assert rows["site_name"].unique().tolist() == ["B"]
    assert len(rows) == 2
