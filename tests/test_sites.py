import numpy as np
import pandas as pd

from radmuenster.data import parse_month_channels
from radmuenster.sites import (
    SITES,
    SITES_BY_ID,
    almost_zero_days,
    load_all_sites,
    low_count_days,
    mark_site_faults,
    to_site_daily,
)

SITE = "300037405"
CSV = (
    "Datetime,300037405 (Promenade (westl. Hals)),"
    "353409612 (Promenade (westl. Hals) Fahrräder FR Ludgerikreisel),"
    "353409613 (Promenade (westl. Hals) Fahrräder FR Aasee),"
    "300037405-status,353409612-status,353409613-status\n"
    "2026-09-01 00:00,5,2,3,0,0,0\n"
    "2026-09-01 00:15,4,4,0,4,0,0\n"
)


def test_parse_month_channels_keeps_total_status_and_channel_ids():
    df = parse_month_channels(CSV, SITE)
    assert list(df.columns) == ["total", "status", "353409612", "353409613"]
    assert df["total"].tolist() == [5, 4]
    assert df["status"].tolist() == ["0", "4"]


def test_site_list_has_unique_ids():
    assert len(SITES_BY_ID) == len(SITES) == 23


def make_intervals(days: pd.DatetimeIndex, per_channel: dict[str, list[float]], freq="15min"):
    """Intervals with a constant count per channel and day; per_channel gives
    the daily value of each channel for each day."""
    frames = []
    steps = int(pd.Timedelta(days=1) / pd.Timedelta(freq))
    for i, day in enumerate(days):
        index = pd.date_range(day, periods=steps, freq=freq)
        frame = pd.DataFrame({ch: values[i] / steps for ch, values in per_channel.items()}, index=index)
        frame.insert(0, "total", frame.sum(axis=1))
        frame.insert(1, "status", "0")
        frames.append(frame)
    return pd.concat(frames)


def test_dead_channel_is_marked_when_the_total_drops():
    days = pd.date_range("2026-01-01", periods=60, freq="D")
    a = [500.0] * 60
    b = [500.0] * 40 + [0.0] * 20  # channel b dies on day 40
    daily = to_site_daily(make_intervals(days, {"a": a, "b": b}))
    assert not daily["dead_channel"].iloc[:40].any()
    assert daily["dead_channel"].iloc[40:].all()
    assert daily["note"].iloc[45] == "channel b counted nothing (50% of the trips before)"
    assert not daily["ok"].iloc[45]


def test_rebuilt_channels_are_not_marked():
    # Trips move from channels a and b to c and d; the total stays the same
    days = pd.date_range("2026-01-01", periods=60, freq="D")
    old = [500.0] * 30 + [0.0] * 30
    new = [0.0] * 30 + [500.0] * 30
    daily = to_site_daily(make_intervals(days, {"a": old, "b": old, "c": new, "d": new}))
    assert not daily["dead_channel"].any()
    assert daily["ok"].all()


def test_hourly_days_are_not_partial():
    days = pd.date_range("2026-01-01", periods=3, freq="D")
    daily = to_site_daily(make_intervals(days, {"a": [100.0] * 3}, freq="1h"))
    assert not daily["partial"].any()


def test_missing_partial_and_zero_days():
    days = pd.date_range("2026-01-01", periods=5, freq="D")
    intervals = make_intervals(days, {"a": [100.0, 100.0, 0.0, 100.0, 100.0]})
    intervals = intervals.drop(intervals.loc["2026-01-02"].index)  # whole day missing
    intervals = intervals.drop(intervals.loc["2026-01-04 12:00":"2026-01-04 23:45"].index)  # cut short
    intervals.loc["2026-01-05 08:00", "status"] = "4"  # flagged by the city
    daily = to_site_daily(intervals)
    assert daily["missing"].tolist() == [False, True, False, False, False]
    assert np.isnan(daily.loc["2026-01-02", "total"])
    assert daily["zero"].tolist() == [False, False, True, False, False]
    assert daily["partial"].tolist() == [False, False, False, True, False]
    assert daily["status_fault"].tolist() == [False, False, False, False, True]
    assert daily["ok"].tolist() == [True, False, False, False, False]


def test_load_all_sites_reports_broken_sites_and_keeps_the_others(monkeypatch):
    days = pd.date_range("2026-01-01", periods=3, freq="D")

    refs = []

    def fake_load(site_id, start, end=None, cache_dir=None, ref=None):
        refs.append(ref)
        if site_id == "100034980":  # Hammer Straße
            raise ValueError("new file format")
        return make_intervals(days, {"a": [100.0] * 3})

    monkeypatch.setattr("radmuenster.sites.load_site_intervals", fake_load)
    monkeypatch.setattr("radmuenster.sites.latest_commit", lambda: "abc123")
    table, failed = load_all_sites()
    # All sites from the same commit
    assert set(refs) == {"abc123"}
    assert failed == {"Hammer Straße": "ValueError: new file format"}
    assert table["site_id"].nunique() == len(SITES) - 1
    assert list(table.columns[:3]) == ["site_id", "site_name", "date"]


def make_sites(n_sites: int = 6, start: str = "2025-08-04", weeks: int = 12) -> pd.DataFrame:
    """Table like load_all_sites without flags: sites of different size with
    the same weekly pattern (weekends half as busy) and a little noise."""
    days = pd.date_range(start, periods=7 * weeks, freq="D")
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n_sites):
        weekly = np.where(days.dayofweek >= 5, 0.5, 1.0)
        frames.append(
            pd.DataFrame(
                {
                    "site_id": f"s{i}",
                    "site_name": f"Site {i}",
                    "date": days,
                    "total": 1000.0 * (i + 1) * weekly * rng.normal(1.0, 0.03, len(days)),
                    "note": "",
                    "ok": True,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def scale(table: pd.DataFrame, site_id: str | None, first: str, last: str, factor: float) -> None:
    """Multiply the counts of one site (None: all sites) between two days."""
    rows = table["date"].between(first, last)
    if site_id is not None:
        rows &= table["site_id"] == site_id
    table.loc[rows, "total"] *= factor


def low_dates(table: pd.DataFrame, site_id: str) -> list[str]:
    low = table[low_count_days(table) & (table["site_id"] == site_id)]
    return [f"{day:%Y-%m-%d}" for day in low["date"]]


def test_low_count_days_finds_a_site_far_below_the_others():
    table = make_sites()
    scale(table, "s0", "2025-09-15", "2025-09-19", 0.3)
    assert low_dates(table, "s0") == [f"2025-09-{day}" for day in range(15, 20)]
    assert low_count_days(table)[table["site_id"] != "s0"].sum() == 0


def test_low_count_days_ignore_short_and_city_wide_drops():
    table = make_sites()
    # Two days are not enough
    scale(table, "s0", "2025-09-15", "2025-09-16", 0.3)
    # Rain everywhere: all sites lower alike
    scale(table, None, "2025-09-22", "2025-09-26", 0.6)
    assert low_count_days(table).sum() == 0


def test_low_count_days_keep_a_run_across_flagged_days():
    table = make_sites()
    scale(table, "s0", "2025-09-15", "2025-09-19", 0.3)
    # The city flagged the middle day: it is not judged, but does not end the run
    table.loc[(table["site_id"] == "s0") & (table["date"] == "2025-09-17"), "ok"] = False
    assert low_dates(table, "s0") == ["2025-09-15", "2025-09-16", "2025-09-18", "2025-09-19"]


def test_mark_site_faults_flags_the_days_with_their_reason():
    table = make_sites(n_sites=2, weeks=1)
    marked = mark_site_faults(table, [("s1", "2025-08-05", "2025-08-06", "half the usual count")])
    fault = marked["known_fault"]
    assert marked.loc[fault, ["site_id", "date"]].astype(str).values.tolist() == [
        ["s1", "2025-08-05"],
        ["s1", "2025-08-06"],
    ]
    assert (marked.loc[fault, "note"] == "half the usual count").all()
    assert (marked["ok"] == ~fault).all()


def test_almost_zero_days_flag_a_single_day_but_not_the_whole_city():
    table = make_sites()
    scale(table, "s0", "2025-09-16", "2025-09-16", 0.03)
    # Snow everywhere: all sites far below normal, no judgement
    scale(table, None, "2025-09-23", "2025-09-23", 0.03)
    flagged = table[almost_zero_days(table)]
    assert flagged[["site_id", "date"]].astype(str).values.tolist() == [["s0", "2025-09-16"]]
