from datetime import date

import pandas as pd

from radmuenster.data import month_range, parse_month_csv, to_daily

SITE = "100034983"
HEADER = (
    "Datetime,100034983 (Warendorfer Straße),"
    "101034983 (Warendorfer Straße stadteinwärts),"
    "102034983 (Warendorfer Straße stadtauswärts),"
    "100034983-status,101034983-status,102034983-status\n"
)


def make_csv(day: str, counts: list[int], status: int | str = 0) -> str:
    """Eine Monats-CSV im Format der Stadt Münster für einen Tag bauen."""
    times = pd.date_range(day, periods=len(counts), freq="15min")
    rows = [
        f"{t:%Y-%m-%d %H:%M},{c},{c // 2},{c - c // 2},{status},{status},{status}"
        for t, c in zip(times, counts)
    ]
    return HEADER + "\n".join(rows) + "\n"


def test_month_range_spans_year_boundary():
    assert month_range(date(2019, 11, 15), date(2020, 2, 1)) == [
        "2019-11",
        "2019-12",
        "2020-01",
        "2020-02",
    ]


def test_parse_keeps_only_total_column():
    df = parse_month_csv(make_csv("2024-06-01", [10, 20]), SITE)
    assert list(df.columns) == ["count", "status"]
    assert df["count"].tolist() == [10, 20]


def test_daily_sum_matches_intervals():
    df = parse_month_csv(make_csv("2024-06-01", [1] * 96), SITE)
    daily = to_daily(df)
    assert len(daily) == 1
    assert daily["count"].iloc[0] == 96
    assert bool(daily["complete"].iloc[0])


def test_day_with_missing_intervals_is_incomplete():
    df = parse_month_csv(make_csv("2024-06-01", [1] * 50), SITE)
    assert not bool(to_daily(df)["complete"].iloc[0])


def test_day_with_status_flag_is_incomplete():
    df = parse_month_csv(make_csv("2024-06-01", [1] * 96, status=1), SITE)
    assert not bool(to_daily(df)["complete"].iloc[0])


def test_validated_status_counts_as_ok():
    df = parse_month_csv(make_csv("2024-06-01", [1] * 96, status="validated"), SITE)
    assert bool(to_daily(df)["complete"].iloc[0])
