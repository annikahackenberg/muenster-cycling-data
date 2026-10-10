from datetime import date

import pandas as pd
import pytest
import requests

from radmuenster.data import (
    download_month,
    load_intervals,
    month_range,
    parse_month_csv,
    to_daily,
)

SITE = "100034983"
HEADER = (
    "Datetime,100034983 (Warendorfer Straße),"
    "101034983 (Warendorfer Straße stadteinwärts),"
    "102034983 (Warendorfer Straße stadtauswärts),"
    "100034983-status,101034983-status,102034983-status\n"
)


def make_csv(
    day: str, counts: list[int], status: int | str = 0, dead_inbound: str | None = None
) -> str:
    """Build a monthly CSV in the City of Münster format for a single day.

    Each count is split evenly into both directions. With dead_inbound set, the
    inbound cell contains that value instead (e.g. "0" or "" for an empty cell)
    and the outbound channel gets the whole count.
    """
    times = pd.date_range(day, periods=len(counts), freq="15min")
    rows = []
    for t, c in zip(times, counts):
        if dead_inbound is None:
            c_in, c_out = str(c // 2), c - c // 2
        else:
            c_in, c_out = dead_inbound, c
        rows.append(f"{t:%Y-%m-%d %H:%M},{c},{c_in},{c_out},{status},{status},{status}")
    return HEADER + "\n".join(rows) + "\n"


def test_month_range_spans_year_boundary():
    assert month_range(date(2019, 11, 15), date(2020, 2, 1)) == [
        "2019-11",
        "2019-12",
        "2020-01",
        "2020-02",
    ]


def test_parse_keeps_total_and_directions():
    df = parse_month_csv(make_csv("2024-06-01", [10, 20]), SITE)
    assert list(df.columns) == ["count", "count_in", "count_out", "status"]
    assert df["count"].tolist() == [10, 20]


def test_daily_sum_matches_intervals():
    df = parse_month_csv(make_csv("2024-06-01", [2] * 96), SITE)
    daily = to_daily(df)
    assert len(daily) == 1
    assert daily["count"].iloc[0] == 192
    assert daily["count_in"].iloc[0] == 96
    assert daily["count_out"].iloc[0] == 96
    assert bool(daily["complete"].iloc[0])


def test_day_with_missing_intervals_is_incomplete():
    df = parse_month_csv(make_csv("2024-06-01", [2] * 50), SITE)
    assert not bool(to_daily(df)["complete"].iloc[0])


def test_day_with_status_flag_is_incomplete():
    df = parse_month_csv(make_csv("2024-06-01", [2] * 96, status=1), SITE)
    assert not bool(to_daily(df)["complete"].iloc[0])


def test_validated_status_counts_as_ok():
    df = parse_month_csv(make_csv("2024-06-01", [2] * 96, status="validated"), SITE)
    assert bool(to_daily(df)["complete"].iloc[0])


@pytest.mark.parametrize("dead_value", ["0", "", "1"], ids=["zeros", "empty_cells", "almost_dead"])
def test_day_with_dead_direction_is_incomplete(dead_value):
    # 40 trips per quarter hour = 3840 per day, well above MIN_COUNT_FOR_SHARE_CHECK
    df = parse_month_csv(make_csv("2023-11-01", [40] * 96, dead_inbound=dead_value), SITE)
    daily = to_daily(df)
    assert daily["count"].iloc[0] == 3840
    assert bool(daily["direction_fault"].iloc[0])
    assert not bool(daily["complete"].iloc[0])


def test_quiet_day_with_uneven_directions_stays_complete():
    # Per quarter hour 1 inbound and 4 outbound: 480 trips per day with a 20 %
    # inbound share, but too few trips to judge the share
    df = parse_month_csv(make_csv("2021-02-08", [4] * 96, dead_inbound="1"), SITE)
    daily = to_daily(df)
    assert not bool(daily["direction_fault"].iloc[0])
    assert bool(daily["complete"].iloc[0])


def test_quiet_day_with_dead_direction_is_incomplete():
    # Few trips, but a direction with nothing at all is still a fault
    # (e.g. Christmas 2023 during the inbound outage)
    df = parse_month_csv(make_csv("2023-12-25", [2] * 96, dead_inbound=""), SITE)
    assert bool(to_daily(df)["direction_fault"].iloc[0])


class FakeResponse:
    """Stand-in for requests.Response with only what download_month uses."""

    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeServer:
    """Fake for requests.get: answers from a dict, unknown months return 404."""

    def __init__(self):
        self.responses: dict[str, FakeResponse] = {}
        self.calls: list[str] = []

    def get(self, url: str, timeout: float) -> FakeResponse:
        month = url.rsplit("/", 1)[-1].removesuffix(".csv")
        self.calls.append(month)
        return self.responses.get(month, FakeResponse(404))


@pytest.fixture
def server(monkeypatch):
    """Replace requests.get with a FakeServer, so no test touches the network."""
    fake = FakeServer()
    monkeypatch.setattr(requests, "get", fake.get)
    return fake


def test_download_month_returns_none_if_month_missing(server, tmp_path):
    assert download_month(SITE, "2018-01", tmp_path) is None
    assert not (tmp_path / SITE / "2018-01.csv").exists()


def test_download_month_caches_past_month(server, tmp_path):
    server.responses["2024-06"] = FakeResponse(200, "csv text")
    assert download_month(SITE, "2024-06", tmp_path) == "csv text"
    assert (tmp_path / SITE / "2024-06.csv").read_text(encoding="utf-8") == "csv text"


def test_download_month_does_not_cache_current_month(server, tmp_path):
    current = pd.Timestamp.today().strftime("%Y-%m")
    server.responses[current] = FakeResponse(200, "csv text")
    assert download_month(SITE, current, tmp_path) == "csv text"
    assert not (tmp_path / SITE / f"{current}.csv").exists()


def test_download_month_reads_cache_without_request(server, tmp_path):
    cache_file = tmp_path / SITE / "2024-06.csv"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("cached text", encoding="utf-8")
    assert download_month(SITE, "2024-06", tmp_path) == "cached text"
    assert server.calls == []


def test_download_month_raises_on_server_error(server, tmp_path):
    server.responses["2024-06"] = FakeResponse(500)
    with pytest.raises(requests.HTTPError):
        download_month(SITE, "2024-06", tmp_path)


def test_load_intervals_skips_missing_months(server, tmp_path):
    server.responses["2024-06"] = FakeResponse(200, make_csv("2024-06-01", [1, 2]))
    server.responses["2024-08"] = FakeResponse(200, make_csv("2024-08-01", [3]))
    df = load_intervals(SITE, date(2024, 6, 1), date(2024, 8, 31), tmp_path)
    assert df["count"].tolist() == [1, 2, 3]
    assert df.index.is_monotonic_increasing


def test_load_intervals_raises_if_no_data(server, tmp_path):
    with pytest.raises(ValueError):
        load_intervals(SITE, date(2018, 1, 1), date(2018, 3, 31), tmp_path)
