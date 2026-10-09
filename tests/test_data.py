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


def make_csv(day: str, counts: list[int], status: int | str = 0) -> str:
    """Build a monthly CSV in the City of Münster format for a single day."""
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
