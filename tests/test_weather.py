import json
from datetime import date

import pytest
import requests

from radmuenster.weather import (
    DAILY_VARIABLES,
    download_weather_year,
    load_weather,
    parse_weather_json,
)


def make_json(days: list[str], sunshine_seconds: float = 3600.0) -> str:
    """An Open-Meteo response with the same value for every variable and day."""
    daily = {"time": days}
    for variable in DAILY_VARIABLES:
        daily[variable] = [1.0] * len(days)
    daily["sunshine_duration"] = [sunshine_seconds] * len(days)
    return json.dumps({"daily": daily})


class FakeResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeWeatherServer:
    """Fake for requests.get: answers per year, records requested years."""

    def __init__(self):
        self.responses: dict[int, FakeResponse] = {}
        self.calls: list[int] = []

    def get(self, url: str, params: dict, timeout: float) -> FakeResponse:
        year = int(params["start_date"][:4])
        self.calls.append(year)
        return self.responses.get(year, FakeResponse(500))


@pytest.fixture
def server(monkeypatch):
    fake = FakeWeatherServer()
    monkeypatch.setattr(requests, "get", fake.get)
    return fake


def test_parse_renames_columns_and_converts_sunshine():
    weather = parse_weather_json(make_json(["2024-06-01", "2024-06-02"], sunshine_seconds=7200))
    assert list(weather.columns) == list(DAILY_VARIABLES.values())
    assert weather.index.name == "date"
    assert weather["sunshine"].tolist() == [2.0, 2.0]


def test_past_year_is_cached(server, tmp_path):
    server.responses[2023] = FakeResponse(200, make_json(["2023-01-01"]))
    download_weather_year(2023, tmp_path)
    assert (tmp_path / "2023.json").exists()
    # Second call reads the cache instead of asking the server again
    download_weather_year(2023, tmp_path)
    assert server.calls == [2023]


def test_current_year_is_not_cached(server, tmp_path):
    year = date.today().year
    server.responses[year] = FakeResponse(200, make_json([f"{year}-01-01"]))
    download_weather_year(year, tmp_path)
    assert not (tmp_path / f"{year}.json").exists()


def test_server_error_raises(server, tmp_path):
    with pytest.raises(requests.HTTPError):
        download_weather_year(2023, tmp_path)


def test_load_weather_spans_years_and_cuts_to_range(server, tmp_path):
    server.responses[2023] = FakeResponse(200, make_json(["2023-12-30", "2023-12-31"]))
    server.responses[2024] = FakeResponse(200, make_json(["2024-01-01", "2024-01-02"]))
    weather = load_weather(date(2023, 12, 31), date(2024, 1, 1), tmp_path)
    assert [d.strftime("%Y-%m-%d") for d in weather.index] == ["2023-12-31", "2024-01-01"]
