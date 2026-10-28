import json
from urllib.error import HTTPError, URLError

from radmuenster import published


def test_a_log_that_does_not_exist_yet_reads_as_an_empty_table(monkeypatch, tmp_path):
    def not_on_main(url):
        raise HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(published, "read_forecasts", not_on_main)
    log = published._read_optional_log(tmp_path / "missing.csv")
    assert log.empty
    assert {"forecast_date", "site_id", "model", "prediction"} <= set(log.columns)


def test_an_old_status_gets_empty_lists_for_new_keys(monkeypatch, tmp_path):
    def offline(url, timeout):
        raise URLError("offline")

    path = tmp_path / "status.json"
    path.write_text(json.dumps({"checked_on": "2026-10-27", "warnings": ["A warning."]}))
    monkeypatch.setattr(published, "urlopen", offline)
    monkeypatch.setattr(published, "STATUS_FILE", path)
    status = published.read_status()
    assert status["warnings"] == ["A warning."]
    assert status["actions"] == [] and status["checked_warnings"] == []
