from urllib.error import HTTPError

from radmuenster import published


def test_a_log_that_does_not_exist_yet_reads_as_an_empty_table(monkeypatch, tmp_path):
    def not_on_main(url):
        raise HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(published, "read_forecasts", not_on_main)
    log = published._read_optional_log(tmp_path / "missing.csv")
    assert log.empty
    assert {"forecast_date", "site_id", "model", "prediction"} <= set(log.columns)
