"""Laden der Münsteraner Radzähldaten und Aggregation zu Tageswerten.

Quelle: https://github.com/od-ms/radverkehr-zaehlstellen
Lizenz: Datenlizenz Deutschland Namensnennung 2.0 (Stadt Münster)
"""

from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

RAW_URL = "https://raw.githubusercontent.com/od-ms/radverkehr-zaehlstellen/main/{site}/{month}.csv"

# Warendorfer Straße: Daten seit Januar 2019
DEFAULT_SITE = "100034983"

# 96 Viertelstunden pro Tag; am Tag der Umstellung auf Sommerzeit sind es 92.
MIN_INTERVALS_PER_DAY = 92

# Statuswerte, die keine Störung bedeuten. In den Daten kommen außerdem "4"
# (z. B. Herbst 2019, Januar 2024) vor, das werten wir als Störung.
OK_STATUS = {"0", "validated"}


def month_range(start: date, end: date) -> list[str]:
    """Alle Monate von start bis end (inklusive) im Format 'YYYY-MM'."""
    months = pd.period_range(start=start, end=end, freq="M")
    return [str(p) for p in months]


def parse_month_csv(text: str, site: str) -> pd.DataFrame:
    """Eine Monats-CSV in ein DataFrame mit Spalten count und status umwandeln.

    Die CSV enthält die Gesamtzählung der Zählstelle (Spalte, die mit der
    Zählstellen-ID beginnt), Einzelrichtungen und je Kanal eine Statusspalte.
    Wir behalten nur die Gesamtzählung und ihren Status.
    """
    df = pd.read_csv(StringIO(text), parse_dates=["Datetime"])
    total_col = next(c for c in df.columns if c.startswith(f"{site} ("))
    status_col = f"{site}-status"
    out = pd.DataFrame(
        {
            "count": pd.to_numeric(df[total_col], errors="coerce"),
            "status": df[status_col].astype(str),
        }
    )
    out.index = pd.DatetimeIndex(df["Datetime"], name="datetime")
    return out


def download_month(site: str, month: str, cache_dir: Path) -> str | None:
    """Monats-CSV holen und lokal zwischenspeichern. None, wenn es sie nicht gibt."""
    cache_file = cache_dir / site / f"{month}.csv"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    resp = requests.get(RAW_URL.format(site=site, month=month), timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    # Der laufende Monat ändert sich noch täglich, also nicht cachen.
    if month != pd.Timestamp.today().strftime("%Y-%m"):
        cache_file.write_text(resp.text, encoding="utf-8")
    return resp.text


def load_intervals(
    site: str = DEFAULT_SITE,
    start: date = date(2019, 1, 1),
    end: date | None = None,
    cache_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
    """Alle 15-Minuten-Werte einer Zählstelle im Zeitraum laden."""
    end = end or date.today()
    frames = []
    for month in month_range(start, end):
        text = download_month(site, month, cache_dir)
        if text is not None:
            frames.append(parse_month_csv(text, site))
    if not frames:
        raise ValueError(f"Keine Daten für Zählstelle {site} gefunden")
    return pd.concat(frames).sort_index()


def to_daily(intervals: pd.DataFrame) -> pd.DataFrame:
    """15-Minuten-Werte zu Tagessummen aggregieren.

    Spalten:
      count       Summe der Radfahrten am Tag
      n_intervals Anzahl vorhandener Viertelstunden
      n_flagged   Anzahl Viertelstunden mit Störungsstatus
      complete    True, wenn der Tag vollständig und ohne Störungsmeldung ist
    """
    grouped = intervals.groupby(intervals.index.normalize())
    daily = pd.DataFrame(
        {
            "count": grouped["count"].sum(min_count=1),
            "n_intervals": grouped["count"].count(),
            "n_flagged": grouped["status"].apply(lambda s: int((~s.isin(OK_STATUS)).sum())),
        }
    )
    daily.index.name = "date"
    daily["complete"] = (daily["n_intervals"] >= MIN_INTERVALS_PER_DAY) & (
        daily["n_flagged"] == 0
    )
    return daily
