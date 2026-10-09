"""Load Münster's bicycle counter data and aggregate it to daily counts.

Source: https://github.com/od-ms/radverkehr-zaehlstellen
License: Datenlizenz Deutschland Namensnennung 2.0 (City of Münster)
"""

from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

RAW_URL = "https://raw.githubusercontent.com/od-ms/radverkehr-zaehlstellen/main/{site}/{month}.csv"

# Warendorfer Straße: data since January 2019
DEFAULT_SITE = "100034983"

# 96 quarter hours per day; on the day daylight saving time starts there are 92.
MIN_INTERVALS_PER_DAY = 92

# Status values that do not indicate a fault. The data also contains "4"
# (e.g. autumn 2019, January 2024), which we treat as a fault.
OK_STATUS = {"0", "validated"}


def month_range(start: date, end: date) -> list[str]:
    """All months from start to end (inclusive) as 'YYYY-MM'."""
    months = pd.period_range(start=start, end=end, freq="M")
    return [str(p) for p in months]


def parse_month_csv(text: str, site: str) -> pd.DataFrame:
    """Convert a monthly CSV into a DataFrame with columns count and status.

    The CSV contains the site's total count (the column starting with the
    site ID), the individual directions and one status column per channel.
    We keep only the total count and its status.
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
    """Fetch a monthly CSV and cache it locally. None if it does not exist."""
    cache_file = cache_dir / site / f"{month}.csv"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    resp = requests.get(RAW_URL.format(site=site, month=month), timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    # The current month still changes daily, so do not cache it.
    if month != pd.Timestamp.today().strftime("%Y-%m"):
        cache_file.write_text(resp.text, encoding="utf-8")
    return resp.text


def load_intervals(
    site: str = DEFAULT_SITE,
    start: date = date(2019, 1, 1),
    end: date | None = None,
    cache_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
    """Load all 15-minute counts of a site for the given period."""
    end = end or date.today()
    frames = []
    for month in month_range(start, end):
        text = download_month(site, month, cache_dir)
        if text is not None:
            frames.append(parse_month_csv(text, site))
    if not frames:
        raise ValueError(f"No data found for site {site}")
    return pd.concat(frames).sort_index()


def to_daily(intervals: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 15-minute counts to daily totals.

    Columns:
      count       total bicycle trips on that day
      n_intervals number of quarter hours present
      n_flagged   number of quarter hours with a fault status
      complete    True if the day is complete and has no fault status
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
