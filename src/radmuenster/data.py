"""Load Münster's bicycle counter data and aggregate it to daily counts.

Source: https://github.com/od-ms/radverkehr-zaehlstellen
License: Datenlizenz Deutschland Namensnennung 2.0 (City of Münster)
"""

from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
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

# Normally 47-64 % of the trips go into the city. A direction with less than
# 25 % of the day's trips points to a (partly) dead channel, even with status 0.
MIN_DIRECTION_SHARE = 0.25

# On very quiet days (e.g. snow in February 2021 with 63 trips) the share
# varies by chance, so the share check only applies above this many trips.
MIN_COUNT_FOR_SHARE_CHECK = 500

# Faults of DEFAULT_SITE that the checks above miss, found by looking at the
# data: (first day, last day, reason), both days inclusive. The city never
# corrects old months, so this list only grows.
KNOWN_FAULTS = [
    (
        "2026-05-05",
        "2026-06-02",
        "inbound counted 30-50 % less than a year earlier while outbound was normal; "
        "the inbound share fell gradually from 0.45 to 0.35 and recovered overnight on 3 June",
    ),
    ("2026-07-13", "2026-07-13", "first day of the inbound outage of 14-18 July, inbound share 0.40"),
    ("2026-07-19", "2026-07-19", "last day of the inbound outage of 14-18 July, inbound share 0.38"),
]

# Days after the end of a month until its file counts as final (see is_final)
CACHE_AFTER_DAYS = 3


def month_range(start: date, end: date) -> list[str]:
    """All months from start to end (inclusive) as 'YYYY-MM'."""
    months = pd.period_range(start=start, end=end, freq="M")
    return [str(p) for p in months]


def parse_month_csv(text: str, site: str) -> pd.DataFrame:
    """Convert a monthly CSV into a DataFrame with columns count, count_in,
    count_out and status.

    The CSV contains the site's total count (the column starting with the
    site ID), the individual directions and one status column per channel.
    We keep the total count, both directions and the status of the total.
    The directions reveal outages the status misses, e.g. autumn 2023, when
    the inbound channel was empty for about 80 days with status 0.
    """
    df = pd.read_csv(StringIO(text), parse_dates=["Datetime"])
    total_col = next(c for c in df.columns if c.startswith(f"{site} ("))
    in_col = next(c for c in df.columns if c.endswith("stadteinwärts)"))
    out_col = next(c for c in df.columns if c.endswith("stadtauswärts)"))
    status_col = f"{site}-status"
    out = pd.DataFrame(
        {
            "count": pd.to_numeric(df[total_col], errors="coerce"),
            "count_in": pd.to_numeric(df[in_col], errors="coerce"),
            "count_out": pd.to_numeric(df[out_col], errors="coerce"),
            "status": df[status_col].astype(str),
        }
    )
    out.index = pd.DatetimeIndex(df["Datetime"], name="datetime")
    return out


def parse_month_channels(text: str, site: str) -> pd.DataFrame:
    """Convert a monthly CSV of any site into total, status and channels.

    Columns: total (the site's count, always the sum of its channels),
    status (of the total, as string) and one column per channel, named by
    the channel's ID (e.g. "353409612"). The channel labels are left out:
    they differ from site to site and changed over time (see sites.py).
    """
    df = pd.read_csv(StringIO(text), parse_dates=["Datetime"])
    total_col = next(c for c in df.columns if c.startswith(f"{site} ("))
    channel_cols = [
        c for c in df.columns if c not in ("Datetime", total_col) and not c.endswith("-status")
    ]
    out = pd.DataFrame(
        {
            "total": pd.to_numeric(df[total_col], errors="coerce"),
            "status": df[f"{site}-status"].astype(str),
        }
    )
    for col in channel_cols:
        # "353409612 (Promenade (westl. Hals) Fahrräder IN)" -> "353409612"
        out[col.split(" ", 1)[0]] = pd.to_numeric(df[col], errors="coerce")
    out.index = pd.DatetimeIndex(df["Datetime"], name="datetime")
    return out


def is_final(month: str, today: date) -> bool:
    """True if the city will not change the file of month any more.

    The city updates around 5 am with the previous day, and a failed update
    is delivered a day later (31 August 2026 arrived on 2 September), so a
    file is final from the 3rd of the next month on. Older files never change.
    """
    month_end = pd.Period(month, freq="M").end_time.date()
    return (today - month_end).days >= CACHE_AFTER_DAYS


def download_month(site: str, month: str, cache_dir: Path) -> str | None:
    """Fetch a monthly CSV and cache it locally. None if it does not exist."""
    cache_file = cache_dir / site / f"{month}.csv"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    resp = requests.get(RAW_URL.format(site=site, month=month), timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    # Only cache final files, see is_final
    if is_final(month, date.today()):
        cache_file.parent.mkdir(parents=True, exist_ok=True)
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


def load_site_intervals(
    site: str,
    start: date,
    end: date | None = None,
    cache_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
    """15-minute total, status and channels of any site (parse_month_channels).

    Channels that only exist in some months are NaN in the others.
    """
    end = end or date.today()
    frames = []
    for month in month_range(start, end):
        text = download_month(site, month, cache_dir)
        if text is not None:
            frames.append(parse_month_channels(text, site))
    if not frames:
        raise ValueError(f"No data found for site {site}")
    return pd.concat(frames).sort_index()


def to_daily(
    intervals: pd.DataFrame, known_faults: list[tuple[str, str, str]] = KNOWN_FAULTS
) -> pd.DataFrame:
    """Aggregate 15-minute counts to daily totals.

    Columns:
      count         total bicycle trips on that day
      count_in      trips into the city
      count_out     trips out of the city
      n_intervals   number of quarter hours present
      n_flagged     number of quarter hours with a fault status
      direction_fault  True if one direction counted nothing all day while the
                       other did, or if one direction has less than
                       MIN_DIRECTION_SHARE of the day's trips (only on days with
                       at least MIN_COUNT_FOR_SHARE_CHECK trips); both mean a
                       (partly) dead channel
      known_fault      True if the day lies in one of known_faults
      complete         True if the day is complete, has no fault status, no
                       direction fault and no known fault
    """
    grouped = intervals.groupby(intervals.index.normalize())
    daily = pd.DataFrame(
        {
            "count": grouped["count"].sum(min_count=1),
            "count_in": grouped["count_in"].sum(min_count=1),
            "count_out": grouped["count_out"].sum(min_count=1),
            "n_intervals": grouped["count"].count(),
            "n_flagged": grouped["status"].apply(lambda s: int((~s.isin(OK_STATUS)).sum())),
        }
    )
    daily.index.name = "date"
    # A dead channel shows up either as zeros or as empty cells (NaN after
    # parsing, e.g. autumn 2023), so treat NaN like 0 here.
    count_in = daily["count_in"].fillna(0)
    count_out = daily["count_out"].fillna(0)
    both = count_in + count_out
    weaker = np.minimum(count_in, count_out)
    # A whole day without a single trip in one direction is implausible even on
    # the quietest day (snow, February 2021: 42 trips, both directions > 0)
    one_dead = (weaker == 0) & (both > 0)
    one_weak = (both >= MIN_COUNT_FOR_SHARE_CHECK) & (weaker / both < MIN_DIRECTION_SHARE)
    daily["direction_fault"] = one_dead | one_weak
    daily["known_fault"] = False
    for first, last, _ in known_faults:
        daily.loc[first:last, "known_fault"] = True
    daily["complete"] = (
        (daily["n_intervals"] >= MIN_INTERVALS_PER_DAY)
        & (daily["n_flagged"] == 0)
        & ~daily["direction_fault"]
        & ~daily["known_fault"]
    )
    return daily
