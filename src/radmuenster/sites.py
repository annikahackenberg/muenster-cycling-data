"""All bicycle counting sites of the City of Münster.

Kept by hand, like calendars.py: the first month tells the loader where the
data starts, so that it does not ask for dozens of months that do not exist.
The names are the current ones; some sites were renamed (e.g. "Lütkenbecker
Str." became "Lütkenbecker Weg").

Unlike the Warendorfer Straße, most sites do not label their channels as
inbound and outbound, some relabelled or rebuilt their channels over time,
and at the Bismarckallee the labels IN and OUT swapped. So only the total of
a site is used here; the channels are kept by ID, without interpreting their
labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from radmuenster.data import MIN_INTERVALS_PER_DAY, OK_STATUS, latest_commit, load_site_intervals


@dataclass(frozen=True)
class Site:
    id: str
    name: str
    first_month: str  # "YYYY-MM"


SITES = [
    Site("100034983", "Warendorfer Straße", "2019-01"),
    Site("100034980", "Hammer Straße", "2019-01"),
    Site("100034981", "Weseler Straße", "2019-01"),
    Site("100034982", "Hüfferstraße", "2019-01"),
    Site("100034978", "Gartenstraße", "2019-01"),
    Site("100035541", "Neutor", "2019-01"),
    Site("100031297", "Promenade (nördl. Salzstraße)", "2019-01"),
    Site("100053305", "Kanalpromenade, Abschnitt 5", "2019-07"),
    Site("100031300", "Hafenstraße", "2020-01"),
    Site("100020113", "Wolbecker Straße", "2023-01"),
    Site("300037405", "Promenade (westl. Hals)", "2023-11"),
    Site("300037544", "Kanalpromenade, Abschnitt 1 (Dingstiege)", "2023-11"),
    Site("300037920", "Lütkenbecker Weg", "2023-12"),
    Site("300037925", "Goldstraße", "2023-12"),
    Site("300037926", "Bohlweg", "2023-12"),
    Site("300037928", "Kinderhauser Straße", "2023-12"),
    Site("300037931", "Gasselstiege", "2023-12"),
    Site("300037932", "Schmeddingstraße", "2023-12"),
    Site("300037933", "Weißenburgstraße", "2023-12"),
    Site("300037936", "Kanalpromenade, Abschnitt 6", "2023-12"),
    Site("300039331", "Grevener Straße", "2023-12"),
    Site("300038855", "Bismarckallee", "2024-01"),
    Site("300039328", "Coesfelder Kreuz", "2024-01"),
]

SITES_BY_ID = {site.id: site for site in SITES}


# A channel that carried at least this share of the trips in the 4 weeks
# before ...
MIN_CHANNEL_SHARE = 0.10
# ... and counts nothing for whole days, while the site's total in the
# following 2 weeks is at least this much lower, is marked as dead. A
# rebuild of the channels moves the trips to other channels and keeps the
# total; a dead channel loses them. Public holidays lower the total too, but
# no channel drops to exactly 0.
MIN_TOTAL_DROP = 0.25
BEFORE_DAYS = 28
AFTER_DAYS = 14
# Zero days at most this far apart belong to the same outage
MAX_GAP_DAYS = 3


def _dead_channel_days(daily: pd.DataFrame, channels: list[str]) -> pd.Series:
    """For each day, the dead channels (see MIN_TOTAL_DROP) with their share
    of the trips before the outage, as text; empty if none."""
    notes = pd.Series("", index=daily.index)
    full_day = ~daily["missing"] & ~daily["partial"]
    for channel in channels:
        zero = (daily[channel].fillna(0) == 0) & (daily["total"] > 0) & full_day
        days = daily.index[zero].to_series()
        if days.empty:
            continue
        # Number the outages: a new one starts after a gap of more than MAX_GAP_DAYS
        outage_id = (days.diff() > pd.Timedelta(days=MAX_GAP_DAYS)).cumsum()
        for _, outage in days.groupby(outage_id.to_numpy()):
            start = outage.iloc[0]
            before = daily.loc[start - pd.Timedelta(days=BEFORE_DAYS) : start - pd.Timedelta(days=1)]
            after = daily.loc[start : start + pd.Timedelta(days=AFTER_DAYS - 1)]
            if before["total"].sum() == 0:
                continue
            share = before[channel].sum() / before["total"].sum()
            drop = 1 - after["total"].mean() / before["total"].mean()
            if share >= MIN_CHANNEL_SHARE and drop >= MIN_TOTAL_DROP:
                note = f"channel {channel} counted nothing ({share:.0%} of the trips before)"
                notes[outage.index] = np.where(
                    notes[outage.index] == "", note, notes[outage.index] + "; " + note
                )
    return notes


def to_site_daily(intervals: pd.DataFrame) -> pd.DataFrame:
    """Daily totals of a site from load_site_intervals, with flags.

    The counts are not changed: suspicious days are only marked, so that a
    page can show the raw data and summaries can leave them out.

    One row per calendar day from the first to the last day with data.
    Columns:
      total         trips that day (NaN if the city delivered nothing)
      n_intervals   quarter hours present
      missing       no data at all for the day
      partial       fewer intervals than a full day (92 of 96 quarter hours, or
                    23 of 24 hours where the site delivers hours), e.g. the
                    last day of a month, which the city cuts short
      status_fault  the city flagged at least one quarter hour
      zero          a full day with a total of 0: the site was off
      dead_channel  a channel stopped counting and the total dropped
                    (see MIN_TOTAL_DROP)
      note          the dead channels, as text
      ok            none of the flags
    """
    channels = [c for c in intervals.columns if c not in ("total", "status")]
    day = intervals.index.normalize()
    grouped = intervals.groupby(day)
    daily = grouped[["total", *channels]].sum(min_count=1)
    daily["n_intervals"] = grouped["total"].count()
    # Most sites deliver quarter hours, but some switched to hours for a while
    # (Schmeddingstraße from March 2025 to June 2026). The median step between
    # two timestamps of a day tells which.
    step = intervals.index.to_series().diff().groupby(day).median()
    daily["expected"] = pd.Timedelta(days=1) / step.fillna(pd.Timedelta(minutes=15))
    flagged = ~intervals["status"].isin(OK_STATUS)
    daily["n_flagged"] = flagged.groupby(intervals.index.normalize()).sum()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="date"))
    daily["n_intervals"] = daily["n_intervals"].fillna(0).astype(int)

    daily["missing"] = daily["n_intervals"] == 0
    # 92 of 96 quarter hours (the day daylight saving time starts), 23 of 24 hours
    min_intervals = daily["expected"] * MIN_INTERVALS_PER_DAY / 96
    daily["partial"] = ~daily["missing"] & (daily["n_intervals"] < min_intervals)
    daily["status_fault"] = daily["n_flagged"].fillna(0) > 0
    daily["zero"] = ~daily["missing"] & ~daily["partial"] & (daily["total"].fillna(0) == 0)
    daily["note"] = _dead_channel_days(daily, channels)
    daily["dead_channel"] = daily["note"] != ""
    flags = ["missing", "partial", "status_fault", "zero", "dead_channel"]
    daily["ok"] = ~daily[flags].any(axis=1)
    return daily[["total", "n_intervals", *flags, "note", "ok"]]


def load_all_sites(
    end: date | None = None, cache_dir: Path = Path("data/raw")
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Daily totals with flags (to_site_daily) of all SITES, in one table.

    Returns the table and the sites that could not be loaded, with the error.
    One broken site (e.g. a new file format) must not stop the others or the
    forecast; the daily job reports it as a monitoring warning instead.

    Columns: site_id, site_name, date and the columns of to_site_daily.
    """
    frames, failed = [], {}
    # One commit for all sites: one API request, and all files from the same day
    ref = latest_commit()
    for site in SITES:
        year, month = map(int, site.first_month.split("-"))
        try:
            intervals = load_site_intervals(
                site.id, start=date(year, month, 1), end=end, cache_dir=cache_dir, ref=ref
            )
            daily = to_site_daily(intervals).reset_index()
        # Deliberately broad: whatever breaks one site is reported, not raised
        except Exception as error:
            failed[site.name] = f"{type(error).__name__}: {error}"
            continue
        daily.insert(0, "site_id", site.id)
        daily.insert(1, "site_name", site.name)
        frames.append(daily)
    return pd.concat(frames, ignore_index=True), failed
