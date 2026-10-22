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

import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from radmuenster.data import (
    DEFAULT_SITE,
    KNOWN_FAULTS,
    MIN_INTERVALS_PER_DAY,
    OK_STATUS,
    latest_commit,
    load_site_intervals,
)
from radmuenster.features import SPECIAL_DAYS, calendar_features


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

# Partial outages that no automatic flag catches, checked by hand: the site
# counted about half or less of its usual count compared with the other
# sites, from one day to the next, and recovered as suddenly. Candidates come
# from low_count_days (a monitoring warning) and notebook 03; whether a drop
# is an outage, building works or the weather stays a human decision.
# (site id, first day, last day, reason)
SITE_FAULTS = [
    ("100035541", "2021-01-31", "2021-03-13", "about half the usual count from one day to the next"),
    ("100035541", "2021-10-03", "2021-10-17", "about half the usual count, between days the city flagged"),
    ("100034982", "2024-06-14", "2024-06-27", "about half the usual count from one day to the next"),
    ("300039328", "2024-06-16", "2024-07-30", "less than half the usual count from one day to the next"),
    ("300039331", "2024-07-01", "2024-07-04", "less than half the usual count, before days without counts"),
    ("300039331", "2024-07-21", "2024-07-30", "less than half the usual count from one day to the next"),
    ("300039331", "2026-05-16", "2026-05-30", "about half the usual count from one day to the next"),
    ("300037405", "2026-04-04", "2026-05-01", "a third of the usual count or less, some days nothing"),
    ("300037405", "2026-05-07", "2026-05-22", "nothing or almost nothing for two weeks"),
    ("300037925", "2026-03-30", "2026-04-14", "almost nothing on some days, a third of the usual on others"),
    ("300037925", "2026-05-04", "2026-05-20", "nothing on some days, about half the usual on others"),
    # The Warendorfer Straße: its direction checks (to_daily) need the labelled
    # directions, which this table does not use
    (DEFAULT_SITE, "2024-06-03", "2024-06-11", "inbound counted almost nothing"),
] + [(DEFAULT_SITE, first, last, reason) for first, last, reason in KNOWN_FAULTS]


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
    return mark_almost_zero(mark_site_faults(pd.concat(frames, ignore_index=True))), failed


def mark_site_faults(
    sites_daily: pd.DataFrame, faults: list[tuple[str, str, str, str]] = SITE_FAULTS
) -> pd.DataFrame:
    """sites_daily with the column known_fault (a day in faults), its reason
    in note and ok updated."""
    marked = sites_daily.copy()
    marked.insert(marked.columns.get_loc("note"), "known_fault", False)
    for site_id, first, last, reason in faults:
        days = (
            (marked["site_id"] == site_id)
            & (marked["date"] >= pd.Timestamp(first))
            & (marked["date"] <= pd.Timestamp(last))
        )
        marked.loc[days, "known_fault"] = True
        marked.loc[days, "note"] = reason
    marked["ok"] = marked["ok"] & ~marked["known_fault"]
    return marked


# A day is far below the usual count of its site if its count, relative to
# the usual count of the site, is below LOW_LEVEL times the same relative
# count of the other sites (the median over them): rain, holidays and the
# season lower all sites, a partial outage only one.
LOW_LEVEL = 0.5
# ... on at least this many days in a row. Rainy days hit leisure routes
# harder than the city, but rarely for three days in a row at half the count.
LOW_RUN_DAYS = 3
# The usual count: median of the last BASELINE_DAYS usable days of the same
# weekday, going back at most BASELINE_WEEKS weeks, so that it follows the
# season, and reaches back to before an outage; with fewer than
# MIN_BASELINE_DAYS usable days the day is not judged
BASELINE_DAYS = 4
BASELINE_WEEKS = 12
MIN_BASELINE_DAYS = 3
# Days on which all sites together are below this level are not judged:
# snow, Christmas and holidays affect the sites too differently
MIN_CITY_LEVEL = 0.5
# Far less than that, on a single day, is flagged: the counter (almost) stopped,
# like at the Hammer Straße on 1 August 2026 (29 trips instead of about 7,000)
ALMOST_ZERO_LEVEL = 0.1
# Low days are left out of the usual counts, which changes the levels;
# repeat until nothing changes
MAX_ITERATIONS = 10


def _usual_counts(counts: pd.DataFrame, usable: pd.DataFrame) -> pd.DataFrame:
    """Median of the last BASELINE_DAYS usable days of the same weekday, per
    site and day (see BASELINE_DAYS); NaN with too few."""
    usable_counts = counts.where(usable)
    # weeks x days x sites, the most recent week first
    earlier = np.stack([usable_counts.shift(7 * k).to_numpy() for k in range(1, BASELINE_WEEKS + 1)])
    present = ~np.isnan(earlier)
    recent = np.where(present & (np.cumsum(present, axis=0) <= BASELINE_DAYS), earlier, np.nan)
    n_days = np.minimum(present.sum(axis=0), BASELINE_DAYS)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # A median over nothing is NaN, and a warning
        warnings.simplefilter("ignore", RuntimeWarning)
        usual = np.where(n_days >= MIN_BASELINE_DAYS, np.nanmedian(recent, axis=0), np.nan)
    return pd.DataFrame(usual, index=counts.index, columns=counts.columns)


def _long_runs(below: pd.DataFrame, judged: pd.DataFrame) -> pd.DataFrame:
    """True on days in runs of at least LOW_RUN_DAYS judged days below.
    Days that are not judged neither count nor end a run, unless there are
    more than MAX_GAP_DAYS of them."""
    runs = pd.DataFrame(False, index=below.index, columns=below.columns)
    for site in below.columns:
        days = below.loc[judged[site], site]
        gap = days.index.to_series().diff() > pd.Timedelta(days=MAX_GAP_DAYS + 1)
        run_id = ((days != days.shift()) | gap).cumsum()
        long = days & (days.groupby(run_id).transform("size") >= LOW_RUN_DAYS)
        runs.loc[long.index, site] = long
    return runs


def _wide(sites_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Counts and ok flags with one row per calendar day and one column per
    site, and which days are special days off."""
    counts = sites_daily.pivot(index="date", columns="site_id", values="total")
    counts = counts.reindex(pd.date_range(counts.index.min(), counts.index.max(), freq="D"))
    ok = sites_daily.pivot(index="date", columns="site_id", values="ok").reindex(counts.index)
    ok = ok.fillna(False).astype(bool)
    special = calendar_features(counts.index)[list(SPECIAL_DAYS)].any(axis=1).to_numpy()
    return counts, ok, special


def _adjusted_levels(
    counts: pd.DataFrame, ok: pd.DataFrame, special: np.ndarray, low: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each day's count relative to the usual count of its site, divided by
    the median of the same over the other sites, and which days are judged
    (see LOW_LEVEL). Days in low are left out of the usual counts and the
    medians."""
    usable = ok & ~low
    level = counts / _usual_counts(counts, usable)
    usable_level = level.where(usable)
    # The other sites only, so that a site's own outage does not lower its yardstick
    city = pd.DataFrame({site: usable_level.drop(columns=site).median(axis=1) for site in counts.columns})
    judged = ok & (city >= MIN_CITY_LEVEL) & ~special[:, None] & level.notna()
    return level / city, judged


def _aligned(wide: pd.DataFrame, sites_daily: pd.DataFrame) -> pd.Series:
    """A True/False table (days x sites) as a column aligned with sites_daily."""
    lookup = wide.stack()
    keys = pd.MultiIndex.from_arrays([pd.DatetimeIndex(sites_daily["date"]), sites_daily["site_id"]])
    return pd.Series(lookup.reindex(keys).fillna(False).to_numpy(dtype=bool), index=sites_daily.index)


def low_count_days(sites_daily: pd.DataFrame) -> pd.Series:
    """True for usable days far below the usual count of their site compared
    with the other sites (see LOW_LEVEL), aligned with sites_daily.

    Meant as a warning, not a flag: the rule cannot tell a partial outage
    from building works or a leisure route in bad weather. Real outages go
    into SITE_FAULTS by hand.
    """
    counts, ok, special = _wide(sites_daily)
    low = pd.DataFrame(False, index=counts.index, columns=counts.columns)
    for _ in range(MAX_ITERATIONS):
        adjusted, judged = _adjusted_levels(counts, ok, special, low)
        new = _long_runs(judged & (adjusted < LOW_LEVEL), judged)
        if new.equals(low):
            break
        low = new
    return _aligned(low, sites_daily)


def almost_zero_days(sites_daily: pd.DataFrame) -> pd.Series:
    """True for usable days with less than ALMOST_ZERO_LEVEL of the usual
    count of their site compared with the other sites, aligned with
    sites_daily. Unlike a low count, this has no harmless explanation: not
    even heavy rain takes nine in ten cyclists off one road but not the
    others. So it is a flag, already from a single day."""
    counts, ok, special = _wide(sites_daily)
    adjusted, judged = _adjusted_levels(counts, ok, special, pd.DataFrame(False, index=ok.index, columns=ok.columns))
    return _aligned(judged & (adjusted < ALMOST_ZERO_LEVEL), sites_daily)


def mark_almost_zero(sites_daily: pd.DataFrame) -> pd.DataFrame:
    """sites_daily with the column almost_zero (almost_zero_days) and ok updated."""
    marked = sites_daily.copy()
    marked.insert(marked.columns.get_loc("note"), "almost_zero", almost_zero_days(sites_daily))
    marked["ok"] = marked["ok"] & ~marked["almost_zero"]
    return marked
