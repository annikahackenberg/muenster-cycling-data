"""Data preparation and charts for the "Explore" page of the Streamlit app.

Works on the daily counts of all sites (load_all_sites, published as
data/sites_daily.parquet). The counts are shown as they are; days with a flag
are marked, and summaries say how many days they are based on.
"""

from __future__ import annotations

from datetime import date

import altair as alt
import pandas as pd

# Categorical palette (8 hues, fixed order), validated for light and dark mode
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
MAX_SITES = len(PALETTE["light"])

# Pandas period of each resolution; "W" weeks run from Monday to Sunday
RESOLUTIONS = {"Day": "D", "Week": "W", "Month": "M"}

# Flags in the order they explain a day best: a missing day is not also partial
FLAG_REASONS = {
    "missing": "no data",
    "known_fault": "partial outage",
    "zero": "site counted nothing",
    "dead_channel": "channel outage",
    "partial": "incomplete day",
    "status_fault": "flagged by the city",
}

# A site needs complete data on at least this share of the days of the period
# to be ranked without a warning
MIN_COVERAGE = 0.8


def assign_colors(selected: list[str], previous: dict[str, int]) -> dict[str, int]:
    """Palette slot of each selected site.

    Sites keep their slot as long as they stay selected; a new site gets the
    first free slot. So removing one site does not repaint the others.
    """
    slots = {site: slot for site, slot in previous.items() if site in selected}
    for site in selected:
        if site not in slots:
            slots[site] = min(set(range(MAX_SITES)) - set(slots.values()))
    return slots


def day_reasons(sites_daily: pd.DataFrame) -> pd.Series:
    """Why a day is flagged, as text ("" for days without flags).

    A channel outage shows its note (which channel stopped counting), a
    partial outage the reason from SITE_FAULTS. Data published before a flag
    existed lacks its column; such flags are skipped.
    """
    reason = pd.Series("", index=sites_daily.index)
    for flag, text in reversed(FLAG_REASONS.items()):
        if flag in sites_daily:
            reason = reason.mask(sites_daily[flag], text)
    outage = sites_daily["dead_channel"] & ~sites_daily["missing"]
    reason = reason.mask(outage, "channel outage: " + sites_daily["note"])
    if "known_fault" in sites_daily:
        fault = sites_daily["known_fault"] & ~sites_daily["missing"]
        reason = reason.mask(fault, "partial outage: " + sites_daily["note"])
    return reason


def is_routine(rows: pd.DataFrame) -> pd.Series:
    """Flagged days that are routine rather than a problem: the last day of a
    month, when it is only incomplete or missing. The city cuts it short every
    month (and left it out completely from 2021 to 2025)."""
    only_short = (rows["partial"] | rows["missing"]) & ~rows[["zero", "dead_channel", "status_fault"]].any(axis=1)
    return rows["date"].dt.is_month_end & only_short


def select(sites_daily: pd.DataFrame, sites: list[str], start: date, end: date) -> pd.DataFrame:
    """Rows of the given site names between start and end (inclusive)."""
    in_range = sites_daily["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    return sites_daily[in_range & sites_daily["site_name"].isin(sites)]


def aggregate(rows: pd.DataFrame, resolution: str, leave_out_flagged: bool) -> pd.DataFrame:
    """Trips per day for each site and period of the resolution ("Day",
    "Week" or "Month").

    For weeks and months, the value is the mean over the days with data, so
    that periods with missing days or different lengths stay comparable.
    With leave_out_flagged, flagged days are left out of that mean (for days:
    shown as gaps).

    flagged_days counts the flagged days except the routine ones (is_routine),
    so that a chart can mark the periods with a real problem; reasons lists
    all of them.

    Columns: site_name, period, trips (per day), days, flagged_days, reasons.
    """
    rows = rows.assign(
        value=rows["total"].where(rows["ok"]) if leave_out_flagged else rows["total"],
        flagged=~rows["ok"] & ~is_routine(rows),
        reason=day_reasons(rows),
    )
    # First day of the period each date falls into (for "Day" the date itself)
    period = rows["date"].dt.to_period(RESOLUTIONS[resolution]).dt.start_time
    grouped = rows.groupby(["site_name", period.rename("period")])
    table = grouped.agg(
        trips=("value", "mean"),
        days=("date", "size"),
        flagged_days=("flagged", "sum"),
        reasons=("reason", lambda r: "; ".join(sorted(set(r) - {""}))),
    )
    return table.reset_index()


def ranking(sites_daily: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Mean trips per day of every site between start and end, from days
    without flags only.

    coverage is the share of the period's days that were used. A site that
    started later or had a long outage has a low coverage; its mean is less
    reliable (and from a different part of the year).

    Columns: site_name, trips, ok_days, coverage, reliable; sorted by trips.
    """
    days_in_period = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    rows = sites_daily[sites_daily["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    ok = rows[rows["ok"]]
    table = ok.groupby("site_name").agg(trips=("total", "mean"), ok_days=("total", "size"))
    table = table.reindex(sorted(sites_daily["site_name"].unique()))
    table["ok_days"] = table["ok_days"].fillna(0).astype(int)
    table["coverage"] = table["ok_days"] / days_in_period
    table["reliable"] = table["coverage"] >= MIN_COVERAGE
    return table.sort_values("trips", ascending=False).reset_index()


def flagged_periods(rows: pd.DataFrame) -> pd.DataFrame:
    """Flagged periods per site: consecutive flagged days with the same reason.

    Routine days (is_routine) are left out: they would cut every longer
    outage into monthly pieces. A single day in between does not end a period.

    Columns: site_name, first, last, days, reason.
    """
    flagged = rows[~rows["ok"] & ~is_routine(rows)]
    flagged = flagged.assign(reason=day_reasons(flagged)).sort_values(["site_name", "date"])
    # A new period starts with a new site, a new reason or a gap of more than one day
    new = (
        (flagged["site_name"] != flagged["site_name"].shift())
        | (flagged["reason"] != flagged["reason"].shift())
        | (flagged["date"].diff() > pd.Timedelta(days=2))
    )
    periods = flagged.groupby(new.cumsum().to_numpy()).agg(
        site_name=("site_name", "first"),
        first=("date", "min"),
        last=("date", "max"),
        days=("date", "size"),
        reason=("reason", "first"),
    )
    return periods.reset_index(drop=True)


def sites_chart(
    data: pd.DataFrame, colors: dict[str, int], theme: str, resolution: str
) -> alt.LayerChart:
    """Trips per day over time, one line per site; periods with flagged days
    are marked with a hollow ring and explained in the tooltip."""
    palette = PALETTE[theme]
    domain = list(colors)
    color = alt.Color(
        "site_name:N",
        scale=alt.Scale(domain=domain, range=[palette[colors[s]] for s in domain]),
        legend=alt.Legend(orient="top", title=None, columns=4, labelLimit=220, symbolType="stroke"),
    )
    time_format = {"Day": "%a %d %b %Y", "Week": "week of %d %b %Y", "Month": "%B %Y"}[resolution]
    tooltip = [
        alt.Tooltip("site_name:N", title="Site"),
        alt.Tooltip("period:T", title="Date", format=time_format),
        alt.Tooltip("trips:Q", title="Trips per day", format=",.0f"),
        alt.Tooltip("flagged_days:Q", title="Flagged days"),
        alt.Tooltip("reasons:N", title="Why"),
    ]
    base = alt.Chart(data).encode(
        x=alt.X("period:T", title=None, axis=alt.Axis(grid=False, labelSeparation=8)),
        y=alt.Y("trips:Q", title="Trips per day", scale=alt.Scale(zero=True)),
        color=color,
    )
    lines = base.mark_line(strokeWidth=2, invalid="break-paths-show-domains")
    # Invisible points as a wide hover target for the tooltip
    hover = base.mark_point(size=120, opacity=0).encode(tooltip=tooltip)
    flagged = (
        base.transform_filter(alt.datum.flagged_days > 0)
        .mark_point(size=60, filled=False, strokeWidth=1.5)
        .encode(tooltip=tooltip)
    )
    return alt.layer(lines, flagged, hover).properties(height=380)


def ranking_chart(table: pd.DataFrame, theme: str) -> alt.Chart:
    """Horizontal bars of the mean trips per day; sites with low coverage in
    grey and labelled, so that the difference does not rely on color."""
    color = PALETTE[theme][0]
    table = table.dropna(subset=["trips"]).assign(
        label=lambda t: t["site_name"] + t["reliable"].map({True: "", False: " (few days)"})
    )
    return (
        alt.Chart(table)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("trips:Q", title="Trips per day"),
            color=alt.condition(alt.datum.reliable, alt.value(color), alt.value("#9a9a96")),
            tooltip=[
                alt.Tooltip("site_name:N", title="Site"),
                alt.Tooltip("trips:Q", title="Trips per day", format=",.0f"),
                alt.Tooltip("ok_days:Q", title="Days used"),
                alt.Tooltip("coverage:Q", title="Share of the period", format=".0%"),
            ],
        )
        .properties(height=alt.Step(22))
    )
