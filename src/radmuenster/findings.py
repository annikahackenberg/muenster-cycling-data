"""Key findings of the exploratory analysis, recomputed from the current data.

The same methods as in notebooks/01_exploratory_analysis.ipynb, as functions
that return small tables, plus one Altair chart per finding. key_findings
puts them together with their texts, for the static page and the Streamlit
app alike. Only complete days are used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd

from radmuenster.dashboard import SERIES_COLORS
from radmuenster.features import PERIOD_STARTS, SPECIAL_DAYS, calendar_features

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# A month counts only with at least this many complete days (as in notebook 01)
MIN_DAYS_PER_MONTH = 15

# Typical day: median of the same weekday over +-4 weeks, i.e. 9 values (as in
# the outlier analysis of notebook 01)
TYPICAL_WINDOW = 9

RAIN_BINS = [-0.1, 0, 3, 7, 24]
RAIN_LABELS = ["Dry", "1–3 h", "4–7 h", "8+ h"]
WORKDAY = "Workday"
WEEKEND = "Weekend"

# Outages of at least this many faulty days are marked in the share chart
MIN_OUTAGE_DAYS = 14
MAX_GAP_DAYS = 3

CHART_HEIGHT = 220


def _complete(daily: pd.DataFrame) -> pd.DataFrame:
    return daily[daily["complete"]]


def weekday_profile(daily: pd.DataFrame) -> pd.DataFrame:
    """Median count per weekday and its ratio to the average workday.

    Columns: weekday (Mon ... Sun), median, factor (1.0 = average of the
    medians of Monday to Friday).
    """
    complete = _complete(daily)
    median = complete.groupby(complete.index.dayofweek)["count"].median()
    return pd.DataFrame(
        {"weekday": WEEKDAYS, "median": median.to_numpy(), "factor": (median / median.iloc[:5].mean()).to_numpy()}
    )


def month_profile(daily: pd.DataFrame) -> pd.DataFrame:
    """Typical annual curve: each month's mean divided by the mean of its
    year, median over all full years.

    Dividing by the annual mean removes the level of the year (Corona, the
    measurement jumps), so that only the shape remains. Only years with all
    twelve months, each with at least MIN_DAYS_PER_MONTH complete days.

    Columns: month (Jan ... Dec), factor; attribute years: the years used.
    """
    complete = _complete(daily)
    grouped = complete["count"].groupby([complete.index.year, complete.index.month])
    monthly = grouped.mean().where(grouped.size() >= MIN_DAYS_PER_MONTH).unstack()
    full_years = monthly.dropna()
    shape = full_years.div(full_years.mean(axis=1), axis=0)
    profile = pd.DataFrame({"month": MONTHS, "factor": shape.median().to_numpy()})
    profile.attrs["years"] = full_years.index.tolist()
    return profile


def ratio_to_typical(daily: pd.DataFrame) -> pd.Series:
    """Each regular day's count divided by the typical count of that weekday
    at that time of year (median of the same weekday over +-4 weeks).

    Special days off (public holidays, bridge days, Christmas, Rosenmontag)
    are left out: they are no typical day and no regular day.
    """
    counts = daily["count"].where(daily["complete"])
    counts = counts.reindex(pd.date_range(counts.index.min(), counts.index.max(), freq="D"))
    special = calendar_features(counts.index)[list(SPECIAL_DAYS)].any(axis=1)
    regular = counts.where(~special)
    # Within one weekday, consecutive rows are exactly 7 days apart
    typical = regular.groupby(regular.index.dayofweek).transform(
        lambda s: s.rolling(TYPICAL_WINDOW, center=True, min_periods=5).median()
    )
    return (regular / typical).dropna().rename("ratio")


def rain_effect(daily: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Median ratio to the typical day by hours of rain, for workdays and
    weekends separately.

    Comparing with the typical day of the same weekday and time of year takes
    the weekly pattern and the season out, which would otherwise hide the
    weather (rainy months are also dark months).

    Columns: day_type, rain, factor, days.
    """
    table = ratio_to_typical(daily).to_frame().join(weather["precipitation_hours"], how="inner")
    table["rain"] = pd.cut(table["precipitation_hours"], RAIN_BINS, labels=RAIN_LABELS)
    table["day_type"] = np.where(table.index.dayofweek >= 5, WEEKEND, WORKDAY)
    grouped = table.groupby(["day_type", "rain"], observed=True)["ratio"]
    effect = grouped.agg(factor="median", days="size").reset_index()
    effect["rain"] = effect["rain"].astype(str)
    return effect


def inbound_share(daily: pd.DataFrame) -> pd.DataFrame:
    """Share of trips into the city, as a 28-day median of complete days.

    Columns: date, share. Days without enough complete days around them are
    NaN, so outages show up as gaps.
    """
    complete = _complete(daily)
    share = (complete["count_in"] / complete["count"]).reindex(
        pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    )
    rolling = share.rolling(28, center=True, min_periods=14).median()
    return pd.DataFrame({"date": rolling.index, "share": rolling.to_numpy()})


def unflagged_outages(daily: pd.DataFrame) -> pd.DataFrame:
    """Outages of at least MIN_OUTAGE_DAYS days with a direction fault that
    the city's status did not flag.

    Gaps of up to MAX_GAP_DAYS do not end an outage: from 2021 to 2025 the
    last day of every month is missing, which would cut an outage into
    monthly pieces.

    Columns: start, end, days (number of faulty days).
    """
    faulty = daily.index[daily["direction_fault"] & (daily["n_flagged"] == 0)].to_series()
    # A new outage starts where the distance to the previous faulty day is
    # larger than the allowed gap; cumsum numbers the outages
    outage_id = (faulty.diff() > pd.Timedelta(days=MAX_GAP_DAYS)).cumsum()
    runs = faulty.groupby(outage_id.to_numpy()).agg(["min", "max", "size"])
    runs.columns = ["start", "end", "days"]
    return runs[runs["days"] >= MIN_OUTAGE_DAYS].reset_index(drop=True)


def _bar_chart(data: pd.DataFrame, x: str, order: list[str], theme: str, title: str) -> alt.LayerChart:
    """Bars of factor per category, with a rule at 1.0 (one series, no legend)."""
    color = SERIES_COLORS[theme][next(iter(SERIES_COLORS[theme]))]
    bars = (
        alt.Chart(data)
        .mark_bar(color=color, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{x}:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("factor:Q", title=title, scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip("factor:Q", title="Factor", format=".2f")],
        )
    )
    one = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color="gray", strokeDash=[4, 3]).encode(y="y:Q")
    return alt.layer(bars, one).properties(height=CHART_HEIGHT)


def weekday_chart(profile: pd.DataFrame, theme: str) -> alt.LayerChart:
    return _bar_chart(profile, "weekday", WEEKDAYS, theme, "Relative to an average workday")


def month_chart(profile: pd.DataFrame, theme: str) -> alt.LayerChart:
    return _bar_chart(profile, "month", MONTHS, theme, "Relative to the annual mean")


def rain_chart(effect: pd.DataFrame, theme: str) -> alt.LayerChart:
    """Grouped bars: hours of rain on the x axis, workday and weekend side by side."""
    colors = list(SERIES_COLORS[theme].values())
    bars = (
        alt.Chart(effect)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("rain:N", sort=RAIN_LABELS, title="Hours of rain", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("day_type:N", sort=[WORKDAY, WEEKEND]),
            y=alt.Y("factor:Q", title="Relative to a typical day", scale=alt.Scale(zero=True)),
            color=alt.Color(
                "day_type:N",
                scale=alt.Scale(domain=[WORKDAY, WEEKEND], range=colors),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("day_type:N", title="Day"),
                alt.Tooltip("rain:N", title="Rain"),
                alt.Tooltip("factor:Q", title="Factor", format=".2f"),
                alt.Tooltip("days:Q", title="Days"),
            ],
        )
    )
    one = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color="gray", strokeDash=[4, 3]).encode(y="y:Q")
    return alt.layer(bars, one).properties(height=CHART_HEIGHT)


def share_chart(share: pd.DataFrame, outages: pd.DataFrame, theme: str) -> alt.LayerChart:
    """Inbound share over time, with unflagged outages as shaded bands and
    the starts of the measurement periods as labelled rules."""
    color = SERIES_COLORS[theme][next(iter(SERIES_COLORS[theme]))]
    line = (
        alt.Chart(share)
        .mark_line(color=color, strokeWidth=2, invalid="break-paths-show-domains")
        .encode(
            # Grid lines at each 1 January separate the years
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", grid=True, tickCount="year")),
            y=alt.Y("share:Q", title="Share of trips into the city", scale=alt.Scale(domain=[0.3, 0.7])),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                alt.Tooltip("share:Q", title="Share (28-day median)", format=".2f"),
            ],
        )
    )
    bands = (
        alt.Chart(outages.assign(label="Unflagged outage"))
        .mark_rect(color="gray", opacity=0.25)
        .encode(
            x="start:T",
            x2="end:T",
            tooltip=[
                alt.Tooltip("label:N", title=""),
                alt.Tooltip("start:T", title="From", format="%d %b %Y"),
                alt.Tooltip("end:T", title="To", format="%d %b %Y"),
                alt.Tooltip("days:Q", title="Days"),
            ],
        )
    )
    # Short labels left of the rules, so that the last one fits on a phone
    periods = pd.DataFrame({"date": list(PERIOD_STARTS), "label": [f"{d:%b %y}" for d in PERIOD_STARTS]})
    rules = alt.Chart(periods).mark_rule(color="gray", strokeDash=[4, 3]).encode(x="date:T")
    labels = (
        alt.Chart(periods)
        .mark_text(align="right", dx=-4, dy=4, baseline="top", color="gray")
        .encode(x="date:T", y=alt.value(0), text="label:N")
    )
    return alt.layer(bands, line, rules, labels).properties(height=CHART_HEIGHT)


@dataclass(frozen=True)
class Finding:
    """One key finding: a title, a chart and a text about it.

    name         short id, e.g. for the chart element on the static page
    text         plain text; its numbers come from the current data
    description  what the chart shows, for screen readers
    chart        the chart for a theme ("light" or "dark")
    width        share of the page width the chart needs on a wide screen:
                 a few bars look lost across the full width
    """

    name: str
    title: str
    text: str
    description: str
    chart: Callable[[str], alt.TopLevelMixin]
    width: float = 1.0


def _percent(factor: float) -> str:
    """0.41 -> "41%"."""
    return f"{factor:.0%}"


def key_findings(daily: pd.DataFrame, weather: pd.DataFrame) -> list[Finding]:
    """The key findings, computed from daily (to_daily) and weather
    (load_weather), so that the numbers in the texts match the charts."""
    weekday = weekday_profile(daily)
    by_weekday = weekday.set_index("weekday")["factor"]
    month = month_profile(daily)
    by_month = month.set_index("month")["factor"]
    busiest = by_month.nlargest(2)
    effect = rain_effect(daily, weather)
    rain = effect.set_index(["day_type", "rain"])["factor"]
    outages = unflagged_outages(daily)
    share = inbound_share(daily)

    def drop(day_type: str) -> str:
        # Difference between a dry day and a day with 8+ hours of rain
        return f"{(rain[(day_type, 'Dry')] - rain[(day_type, '8+ h')]) * 100:.0f}"

    if outages.empty:
        outage_text = (
            "The city marks faults in a status column; outages it misses are found by comparing the "
            "two directions."
        )
    else:
        starts = " and ".join(f"{row.start:%B %Y} ({row.days} days)" for row in outages.itertuples())
        outage_text = (
            f"The city marks faults in a status column, but it missed the longest outages: from {starts} "
            "one direction counted nothing while the status said OK (grey bands). They are found by "
            "comparing the two directions."
        )
    years = ", ".join(map(str, month.attrs["years"]))
    return [
        Finding(
            name="weekday",
            title="The weekday matters most",
            text=(
                f"On Sundays about {_percent(by_weekday['Sun'])} as many people cycle as on an average "
                f"workday, on Saturdays {_percent(by_weekday['Sat'])}. Wednesday is the busiest day "
                f"({_percent(by_weekday['Wed'])}); Wednesday and Saturday are market days on the "
                "Domplatz, which may explain it (a hypothesis, not checked)."
            ),
            description="Bar chart of the median count per weekday relative to an average workday.",
            chart=lambda theme: weekday_chart(weekday, theme),
            width=0.5,
        ),
        Finding(
            name="month",
            title="The season matters less",
            text=(
                f"January and December reach {_percent(by_month['Jan'])} and "
                f"{_percent(by_month['Dec'])} of the annual mean, {busiest.index[0]} and "
                f"{busiest.index[1]} {_percent(busiest.iloc[0])}. April dips: Easter, school holidays "
                "and the lecture-free period of the university fall into it. Median over the full "
                f"years {years}."
            ),
            description="Bar chart of the typical monthly mean relative to the annual mean.",
            chart=lambda theme: month_chart(month, theme),
            width=0.75,
        ),
        Finding(
            name="rain",
            title="Rain keeps leisure cyclists at home",
            text=(
                "Each day is compared with the typical day of the same weekday at the same time of "
                "year, which takes the weekly pattern and the season out. From a dry day to a day with "
                f"eight or more hours of rain, the count drops by {drop(WORKDAY)} percentage points on "
                f"workdays, but by {drop(WEEKEND)} at weekends: commuters ride anyway, leisure trips "
                "are dropped. Temperature hardly matters once the season is taken out."
            ),
            description=(
                "Grouped bar chart of the count relative to a typical day by hours of rain, for "
                "workdays and weekends."
            ),
            chart=lambda theme: rain_chart(effect, theme),
            width=0.6,
        ),
        Finding(
            name="share",
            title="The data cannot be taken at face value",
            text=(
                f"{outage_text} The share of trips into the city also jumps twice and stays on the new "
                "level (dashed lines), probably after maintenance at the counting site. Since October "
                "2025 the site counts about 40% more than before, more than the season explains. "
                "Counts from different periods are therefore not fully comparable, and the model gets "
                "the measurement period as a feature. In May 2026 the inbound channel slowly lost "
                "about a third of its trips while the outbound channel counted as usual, until it "
                "recovered overnight on 3 June. These four weeks are marked as faulty by hand and left "
                "out (the gap in the line)."
            ),
            description=(
                "Line chart of the share of trips into the city over time, with shaded outages and "
                "dashed lines at two jumps."
            ),
            chart=lambda theme: share_chart(share, outages, theme),
        ),
    ]
