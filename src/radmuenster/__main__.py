"""Build daily counts: python -m radmuenster"""

from pathlib import Path

from radmuenster.data import DEFAULT_SITE, load_intervals, to_daily


def main() -> None:
    daily = to_daily(load_intervals(DEFAULT_SITE))
    out = Path("data/processed") / f"daily_{DEFAULT_SITE}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out)

    n_bad = int((~daily["complete"]).sum())
    print(f"Saved {len(daily)} days to {out}")
    print(f"Period: {daily.index.min():%Y-%m-%d} to {daily.index.max():%Y-%m-%d}")
    print(f"Incomplete or flagged days: {n_bad}")


if __name__ == "__main__":
    main()
