import argparse
from getpass import getpass

from cohort_metrics.core import (
    compute_cohort_metrics,
    build_symbol_baseline,
    enrich_current_with_baseline,
)
from cohort_metrics.db import get_db_engine, fetch_history_months


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute per-symbol metrics and baseline classifications for overlap cohort")
    p.add_argument("--input", default="habitual_overlaps.csv", help="Input CSV path")
    p.add_argument(
        "--output",
        default="overlap_metrics.csv",
        help="Output CSV path for per-symbol metrics",
    )
    p.add_argument("--months", type=int, default=3, help="Months of history for baselines (default: 3)")
    p.add_argument("--host", help="MySQL host")
    p.add_argument("--user", help="MySQL user")
    p.add_argument("--database", help="Database name")
    p.add_argument("--port", type=int, default=3306, help="MySQL port")
    p.add_argument("--password", help="MySQL password (prompt if omitted)")
    p.add_argument("--table", default="ohlcvt", help="OHLCVT table name")
    p.add_argument("--no-baseline", action="store_true", help="Skip baseline enrichment and only output current metrics")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    current_df = compute_cohort_metrics(args.input, args.output)

    if args.no_baseline or current_df.empty:
        return

    # Gather symbols to baseline
    symbols = current_df["symbol"].dropna().unique().tolist()

    # Prompt for any missing DB args like leaderboards.py
    if args.host is None:
        args.host = input("MySQL host: ")
    if args.user is None:
        args.user = input("MySQL user: ")
    if args.database is None:
        args.database = input("Database name: ")
    if args.password is None:
        args.password = getpass("MySQL password: ")

    engine = get_db_engine(args.host, args.user, args.password, args.database, args.port)
    hist_df = fetch_history_months(engine, symbols, args.months, table=args.table)
    engine.dispose()

    baselines = {}
    for sym, g in hist_df.groupby("symbol"):
        baselines[sym] = build_symbol_baseline(g)

    enriched = enrich_current_with_baseline(current_df, baselines)
    enriched.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
