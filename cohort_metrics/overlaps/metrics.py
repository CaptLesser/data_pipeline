import argparse
from getpass import getpass

from cohort_metrics.core import (
    compute_cohort_metrics,
    build_symbol_baseline,
    enrich_current_with_baseline,
    resolve_input_path,
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
    p.add_argument("--emit-series", action="store_true", help="Also emit per-window metrics time series")
    p.add_argument("--series-output", default="overlap_metrics_series.csv", help="Output CSV path for metrics time series")
    p.add_argument("--dense", action="store_true", help="Emit dense per-minute rolling metrics")
    p.add_argument("--dense-windows", help="CSV windows for dense mode (e.g., 60,240,720,1440)")
    p.add_argument("--dense-output", default="overlap_metrics_dense.csv", help="Output CSV path for dense metrics")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve input path robustly and run metrics
    input_path = resolve_input_path(args.input)
    current_df = compute_cohort_metrics(input_path, args.output)

    # Optionally emit time series of window metrics (independent of baseline)
    if bool(args.emit_series):
        from cohort_metrics.core import compute_cohort_metrics_series, WINDOWS_MINUTES
        compute_cohort_metrics_series(input_path, args.series_output, windows_minutes=WINDOWS_MINUTES)

    # Optionally emit dense per-minute rolling metrics
    if bool(args.dense):
        from cohort_metrics.core import compute_cohort_metrics_dense, WINDOWS_MINUTES
        if args.dense_windows:
            # Parse windows
            wins = []
            for p in str(args.dense_windows).split(','):
                p = p.strip().lower()
                if not p:
                    continue
                if p.endswith('h'):
                    wins.append(int(float(p[:-1]) * 60))
                else:
                    wins.append(int(p))
        else:
            wins = WINDOWS_MINUTES
        compute_cohort_metrics_dense(input_path, args.dense_output, windows_minutes=wins)

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
