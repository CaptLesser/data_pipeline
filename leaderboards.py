"""leaderboards.py

Generate multi-timeframe leaderboards for habitual losers, gainers, and
overlap coins based on OHLCVT minute bars stored in MySQL.

Workflow:
1. Filter for coins with recent data (within 1 day of now)
2. Pull 30-day minute-level data for all coins in a single query
3. Slice into disjoint windows: 6h, 24h, 72h, 168h
4. Calculate:
   - Price % change per window
   - Total volume per window
   - Volume % change vs. previous disjoint window
5. Rank coins for each window & metric using top-N appearances
6. Aggregate counts over 30 days to find habitual losers/gainers/overlaps
7. Analyze bucket tallies against a global baseline to detect per-bucket
   skew for top coins
8. Output CSVs containing full time series for qualifying coins
9. Print summary lists and skew reports to the terminal

Note: Ensure the MySQL table is indexed on (symbol, timestamp) for performance.
"""

import argparse
from collections import Counter, defaultdict
from getpass import getpass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import mysql.connector

TABLE_NAME = "ohlcvt"

# ------------------------
# Database connection
# ------------------------
def get_db_connection(host: str, user: str, password: str, database: str, port: int = 3306):
    """Create and return a connection to the MySQL database."""
    return mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port,
    )

# ------------------------
# Step 1: Filter recent coins
# ------------------------
def get_recent_coins(conn: mysql.connector.MySQLConnection) -> List[str]:
    """Return coins with data within the last day (UTC)."""
    query = f"""
    SELECT symbol
    FROM {TABLE_NAME}
    GROUP BY symbol
    HAVING MAX(timestamp) >= UTC_TIMESTAMP() - INTERVAL 1 DAY
    """
    return pd.read_sql(query, conn)["symbol"].tolist()

# ------------------------
# Step 2: Pull full 30-day data
# ------------------------
def get_all_coin_history(conn: mysql.connector.MySQLConnection, symbols: List[str]) -> pd.DataFrame:
    """Retrieve full 30-day minute OHLCVT history for all symbols in one pull."""
    if not symbols:
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    placeholders = ",".join(["%s"] * len(symbols))
    query = f"""
    SELECT symbol, timestamp, open, high, low, close, volume
    FROM {TABLE_NAME}
    WHERE symbol IN ({placeholders})
      AND timestamp >= UTC_TIMESTAMP() - INTERVAL 30 DAY
    ORDER BY symbol ASC, timestamp ASC
    """
    return pd.read_sql(query, conn, params=symbols)

# ------------------------
# Step 3-4: Compute window metrics
# ------------------------
def compute_window_stats(df: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    """Slice data into disjoint windows and compute metrics.

    Parameters
    ----------
    df : DataFrame
        Minute-level OHLCVT data sorted by timestamp ascending.
    window_hours : int
        Size of the lookback window in hours.

    Returns
    -------
    DataFrame
        Columns: window_start (timestamp at start of window), price_change,
        total_volume, pct_vol_change.

    Notes
    -----
    Price change is calculated as the percent change between the current
    window's close and the previous window's close. Volume change compares
    total volume against the previous disjoint window.
    """

    if df.empty:
        return pd.DataFrame(columns=["window_start", "price_change", "total_volume", "pct_vol_change"])

    df = df.sort_values("timestamp").set_index("timestamp")

    resampled = (
        df.resample(
            f"{window_hours}H",
            label="left",
            closed="left",
            origin=df.index[0],
        )
        .agg({"close": "last", "volume": "sum"})
        .dropna()
    )

    price_change = resampled["close"].pct_change() * 100
    total_volume = resampled["volume"]
    pct_vol_change = total_volume.pct_change() * 100

    stats = pd.DataFrame(
        {
            "window_start": resampled.index,
            "price_change": price_change,
            "total_volume": total_volume,
            "pct_vol_change": pct_vol_change,
        }
    ).dropna().reset_index(drop=True)

    return stats

# ------------------------
# Step 5-6: Ranking & aggregation
# ------------------------
def build_leaderboards(
    all_window_stats: Dict[int, pd.DataFrame],
    top_n: int = 20,
) -> Tuple[List[str], List[str], List[str], Dict[str, Counter[str]]]:
    """Aggregate top-N window appearances to produce leaderboards."""

    loser_counts: Counter[str] = Counter()
    gainer_counts: Counter[str] = Counter()
    bucket_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for window, df in all_window_stats.items():
        bucket = f"{window}h"
        if df.empty:
            continue
        for _, group in df.groupby("window_start"):
            top_losers = (
                group[group["price_change"] < 0]
                .nsmallest(top_n, "price_change")["symbol"]
            )
            for sym in top_losers:
                loser_counts[sym] += 1
                bucket_counts[sym][f"price_{bucket}_loss"] += 1

            top_gainers = (
                group[group["price_change"] > 0]
                .nlargest(top_n, "price_change")["symbol"]
            )
            for sym in top_gainers:
                gainer_counts[sym] += 1
                bucket_counts[sym][f"price_{bucket}_gain"] += 1

            vol_drop = (
                group[group["pct_vol_change"] < 0]
                .nsmallest(top_n, "pct_vol_change")["symbol"]
            )
            for sym in vol_drop:
                bucket_counts[sym][f"vol_{bucket}_drop"] += 1

            vol_spike = (
                group[group["pct_vol_change"] > 0]
                .nlargest(top_n, "pct_vol_change")["symbol"]
            )
            for sym in vol_spike:
                bucket_counts[sym][f"vol_{bucket}_spike"] += 1

    habitual_losers = [c for c, _ in loser_counts.most_common(top_n)]
    habitual_gainers = [c for c, _ in gainer_counts.most_common(top_n)]
    overlaps = list(set(habitual_losers) & set(habitual_gainers))
    return habitual_losers, habitual_gainers, overlaps, bucket_counts

# ------------------------
# Step 7: Bucket skew detection vs. global baseline
# ------------------------
def detect_skew(bucket_counts: Dict[str, Counter[str]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Measure per-bucket deviation from the global average.

    Parameters
    ----------
    bucket_counts : Dict[str, Counter[str]]
        Mapping of coin -> bucket tally counts.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, float]]]
        Nested mapping of coin -> bucket -> {"level", "deviation_pct"} for
        buckets that deviate from the global baseline by at least 10%.
    """

    if not bucket_counts:
        return {}

    global_totals: Counter[str] = Counter()
    for counts in bucket_counts.values():
        global_totals.update(counts)

    total_global = sum(global_totals.values())
    if total_global == 0:
        return {}

    global_share = {
        bucket: count / total_global for bucket, count in global_totals.items()
    }

    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    for coin, counts in bucket_counts.items():
        coin_total = sum(counts.values())
        if coin_total == 0:
            continue
        coin_share = {bucket: counts.get(bucket, 0) / coin_total for bucket in global_share}

        deviations: Dict[str, Dict[str, float]] = {}
        for bucket, baseline in global_share.items():
            share = coin_share.get(bucket, 0)
            if baseline == 0:
                continue
            deviation_pct = ((share - baseline) / baseline) * 100
            abs_dev = abs(deviation_pct)
            if abs_dev >= 60:
                level = "heavy"
            elif abs_dev >= 30:
                level = "moderate"
            elif abs_dev >= 10:
                level = "slight"
            else:
                continue
            deviations[bucket] = {"level": level, "deviation_pct": deviation_pct}

        if deviations:
            result[coin] = deviations

    return result

# ------------------------
# Step 8: Save CSVs
# ------------------------
def save_full_timeseries_csv(symbol_list: Iterable[str], history: pd.DataFrame, filename: str) -> None:
    """Save full 30-day timeseries for symbols to a CSV."""
    subset = history[history["symbol"].isin(list(symbol_list))]
    if not subset.empty:
        subset.to_csv(filename, index=False)

# ------------------------
# Step 9: Main routine
# ------------------------
def parse_args() -> argparse.Namespace:
    """Collect database connection and leaderboard options from the CLI."""

    parser = argparse.ArgumentParser(
        description="Generate leaderboards from OHLCVT data"
    )
    parser.add_argument("--host", required=True, help="MySQL host")
    parser.add_argument("--user", required=True, help="MySQL user")
    parser.add_argument("--database", required=True, help="Database name")
    parser.add_argument("--port", type=int, default=3306, help="MySQL port")
    parser.add_argument("--password", help="MySQL password (prompt if omitted)")
    parser.add_argument("--top-n", type=int, default=20, help="Number of coins per leaderboard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.password is None:
        args.password = getpass("MySQL password: ")

    conn = get_db_connection(args.host, args.user, args.password, args.database, args.port)
    coins = get_recent_coins(conn)
    print(f"Found {len(coins)} coins with fresh data.")

    history_df = get_all_coin_history(conn, coins)

    window_stats: Dict[int, List[pd.DataFrame]] = {6: [], 24: [], 72: [], 168: []}
    for symbol, sym_df in history_df.groupby("symbol"):
        for window in window_stats.keys():
            stats = compute_window_stats(sym_df.copy(), window)
            stats["symbol"] = symbol
            window_stats[window].append(stats)

    aggregated_stats = {
        window: pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        for window, dfs in window_stats.items()
    }

    losers, gainers, overlaps, bucket_counts = build_leaderboards(
        aggregated_stats, top_n=args.top_n
    )

    top_coins = set(losers) | set(gainers) | set(overlaps)
    filtered_counts = {c: bucket_counts[c] for c in top_coins if c in bucket_counts}
    skew_info = detect_skew(filtered_counts)

    print("\nHabitual Losers:", losers)
    print("\nHabitual Gainers:", gainers)
    print("\nOverlaps:", overlaps)

    print("\nSkew Report (deviation from global baseline):")
    print(f"{'Coin':<10}{'Bucket':<20}{'Level':<10}{'Deviation%':>12}")
    for coin in sorted(top_coins):
        buckets = skew_info.get(coin)
        if not buckets:
            print(f"{coin:<10}{'None':<20}{'--':<10}{'--':>12}")
            continue
        for bucket, info in buckets.items():
            print(
                f"{coin:<10}{bucket:<20}{info['level']:<10}{info['deviation_pct']:+12.2f}"
            )

    save_full_timeseries_csv(losers, history_df, "habitual_losers.csv")
    save_full_timeseries_csv(gainers, history_df, "habitual_gainers.csv")
    save_full_timeseries_csv(overlaps, history_df, "habitual_overlaps.csv")

    conn.close()


if __name__ == "__main__":
    main()
