"""leaderboards.py

Generate multi-timeframe leaderboards for habitual losers, gainers, and
overlap coins based on OHLCVT minute bars stored in MySQL.

Workflow:
1. Filter for coins with recent data (within 1 day of now)
2. For each coin, pull 30-day minute-level data
3. Slice into rolling windows: 6h, 24h, 72h, 168h
4. Calculate:
   - Price % change per window
   - Total volume per window
   - Volume % change vs. previous window
5. Rank coins for each window & metric
6. Aggregate counts over 30 days to find habitual losers/gainers/overlaps
7. Perform clustering (HDBSCAN) to detect disproportional "bucket skew"
8. Output CSVs containing full time series for qualifying coins
9. Print summary lists to terminal

Note: Ensure the MySQL table is indexed on (symbol, timestamp) for performance.
"""

import argparse
from collections import Counter, defaultdict
from getpass import getpass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import numpy as np
import mysql.connector
import hdbscan  # For skew detection

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
    """Return coins with data within the last day."""
    query = f"""
    SELECT symbol
    FROM {TABLE_NAME}
    GROUP BY symbol
    HAVING MAX(timestamp) >= NOW() - INTERVAL 1 DAY
    """
    return pd.read_sql(query, conn)["symbol"].tolist()

# ------------------------
# Step 2: Pull full 30-day data
# ------------------------
def get_coin_history(conn: mysql.connector.MySQLConnection, symbol: str) -> pd.DataFrame:
    """Retrieve full 30-day minute OHLCVT history for a symbol."""
    query = f"""
    SELECT timestamp, open, high, low, close, volume
    FROM {TABLE_NAME}
    WHERE symbol = %s
      AND timestamp >= NOW() - INTERVAL 30 DAY
    ORDER BY timestamp ASC
    """
    return pd.read_sql(query, conn, params=[symbol])

# ------------------------
# Step 3-4: Compute window metrics
# ------------------------
def compute_window_stats(df: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    """Create rolling lookback windows and compute metrics.

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
    """

    window_bars = window_hours * 60
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Rolling metrics using trailing windows for efficiency
    open_shifted = df["open"].shift(window_bars - 1)
    close_price = df["close"]
    price_change = ((close_price - open_shifted) / open_shifted) * 100

    total_volume = df["volume"].rolling(window_bars).sum()
    pct_vol_change = total_volume.pct_change() * 100

    window_start = df["timestamp"].shift(window_bars - 1)
    stats = pd.DataFrame(
        {
            "window_start": window_start,
            "price_change": price_change,
            "total_volume": total_volume,
            "pct_vol_change": pct_vol_change,
        }
    )

    # Drop incomplete windows at the start of the series
    return stats.dropna().reset_index(drop=True)

# ------------------------
# Step 5-6: Ranking & aggregation
# ------------------------
def build_leaderboards(
    all_stats: Dict[str, Dict[int, pd.DataFrame]],
    top_n: int = 20,
) -> Tuple[List[str], List[str], List[str], Dict[str, Counter[str]]]:
    """Aggregate window metrics to produce leaderboards and bucket counts."""

    loser_counts: Counter[str] = Counter()
    gainer_counts: Counter[str] = Counter()
    bucket_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for coin, windows in all_stats.items():
        for window, window_df in windows.items():
            bucket = f"{window}h"

            changes = window_df["price_change"]
            losers = int((changes < 0).sum())
            gainers = int((changes > 0).sum())
            loser_counts[coin] += losers
            gainer_counts[coin] += gainers
            bucket_counts[coin][f"price_{bucket}_loss"] += losers
            bucket_counts[coin][f"price_{bucket}_gain"] += gainers

            vol_changes = window_df["pct_vol_change"]
            vol_drop = int((vol_changes < 0).sum())
            vol_spike = int((vol_changes > 0).sum())
            bucket_counts[coin][f"vol_{bucket}_drop"] += vol_drop
            bucket_counts[coin][f"vol_{bucket}_spike"] += vol_spike

    habitual_losers = [c for c, _ in loser_counts.most_common(top_n)]
    habitual_gainers = [c for c, _ in gainer_counts.most_common(top_n)]
    overlaps = list(set(habitual_losers) & set(habitual_gainers))
    return habitual_losers, habitual_gainers, overlaps, bucket_counts

# ------------------------
# Step 7: Bucket skew detection with HDBSCAN
# ------------------------
def detect_skew(bucket_counts: Dict[str, Counter[str]]) -> Dict[str, Dict[str, int]]:
    """Cluster bucket appearance vectors to detect skew using HDBSCAN."""
    if not bucket_counts:
        return {}

    # Determine full list of buckets
    buckets = sorted({b for counts in bucket_counts.values() for b in counts})

    data = []
    coins = []
    for coin, counts in bucket_counts.items():
        data.append([counts.get(b, 0) for b in buckets])
        coins.append(coin)

    # If there aren't enough samples, return non-skewed results without clustering
    if len(data) < 2:
        result: Dict[str, Dict[str, int]] = {}
        for coin, counts in zip(coins, data):
            dominant_idx = int(np.argmax(counts)) if counts else 0
            result[coin] = {
                "skewed": 0,
                "cluster": 0,
                "dominant_bucket": buckets[dominant_idx] if buckets else "",
            }
        return result

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(np.array(data))

    result: Dict[str, Dict[str, int]] = {}
    for coin, label, counts in zip(coins, labels, data):
        dominant_idx = int(np.argmax(counts)) if counts else 0
        dominant_bucket = buckets[dominant_idx] if buckets else ""
        result[coin] = {
            "skewed": int(label == -1),
            "cluster": int(label),
            "dominant_bucket": dominant_bucket,
        }
    return result

# ------------------------
# Step 8: Save CSVs
# ------------------------
def save_full_timeseries_csv(symbol_list: Iterable[str], conn: mysql.connector.MySQLConnection, filename: str) -> None:
    """Save full 30-day timeseries for symbols to a CSV."""
    all_data: List[pd.DataFrame] = []
    for sym in symbol_list:
        df = get_coin_history(conn, sym)
        df["symbol"] = sym
        all_data.append(df)
    if all_data:
        pd.concat(all_data).to_csv(filename, index=False)

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

    all_stats: Dict[str, Dict[int, pd.DataFrame]] = {}
    for coin in coins:
        df = get_coin_history(conn, coin)
        all_stats[coin] = {}
        for window in [6, 24, 72, 168]:
            all_stats[coin][window] = compute_window_stats(df, window)

    losers, gainers, overlaps, bucket_counts = build_leaderboards(
        all_stats, top_n=args.top_n
    )
    skew_info = detect_skew(bucket_counts)

    print("\nHabitual Losers:", losers)
    print("\nHabitual Gainers:", gainers)
    print("\nOverlaps:", overlaps)
    print("\nSkew Analysis:", skew_info)

    save_full_timeseries_csv(losers, conn, "habitual_losers.csv")
    save_full_timeseries_csv(gainers, conn, "habitual_gainers.csv")
    save_full_timeseries_csv(overlaps, conn, "habitual_overlaps.csv")

    conn.close()


if __name__ == "__main__":
    main()
