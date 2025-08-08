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
        .agg({"open": "first", "close": "last", "volume": "sum"})
        .dropna()
    )

    price_change = ((resampled["close"] - resampled["open"]) / resampled["open"]) * 100
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
# Step 7: Bucket skew detection with HDBSCAN
# ------------------------
def detect_skew(bucket_counts: Dict[str, Counter[str]]) -> Dict[str, Dict[str, float]]:
    """Cluster bucket proportion vectors and flag disproportionate skews."""
    if not bucket_counts:
        return {}

    buckets = sorted({b for counts in bucket_counts.values() for b in counts})

    data: List[List[float]] = []
    coins: List[str] = []
    for coin, counts in bucket_counts.items():
        total = sum(counts.values())
        vec = [counts.get(b, 0) / total if total else 0 for b in buckets]
        data.append(vec)
        coins.append(coin)

    X = np.array(data)

    # For very small samples, simply report dominant buckets without skewing.
    if len(X) < 2:
        result: Dict[str, Dict[str, float]] = {}
        for coin, vec in zip(coins, X):
            dominant_idx = int(np.argmax(vec)) if len(vec) else 0
            dominant_bucket = buckets[dominant_idx] if buckets else ""
            result[coin] = {
                "skewed": 0,
                "cluster": 0,
                "dominant_bucket": dominant_bucket,
                "skew_strength": float(vec[dominant_idx]) if len(vec) else 0.0,
            }
        return result

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(X)

    # Baseline medians for each bucket globally and within each cluster
    global_median = np.median(X, axis=0)
    cluster_medians: Dict[int, np.ndarray] = {}
    for label in set(labels):
        cluster_mask = labels == label
        cluster_medians[int(label)] = np.median(X[cluster_mask], axis=0)

    result: Dict[str, Dict[str, float]] = {}
    for idx, (coin, vec) in enumerate(zip(coins, X)):
        label = int(labels[idx])
        dominant_idx = int(np.argmax(vec)) if len(vec) else 0
        dominant_bucket = buckets[dominant_idx] if buckets else ""
        dominant_share = float(vec[dominant_idx]) if len(vec) else 0.0

        baseline = (
            cluster_medians[label][dominant_idx]
            if label != -1
            else global_median[dominant_idx]
        )
        skewed = int(dominant_share > baseline + 0.1)

        result[coin] = {
            "skewed": skewed,
            "cluster": label,
            "dominant_bucket": dominant_bucket,
            "skew_strength": dominant_share,
        }

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
    skew_info = detect_skew(bucket_counts)

    print("\nHabitual Losers:", losers)
    print("\nHabitual Gainers:", gainers)
    print("\nOverlaps:", overlaps)
    print("\nSkew Analysis:", skew_info)

    save_full_timeseries_csv(losers, history_df, "habitual_losers.csv")
    save_full_timeseries_csv(gainers, history_df, "habitual_gainers.csv")
    save_full_timeseries_csv(overlaps, history_df, "habitual_overlaps.csv")

    conn.close()


if __name__ == "__main__":
    main()
