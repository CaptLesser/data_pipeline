"""Generate multi-timeframe leaderboards for habitual losers, gainers, and
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
7. Analyze bucket tallies to compute centroid distances, profile tags, and
   per-bucket deviations for top coins
8. Output CSVs containing full time series for qualifying coins
9. Print summary lists and skew reports to the terminal

Note: Ensure the MySQL table is indexed on (symbol, timestamp) for performance.
"""

import argparse
from collections import Counter, defaultdict
from getpass import getpass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import numpy as np
import mysql.connector

TABLE_NAME = "ohlcvt"


# ------------------------
# Database connection
# ------------------------
def get_db_connection(
    host: str, user: str, password: str, database: str, port: int = 3306
):
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
def get_all_coin_history(
    conn: mysql.connector.MySQLConnection, symbols: List[str]
) -> pd.DataFrame:
    """Retrieve full 30-day minute OHLCVT history for all symbols in one pull."""
    if not symbols:
        return pd.DataFrame(
            columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        )
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
    total volume against the previous disjoint window. All resample windows
    align to UTC boundaries using ``origin='epoch'``.
    """

    if df.empty:
        return pd.DataFrame(
            columns=["window_start", "price_change", "total_volume", "pct_vol_change"]
        )

    df = df.sort_values("timestamp").set_index("timestamp")

    resampled = (
        df.resample(
            f"{window_hours}H",
            label="left",
            closed="left",
            origin="epoch",
        )
        .agg({"close": "last", "volume": "sum"})
        .dropna()
    )

    price_change = resampled["close"].pct_change() * 100
    total_volume = resampled["volume"]
    pct_vol_change = total_volume.pct_change() * 100

    stats = (
        pd.DataFrame(
            {
                "window_start": resampled.index,
                "price_change": price_change,
                "total_volume": total_volume,
                "pct_vol_change": pct_vol_change,
            }
        )
        .dropna()
        .reset_index(drop=True)
    )

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
            top_losers = group[group["price_change"] < 0].nsmallest(
                top_n, "price_change"
            )["symbol"]
            for sym in top_losers:
                loser_counts[sym] += 1
                bucket_counts[sym][f"price_{bucket}_loss"] += 1

            top_gainers = group[group["price_change"] > 0].nlargest(
                top_n, "price_change"
            )["symbol"]
            for sym in top_gainers:
                gainer_counts[sym] += 1
                bucket_counts[sym][f"price_{bucket}_gain"] += 1

            vol_drop = group[group["pct_vol_change"] < 0].nsmallest(
                top_n, "pct_vol_change"
            )["symbol"]
            for sym in vol_drop:
                bucket_counts[sym][f"vol_{bucket}_drop"] += 1

            vol_spike = group[group["pct_vol_change"] > 0].nlargest(
                top_n, "pct_vol_change"
            )["symbol"]
            for sym in vol_spike:
                bucket_counts[sym][f"vol_{bucket}_spike"] += 1

    habitual_losers = [c for c, _ in loser_counts.most_common(top_n)]
    habitual_gainers = [c for c, _ in gainer_counts.most_common(top_n)]
    max_g = max(gainer_counts.values(), default=0)
    max_l = max(loser_counts.values(), default=0)
    overlap_scores = {}
    all_coins = set(gainer_counts) | set(loser_counts)
    for coin in all_coins:
        gain_frac = gainer_counts[coin] / max_g if max_g else 0
        loss_frac = loser_counts[coin] / max_l if max_l else 0
        overlap_scores[coin] = min(gain_frac, loss_frac)
    overlaps = [
        c
        for c, _ in sorted(overlap_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
    ]
    return habitual_losers, habitual_gainers, overlaps, bucket_counts


# ------------------------
# Step 7: Bucket skew detection vs. centroid baseline
# ------------------------
def detect_skew(
    bucket_counts: Dict[str, Counter[str]],
) -> Tuple[
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, float],
]:
    """Measure per-bucket deviation from a centroid of top coins.

    Parameters
    ----------
    bucket_counts : Dict[str, Counter[str]]
        Mapping of coin -> bucket tally counts.

    Returns
    -------
    Tuple
        1) Nested mapping of coin -> bucket -> {"level", "deviation_pct"}
           for buckets that deviate from the centroid baseline.
        2) Mapping of coin -> centroid distance.
    """

    if not bucket_counts:
        return {}, {}

    all_buckets = sorted({b for counts in bucket_counts.values() for b in counts})
    if not all_buckets:
        return {}, {}

    vectors: List[np.ndarray] = []
    coins: List[str] = []
    for coin, counts in bucket_counts.items():
        vec = np.array([counts.get(b, 0) for b in all_buckets], dtype=float)
        total = vec.sum()
        if total == 0:
            continue
        vectors.append(vec / total)
        coins.append(coin)

    if not vectors:
        return {}, {}

    X = np.vstack(vectors)
    centroid = X.mean(axis=0)

    coin_devs: Dict[str, Dict[str, float]] = {}
    all_abs_devs: List[float] = []
    for coin, vec in zip(coins, X):
        with np.errstate(divide="ignore", invalid="ignore"):
            dev_vec = np.where(centroid != 0, (vec - centroid) / centroid * 100, 0)
        coin_devs[coin] = dict(zip(all_buckets, dev_vec))
        all_abs_devs.extend([abs(d) for d in dev_vec])

    abs_array = np.array(all_abs_devs)
    heavy_cut = float(np.percentile(abs_array, 95))
    moderate_cut = float(np.percentile(abs_array, 75))
    slight_cut = float(np.percentile(abs_array, 50))

    distances_arr = np.linalg.norm(X - centroid, axis=1)

    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    distances: Dict[str, float] = {}
    for coin, devs, dist in zip(coins, [coin_devs[c] for c in coins], distances_arr):
        distances[coin] = float(dist)
        bucket_info: Dict[str, Dict[str, float]] = {}
        for bucket, dev in devs.items():
            abs_dev = abs(dev)
            if abs_dev >= heavy_cut:
                level = "heavy"
            elif abs_dev >= moderate_cut:
                level = "moderate"
            elif abs_dev >= slight_cut:
                level = "slight"
            else:
                continue
            bucket_info[bucket] = {"level": level, "deviation_pct": float(dev)}
        if bucket_info:
            result[coin] = bucket_info

    return result, distances


def compute_profile_tags(
    counts_map: Dict[str, Counter[str]],
    distances: Dict[str, float],
    vol_price_ratio: float = 1.5,
    short_long_ratio: float = 1.5,
) -> Dict[str, List[str]]:
    """Derive profile tags for each coin based on bucket counts and distance.

    Tags include:
    - ``Vol Mover``: volume buckets dominate price buckets
    - ``Short-Term``: short windows (6h/24h) dominate long ones (72h/168h)
    - ``Long-Term``: long windows dominate short ones
    - ``Balanced``: centroid distance is within one standard deviation of the mean
    """

    tags: Dict[str, List[str]] = {}
    dist_values = list(distances.values())
    threshold = (
        float(np.mean(dist_values) + np.std(dist_values)) if dist_values else 0.0
    )

    for coin, counts in counts_map.items():
        vol_sum = sum(v for k, v in counts.items() if k.startswith("vol_"))
        price_sum = sum(v for k, v in counts.items() if k.startswith("price_"))
        short_sum = sum(
            v for k, v in counts.items() if k.split("_")[1] in {"6h", "24h"}
        )
        long_sum = sum(
            v for k, v in counts.items() if k.split("_")[1] in {"72h", "168h"}
        )

        coin_tags: List[str] = []
        if vol_sum > vol_price_ratio * price_sum:
            coin_tags.append("Vol Mover")
        if short_sum > short_long_ratio * long_sum:
            coin_tags.append("Short-Term")
        elif long_sum > short_long_ratio * short_sum:
            coin_tags.append("Long-Term")
        if distances.get(coin, float("inf")) < threshold:
            coin_tags.append("Balanced")
        tags[coin] = coin_tags

    return tags


# ------------------------
# Step 8: Save CSVs
# ------------------------
def save_full_timeseries_csv(
    symbol_list: Iterable[str], history: pd.DataFrame, filename: str
) -> None:
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
        description="Generate leaderboards from OHLCVT data",
    )
    parser.add_argument("--host", help="MySQL host")
    parser.add_argument("--user", help="MySQL user")
    parser.add_argument("--database", help="Database name")
    parser.add_argument("--port", type=int, default=3306, help="MySQL port")
    parser.add_argument("--password", help="MySQL password (prompt if omitted)")
    parser.add_argument(
        "--top-n", type=int, default=20, help="Number of coins per leaderboard",
    )
    args = parser.parse_args()

    if args.host is None:
        args.host = input("MySQL host: ")
    if args.user is None:
        args.user = input("MySQL user: ")
    if args.database is None:
        args.database = input("Database name: ")
    if args.password is None:
        args.password = getpass("MySQL password: ")

    return args


def main() -> None:
    args = parse_args()

    conn = get_db_connection(
        args.host, args.user, args.password, args.database, args.port
    )
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

    latest_24h = (
        aggregated_stats[24]
        .sort_values("window_start")
        .groupby("symbol")
        .tail(1)
        .set_index("symbol")[["price_change", "pct_vol_change"]]
        .to_dict("index")
    )

    def print_section(name: str, coins: List[str]) -> None:
        counts_map = {c: bucket_counts[c] for c in coins if c in bucket_counts}
        skew_info, distances = detect_skew(counts_map)
        tags = compute_profile_tags(counts_map, distances)

        print(f"\n{name}:")
        for c in coins:
            metrics = latest_24h.get(c, {})
            price = metrics.get("price_change", float("nan"))
            vol = metrics.get("pct_vol_change", float("nan"))
            tag_str = ", ".join(tags.get(c, [])) or "None"
            print(f"[{c}] PriceΔ: {price:+.2f}% VolΔ: {vol:+.2f}% Skew: {tag_str}")

        if skew_info:
            print(f"\nSkew Details ({name}):")
            print(f"{'Coin':<10}{'Bucket':<20}{'Level':<10}{'Deviation%':>12}")
            for coin in coins:
                for bucket, info in skew_info.get(coin, {}).items():
                    print(
                        f"{coin:<10}{bucket:<20}{info['level']:<10}{info['deviation_pct']:+12.2f}"
                    )

    print_section("Top Gainers", gainers)
    print_section("Top Losers", losers)
    print_section("Top Overlap", overlaps)

    save_full_timeseries_csv(losers, history_df, "habitual_losers.csv")
    save_full_timeseries_csv(gainers, history_df, "habitual_gainers.csv")
    save_full_timeseries_csv(overlaps, history_df, "habitual_overlaps.csv")

    conn.close()


if __name__ == "__main__":
    main()
