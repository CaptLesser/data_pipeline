"""
leaderboard_generator.py

Generates multi-timeframe leaderboards for habitual losers, gainers, 
and overlap coins based on OHLCV minute bars stored in MySQL.

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

import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime, timedelta
import hdbscan  # For skew detection

# ------------------------
# Database connection
# ------------------------
def get_db_connection():
    """
    Create and return a connection to the MySQL database.
    Replace placeholders with actual credentials locally.
    """
    return mysql.connector.connect(
        host="DB_HOST",
        user="DB_USER",
        password="DB_PASSWORD",
        database="DB_NAME"
    )

# ------------------------
# Step 1: Filter recent coins
# ------------------------
def get_recent_coins(conn):
    """
    Get list of coins whose most recent bar is within 1 day of current time.
    SQL handles the filtering to reduce local load.
    """
    query = """
    SELECT symbol
    FROM ohlcv_minute
    GROUP BY symbol
    HAVING MAX(timestamp) >= NOW() - INTERVAL 1 DAY
    """
    return pd.read_sql(query, conn)["symbol"].tolist()

# ------------------------
# Step 2: Pull full 30-day data
# ------------------------
def get_coin_history(conn, symbol):
    """
    Retrieve full 30-day minute OHLCV history for given symbol.
    """
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM ohlcv_minute
    WHERE symbol = %s
      AND timestamp >= NOW() - INTERVAL 30 DAY
    ORDER BY timestamp ASC
    """
    return pd.read_sql(query, conn, params=[symbol])

# ------------------------
# Step 3-4: Compute window metrics
# ------------------------
def compute_window_stats(df, window_hours):
    """
    Slice dataframe into rolling windows and compute:
      - Price % change
      - Total volume
      - Volume % change from previous
    """
    window_bars = window_hours * 60
    stats = []
    prev_total_volume = None
    
    for start in range(0, len(df) - window_bars, window_bars):
        window_df = df.iloc[start:start + window_bars]
        if len(window_df) < window_bars:
            continue
        
        open_price = window_df.iloc[0]["open"]
        close_price = window_df.iloc[-1]["close"]
        pct_price_change = ((close_price - open_price) / open_price) * 100
        
        total_volume = window_df["volume"].sum()
        pct_vol_change = None
        if prev_total_volume is not None:
            pct_vol_change = ((total_volume - prev_total_volume) / prev_total_volume) * 100
        prev_total_volume = total_volume
        
        stats.append((window_df.iloc[0]["timestamp"], pct_price_change, total_volume, pct_vol_change))
    
    return pd.DataFrame(stats, columns=["window_start", "price_change", "total_volume", "pct_vol_change"])

# ------------------------
# Step 5-6: Ranking & aggregation
# ------------------------
def build_leaderboards(all_stats, top_n=20):
    """
    Rank coins in each bucket, aggregate appearances for:
      - Habitual losers
      - Habitual gainers
      - Overlaps
    """
    # Placeholder logic — Codex will expand this
    habitual_losers = []
    habitual_gainers = []
    overlaps = []
    return habitual_losers, habitual_gainers, overlaps

# ------------------------
# Step 7: Bucket skew detection with HDBSCAN
# ------------------------
def detect_skew(appearance_counts):
    """
    Use HDBSCAN clustering to detect disproportionate appearances 
    in certain buckets.
    Returns dict: {symbol: {"skewed": True/False, "towards": bucket_name}}
    """
    # Placeholder — Codex to refine
    pass

# ------------------------
# Step 8: Save CSVs
# ------------------------
def save_full_timeseries_csv(symbol_list, conn, filename):
    """
    For all qualifying coins, save their full 30-day timeseries to a CSV.
    """
    all_data = []
    for sym in symbol_list:
        df = get_coin_history(conn, sym)
        df["symbol"] = sym
        all_data.append(df)
    
    pd.concat(all_data).to_csv(filename, index=False)

# ------------------------
# Step 9: Main routine
# ------------------------
def main():
    conn = get_db_connection()
    coins = get_recent_coins(conn)
    print(f"Found {len(coins)} coins with fresh data.")
    
    all_stats = {}
    for coin in coins:
        df = get_coin_history(conn, coin)
        all_stats[coin] = {}
        for window in [6, 24, 72, 168]:
            all_stats[coin][window] = compute_window_stats(df, window)
    
    losers, gainers, overlaps = build_leaderboards(all_stats)
    print("\nHabitual Losers:", losers)
    print("\nHabitual Gainers:", gainers)
    print("\nOverlaps:", overlaps)
    
    save_full_timeseries_csv(losers, conn, "habitual_losers.csv")
    save_full_timeseries_csv(gainers, conn, "habitual_gainers.csv")
    save_full_timeseries_csv(overlaps, conn, "habitual_overlaps.csv")

if __name__ == "__main__":
    main()
