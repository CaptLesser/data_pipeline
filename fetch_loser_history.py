"""Fetch 30-day OHLCVT history for top losers and save to CSV."""

from __future__ import annotations

import logging
import os
from typing import Sequence

import mysql.connector
import pandas as pd


# --- Configurations ---
# Values can be overridden with environment variables to avoid hardcoding
# credentials in source control.
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "YOUR_USER"),
    "password": os.getenv("DB_PASSWORD", "YOUR_PASS"),
    "database": os.getenv("DB_NAME", "YOUR_DB"),
}
INPUT_CSV = os.getenv("LOSERS_CSV", "kraken_top_losers.csv")
OUTPUT_CSV = os.getenv("HISTORY_OUTPUT_CSV", "losers_30day_history.csv")
TABLE_NAME = os.getenv("OHLCVT_TABLE", "ohlcvt")


def fetch_history(symbols: Sequence[str], conn: mysql.connector.MySQLConnection) -> pd.DataFrame:
    """Return 30-day history for each symbol from the database."""

    all_dfs = []
    query = (
        f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE symbol = %s AND timestamp >= NOW() - INTERVAL 30 DAY
        ORDER BY timestamp ASC
        """
    )
    for symbol in symbols:
        df = pd.read_sql(query, conn, params=(symbol,))
        if not df.empty:
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def main() -> None:
    """Load symbols from CSV, fetch their history, and export to another CSV."""

    logging.basicConfig(level=logging.INFO)

    losers = pd.read_csv(INPUT_CSV)
    symbols = losers["symbol"].tolist()

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as exc:
        raise SystemExit(f"Error connecting to database: {exc}") from exc

    try:
        df = fetch_history(symbols, conn)
    finally:
        conn.close()

    df.to_csv(OUTPUT_CSV, index=False)
    logging.info("Saved history for %d symbols to %s", len(symbols), OUTPUT_CSV)


if __name__ == "__main__":
    main()
