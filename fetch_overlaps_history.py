"""Fetch 30-day OHLCVT history for overlap cohort and save to CSV.

Reads symbols from a cohort CSV (default: habitual_overlaps.csv) and
pulls the last 30 days of minute bars from MySQL `ohlcvt`.

Credentials can be provided via CLI flags, environment variables, or
interactive prompts (for any missing values).
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Sequence

import mysql.connector
import pandas as pd
from getpass import getpass


# --- Defaults / Env ---
DEFAULT_INPUT_CSV = os.getenv("OVERLAPS_CSV", "habitual_overlaps.csv")
DEFAULT_OUTPUT_CSV = os.getenv("OVERLAPS_HISTORY_OUTPUT_CSV", "overlaps_30day_history.csv")
TABLE_NAME = os.getenv("OHLCVT_TABLE", "ohlcvt")


def build_db_config(args: argparse.Namespace) -> dict:
    host = args.host or os.getenv("DB_HOST", "localhost")
    port_env = os.getenv("DB_PORT")
    port = int(args.port if args.port is not None else (port_env if port_env else 3306))
    user = args.user or os.getenv("DB_USER")
    database = args.database or os.getenv("DB_NAME")
    password = args.password or os.getenv("DB_PASSWORD")

    if user is None:
        user = input("MySQL user: ")
    if database is None:
        database = input("Database name: ")
    if password is None:
        password = getpass("MySQL password: ")

    return {"host": host, "port": port, "user": user, "password": password, "database": database}


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

    parser = argparse.ArgumentParser(description="Fetch 30-day OHLCVT history for overlaps cohort")
    parser.add_argument("--host", help="MySQL host")
    parser.add_argument("--user", help="MySQL user")
    parser.add_argument("--database", help="MySQL database name")
    parser.add_argument("--password", help="MySQL password")
    parser.add_argument("--port", type=int, default=None, help="MySQL port (default 3306)")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Input CSV with symbols column")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Output CSV filename")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cohort = pd.read_csv(args.input_csv)
    symbols = cohort["symbol"].drop_duplicates().tolist()

    db_config = build_db_config(args)

    try:
        conn = mysql.connector.connect(**db_config)
    except mysql.connector.Error as exc:
        raise SystemExit(f"Error connecting to database: {exc}") from exc

    try:
        df = fetch_history(symbols, conn)
    finally:
        conn.close()

    df.to_csv(args.output_csv, index=False)
    logging.info("Saved history for %d symbols to %s", len(symbols), args.output_csv)


if __name__ == "__main__":
    main()
