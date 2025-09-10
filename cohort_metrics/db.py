from typing import List

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_db_engine(host: str, user: str, password: str, database: str, port: int = 3306) -> Engine:
    url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return create_engine(url)


def fetch_history_months(
    engine: Engine,
    symbols: List[str],
    months: int,
    table: str = "ohlcvt",
) -> pd.DataFrame:
    """Fetch N months of 1-minute OHLCV for given symbols in a single query."""
    if not symbols:
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    placeholders = ",".join(["%s"] * len(symbols))
    query = f"""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM {table}
        WHERE symbol IN ({placeholders})
          AND timestamp >= UTC_TIMESTAMP() - INTERVAL {months} MONTH
        ORDER BY symbol ASC, timestamp ASC
    """
    return pd.read_sql(query, engine, params=tuple(symbols))

