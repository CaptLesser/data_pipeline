import pandas as pd
import mysql.connector

# TODO: Set your own DB config, or load from environment variables for security!

# --- Configurations ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'YOUR_USER',
    'password': 'YOUR_PASS',
    'database': 'YOUR_DB'
}
INPUT_CSV = 'kraken_top_losers.csv'
OUTPUT_CSV = 'losers_30day_history.csv'
TABLE_NAME = 'ohlcvt'  # change if your table name is different

def fetch_history(symbols, conn):
    all_dfs = []
    for symbol in symbols:
        query = f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE symbol = %s AND timestamp >= NOW() - INTERVAL 30 DAY
            ORDER BY timestamp ASC
        """
        df = pd.read_sql(query, conn, params=(symbol,))
        if not df.empty:
            all_dfs.append(df)
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def main():
    losers = pd.read_csv(INPUT_CSV)
    symbols = losers['symbol'].tolist()
    conn = mysql.connector.connect(**DB_CONFIG)
    df = fetch_history(symbols, conn)
    conn.close()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved history for {len(symbols)} symbols to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
