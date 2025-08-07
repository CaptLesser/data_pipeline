"""
kraken_losers.py

Fetches all USD-quoted crypto asset pairs from Kraken,
retrieves 24-hour price data, computes % change, and
outputs the top N "losers" (biggest price drops).
"""

import requests
import pandas as pd

# --- Configurable parameters ---
TOP_N = 20        # Number of top losers to display/save
MIN_VOLUME = 1000 # Minimum 24h volume to include (adjust as needed)
EXCLUDE_STABLES = True  # Exclude known stables (rudimentary, improve as needed)

def get_usd_pairs():
    """Get all USD trading pairs from Kraken"""
    url = "https://api.kraken.com/0/public/AssetPairs"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch asset pairs: {e}") from e
    if data.get('error'):
        raise RuntimeError(f"Kraken API error: {data['error']}")
    pairs = data['result']
    usd_pairs = [k for k, v in pairs.items() if v.get('wsname', '').endswith('USD')]
    return usd_pairs

def fetch_ticker_data(pairs):
    """Get ticker info for a list of pairs (up to 100 at a time)"""
    url = "https://api.kraken.com/0/public/Ticker"
    chunk_size = 100
    all_data = {}
    for i in range(0, len(pairs), chunk_size):
        chunk = ",".join(pairs[i:i+chunk_size])
        try:
            resp = requests.get(url, params={"pair": chunk}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch ticker data: {e}") from e
        if data.get('error'):
            raise RuntimeError(f"Kraken API error for chunk {chunk}: {data['error']}")
        resp_data = data['result']
        all_data.update(resp_data)
    return all_data

def main():
    usd_pairs = get_usd_pairs()
    print(f"Found {len(usd_pairs)} USD pairs on Kraken.")

    ticker = fetch_ticker_data(usd_pairs)
    rows = []
    for sym, vals in ticker.items():
        try:
            open_ = float(vals['o'])
            last = float(vals['c'][0])
            volume = float(vals['v'][1])  # 24h rolling volume
            pct_change = 100 * (last - open_) / open_
            rows.append({
                "symbol": sym,
                "open": open_,
                "last": last,
                "pct_change": pct_change,
                "volume": volume
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if EXCLUDE_STABLES:
        # Quick filter for stables (improve as needed)
        stables = ['USDT', 'USDC', 'USD', 'DAI']
        df = df[~df['symbol'].str.contains('|'.join(stables))]
    df = df[df['volume'] >= MIN_VOLUME]
    df = df.sort_values('pct_change').reset_index(drop=True)

    print(f"\nTop {TOP_N} 24h losers (by %):")
    print(df[['symbol', 'pct_change', 'volume']].head(TOP_N))

    # Optionally, export to CSV
    df.head(TOP_N).to_csv('kraken_top_losers.csv', index=False)
    print("\nExported to kraken_top_losers.csv.")

if __name__ == "__main__":
    main()
