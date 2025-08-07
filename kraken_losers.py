"""kraken_losers.py

Fetches all USD-quoted crypto asset pairs from Kraken, retrieves 24-hour
price data, computes % change, and outputs the top N "losers" (biggest
price drops).
"""

import argparse
import logging
import requests
import pandas as pd


class KrakenAPIError(Exception):
    """Raised when the Kraken API returns an error or invalid data."""

STABLE_COINS = {"USDT", "USDC", "DAI"}

def get_usd_pairs():
    """Get all USD trading pairs from Kraken"""
    url = "https://api.kraken.com/0/public/AssetPairs"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise KrakenAPIError(f"Error fetching asset pairs: {exc}") from exc
    if data.get("error"):
        raise KrakenAPIError(f"Kraken API error: {data['error']}")
    try:
        pairs = data["result"]
    except KeyError as exc:
        raise KrakenAPIError("Malformed response: missing 'result'") from exc
    usd_pairs = [k for k, v in pairs.items() if v.get("wsname", "").endswith("USD")]
    return usd_pairs


def fetch_ticker_data(pairs):
    """Get ticker info for a list of pairs (up to 100 at a time)"""
    url = "https://api.kraken.com/0/public/Ticker"
    chunk_size = 100
    all_data = {}
    for i in range(0, len(pairs), chunk_size):
        chunk = ",".join(pairs[i : i + chunk_size])
        try:
            resp = requests.get(url, params={"pair": chunk}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise KrakenAPIError(f"Error fetching ticker data: {exc}") from exc
        if data.get("error"):
            raise KrakenAPIError(f"Kraken API error for chunk {chunk}: {data['error']}")
        try:
            all_data.update(data["result"])
        except KeyError as exc:
            raise KrakenAPIError("Malformed response: missing 'result'") from exc
    return all_data


def exclude_stable_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where the base asset is a stablecoin."""
    base_assets = df["symbol"].str[:-3]
    return df[~base_assets.isin(STABLE_COINS)]


def main():
    parser = argparse.ArgumentParser(
        description="Fetch top 24h losers (biggest price drops) on Kraken.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top losers to display/save",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=1000,
        help="Minimum 24h volume to include",
    )
    parser.add_argument(
        "--include-stables",
        action="store_true",
        help="Include known stablecoins in results",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        usd_pairs = get_usd_pairs()
        logging.info("Found %d USD pairs on Kraken.", len(usd_pairs))
        ticker = fetch_ticker_data(usd_pairs)
    except KrakenAPIError as exc:
        raise SystemExit(str(exc)) from exc

    rows = []
    for sym, vals in ticker.items():
        try:
            open_ = float(vals["o"])
            last = float(vals["c"][0])
            volume = float(vals["v"][1])  # 24h rolling volume
            pct_change = 100 * (last - open_) / open_
        except (KeyError, TypeError, ValueError) as exc:
            logging.debug("Skipping %s due to invalid data: %s", sym, exc)
            continue
        rows.append(
            {
                "symbol": sym,
                "open": open_,
                "last": last,
                "pct_change": pct_change,
                "volume": volume,
            }
        )

    df = pd.DataFrame(rows)
    if not args.include_stables:
        df = exclude_stable_pairs(df)
    df = df[df["volume"] >= args.min_volume]
    df = df.sort_values("pct_change").reset_index(drop=True)

    print(f"\nTop {args.top_n} 24h losers (by %):")
    print(df[["symbol", "pct_change", "volume"]].head(args.top_n))

    # Optionally, export to CSV
    df.head(args.top_n).to_csv("kraken_top_losers.csv", index=False)
    print("\nExported to kraken_top_losers.csv.")


if __name__ == "__main__":
    main()

