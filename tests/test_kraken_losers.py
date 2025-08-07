import pandas as pd
from kraken_losers import exclude_stable_pairs


def test_exclude_stable_pairs_filters_only_stables():
    df = pd.DataFrame({
        "symbol": ["BTCUSD", "USDTUSD", "ETHUSD", "DAIUSD"],
        "volume": [10000, 10000, 10000, 10000],
    })

    filtered = exclude_stable_pairs(df)

    assert set(filtered["symbol"]) == {"BTCUSD", "ETHUSD"}
