import pandas as pd
import pytest
import requests
from kraken_losers import (
    KrakenAPIError,
    exclude_stable_pairs,
    fetch_ticker_data,
    get_usd_pairs,
)


def test_exclude_stable_pairs_filters_only_stables():
    df = pd.DataFrame({
        "symbol": ["BTCUSD", "USDTUSD", "ETHUSD", "DAIUSD"],
        "volume": [10000, 10000, 10000, 10000],
    })

    filtered = exclude_stable_pairs(df)

    assert set(filtered["symbol"]) == {"BTCUSD", "ETHUSD"}


def test_get_usd_pairs_raises_on_request_error(monkeypatch):
    def fake_get(*args, **kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(KrakenAPIError):
        get_usd_pairs()


def test_fetch_ticker_data_raises_on_api_error(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"error": ["EQuery:Unknown asset pair"]}

    def fake_get(*args, **kwargs):
        return FakeResp()

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(KrakenAPIError):
        fetch_ticker_data(["BTCUSD"])
