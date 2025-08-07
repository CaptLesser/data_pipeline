"""Tests for the kraken_losers module.

The network-dependent functions are patched so the suite can run without
external HTTP requests.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest
import requests

import kraken_losers as kl

def test_exclude_stable_pairs_filters_only_stables():
    """Stablecoin pairs should be removed from the DataFrame."""
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSD", "USDTUSD", "ETHUSD", "DAIUSD"],
            "volume": [10000, 10000, 10000, 10000],
        }
    )

    filtered = kl.exclude_stable_pairs(df)

    assert set(filtered["symbol"]) == {"BTCUSD", "ETHUSD"}

def test_get_usd_pairs_fetches_usd_pairs(monkeypatch):
    """Only trading pairs quoted in USD should be returned."""

    def mock_get(url, timeout):
        class MockResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "error": [],
                    "result": {
                        "XXBTZUSD": {"wsname": "BTC/USD"},
                        "XETHZEUR": {"wsname": "ETH/EUR"},
                        "XETHZUSD": {"wsname": "ETH/USD"},
                        "USDTZUSD": {"wsname": "USDT/USD"},
                    },
                }

        return MockResp()

    monkeypatch.setattr(kl.requests, "get", mock_get)

    assert set(kl.get_usd_pairs()) == {"XXBTZUSD", "XETHZUSD", "USDTZUSD"}


def test_get_usd_pairs_raises_on_api_error(monkeypatch):
    """SystemExit should be raised if Kraken returns an API error."""

    def mock_get(url, timeout):
        class MockResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"error": ["EGeneral:Internal error"]}

        return MockResp()

    monkeypatch.setattr(kl.requests, "get", mock_get)

    with pytest.raises(SystemExit):
        kl.get_usd_pairs()


def test_fetch_ticker_data_merges_results(monkeypatch):
    """Ticker data for each pair is combined into a single dictionary."""

    def mock_get(url, params, timeout):
        assert params["pair"] == "BTCUSD,ETHUSD"

        class MockResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "error": [],
                    "result": {"BTCUSD": {"o": "1"}, "ETHUSD": {"o": "2"}},
                }

        return MockResp()

    monkeypatch.setattr(kl.requests, "get", mock_get)

    data = kl.fetch_ticker_data(["BTCUSD", "ETHUSD"])

    assert data == {"BTCUSD": {"o": "1"}, "ETHUSD": {"o": "2"}}


def test_fetch_ticker_data_raises_on_request_error(monkeypatch):
    """SystemExit should be raised when the HTTP request fails."""

    def mock_get(*args, **kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr(kl.requests, "get", mock_get)

    with pytest.raises(SystemExit):
        kl.fetch_ticker_data(["BTCUSD"])
