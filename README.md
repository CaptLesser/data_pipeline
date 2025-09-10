# data_pipeline

Python-based crypto quant research toolkit that operates on minute-level OHLCVT data stored in MySQL and generates multi-timeframe leaderboards, downstream clustering analyses, and helper artifacts.

The initial assumption is that you already have 1-minute OHLCVT bars in a MySQL table (see Prerequisites) and want to explore leaders/laggards and their behavioral profiles.

## Features
- Leaderboards from MySQL minute bars (`leaderboards.py`): habitual gainers, losers, and overlaps across 6h/24h/72h/168h windows with skew tagging and summaries.
- IMF-based clustering on leader cohorts (`imf_cluster_{gainers,losers,overlaps}.py`): EMD cycle extraction + GMM/DBSCAN clustering with JSON and text summaries.
- Kraken 24h top losers (`kraken_losers.py`): fetch USD-quoted pairs, compute daily % change, export CSV for quick research.
- Fetch 30-day history for losers (`fetch_loser_history.py`): read tickers from CSV and pull last 30 days of MySQL OHLCVT.
- Cohort metrics (CSV-based, leaderboard-aligned): compute multi-window technical features per symbol directly from `habitual_{gainers,losers,overlaps}.csv`.

## Prerequisites
- MySQL with table `ohlcvt` containing at least: `symbol` (VARCHAR), `timestamp` (DATETIME, UTC), `open`, `high`, `low`, `close`, `volume` (numeric), optionally `trades`.
- Indexing: for performance, create an index on `(symbol, timestamp)`.
- 30 days of reasonably complete minute data for leaderboard generation.
- Python 3.9+ and packages from `requirements.txt`. Additional packages are needed for the IMF clustering scripts (see below).

## Installation

1) Clone and install base dependencies

```sh
git clone https://github.com/CaptLesser/data_pipeline
cd data_pipeline
pip install -r requirements.txt
```

2) Extras for IMF clustering (if you plan to run those scripts):

```sh
pip install PyEMD scikit-learn scipy kneed
```

## Usage

### 1) Generate Leaderboards from MySQL
Computes disjoint-window metrics, ranks top-N per bucket, aggregates habitual gainers/losers/overlaps, tags skew, and writes a summary plus full time series CSVs for the top cohorts.

```sh
python leaderboards.py \
  --host <MYSQL_HOST> \
  --user <MYSQL_USER> \
  --database <DB_NAME> \
  [--port 3306] [--password <PASSWORD>] [--top-n 20]
```

If args are omitted, the script will prompt interactively. Outputs:
- `leaderboards_summary.json`
- `habitual_losers.csv`, `habitual_gainers.csv`, `habitual_overlaps.csv` (full 30-day timeseries for qualifying coins)

Tips:
- Ensure `ohlcvt` is in UTC and indexed by `(symbol, timestamp)`.

### 2) IMF Clustering on Leader Cohorts
Takes the CSVs produced by leaderboards and performs EMD decomposition and clustering, emitting JSON cluster summaries and a compact text file.

Run one (or all) of:

```sh
python imf_cluster_gainers.py
python imf_cluster_losers.py
python imf_cluster_overlaps.py
```

Inputs: `habitual_gainers.csv`, `habitual_losers.csv`, `habitual_overlaps.csv` (must include `symbol,timestamp,close,volume`).

Outputs:
- `imf_clusters_gainers.json` and `imf_clusters_gainers.txt`
- `imf_clusters_losers.json` and `imf_clusters_losers.txt`
- `imf_clusters_overlaps.json` and `imf_clusters_overlaps.txt`

Notes:
- Requires extra packages: `PyEMD`, `scikit-learn`, `scipy`, `kneed`.
- Uses a minimum sample threshold and may skip a coin/window if insufficient data are present.

### 3) Fetch Top 24h Losers from Kraken
Fetch USD-quoted pairs, compute 24h % change, filter by minimum volume, optionally exclude stables.

```sh
python kraken_losers.py [--top-n 20] [--min-volume 1000] [--include-stables]
```

Output: `kraken_top_losers.csv` and a printed table.

### 4) Fetch 30-Day History for Losers from MySQL
Reads symbols from `kraken_top_losers.csv` and saves the last 30 days of OHLCVT history.

Configuration via environment variables (with defaults):

```sh
export DB_HOST=localhost
export DB_USER=YOUR_USER
export DB_PASSWORD=YOUR_PASS
export DB_NAME=YOUR_DB
export LOSERS_CSV=kraken_top_losers.csv
export HISTORY_OUTPUT_CSV=losers_30day_history.csv
export OHLCVT_TABLE=ohlcvt

python fetch_loser_history.py [--host ... --user ... --database ... --port 3306 --password ...] \
  [--input-csv kraken_top_losers.csv] [--output-csv losers_30day_history.csv]

Notes:
- You can provide DB credentials via flags as above, via environment variables, or omit them to be prompted interactively.
```

### 4b) Fetch 30-Day History for Gainers from MySQL
Reads symbols from `habitual_gainers.csv` and saves the last 30 days of OHLCVT history.

Environment variables (defaults shown):

```sh
export DB_HOST=localhost
export DB_USER=YOUR_USER
export DB_PASSWORD=YOUR_PASS
export DB_NAME=YOUR_DB
export GAINERS_CSV=habitual_gainers.csv
export GAINERS_HISTORY_OUTPUT_CSV=gainers_30day_history.csv
export OHLCVT_TABLE=ohlcvt

python fetch_gainers_history.py [--host ... --user ... --database ... --port 3306 --password ...] \
  [--input-csv habitual_gainers.csv] [--output-csv gainers_30day_history.csv]

Notes:
- Credentials accepted via CLI flags/env; any missing values will be prompted.
```

### 4c) Fetch 30-Day History for Overlaps from MySQL
Reads symbols from `habitual_overlaps.csv` and saves the last 30 days of OHLCVT history.

Environment variables (defaults shown):

```sh
export DB_HOST=localhost
export DB_USER=YOUR_USER
export DB_PASSWORD=YOUR_PASS
export DB_NAME=YOUR_DB
export OVERLAPS_CSV=habitual_overlaps.csv
export OVERLAPS_HISTORY_OUTPUT_CSV=overlaps_30day_history.csv
export OHLCVT_TABLE=ohlcvt

python fetch_overlaps_history.py [--host ... --user ... --database ... --port 3306 --password ...] \
  [--input-csv habitual_overlaps.csv] [--output-csv overlaps_30day_history.csv]

Notes:
- Credentials accepted via CLI flags/env; any missing values will be prompted.
```

### 5) Cohort Metrics (CSV-based)
Compute per-symbol technical metrics for each leaderboard cohort using the CSVs generated by `leaderboards.py`. Optionally, enrich the current metrics with asset-specific historical baselines pulled from MySQL to classify how far from typical the current state is (percentile + quintile).

Outputs are compact per-symbol metric summaries with multi-window features across 6h/24h/72h/168h. When DB credentials are provided, outputs also include `_pctile`, `_quintile`, `_p25/_p50/_p75`, `_n`, and `_direction` per metric.

Run one (or all) of:

```sh
# Overlaps
python -m cohort_metrics.overlaps.metrics --input habitual_overlaps.csv --output overlap_metrics.csv \
  [--months 3] [--host ... --user ... --database ... --port 3306 --password ...] [--table ohlcvt] [--no-baseline]

# Gainers
python -m cohort_metrics.gainers.metrics  --input habitual_gainers.csv  --output gainers_metrics.csv \
  [--months 3] [--host ... --user ... --database ... --port 3306 --password ...] [--table ohlcvt] [--no-baseline]

# Losers
python -m cohort_metrics.losers.metrics   --input habitual_losers.csv   --output losers_metrics.csv \
  [--months 3] [--host ... --user ... --database ... --port 3306 --password ...] [--table ohlcvt] [--no-baseline]
```

Notes:
- Inputs must include columns: `symbol,timestamp,open,high,low,close,volume`.
- Features include mean/std/min/max close, VWAP, price deviation vs VWAP, ROC, log returns, EMA/SMA, MACD (+signal/hist), Bollinger (upper/lower/width), RSI, ATR, Stochastic (%K/%D), OBV, CCI, ADX — computed on the last N minutes for each window.
- Baselines are computed per-asset from non-overlapping windows aligned to UTC (6h/24h/72h/168h) over the last N months (default 3). Incomplete windows are skipped; edge cases with limited history are flagged via sample counts (`*_n`).
- Outputs are single-row-per-symbol summaries, with columns suffixed by window (e.g., `rsi_24h`, `macd_histogram_6h`). When baselines are enabled, extra columns such as `rsi_24h_pctile`, `rsi_24h_quintile`, `rsi_24h_p50`, `rsi_24h_n`, and `rsi_24h_direction` are included.

### 6) Cohort Metrics (CSV-based)
Compute per-symbol technical metrics for each leaderboard cohort using the CSVs generated by `leaderboards.py`. These scripts do not require database access and operate purely on the cohort CSVs.

Outputs are compact per-symbol metric summaries with multi-window features across 6h/24h/72h/168h.

Run one (or all) of:

```sh
python -m cohort_metrics.overlaps.metrics --input habitual_overlaps.csv --output overlap_metrics.csv
python -m cohort_metrics.gainers.metrics  --input habitual_gainers.csv  --output gainers_metrics.csv
python -m cohort_metrics.losers.metrics   --input habitual_losers.csv   --output losers_metrics.csv
```

Notes:
- Inputs must include columns: `symbol,timestamp,open,high,low,close,volume`.
- Features include mean/std/min/max close, VWAP, price deviation vs VWAP, ROC, log returns, EMA/SMA, MACD (+signal/hist), Bollinger (upper/lower/width), RSI, ATR, Stochastic (%K/%D), OBV, CCI, ADX — computed on the last N minutes for each window.
- Outputs are single-row-per-symbol summaries, with columns suffixed by window (e.g., `rsi_24h`, `macd_histogram_6h`).

## Data Flow Overview
MySQL OHLCVT (minute bars) → Leaderboards (`leaderboards.py`) → Cohort CSVs → IMF Clustering scripts → Cluster JSON/TXT summaries.

Optionally, use `kraken_losers.py` to seed a quick losers list directly from exchange data and `fetch_loser_history.py` to hydrate DB time series for those symbols.

## Testing

Run the unit tests with [pytest](https://docs.pytest.org/):

```sh
pytest
```

Current tests cover the Kraken module and are network-free (API responses are mocked).

## Notes & Recommendations
- Keep `ohlcvt` timestamps in UTC. The leaderboards logic uses UTC window boundaries.
- Add/verify the `(symbol, timestamp)` index for fast scans and resampling.
- The IMF clustering steps can be CPU-intensive; prefer running after leaderboards narrow the universe.

## Contributing

Issues and PRs welcome.
