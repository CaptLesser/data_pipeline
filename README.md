# data_pipeline

Python-based crypto quant research toolkit that operates on minute-level OHLCVT data stored in MySQL and generates multi-timeframe leaderboards, downstream clustering analyses, and helper artifacts.

The initial assumption is that you already have 1-minute OHLCVT bars in a MySQL table (see Prerequisites) and want to explore leaders/laggards and their behavioral profiles.

## Features
- Leaderboards from MySQL minute bars (`leaderboards.py`): habitual gainers, losers, and overlaps across 6h/24h/72h/168h windows with skew tagging and summaries.
- IMF-based clustering on leader cohorts (`imf_cluster_{gainers,losers,overlaps}.py`): EMD cycle extraction + GMM/DBSCAN clustering with JSON and text summaries.
- Kraken 24h top losers (`kraken_losers.py`): fetch USD-quoted pairs, compute daily % change, export CSV for quick research.
- Fetch 30-day history for losers (`fetch_loser_history.py`): read tickers from CSV and pull last 30 days of MySQL OHLCVT.
- Daily metrics pipeline scaffold (`metrics.py`): per-(date,symbol) multi-window technical metrics table with backfill + live updates (work in progress).

## Prerequisites
- MySQL with table `ohlcvt` containing at least: `symbol` (VARCHAR), `timestamp` (DATETIME, UTC), `open`, `high`, `low`, `close`, `volume` (numeric), optionally `trades` (used by `metrics.py`).
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

python fetch_loser_history.py
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

python fetch_gainers_history.py
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

python fetch_overlaps_history.py
```

### 5) Daily Metrics Pipeline (WIP)
Computes a large set of per-window technical features into `ohlc_metrics_daily` with backfill + live intraday/EOD updates.

Status: scaffolding present, including table creation and computation logic. Database credentials are currently expected to be provided in code (see TODO in `metrics.py`).

High-level responsibilities:
- Ensure table exists (`ohlc_metrics_daily`) with many per-window columns.
- Backfill: for each symbol, compute historical daily snapshots using the full time series.
- Live loop: intraday partial updates for due windows; EOD closeout recomputes yesterday fully and advances `state.json`.

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
