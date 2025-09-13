# data_pipeline

Python-based crypto quant research toolkit that operates on minute-level OHLCVT data stored in MySQL and generates multi-timeframe leaderboards, downstream clustering analyses, and helper artifacts.

The initial assumption is that you already have 1-minute OHLCVT bars in a MySQL table (see Prerequisites) and want to explore leaders/laggards and their behavioral profiles.

## Features
- Leaderboards from MySQL minute bars (`leaderboards.py`): habitual gainers, losers, and overlaps across 1h/4h/12h/24h windows (optionally 72h) with skew tagging and summaries.
- Intraday State & Daily Regime snapshots (CLIs): compute resample-first, window-aligned features with baseline enrichment, neutral imputation, and Parquet output (partitioned by UTC date).
 - IMF Stage (CLI): CEEMDAN-based cycle decomposition of price/volume, feature extraction, per-series clustering (HDBSCAN + KMeans fallback), and reliability via MAD%, with Parquet + JSON outputs.
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
pip install EMD-signal scikit-learn scipy kneed
```

## Usage

### Recommended Sequence (Overlap Analysis)

1) Install dependencies

```sh
pip install -r requirements.txt
```

2) Generate leaderboards (produce `habitual_overlaps.csv`)

```sh
python leaderboards.py \
  --host <MYSQL_HOST> \
  --user <MYSQL_USER> \
  --database <DB_NAME> \
  --port 3306 \
  --top-n 20
```

- Outputs: `leaderboards_summary.json`, `habitual_overlaps.csv` (plus gainers/losers CSVs)
- If flags are omitted, the script prompts interactively (no password flag required).

3) Quantify overlap behaviors (metrics + baselines)

```sh
python -m cohort_metrics.overlaps.metrics \
  --input habitual_overlaps.csv \
  --output overlap_metrics.csv \
  --months 3 \
  --host <MYSQL_HOST> --user <MYSQL_USER> --database <DB_NAME> --port 3306 --table ohlcvt
```

- Output: `overlap_metrics.csv` with multi-window features and baseline enrichment.
- Omitting `--password` will prompt securely. To skip baselines, add `--no-baseline`.

4) IMF clustering on overlaps (cycle profiles)

```sh
python imf_cluster_overlaps.py
```

- Outputs: `imf_clusters_overlaps.json`, `imf_clusters_overlaps.txt` (and `pipeline.log`).

Optional (joinable context):

```sh
python -m cohort_metrics.state  --input habitual_overlaps.csv --out data/state  --windows 1h,4h,12h,24h
python -m cohort_metrics.regime --input habitual_overlaps.csv --out data/regime --windows 3d,7d,14d,30d,90d
```

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
Compute per-symbol technical metrics for each leaderboard cohort using the CSVs generated by `leaderboards.py`. Optionally, enrich the current metrics with asset-specific historical baselines pulled from MySQL to classify how far from typical the current state is (percentile + quintile + robust z).

Outputs are compact per-symbol metric summaries with multi-window features across 1h/4h/12h/24h (optional 72h). When DB credentials are provided, outputs also include `_pctile` (0â€“100), `_quintile` (1â€“5), `_p25/_p50/_p75`, `_n`, `_direction`, plus `_rz` and `_abs_rz` per metric.

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
- Features include mean/std/min/max close, VWAP, price deviation vs VWAP, ROC, log returns, EMA/SMA, MACD (+signal/hist), Bollinger (upper/lower/width), RSI, ATR, Stochastic (%K/%D), OBV, CCI, ADX ” computed on the last N minutes for each window.
- Baselines are computed per-asset from non-overlapping windows aligned to UTC (1h/4h/12h/24h, optional 72h) over the last N months (default 3). Incomplete windows are flagged via `window_coverage_*` and `window_ok_*`.
- Outputs are single-row-per-symbol summaries, with columns suffixed by window (e.g., `rsi_24h`, `macd_histogram_4h`). When baselines are enabled, extra columns such as `rsi_24h_pctile`, `rsi_24h_quintile`, `rsi_24h_p50`, `rsi_24h_n`, `rsi_24h_rz`, and `rsi_24h_direction` are included.

### 6) Intraday State Snapshot (CLI)

Compute current, resample-first, window-aligned metrics on intraday bars with optional baseline enrichment and neutral imputation.

Usage:

```sh
python -m cohort_metrics.state \
  --input habitual_overlaps.csv \
  [--windows 1h,4h,12h,24h] [--enable-72h] \
  [--coverage-threshold 0.8] [--asof 2024-05-18T12:00:00Z] \
  [--out data/state] [--compression snappy|zstd] [--schema-version v1] [--dry-run] \
  [--months 3 --host ... --user ... --database ... --port 3306 --password ... --table ohlcvt --no-baseline]
```

Details:

- Windows: defaults 1h,4h,12h,24h; 72h is optional and off by default.
- Resample-first: 1h+1m, 4h+3m, 12h+10m, 24h+20m (72h+60m).
- Alignment: uses last complete UTC bucket per window; anchor `timestamp_asof_utc` is min end-bucket across selected windows or `--asof` if provided.
- Coverage flags: emits `window_coverage_*` and `window_ok_*` per window.
- Opportunity primitives: bb_position, squeeze_index, realized vol family (rv_close/parkinson/garman_klass/yang_zhang), range_intensity, dist_to_window_high/low and abs, ret_share_pos, run_up_max/run_dn_max, vwap_pos_share, vol_spike_share, quote_volume_sum, up/down volume shares.
- Indicators: Wilder RSI/ATR, classic ADX/DMI, Bollinger (ddof=1), stochastic D = SMA(K,3).
- Baselines & percentiles: per (symbol, window, metric) store p01,p05,p10,p20,p25,p40,p50,p60,p75,p80,p90,p95,p99,n. `_pctile` is 0â€“100 via piecewise-linear interpolation between knots in transform space (log1p for positive, ASL for signed). Robust z: `_rz`, `_abs_rz`.
- Enrichment skips non-signals: no `_pctile/_rz` for `window_*`, `metrics_valid_*`, `sma_*`, `ema_fast_*`, `ema_slow_*`, or `macd_*` except `macd_histogram_*`.
- Validity flags: per-family `metrics_valid_*` plus rollup `_valid` flags.
- Output: Parquet under `dt=YYYY-MM-DD/state_snapshot.parquet` with compression (`snappy` default) and sidecar schema JSON.
- Deterministic: same inputs + `--asof` produce identical outputs.

Imputation (always numeric output):

- Percentile: `*_pctile=50.0`, mask `*_pctile_imputed=1` if imputed.
- Robust-z: `*_rz=0.0`, `*_abs_rz=0.0` with `*_rz_imputed`, `*_abs_rz_imputed`.
- Quintile: `*_quintile=3` with `*_quintile_imputed`.
- Positions: `*_pos=0.5` with `*_pos_imputed`.
- Directional: `*_dir=0.0` with `*_dir_imputed`.
- Ratios: `*_over_* = 1.0` with `*_imputed`.
- Differences: `*_minus_* = 0.0` with `*_imputed`.
- ATR-scaled distances: `dist_*_mag = 0.0` with `*_imputed`.
- Magnitudes: `*_mag = p50` (if enriched) else `0.0`, with `*_mag_imputed`.

Schema sidecar describes each column's dtype, family tag, and window.

### 7) Daily Regime Snapshot (CLI)

Compute slow-context, math-only daily features across configurable windows with optional baseline enrichment and neutral imputation.

Usage:

```sh
python -m cohort_metrics.regime \
  --input habitual_overlaps.csv \
  [--windows 3d,7d,14d,30d,90d] [--breakout-lookback-days 20] \
  [--asof 2024-05-18] [--out data/regime] [--compression snappy|zstd] [--schema-version v1] [--dry-run]
```

Details:

- Cadence: daily (UTC), resampled from intraday when needed.
- Windows: defaults 3d,7d,14d,30d,90d.
- Breakout lookback: default 20 UTC calendar days, configurable.
- Features: ROC, realized vol, vol-of-vol, return skew/kurtosis, MA50/MA200 slopes and cross, above-MA share, ATR/ADX, Bollinger squeeze, range intensity, breakout shares.
- Alignment: `timestamp_asof_utc` is the chosen UTC date (last daily as-of).
- Output: Parquet under `dt=YYYY-MM-DD/regime_snapshot.parquet` with sidecar schema JSON.
- Join contract: join with Intraday State on `(symbol, timestamp_asof_utc)`.

Imputation follows the same neutral rules as Intraday State.

### 8) IMF Stage (CLI)

Decompose minute-level price and volume for a leaders set using CEEMDAN, detect robust cycles (Hilbert + extrema with Â¼-period edge guard and a hard depth gate), build per-IMF features, and cluster per series (price, volume) â€” or as a mixed pool if requested. Emits join-friendly Parquet and cluster â€œcardsâ€ JSON.

Usage:

```sh
python -m cohort_metrics.imf_stage \
  --leaders-csv leaders.csv \
  --asof 2024-05-18T12:00:00Z \
  --lookback-months 3 \
  --host <MYSQL_HOST> --user <MYSQL_USER> --database <DB_NAME> --password <PASSWORD> [--port 3306] [--table ohlcvt] \
  [--persist-imfs] [--cluster-mixed] [--n-jobs 7] [--max-seconds-per-asset 30] \
  [--min-cycle-depth-pct 2.0] [--detrend] [--no-winsorize] [--out data/imf]
```

Key details:

- Inputs: leaders from `--leaders-csv` (column `symbol`) or `--leaders` list. Minute OHLCV fetched with the same `lookback_months` as metrics.
- Windowing: `asof` anchors the end; window_start = `asof - lookback_months` (calendar months). Strict 1-min UTC grid; de-dupe by last; `close` ffill/backfill at window start; `volume` zeros for gaps.
- Transforms: price `log(close)` with optional robust Theil-Sen detrend; volume winsorize raw at [1,99] then `log1p`.
- Decomposition: CEEMDAN `trials=100, seed=1337`; if `--max-seconds-per-asset` is exceeded (and trials > 20), reruns once with `trials=20` and sets `status=time_budget_reduced`.
- Cycle gate: Hilbert + extrema must pass; hard minimum depth `--min-cycle-depth-pct` (default 2.0) on back-transformed midline % depth; min period bars=3; max period <= 2/3 of window bars; edge guard = 1/4 median period for summaries.
- Features per IMF: `period_days`, `amp_pct`, `energy_share`, `cycles_count`, `pct_cycles_ge_2pct`, `phase_var`. For price IMFs, fusion to closest-volume IMF adds `vol_amp_corr` (Hilbert envelope corr), `vol_period_ratio`, `vol_energy_share`.
- Clustering: per-series pools by default (price->GP####, volume->GV####); `--cluster-mixed` for one pool (GM####). HDBSCAN (leaf), noise as unique singletons; fallback to KMeans (best silhouette). Per-asset merge via Ward on ln(period) to A:<SYMBOL>:####.
- Reliability: MAD% (period, amp) and `exp(-MAD%/25)` scores; singletons keep MAD%=NaN and reliability=0.0.
- Outputs (under `data/imf/dt=YYYY-MM-DD/`):
  - `imf_summary.parquet` (snappy) - one row per kept IMF with features, labels, and redundant cluster stats (global/asset).
  - Global cards: `cluster_cards_global_price.json` and `cluster_cards_global_volume.json` (or `cluster_cards_global_mixed.json` when `--cluster-mixed`).
  - Asset cards: `cluster_cards_asset.json` per symbol.
  - Optional IMFs: `imfs/<SYMBOL>_{price,volume}.npz` when `--persist-imfs`.
  - `run_config.json` echoing parameters, seeds, library versions, scaler metadata (center/scale/features per pool), and bars coverage per symbol (`bars_before`, `bars_after`, `missing_share`).

Notes:

- Deterministic by seeds; CEEMDAN runs single-threaded for reproducibility; parallelism is per symbol (`--n-jobs`).
- "Include everything": once an IMF passes the gate and period sanity, it's clustered and emitted (singletons allowed).

### 9) IMF Post-Processing (CLI)

Summarize the IMF stage into compact, LLM/ML-friendly tables per symbol/series with reliability-adjusted dominant cycles, weighted aggregates, interpretability bands, and templated text.

Usage:

```sh
python -m cohort_metrics.imf_postprocess \
  --summary data/imf/dt=YYYY-MM-DD/imf_summary.parquet \
  [--cards-price .../cluster_cards_global_price.json] \
  [--cards-volume .../cluster_cards_global_volume.json] \
  [--cards-asset .../cluster_cards_asset.json] \
  [--run-config .../run_config.json] \
  [--series price|volume|both] [--asof 2024-05-18T12:00:00Z] [--window-days 90] \
  [--min-energy-share 0.0] [--amp-bands 5 12] \
  [--format parquet|csv] [--out-dir data/imf/dt=YYYY-MM-DD/]
```

Details:

- Drops only rows with `status != 'ok'`; preserves long/rare cycles. Flags per IMF: `singleton_global_flag`, `long_cycle_flag`, `undersampled_flag`.
- Reliability: blends cluster MAD% (period/amp) with undersampling shrinkage and a gentle size boost; missing stats fall back to neutral 0.5.
- Dominant cycle (per symbol/series): score = `reliability * (0.6*energy_share + 0.4*min(1, amp_pct/20))`; ties -> higher amp, then shorter period.
- Aggregates: energy-weighted period/amp; diversity via unweighted MAD% of periods.
- Amplitude bands: small <5%, medium 5-12%, large >12% (configurable).
- Outputs (series retained when `--series both`):
  - `imf_post_summary.parquet` - one row per symbol/series with aggregates, dominant IMF, flags, and `text_summary` (<=120 chars).
  - `imf_post_top_imfs.parquet` - top 3 IMFs per symbol/series with key fields for analysis and QA.

Optional Parquet snapshots:
- Intraday State: `data/state/{UTC_date}/state_snapshot.parquet`
- Daily Regime: `data/regime/{UTC_date}/regime_snapshot.parquet`

See helpers in `cohort_metrics/state.py` and `cohort_metrics/regime.py`. If Parquet engines are not installed, a CSV fallback is written.

### 6) Cohort Metrics (CSV-based)
Compute per-symbol technical metrics for each leaderboard cohort using the CSVs generated by `leaderboards.py`. These scripts do not require database access and operate purely on the cohort CSVs.

Outputs are compact per-symbol metric summaries with multi-window features across 1h/4h/12h/24h (optional 72h).

Run one (or all) of:

```sh
python -m cohort_metrics.overlaps.metrics --input habitual_overlaps.csv --output overlap_metrics.csv
python -m cohort_metrics.gainers.metrics  --input habitual_gainers.csv  --output gainers_metrics.csv
python -m cohort_metrics.losers.metrics   --input habitual_losers.csv   --output losers_metrics.csv
```

Notes:
- Inputs must include columns: `symbol,timestamp,open,high,low,close,volume`.
 - Features include mean/std/min/max close, VWAP, price deviation vs VWAP, ROC, log returns, EMA/SMA, MACD (+signal/hist), Bollinger (upper/lower/width), RSI, ATR, Stochastic (%K/%D), OBV, CCI, ADX, computed on the last N minutes for each window.
- Outputs are single-row-per-symbol summaries, with columns suffixed by window (e.g., `rsi_24h`, `macd_histogram_4h`).

## Data Flow Overview
MySQL OHLCVT (minute bars) -> Leaderboards (`leaderboards.py`) -> Cohort CSVs -> IMF Clustering scripts -> Cluster JSON/TXT summaries.

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









