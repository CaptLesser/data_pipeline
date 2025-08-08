#!/usr/bin/env python3
import json
import logging
from datetime import timedelta
import numpy as np
import pandas as pd
from PyEMD import EMD
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import functools
from typing import List, Dict, Any

# ─── CONFIG ─────────────────────────────────────────────────────────────

# Load leaderboard data directly from CSV (no database dependency).
DF = pd.read_csv('habitual_overlaps.csv', parse_dates=['timestamp'])
COINS = DF['symbol'].unique().tolist()

# Time windows (in days) to analyze: 6h, 12h, 24h, 3d, 7d, 30d
TARGET_DAYS = [0.25, 0.5, 1, 3, 7, 30]
WINDOW_LABELS = {
    0.25: '6h',
    0.5:  '12h',
    1:    '24h',
    3:    '3d',
    7:    '7d',
    30:   '30d'
}

MIN_AMPLITUDE_PCT  = 2.0
# Maximum number of GMM components to try for each window
GMM_MAX_COMPONENTS = {0.25: 5, 0.5: 5, 1: 5, 3: 10, 7: 10, 30: 15}
# DBSCAN grid search ranges; these can be modified if needed.
DBSCAN_MIN_SAMPLES_RANGE = range(2, 8)  # trying min_samples between 2 and 7 inclusive.
DBSCAN_EPS_PERCENTILES   = (10, 90)     # Use 10th to 90th percentile of kth distances for eps grid.
DBSCAN_EPS_GRID_SIZE     = 10           # Number of eps candidates.
MIN_REQUIRED_POINTS      = 100

OUTPUT_FILE = 'imf_clusters_overlaps.json'
LOG_FILE    = 'pipeline.log'

# ─── LOGGING ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger('pipeline')

def log_enter_exit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug("ENTER %s: args=%s kwargs=%s", func.__name__, args, kwargs)
        try:
            res = func(*args, **kwargs)
        except Exception:
            logger.exception("EXCEPTION in %s", func.__name__)
            raise
        size = None
        try:
            size = res.shape if hasattr(res, 'shape') else len(res)
        except Exception:
            pass
        logger.debug("EXIT %s: returned %s (size=%s)", func.__name__,
                     type(res).__name__, size)
        return res
    return wrapper

# ─── FETCH & PAD SERIES ───────────────────────────────────────────────────
@log_enter_exit
def fetch_clean_series(symbol: str) -> pd.Series:
    """Fetch closing-price series at 1-min cadence from the CSV dataframe."""
    symbol_df = DF[DF['symbol'] == symbol].sort_values('timestamp')
    if symbol_df.empty:
        return pd.Series(dtype=float)
    full_idx = pd.date_range(start=symbol_df['timestamp'].min(),
                             end=symbol_df['timestamp'].max(),
                             freq='min')
    symbol_df = symbol_df.set_index('timestamp').reindex(full_idx)
    symbol_df['close'] = symbol_df['close'].ffill().bfill().astype(float)
    return symbol_df['close']

@log_enter_exit
def fetch_volume_series(symbol: str) -> pd.Series:
    """Fetch volume series at 1-min cadence from the CSV dataframe."""
    symbol_df = DF[DF['symbol'] == symbol].sort_values('timestamp')
    if symbol_df.empty:
        return pd.Series(dtype=float)
    full_idx = pd.date_range(start=symbol_df['timestamp'].min(),
                             end=symbol_df['timestamp'].max(),
                             freq='min')
    symbol_df = symbol_df.set_index('timestamp').reindex(full_idx)
    symbol_df['volume'] = symbol_df['volume'].fillna(0).astype(float)
    return symbol_df['volume']

# ─── DECOMPOSITION HELPERS ────────────────────────────────────────────────
def decompose_to_imf_dfs(x: pd.Series, value_col: str) -> List[pd.DataFrame]:
    """
    Run EMD on x.values and return a list of DataFrames with columns:
      ['timestamp','imf_idx', <value_col>]
    """
    imfs  = EMD()(x.values)
    times = x.index
    out   = []
    for i, arr in enumerate(imfs):
        out.append(pd.DataFrame({
            'timestamp': times,
            'imf_idx':   i,
            value_col:   arr
        }))
    return out

def extract_cycles(df_imf: pd.DataFrame, val_col: str,
                   ref_series: pd.Series = None) -> List[Dict[str,Any]]:
    """
    Extract cycles from an IMF DataFrame via peak-to-peak intervals.
    If a reference series is provided, amplitude_pct is computed using the 
    original series values at the start and end times.
    Otherwise, the raw IMF values are used.
    Returns a list of dicts with keys:
      imf_idx, start_time, end_time, amplitude_pct, duration_min
    """
    vals  = df_imf[val_col].values
    times = df_imf['timestamp'].values
    peaks, _ = find_peaks(vals)
    out = []
    for p0, p1 in zip(peaks, peaks[1:]):
        t0, t1 = times[p0], times[p1]
        # Compute cycle duration in minutes.
        dur_min = (t1 - t0) / np.timedelta64(1, 'm')
        if dur_min <= 0:
            continue

        # If a reference series is provided, use it for amplitude calculation.
        if ref_series is not None:
            try:
                v0 = ref_series.loc[t0]
                v1 = ref_series.loc[t1]
            except KeyError:
                continue
            if v0 == 0:
                continue
            amp = (v1 - v0) / v0 * 100.0
        else:
            v0 = vals[p0]
            v1 = vals[p1]
            if v0 == 0:
                continue
            amp = (v1 - v0) / abs(v0) * 100.0

        # Only keep cycles with a positive amplitude of at least MIN_AMPLITUDE_PCT.
        if val_col == 'imf_price' and amp < MIN_AMPLITUDE_PCT:
            continue

        out.append({
            'imf_idx':       int(df_imf.at[p0, 'imf_idx']),
            'start_time':    t0,
            'end_time':      t1,
            'amplitude_pct': float(amp),
            'duration_min':  float(dur_min)
        })
    return out

# ─── CLUSTERING ──────────────────────────────────────────────────────────
@log_enter_exit
def cluster_and_summarize(df: pd.DataFrame, days: int) -> dict:
    result = {'gmm': [], 'dbscan': []}
    if df.empty or len(df) < 2:
        logger.warning("Insufficient samples for clustering (found %d), skipping clustering.", len(df))
        return result

    # Use only the three features to preserve EMD alignment and interpretability.
    X = df[['amplitude_pct','duration_min','total_volume']].values
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    # ─── AUTOMATIC GMM CLUSTER SELECTION ──────────────────────────────
    # Set maximum number of GMM components based on configured maximum
    max_components = min(GMM_MAX_COMPONENTS.get(days, max(GMM_MAX_COMPONENTS.values())), len(Xs))
    lowest_bic = np.inf
    best_gmm = None
    best_n_components = 2
    # Try models with 2 to max_components clusters.
    for n_components in range(2, max_components + 1):
        gmm_candidate = GaussianMixture(n_components=n_components, random_state=0)
        gmm_candidate.fit(Xs)
        bic = gmm_candidate.bic(Xs)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm_candidate
            best_n_components = n_components
    if best_gmm is None:
        logger.warning("No valid GMM found, skipping GMM clustering.")
    else:
        df['gmm_cluster'] = best_gmm.predict(Xs)
        # Summarize clusters using robust statistics.
        for cl in sorted(df['gmm_cluster'].unique()):
            sub = df[df['gmm_cluster'] == cl]
            med_amp = float(sub['amplitude_pct'].median())
            med_dur = float(sub['duration_min'].median())
            med_vol = float(sub['total_volume'].median())
            # Compute the median absolute deviation (MAD)
            mad_amp = float(np.median(np.abs(sub['amplitude_pct'] - med_amp)))
            mad_dur = float(np.median(np.abs(sub['duration_min'] - med_dur)))
            mad_vol = float(np.median(np.abs(sub['total_volume'] - med_vol)))
            result['gmm'].append({
                'cluster':         int(cl),
                'count':           len(sub),
                'median_amplitude_pct': med_amp,
                'mad_amplitude_pct':    mad_amp,
                'median_duration_min':  med_dur,
                'mad_duration_min':     mad_dur,
                'median_total_volume':  med_vol,
                'mad_total_volume':     mad_vol
            })

    # ─── DYNAMIC DBSCAN CLUSTERING VIA GRID SEARCH ───────────────
    # First, compute kth nearest neighbor distances (using k=DBSCAN_MIN_SAMPLES_RANGE.start is not used here).
    # We'll use the maximum k from our min_samples grid search for initial candidate eps values.
    k_for_dist = min(max(DBSCAN_MIN_SAMPLES_RANGE), len(Xs))
    nbrs = NearestNeighbors(n_neighbors=k_for_dist).fit(Xs)
    dists, _ = nbrs.kneighbors(Xs)
    kth_dists = dists[:, -1]
    # Define eps candidate range based on specified percentiles.
    eps_lower = np.percentile(kth_dists, DBSCAN_EPS_PERCENTILES[0])
    eps_upper = np.percentile(kth_dists, DBSCAN_EPS_PERCENTILES[1])
    eps_candidates = np.linspace(eps_lower, eps_upper, DBSCAN_EPS_GRID_SIZE)

    best_score = -1
    best_params = None
    best_labels = None
    # Grid search over eps and min_samples.
    for eps in eps_candidates:
        for min_samples in DBSCAN_MIN_SAMPLES_RANGE:
            db_candidate = DBSCAN(eps=eps, min_samples=min_samples)
            candidate_labels = db_candidate.fit_predict(Xs)
            # Exclude noise points for silhouette evaluation.
            core_mask = candidate_labels != -1
            if len(set(candidate_labels[core_mask])) < 2:
                continue
            try:
                score = silhouette_score(Xs[core_mask], candidate_labels[core_mask])
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}
                best_labels = candidate_labels.copy()

    if best_params is None:
        logger.warning("Could not determine optimal DBSCAN parameters; using default knee locator eps.")
        # Fallback: use knee locator method if grid search fails.
        sorted_kth = np.sort(kth_dists)
        if len(sorted_kth) == 0:
            eps = 0.5
        elif np.allclose(sorted_kth[0], sorted_kth[-1]):
            eps = float(sorted_kth[0])
        else:
            idxs = np.arange(len(sorted_kth))
            kl = KneeLocator(idxs, sorted_kth, curve="convex", direction="increasing")
            eps = float(sorted_kth[int(kl.knee)]) if kl.knee is not None else float(np.percentile(sorted_kth, 50))
        best_params = {'eps': eps, 'min_samples': 2}
        db = DBSCAN(eps=eps, min_samples=2).fit(Xs)
        best_labels = db.labels_
    else:
        db = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples']).fit(Xs)
        best_labels = db.labels_
    df['dbscan_cluster'] = best_labels

    # Optionally flag small DBSCAN clusters (for example, clusters with fewer than 5 points)
    SMALL_CLUSTER_THRESHOLD = 5
    for cl in sorted(set(best_labels)):
        if cl < 0:
            continue
        sub = df[df['dbscan_cluster'] == cl]
        med_amp = float(sub['amplitude_pct'].median())
        med_dur = float(sub['duration_min'].median())
        med_vol = float(sub['total_volume'].median())
        mad_amp = float(np.median(np.abs(sub['amplitude_pct'] - med_amp)))
        mad_dur = float(np.median(np.abs(sub['duration_min'] - med_dur)))
        mad_vol = float(np.median(np.abs(sub['total_volume'] - med_vol)))
        cluster_summary = {
            'cluster':         int(cl),
            'count':           len(sub),
            'median_amplitude_pct': med_amp,
            'mad_amplitude_pct':    mad_amp,
            'median_duration_min':  med_dur,
            'mad_duration_min':     mad_dur,
            'median_total_volume':  med_vol,
            'mad_total_volume':     mad_vol,
            'parameters': {
                'eps': best_params['eps'],
                'min_samples': best_params['min_samples']
            }
        }
        # Flag small clusters
        if len(sub) < SMALL_CLUSTER_THRESHOLD:
            cluster_summary['flag_small'] = True
        else:
            cluster_summary['flag_small'] = False
        result['dbscan'].append(cluster_summary)

    return result

# ─── RUN PER ASSET ───────────────────────────────────────────────────────
@log_enter_exit
def run_asset(symbol: str, days: float) -> dict:
    label = WINDOW_LABELS.get(days, f"{days}d")
    # 1) Fetch the original price and volume series.
    price_full = fetch_clean_series(symbol)
    vol_full   = fetch_volume_series(symbol)
    if price_full.empty or vol_full.empty:
        logger.warning("Missing series for %s %s", symbol, label)
        return {'gmm': [], 'dbscan': []}
    latest = price_full.index.max()
    start = latest - timedelta(days=days)
    price_s = price_full[price_full.index >= start]
    vol_s   = vol_full[vol_full.index >= start]
    if price_s.empty or price_s.isna().all() or len(price_s.dropna()) < MIN_REQUIRED_POINTS:
        print(f"Skipping {symbol}: insufficient data")
        logger.warning("Skipping %s %s: insufficient data", symbol, label)
        return {'gmm': [], 'dbscan': []}

    # 2) Decompose into IMF dataframes for price and volume separately.
    price_imfs  = decompose_to_imf_dfs(price_s, 'imf_price')
    volume_imfs = decompose_to_imf_dfs(vol_s, 'imf_volume')

    # 3) Extract cycles.
    price_cycles_list = [c for df in price_imfs for c in extract_cycles(df, 'imf_price', ref_series=price_s)]
    volume_cycles_list = [c for df in volume_imfs for c in extract_cycles(df, 'imf_volume', ref_series=vol_s)]
    
    if not price_cycles_list or not volume_cycles_list:
        logger.warning("No price or volume cycles for %s %s", symbol, label)
        return {'gmm': [], 'dbscan': []}

    price_cycles  = pd.DataFrame(price_cycles_list)
    volume_cycles = pd.DataFrame(volume_cycles_list)

    volume_cycles = volume_cycles[volume_cycles['amplitude_pct'] != 0]
    if volume_cycles.empty:
        logger.warning("No volume cycles (after dropping flat cycles) for %s %s", symbol, label)
        return {'gmm': [], 'dbscan': []}

    # 5) Fuse cycles using the price cycles as anchors.
    fused_records = []
    for _, price_row in price_cycles.iterrows():
        overlaps = volume_cycles[
            ((volume_cycles['start_time'] >= price_row['start_time']) & 
             (volume_cycles['start_time'] <= price_row['end_time'])) |
            ((volume_cycles['end_time']   >= price_row['start_time']) & 
             (volume_cycles['end_time']   <= price_row['end_time']))
        ]
        if overlaps.empty:
            continue

        vol_sum = 0.0
        for _, vol_row in overlaps.iterrows():
            try:
                vol_segment_sum = vol_s.loc[vol_row['start_time']: vol_row['end_time']].sum()
            except Exception:
                vol_segment_sum = 0.0
            vol_sum += vol_segment_sum

        fused_records.append({
            'imf_idx':       price_row['imf_idx'],
            'start_time':    price_row['start_time'],
            'end_time':      price_row['end_time'],
            'amplitude_pct': price_row['amplitude_pct'],
            'duration_min':  price_row['duration_min'],
            'total_volume':  float(vol_sum)
        })

    logger.info("Asset %s over %s: minute bars: price=%d, volume=%d",
                symbol, label, len(price_s), len(vol_s))
    logger.info("Asset %s over %s: extracted %d price cycles and %d volume cycles",
                symbol, label, len(price_cycles), len(volume_cycles))
    logger.info("Asset %s over %s: created %d fused cycle records",
                symbol, label, len(fused_records))

    if not fused_records:
        logger.warning("No fused cycles for %s %s after overlapping join", symbol, label)
        return {'gmm': [], 'dbscan': []}

    fused = pd.DataFrame(fused_records)
    return cluster_and_summarize(fused, days)

def main():
    logger.info("Pipeline started")
    summary = {}
    # Iterate through each coin in the global COINS list.
    for coin in COINS:
        summary[coin] = {}
        for days in TARGET_DAYS:
            key = WINDOW_LABELS.get(days, f"{days}d")
            logger.info("Running asset for coin %s with time period %s", coin, key)
            summary[coin][key] = run_asset(coin, days)
    with open(OUTPUT_FILE, 'w') as fp:
        json.dump(summary, fp, indent=2)
    logger.info("Wrote %s", OUTPUT_FILE)

    # Write human-readable summary
    lines = []
    for coin, windows in summary.items():
        dominant = None
        for result in windows.values():
            clusters = result.get('gmm', []) or result.get('dbscan', [])
            for cl in clusters:
                if dominant is None or cl.get('median_amplitude_pct', 0) > dominant.get('median_amplitude_pct', 0):
                    dominant = cl
        if dominant:
            line = f"{coin}, {dominant['median_duration_min']}, {dominant['median_amplitude_pct']}"
        else:
            line = f"{coin}, NA, NA"
        lines.append(line)
    with open('imf_clusters_overlaps.txt', 'w') as tf:
        tf.write("\n".join(lines))

    logger.info("Wrote imf_clusters_overlaps.txt")
    logger.info("Pipeline finished")

if __name__ == '__main__':
    main()