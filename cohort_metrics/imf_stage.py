import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyEMD import CEEMDAN
from scipy.signal import hilbert, medfilt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None

from .db import get_db_engine, fetch_history_months


EPS = 1e-9


@dataclass
class Config:
    leaders: List[str]
    asof: pd.Timestamp
    lookback_months: int
    persist_imfs: bool
    cluster_mixed: bool
    n_jobs: int
    max_seconds_per_asset: Optional[float]
    detrend: bool
    winsorize: bool
    out_dir: str
    db_host: Optional[str]
    db_user: Optional[str]
    db_password: Optional[str]
    db_name: Optional[str]
    db_port: int
    db_table: str


def _parse_leaders(leaders_csv: Optional[str], leaders_list: Optional[str]) -> List[str]:
    syms: List[str] = []
    if leaders_csv:
        df = pd.read_csv(leaders_csv)
        if "symbol" not in df.columns:
            raise ValueError("leaders CSV must include 'symbol' column")
        syms = df["symbol"].dropna().astype(str).tolist()
    if leaders_list:
        syms.extend([s.strip() for s in leaders_list.split(",") if s.strip()])
    # Canonicalize formatting exactly as provided (state/regime used CSV inputs too)
    # De-duplicate preserving order
    seen = set()
    out: List[str] = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _asof_ts(s: str) -> pd.Timestamp:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid --asof value: {s}")
    # If date-only, anchor at 23:59:59
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.tzinfo is not None and len(s) <= 10:
        ts = ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return ts


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _window_start(asof: pd.Timestamp, lookback_months: int) -> pd.Timestamp:
    return (asof - pd.DateOffset(months=lookback_months)).floor("T")


def _reindex_1min(df: pd.DataFrame, asof: pd.Timestamp, win_start: pd.Timestamp) -> pd.DataFrame:
    g = df.copy()
    g = g.sort_values("timestamp")
    # keep last for duplicates
    g = g.drop_duplicates(subset=["timestamp"], keep="last")
    # restrict window (window_start, asof] inclusive
    g = g[(g["timestamp"] > win_start) & (g["timestamp"] <= asof)]
    g = g.set_index("timestamp")
    full_idx = pd.date_range(start=win_start + pd.Timedelta(minutes=1), end=asof, freq="T", tz="UTC")
    g = g.reindex(full_idx)
    # fill close
    if "close" in g.columns:
        g["close"] = g["close"].astype(float)
        g["close"] = g["close"].ffill()
        g["close"] = g["close"].bfill()
    # fill volume
    if "volume" in g.columns:
        g["volume"] = g["volume"].fillna(0.0).astype(float)
    g = g.reset_index().rename(columns={"index": "timestamp"})
    return g


def _theilsen_detrend(y: np.ndarray, seed: int = 1337, biweight_c: float = 4.685, passes: int = 3) -> np.ndarray:
    from sklearn.linear_model import TheilSenRegressor

    n = len(y)
    x = np.arange(n).reshape(-1, 1)
    rng = np.random.RandomState(seed)
    model = TheilSenRegressor(random_state=rng, fit_intercept=True)
    model.fit(x, y)
    trend = model.predict(x)
    resid = y - trend
    for _ in range(passes):
        med = np.median(resid)
        mad = np.median(np.abs(resid - med)) + EPS
        u = (resid - med) / (biweight_c * mad)
        w = (1 - u ** 2) ** 2
        w[np.abs(u) >= 1] = 0.0
        # Weighted least squares for line fit
        W = np.diag(w)
        X = np.c_[np.ones(n), np.arange(n)]
        try:
            beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
            trend = beta[0] + beta[1] * np.arange(n)
        except Exception:
            break
        resid = y - trend
    return y - trend


def _transform_series(close: pd.Series, volume: pd.Series, detrend: bool, winsorize: bool) -> Tuple[np.ndarray, np.ndarray]:
    c = close.to_numpy(dtype=float)
    v = volume.to_numpy(dtype=float)
    y_price = np.log(np.maximum(c, EPS))
    if detrend:
        y_price = _theilsen_detrend(y_price, seed=1337)
    if winsorize:
        lo = np.nanpercentile(v, 1)
        hi = np.nanpercentile(v, 99)
        v_clip = np.clip(v, lo, hi)
    else:
        v_clip = v
    y_vol = np.log1p(np.maximum(v_clip, 0.0))
    return y_price, y_vol


def _ceemdan(y: np.ndarray, trials: int, seed: int, max_seconds: Optional[float]) -> Tuple[np.ndarray, np.ndarray, str, int]:
    start = time.time()
    status = "ok"
    used_trials = trials
    try:
        ce = CEEMDAN(trials=trials, random_state=seed)
        imfs = ce.ceemdan(y)
        residue = y - imfs.sum(axis=0)
    except Exception:
        # Retry with lower trials
        used_trials = max(20, min(trials, 20))
        status = "time_budget_reduced"
        ce = CEEMDAN(trials=used_trials, random_state=seed)
        imfs = ce.ceemdan(y)
        residue = y - imfs.sum(axis=0)
    # budget check after
    if max_seconds is not None and (time.time() - start) > max_seconds and status == "ok":
        status = "time_budget_reduced"
    return imfs, residue, status, used_trials


def _hilbert_phase(imf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    analytic = hilbert(imf)
    amp = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    dphi = np.diff(phase)
    inst_period = (2.0 * np.pi) / np.clip(dphi, EPS, None)
    # pad to match length
    inst_period = np.r_[inst_period, inst_period[-1]]
    return amp, phase, inst_period


def _circular_variance(theta: np.ndarray) -> float:
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.hypot(c.mean(), s.mean())
    return float(1.0 - R)


def _extrema_brackets(imf: np.ndarray) -> List[Tuple[int, int, int]]:
    # Return list of (peak, trough, peak) indices or (trough, peak, trough)
    # Use simple 3-point median smoothing to reduce jitter
    from scipy.signal import find_peaks

    sm = medfilt(imf, kernel_size=3)
    peaks, _ = find_peaks(sm)
    troughs, _ = find_peaks(-sm)
    pts = sorted([(i, "p") for i in peaks] + [(i, "t") for i in troughs])
    brackets: List[Tuple[int, int, int]] = []
    for i in range(len(pts) - 2):
        a, b, c = pts[i], pts[i + 1], pts[i + 2]
        if a[1] == "p" and b[1] == "t" and c[1] == "p":
            brackets.append((a[0], b[0], c[0]))
        if a[1] == "t" and b[1] == "p" and c[1] == "t":
            brackets.append((a[0], b[0], c[0]))
    return brackets


def _cycle_depth_pct(series_orig: np.ndarray, i_peak: int, i_trough: int) -> float:
    p = series_orig[i_peak]
    t = series_orig[i_trough]
    denom = (p + t) / 2.0
    if denom == 0:
        return 0.0
    return float(abs(p - t) / abs(denom) * 100.0)


def _analyze_imf(
    series_name: str,
    y_transformed: np.ndarray,
    series_orig: np.ndarray,
    min_period_bars: int,
    max_period_frac: float,
) -> Optional[Dict[str, float]]:
    # Smooth IMF
    imf = y_transformed
    amp, phase, inst_period = _hilbert_phase(imf)
    median_period = float(np.nanmedian(inst_period[np.isfinite(inst_period)]))
    if not math.isfinite(median_period) or median_period < min_period_bars:
        return None
    # period sanity
    if median_period > max_period_frac * len(imf):
        return None
    # Edge guard
    guard = max(1, int(round(0.25 * median_period)))
    valid_slice = slice(guard, len(imf) - guard)
    if valid_slice.stop - valid_slice.start < min_period_bars * 2:
        return None
    median_period_guarded = float(np.nanmedian(inst_period[valid_slice]))
    phase_var = _circular_variance(phase[valid_slice])
    # Extrema brackets
    brackets = _extrema_brackets(imf)
    # filter brackets to be inside valid slice and not too close to edges
    good_brackets = [b for b in brackets if b[0] >= guard and b[2] <= (len(imf) - guard)]
    # Compute cycle depths in original scale; select >=2%
    kept_depths: List[float] = []
    for a, b, c in good_brackets:
        d = _cycle_depth_pct(series_orig, a, b)
        d2 = _cycle_depth_pct(series_orig, b, c)
        depth = max(d, d2)
        if depth >= 2.0:
            kept_depths.append(depth)
    cycles_count = len(good_brackets)
    pct_cycles_ge_2pct = float(len(kept_depths) / cycles_count * 100.0) if cycles_count > 0 else float("nan")
    if len(kept_depths) == 0:
        return None
    amp_pct = float(np.median(kept_depths))
    energy = float(np.var(imf))
    return {
        "period_bars": median_period_guarded,
        "amp_pct": amp_pct,
        "phase_var": phase_var,
        "cycles_count": float(cycles_count),
        "pct_cycles_ge_2pct": pct_cycles_ge_2pct,
        "energy": energy,
    }


def _compute_imf_features_for_series(
    series_name: str,
    y: np.ndarray,
    series_orig: np.ndarray,
    min_period_bars: int,
    max_period_frac: float,
    trials: int,
    seed: int,
    max_seconds_per_asset: Optional[float],
    persist_path: Optional[str] = None,
) -> Tuple[List[Dict[str, float]], str]:
    imfs, residue, status, used_trials = _ceemdan(y, trials=trials, seed=seed, max_seconds=max_seconds_per_asset)
    n = len(y)
    feats: List[Dict[str, float]] = []
    if imfs is None or imfs.size == 0:
        return feats, ("decomp_error" if status == "ok" else status)
    # Energy shares per IMF (exclude residue)
    energies = np.var(imfs, axis=1)
    energy_total = float(np.sum(energies)) + EPS
    for k in range(imfs.shape[0]):
        info = _analyze_imf(series_name, imfs[k], series_orig, min_period_bars, max_period_frac)
        if info is None:
            continue
        info["energy_share"] = float(energies[k] / energy_total)
        info["imf_id"] = float(k)
        feats.append(info)
    if persist_path is not None:
        _ensure_dir(os.path.dirname(persist_path))
        np.savez_compressed(
            persist_path,
            imfs=imfs,
            residue=residue,
            imf_variances=energies,
        )
    if len(feats) == 0:
        return feats, "no_kept_imfs" if status == "ok" else status
    return feats, status


def _pair_price_volume(price_feats: List[Dict[str, float]], vol_feats: List[Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    # Pair each price IMF to closest-period volume IMF; tie-break by amp corr (placeholder: use period proximity only)
    if not price_feats or not vol_feats:
        return {}
    out: Dict[int, Dict[str, float]] = {}
    vol_periods = np.array([vf["period_bars"] for vf in vol_feats], dtype=float)
    for pf in price_feats:
        pid = int(pf["imf_id"])
        pper = float(pf["period_bars"])
        idx = int(np.argmin(np.abs(vol_periods - pper)))
        vf = vol_feats[idx]
        out[pid] = {
            "vol_period_ratio": float((pper + EPS) / (vf["period_bars"] + EPS)),
            "vol_energy_share": float(vf.get("energy_share", float("nan"))),
            # amp corr left as NaN placeholder; could compute via Hilbert envelopes with original series
            "vol_amp_corr": float("nan"),
        }
    return out


def _mad_pct(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    if med == 0:
        return float("nan")
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad / abs(med) * 100.0)


def _global_cluster(labels_prefix: str, X: np.ndarray, seed: int = 1337) -> Tuple[np.ndarray, Dict[int, str]]:
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int), {}
    labels = None
    model = None
    if hdbscan is not None:
        model = hdbscan.HDBSCAN(min_cluster_size=8, metric="euclidean", cluster_selection_method="leaf")
        labels = model.fit_predict(X)
        # Convert noise (-1) to unique singletons starting at 90001
        if (labels == -1).any():
            next_id = 90001
            for i in range(n):
                if labels[i] == -1:
                    labels[i] = next_id
                    next_id += 1
    if model is None or (hasattr(model, "labels_") and len(set([l for l in labels if l < 90000])) == 0):
        # Fallback KMeans
        best = None
        best_k = None
        rng = np.random.RandomState(seed)
        for k in range(2, max(2, min(12, n - 1)) + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=rng)
            l = km.fit_predict(X)
            try:
                score = silhouette_score(X, l)
            except Exception:
                continue
            if best is None or score > best:
                best = score
                best_k = (l, km)
        labels, _ = best_k if best_k is not None else (np.zeros(n, dtype=int), None)
    # Relabel to GP#### ordering by cluster size then lower median period not available here; we'll map later outside
    return labels, {}


def _label_clusters(series: str, labels: np.ndarray, periods: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
    # Map integer labels to stable IDs like GP0001, GV0001, GP90001, etc.
    n = len(labels)
    unique = {}
    # Separate noise singletons (>=90000) from real clusters (<90000)
    real_ids = sorted(set([int(l) for l in labels if int(l) < 90000]))
    # Order by size desc, then lower median period
    order = []
    for cid in real_ids:
        idx = np.where(labels == cid)[0]
        medp = float(np.median(periods[idx])) if len(idx) else float("inf")
        order.append((cid, len(idx), medp))
    order.sort(key=lambda x: (-x[1], x[2]))
    prefix = "GP" if series == "price" else "GV"
    mapping: Dict[int, str] = {}
    for rank, (cid, _, _) in enumerate(order, start=1):
        mapping[cid] = f"{prefix}{rank:04d}"
    # Assign noise/singletons GP/GV9xxxx using their existing numbers
    for i in range(n):
        l = int(labels[i])
        if l >= 90000:
            mapping[l] = f"{prefix}{l}"
    # Build labeled
    out = np.array([mapping[int(l)] for l in labels], dtype=object)
    return out, mapping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IMF stage: CEEMDAN decomposition, features, and clustering")
    p.add_argument("--leaders-csv", help="CSV with column 'symbol' for leaders")
    p.add_argument("--leaders", help="Comma-separated list of symbols")
    p.add_argument("--asof", required=True, help="UTC timestamp or YYYY-MM-DD (anchors 23:59:59Z)")
    p.add_argument("--lookback-months", type=int, required=True, help="Calendar months back from asof")
    p.add_argument("--persist-imfs", action="store_true", help="Persist per-symbol IMFs to NPZ")
    p.add_argument("--cluster-mixed", action="store_true", help="Pool price and volume IMFs together")
    p.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--max-seconds-per-asset", type=float, help="Time budget per (symbol,series)")
    p.add_argument("--detrend", action="store_true", help="Detrend price with robust Theil–Sen + Tukey capping")
    p.add_argument("--no-winsorize", dest="winsorize", action="store_false", help="Disable volume winsorization")
    p.add_argument("--out", default="data/imf", help="Output base directory")
    # DB flags
    p.add_argument("--host", help="MySQL host")
    p.add_argument("--user", help="MySQL user")
    p.add_argument("--database", help="Database name")
    p.add_argument("--port", type=int, default=3306, help="MySQL port")
    p.add_argument("--password", help="MySQL password")
    p.add_argument("--table", default="ohlcvt", help="OHLCVT table name")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    leaders = _parse_leaders(args.leaders_csv, args.leaders)
    if not leaders:
        raise SystemExit("No leaders provided")
    asof = _asof_ts(args.asof)
    win_start = _window_start(asof, args.lookback_months)
    dt_dir = os.path.join(args.out, f"dt={asof.date().isoformat()}")
    _ensure_dir(dt_dir)

    # Fetch minute OHLCV history for leaders
    if not (args.host and args.user and args.database and args.password):
        raise SystemExit("DB credentials required to fetch minute OHLCV")
    engine = get_db_engine(args.host, args.user, args.password, args.database, args.port)
    hist = fetch_history_months(engine, leaders, args.lookback_months, table=args.table)
    engine.dispose()
    if hist.empty:
        raise SystemExit("No history fetched")
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True, errors="coerce")

    # Prepare per-symbol time series
    rows: List[Dict[str, object]] = []
    bars_per_symbol: Dict[str, int] = {}
    persist_paths: Dict[Tuple[str, str], str] = {}

    for sym, g in hist.groupby("symbol"):
        g = g.dropna(subset=["timestamp"]).copy()
        g = _reindex_1min(g, asof, win_start)
        bars_per_symbol[sym] = len(g)
        close = g["close"].astype(float)
        volume = g["volume"].astype(float)
        y_price, y_vol = _transform_series(close, volume, detrend=bool(args.detrend), winsorize=bool(args.winsorize))
        # Decompose both series
        status_price = "ok"
        status_vol = "ok"
        price_feats: List[Dict[str, float]] = []
        vol_feats: List[Dict[str, float]] = []
        min_period_bars = 3
        max_period_frac = 2.0 / 3.0
        if args.persist_imfs:
            persist_paths[(sym, "price")] = os.path.join(dt_dir, "imfs", f"{sym}_price.npz")
            persist_paths[(sym, "volume")] = os.path.join(dt_dir, "imfs", f"{sym}_volume.npz")
        price_feats, status_price = _compute_imf_features_for_series(
            "price", y_price, close.to_numpy(dtype=float), min_period_bars, max_period_frac,
            trials=100, seed=1337, max_seconds_per_asset=args.max_seconds_per_asset,
            persist_path=persist_paths.get((sym, "price")) if args.persist_imfs else None,
        )
        vol_feats, status_vol = _compute_imf_features_for_series(
            "volume", y_vol, volume.to_numpy(dtype=float), min_period_bars, max_period_frac,
            trials=100, seed=1337, max_seconds_per_asset=args.max_seconds_per_asset,
            persist_path=persist_paths.get((sym, "volume")) if args.persist_imfs else None,
        )
        # Fusion: pair volume to price IMFs
        fusion = _pair_price_volume(price_feats, vol_feats)
        # Accumulate rows
        for pf in price_feats:
            row: Dict[str, object] = {
                "symbol": sym,
                "asof": asof.to_pydatetime(),
                "series": "price",
                "imf_id": int(pf["imf_id"]),
                "period_days": float(pf["period_bars"]) / (60.0 * 24.0),
                "amp_pct": float(pf["amp_pct"]),
                "energy_share": float(pf["energy_share"]),
                "cycles_count": float(pf["cycles_count"]),
                "pct_cycles_ge_2pct": float(pf["pct_cycles_ge_2pct"]),
                "phase_var": float(pf["phase_var"]),
                "status": status_price,
            }
            if pf["imf_id"] in fusion:
                f = fusion[int(pf["imf_id"])]
                row.update({
                    "vol_amp_corr": f.get("vol_amp_corr", float("nan")),
                    "vol_period_ratio": f.get("vol_period_ratio", float("nan")),
                    "vol_energy_share": f.get("vol_energy_share", float("nan")),
                })
            rows.append(row)
        for vf in vol_feats:
            rows.append({
                "symbol": sym,
                "asof": asof.to_pydatetime(),
                "series": "volume",
                "imf_id": int(vf["imf_id"]),
                "period_days": float(vf["period_bars"]) / (60.0 * 24.0),
                "amp_pct": float(vf["amp_pct"]),
                "energy_share": float(vf["energy_share"]),
                "cycles_count": float(vf["cycles_count"]),
                "pct_cycles_ge_2pct": float(vf["pct_cycles_ge_2pct"]),
                "phase_var": float(vf["phase_var"]),
                "status": status_vol,
            })

    if not rows:
        print("No IMF features computed")
        return

    df_feat = pd.DataFrame(rows)

    # Clustering per series (default) or mixed pool
    clusters_global: Dict[str, Dict[str, object]] = {}
    cards_global: Dict[str, List[Dict[str, object]]] = {"price": [], "volume": []}

    def cluster_pool(series: str, sub: pd.DataFrame) -> pd.DataFrame:
        if sub.empty:
            return sub
        # Scale features
        feats = ["period_days", "amp_pct", "energy_share", "cycles_count", "pct_cycles_ge_2pct", "phase_var"]
        X = sub[feats].to_numpy(dtype=float)
        scaler = RobustScaler(quantile_range=(25, 75))
        Xs = scaler.fit_transform(X)
        labels_int, _ = _global_cluster(series, Xs, seed=1337)
        label_ids, mapping = _label_clusters(series, labels_int, sub["period_days"].to_numpy(dtype=float))
        sub = sub.copy()
        sub["cluster_id_global"] = label_ids
        # Global cluster stats for cards
        cluster_stats: Dict[str, Dict[str, object]] = {}
        for cid in np.unique(label_ids):
            mask = (sub["cluster_id_global"] == cid)
            g = sub.loc[mask]
            med_period = float(np.median(g["period_days"])) if not g.empty else float("nan")
            med_amp = float(np.median(g["amp_pct"])) if not g.empty else float("nan")
            madp = _mad_pct(g["period_days"].to_numpy(dtype=float))
            mada = _mad_pct(g["amp_pct"].to_numpy(dtype=float))
            rel_p = float(math.exp(-(madp if math.isfinite(madp) else 0.0) / 25.0)) if g.shape[0] > 1 else 0.0
            rel_a = float(math.exp(-(mada if math.isfinite(mada) else 0.0) / 25.0)) if g.shape[0] > 1 else 0.0
            cluster_stats[str(cid)] = {
                "cluster_id": str(cid),
                "n_imfs": int(g.shape[0]),
                "median_period_days": med_period,
                "MAD%_period": madp,
                "median_amp_pct": med_amp,
                "MAD%_amp": mada,
                "median_energy_share": float(np.median(g["energy_share"])) if not g.empty else float("nan"),
                "reliability_period": rel_p,
                "reliability_amp": rel_a,
                "example_symbols": g.sort_values("energy_share", ascending=False)["symbol"].head(5).tolist(),
            }
        clusters_global[series] = {"scaler_center": scaler.center_.tolist(), "scaler_scale": scaler.scale_.tolist()}
        cards_global[series] = list(cluster_stats.values())
        # Per-asset clustering on log(period)
        asset_ids: List[str] = []
        asset_stats: List[Dict[str, object]] = []
        for sym, gg in sub.groupby("symbol"):
            per = gg["period_days"].to_numpy(dtype=float)
            if per.size == 0:
                asset_ids.extend(["" for _ in range(len(gg))])
                continue
            X1 = np.log(np.maximum(per, EPS)).reshape(-1, 1)
            try:
                hac = AgglomerativeClustering(n_clusters=None, distance_threshold=0.10, linkage="ward")
                la = hac.fit_predict(X1)
            except Exception:
                la = np.arange(len(gg))
            # Labeling A:SYMBOL:#### by size
            order = []
            for cid in np.unique(la):
                idx = np.where(la == cid)[0]
                medp = float(np.median(per[idx])) if idx.size else float("inf")
                order.append((int(cid), len(idx), medp))
            order.sort(key=lambda x: (-x[1], x[2]))
            mapping_asset: Dict[int, str] = {}
            for rank, (cid, _, _) in enumerate(order, start=1):
                mapping_asset[cid] = f"A:{sym}:{rank:04d}"
            asset_ids.extend([mapping_asset[int(cid)] for cid in la])
        sub["cluster_id_asset"] = asset_ids
        return sub

    if args.cluster_mixed:
        df_feat = cluster_pool("price", df_feat)
    else:
        parts = []
        for series in ["price", "volume"]:
            parts.append(cluster_pool(series, df_feat[df_feat["series"] == series]))
        df_feat = pd.concat(parts, ignore_index=True) if parts else df_feat

    # Attach global and asset stats redundantly per IMF for join convenience
    # For simplicity, recompute stats per series cluster id
    def attach_stats(sub: pd.DataFrame) -> pd.DataFrame:
        out = sub.copy()
        # Global
        stats_g = {}
        for cid in out["cluster_id_global"].dropna().unique():
            g = out[out["cluster_id_global"] == cid]
            medp = float(np.median(g["period_days"]))
            meda = float(np.median(g["amp_pct"]))
            madp = _mad_pct(g["period_days"].to_numpy(dtype=float))
            mada = _mad_pct(g["amp_pct"].to_numpy(dtype=float))
            relp = float(math.exp(-(madp if math.isfinite(madp) else 0.0) / 25.0)) if g.shape[0] > 1 else 0.0
            rela = float(math.exp(-(mada if math.isfinite(mada) else 0.0) / 25.0)) if g.shape[0] > 1 else 0.0
            stats_g[str(cid)] = {
                "cluster_size_global": int(g.shape[0]),
                "cluster_median_period_global": medp,
                "cluster_mad_pct_period_global": madp,
                "cluster_median_amp_global": meda,
                "cluster_mad_pct_amp_global": mada,
                "cluster_reliability_period_global": relp,
                "cluster_reliability_amp_global": rela,
            }
        rows2 = []
        for _, r in out.iterrows():
            d = r.to_dict()
            cid = str(r.get("cluster_id_global"))
            d.update(stats_g.get(cid, {}))
            rows2.append(d)
        out = pd.DataFrame(rows2)
        # Asset-level stats
        stats_a = {}
        for (sym, cid), g in out.groupby(["symbol", "cluster_id_asset"]):
            medp = float(np.median(g["period_days"]))
            meda = float(np.median(g["amp_pct"]))
            madp = _mad_pct(g["period_days"].to_numpy(dtype=float))
            mada = _mad_pct(g["amp_pct"].to_numpy(dtype=float))
            relp = float(math.exp(-(madp if math.isfinite(madp) else 0.0) / 25.0)) if g.shape[0] > 1 else 0.0
            rela = float(math.exp(-(mada if math.isfinite(mada) else 0.0) / 25.0)) if g.shape[0] > 1 else 0.0
            stats_a[(sym, cid)] = {
                "cluster_size_asset": int(g.shape[0]),
                "cluster_median_period_asset": medp,
                "cluster_mad_pct_period_asset": madp,
                "cluster_median_amp_asset": meda,
                "cluster_mad_pct_amp_asset": mada,
                "cluster_reliability_period_asset": relp,
                "cluster_reliability_amp_asset": rela,
            }
        rows3 = []
        for _, r in out.iterrows():
            d = r.to_dict()
            key = (r.get("symbol"), r.get("cluster_id_asset"))
            d.update(stats_a.get(key, {}))
            rows3.append(d)
        return pd.DataFrame(rows3)

    df_feat = attach_stats(df_feat)

    # Write outputs
    summary_path = os.path.join(dt_dir, "imf_summary.parquet")
    try:
        df_feat.to_parquet(summary_path, index=False, compression="snappy")
    except Exception:
        summary_path = os.path.splitext(summary_path)[0] + ".csv"
        df_feat.to_csv(summary_path, index=False)

    # Cluster cards
    with open(os.path.join(dt_dir, "cluster_cards_global_price.json"), "w", encoding="utf-8") as f:
        json.dump(cards_global.get("price", []), f, indent=2)
    with open(os.path.join(dt_dir, "cluster_cards_global_volume.json"), "w", encoding="utf-8") as f:
        json.dump(cards_global.get("volume", []), f, indent=2)

    # Asset cards
    cards_asset: Dict[str, List[Dict[str, object]]] = {}
    for sym, g in df_feat.groupby("symbol"):
        cl = []
        for cid, gg in g.groupby("cluster_id_asset"):
            cl.append({
                "cluster_id": cid,
                "n_imfs": int(gg.shape[0]),
                "median_period_days": float(np.median(gg["period_days"])) if not gg.empty else float("nan"),
                "MAD%_period": _mad_pct(gg["period_days"].to_numpy(dtype=float)),
                "median_amp_pct": float(np.median(gg["amp_pct"])) if not gg.empty else float("nan"),
                "MAD%_amp": _mad_pct(gg["amp_pct"].to_numpy(dtype=float)),
                "median_energy_share": float(np.median(gg["energy_share"])) if not gg.empty else float("nan"),
            })
        cards_asset[sym] = cl
    with open(os.path.join(dt_dir, "cluster_cards_asset.json"), "w", encoding="utf-8") as f:
        json.dump(cards_asset, f, indent=2)

    # Run config echo
    run_config = {
        "leaders": leaders,
        "asof": asof.isoformat(),
        "window_start": win_start.isoformat(),
        "lookback_months": int(args.lookback_months),
        "persist_imfs": bool(args.persist_imfs),
        "cluster_mixed": bool(args.cluster_mixed),
        "n_jobs": int(args.n_jobs),
        "max_seconds_per_asset": float(args.max_seconds_per_asset) if args.max_seconds_per_asset else None,
        "detrend": bool(args.detrend),
        "winsorize": bool(args.winsorize),
        "scalers": clusters_global,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "bars_per_symbol": bars_per_symbol,
        "paths": {
            "summary": summary_path,
            "dt_dir": dt_dir,
        },
    }
    with open(os.path.join(dt_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(
        f"IMF stage complete: symbols={df_feat['symbol'].nunique()} rows={len(df_feat)} out={summary_path}"
    )


if __name__ == "__main__":
    main()

