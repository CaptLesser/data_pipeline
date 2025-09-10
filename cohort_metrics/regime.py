"""
Daily Regime snapshot computation.

Computes slow-context features on daily bars across 3d, 7d, 14d, 30d, 90d
windows (math-only). Optionally enriches with historical baselines for
percentiles/quintiles/robust-z when provided.
"""

import os
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, List

import pandas as pd
import numpy as np

from .core import EPS, _bollinger, _atr, _adx, enrich_current_with_baseline
import argparse
import json


REGIME_WINDOWS_DAYS = [3, 7, 14, 30, 90]


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample("1D").agg(agg).dropna(how="all").reset_index()


def _skew_kurt(log_rets: np.ndarray) -> (float, float):
    x = np.asarray(log_rets, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan"), float("nan")
    m = np.mean(x)
    c = x - m
    m2 = np.mean(c ** 2)
    if m2 == 0:
        return float("nan"), float("nan")
    m3 = np.mean(c ** 3)
    m4 = np.mean(c ** 4)
    skew = float(m3 / (m2 ** 1.5))
    kurt_excess = float(m4 / (m2 ** 2) - 3.0)
    return skew, kurt_excess


def compute_regime_window_metrics(daily: pd.DataFrame, window_days: int, suffix: str, breakout_lookback_days: int = 20) -> Dict[str, float]:
    if daily.empty:
        return {}
    win = daily.tail(window_days)
    if win.empty:
        return {}
    c = win["close"].astype(float)
    h = win["high"].astype(float)
    l = win["low"].astype(float)
    v = win["volume"].astype(float)

    last = float(c.iloc[-1]) if len(c) else float("nan")
    first = float(c.iloc[0]) if len(c) else float("nan")

    # Returns (log)
    log_rets = np.log(np.maximum(c, EPS)).diff().dropna()
    rv_close = float(log_rets.std(ddof=1)) if len(log_rets) > 1 else float("nan")
    vol_of_vol = float(np.abs(log_rets).std(ddof=1)) if len(log_rets) > 1 else float("nan")
    skew, kurt = _skew_kurt(log_rets.to_numpy()) if len(log_rets) else (float("nan"), float("nan"))

    # MAs
    full_c = daily["close"].astype(float)
    ma50 = full_c.rolling(50, min_periods=1).mean()
    ma200 = full_c.rolling(200, min_periods=1).mean()
    ma50_win = ma50.tail(window_days)
    ma200_win = ma200.tail(window_days)
    ma50_slope = float((ma50_win.iloc[-1] - ma50_win.iloc[0]) / (abs(ma50_win.iloc[0]) + EPS)) if len(ma50_win) > 1 else float("nan")
    ma200_slope = float((ma200_win.iloc[-1] - ma200_win.iloc[0]) / (abs(ma200_win.iloc[0]) + EPS)) if len(ma200_win) > 1 else float("nan")
    ma_diff = float((ma50.iloc[-1] - ma200.iloc[-1]) / (abs(ma200.iloc[-1]) + EPS)) if len(ma50) and len(ma200) else float("nan")
    above_ma50_share = float(np.mean((full_c.tail(window_days) > ma50.tail(window_days)).astype(float))) if len(full_c) else float("nan")

    # ADX and ATR on window
    atr = _atr(h, l, c, period=14)
    adx = _adx(h, l, c, period=14)

    # Bollinger & squeeze
    bb = _bollinger(c, period=min(20, len(c)), num_std=2.0)
    squeeze_index = float((bb["width"]) / (atr * 4.0 + EPS)) if np.isfinite([bb.get("width", np.nan), atr]).all() else float("nan")

    # Range intensity
    range_intensity = float((float(h.max()) - float(l.min())) / (abs(last) + EPS)) if len(h) and len(l) and np.isfinite(last) else float("nan")

    # Breakout frequency (configurable lookback days, constrained by available history)
    full = daily.copy()
    full_c_rollmax = full["close"].rolling(breakout_lookback_days, min_periods=1).max()
    full_c_rollmin = full["close"].rolling(breakout_lookback_days, min_periods=1).min()
    win_idx_start = len(full) - len(win)
    hi_hits = (full["close"].iloc[win_idx_start:] >= full_c_rollmax.iloc[win_idx_start:]).astype(float)
    lo_hits = (full["close"].iloc[win_idx_start:] <= full_c_rollmin.iloc[win_idx_start:]).astype(float)
    breakout_hi_share = float(hi_hits.mean()) if len(hi_hits) else float("nan")
    breakout_lo_share = float(lo_hits.mean()) if len(lo_hits) else float("nan")

    # ROC over the window
    roc = float(((last - first) / first) * 100.0) if np.isfinite([last, first]).all() and first != 0 else float("nan")

    out = {
        f"roc_{suffix}": roc,
        f"rv_close_{suffix}_mag": rv_close,
        f"vol_of_vol_{suffix}_mag": vol_of_vol,
        f"ret_skew_{suffix}_dir": skew,
        f"ret_kurtosis_{suffix}_mag": kurt,
        f"ma50_slope_{suffix}_dir": ma50_slope,
        f"ma200_slope_{suffix}_dir": ma200_slope,
        f"ma50_minus_ma200_{suffix}_dir": ma_diff,
        f"above_ma50_share_{suffix}_pos": above_ma50_share,
        f"adx_{suffix}_mag": adx,
        f"atr_{suffix}_mag": atr,
        f"parkinson_vol_{suffix}_mag": float(np.sqrt((1.0 / (4.0 * np.log(2.0))) * np.mean((np.log(h / l)) ** 2))) if (h > 0).all() and (l > 0).all() else float("nan"),
        f"garman_klass_{suffix}_mag": (
            float(np.sqrt(max(0.5 * np.mean((np.log(h / l)) ** 2) - (2.0 * np.log(2.0) - 1.0) * np.mean((np.log(c / win['open'].astype(float))) ** 2), 0.0)))
            if "open" in win.columns and (win["open"].astype(float) > 0).all() and (c > 0).all()
            else float("nan")
        ),
        f"squeeze_index_{suffix}_mag": squeeze_index,
        f"range_intensity_{suffix}_mag": range_intensity,
        f"breakout_hi_share_{suffix}_pos": breakout_hi_share,
        f"breakout_lo_share_{suffix}_pos": breakout_lo_share,
    }
    out.update({
        f"metrics_valid_volatility_{suffix}": bool(np.isfinite([rv_close, vol_of_vol]).any()),
        f"metrics_valid_trend_{suffix}": bool(np.isfinite([ma50_slope, ma200_slope, ma_diff, adx]).any()),
        f"metrics_valid_range_{suffix}": bool(np.isfinite([range_intensity]).any()),
        f"metrics_valid_breakout_{suffix}": bool(np.isfinite([breakout_hi_share, breakout_lo_share]).any()),
    })
    return out


def compute_regime_series(daily: pd.DataFrame, window_days: int, suffix: str, breakout_lookback_days: int = 20) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    df = daily.copy()
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")
    freq = f"{window_days}D"
    win_start = df.index.floor(freq)
    df = df.assign(_win_start=win_start)
    rows: List[Dict[str, float]] = []
    for wstart, g in df.groupby("_win_start"):
        metrics = compute_regime_window_metrics(g.reset_index(), window_days, suffix, breakout_lookback_days=breakout_lookback_days)
        if not metrics:
            continue
        wend = wstart + pd.to_timedelta(window_days, unit="D")
        row = {"window_start": wstart.to_pydatetime(), "window_end": wend.to_pydatetime()}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_regime_snapshot(
    cohort_csv: str,
    windows_days: Iterable[int] = REGIME_WINDOWS_DAYS,
    baselines: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    db_fetch_fn=None,
    db_engine=None,
    asof: Optional[datetime] = None,
    breakout_lookback_days: int = 20,
) -> pd.DataFrame:
    df = pd.read_csv(cohort_csv)
    if df.empty:
        return pd.DataFrame(columns=["symbol"])  # empty
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # Anchor date (UTC day) for snapshot
    if asof is None:
        asof = df["timestamp"].max().floor("1D")
    rows = []
    for sym, g in df.groupby("symbol"):
        daily = resample_to_daily(g)
        if daily.empty:
            rows.append({"symbol": sym})
            continue
        row = {"symbol": sym}
        for d in windows_days:
            sfx = f"{d}d"
            row.update(compute_regime_window_metrics(daily, d, sfx, breakout_lookback_days=breakout_lookback_days))
        row["timestamp_asof_utc"] = pd.to_datetime(asof).to_pydatetime()
        rows.append(row)
    regime_df = pd.DataFrame(rows)

    if baselines is None and db_fetch_fn is not None and db_engine is not None:
        # Optional: enrich using DB-provided daily history
        symbols = regime_df["symbol"].dropna().unique().tolist()
        hist_df = db_fetch_fn(db_engine, symbols)  # expected daily bars
        baselines = build_daily_baselines(hist_df, windows_days)

    if baselines:
        regime_df = enrich_current_with_baseline(regime_df, baselines)
    # Impute neutral (no NaNs)
    regime_df = impute_neutral_regime(regime_df)
    return regime_df


def write_regime_parquet(
    df: pd.DataFrame,
    utc_date: Optional[datetime] = None,
    base_dir: str = "data/regime",
    filename: str = "regime_snapshot.parquet",
    also_csv: bool = False,
    compression: str = "snappy",
    schema_meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    if utc_date is None:
        utc_date = datetime.now(timezone.utc)
    date_dir = os.path.join(base_dir, f"dt={utc_date.strftime('%Y-%m-%d')}")
    out_path = os.path.join(date_dir, filename)
    _ensure_dir(out_path)
    try:
        df.to_parquet(out_path, index=False, compression=compression)
    except Exception:
        out_path = os.path.splitext(out_path)[0] + ".csv"
        df.to_csv(out_path, index=False)
    if also_csv and not out_path.endswith(".csv"):
        csv_path = os.path.splitext(out_path)[0] + ".csv"
        df.to_csv(csv_path, index=False)
    if schema_meta is not None:
        meta_path = os.path.join(date_dir, os.path.splitext(filename)[0] + ".schema.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(schema_meta, f, indent=2)
    return out_path


def build_daily_baselines(daily_hist: pd.DataFrame, windows_days: Iterable[int] = REGIME_WINDOWS_DAYS) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build per-symbol baselines for regime metrics on daily bars.

    Returns: mapping symbol -> {metric_name -> {p01..p99, n}}
    """
    from .core import quantiles_summary  # reuse

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    if daily_hist.empty:
        return out
    if "timestamp" in daily_hist.columns:
        daily_hist = daily_hist.copy()
        daily_hist["timestamp"] = pd.to_datetime(daily_hist["timestamp"], utc=True, errors="coerce")

    for sym, g in daily_hist.groupby("symbol"):
        g = g.dropna(subset=["timestamp"]).sort_values("timestamp")
        g_daily = resample_to_daily(g)
        sym_base: Dict[str, Dict[str, float]] = {}
        for d in windows_days:
            sfx = f"{d}d"
            series_df = compute_regime_series(g_daily, d, sfx, breakout_lookback_days=20)
            if series_df.empty:
                continue
            metric_cols = [c for c in series_df.columns if c not in ("window_start", "window_end")]
            for col in metric_cols:
                vals = series_df[col].to_numpy(dtype=float)
                stats = quantiles_summary(vals)
                stats["n"] = float(np.isfinite(vals).sum())
                sym_base[col] = stats
        out[sym] = sym_base
    return out


def _schema_meta(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    def family(col: str) -> str:
        if col.endswith("_pctile"): return "pctile"
        if col.endswith("_quintile"): return "quintile"
        if col.endswith("_rz"): return "rz"
        if col.endswith("_abs_rz"): return "abs_rz"
        if col.endswith("_pos"): return "pos"
        if col.endswith("_mag"): return "mag"
        if col.endswith("_dir"): return "dir"
        if col.endswith("_imputed"): return "imputed_flag"
        if col.startswith("window_") or col.startswith("metrics_valid_"): return "flag"
        return "raw"

    def window(col: str) -> str:
        for sfx in ["3d","7d","14d","30d","90d"]:
            if f"_{sfx}" in col:
                return sfx
        return ""

    meta: Dict[str, Dict[str, str]] = {}
    for c in df.columns:
        meta[c] = {
            "dtype": str(df[c].dtype),
            "family": family(c),
            "window": window(c),
        }
    return meta


def impute_neutral_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def fill(col: str, neutral: float) -> None:
        if col not in out.columns:
            return
        mask = ~np.isfinite(out[col].to_numpy(dtype=float))
        if mask.any():
            out[col] = out[col].astype(float)
            out.loc[mask, col] = neutral
            out[f"{col}_imputed"] = 0
            out.loc[mask, f"{col}_imputed"] = 1
        else:
            out[f"{col}_imputed"] = 0

    for col in list(out.columns):
        if col == "symbol" or col.startswith("timestamp_asof"):
            continue
        s = out[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        cname = str(col)
        if cname.endswith("_pctile"): fill(cname, 50.0); continue
        if cname.endswith("_quintile"): fill(cname, 3); continue
        if cname.endswith("_rz") or cname.endswith("_abs_rz"): fill(cname, 0.0); continue
        if cname.endswith("_pos"): fill(cname, 0.5); continue
        if "_over_" in cname: fill(cname, 1.0); continue
        if "_minus_" in cname: fill(cname, 0.0); continue
        if cname.startswith("dist_") or ("dist_to_" in cname and cname.endswith("_mag")): fill(cname, 0.0); continue
        if cname.endswith("_mag"): fill(cname, 0.0); continue
        if cname.endswith("_dir"): fill(cname, 0.0); continue
        fill(cname, 0.0)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily Regime snapshot generator")
    p.add_argument("--input", default="habitual_overlaps.csv", help="Input cohort CSV path")
    p.add_argument("--out", default="data/regime", help="Output base directory (partitioned by dt=YYYY-MM-DD)")
    p.add_argument("--windows", help="CSV list of windows (e.g., 3d,7d,14d,30d,90d)")
    p.add_argument("--breakout-lookback-days", type=int, default=20, help="Lookback for breakout shares")
    p.add_argument("--asof", help="UTC ISO date (YYYY-MM-DD) for snapshot; else auto from data")
    p.add_argument("--compression", choices=["snappy", "zstd"], default="snappy")
    p.add_argument("--schema-version", default="v1", help="Schema version tag for sidecar metadata")
    p.add_argument("--dry-run", action="store_true", help="Print coverage and planned date; no write")
    return p.parse_args()


def _parse_windows_days(arg: Optional[str]) -> List[int]:
    if arg:
        parts = [p.strip().lower() for p in arg.split(",") if p.strip()]
        out: List[int] = []
        for p in parts:
            if p.endswith("d"):
                out.append(int(p[:-1]))
            else:
                out.append(int(p))
        return out
    return REGIME_WINDOWS_DAYS


def main() -> None:
    args = parse_args()
    windows = _parse_windows_days(args.windows)
    # Load
    df = pd.read_csv(args.input)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df.empty:
        print("No rows in input; nothing to do.")
        return
    asof = pd.to_datetime(args.asof, utc=True) if args.asof else df["timestamp"].max().floor("1D")
    if args.dry_run:
        # Simple symbol/day coverage count
        day_counts = df.assign(day=df["timestamp"].dt.floor("1D")).groupby(["symbol","day"]).size().groupby(level=0).size()
        print(f"Dry run: symbols={df['symbol'].nunique()} windows={','.join([str(d)+'d' for d in windows])} asof={asof.date().isoformat()} symbol_days={int(day_counts.sum())}")
        return

    # Compute
    snapshot = compute_regime_snapshot(
        cohort_csv=args.input,
        windows_days=windows,
        baselines=None,
        db_fetch_fn=None,
        db_engine=None,
        asof=asof,
        breakout_lookback_days=args.breakout_lookback_days,
    )
    # Sidecar schema
    schema_meta = _schema_meta(snapshot)
    dt = asof.date()
    out_path = write_regime_parquet(
        snapshot,
        utc_date=datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc),
        base_dir=args.out,
        filename="regime_snapshot.parquet",
        compression=args.compression,
        schema_meta={"schema_version": args.schema_version, "columns": schema_meta},
    )
    null_rate = float((~np.isfinite(snapshot.select_dtypes(include=[np.number]).to_numpy())).mean()) if not snapshot.empty else 0.0
    print(f"Regime snapshot: symbols={snapshot['symbol'].nunique()} windows={','.join([str(d)+'d' for d in windows])} asof={asof.date().isoformat()} rows={len(snapshot)} cols={snapshot.shape[1]} out={out_path} null_rate={null_rate:.4f}")


if __name__ == "__main__":
    main()
