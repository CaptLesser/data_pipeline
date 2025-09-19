import os
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, List, Tuple

import pandas as pd
import numpy as np
import json
import argparse

from .core import (
    WINDOWS_MINUTES,
    compute_symbol_metrics,
    build_symbol_baseline,
    enrich_current_with_baseline,
    resolve_input_path,
    normalize_ohlcv_columns,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def compute_state_snapshot(
    cohort_csv: str,
    months_history: int = 3,
    baselines: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    db_fetch_fn=None,
    db_engine=None,
    windows_minutes: Iterable[int] = WINDOWS_MINUTES,
    asof: Optional[datetime] = None,
    coverage_threshold: float = 0.8,
) -> pd.DataFrame:
    """Compute the Intraday State snapshot for symbols in cohort_csv.

    If baselines are provided (or db_fetch_fn+engine are given), enrich with
    percentiles/quintiles and robust-z.
    """
    # Robust load: resolve path + normalize columns
    path = resolve_input_path(cohort_csv)
    df = pd.read_csv(path)
    df = normalize_ohlcv_columns(df)
    if df.empty:
        return pd.DataFrame(columns=["symbol"])  # empty template
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Determine anchor (as-of) as the min across selected windows of floored max ts
    if asof is None:
        tmax = df["timestamp"].max()
        anchors = []
        for w in windows_minutes:
            rule = f"{w // 60}h" if w % 60 == 0 else f"{w}min"
            anchors.append(tmax.floor(rule))
        asof = min(anchors) if anchors else tmax

    rows = []
    for sym, g in df.groupby("symbol"):
        g = g[g["timestamp"] <= asof]
        metrics = compute_symbol_metrics(g, windows_minutes=windows_minutes)
        row = {"symbol": sym}
        row.update(metrics)
        rows.append(row)
    state_df = pd.DataFrame(rows)

    if baselines is None and db_fetch_fn is not None and db_engine is not None:
        symbols = state_df["symbol"].dropna().unique().tolist()
        hist_df = db_fetch_fn(db_engine, symbols, months_history)
        baselines = {sym: build_symbol_baseline(g) for sym, g in hist_df.groupby("symbol")}

    if baselines:
        state_df = enrich_current_with_baseline(state_df, baselines)
    # Impute neutral values per spec
    state_df = impute_neutral(state_df, baselines)
    return state_df


def _parse_windows(arg: Optional[str], enable_72h: bool) -> List[int]:
    if arg:
        parts = [p.strip().lower() for p in arg.split(",") if p.strip()]
        out: List[int] = []
        for p in parts:
            if p.endswith("h"):
                out.append(int(float(p[:-1]) * 60))
            elif p.endswith("m"):
                out.append(int(p[:-1]))
            else:
                out.append(int(p))
        return out
    # default
    ws = [60, 240, 720, 1440]
    if enable_72h:
        ws.append(4320)
    return ws


def _schema_meta(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    # Column → {dtype, family, window}
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
        for sfx in ["1h","4h","6h","12h","24h","72h","168h","3d","7d","14d","30d","90d"]:
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


def impute_neutral(df: pd.DataFrame, baselines: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None) -> pd.DataFrame:
    out = df.copy()
    # Helper to mark and fill
    def fill(col: str, neutral: float) -> None:
        mask = ~np.isfinite(out[col].to_numpy(dtype=float)) if col in out.columns else None
        if mask is None:
            return
        if mask.any():
            out[col] = out[col].astype(float)
            out.loc[mask, col] = neutral
            out[f"{col}_imputed"] = 0
            out.loc[mask, f"{col}_imputed"] = 1
        else:
            out[f"{col}_imputed"] = 0

    for col in list(out.columns):
        if col == "symbol" or col.startswith("timestamp_asof") or col.startswith("window_") or col.startswith("metrics_valid_"):
            continue
        s = out[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        cname = str(col)
        # Percentile/quintile/z family
        if cname.endswith("_pctile"):
            fill(cname, 50.0)
            continue
        if cname.endswith("_quintile"):
            fill(cname, 3)
            continue
        if cname.endswith("_rz") or cname.endswith("_abs_rz"):
            fill(cname, 0.0)
            continue
        if cname.endswith("_pos"):
            fill(cname, 0.5)
            continue
        if "_over_" in cname:
            fill(cname, 1.0)
            continue
        if "_minus_" in cname:
            fill(cname, 0.0)
            continue
        if cname.startswith("dist_") or ("dist_to_" in cname and cname.endswith("_mag")):
            fill(cname, 0.0)
            continue
        # magnitude columns: try baseline p50 if available, else 0.0
        if cname.endswith("_mag"):
            neutral = 0.0
            # if we have p50 column, use it
            p50_col = f"{cname}_p50"
            if p50_col in out.columns and pd.api.types.is_numeric_dtype(out[p50_col]):
                # If any NaN in p50, default to 0.0
                try:
                    neutral = float(out[p50_col].fillna(0.0).iloc[0])
                except Exception:
                    neutral = 0.0
            fill(cname, neutral)
            continue
        # directional numeric
        if cname.endswith("_dir"):
            fill(cname, 0.0)
            continue
        # For any other numeric, default-impute to 0.0
        fill(cname, 0.0)
    return out


def write_state_parquet(
    df: pd.DataFrame,
    utc_date: Optional[datetime] = None,
    base_dir: str = "data/state",
    filename: str = "state_snapshot.parquet",
    also_csv: bool = False,
    compression: str = "snappy",
    schema_meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """Write Intraday State to partitioned Parquet path.

    Returns the written Parquet file path.
    """
    if utc_date is None:
        utc_date = datetime.now(timezone.utc)
    date_dir = os.path.join(base_dir, f"dt={utc_date.strftime('%Y-%m-%d')}")
    out_path = os.path.join(date_dir, filename)
    _ensure_dir(out_path)
    try:
        df.to_parquet(out_path, index=False, compression=compression)
    except Exception:
        # Fallback to CSV if Parquet engine not available
        out_path = os.path.splitext(out_path)[0] + ".csv"
        df.to_csv(out_path, index=False)

    if also_csv and not out_path.endswith(".csv"):
        csv_path = os.path.splitext(out_path)[0] + ".csv"
        df.to_csv(csv_path, index=False)
    # Sidecar schema JSON
    if schema_meta is not None:
        meta_path = os.path.join(date_dir, os.path.splitext(filename)[0] + ".schema.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(schema_meta, f, indent=2)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intraday State snapshot generator")
    p.add_argument("--input", default="habitual_overlaps.csv", help="Input cohort CSV path")
    p.add_argument("--out", default="data/state", help="Output base directory (partitioned by dt=YYYY-MM-DD)")
    p.add_argument("--windows", help="CSV list of windows (e.g., 1h,4h,12h,24h)")
    p.add_argument("--enable-72h", dest="enable_72h", action="store_true", help="Enable 72h window")
    p.add_argument("--coverage-threshold", type=float, default=0.8, help="Minimum coverage fraction per window")
    p.add_argument("--asof", help="UTC ISO timestamp to anchor snapshot; else auto from data")
    p.add_argument("--compression", choices=["snappy", "zstd"], default="snappy")
    p.add_argument("--schema-version", default="v1", help="Schema version tag for sidecar metadata")
    p.add_argument("--dry-run", action="store_true", help="Print planned buckets and coverage summary, no write")
    # Optional baseline via DB
    p.add_argument("--months", type=int, default=3, help="Months of history for baselines")
    p.add_argument("--host", help="MySQL host")
    p.add_argument("--user", help="MySQL user")
    p.add_argument("--database", help="Database name")
    p.add_argument("--port", type=int, default=3306, help="MySQL port")
    p.add_argument("--password", help="MySQL password")
    p.add_argument("--table", default="ohlcvt", help="OHLCVT table name")
    p.add_argument("--no-baseline", action="store_true", help="Skip baseline enrichment")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    windows = _parse_windows(args.windows, args.enable_72h)
    asof = pd.to_datetime(args.asof, utc=True) if args.asof else None

    # Optionally dry-run: report planned anchor and coverage
    path0 = resolve_input_path(args.input)
    df = pd.read_csv(path0)
    df = normalize_ohlcv_columns(df)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df.empty:
        print("No rows in input; nothing to do.")
        return
    if asof is None:
        tmax = df["timestamp"].max()
        anchors = []
        for w in windows:
            rule = f"{w // 60}h" if w % 60 == 0 else f"{w}min"
            anchors.append(tmax.floor(rule))
        asof = min(anchors)

    if args.dry_run:
        # Compute simple coverage summaries
        covs: List[Tuple[str, int, float]] = []
        for sym, g in df.groupby("symbol"):
            for w in windows:
                start = asof - pd.to_timedelta(w, unit="m")
                cov = float(len(g[(g["timestamp"] > start) & (g["timestamp"] <= asof)])) / max(1, w)
                covs.append((sym, w, cov))
        # Print one-line summary
        print(f"Dry run: symbols={df['symbol'].nunique()} windows={','.join([str(int(w/60))+'h' for w in windows])} asof={asof.isoformat()} avg_coverage={np.mean([c for _,_,c in covs]):.3f}")
        return

    # Optional DB for baseline
    baselines = None
    if not args.no_baseline and args.host and args.user and args.database and args.password:
        from .db import get_db_engine, fetch_history_months
        engine = get_db_engine(args.host, args.user, args.password, args.database, args.port)
        symbols = df["symbol"].dropna().unique().tolist()
        hist_df = fetch_history_months(engine, symbols, args.months, table=args.table)
        engine.dispose()
        baselines = {sym: build_symbol_baseline(g) for sym, g in hist_df.groupby("symbol")}

    # Compute snapshot and enrichment
    snapshot = compute_state_snapshot(
        cohort_csv=path0,
        months_history=args.months,
        baselines=baselines,
        db_fetch_fn=None,
        db_engine=None,
        windows_minutes=windows,
        asof=asof,
        coverage_threshold=args.coverage_threshold,
    )

    # Schema sidecar
    schema_meta = _schema_meta(snapshot)
    dt = asof.date()
    out_path = write_state_parquet(
        snapshot,
        utc_date=datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc),
        base_dir=args.out,
        filename="state_snapshot.parquet",
        compression=args.compression,
        schema_meta={"schema_version": args.schema_version, "columns": schema_meta},
    )
    # Log summary
    null_rate = float((~np.isfinite(snapshot.select_dtypes(include=[np.number]).to_numpy())).mean()) if not snapshot.empty else 0.0
    print(f"State snapshot: symbols={snapshot['symbol'].nunique()} windows={','.join([str(int(w/60))+'h' for w in windows])} asof={asof.isoformat()} rows={len(snapshot)} cols={snapshot.shape[1]} out={out_path} null_rate={null_rate:.4f}")


if __name__ == "__main__":
    main()
