from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DatasetSpec:
    horizons: List[int]
    short_win: str = "1h"
    dense: bool = True
    windows: Optional[List[str]] = None  # e.g., ["1h","4h","12h","24h"]
    require_wins: Optional[List[str]] = None  # windows that must be valid
    embargo_minutes: int = 60
    cross_asset: Optional[str] = None  # ranks|spreads|both (optional)
    base_symbol: str = "BTCUSDT"
    calendar: bool = False
    calendar_extended: bool = False
    rank_top: Optional[int] = None
    liq_col: str = "quote_volume_sum_24h_mag"
    liq_threshold: Optional[float] = None  # percentile (0..100)
    base_missing: str = "drop"  # drop|ffill|error


def _to_utc(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True, errors="coerce")


def load_ohlcvt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    # Normalize expected columns (simple case: files in repo already match schema)
    cols = {c.lower(): c for c in df.columns}
    need = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in cols]
    # If any missing, try case-insensitive rename
    if missing:
        rename = {}
        for c in need:
            for k, v in list(cols.items()):
                if k == c:
                    rename[v] = c
        df = df.rename(columns=rename)
    df["timestamp"] = _to_utc(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values(["symbol", "timestamp"])  # type: ignore
    return df


def load_metrics_series(path: str, windows: Optional[Iterable[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    # Expect long format with 'window' column and 'timestamp' == window_end
    if "timestamp" not in df.columns:
        raise ValueError("metrics_series CSV must contain 'timestamp' column")
    df["timestamp"] = _to_utc(df["timestamp"])  # window_end
    if windows:
        if "window" not in df.columns:
            # Heuristic: keep columns whose suffix matches requested windows
            # but still filter by available rows
            pass
        else:
            sel = set([w.strip().lower() for w in windows])
            df = df[df["window"].astype(str).str.lower().isin(sel)]
    df = df.dropna(subset=["timestamp"]).sort_values(["symbol", "timestamp"])  # type: ignore
    return df


def _merge_asof_per_window(base: pd.DataFrame, mseries: pd.DataFrame, windows: Iterable[str]) -> pd.DataFrame:
    """For each window suffix, asof-merge metrics into the minute grid.

    base: per-minute OHLCVT rows (symbol, timestamp, open..)
    mseries: long-format metrics rows with 'window' and 'timestamp'
    """
    base = base.copy()
    # speed/memory: category for symbol and pre-sort once
    if base["symbol"].dtype != "category":
        base["symbol"] = base["symbol"].astype("category")
    if "symbol" in mseries.columns and mseries["symbol"].dtype != "category":
        mseries["symbol"] = mseries["symbol"].astype("category")
    out = base.sort_values(["symbol", "timestamp"]).copy()
    if mseries.empty:
        return out
    if "window" not in mseries.columns:
        # Assume all rows correspond to a single window already (rare)
        win_dfs = {"unknown": mseries}
    else:
        win_dfs = {
            w: mseries[mseries["window"].astype(str).str.lower() == w.lower()].copy()
            for w in windows
            if not mseries.empty
        }
    # columns to keep from metrics (exclude id/time/helper cols)
    drop_cols = {"window", "window_start", "window_end", "timestamp"}
    for w, dfw in win_dfs.items():
        if dfw.empty:
            continue
        keep_cols = [c for c in dfw.columns if c not in drop_cols]
        dfw = dfw[["symbol", "timestamp"] + keep_cols].sort_values(["symbol", "timestamp"])  # type: ignore
        out = pd.merge_asof(
            out,
            dfw,
            by="symbol",
            on="timestamp",
            direction="backward",
        )
    return out


def _apply_filters(df: pd.DataFrame, require_wins: Optional[List[str]], short_win: str) -> pd.DataFrame:
    if df.empty:
        return df
    wins_req = require_wins or [short_win]
    mask = pd.Series(True, index=df.index)
    for w in wins_req:
        wk = w.lower()
        ok_col = f"window_ok_{wk}"
        if ok_col in df.columns:
            mask &= df[ok_col] == True  # noqa: E712
        # family valid flags
        for fam in ["volume", "volatility", "vwap", "bbands", "trend", "range", "momentum"]:
            col = f"metrics_valid_{fam}_{wk}"
            if col in df.columns:
                mask &= df[col] == True  # noqa: E712
    return df[mask].copy()


def _add_calendar(df: pd.DataFrame, extended: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    t = df["timestamp"].dt.tz_convert("UTC") if df["timestamp"].dt.tz is not None else df["timestamp"]
    df["minute_of_hour"] = t.dt.minute
    df["hour_of_day"] = t.dt.hour
    df["day_of_week"] = t.dt.dayofweek
    if extended:
        df["day_of_month"] = t.dt.day
        df["month"] = t.dt.month
    return df


def _add_ret_1m(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["symbol", "timestamp"]).copy()
    df["close_shift1"] = df.groupby("symbol")["close"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ret_1m"] = np.log(df["close"]) - np.log(df["close_shift1"])
    df = df.drop(columns=["close_shift1"])
    return df


def _make_lags(df: pd.DataFrame, cols: Iterable[str], k: int) -> Tuple[pd.DataFrame, List[str]]:
    lag_cols: List[str] = []
    if df.empty or k <= 0:
        return df, lag_cols
    for c in cols:
        if c not in df.columns:
            continue
        for i in range(1, k + 1):
            name = f"{c}_lag{i}"
            df[name] = df.groupby("symbol")[c].shift(i)
            lag_cols.append(name)
    return df, lag_cols


def build_dense_dataset(
    ohlcvt_csv: str,
    metrics_series_csv: str,
    spec: DatasetSpec,
    lags: int,
    features_preset: str,
    short_win: str,
    windows: List[str],
    include_cols_regex: Optional[str] = None,
    exclude_cols_regex: Optional[str] = None,
    symbols_filter: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Return a dense minute-level dataset with features ready for label creation.

    Returns: (df, info) where info contains lists of feature columns by group.
    """
    base = load_ohlcvt(ohlcvt_csv)
    if symbols_filter:
        base = base[base["symbol"].astype(str).isin(list(symbols_filter))]
    if base.empty:
        raise SystemExit("OHLCVT input is empty")
    mser = load_metrics_series(metrics_series_csv, windows=windows)
    if mser.empty:
        raise SystemExit("metrics_series input is empty. Generate it with --emit-series.")

    # As-of merge per window
    df = _merge_asof_per_window(base, mser, windows)
    row_acc = {"post_merge": int(len(df))}

    # Filters on validity
    df = _apply_filters(df, spec.require_wins, short_win)
    row_acc["post_hygiene"] = int(len(df))

    # Calendar features
    if spec.calendar:
        df = _add_calendar(df, extended=spec.calendar_extended)

    # 1-minute return baseline
    df = _add_ret_1m(df)

    # Cross-asset features (optional)
    universe_mask: Optional[pd.DataFrame] = None
    if spec.cross_asset:
        df, universe_mask = _add_cross_asset_features(df, spec)

    # Feature presets
    # min preset: lags of close + ret_1m + volume, and current rsi/atr at short_win if present
    feat_cols: List[str] = []
    info: Dict[str, List[str]] = {"base": [], "lags": [], "metrics": []}
    if features_preset == "min":
        base_cols = [c for c in ["close", "volume", "ret_1m"] if c in df.columns]
        info["base"] = base_cols
        df, lag_cols = _make_lags(df, base_cols, lags)
        info["lags"] = lag_cols
        # add momentum/vol if present
        for c in [f"rsi_{short_win}", f"atr_{short_win}"]:
            if c in df.columns:
                feat_cols.append(c)
        feat_cols.extend(lag_cols)
    else:
        # max preset: all numeric metric columns excluding hygiene/labels and identifiers
        drop_prefixes = (
            "window_coverage_",
            "window_ok_",
            "metrics_valid_",
            "window_start",
            "window_end",
        )
        for c in df.columns:
            if c in ("symbol", "timestamp", "open", "high", "low"):
                continue
            if any(c.startswith(p) for p in drop_prefixes):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                feat_cols.append(c)
        # add lags for a subset with reasonable size
        lag_bases = [x for x in ["close", "volume", "ret_1m"] if x in df.columns]
        df, lag_cols = _make_lags(df, lag_bases, min(lags, 60))
        feat_cols.extend(lag_cols)
        info["lags"] = lag_cols
    # Apply include/exclude regex filters if provided
    import re
    if include_cols_regex:
        pat = re.compile(include_cols_regex)
        feat_cols = [c for c in feat_cols if pat.search(c)]
    if exclude_cols_regex:
        pat = re.compile(exclude_cols_regex)
        feat_cols = [c for c in feat_cols if not pat.search(c)]
    # Deterministic feature ordering
    feat_cols = sorted(dict.fromkeys(feat_cols))
    info["metrics"] = [c for c in feat_cols if c not in info.get("lags", [])]

    # Fill any remaining inf/-inf to NaN; leave NaNs for model-specific handling or later drop
    for c in feat_cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Drop rows with any NaN in selected features (dense runs favor completeness)
    feat_cols = [c for c in feat_cols if c in df.columns]
    row_acc["pre_dropna"] = int(len(df))
    df = df.dropna(subset=feat_cols)
    row_acc["post_dropna"] = int(len(df))

    info_dict: Dict[str, List[str]] = {"features": feat_cols}
    if universe_mask is not None:
        info_dict["_universe_csv"] = []  # signal presence
        # attach as attribute for caller access (train CLI will persist it)
        df.__dict__["_universe_mask_df"] = universe_mask
    # attach row accounting
    df.__dict__["_row_accounting"] = row_acc
    return df, info_dict


def _parse_win_to_minutes(w: str) -> int:
    s = w.strip().lower()
    if s.endswith("h"):
        return int(float(s[:-1]) * 60)
    if s.endswith("min"):
        return int(s[:-3])
    if s.endswith("m"):
        return int(s[:-1])
    return int(s)


def _add_cross_asset_features(df: pd.DataFrame, spec: DatasetSpec) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if df.empty:
        return df, None
    df = df.copy()
    # Liquidity filter universe per timestamp (optional)
    liq_col = spec.liq_col
    have_liq = liq_col in df.columns
    # Rank universe mask per ts
    if spec.rank_top is not None and have_liq:
        df["_rank_universe"] = False
        for ts, g in df.groupby("timestamp"):
            topn = g.nlargest(int(spec.rank_top), liq_col).index
            df.loc[topn, "_rank_universe"] = True
        rank_mask = df["_rank_universe"]
    elif spec.liq_threshold is not None and have_liq:
        # per-timestamp percentile threshold
        df["_rank_universe"] = False
        for ts, g in df.groupby("timestamp"):
            thr = np.nanpercentile(pd.to_numeric(g[liq_col], errors="coerce").to_numpy(dtype=float), float(spec.liq_threshold))
            df.loc[g.index, "_rank_universe"] = pd.to_numeric(g[liq_col], errors="coerce") >= thr
        rank_mask = df["_rank_universe"]
    else:
        rank_mask = pd.Series(True, index=df.index)

    # Ranks/zscores for default columns at 1h
    rank_cols = [
        "returns_log_1h",
        "rv_close_1h_mag",
        "rsi_1h",
        "adx_1h",
        "bollinger_width_1h",
    ]
    for c in rank_cols:
        if c not in df.columns:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        # Apply only within universe per timestamp
        # rank_pct
        grp = vals.where(rank_mask)
        rank_pct = grp.groupby(df["timestamp"]).rank(pct=True)
        df[f"rank_{c}"] = rank_pct
        # zscore per timestamp
        gmean = grp.groupby(df["timestamp"]).transform("mean")
        gstd = grp.groupby(df["timestamp"]).transform("std")
        z = (vals - gmean) / (gstd + 1e-12)
        # winsorize at 3σ for stability
        df[f"z_{c}"] = z.clip(lower=-3.0, upper=3.0)

    # Spreads vs base symbol
    if spec.cross_asset in ("spreads", "both"):
        base = spec.base_symbol
        base_df = df[df["symbol"] == base].sort_values("timestamp").copy()
        # Merge-asof base close onto all timestamps for controlled ffill
        ts_df = df[["timestamp"]].drop_duplicates().sort_values("timestamp").copy()
        close_asof = pd.merge_asof(ts_df, base_df[["timestamp", "close"]].rename(columns={"timestamp": "_ts_base"}).sort_values("_ts_base"), left_on="timestamp", right_on="_ts_base", direction="backward")
        # Compute gap minutes for ffill guard
        gap_min = (close_asof["timestamp"] - close_asof["_ts_base"]).dt.total_seconds() / 60.0
        close_asof["close_base"] = close_asof["close"]
        if spec.base_missing == "ffill":
            lim = getattr(spec, "base_ffill_max_minutes", 5)
            close_asof.loc[(gap_min > float(lim)) | (~np.isfinite(gap_min)), "close_base"] = np.nan
        elif spec.base_missing == "drop":
            # leave NaNs for dropping later
            pass
        elif spec.base_missing == "error":
            if close_asof["close"].isna().any():
                raise ValueError("Base symbol missing for some timestamps; use --base-missing ffill or drop.")
        df = df.merge(close_asof[["timestamp", "close_base"]], on="timestamp", how="left")
        df["spread_close"] = pd.to_numeric(df.get("close"), errors="coerce") - pd.to_numeric(df.get("close_base"), errors="coerce")
        # returns_log_1h asof if present on base
        if "returns_log_1h" in df.columns and "returns_log_1h" in base_df.columns:
            ret_asof = pd.merge_asof(ts_df, base_df[["timestamp", "returns_log_1h"]].rename(columns={"timestamp": "_ts_base_ret", "returns_log_1h": "returns_log_1h_base"}).sort_values("_ts_base_ret"), left_on="timestamp", right_on="_ts_base_ret", direction="backward")
            gap_min_r = (ret_asof["timestamp"] - ret_asof["_ts_base_ret"]).dt.total_seconds() / 60.0
            if spec.base_missing == "ffill":
                lim = getattr(spec, "base_ffill_max_minutes", 5)
                ret_asof.loc[(gap_min_r > float(lim)) | (~np.isfinite(gap_min_r)), "returns_log_1h_base"] = np.nan
            df = df.merge(ret_asof[["timestamp", "returns_log_1h_base"]], on="timestamp", how="left")
            df["spread_returns_log_1h"] = pd.to_numeric(df.get("returns_log_1h"), errors="coerce") - pd.to_numeric(df.get("returns_log_1h_base"), errors="coerce")
        # Spread vol over 1h
        if "spread_close" in df.columns:
            df = df.sort_values(["symbol", "timestamp"]).copy()
            d_spread = df.groupby("symbol")["spread_close"].diff()
            df["spread_vol_1h"] = d_spread.groupby(df["symbol"]).rolling(window=60, min_periods=30).std().reset_index(level=0, drop=True)
        # Drop rows per base_missing
        if spec.base_missing == "drop":
            drop_cols = [c for c in ["close_base", "returns_log_1h_base"] if c in df.columns]
            if drop_cols:
                df = df.dropna(subset=drop_cols)
    # Build universe mask dataframe
    uni_df = None
    if "_rank_universe" in df.columns:
        uni_df = df[["symbol", "timestamp", "_rank_universe"]].rename(columns={"_rank_universe": "in_universe"}).copy()

    # Clean temp cols
    for col in ["_rank_universe", "close_base", "returns_log_1h_base"]:
        if col in df.columns:
            try:
                df = df.drop(columns=[col])
            except Exception:
                pass
    return df, uni_df


def compute_labels(
    df: pd.DataFrame,
    horizons: Iterable[int],
    task: str,
    dir_bps: int = 5,
    classes: int = 3,
) -> Dict[int, pd.Series]:
    labels: Dict[int, pd.Series] = {}
    if df.empty:
        return labels
    df = df.sort_values(["symbol", "timestamp"]).copy()
    eps = 1e-12
    thresh = dir_bps / 10000.0  # bps to decimal
    for H in horizons:
        # shift future close within symbol
        fut = df.groupby("symbol")["close"].shift(-H)
        if task == "price":
            y = fut
        elif task == "return":
            y = np.log((fut + eps) / (df["close"] + eps))
        elif task == "direction":
            r = np.log((fut + eps) / (df["close"] + eps))
            if classes == 2:
                # up vs. non-up
                y = (r >= thresh).astype(int)
            else:
                y = pd.Series(np.where(r >= thresh, 2, np.where(r <= -thresh, 0, 1)), index=df.index)
        elif task == "vol":
            # std of 1m returns over (t+1..t+H)
            ret1 = df.groupby("symbol")["ret_1m"].shift(-1)
            roll = (
                ret1.groupby(df["symbol"]).rolling(window=H, min_periods=max(2, int(H * 0.8))).std().reset_index(level=0, drop=True)
            )
            y = roll
        elif task == "event":
            # Placeholder: events wired later
            y = pd.Series(np.nan, index=df.index)
        else:
            raise ValueError(f"Unknown task: {task}")
        labels[H] = y
    return labels
