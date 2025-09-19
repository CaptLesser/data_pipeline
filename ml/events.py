from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelSpec:
    kind: Literal["breakout", "squeeze", "spike"]
    horizons: Tuple[int, ...]
    wins: Tuple[str, ...]
    params: Dict[str, float]
    prior_window: Optional[str] = None
    embargo_min: int = 0
    train_cutoff: Optional[pd.Timestamp] = None


@dataclass
class EventSpec:
    event: str  # breakout|squeeze|spike
    short_win: str = "1h"
    breakout_bps: float = 10.0
    squeeze_pct: float = 10.0
    squeeze_expand: float = 0.5
    spike_ret_pctl: float = 99.5
    spike_vol_pctl: float = 80.0


def _minutes_from_win(win: str) -> int:
    w = win.strip().lower()
    if w.endswith("h"):
        return int(float(w[:-1]) * 60)
    if w.endswith("min"):
        return int(w[:-3])
    if w.endswith("m"):
        return int(w[:-1])
    return int(w)


def _future_window_max(series: pd.Series, H: int) -> pd.Series:
    # Pure future max: max over (t+1..t+H)
    arr = series.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        j0 = i + 1
        j1 = min(n, i + H + 1)
        if j0 < j1:
            window = arr[j0:j1]
            # ignore NaNs in window
            if np.isfinite(window).any():
                out[i] = np.nanmax(window)
    return pd.Series(out, index=series.index)


def _group_future_max(df: pd.DataFrame, col: str, H: int) -> pd.Series:
    return df.groupby("symbol")[col].apply(lambda s: _future_window_max(s, H))


def _rolling_window_high(df: pd.DataFrame, minutes: int) -> pd.Series:
    # Prior window high using OHLCVT 'high', aligned to t (inclusive)
    # Require sufficient coverage (~80%) to consider valid
    minp = max(1, int(round(minutes * 0.8)))
    return (
        df.sort_values(["symbol", "timestamp"]).groupby("symbol")["high"].rolling(window=minutes, min_periods=minp).max().reset_index(level=0, drop=True)
    )


def compute_event_thresholds(
    train_df: pd.DataFrame, spec: EventSpec
) -> Dict[str, float]:
    # Compute training-split thresholds for squeeze/spike percentiles
    out: Dict[str, float] = {}
    W = spec.short_win
    bw_col = f"bollinger_width_{W}"
    vs_col = f"vol_spike_share_{W}_pos"
    # abs 1m returns
    ar1 = train_df.get("ret_1m")
    if isinstance(ar1, pd.Series):
        ar1 = ar1.abs()
        if np.isfinite(ar1).any():
            out["spike_ret_thr"] = float(np.nanpercentile(ar1.to_numpy(dtype=float), spec.spike_ret_pctl))
    if vs_col in train_df.columns:
        vs = pd.to_numeric(train_df[vs_col], errors="coerce")
        if np.isfinite(vs).any():
            out["spike_vol_thr"] = float(np.nanpercentile(vs.to_numpy(dtype=float), spec.spike_vol_pctl))
    if bw_col in train_df.columns:
        bw = pd.to_numeric(train_df[bw_col], errors="coerce")
        if np.isfinite(bw).any():
            out["squeeze_bw_low"] = float(np.nanpercentile(bw.to_numpy(dtype=float), spec.squeeze_pct))
    return out


def compute_event_labels(
    df: pd.DataFrame,
    horizons: Iterable[int],
    spec: EventSpec,
    thresholds: Dict[str, float],
) -> Dict[int, pd.Series]:
    labels: Dict[int, pd.Series] = {}
    Wm = _minutes_from_win(spec.short_win)
    # breakout reference high at t
    if spec.event == "breakout":
        ref_high = _rolling_window_high(df, Wm)
        bp = spec.breakout_bps / 10000.0
        for H in horizons:
            max_future_close = _group_future_max(df, "close", H)
            y = (max_future_close >= (ref_high * (1.0 + bp))).astype(float)
            # invalid ref_high rows -> NaN
            y[~np.isfinite(ref_high)] = np.nan
            labels[H] = y
        return labels

    # squeeze
    if spec.event == "squeeze":
        bw_col = f"bollinger_width_{spec.short_win}"
        if bw_col not in df.columns:
            # cannot label
            for H in horizons:
                labels[H] = pd.Series(np.nan, index=df.index)
            return labels
        bw_t = pd.to_numeric(df[bw_col], errors="coerce")
        bw_low = thresholds.get("squeeze_bw_low", np.nan)
        # guard invalid current bandwidth
        valid_now = np.isfinite(bw_t)
        for H in horizons:
            max_future_bw = _group_future_max(df.assign(_bw=bw_t.rename("_bw")), "_bw", H)
            expand_ok = max_future_bw >= (bw_t * (1.0 + float(spec.squeeze_expand)))
            y = (valid_now & (bw_t <= bw_low) & expand_ok).astype(float)
            y[~valid_now | ~np.isfinite(bw_low)] = np.nan
            labels[H] = y
        return labels

    # spike
    if spec.event == "spike":
        # abs 1m return now and in future
        ar1 = pd.to_numeric(df.get("ret_1m"), errors="coerce").abs()
        vs_col = f"vol_spike_share_{spec.short_win}_pos"
        vs = pd.to_numeric(df[vs_col], errors="coerce") if vs_col in df.columns else pd.Series(np.nan, index=df.index)
        ret_thr = float(thresholds.get("spike_ret_thr", np.nan))
        vol_thr = float(thresholds.get("spike_vol_thr", np.nan))
        for H in horizons:
            max_future_ar1 = _group_future_max(df.assign(_ar1=ar1.rename("_ar1")), "_ar1", H)
            max_future_vs = _group_future_max(df.assign(_vs=vs.rename("_vs")), "_vs", H) if vs_col in df.columns else pd.Series(np.nan, index=df.index)
            cond_ret = max_future_ar1 >= ret_thr if np.isfinite(ret_thr) else pd.Series(False, index=df.index)
            cond_vol = max_future_vs >= vol_thr if (vs_col in df.columns and np.isfinite(vol_thr)) else pd.Series(False, index=df.index)
            y = (cond_ret | cond_vol).astype(float)
            # mark NaN when both sides invalid
            invalid = (~np.isfinite(max_future_ar1)) & (~np.isfinite(max_future_vs))
            y[invalid] = np.nan
            labels[H] = y
        return labels

    # unknown event
    for H in horizons:
        labels[H] = pd.Series(np.nan, index=df.index)
    return labels
