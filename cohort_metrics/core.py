import math
from typing import Dict, Iterable, List, Tuple, Optional
import os
import difflib

import numpy as np
import pandas as pd


# Window sizes expressed in minutes (since input is 1-minute bars)
# Intraday State defaults
WINDOWS_MINUTES = [60, 240, 720, 1440]  # 1h, 4h, 12h, 24h (72h optional)
SUFFIX = {
    60: "1h",
    240: "4h",
    720: "12h",
    1440: "24h",
    4320: "72h",
}

# Resample map to target ~60–80 bars per window (resample-first default)
RESAMPLE_MAP_MIN = {
    60: 1,     # 1h → 1m
    240: 3,    # 4h → 3m
    720: 10,   # 12h → 10m
    1440: 20,  # 24h → 20m
    4320: 60,  # 72h → 60m
    360: 5,    # 6h → 5m (legacy)
    10080: 120 # 168h → 120m (legacy)
}

# Global epsilon
EPS = 1e-9


def _safe_div(a: float, b: float) -> float:
    if b == 0 or pd.isna(b):
        return float("nan")
    return a / b


def _ema(series: pd.Series, span: int) -> float:
    if series.empty:
        return float("nan")
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _rsi(close: pd.Series, period: int = 14) -> float:
    """RSI using Wilder's smoothing (EMA-like)."""
    if len(close) < period + 1:
        return float("nan")
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    val = rsi.iloc[-1]
    if pd.isna(val):
        return float("nan")
    return float(val)


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> Dict[str, float]:
    if close.empty:
        return {"upper": float("nan"), "lower": float("nan"), "width": float("nan")}
    ma = close.rolling(window=period, min_periods=period).mean()
    sd = close.rolling(window=period, min_periods=period).std(ddof=1)
    mu = ma.iloc[-1] if not pd.isna(ma.iloc[-1]) else close.mean()
    sigma = sd.iloc[-1] if not pd.isna(sd.iloc[-1]) else close.std(ddof=1)
    upper = mu + num_std * sigma
    lower = mu - num_std * sigma
    width = upper - lower
    return {"upper": float(upper), "lower": float(lower), "width": float(width)}


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Wilder ATR (EWMA with alpha=1/period)."""
    if len(close) < 2:
        return float("nan")
    tr = _true_range(high, low, close).dropna()
    if tr.empty:
        return float("nan")
    atr_series = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return float(atr_series.iloc[-1]) if len(atr_series) else float("nan")


def _stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, d_period: int = 3) -> Dict[str, float]:
    if close.empty:
        return {"k": float("nan"), "d": float("nan")}
    ll = low.rolling(window=period, min_periods=period).min()
    hh = high.rolling(window=period, min_periods=period).max()
    denom = (hh - ll)
    denom = denom.replace(0, np.nan)
    k_series = ((close - ll) / denom) * 100.0
    k_val = float(k_series.iloc[-1]) if not pd.isna(k_series.iloc[-1]) else float("nan")
    d_series = k_series.rolling(window=d_period, min_periods=1).mean()
    d_val = float(d_series.iloc[-1]) if len(d_series) else float("nan")
    return {"k": k_val, "d": d_val}


def _obv(close: pd.Series, volume: pd.Series) -> float:
    if close.empty or volume.empty:
        return float("nan")
    sign = np.sign(close.diff().fillna(0.0))
    obv = (sign * volume).cumsum()
    return float(obv.iloc[-1])


def _cci(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    if high.empty:
        return float("nan")
    tp = (high + low + close) / 3.0
    sma = tp.mean()
    mad = np.mean(np.abs(tp - sma))
    if mad == 0:
        return float("nan")
    return float((tp.iloc[-1] - sma) / (0.015 * mad))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Classic ADX via Wilder smoothing and DX."""
    if len(close) < period + 1:
        return float("nan")
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0.0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0.0)
    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    val = adx.iloc[-1]
    return float(val) if pd.notna(val) else float("nan")


def _asl(x: np.ndarray) -> np.ndarray:
    """Signed log transform: sign(x) * log1p(|x|)."""
    return np.sign(x) * np.log1p(np.abs(x))


def _mad(x: pd.Series) -> float:
    med = x.median()
    return float((np.abs(x - med)).median())


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(rule).agg(agg).dropna(how="all")


def _target_resample_rule(window_size: int) -> Optional[str]:
    mins = RESAMPLE_MAP_MIN.get(window_size)
    if mins is None:
        return None
    if mins % 60 == 0:
        return f"{mins // 60}h"
    return f"{mins}min"


def compute_window_metrics(
    df: pd.DataFrame,
    window_size: int,
    suffix: str,
    min_fraction: float = 0.8,
    resample_first: bool = True,
    scale_lookbacks: bool = False,
    prealigned: bool = False,
) -> Dict[str, float]:
    """Compute a set of technical/statistical features on the last N rows.

    Parameters
    ----------
    df : DataFrame with columns open, high, low, close, volume
    window_size : int, number of rows (minutes)
    suffix : string appended to column names (e.g., '6h')
    """
    if df.empty:
        return {}

    # Ensure timestamp index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("compute_window_metrics expects timestamp index or column")
        df = df.sort_index()

    # Align to last complete UTC bucket (non-overlapping) unless pre-aligned
    if prealigned:
        win = df
    else:
        bucket_rule = f"{window_size // 60}h" if window_size % 60 == 0 else f"{window_size}min"
        end_bucket = df.index.max().floor(bucket_rule)
        win_start = end_bucket - pd.to_timedelta(window_size, unit="m")
        win = df[(df.index > win_start) & (df.index <= end_bucket)]

    # Coverage
    coverage = len(win) / max(1, window_size)
    window_ok = coverage >= min_fraction
    # If poor coverage, still emit full schema with NaNs to keep columns stable
    if len(win) == 0:
        def _empty_window_metrics(sfx: str) -> Dict[str, float]:
            keys = [
                # Legacy metrics
                f"close_mean_{sfx}", f"close_stddev_{sfx}", f"close_min_{sfx}", f"close_max_{sfx}",
                f"volume_sum_{sfx}", f"vwap_{sfx}", f"price_deviation_vwap_{sfx}", f"roc_{sfx}", f"returns_log_{sfx}",
                f"sma_{sfx}", f"ema_fast_{sfx}", f"ema_slow_{sfx}", f"macd_{sfx}", f"macd_signal_{sfx}", f"macd_histogram_{sfx}",
                f"bollinger_upper_{sfx}", f"bollinger_lower_{sfx}", f"bollinger_width_{sfx}", f"rsi_{sfx}", f"atr_{sfx}",
                f"stochastic_k_{sfx}", f"stochastic_d_{sfx}", f"obv_{sfx}", f"cci_{sfx}", f"adx_{sfx}",
                # Opportunity primitives
                f"quote_volume_sum_{sfx}_mag", f"up_volume_share_{sfx}_pos", f"down_volume_share_{sfx}_pos",
                f"vol_spike_share_{sfx}_pos", f"rv_close_{sfx}_mag", f"parkinson_vol_{sfx}_mag", f"garman_klass_{sfx}_mag", f"yang_zhang_{sfx}_mag",
                f"squeeze_index_{sfx}_mag", f"range_intensity_{sfx}_mag", f"bb_position_{sfx}_pos",
                f"dist_to_window_high_{sfx}_dir", f"dist_to_window_low_{sfx}_dir",
                f"dist_to_window_high_abs_{sfx}_mag", f"dist_to_window_low_abs_{sfx}_mag",
                f"ret_share_pos_{sfx}_pos", f"run_up_max_{sfx}_mag", f"run_dn_max_{sfx}_mag", f"vwap_pos_share_{sfx}_pos",
                f"vwap_deviation_abs_{sfx}_mag",
            ]
            out_empty = {k: float("nan") for k in keys}
            # Hygiene + family validity flags
            out_empty.update({
                f"window_coverage_{sfx}": 0.0,
                f"window_ok_{sfx}": False,
                f"metrics_valid_volume_{sfx}": False,
                f"metrics_valid_volatility_{sfx}": False,
                f"metrics_valid_vwap_{sfx}": False,
                f"metrics_valid_bbands_{sfx}": False,
                f"metrics_valid_trend_{sfx}": False,
                f"metrics_valid_range_{sfx}": False,
                f"metrics_valid_momentum_{sfx}": False,
            })
            return out_empty

        return _empty_window_metrics(suffix)

    if not window_ok:
        # Emit full schema with NaNs but include flags
        def _empty_window_metrics(sfx: str) -> Dict[str, float]:
            keys = [
                f"close_mean_{sfx}", f"close_stddev_{sfx}", f"close_min_{sfx}", f"close_max_{sfx}",
                f"volume_sum_{sfx}", f"vwap_{sfx}", f"price_deviation_vwap_{sfx}", f"roc_{sfx}", f"returns_log_{sfx}",
                f"sma_{sfx}", f"ema_fast_{sfx}", f"ema_slow_{sfx}", f"macd_{sfx}", f"macd_signal_{sfx}", f"macd_histogram_{sfx}",
                f"bollinger_upper_{sfx}", f"bollinger_lower_{sfx}", f"bollinger_width_{sfx}", f"rsi_{sfx}", f"atr_{sfx}",
                f"stochastic_k_{sfx}", f"stochastic_d_{sfx}", f"obv_{sfx}", f"cci_{sfx}", f"adx_{sfx}",
                f"quote_volume_sum_{sfx}_mag", f"up_volume_share_{sfx}_pos", f"down_volume_share_{sfx}_pos",
                f"vol_spike_share_{sfx}_pos", f"rv_close_{sfx}_mag", f"parkinson_vol_{sfx}_mag", f"garman_klass_{sfx}_mag", f"yang_zhang_{sfx}_mag",
                f"squeeze_index_{sfx}_mag", f"range_intensity_{sfx}_mag", f"bb_position_{sfx}_pos",
                f"dist_to_window_high_{sfx}_dir", f"dist_to_window_low_{sfx}_dir",
                f"dist_to_window_high_abs_{sfx}_mag", f"dist_to_window_low_abs_{sfx}_mag",
                f"ret_share_pos_{sfx}_pos", f"run_up_max_{sfx}_mag", f"run_dn_max_{sfx}_mag", f"vwap_pos_share_{sfx}_pos",
                f"vwap_deviation_abs_{sfx}_mag",
            ]
            out_empty = {k: float("nan") for k in keys}
            out_empty.update({
                f"window_coverage_{sfx}": float(coverage),
                f"window_ok_{sfx}": False,
                f"metrics_valid_volume_{sfx}": False,
                f"metrics_valid_volatility_{sfx}": False,
                f"metrics_valid_vwap_{sfx}": False,
                f"metrics_valid_bbands_{sfx}": False,
                f"metrics_valid_trend_{sfx}": False,
                f"metrics_valid_range_{sfx}": False,
                f"metrics_valid_momentum_{sfx}": False,
            })
            return out_empty
        return _empty_window_metrics(suffix)

    # Resample-first mechanics
    if resample_first:
        rule = _target_resample_rule(window_size)
        if rule is not None:
            rwin = _resample_ohlcv(win, rule)
        else:
            rwin = win.copy()
    else:
        rwin = win.copy()

    c = rwin["close"].astype(float)
    h = rwin["high"].astype(float)
    l = rwin["low"].astype(float)
    v = rwin["volume"].astype(float)

    last = float(c.iloc[-1]) if len(c) else float("nan")
    first = float(c.iloc[0]) if len(c) else float("nan")

    close_mean = float(c.mean()) if len(c) else float("nan")
    close_std = float(c.std(ddof=1)) if len(c) > 1 else float("nan")
    close_min = float(c.min()) if len(c) else float("nan")
    close_max = float(c.max()) if len(c) else float("nan")

    v_sum = float(v.sum()) if len(v) else float("nan")
    # Window-anchored VWAP using typical price
    tp = (h + l + c) / 3.0
    tpv = (tp * v).sum()
    vwap = float(_safe_div(tpv, v.sum())) if len(v) else float("nan")
    dev_vwap = float(_safe_div(last - vwap, vwap)) if math.isfinite(vwap) else float("nan")

    roc = float(_safe_div(last - first, first) * 100.0) if math.isfinite(first) else float("nan")
    # Guard against zeros: safe log returns
    log_rets = np.log(np.maximum(c, EPS)).diff()
    log_ret_sum = float(log_rets.sum()) if len(log_rets) else float("nan")

    # Indicator lookbacks
    if not resample_first and scale_lookbacks:
        window_bars = max(1, len(c))
        rsi_period = max(2, int(round(window_bars / 4)))
        macd_fast = max(2, int(round(window_bars / 6)))
        macd_slow = max(macd_fast + 1, int(round(window_bars / 3)))
        macd_signal = max(2, int(round(window_bars / 4.5)))
        bb_period = max(5, int(round(window_bars / 3)))
        atr_period = max(5, int(round(window_bars / 4)))
        adx_period = max(5, int(round(window_bars / 4)))
    else:
        rsi_period = 14
        macd_fast = 12
        macd_slow = 26
        macd_signal = 9
        bb_period = 20
        atr_period = 14
        adx_period = 14

    sma = close_mean
    ema_fast_val = _ema(c, span=macd_fast)
    ema_slow_val = _ema(c, span=macd_slow)
    macd_val = float(ema_fast_val - ema_slow_val) if (math.isfinite(ema_fast_val) and math.isfinite(ema_slow_val)) else float("nan")
    macd_series = c.ewm(span=macd_fast, adjust=False).mean() - c.ewm(span=macd_slow, adjust=False).mean()
    macd_signal_val = float(macd_series.ewm(span=macd_signal, adjust=False).mean().iloc[-1]) if len(macd_series) else float("nan")
    macd_hist = float(macd_val - macd_signal_val) if (math.isfinite(macd_val) and math.isfinite(macd_signal_val)) else float("nan")

    bb = _bollinger(c, period=bb_period, num_std=2.0)
    rsi_val = _rsi(c, period=rsi_period)
    atr_val = _atr(h, l, c, period=atr_period)
    st = _stochastic_kd(h, l, c, period=14, d_period=3)
    obv_val = _obv(c, v)
    cci_val = _cci(h, l, c)
    adx_val = _adx(h, l, c, period=14)

    # Realized vol family (per-window)
    rv_close = float(log_rets.std(ddof=1)) if len(log_rets) > 1 else float("nan")
    # Parkinson
    if len(h) > 0 and len(l) > 0 and (h > 0).all() and (l > 0).all():
        hl = np.log(h / l)
        parkinson = float(np.sqrt((1.0 / (4.0 * np.log(2.0))) * np.mean(hl ** 2)))
    else:
        parkinson = float("nan")
    # Garman-Klass
    if "open" in rwin.columns and (rwin["open"] > 0).all() and (c > 0).all():
        o = rwin["open"].astype(float)
        log_hl = np.log(h / l)
        log_co = np.log(c / o)
        gk_var = 0.5 * np.mean(log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * np.mean(log_co ** 2)
        garman_klass = float(np.sqrt(max(gk_var, 0.0)))
    else:
        garman_klass = float("nan")
    # Yang-Zhang (optional)
    yang_zhang = float("nan")
    if "open" in rwin.columns and (rwin["open"] > 0).all() and (c > 0).all():
        o = rwin["open"].astype(float)
        k = 0.34 / (1.34 + (len(c) + 1) / (len(c) - 1)) if len(c) > 1 else 0.0
        log_oo = np.log(o / o.shift(1)).dropna()
        log_cc = np.log(c / c.shift(1)).dropna()
        log_oc = np.log(c / o)
        log_co = np.log(o / c.shift(1)).dropna()
        if len(log_oo) and len(log_cc) and len(log_co):
            s2o = np.var(log_oo, ddof=1) if len(log_oo) > 1 else 0.0
            s2c = np.var(log_cc, ddof=1) if len(log_cc) > 1 else 0.0
            rs = np.mean((np.log(h / l)) ** 2) - np.mean(log_co ** 2) - np.mean(log_oc ** 2)
            yz_var = s2o + k * s2c + (1 - k) * max(rs, 0.0)
            yang_zhang = float(np.sqrt(max(yz_var, 0.0)))

    # Bollinger position [0,1]
    bb_pos = float(_safe_div(last - bb["lower"], (bb["upper"] - bb["lower"]) + EPS)) if (math.isfinite(bb.get("upper", np.nan)) and math.isfinite(bb.get("lower", np.nan))) else float("nan")
    if math.isfinite(bb_pos):
        bb_pos = min(1.0, max(0.0, bb_pos))

    # Range intensity and distance to extremes
    range_intensity = float(_safe_div((float(h.max()) - float(l.min())), vwap + EPS)) if len(h) and len(l) and math.isfinite(vwap) else float("nan")
    dist_to_high = float(_safe_div(last - float(c.max()), atr_val + EPS)) if len(c) and math.isfinite(atr_val) and math.isfinite(last) else float("nan")
    dist_to_low = float(_safe_div(last - float(c.min()), atr_val + EPS)) if len(c) and math.isfinite(atr_val) and math.isfinite(last) else float("nan")
    dist_to_high_abs = abs(dist_to_high) if math.isfinite(dist_to_high) else float("nan")
    dist_to_low_abs = abs(dist_to_low) if math.isfinite(dist_to_low) else float("nan")

    # Volume & liquidity
    quote_volume_sum = float((c * v).sum()) if len(c) and len(v) else float("nan")
    # Log returns for streaks and shares
    pos_mask = (log_rets > 0).fillna(False) if len(c) else pd.Series(dtype=bool)
    up_volume = float(v[pos_mask].sum()) if len(v) else float("nan")
    up_volume_share = float(_safe_div(up_volume, v.sum())) if len(v) else float("nan")
    down_volume_share = float(1.0 - up_volume_share) if math.isfinite(up_volume_share) else float("nan")
    vol_med = float(v.median()) if len(v) else float("nan")
    mad_raw = _mad(v) if len(v) else float("nan")
    mad_scaled = float(1.4826 * mad_raw) if math.isfinite(mad_raw) else float("nan")
    if math.isfinite(vol_med) and math.isfinite(mad_scaled):
        spike_thr = vol_med + 3.0 * mad_scaled
        vol_spike_share = float(np.mean((v > spike_thr).astype(float))) if len(v) else float("nan")
    else:
        vol_spike_share = float("nan")

    # Squeeze index
    squeeze_index = float(_safe_div(bb.get("width", np.nan), (atr_val * 4.0) + EPS)) if math.isfinite(bb.get("width", np.nan)) and math.isfinite(atr_val) else float("nan")

    # Shares
    ret_share_pos = float(np.mean((log_rets > 0).astype(float))) if len(log_rets) else float("nan")
    # longest streaks of up/down returns
    def _longest_streak(mask: np.ndarray) -> int:
        best = 0
        cur = 0
        for m in mask:
            if m:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    run_up_max = float(_longest_streak((log_rets > 0).to_numpy())) if len(log_rets) else float("nan")
    run_dn_max = float(_longest_streak((log_rets < 0).to_numpy())) if len(log_rets) else float("nan")
    vwap_pos_share = float(np.mean((c > vwap).astype(float))) if math.isfinite(vwap) and len(c) else float("nan")

    out = {
        # Hygiene flags
        f"window_coverage_{suffix}": float(coverage),
        f"window_ok_{suffix}": bool(window_ok),
        # Legacy metrics (kept for compatibility)
        f"close_mean_{suffix}": close_mean,
        f"close_stddev_{suffix}": close_std,
        f"close_min_{suffix}": close_min,
        f"close_max_{suffix}": close_max,
        f"volume_sum_{suffix}": v_sum,
        f"vwap_{suffix}": vwap,
        f"price_deviation_vwap_{suffix}": dev_vwap,
        f"roc_{suffix}": roc,
        f"returns_log_{suffix}": log_ret_sum,
        f"sma_{suffix}": sma,
        f"ema_fast_{suffix}": ema_fast_val,
        f"ema_slow_{suffix}": ema_slow_val,
        f"macd_{suffix}": macd_val,
        f"macd_signal_{suffix}": macd_signal_val,
        f"macd_histogram_{suffix}": macd_hist,
        f"bollinger_upper_{suffix}": bb["upper"],
        f"bollinger_lower_{suffix}": bb["lower"],
        f"bollinger_width_{suffix}": bb["width"],
        f"rsi_{suffix}": rsi_val,
        f"atr_{suffix}": atr_val,
        f"stochastic_k_{suffix}": st["k"],
        f"stochastic_d_{suffix}": st["d"],
        f"obv_{suffix}": obv_val,
        f"cci_{suffix}": cci_val,
        f"adx_{suffix}": adx_val,
        # Opportunity primitives with taxonomy tags
        f"quote_volume_sum_{suffix}_mag": quote_volume_sum,
        f"up_volume_share_{suffix}_pos": up_volume_share,
        f"down_volume_share_{suffix}_pos": down_volume_share,
        f"vol_spike_share_{suffix}_pos": vol_spike_share,
        f"rv_close_{suffix}_mag": rv_close,
        f"parkinson_vol_{suffix}_mag": parkinson,
        f"garman_klass_{suffix}_mag": garman_klass,
        f"yang_zhang_{suffix}_mag": yang_zhang,
        f"squeeze_index_{suffix}_mag": squeeze_index,
        f"range_intensity_{suffix}_mag": range_intensity,
        f"vwap_deviation_abs_{suffix}_mag": abs(dev_vwap) if math.isfinite(dev_vwap) else float("nan"),
        f"bb_position_{suffix}_pos": bb_pos,
        f"dist_to_window_high_{suffix}_dir": dist_to_high,
        f"dist_to_window_low_{suffix}_dir": dist_to_low,
        f"dist_to_window_high_abs_{suffix}_mag": dist_to_high_abs,
        f"dist_to_window_low_abs_{suffix}_mag": dist_to_low_abs,
        f"ret_share_pos_{suffix}_pos": ret_share_pos,
        f"run_up_max_{suffix}_mag": run_up_max,
        f"run_dn_max_{suffix}_mag": run_dn_max,
        f"vwap_pos_share_{suffix}_pos": vwap_pos_share,
    }
    # Validity flags per family
    out.update({
        f"metrics_valid_volume_{suffix}": bool(np.isfinite([quote_volume_sum, up_volume_share, down_volume_share]).any()) if len(v) else False,
        f"metrics_valid_volatility_{suffix}": bool(np.isfinite([rv_close, parkinson, garman_klass, yang_zhang]).any()),
        f"metrics_valid_vwap_{suffix}": bool(np.isfinite([vwap, dev_vwap]).all()) if math.isfinite(vwap) else False,
        f"metrics_valid_bbands_{suffix}": bool(np.isfinite([bb.get('upper', np.nan), bb.get('lower', np.nan), bb.get('width', np.nan)]).all()),
        f"metrics_valid_trend_{suffix}": bool(np.isfinite([adx_val]).all()),
        f"metrics_valid_range_{suffix}": bool(np.isfinite([range_intensity, dist_to_high, dist_to_low]).any()),
        f"metrics_valid_momentum_{suffix}": bool(np.isfinite([roc, ret_share_pos, run_up_max, run_dn_max]).any()),
    })
    return out


def compute_metrics_series(
    df: pd.DataFrame,
    window_size: int,
    suffix: str,
    min_fraction: float = 0.8,
) -> pd.DataFrame:
    """Compute metrics for each non-overlapping window aligned to UTC.

    Returns a DataFrame with one row per window end (inclusive) containing
    the same columns as compute_window_metrics, plus 'window_start' and 'window_end'.
    Incomplete windows (insufficient bars) are skipped.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("timestamp")
    # Parse with UTC; inputs are expected UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.set_index("timestamp")

    # Build window buckets aligned to UTC boundaries
    # Convert minutes to pandas offset string (H for hours when divisible by 60)
    if window_size % 60 == 0:
        freq = f"{window_size // 60}h"
    else:
        freq = f"{window_size}min"

    # Map each row to its window start
    win_start = df.index.floor(freq)
    df = df.assign(_win_start=win_start)
    rows: List[Dict[str, float]] = []
    for wstart, g in df.groupby("_win_start"):
        # expect ~ window_size rows; enforce coverage for baselines
        if len(g) < max(1, int(window_size * min_fraction)):
            continue
        metrics = compute_window_metrics(
            g, window_size=window_size, suffix=suffix, min_fraction=min_fraction, prealigned=True
        )
        if not metrics:
            continue
        wend = wstart + pd.to_timedelta(window_size, unit="m")
        row = {"window_start": wstart.to_pydatetime(), "window_end": wend.to_pydatetime()}
        row.update(metrics)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out


# removed unused percentile_rank helper


def quantiles_summary(values: np.ndarray, quantiles: Iterable[float] = (1, 5, 10, 20, 25, 40, 50, 60, 75, 80, 90, 95, 99)) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{int(q):02d}": float("nan") for q in quantiles}
    qs = np.nanpercentile(arr, list(quantiles))
    return {f"p{int(q):02d}": float(v) for q, v in zip(quantiles, qs)}


def build_symbol_baseline(
    symbol_df: pd.DataFrame,
    windows_minutes: Iterable[int] = WINDOWS_MINUTES,
    min_fraction: float = 0.8,
) -> Dict[str, Dict[str, float]]:
    """Build baseline percentiles and sample sizes for a single symbol.

    Returns mapping metric_name -> {p01,p05,p10,p20,p25,p40,p50,p60,p75,p80,p90,p95,p99,n}
    for each window suffix.
    """
    out: Dict[str, Dict[str, float]] = {}
    if symbol_df.empty:
        return out
    for w in windows_minutes:
        sfx = SUFFIX.get(w, f"{w}m")
        series_df = compute_metrics_series(symbol_df.copy(), w, sfx, min_fraction=min_fraction)
        if series_df.empty:
            continue
        # Drop non-metric columns, hygiene flags and deny-list non-signals
        metric_cols = []
        for c in series_df.columns:
            if c in ("window_start", "window_end"):
                continue
            if c.startswith("window_") or c.startswith("metrics_valid_"):
                continue
            if c.startswith("sma_") or c.startswith("ema_fast_") or c.startswith("ema_slow_"):
                continue
            if c.startswith("macd_") and not c.startswith("macd_histogram_"):
                continue
            # Only numeric dtypes
            if not pd.api.types.is_numeric_dtype(series_df[c]):
                continue
            metric_cols.append(c)
        for col in metric_cols:
            vals = series_df[col].to_numpy(dtype=float)
            stats = quantiles_summary(vals)
            stats["n"] = float(np.isfinite(vals).sum())
            out[col] = stats
    return out


def enrich_current_with_baseline(
    current_df: pd.DataFrame,
    baselines: Dict[str, Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    """Append percentile/quintile and key baseline quantiles to current metrics.

    Adds for each metric column M:
      - M_pctile (0-100)
      - M_quintile (1..5)
      - M_p25, M_p50, M_p75
      - M_n (number of baseline samples)
      - M_direction ('under' if value < p50 else 'over')
    """
    if current_df.empty:
        return current_df

    # Define metric transform categories
    POSITIVE_PREFIXES = (
        "atr_", "bollinger_width_", "volume_sum_", "quote_volume_sum_", "range_intensity_",
        "rv_close_", "parkinson_vol_", "garman_klass_", "yang_zhang_", "vwap_deviation_abs_",
        "dist_to_window_high_abs_", "dist_to_window_low_abs_",
    )
    SIGNED_PREFIXES = (
        "macd_histogram_", "roc_", "price_deviation_vwap_", "dist_to_window_high_", "dist_to_window_low_", "obv_",
    )

    def _transform_metric_value(col: str, x: float) -> float:
        # Determine transform for interpolation
        try:
            base = col.rsplit("_", 1)[0]  # remove suffix tag like _1h or taxonomy tag
        except Exception:
            base = col
        if any(base.startswith(p) for p in POSITIVE_PREFIXES):
            return float(np.log1p(max(x, 0.0))) if math.isfinite(x) else float("nan")
        if any(base.startswith(p) for p in SIGNED_PREFIXES):
            return float(np.sign(x) * np.log1p(abs(x))) if math.isfinite(x) else float("nan")
        # default: identity
        return x

    rows: List[Dict[str, float]] = []
    for _, r in current_df.iterrows():
        sym = r["symbol"]
        b = baselines.get(sym, {})
        enriched = r.to_dict()
        for col, val in r.items():
            if col == "symbol":
                continue
            # Skip non-signal columns and hygiene flags for enrichment
            cname = str(col)
            if cname.startswith("timestamp_asof") or cname.startswith("window_") or cname.startswith("metrics_valid_") or \
               cname.startswith("sma_") or cname.startswith("ema_fast_") or cname.startswith("ema_slow_") or \
               (cname.startswith("macd_") and not cname.startswith("macd_histogram_")):
                continue
            base = b.get(col)
            if base is None:
                continue
            p25 = base.get("p25")
            p50 = base.get("p50")
            p75 = base.get("p75")
            n = base.get("n", 0.0)

            # Percentile via piecewise-linear interpolation between stored knots
            # using transformed space for heavy-tailed metrics.
            q_list = (1, 5, 10, 25, 50, 75, 90, 95, 99)
            knots = []
            knot_pct = []
            for q in q_list:
                pv = base.get(f"p{q:02d}")
                if pv is not None and math.isfinite(pv):
                    knots.append(_transform_metric_value(col, float(pv)))
                    knot_pct.append(float(q))
            pctile = float("nan")
            if len(knots) >= 2 and math.isfinite(val):
                xs = np.array(knots, dtype=float)
                ys = np.array(knot_pct, dtype=float)
                order = np.argsort(xs)
                xs = xs[order]
                ys = ys[order]
                tval = _transform_metric_value(col, float(val))
                if np.ptp(xs) == 0 or not np.isfinite(tval):
                    pctile = 50.0
                else:
                    pctile = float(np.interp(tval, xs, ys, left=1.0, right=99.0))

            # Quintile by comparing to p20,p40,p60,p80 (approx via p25/p50/p75)
            q_breaks = [
                base.get("p20", base.get("p25")),
                base.get("p40", base.get("p50")),
                base.get("p60", base.get("p75")),
                base.get("p80", base.get("p90")),
            ]
            quintile = 1
            if math.isfinite(val):
                quintile = 1
                for i, brk in enumerate(q_breaks, start=1):
                    if brk is None or not math.isfinite(brk):
                        continue
                    if val > brk:
                        quintile = i + 1
            # Clamp 1..5
            quintile = max(1, min(5, quintile))

            direction = None
            if p50 is not None and math.isfinite(p50) and math.isfinite(val):
                direction = "over" if val >= p50 else "under"

            enriched[f"{col}_pctile"] = pctile
            enriched[f"{col}_quintile"] = quintile
            if p25 is not None:
                enriched[f"{col}_p25"] = p25
            if p50 is not None:
                enriched[f"{col}_p50"] = p50
            if p75 is not None:
                enriched[f"{col}_p75"] = p75
            enriched[f"{col}_n"] = n
            if direction is not None:
                enriched[f"{col}_direction"] = direction

            # Robust z and abs robust z
            if all(p is not None and math.isfinite(p) for p in [p25, p50, p75]) and math.isfinite(val):
                scale = (p75 - p25) if (p75 - p25) != 0 else 0.0
                rz = float((val - p50) / (scale + EPS))
                enriched[f"{col}_rz"] = rz
                enriched[f"{col}_abs_rz"] = abs(rz)

        rows.append(enriched)
    return pd.DataFrame(rows)



def compute_symbol_metrics(symbol_df: pd.DataFrame, windows_minutes: Iterable[int] = WINDOWS_MINUTES) -> Dict[str, float]:
    """Compute metrics for one symbol across requested windows."""
    df = symbol_df.copy()
    if df.empty:
        return {}
    df = df.sort_values("timestamp")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # Ensure expected columns exist
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in input data")

    out: Dict[str, float] = {}
    per_window: Dict[int, Dict[str, float]] = {}
    for w in windows_minutes:
        sfx = SUFFIX.get(w, f"{w}m")
        metrics = compute_window_metrics(df, w, sfx)
        per_window[w] = metrics
        out.update(metrics)

    # Anchor snapshot with as-of timestamp (UTC max timestamp in input)
    try:
        if "timestamp" in df.columns:
            tmax = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").max()
            # Compute last complete bucket for each window and take the minimum
            anchors = []
            for w in windows_minutes:
                rule = f"{w // 60}h" if w % 60 == 0 else f"{w}min"
                anchors.append(tmax.floor(rule))
            anchor = min(anchors) if anchors else tmax
            out["timestamp_asof_utc"] = anchor.to_pydatetime()
            # Include current (as-of) price for LLM/forecast downstreams
            try:
                last_close = float(df.loc[df["timestamp"] <= anchor, "close"].astype(float).iloc[-1])
            except Exception:
                last_close = float("nan")
            out["close_current"] = last_close
    except Exception:
        pass

    # Cross-horizon rollups (using available windows)
    def getm(base: str, w: int) -> float:
        sfx = SUFFIX.get(w, f"{w}m")
        return out.get(f"{base}_{sfx}")

    # Sign agreement for selected metrics across intraday windows
    intraday_ws = [w for w in [60, 240, 720, 1440] if w in windows_minutes]
    def sign_agree(bases: List[str]) -> None:
        for base in bases:
            signs = []
            for w in intraday_ws:
                sfx = SUFFIX[w]
                val = out.get(f"{base}_{sfx}")
                if val is None or not isinstance(val, (int, float)) or not math.isfinite(val):
                    continue
                if abs(val) < EPS:
                    continue
                signs.append(1 if val > 0 else -1)
            agree_key = f"{base}_sign_agree_rollup"
            if signs:
                out[agree_key] = int(sum(signs))
                out[f"{agree_key}_valid"] = True
            else:
                out[f"{agree_key}_valid"] = False

    sign_agree(["macd_histogram", "roc", "price_deviation_vwap"])

    # Slope-ish ratios/diffs
    def safe_ratio(a: Optional[float], b: Optional[float]) -> float:
        if a is None or b is None or not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            return float("nan")
        if not (math.isfinite(a) and math.isfinite(b)):
            return float("nan")
        return float(a / (abs(b) + EPS))

    def safe_diff(a: Optional[float], b: Optional[float]) -> float:
        if a is None or b is None or not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            return float("nan")
        if not (math.isfinite(a) and math.isfinite(b)):
            return float("nan")
        return float(a - b)

    if all(w in out for w in []):
        pass
    # rv ratios
    if all(SUFFIX.get(w) for w in [60, 240]):
        rv1 = out.get(f"rv_close_{SUFFIX[60]}_mag")
        rv4 = out.get(f"rv_close_{SUFFIX[240]}_mag")
        r = safe_ratio(rv1, rv4)
        out["rv_close_1h_over_4h_mag"] = r
        out["rv_close_1h_over_4h_mag_valid"] = bool(isinstance(r, (int, float)) and math.isfinite(r))
    if all(SUFFIX.get(w) for w in [240, 1440]):
        rv4 = out.get(f"rv_close_{SUFFIX[240]}_mag")
        rv24 = out.get(f"rv_close_{SUFFIX[1440]}_mag")
        r = safe_ratio(rv4, rv24)
        out["rv_close_4h_over_24h_mag"] = r
        out["rv_close_4h_over_24h_mag_valid"] = bool(isinstance(r, (int, float)) and math.isfinite(r))

    # adx diffs
    if all(SUFFIX.get(w) for w in [60, 240]):
        a1 = out.get(f"adx_{SUFFIX[60]}")
        a4 = out.get(f"adx_{SUFFIX[240]}")
        d = safe_diff(a1, a4)
        out["adx_1h_minus_4h_mag"] = d
        out["adx_1h_minus_4h_mag_valid"] = bool(isinstance(d, (int, float)) and math.isfinite(d))
    if all(SUFFIX.get(w) for w in [240, 1440]):
        a4 = out.get(f"adx_{SUFFIX[240]}")
        a24 = out.get(f"adx_{SUFFIX[1440]}")
        d = safe_diff(a4, a24)
        out["adx_4h_minus_24h_mag"] = d
        out["adx_4h_minus_24h_mag_valid"] = bool(isinstance(d, (int, float)) and math.isfinite(d))

    # momentum ratio
    if all(SUFFIX.get(w) for w in [240, 1440]):
        r4 = out.get(f"roc_{SUFFIX[240]}")
        r24 = out.get(f"roc_{SUFFIX[1440]}")
        r = safe_ratio(r4, r24)
        out["roc_4h_over_24h_dir"] = r
        out["roc_4h_over_24h_dir_valid"] = bool(isinstance(r, (int, float)) and math.isfinite(r))
    return out


def compute_cohort_metrics(
    input_csv: str,
    output_csv: str,
    windows_minutes: Iterable[int] = WINDOWS_MINUTES,
    impute_neutral: bool = False,
) -> pd.DataFrame:
    """Read a cohort CSV (symbol,timestamp,open,high,low,close,volume) and
    write per-symbol metrics to output_csv. Returns the DataFrame written.
    """
    # Resolve input path robustly (case-insensitive + fuzzy match)
    input_csv = resolve_input_path(input_csv)
    df = pd.read_csv(input_csv)
    # Normalize column names (case-insensitive + common synonyms)
    df = normalize_ohlcv_columns(df)
    if df.empty:
        result = pd.DataFrame(columns=["symbol"])  # empty template
        result.to_csv(output_csv, index=False)
        return result

    # Parse timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    rows: List[Dict[str, float]] = []
    for sym, g in df.groupby("symbol"):
        metrics = compute_symbol_metrics(g, windows_minutes=windows_minutes)
        metrics_row: Dict[str, float] = {"symbol": sym}
        metrics_row.update(metrics)
        rows.append(metrics_row)

    result = pd.DataFrame(rows)
    if impute_neutral:
        result = impute_neutral_metrics(result)
    result.to_csv(output_csv, index=False)
    return result


def compute_cohort_metrics_series(
    input_csv: str,
    output_csv: str,
    windows_minutes: Iterable[int] = WINDOWS_MINUTES,
    min_fraction: float = 0.8,
    impute_neutral: bool = False,
) -> pd.DataFrame:
    """Compute a time series of metrics per symbol and per non-overlapping window.

    Produces a long-format DataFrame with rows at each window end (UTC-aligned),
    safe to join to OHLCVT by joining on (symbol, timestamp=window_end).

    Columns include: symbol, window (suffix), window_start, window_end, and the
    same metrics as compute_window_metrics for that window.
    """
    df = pd.read_csv(input_csv)
    if df.empty:
        result = pd.DataFrame(columns=["symbol", "window", "window_start", "window_end"])  # empty template
        result.to_csv(output_csv, index=False)
        return result

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    all_rows: List[pd.DataFrame] = []
    for sym, g in df.groupby("symbol"):
        for w in windows_minutes:
            sfx = SUFFIX.get(w, f"{w}m")
            series_df = compute_metrics_series(g.copy(), w, sfx, min_fraction=min_fraction)
            if series_df.empty:
                continue
            out = series_df.copy()
            out.insert(0, "window", sfx)
            out.insert(0, "symbol", sym)
            # Add a join-friendly timestamp equal to window_end
            if "window_end" in out.columns and "timestamp" not in out.columns:
                out["timestamp"] = out["window_end"]
            all_rows.append(out)

    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["symbol", "window", "window_start", "window_end", "timestamp"])
    if impute_neutral:
        result = impute_neutral_metrics(result)
    result.to_csv(output_csv, index=False)
    return result


def _parse_windows_arg(arg: Optional[str], default_minutes: Iterable[int]) -> List[int]:
    if arg is None or str(arg).strip() == "":
        return list(default_minutes)
    parts = [p.strip().lower() for p in str(arg).split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        if p.endswith("h"):
            out.append(int(float(p[:-1]) * 60))
        elif p.endswith("min"):
            out.append(int(p[:-3]))
        elif p.endswith("m"):
            out.append(int(p[:-1]))
        else:
            out.append(int(p))
    return out


def compute_cohort_metrics_dense(
    input_csv: str,
    output_csv: str,
    windows_minutes: Iterable[int] = WINDOWS_MINUTES,
    min_fraction: float = 0.8,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    impute_neutral: bool = False,
) -> pd.DataFrame:
    """Compute per-minute rolling metrics (dense mode) for selected windows.

    - Emits one row per (symbol, timestamp) present in the input minute bars.
    - For each timestamp, computes metrics for each requested window using the
      same feature family as compute_window_metrics with prealigned=True.
    - Joins all window-suffixed metrics into a single wide row.
    - Intended for ML pipelines where features must align to every minute bar.
    """
    df = pd.read_csv(input_csv)
    if df.empty:
        result = pd.DataFrame(columns=["symbol", "timestamp"])  # empty template
        result.to_csv(output_csv, index=False)
        return result

    # Parse timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if start_ts:
        start = pd.to_datetime(start_ts, utc=True, errors="coerce")
        if pd.notna(start):
            df = df[df["timestamp"] >= start]
    if end_ts:
        end = pd.to_datetime(end_ts, utc=True, errors="coerce")
        if pd.notna(end):
            df = df[df["timestamp"] <= end]

    rows: List[Dict[str, float]] = []
    for sym, g in df.groupby("symbol"):
        g = g.dropna(subset=["timestamp"]).sort_values("timestamp")
        if g.empty:
            continue
        ts_list = g["timestamp"].unique().tolist()
        for ts in ts_list:
            row: Dict[str, float] = {"symbol": sym, "timestamp": ts}
            # For each window, slice trailing data and compute metrics
            for w in windows_minutes:
                sfx = SUFFIX.get(w, f"{w}m")
                start = ts - pd.to_timedelta(w, unit="m")
                win = g[(g["timestamp"] > start) & (g["timestamp"] <= ts)]
                if win.empty:
                    # Fill with empty window metrics schema for consistency
                    m = compute_window_metrics(pd.DataFrame(columns=["timestamp","open","high","low","close","volume"]), w, sfx, min_fraction=min_fraction, prealigned=True)
                else:
                    m = compute_window_metrics(win.copy(), w, sfx, min_fraction=min_fraction, prealigned=True)
                # Merge
                row.update(m)
            rows.append(row)

    result = pd.DataFrame(rows)
    if impute_neutral:
        result = impute_neutral_metrics(result)
    result.to_csv(output_csv, index=False)
    return result


def impute_neutral_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Impute neutral values for numeric metrics to eliminate NaNs for ML.

    Rules (mirrors state/regime imputation):
      - *_pctile = 50.0
      - *_quintile = 3
      - *_rz, *_abs_rz = 0.0
      - *_pos = 0.5
      - *_dir = 0.0
      - *_over_* = 1.0
      - *_minus_* = 0.0
      - dist_*_mag and *_mag = 0.0 (or p50 if provided; fallback 0.0)
      - All other numeric metrics: 0.0
    Also emits {col}_imputed flags (0/1) per column imputed.
    """
    out = df.copy()
    import numpy as _np
    import pandas as _pd

    def fill(col: str, neutral) -> None:
        if col not in out.columns:
            return
        s = out[col]
        if not _pd.api.types.is_numeric_dtype(s):
            return
        vals = s.to_numpy(dtype=float)
        mask = ~_np.isfinite(vals)
        flag_col = f"{col}_imputed"
        if mask.any():
            out[col] = s.astype(float)
            out.loc[mask, col] = neutral
            out[flag_col] = 0
            out.loc[mask, flag_col] = 1
        else:
            out[flag_col] = 0

    # Iterate all numeric columns and apply family-specific rules
    for col in list(out.columns):
        if col in ("symbol",):
            continue
        if any(col.startswith(p) for p in ("timestamp_asof", "window_coverage_", "window_ok_", "metrics_valid_", "window_start", "window_end", "timestamp")):
            # Skip flags/timestamps
            continue
        # Percentile/quintile/z families
        if col.endswith("_pctile"):
            fill(col, 50.0); continue
        if col.endswith("_quintile"):
            fill(col, 3); continue
        if col.endswith("_rz") or col.endswith("_abs_rz"):
            fill(col, 0.0); continue
        # Positions/dirs
        if col.endswith("_pos"):
            fill(col, 0.5); continue
        if col.endswith("_dir"):
            fill(col, 0.0); continue
        # Ratios/differences
        if "_over_" in col:
            fill(col, 1.0); continue
        if "_minus_" in col:
            fill(col, 0.0); continue
        # Magnitudes & distances
        if col.startswith("dist_") or col.endswith("_mag"):
            # Try p50 sidecar if present
            p50_col = f"{col}_p50"
            neutral = 0.0
            if p50_col in out.columns:
                try:
                    neutral = float(out[p50_col].fillna(0.0).iloc[0])
                except Exception:
                    neutral = 0.0
            fill(col, neutral); continue
        # All other numeric metrics -> 0.0
        fill(col, 0.0)
    return out


def resolve_input_path(path: str, search_dir: Optional[str] = None, exts: Tuple[str, ...] = (".csv", ".parquet")) -> str:
    """Resolve a possibly misspelled or case-variant file path.

    - Returns the existing path if it exists.
    - Else, searches `search_dir` (default cwd) for case-insensitive and fuzzy filename matches
      among files with extensions in `exts`.
    - Raises FileNotFoundError with suggestions when not found.
    """
    if not path:
        raise FileNotFoundError("Input path is empty")
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    root = search_dir or os.getcwd()
    candidates: List[str] = []
    for fname in os.listdir(root):
        fp = os.path.join(root, fname)
        if not os.path.isfile(fp):
            continue
        fstem, fext = os.path.splitext(fname)
        if fext.lower() not in exts:
            continue
        candidates.append(fname)
        # exact case-insensitive match
        if fname.lower() == base.lower():
            return fp
    # Fuzzy match by stem
    close = difflib.get_close_matches(stem.lower(), [os.path.splitext(c)[0].lower() for c in candidates], n=3, cutoff=0.6)
    if close:
        # Map back to original case filenames
        mapping = {os.path.splitext(c)[0].lower(): c for c in candidates}
        suggestion = mapping.get(close[0])
        if suggestion:
            return os.path.join(root, suggestion)
    # No match found
    msg = f"Input file not found: {path}"
    if candidates:
        msg += f". Nearby candidates: {', '.join(sorted(candidates)[:5])}"
    raise FileNotFoundError(msg)


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to expected OHLCV schema with case-insensitive mapping and synonyms.

    Expected: symbol, timestamp, open, high, low, close, volume (trades optional)
    Synonyms handled:
      - symbol: [symbol, sym, ticker, asset]
      - timestamp: [timestamp, time, date, datetime]
      - volume: [volume, vol, qty, quantity, amount]
    Non-matching columns remain unchanged.
    """
    if df is None or df.empty:
        return df
    original = list(df.columns)
    lower_map: Dict[str, str] = {str(c).lower().strip(): c for c in original}

    def have(*names: str) -> Optional[str]:
        for n in names:
            if n in lower_map:
                return lower_map[n]
        return None

    rename: Dict[str, str] = {}
    # symbol
    sym_col = have("symbol", "sym", "ticker", "asset")
    if sym_col and sym_col != "symbol":
        rename[sym_col] = "symbol"
    # timestamp
    ts_col = have("timestamp", "time", "date", "datetime")
    if ts_col and ts_col != "timestamp":
        rename[ts_col] = "timestamp"
    # open/high/low/close
    for k in ["open", "high", "low", "close"]:
        c = have(k)
        if c and c != k:
            rename[c] = k
    # volume
    vol_col = have("volume", "vol", "qty", "quantity", "amount")
    if vol_col and vol_col != "volume":
        rename[vol_col] = "volume"

    if rename:
        df = df.rename(columns=rename)
    return df
