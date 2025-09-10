import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# Window sizes expressed in minutes (since input is 1-minute bars)
WINDOWS_MINUTES = [360, 1440, 4320, 10080]  # 6h, 24h, 72h, 168h
SUFFIX = {
    360: "6h",
    1440: "24h",
    4320: "72h",
    10080: "168h",
}


def _safe_div(a: float, b: float) -> float:
    if b == 0 or pd.isna(b):
        return 0.0
    return a / b


def _ema(series: pd.Series, span: int) -> float:
    if series.empty:
        return 0.0
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(window=period, min_periods=period).mean()
    roll_down = down.rolling(window=period, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    val = rsi.iloc[-1]
    if pd.isna(val):
        return 50.0
    return float(val)


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> Dict[str, float]:
    if close.empty:
        return {"upper": 0.0, "lower": 0.0, "width": 0.0}
    ma = close.rolling(window=period, min_periods=period).mean()
    sd = close.rolling(window=period, min_periods=period).std(ddof=0)
    mu = ma.iloc[-1] if not pd.isna(ma.iloc[-1]) else close.mean()
    sigma = sd.iloc[-1] if not pd.isna(sd.iloc[-1]) else close.std(ddof=0)
    upper = mu + num_std * sigma
    lower = mu - num_std * sigma
    width = upper - lower
    return {"upper": float(upper), "lower": float(lower), "width": float(width)}


def _atr(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    if len(close) < 2:
        return 0.0
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr = tr.dropna()
    if tr.empty:
        return 0.0
    return float(tr.mean())


def _stochastic_kd(close: pd.Series) -> Dict[str, float]:
    if close.empty:
        return {"k": 0.0, "d": 0.0}
    mn = float(close.min())
    mx = float(close.max())
    last = float(close.iloc[-1])
    if mx == mn:
        k = 0.0
    else:
        k = 100.0 * (last - mn) / (mx - mn)
    return {"k": float(k), "d": float(k)}  # basic %D as same-window mean


def _obv(close: pd.Series, volume: pd.Series) -> float:
    if close.empty or volume.empty:
        return 0.0
    sign = np.sign(close.diff().fillna(0.0))
    obv = (sign * volume).cumsum()
    return float(obv.iloc[-1])


def _cci(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    if high.empty:
        return 0.0
    tp = (high + low + close) / 3.0
    sma = tp.mean()
    mad = np.mean(np.abs(tp - sma))
    if mad == 0:
        return 0.0
    return float((tp.iloc[-1] - sma) / (0.015 * mad))


def _adx(high: pd.Series, low: pd.Series, atr_val: float) -> float:
    if len(high) < 2 or atr_val <= 0:
        return 0.0
    dm_p = high.diff().clip(lower=0.0).dropna()
    dm_m = (-low.diff()).clip(lower=0.0).dropna()
    if dm_p.empty or dm_m.empty:
        return 0.0
    ap = float(dm_p.mean())
    am = float(dm_m.mean())
    if (ap + am) == 0:
        return 0.0
    pdi = 100.0 * ap / atr_val
    mdi = 100.0 * am / atr_val
    return float(100.0 * abs(pdi - mdi) / (pdi + mdi))


def compute_window_metrics(
    df: pd.DataFrame,
    window_size: int,
    suffix: str,
    min_fraction: float = 0.95,
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

    win = df.tail(window_size)
    # Guard: require sufficient bars to avoid skewing early-window results
    if len(win) < max(1, int(window_size * min_fraction)):
        return {}
    c = win["close"].astype(float)
    h = win["high"].astype(float)
    l = win["low"].astype(float)
    v = win["volume"].astype(float)

    last = float(c.iloc[-1])
    first = float(c.iloc[0])

    close_mean = float(c.mean())
    close_std = float(c.std(ddof=0)) if len(c) > 1 else 0.0
    close_min = float(c.min())
    close_max = float(c.max())

    v_sum = float(v.sum())
    vwap = float(_safe_div((c * v).sum(), v_sum)) if v_sum else float(c.mean())
    dev_vwap = float(_safe_div(last - vwap, vwap))

    roc = float(_safe_div(last - first, first) * 100.0) if first else 0.0
    log_ret = float(np.log(c).diff().sum()) if (c > 0).all() else 0.0

    sma = close_mean
    ema_fast = _ema(c, span=12)
    ema_slow = _ema(c, span=26)
    macd = ema_fast - ema_slow
    macd_series = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_signal = float(macd_series.ewm(span=9, adjust=False).mean().iloc[-1])
    macd_hist = float(macd - macd_signal)

    bb = _bollinger(c, period=20, num_std=2.0)
    rsi = _rsi(c, period=14)
    atr = _atr(h, l, c)
    st = _stochastic_kd(c)
    obv = _obv(c, v)
    cci = _cci(h, l, c)
    adx = _adx(h, l, atr)

    out = {
        f"close_mean_{suffix}": close_mean,
        f"close_stddev_{suffix}": close_std,
        f"close_min_{suffix}": close_min,
        f"close_max_{suffix}": close_max,
        f"volume_sum_{suffix}": v_sum,
        f"vwap_{suffix}": vwap,
        f"price_deviation_vwap_{suffix}": dev_vwap,
        f"roc_{suffix}": roc,
        f"returns_log_{suffix}": log_ret,
        f"sma_{suffix}": sma,
        f"ema_fast_{suffix}": ema_fast,
        f"ema_slow_{suffix}": ema_slow,
        f"macd_{suffix}": macd,
        f"macd_signal_{suffix}": macd_signal,
        f"macd_histogram_{suffix}": macd_hist,
        f"bollinger_upper_{suffix}": bb["upper"],
        f"bollinger_lower_{suffix}": bb["lower"],
        f"bollinger_width_{suffix}": bb["width"],
        f"rsi_{suffix}": rsi,
        f"atr_{suffix}": atr,
        f"stochastic_k_{suffix}": st["k"],
        f"stochastic_d_{suffix}": st["d"],
        f"obv_{suffix}": obv,
        f"cci_{suffix}": cci,
        f"adx_{suffix}": adx,
    }

    # sanitize NaN/Inf
    clean = {}
    for k, v in out.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = 0.0
        else:
            clean[k] = float(v)
    return clean


def compute_metrics_series(
    df: pd.DataFrame,
    window_size: int,
    suffix: str,
    min_fraction: float = 0.95,
) -> pd.DataFrame:
    """Compute metrics for each non-overlapping window aligned to UTC.

    Returns a DataFrame with one row per window end (inclusive) containing
    the same columns as compute_window_metrics, plus 'window_start' and 'window_end'.
    Incomplete windows (insufficient bars) are skipped.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    # Build window buckets aligned to UTC boundaries
    # Convert minutes to pandas offset string (H for hours when divisible by 60)
    if window_size % 60 == 0:
        freq = f"{window_size // 60}H"
    else:
        freq = f"{window_size}T"

    # Map each row to its window start
    win_start = df.index.floor(freq)
    df = df.assign(_win_start=win_start)
    rows: List[Dict[str, float]] = []
    for wstart, g in df.groupby("_win_start"):
        # expect ~ window_size rows; enforce coverage
        if len(g) < max(1, int(window_size * min_fraction)):
            continue
        metrics = compute_window_metrics(
            g.reset_index(), window_size=window_size, suffix=suffix, min_fraction=min_fraction
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


def percentile_rank(value: float, population: np.ndarray) -> float:
    """Return the percentile rank (0-100) of value within the population."""
    arr = np.asarray(population, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    arr.sort()
    # number of elements <= value
    rank = np.searchsorted(arr, value, side="right")
    return 100.0 * rank / arr.size


def quantiles_summary(values: np.ndarray, quantiles: Iterable[float] = (5, 10, 25, 50, 75, 90, 95)) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{int(q):02d}": float("nan") for q in quantiles}
    qs = np.nanpercentile(arr, list(quantiles))
    return {f"p{int(q):02d}": float(v) for q, v in zip(quantiles, qs)}


def build_symbol_baseline(
    symbol_df: pd.DataFrame,
    windows_minutes: Iterable[int] = WINDOWS_MINUTES,
    min_fraction: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Build baseline percentiles and sample sizes for a single symbol.

    Returns mapping metric_name -> {p05,p10,p25,p50,p75,p90,p95,n}
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
        # Drop non-metric columns
        metric_cols = [c for c in series_df.columns if c not in ("window_start", "window_end")]
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

    rows: List[Dict[str, float]] = []
    for _, r in current_df.iterrows():
        sym = r["symbol"]
        b = baselines.get(sym, {})
        enriched = r.to_dict()
        for col, val in r.items():
            if col == "symbol":
                continue
            base = b.get(col)
            if base is None:
                continue
            p25 = base.get("p25")
            p50 = base.get("p50")
            p75 = base.get("p75")
            n = base.get("n", 0.0)

            # Percentile rank requires the full distribution; approximate using
            # piecewise linear interpolation between stored percentiles when full
            # distribution is not available. Here we use p10,p25,p50,p75,p90 as knots.
            knots = []
            knot_pct = []
            for q in (10, 25, 50, 75, 90):
                pv = base.get(f"p{q:02d}")
                if pv is not None and math.isfinite(pv):
                    knots.append(pv)
                    knot_pct.append(float(q))
            pctile = float("nan")
            if len(knots) >= 2 and math.isfinite(val):
                xs = np.array(knots, dtype=float)
                ys = np.array(knot_pct, dtype=float)
                # Ensure xs is monotonically non-decreasing
                order = np.argsort(xs)
                xs = xs[order]
                ys = ys[order]
                pctile = float(np.interp(val, xs, ys, left=0.0, right=100.0))

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

        rows.append(enriched)
    return pd.DataFrame(rows)



def compute_symbol_metrics(symbol_df: pd.DataFrame, windows_minutes: Iterable[int] = WINDOWS_MINUTES) -> Dict[str, float]:
    """Compute metrics for one symbol across requested windows."""
    df = symbol_df.copy()
    if df.empty:
        return {}
    df = df.sort_values("timestamp")
    # Ensure expected columns exist
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in input data")

    out: Dict[str, float] = {}
    for w in windows_minutes:
        sfx = SUFFIX.get(w, f"{w}m")
        out.update(compute_window_metrics(df, w, sfx))
    return out


def compute_cohort_metrics(input_csv: str, output_csv: str, windows_minutes: Iterable[int] = WINDOWS_MINUTES) -> pd.DataFrame:
    """Read a cohort CSV (symbol,timestamp,open,high,low,close,volume) and
    write per-symbol metrics to output_csv. Returns the DataFrame written.
    """
    df = pd.read_csv(input_csv)
    if df.empty:
        result = pd.DataFrame(columns=["symbol"])  # empty template
        result.to_csv(output_csv, index=False)
        return result

    # Parse timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")

    rows: List[Dict[str, float]] = []
    for sym, g in df.groupby("symbol"):
        metrics = compute_symbol_metrics(g, windows_minutes=windows_minutes)
        metrics_row: Dict[str, float] = {"symbol": sym}
        metrics_row.update(metrics)
        rows.append(metrics_row)

    result = pd.DataFrame(rows)
    result.to_csv(output_csv, index=False)
    return result
