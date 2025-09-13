import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# Reuse helpers from core
from cohort_metrics.core import resolve_input_path, normalize_ohlcv_columns


TF_DEFAULTS = ["1h", "4h", "12h", "24h"]


def load_metrics(path: str) -> pd.DataFrame:
    p = resolve_input_path(path)
    df = pd.read_csv(p)
    df = normalize_ohlcv_columns(df)
    # Parse timestamp_asof_utc if present
    if "timestamp_asof_utc" in df.columns:
        try:
            df["timestamp_asof_utc"] = pd.to_datetime(df["timestamp_asof_utc"], utc=True, errors="coerce")
        except Exception:
            pass
    return df


def load_imf_summary(path_json: Optional[str], path_txt: Optional[str]) -> Dict[str, Dict[str, float]]:
    """Return mapping symbol -> {median_amplitude_pct, median_duration_min}.

    Prefers JSON; falls back to TXT if provided.
    """
    out: Dict[str, Dict[str, float]] = {}
    if path_json and os.path.exists(path_json):
        try:
            with open(path_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            # JSON structure from imf_cluster_overlaps.py: summary[coin][window_label]['gmm']/['dbscan']
            # We'll derive per-coin dominant cluster across windows using median over available clusters.
            for coin, windows in data.items():
                amps: List[float] = []
                durs: List[float] = []
                for w, res in windows.items():
                    clusters = res.get("gmm", []) or res.get("dbscan", [])
                    for cl in clusters:
                        amps.append(float(cl.get("median_amplitude_pct", np.nan)))
                        durs.append(float(cl.get("median_duration_min", np.nan)))
                a = float(np.nanmedian(amps)) if amps else float("nan")
                d = float(np.nanmedian(durs)) if durs else float("nan")
                out[coin] = {"median_amplitude_pct": a, "median_duration_min": d}
        except Exception:
            pass
    if not out and path_txt and os.path.exists(path_txt):
        try:
            with open(path_txt, "r", encoding="utf-8") as f:
                for line in f:
                    parts = [p.strip() for p in line.strip().split(",")]
                    if len(parts) >= 3:
                        sym = parts[0]
                        try:
                            dur = float(parts[1])
                        except Exception:
                            dur = float("nan")
                        try:
                            amp = float(parts[2])
                        except Exception:
                            amp = float("nan")
                        out[sym] = {"median_amplitude_pct": amp, "median_duration_min": dur}
        except Exception:
            pass
    return out


def load_regime(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = resolve_input_path(path)
    if p.lower().endswith(".parquet"):
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    try:
        df = pd.read_csv(p)
        if "timestamp_asof_utc" in df.columns:
            df["timestamp_asof_utc"] = pd.to_datetime(df["timestamp_asof_utc"], utc=True, errors="coerce")
        return df
    except Exception:
        return None


def pct_to_price(entry: float, pct: float) -> float:
    if not (isinstance(entry, (int, float)) and math.isfinite(entry)):
        return float("nan")
    return float(entry * (1.0 + pct / 100.0))


def compute_confidence(row: pd.Series,
                       tf: str,
                       imf_amp: Optional[float],
                       imf_dur: Optional[float],
                       macro_align: Optional[bool],
                       abnormal_flow: bool) -> float:
    c = 0.5
    # Intraday trend/momentum alignment
    rsi_pct = row.get(f"rsi_{tf}_pctile")
    macd_hist = row.get(f"macd_histogram_{tf}")
    if isinstance(rsi_pct, (int, float)) and math.isfinite(rsi_pct) and rsi_pct > 55 and isinstance(macd_hist, (int, float)) and math.isfinite(macd_hist) and macd_hist > 0:
        c += 0.15
    # Volatility context (breakout or reversion) via width and ATR percentile if available
    bw = row.get(f"bollinger_width_{tf}")
    atr = row.get(f"atr_{tf}")
    bw_ok = isinstance(bw, (int, float)) and math.isfinite(bw)
    atr_ok = isinstance(atr, (int, float)) and math.isfinite(atr)
    if bw_ok and atr_ok:
        c += 0.10
    # IMF quality
    if imf_amp is not None and math.isfinite(imf_amp) and imf_amp >= 10.0:
        c += 0.15
    # Macro alignment bonus
    if macro_align:
        c += 0.10
    # Abnormal flow penalty
    if abnormal_flow:
        c -= 0.15
    return max(0.0, min(1.0, c))


def rules_for_timeframe(row: pd.Series,
                        tf: str,
                        imf: Dict[str, float],
                        regime_row: Optional[pd.Series],
                        cfg: Dict) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    sfx = tf
    # Read inputs
    asof = row.get("timestamp_asof_utc")
    price = row.get("close_current")
    upper = row.get(f"bollinger_upper_{sfx}")
    lower = row.get(f"bollinger_lower_{sfx}")
    bb_pos = row.get(f"bb_position_{sfx}_pos")
    rsi_pct = row.get(f"rsi_{sfx}_pctile")
    adx_mag = row.get(f"adx_{sfx}")
    atr = row.get(f"atr_{sfx}")
    roc_pctile = row.get(f"roc_{sfx}_pctile")
    vol_spike_share = row.get(f"vol_spike_share_{sfx}_pos")
    qv_pct = row.get(f"quote_volume_sum_{sfx}_pctile")
    macd_hist = row.get(f"macd_histogram_{sfx}")

    amp = imf.get("median_amplitude_pct", float("nan")) if imf else float("nan")
    dur = imf.get("median_duration_min", float("nan")) if imf else float("nan")

    # Macro alignment (soft bonus)
    macro_align = None
    if regime_row is not None:
        ma50_ok = bool(regime_row.get("ma50_slope_30d_dir", 0.0) > 0)
        ma200_ok = bool(regime_row.get("ma200_slope_90d_dir", 0.0) > 0)
        above_ok = bool(regime_row.get("above_ma50_share_30d_pos", 0.0) > 0.5)
        macro_align = ma50_ok and ma200_ok and above_ok

    # Abnormal flow tag
    abnormal_flow = bool((isinstance(qv_pct, (int, float)) and qv_pct > cfg["thresholds"]["abnormal_qv_pctile"]) or
                         (isinstance(vol_spike_share, (int, float)) and vol_spike_share > cfg["thresholds"]["abnormal_vol_spike_share"]))

    # Common checks
    adx_ok = isinstance(adx_mag, (int, float)) and adx_mag > cfg["thresholds"]["adx_min"]

    # Breakout trigger/setup
    # Trigger when price above upper band or bb_pos ~ 1.0
    brk_trigger = (isinstance(bb_pos, (int, float)) and bb_pos > 0.98) or (isinstance(price, (int, float)) and isinstance(upper, (int, float)) and price > upper)
    brk_setup = (isinstance(price, (int, float)) and isinstance(upper, (int, float)) and isinstance(atr, (int, float)) and (upper - price) <= max(0.0, 0.5 * atr) and (upper - price) >= 0.0)
    rsi_band_ok = isinstance(rsi_pct, (int, float)) and 60 <= rsi_pct <= 85
    trend_ok = isinstance(macd_hist, (int, float)) and macd_hist > 0 and ((isinstance(rsi_pct, (int, float)) and rsi_pct > 55) or (isinstance(roc_pctile, (int, float)) and roc_pctile > 55))

    if adx_ok and trend_ok and isinstance(atr, (int, float)):
        for status in (["trigger"] if brk_trigger else ( ["setup"] if brk_setup else [])):
            entry = float(price) if isinstance(price, (int, float)) else float("nan")
            stop = entry - cfg["stops"]["breakout_atr_mult"] * float(atr)
            t1_pct = min(float(amp) if math.isfinite(amp) else cfg["targets"]["default_t1_pct"], cfg["targets"]["roi_cap_pct"])
            t2_pct = None
            if macro_align and math.isfinite(amp) and amp > 20.0:
                t2_pct = 1.5 * t1_pct
            exp_hold = float(dur) if math.isfinite(dur) else cfg["timing"]["default_hold_min"]
            roi_per_hour = float(t1_pct / max(exp_hold / 60.0, 1e-6))
            conf = compute_confidence(row, sfx, amp, dur, macro_align, abnormal_flow)
            tags = ["trend", "strong_adx"]
            if abnormal_flow:
                tags.append("abnormal_flow")
            if status == "setup":
                tags.append("pre_breakout")
            notes = "Above band" if status == "trigger" else "Within 0.5*ATR of upper band"
            out.append({
                "symbol": row.get("symbol"),
                "timeframe": sfx,
                "signal_type": "long_breakout",
                "status": status,
                "confidence": round(conf, 3),
                "entry": round(entry, 8) if math.isfinite(entry) else entry,
                "stop": round(stop, 8) if math.isfinite(stop) else stop,
                "target1": round(pct_to_price(entry, t1_pct), 8) if math.isfinite(entry) else float("nan"),
                "target2": round(pct_to_price(entry, t2_pct), 8) if (t2_pct is not None and math.isfinite(entry)) else None,
                "expected_hold_min": int(exp_hold) if math.isfinite(exp_hold) else None,
                "roi_per_hour": round(roi_per_hour, 4),
                "tags": ",".join(tags),
                "notes": notes,
                "asof": asof.isoformat() if hasattr(asof, "isoformat") else str(asof),
            })

    # Mean-reversion trigger/setup
    rev_trigger = (isinstance(bb_pos, (int, float)) and bb_pos < 0.05) or (isinstance(price, (int, float)) and isinstance(lower, (int, float)) and price < lower)
    rev_setup = (isinstance(price, (int, float)) and isinstance(lower, (int, float)) and isinstance(atr, (int, float)) and (price - lower) <= max(0.0, 0.5 * atr) and (price - lower) >= 0.0)
    rsi_low = isinstance(rsi_pct, (int, float)) and rsi_pct < 20
    if rsi_low and isinstance(atr, (int, float)):
        for status in (["trigger"] if rev_trigger else ( ["setup"] if rev_setup else [])):
            entry = float(price) if isinstance(price, (int, float)) else float("nan")
            stop = entry - cfg["stops"]["reversion_atr_mult"] * float(atr)
            # Favor small amplitude targets for reversion if IMF small; else cap by ROI cap
            base_amp = float(amp) if math.isfinite(amp) else cfg["targets"]["default_t1_pct"]
            t1_pct = min(base_amp, cfg["targets"]["roi_cap_pct"])
            t2_pct = None
            exp_hold = float(dur) if math.isfinite(dur) else cfg["timing"]["default_hold_min"]
            roi_per_hour = float(t1_pct / max(exp_hold / 60.0, 1e-6))
            conf = compute_confidence(row, sfx, amp, dur, macro_align, abnormal_flow)
            tags = ["reversion"]
            if abnormal_flow:
                tags.append("abnormal_flow")
            if status == "setup":
                tags.append("pre_reversion")
            notes = "Below band" if status == "trigger" else "Within 0.5*ATR of lower band"
            out.append({
                "symbol": row.get("symbol"),
                "timeframe": sfx,
                "signal_type": "long_reversion",
                "status": status,
                "confidence": round(conf, 3),
                "entry": round(entry, 8) if math.isfinite(entry) else entry,
                "stop": round(stop, 8) if math.isfinite(stop) else stop,
                "target1": round(pct_to_price(entry, t1_pct), 8) if math.isfinite(entry) else float("nan"),
                "target2": round(pct_to_price(entry, t2_pct), 8) if (t2_pct is not None and math.isfinite(entry)) else None,
                "expected_hold_min": int(exp_hold) if math.isfinite(exp_hold) else None,
                "roi_per_hour": round(roi_per_hour, 4),
                "tags": ",".join(tags),
                "notes": notes,
                "asof": asof.isoformat() if hasattr(asof, "isoformat") else str(asof),
            })

    return out


def load_config(path: Optional[str]) -> Dict:
    # Defaults from user feedback
    cfg = {
        "timeframes": TF_DEFAULTS,
        "stops": {
            "breakout_atr_mult": 1.25,
            "reversion_atr_mult": 0.85,
            "band_buffer_atr": 0.1,
        },
        "targets": {
            "roi_cap_pct": 25.0,
            "default_t1_pct": 8.0,
        },
        "thresholds": {
            "squeeze_pctile": 30.0,
            "squeeze_ratio": 1.0,
            "abnormal_qv_pctile": 98.0,
            "abnormal_vol_spike_share": 0.30,
            "adx_min": 25.0,
        },
        "timing": {
            "default_hold_min": 1440.0,
        },
    }
    if path and os.path.exists(path) and yaml is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                u = yaml.safe_load(f) or {}
            # Shallow merge
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        except Exception:
            pass
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rule-based signals emitter for overlap cohort")
    p.add_argument("--metrics", default="overlap_metrics.csv", help="Metrics CSV (overlaps)")
    p.add_argument("--imf-json", default="imf_clusters_overlaps.json", help="IMF clusters JSON (overlaps)")
    p.add_argument("--imf-txt", default="imf_clusters_overlaps.txt", help="IMF clusters TXT fallback (overlaps)")
    p.add_argument("--regime", help="Optional regime snapshot path (daily)")
    p.add_argument("--config", help="Optional rules_config.yaml")
    p.add_argument("--timeframes", help="CSV timeframes to emit (e.g., 1h,4h,12h,24h)")
    p.add_argument("--out-csv", default="signals.csv", help="Output CSV path for signals")
    p.add_argument("--write-cards", action="store_true", help="Also write human-readable signals.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.timeframes:
        cfg["timeframes"] = [t.strip().lower() for t in args.timeframes.split(",") if t.strip()]

    metrics = load_metrics(args.metrics)
    imf_map = load_imf_summary(args.imf_json, args.imf_txt)
    regime_df = load_regime(args.regime)
    if regime_df is not None and "symbol" in regime_df.columns:
        # Join key: (symbol, timestamp_asof_utc)
        regime_key = regime_df.set_index(["symbol", "timestamp_asof_utc"]) if "timestamp_asof_utc" in regime_df.columns else regime_df.set_index(["symbol"])
    else:
        regime_key = None

    rows: List[Dict[str, object]] = []
    for sym, g in metrics.groupby("symbol"):
        imf = imf_map.get(str(sym), {})
        regime_row = None
        if regime_key is not None:
            key = (sym, g["timestamp_asof_utc"].iloc[0]) if "timestamp_asof_utc" in g.columns else sym
            try:
                regime_row = regime_key.loc[key]
            except Exception:
                regime_row = None
            if isinstance(regime_row, pd.DataFrame):
                regime_row = regime_row.iloc[0]
        r = g.iloc[0]
        for tf in cfg["timeframes"]:
            signals = rules_for_timeframe(r, tf, imf, regime_row, cfg)
            rows.extend(signals)

    sig_df = pd.DataFrame(rows)
    if not sig_df.empty:
        sig_df = sig_df.sort_values(by=["confidence", "roi_per_hour"], ascending=[False, False])
    sig_df.to_csv(args.out_csv, index=False)

    # Console summary (top 20)
    if not sig_df.empty:
        print(f"Signals: assets={sig_df['symbol'].nunique()} rows={len(sig_df)} saved={args.out_csv}")
        head = sig_df.head(20)
        cols = ["symbol", "timeframe", "signal_type", "status", "confidence", "roi_per_hour", "tags"]
        print(head[cols].to_string(index=False))
    else:
        print("No signals generated.")

    # Optional play cards
    if args.write_cards and not sig_df.empty:
        lines: List[str] = []
        for sym, sg in sig_df.groupby("symbol"):
            lines.append(f"[{sym}]")
            for _, r2 in sg.head(6).iterrows():
                lines.append(
                    f"- {r2['timeframe']} {r2['signal_type']} ({r2['status']}): conf={r2['confidence']:.2f}, roi/h={r2['roi_per_hour']:.2f}, tags={r2['tags']}"
                )
            lines.append("")
        with open("signals.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("Wrote signals.txt")


if __name__ == "__main__":
    main()

