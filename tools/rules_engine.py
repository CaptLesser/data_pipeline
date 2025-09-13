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


def load_imf_summary(path_json: Optional[str], path_txt: Optional[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return mapping symbol -> window_label -> {median_amplitude_pct, median_duration_min, method}.

    Prefers JSON; falls back to TXT if provided (TXT is window-agnostic; used for 24h mapping).
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    if path_json and os.path.exists(path_json):
        try:
            with open(path_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            # JSON structure: summary[coin][window_label]['gmm']/['dbscan'] → list of clusters
            # Select per-window dominant cluster by preference:
            # 1) DBSCAN with flag_small=False and max count, else any DBSCAN with max count
            # 2) Else GMM with max count
            for coin, windows in data.items():
                out[coin] = {}
                for wlabel, res in windows.items():
                    choice = None
                    method = None
                    # Prefer DBSCAN
                    dbs = res.get("dbscan", [])
                    if dbs:
                        # Build candidates with (not flag_small, count)
                        cand = []
                        for cl in dbs:
                            count = int(cl.get("count", 0))
                            flag_small = bool(cl.get("flag_small", False))
                            cand.append((not flag_small, count, cl))
                        # Sort: not small first, then larger count
                        cand.sort(key=lambda x: (x[0], x[1]), reverse=True)
                        choice = cand[0][2]
                        method = "dbscan"
                    # Fallback GMM
                    if choice is None:
                        gmm = res.get("gmm", [])
                        if gmm:
                            gmm_sorted = sorted(gmm, key=lambda cl: int(cl.get("count", 0)), reverse=True)
                            choice = gmm_sorted[0]
                            method = "gmm"
                    if choice is not None:
                        amp = float(choice.get("median_amplitude_pct", np.nan))
                        dur = float(choice.get("median_duration_min", np.nan))
                        out[coin][wlabel] = {
                            "median_amplitude_pct": amp,
                            "median_duration_min": dur,
                            "method": method or "",
                        }
        except Exception:
            pass
    if (not out) and path_txt and os.path.exists(path_txt):
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
                        out.setdefault(sym, {})
                        # Assume TXT approximates 24h window
                        out[sym]["24h"] = {"median_amplitude_pct": amp, "median_duration_min": dur, "method": "txt"}
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


IMF_TF_MAP = {
    "1h": "6h",
    "4h": "12h",
    "12h": "24h",
    "24h": "24h",
}


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

    # Timeframe-aligned IMF selection
    amp = float("nan")
    dur = float("nan")
    imf_method = ""
    if imf:
        wlabel = IMF_TF_MAP.get(sfx, "24h")
        imf_row = imf.get(wlabel) if isinstance(imf, dict) else None
        if isinstance(imf_row, dict):
            amp = float(imf_row.get("median_amplitude_pct", float("nan")))
            dur = float(imf_row.get("median_duration_min", float("nan")))
            imf_method = str(imf_row.get("method", ""))

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

    # Breakout trigger/setup (asset-relative)
    # Trigger when price above upper band or bb_pos ~ 1.0
    brk_trigger = (isinstance(bb_pos, (int, float)) and bb_pos > 0.98) or (isinstance(price, (int, float)) and isinstance(upper, (int, float)) and price > upper)
    brk_setup = (isinstance(price, (int, float)) and isinstance(upper, (int, float)) and isinstance(atr, (int, float)) and (upper - price) <= max(0.0, 0.5 * atr) and (upper - price) >= 0.0)
    rsi_band_ok = isinstance(rsi_pct, (int, float)) and 60 <= rsi_pct <= 85
    trend_ok = isinstance(macd_hist, (int, float)) and macd_hist > 0 and ((isinstance(rsi_pct, (int, float)) and rsi_pct > 55) or (isinstance(roc_pctile, (int, float)) and roc_pctile > 55))

    # Apply squeeze tag
    squeeze_tag = False
    sq_pct = row.get(f"squeeze_index_{sfx}_mag_pctile")
    bw = row.get(f"bollinger_width_{sfx}")
    sq_idx = row.get(f"squeeze_index_{sfx}_mag")
    squeeze_ratio = None
    if isinstance(bw, (int, float)) and isinstance(atr, (int, float)) and atr > 0:
        squeeze_ratio = float(bw / (atr + 1e-12))
    if (isinstance(sq_pct, (int, float)) and sq_pct < cfg["thresholds"]["squeeze_pctile"]) or (
        isinstance(squeeze_ratio, (int, float)) and squeeze_ratio < cfg["thresholds"]["squeeze_ratio"]):
        squeeze_tag = True

    if adx_ok and trend_ok and rsi_band_ok and isinstance(atr, (int, float)):
        for status in (["trigger"] if brk_trigger else (["setup"] if brk_setup else [])):
            entry = float(price) if isinstance(price, (int, float)) else float("nan")
            # Respect band buffer for breakout: max(entry - k*ATR, upper - buffer*ATR)
            stop_atr = entry - cfg["stops"]["breakout_atr_mult"] * float(atr)
            stop_band = None
            if isinstance(upper, (int, float)):
                stop_band = float(upper) - cfg["stops"]["band_buffer_atr"] * float(atr)
            stop_candidates = [s for s in [stop_atr, stop_band] if isinstance(s, (int, float)) and math.isfinite(s)]
            stop = max(stop_candidates) if stop_candidates else stop_atr
            t1_pct = min(float(amp) if math.isfinite(amp) else cfg["targets"]["default_t1_pct"], cfg["targets"]["roi_cap_pct"])
            t2_pct = 1.5 * t1_pct if (macro_align and math.isfinite(amp) and amp > 20.0) else None
            exp_hold = float(dur) if (math.isfinite(dur) and dur > 0) else cfg["timing"]["default_hold_min"]
            tags = ["trend", "strong_adx"]
            if abnormal_flow:
                tags.append("abnormal_flow")
            if squeeze_tag:
                tags.append("squeeze")
            if status == "setup":
                tags.append("pre_breakout")
            # Build informative notes/audit
            notes = []
            notes.append(f"rsi_{sfx}_pctile={rsi_pct}")
            notes.append(f"adx_{sfx}={adx_mag}")
            notes.append(f"macd_hist>0")
            if squeeze_tag:
                if isinstance(sq_pct, (int, float)):
                    notes.append(f"squeeze_pctile={sq_pct}")
                if isinstance(squeeze_ratio, (int, float)):
                    notes.append(f"squeeze_ratio={squeeze_ratio:.2f}")
            if imf_method:
                notes.append(f"imf_window={IMF_TF_MAP.get(sfx,'24h')}:{imf_method}")
            if status == "trigger":
                notes.append("trigger=above_band")
            else:
                if isinstance(upper, (int, float)) and isinstance(atr, (int, float)):
                    dist = upper - price if (isinstance(price, (int, float))) else float("nan")
                    notes.append(f"setup=within_0.5ATR_upper: gap={dist:.6f}")
                else:
                    notes.append("setup=near_upper")
            note_txt = "; ".join(notes)
            out.append({
                "symbol": row.get("symbol"),
                "timeframe": sfx,
                "signal_type": "long_breakout",
                "status": status,
                "entry": round(entry, 8) if math.isfinite(entry) else entry,
                "stop": round(stop, 8) if math.isfinite(stop) else stop,
                "target1": round(pct_to_price(entry, t1_pct), 8) if math.isfinite(entry) else float("nan"),
                "target2": round(pct_to_price(entry, t2_pct), 8) if (t2_pct is not None and math.isfinite(entry)) else None,
                "expected_hold_min": int(exp_hold) if math.isfinite(exp_hold) else None,
                "tags": ",".join(tags),
                "notes": note_txt,
                "asof": asof.isoformat() if hasattr(asof, "isoformat") else str(asof),
                "imf_window_used": IMF_TF_MAP.get(sfx, "24h"),
                "imf_method": imf_method or "",
            })

    # Mean-reversion trigger/setup
    rev_trigger = (isinstance(bb_pos, (int, float)) and bb_pos < 0.05) or (isinstance(price, (int, float)) and isinstance(lower, (int, float)) and price < lower)
    rev_setup = (isinstance(price, (int, float)) and isinstance(lower, (int, float)) and isinstance(atr, (int, float)) and (price - lower) <= max(0.0, 0.5 * atr) and (price - lower) >= 0.0)
    rsi_low = isinstance(rsi_pct, (int, float)) and rsi_pct < 20
    if rsi_low and isinstance(atr, (int, float)):
        for status in (["trigger"] if rev_trigger else (["setup"] if rev_setup else [])):
            entry = float(price) if isinstance(price, (int, float)) else float("nan")
            # Respect band buffer for reversion: min(entry - k*ATR, lower - buffer*ATR)
            stop_atr = entry - cfg["stops"]["reversion_atr_mult"] * float(atr)
            stop_band = None
            if isinstance(lower, (int, float)):
                stop_band = float(lower) - cfg["stops"]["band_buffer_atr"] * float(atr)
            stop_candidates = [s for s in [stop_atr, stop_band] if isinstance(s, (int, float)) and math.isfinite(s)]
            policy = str(cfg.get("stops", {}).get("reversion_policy", "conservative")).lower()
            if stop_candidates:
                if policy == "wide":
                    stop = min(stop_candidates)
                else:  # conservative default
                    stop = max(stop_candidates)
            else:
                stop = stop_atr
            # Favor small amplitude targets for reversion if IMF small; else cap by ROI cap
            base_amp = float(amp) if math.isfinite(amp) else cfg["targets"]["default_t1_pct"]
            t1_pct = min(base_amp, cfg["targets"]["roi_cap_pct"])
            t2_pct = None
            exp_hold = float(dur) if (math.isfinite(dur) and dur > 0) else cfg["timing"]["default_hold_min"]
            tags = ["reversion"]
            if abnormal_flow:
                tags.append("abnormal_flow")
            if status == "setup":
                tags.append("pre_reversion")
            notes = []
            notes.append(f"rsi_{sfx}_pctile={rsi_pct}")
            if imf_method:
                notes.append(f"imf_window={IMF_TF_MAP.get(sfx,'24h')}:{imf_method}")
            notes.append("trigger=below_band" if status == "trigger" else "setup=near_lower")
            note_txt = "; ".join(notes)
            out.append({
                "symbol": row.get("symbol"),
                "timeframe": sfx,
                "signal_type": "long_reversion",
                "status": status,
                "entry": round(entry, 8) if math.isfinite(entry) else entry,
                "stop": round(stop, 8) if math.isfinite(stop) else stop,
                "target1": round(pct_to_price(entry, t1_pct), 8) if math.isfinite(entry) else float("nan"),
                "target2": round(pct_to_price(entry, t2_pct), 8) if (t2_pct is not None and math.isfinite(entry)) else None,
                "expected_hold_min": int(exp_hold) if math.isfinite(exp_hold) else None,
                "tags": ",".join(tags),
                "notes": note_txt,
                "asof": asof.isoformat() if hasattr(asof, "isoformat") else str(asof),
                "imf_window_used": IMF_TF_MAP.get(sfx, "24h"),
                "imf_method": imf_method or "",
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
            "reversion_policy": "conservative",
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
        "regime": {
            "join_tolerance_days": 3,
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

    rows: List[Dict[str, object]] = []
    for sym, g in metrics.groupby("symbol"):
        imf = imf_map.get(str(sym), {})
        regime_row = None
        if regime_df is not None and "symbol" in regime_df.columns:
            try:
                sym_reg = regime_df[regime_df["symbol"] == sym]
                if not sym_reg.empty and "timestamp_asof_utc" in sym_reg.columns and "timestamp_asof_utc" in g.columns:
                    asof = pd.to_datetime(g["timestamp_asof_utc"].iloc[0], utc=True, errors="coerce")
                    sym_reg = sym_reg.copy()
                    sym_reg["_delta"] = (sym_reg["timestamp_asof_utc"] - asof).abs()
                    rr = sym_reg.sort_values("_delta").iloc[0]
                    tol_days = float(cfg.get("regime", {}).get("join_tolerance_days", 3))
                    if pd.notna(rr.get("_delta")) and getattr(rr.get("_delta"), 'days', 9999) <= tol_days:
                        regime_row = rr
                    else:
                        regime_row = sym_reg.sort_values("timestamp_asof_utc").iloc[-1]
                elif not sym_reg.empty:
                    regime_row = sym_reg.sort_values("timestamp_asof_utc").iloc[-1] if "timestamp_asof_utc" in sym_reg.columns else sym_reg.iloc[-1]
            except Exception:
                regime_row = None
        r = g.iloc[0]
        for tf in cfg["timeframes"]:
            signals = rules_for_timeframe(r, tf, imf, regime_row, cfg)
            rows.extend(signals)

    sig_df = pd.DataFrame(rows)
    sig_df.to_csv(args.out_csv, index=False)

    # Console summary (top 20)
    if not sig_df.empty:
        print(f"Signals: assets={sig_df['symbol'].nunique()} rows={len(sig_df)} saved={args.out_csv}")
        head = sig_df.head(20)
        cols = ["symbol", "timeframe", "signal_type", "status", "entry", "stop", "target1", "target2", "tags"]
        try:
            print(head[cols].to_string(index=False))
        except Exception:
            print(head.to_string(index=False))
    else:
        print("No signals generated.")

    # Optional play cards
    if args.write_cards and not sig_df.empty:
        lines: List[str] = []
        for sym, sg in sig_df.groupby("symbol"):
            lines.append(f"[{sym}]")
            for _, r2 in sg.head(6).iterrows():
                try:
                    lines.append(
                        f"- {r2['timeframe']} {r2['signal_type']} ({r2['status']}): hold_min={r2.get('expected_hold_min','')}, target1={r2.get('target1','')}, tags={r2.get('tags','')}"
                    )
                except Exception:
                    lines.append(f"- {r2}")
            lines.append("")
        with open("signals.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("Wrote signals.txt")


if __name__ == "__main__":
    main()
