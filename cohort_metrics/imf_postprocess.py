import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-9


def _mad_pct(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    if med == 0:
        return float("nan")
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad / abs(med) * 100.0)


def _load_json(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_dt_dir(summary_path: str) -> str:
    # Expect path .../data/imf/dt=YYYY-MM-DD/imf_summary.parquet
    d = os.path.dirname(summary_path)
    return d


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _amplitude_band(amp: float, small: float, medium: float) -> str:
    if not np.isfinite(amp):
        return "unknown"
    if amp < small:
        return "small"
    if amp <= medium:
        return "medium"
    return "large"


def _dominant_score(reliability: float, energy_share: float, amp_pct: float) -> float:
    # Option B: balanced
    comp = 0.6 * float(energy_share) + 0.4 * min(1.0, float(amp_pct) / 20.0)
    return float(reliability) * comp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IMF post-processing to compact LLM/ML-friendly summaries")
    p.add_argument("--summary", help="Path to imf_summary.parquet", required=True)
    p.add_argument("--cards-price", help="Optional path to cluster_cards_global_price.json")
    p.add_argument("--cards-volume", help="Optional path to cluster_cards_global_volume.json")
    p.add_argument("--cards-asset", help="Optional path to cluster_cards_asset.json")
    p.add_argument("--run-config", help="Optional path to run_config.json")
    p.add_argument("--series", choices=["price", "volume", "both"], default="price")
    p.add_argument("--asof", help="UTC asof timestamp (fallback to run_config)")
    p.add_argument("--window-days", type=float, help="Window length in days (fallback to run_config)")
    p.add_argument("--min-energy-share", type=float, default=0.0)
    p.add_argument("--amp-bands", nargs=2, type=float, default=[5.0, 12.0], metavar=("SMALL","MEDIUM"))
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    p.add_argument("--out-dir", help="Output base directory (default inferred from summary)")
    p.add_argument("--emit-fingerprints", action="store_true", help="Optional extra output (reserved)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.summary) if args.summary.endswith(".parquet") else pd.read_csv(args.summary)
    if df.empty:
        raise SystemExit("imf_summary is empty")

    run_cfg = _load_json(args.run_config)
    asof = pd.to_datetime(args.asof, utc=True) if args.asof else None
    if asof is None and run_cfg and "asof" in run_cfg:
        asof = pd.to_datetime(run_cfg.get("asof"), utc=True)
    if asof is None:
        # fallback from data
        asof_vals = pd.to_datetime(df["asof"], utc=True, errors="coerce") if "asof" in df.columns else None
        asof = asof_vals.iloc[0] if asof_vals is not None and len(asof_vals) else pd.Timestamp.utcnow().tz_localize("UTC")

    if args.window_days is not None:
        window_days = float(args.window_days)
    elif run_cfg and "window_start" in run_cfg:
        wstart = pd.to_datetime(run_cfg["window_start"], utc=True)
        window_days = max(1.0, (asof - wstart).total_seconds() / (60.0 * 60.0 * 24.0))
    else:
        raise SystemExit("--window-days is required if run_config is not provided")

    dt_dir = args.out_dir or _infer_dt_dir(args.summary)
    _ensure_dir(dt_dir)

    # Filter series
    series_sel = ["price", "volume"] if args.series == "both" else [args.series]
    df = df[df["series"].isin(series_sel)].copy()

    # Drop statuses != ok
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    # Optional drop by energy_share threshold
    if args.min_energy_share > 0:
        df = df[df["energy_share"] >= float(args.min_energy_share)].copy()

    # Flags
    # singleton_global_flag: cluster_id_global startswith GP9/GV9/GM9
    def singleton_flag(cid: str) -> bool:
        if not isinstance(cid, str):
            return False
        return any(cid.startswith(p) for p in ("GP9", "GV9", "GM9"))

    df["singleton_global_flag"] = df["cluster_id_global"].astype(str).apply(singleton_flag)
    df["expected_cycles"] = window_days / df["period_days"].astype(float)
    df["undersampled_flag"] = df["expected_cycles"] < 1.5
    df["long_cycle_flag"] = df["period_days"] > (2.0 / 3.0) * window_days

    # Reliability
    madp = df.get("cluster_mad_pct_period_global", pd.Series([np.nan] * len(df)))
    mada = df.get("cluster_mad_pct_amp_global", pd.Series([np.nan] * len(df)))
    R_period = np.exp(-(madp.astype(float)) / 25.0)
    R_amp = np.exp(-(mada.astype(float)) / 25.0)
    R_period = R_period.where(np.isfinite(R_period), 0.5)
    R_amp = R_amp.where(np.isfinite(R_amp), 0.5)
    R_base = 0.6 * R_period + 0.4 * R_amp
    alpha = np.minimum(1.0, df["expected_cycles"].astype(float) / 2.0)
    reliability = alpha * R_base + (1.0 - alpha) * 0.4
    # size boost
    csize = df.get("cluster_size_global", pd.Series([np.nan] * len(df)))
    size_boost = np.minimum(0.15, (np.log1p(csize.astype(float)) / np.log(50.0)) * 0.15)
    size_boost = size_boost.where(np.isfinite(size_boost), 0.0)
    reliability = np.minimum(1.0, reliability + size_boost)
    df["reliability"] = reliability.astype(float)

    # Score (Option B)
    df["selection_score"] = df.apply(lambda r: _dominant_score(r["reliability"], r["energy_share"], r["amp_pct"]), axis=1)

    # Amplitude bands
    small, medium = float(args.amp_bands[0]), float(args.amp_bands[1])
    df["amp_band"] = df["amp_pct"].astype(float).apply(lambda a: _amplitude_band(a, small, medium))

    # Aggregates per symbol/series
    rows_summary: List[Dict[str, object]] = []
    rows_topk: List[Dict[str, object]] = []

    # Missing share: from run_config per symbol if provided
    syms_missing: Dict[str, float] = {}
    if run_cfg and isinstance(run_cfg.get("bars_per_symbol"), dict):
        for sym, obj in run_cfg["bars_per_symbol"].items():
            try:
                syms_missing[sym] = float(obj.get("missing_share", 0.0))
            except Exception:
                pass

    for (sym, series), g in df.groupby(["symbol", "series" ]):
        g = g.copy()
        # Top-K selection ordering: by score desc, then amp_pct desc, then period_days asc
        g = g.sort_values(["selection_score", "amp_pct", "period_days"], ascending=[False, False, True])
        # Aggregates (weighted by energy_share)
        w = g["energy_share"].astype(float).to_numpy()
        wsum = float(np.sum(w)) if np.isfinite(w).all() else float("nan")
        if wsum and np.isfinite(wsum) and wsum > 0:
            agg_period = float(np.sum(w * g["period_days"].astype(float).to_numpy()) / wsum)
            agg_amp = float(np.sum(w * g["amp_pct"].astype(float).to_numpy()) / wsum)
        else:
            agg_period = float(np.nan)
            agg_amp = float(np.nan)
        mad_all = _mad_pct(g["period_days"].astype(float).to_numpy())
        dom = g.iloc[0]
        alt1 = g.iloc[1] if len(g) > 1 else None
        alt2 = g.iloc[2] if len(g) > 2 else None
        # Text summary
        dom_period = float(dom["period_days"]) if np.isfinite(dom["period_days"]) else float("nan")
        dom_amp = float(dom["amp_pct"]) if np.isfinite(dom["amp_pct"]) else float("nan")
        dom_rel = float(dom["reliability"]) if np.isfinite(dom["reliability"]) else float("nan")
        dom_band = str(dom["amp_band"]) if isinstance(dom["amp_band"], str) else ""
        alt_txt = "/".join(filter(None, [
            f"{float(alt1['period_days']):.1f}" if alt1 is not None and np.isfinite(alt1["period_days"]) else "",
            f"{float(alt2['period_days']):.1f}" if alt2 is not None and np.isfinite(alt2["period_days"]) else "",
        ]))
        cid = str(dom.get("cluster_id_global", ""))
        cid_part = f" ({cid})" if cid and len(cid) <= 8 else ""
        unders = "yes" if bool(dom.get("undersampled_flag", False)) else "no"
        text_summary = (
            f"Cycle ~{dom_period:.1f}d, {dom_band} (≈{dom_amp:.1f}%). "
            f"Alt: {alt_txt}d. Rel {dom_rel:.2f}{cid_part}; undersampled: {unders}."
        ).strip()
        miss_share = syms_missing.get(sym, None)
        # summary row
        rows_summary.append({
            "symbol": sym,
            "asof": asof.to_pydatetime(),
            "series": series,
            "agg_period_days": agg_period,
            "agg_amp_pct": agg_amp,
            "mad_pct_period_all": float(mad_all),
            "dom_period_days": dom_period,
            "dom_amp_pct": dom_amp,
            "dom_energy_share": float(dom["energy_share"]),
            "dom_reliability": dom_rel,
            "dom_cluster_id_global": cid,
            "dom_cluster_size_global": int(dom.get("cluster_size_global", 0)) if np.isfinite(dom.get("cluster_size_global", np.nan)) else 0,
            "dom_score": float(dom["selection_score"]),
            "dom_amp_band": dom_band,
            "dom_expected_cycles": float(dom["expected_cycles"]),
            "dom_singleton_global_flag": bool(dom.get("singleton_global_flag", False)),
            "dom_undersampled_flag": bool(dom.get("undersampled_flag", False)),
            "missing_share": float(miss_share) if miss_share is not None else float("nan"),
            "text_summary": text_summary,
        })
        # top-K table rows
        for rank, r in enumerate(g.head(3).itertuples(index=False), start=1):
            rows_topk.append({
                "symbol": getattr(r, "symbol"),
                "asof": asof.to_pydatetime(),
                "series": getattr(r, "series"),
                "rank": rank,
                "period_days": float(getattr(r, "period_days")),
                "amp_pct": float(getattr(r, "amp_pct")),
                "energy_share": float(getattr(r, "energy_share")),
                "reliability": float(getattr(r, "reliability")),
                "cluster_id_global": str(getattr(r, "cluster_id_global")),
                "cluster_size_global": int(getattr(r, "cluster_size_global")) if np.isfinite(getattr(r, "cluster_size_global", np.nan)) else 0,
                "singleton_global_flag": bool(getattr(r, "singleton_global_flag")),
                "undersampled_flag": bool(getattr(r, "undersampled_flag")),
            })

    post_summary = pd.DataFrame(rows_summary)
    topk = pd.DataFrame(rows_topk)

    # Write outputs
    out1 = os.path.join(dt_dir, "imf_post_summary.parquet")
    out2 = os.path.join(dt_dir, "imf_post_top_imfs.parquet")
    if args.format == "parquet":
        post_summary.to_parquet(out1, index=False)
        topk.to_parquet(out2, index=False)
    else:
        out1 = out1.replace(".parquet", ".csv")
        out2 = out2.replace(".parquet", ".csv")
        post_summary.to_csv(out1, index=False)
        topk.to_csv(out2, index=False)

    print(f"IMF postprocess: summary rows={len(post_summary)} topK rows={len(topk)} out_dir={dt_dir}")


if __name__ == "__main__":
    main()

