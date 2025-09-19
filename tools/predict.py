from __future__ import annotations

import argparse
import json
import os
import pickle
import pandas as pd
import numpy as np

from ml.predict import predict_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference: apply trained model to a dataset and emit predictions.parquet")
    p.add_argument("--model-dir", required=True, help="Path to models/.../H=<H> directory")
    p.add_argument("--dataset", required=True, help="CSV with required features + symbol,timestamp")
    p.add_argument("--output", default=None, help="Output Parquet path (default predictions.parquet in model dir)")
    p.add_argument("--classes", type=int, default=2, help="2 or 3 (for direction)")
    p.add_argument("--task", required=True, help="price|return|direction|vol|event")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(os.path.join(args.model_dir, "features_used.txt"), "r", encoding="utf-8") as f:
        feats = [l.strip() for l in f if l.strip()]
    with open(os.path.join(args.model_dir, "column_map.json"), "r", encoding="utf-8") as f:
        cmap = json.load(f)
    with open(os.path.join(args.model_dir, "metrics.json"), "r", encoding="utf-8") as f:
        m = json.load(f)
    best_iter = None
    try:
        best_iter = int(m.get("best_iteration") or m.get("val", {}).get("best_iteration") or 0)
    except Exception:
        best_iter = None
    model = pickle.load(open(os.path.join(args.model_dir, "model.pkl"), "rb"))
    calibrator = None
    cal_path = os.path.join(args.model_dir, "calibrator.pkl")
    if os.path.exists(cal_path):
        try:
            calibrator = pickle.load(open(cal_path, "rb"))
        except Exception:
            calibrator = None

    df = pd.read_csv(args.dataset)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing features in dataset: {missing}")
    # Build X
    X = df[feats]
    # Handle CatBoost native symbol, else numeric matrix
    if any(f == "symbol" for f in feats):
        Xmat = X
    else:
        Xmat = X.to_numpy(dtype=float)
    # Predict
    if calibrator is not None and args.task in ("direction", "event") and int(args.classes) == 2:
        try:
            y_prob = calibrator.predict_proba(Xmat)
            y_prob = y_prob[:, 1] if hasattr(y_prob, 'ndim') and y_prob.ndim == 2 else y_prob
            y_hat = (y_prob >= float(cmap.get("decision_threshold", 0.5))).astype(int)
        except Exception:
            y_prob, y_hat = predict_out(model, Xmat, args.task, int(args.classes), best_iter)
    else:
        y_prob, y_hat = predict_out(model, Xmat, args.task, int(args.classes), best_iter)

    out = pd.DataFrame({
        "timestamp": df.get("timestamp"),
        "symbol": df.get("symbol"),
        "proba": y_prob if y_prob is not None else np.nan,
        "label": y_hat,
    })
    out_path = args.output or os.path.join(args.model_dir, "predictions.parquet")
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} predictions to {out_path}")


if __name__ == "__main__":
    main()

