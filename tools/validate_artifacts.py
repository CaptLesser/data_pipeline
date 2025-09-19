from __future__ import annotations

import argparse
import json
import os
import sys
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate model artifacts against a given dataset (schema, dtypes)")
    p.add_argument("--model-dir", required=True, help="Path to models/<task>/<model>/<run_id>/H=<H>")
    p.add_argument("--dataset", required=True, help="CSV to validate (must contain required features)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cm_path = os.path.join(args.model_dir, "column_map.json")
    feats_path = os.path.join(args.model_dir, "features_used.txt")
    if not os.path.exists(cm_path) or not os.path.exists(feats_path):
        print("Missing column_map.json or features_used.txt in model dir", file=sys.stderr)
        sys.exit(2)
    with open(cm_path, "r", encoding="utf-8") as f:
        cmap = json.load(f)
    with open(feats_path, "r", encoding="utf-8") as f:
        feats = [line.strip() for line in f if line.strip()]
    df = pd.read_csv(args.dataset, nrows=100)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        print("Missing columns:")
        for c in missing:
            print(" -", c)
        wins = cmap.get("windows")
        if wins:
            print(f"Hint: ensure metrics were generated with --wins {','.join(wins)}")
        sys.exit(1)
    print("OK: dataset contains all required feature columns")


if __name__ == "__main__":
    main()

