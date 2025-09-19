from __future__ import annotations

import json
import os
from typing import Dict, List, Optional
import pickle


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _dump_feature_importance(model, features: List[str], out_dir: str) -> Optional[str]:
    imp_path = os.path.join(out_dir, "feature_importance.json")
    try:
        import numpy as np  # noqa
        if hasattr(model, "feature_importances_") and getattr(model, "feature_importances_") is not None:
            vals = list(getattr(model, "feature_importances_"))
            data = {f: float(v) for f, v in zip(features, vals)}
            with open(imp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return imp_path
    except Exception:
        pass
    return None


def save_artifacts(out_dir: str, model, metrics: Dict, features: List[str], column_map: Dict) -> Dict[str, str]:
    _ensure_dir(out_dir)
    paths: Dict[str, str] = {}
    model_path = os.path.join(out_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    paths["model"] = model_path

    mpath = os.path.join(out_dir, "metrics.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    paths["metrics"] = mpath

    fpath = os.path.join(out_dir, "features_used.txt")
    with open(fpath, "w", encoding="utf-8") as f:
        for c in features:
            f.write(str(c) + "\n")
    paths["features"] = fpath

    # Augment column_map with run metadata (git sha, hash, etc.)
    try:
        import subprocess
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.getcwd()).decode().strip()
        column_map.setdefault("git_sha", sha)
    except Exception:
        pass
    cmap = os.path.join(out_dir, "column_map.json")
    with open(cmap, "w", encoding="utf-8") as f:
        json.dump(column_map, f, indent=2)
    paths["column_map"] = cmap

    imp = _dump_feature_importance(model, features, out_dir)
    if imp:
        paths["feature_importance"] = imp

    return paths
