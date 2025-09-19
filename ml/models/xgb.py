from __future__ import annotations

from typing import Dict

import numpy as np

from ml.metrics import regression_metrics, classification_metrics
from ml.presets import XGB_PRESETS


def build_model(task: str, tuning: str = "moderate", **overrides):
    try:
        import xgboost as xgb  # noqa: F401
    except Exception:
        print("xgboost not installed; skipping run.")
        raise SystemExit(0)
    params = XGB_PRESETS.get(tuning, XGB_PRESETS["moderate"]).copy()
    params.update({k: v for k, v in overrides.items() if v is not None})
    # Map to correct estimator
    if task in ("price", "return", "vol"):
        from xgboost import XGBRegressor
        params.setdefault("objective", "reg:squarederror")
        params.setdefault("eval_metric", "rmse")
        return XGBRegressor(**params)
    else:
        from xgboost import XGBClassifier
        # default multi vs binary decided downstream by y
        params.setdefault("objective", "binary:logistic")
        params.setdefault("eval_metric", "aucpr")
        return XGBClassifier(**params)


def fit_and_score(model, X_tr, y_tr, X_va, y_va, task: str) -> Dict[str, float]:
    # Early stopping if supported and val provided
    es_rounds = None
    try:
        es_rounds = int(getattr(model, "early_stopping_rounds", 0) or 0)
    except Exception:
        es_rounds = 0
    eval_set = []
    if X_va is not None and y_va is not None:
        eval_set = [(X_va, y_va)]
    try:
        if eval_set:
            model.set_params(early_stopping_rounds=model.get_params().get("early_stopping_rounds", None))
            model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
        else:
            model.fit(X_tr, y_tr)
    except TypeError:
        # older xgb signature
        model.fit(X_tr, y_tr)
    y_pred = model.predict(X_va)
    if task in ("price", "return", "vol"):
        return regression_metrics(y_va, y_pred)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_va)
    except Exception:
        pass
    return classification_metrics(y_va, y_pred, y_prob)
