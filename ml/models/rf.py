from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml.metrics import regression_metrics, classification_metrics
from ml.presets import RF_PRESETS


def build_model(task: str, tuning: str = "moderate"):
    params = RF_PRESETS.get(tuning, RF_PRESETS["moderate"]).copy()
    if task in ("price", "return", "vol"):
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    return RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced", **params)


def fit_and_score(model, X_tr, y_tr, X_va, y_va, task: str) -> Dict[str, float]:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_va)
    if task in ("price", "return", "vol"):
        return regression_metrics(y_va, y_pred)
    # probabilities optional
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_va)
        except Exception:
            y_prob = None
    return classification_metrics(y_va, y_pred, y_prob)

