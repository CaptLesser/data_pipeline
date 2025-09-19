from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ml.metrics import regression_metrics, classification_metrics
from ml.presets import HGBT_PRESETS


def build_model(task: str, tuning: str = "moderate"):
    params = HGBT_PRESETS.get(tuning, HGBT_PRESETS["moderate"]).copy()
    if task in ("price", "return", "vol"):
        return HistGradientBoostingRegressor(**params)
    return HistGradientBoostingClassifier(**params)


def fit_and_score(model, X_tr, y_tr, X_va, y_va, task: str) -> Dict[str, float]:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_va)
    if task in ("price", "return", "vol"):
        return regression_metrics(y_va, y_pred)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_va)
    except Exception:
        y_prob = None
    return classification_metrics(y_va, y_pred, y_prob)

