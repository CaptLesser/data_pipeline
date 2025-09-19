from __future__ import annotations

from typing import Dict

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.metrics import regression_metrics
from ml.presets import ENET_PRESETS


def build_model(task: str, tuning: str = "moderate"):
    if task not in ("price", "return", "vol"):
        raise SystemExit("enet supports regression tasks only")
    params = ENET_PRESETS.get(tuning, ENET_PRESETS["moderate"]).copy()
    reg = ElasticNet(random_state=42, **params)
    return Pipeline([("scaler", StandardScaler(with_mean=False)), ("reg", reg)])


def fit_and_score(model, X_tr, y_tr, X_va, y_va, task: str) -> Dict[str, float]:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_va)
    return regression_metrics(y_va, y_pred)

