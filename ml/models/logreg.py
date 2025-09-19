from __future__ import annotations

from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.metrics import classification_metrics
from ml.presets import LOGREG_PRESETS


def build_model(task: str, tuning: str = "moderate", **overrides):
    if task not in ("direction", "event"):
        raise SystemExit("logreg supports classification tasks only")
    params = LOGREG_PRESETS.get(tuning, LOGREG_PRESETS["moderate"]).copy()
    # Force multinomial when multi-class direction
    if overrides.get("classes", 2) and int(overrides.get("classes", 2)) > 2:
        params.update({"solver": "lbfgs", "multi_class": "multinomial"})
    clf = LogisticRegression(**params)
    return Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", clf)])


def fit_and_score(model, X_tr, y_tr, X_va, y_va, task: str) -> Dict[str, float]:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_va)
    try:
        y_prob = model.predict_proba(X_va)
    except Exception:
        y_prob = None
    return classification_metrics(y_va, y_pred, y_prob)
