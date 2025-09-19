from __future__ import annotations

from typing import Dict

from ml.metrics import regression_metrics, classification_metrics
from ml.presets import CAT_PRESETS


def build_model(task: str, tuning: str = "moderate", **overrides):
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except Exception:
        print("catboost not installed; skipping run.")
        raise SystemExit(0)
    params = CAT_PRESETS.get(tuning, CAT_PRESETS["moderate"]).copy()
    params.update({k: v for k, v in overrides.items() if v is not None})
    # Use default loss by task
    if task in ("price", "return", "vol"):
        params.setdefault("loss_function", "RMSE")
        return CatBoostRegressor(**params)
    else:
        params.setdefault("loss_function", "Logloss")
        return CatBoostClassifier(**params)


def fit_and_score(model, X_tr, y_tr, X_va, y_va, task: str) -> Dict[str, float]:
    try:
        if X_va is not None and y_va is not None:
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        else:
            model.fit(X_tr, y_tr)
    except Exception:
        model.fit(X_tr, y_tr)
    y_pred = model.predict(X_va)
    if task in ("price", "return", "vol"):
        # CatBoost returns shape (n,1)
        try:
            y_pred = y_pred.reshape(-1)
        except Exception:
            pass
        return regression_metrics(y_va, y_pred)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_va)
    except Exception:
        pass
    return classification_metrics(y_va, y_pred, y_prob)
