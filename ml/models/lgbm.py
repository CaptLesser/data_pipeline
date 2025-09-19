from __future__ import annotations

from typing import Dict

from ml.metrics import regression_metrics, classification_metrics
from ml.presets import LGBM_PRESETS


def build_model(task: str, tuning: str = "moderate", **overrides):
    try:
        import lightgbm as lgb  # noqa: F401
    except Exception:
        print("lightgbm not installed; skipping run.")
        raise SystemExit(0)
    params = LGBM_PRESETS.get(tuning, LGBM_PRESETS["moderate"]).copy()
    params.update({k: v for k, v in overrides.items() if v is not None})
    if task in ("price", "return", "vol"):
        from lightgbm import LGBMRegressor
        params.setdefault("objective", "regression")
        params.setdefault("metric", "rmse")
        return LGBMRegressor(**params)
    else:
        from lightgbm import LGBMClassifier
        # default binary
        params.setdefault("objective", "binary")
        params.setdefault("metric", "aucpr")
        return LGBMClassifier(**params)


def fit_and_score(model, X_tr, y_tr, X_va, y_va, task: str) -> Dict[str, float]:
    eval_set = []
    if X_va is not None and y_va is not None:
        eval_set = [(X_va, y_va)]
    try:
        if eval_set:
            model.fit(X_tr, y_tr, eval_set=eval_set, verbose=-1)
        else:
            model.fit(X_tr, y_tr)
    except Exception:
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
