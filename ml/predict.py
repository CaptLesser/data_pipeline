from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def predict_out(model, X, task: str, classes: int = 2, best_iter: Optional[int] = None) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Unified prediction helper.

    Returns (proba, decision_or_yhat):
      - For regression tasks: (None, yhat)
      - For binary classification: (p_pos, y_pred)
      - For multi-class classification: (proba_matrix, y_pred)
    Applies best-iteration caps when supported.
    """
    # Regression
    if task in ("price", "return", "vol"):
        # XGB/LGBM/Cat support limiting trees/iterations
        try:
            if hasattr(model, "predict"):
                # XGB: ntree_limit
                if "ntree_limit" in model.predict.__code__.co_varnames and best_iter is not None:
                    y = model.predict(X, ntree_limit=int(best_iter))
                # LGBM: num_iteration
                elif "num_iteration" in model.predict.__code__.co_varnames and best_iter is not None:
                    y = model.predict(X, num_iteration=int(best_iter))
                else:
                    y = model.predict(X)
            else:
                y = model.predict(X)
        except Exception:
            y = model.predict(X)
        return None, np.asarray(y)

    # Classification
    proba = None
    # Try proba with best-iteration caps
    try:
        if hasattr(model, "predict_proba"):
            if "ntree_limit" in model.predict_proba.__code__.co_varnames and best_iter is not None:
                proba = model.predict_proba(X, ntree_limit=int(best_iter))
            elif "num_iteration" in model.predict_proba.__code__.co_varnames and best_iter is not None:
                proba = model.predict_proba(X, num_iteration=int(best_iter))
            else:
                proba = model.predict_proba(X)
    except Exception:
        proba = None
    # Decision labels
    try:
        y_pred = model.predict(X)
    except Exception:
        # fallback using argmax on proba
        if proba is not None:
            y_pred = np.argmax(proba, axis=1)
        else:
            raise

    if proba is not None and classes == 2:
        if getattr(proba, "ndim", 1) == 2:
            proba = proba[:, 1]
    return None if proba is None else np.asarray(proba), np.asarray(y_pred)

