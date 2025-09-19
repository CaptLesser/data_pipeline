from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    res: Dict[str, float] = {}
    # F1
    try:
        if len(np.unique(y_true)) > 2:
            res["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        else:
            res["f1"] = float(f1_score(y_true, y_pred))
    except Exception:
        pass
    # MCC
    try:
        res["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        pass
    # AUC / PR-AUC
    try:
        if y_prob is not None:
            if y_prob.ndim == 1:
                res["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                res["pr_auc"] = float(average_precision_score(y_true, y_prob))
            else:
                n_classes = y_prob.shape[1]
                aucs = []
                prs = []
                for c in range(n_classes):
                    y_bin = (y_true == c).astype(int)
                    aucs.append(roc_auc_score(y_bin, y_prob[:, c]))
                    prs.append(average_precision_score(y_bin, y_prob[:, c]))
                res["roc_auc_macro"] = float(np.mean(aucs))
                res["pr_auc_macro"] = float(np.mean(prs))
    except Exception:
        pass
    return res

