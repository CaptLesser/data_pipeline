from __future__ import annotations

from . import rf, hgbt, logreg, enet
from . import xgb, lgbm, cat

REGISTRY = {
    "rf": rf,
    "hgbt": hgbt,
    "logreg": logreg,
    "enet": enet,
    "xgb": xgb,
    "lgbm": lgbm,
    "cat": cat,
}
