from __future__ import annotations

# Placeholder for centralized hyperparameter presets per library/tuning profile.
# Each model implementation may import from here to keep profiles consistent.

RF_PRESETS = {
    "light": dict(n_estimators=300, min_samples_split=4, min_samples_leaf=2, max_features="sqrt"),
    "moderate": dict(n_estimators=600, min_samples_split=4, min_samples_leaf=1, max_features="sqrt"),
    "heavy": dict(n_estimators=1200, min_samples_split=2, min_samples_leaf=1, max_features=None),
}

HGBT_PRESETS = {
    "light": dict(max_depth=6, learning_rate=0.08, max_iter=300),
    "moderate": dict(max_depth=8, learning_rate=0.05, max_iter=600),
    "heavy": dict(max_depth=None, learning_rate=0.03, max_iter=1000),
}

LOGREG_PRESETS = {
    "light": dict(penalty="l2", C=1.0, solver="lbfgs", class_weight="balanced", max_iter=200),
    "moderate": dict(penalty="l2", C=0.5, solver="lbfgs", class_weight="balanced", max_iter=300),
    "heavy": dict(penalty="elasticnet", l1_ratio=0.3, C=0.5, solver="saga", class_weight="balanced", max_iter=500),
}

ENET_PRESETS = {
    "light": dict(alpha=0.0005, l1_ratio=0.2, max_iter=2000),
    "moderate": dict(alpha=0.001, l1_ratio=0.3, max_iter=5000),
    "heavy": dict(alpha=0.0003, l1_ratio=0.5, max_iter=8000),
}

# Booster presets (centralized)
XGB_PRESETS = {
    "light": {
        "n_estimators": 300,
        "learning_rate": 0.08,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    },
    "moderate": {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 2,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "colsample_bynode": 0.8,
        "colsample_bylevel": 0.9,
        "tree_method": "hist",
        "early_stopping_rounds": 75,
    },
    "heavy": {
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "max_depth": 10,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "tree_method": "hist",
        "early_stopping_rounds": 100,
    },
    "custom_ranges": {
        "learning_rate": [0.01, 0.2],
        "n_estimators": [200, 2000],
        "max_depth": [3, 12],
        "min_child_weight": [1, 10],
        "subsample": [0.6, 1.0],
        "colsample_bytree": [0.6, 1.0],
        "gamma": [0, 5],
        "reg_lambda": [0, 10],
        "reg_alpha": [0, 5],
    },
}

LGBM_PRESETS = {
    "light": {
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 50,
        "n_estimators": 400,
        "learning_rate": 0.08,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "extra_trees": True,
        "early_stopping_rounds": 50,
    },
    "moderate": {
        "num_leaves": 127,
        "min_data_in_leaf": 40,
        "n_estimators": 800,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "early_stopping_rounds": 75,
    },
    "heavy": {
        "num_leaves": 255,
        "min_data_in_leaf": 20,
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "max_bin": 511,
        "early_stopping_rounds": 100,
    },
}

CAT_PRESETS = {
    "light": {
        "depth": 6,
        "learning_rate": 0.08,
        "l2_leaf_reg": 3.0,
        "iterations": 600,
        "bagging_temperature": 0.5,
        "leaf_estimation_iterations": 1,
        "early_stopping_rounds": 50,
    },
    "moderate": {
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "iterations": 1000,
        "bagging_temperature": 1.0,
        "random_strength": 1.0,
        "leaf_estimation_iterations": 2,
        "early_stopping_rounds": 75,
    },
    "heavy": {
        "depth": 10,
        "learning_rate": 0.03,
        "l2_leaf_reg": 2.0,
        "iterations": 1600,
        "bagging_temperature": 1.0,
        "random_strength": 1.0,
        "leaf_estimation_iterations": 3,
        "early_stopping_rounds": 100,
    },
}
