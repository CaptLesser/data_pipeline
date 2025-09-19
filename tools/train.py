from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

# Local imports (ensure repo root on path when invoked from tools/)
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.dataset import DatasetSpec, build_dense_dataset, compute_labels  # type: ignore
from ml.split import split_time, PurgedKFold  # type: ignore
from ml.models import REGISTRY  # type: ignore
from ml.artifacts import save_artifacts  # type: ignore
from ml.utils import seed_everything  # type: ignore
from ml.predict import predict_out  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ML models on metrics_series + OHLCVT (dense minute grid)")
    p.add_argument("--model", required=True, help="rf|hgbt|logreg|enet|xgb|lgbm|cat (others stubbed)")
    p.add_argument("--task", required=True, help="price|return|direction|vol|event")
    p.add_argument("--horizons", default="1,5,15,60", help="Comma list of horizons in minutes")
    p.add_argument("--features", default="min", help="min|max|custom (custom reserved)")
    p.add_argument("--lags", type=int, default=60, help="Number of lags for selected base columns")
    p.add_argument("--short-win", default="1h", help="Short window label (e.g., 1h)")
    p.add_argument("--wins", default="1h,4h,12h,24h", help="Windows to use from metrics_series")
    p.add_argument("--input-metrics", required=True, help="Path to *_metrics_series.csv")
    p.add_argument("--input-ohlcvt", required=True, help="Path to habitual_*.csv OHLCVT")
    p.add_argument("--dense", action="store_true", help="Dense minute grid (default)")
    p.add_argument("--sparse", action="store_true", help="Use only rows present in metrics_series (quick run)")
    p.add_argument("--cross-asset", choices=["ranks", "spreads", "both"], help="Enable cross-asset features (optional)")
    p.add_argument("--base", default="BTCUSDT", help="Base symbol for spreads")
    p.add_argument("--calendar", action="store_true", help="Add calendar features (minute/hour/dow)")
    p.add_argument("--calendar-extended", action="store_true", help="Add day/month")
    p.add_argument("--symbol-cat", action="store_true", help="Append symbol as categorical code feature (for cat/lgbm)")
    p.add_argument("--dir-bps", type=int, default=5, help="Direction thresholds in bps (±)")
    p.add_argument("--classes", type=int, default=3, help="2 for binary up vs non-up; 3 for up/flat/down")
    p.add_argument("--rv", default="std", choices=["std", "parkinson", "gk", "yz"], help="Realized vol estimator for vol task")
    p.add_argument("--tuning", default="moderate", choices=["light", "moderate", "heavy", "custom"], help="Tuning profile")
    p.add_argument("--gpu", action="store_true", help="Use GPU for supported boosters (xgb/lgbm/cat)")
    p.add_argument("--scale-pos-weight", default=None, help="Imbalance handling for binary tasks: auto or a float value")
    # Event labeling
    p.add_argument("--event", choices=["breakout", "squeeze", "spike"], help="Event type (task=event)")
    p.add_argument("--breakout-bps", type=float, default=10.0, help="Breakout threshold in bps (default 10)")
    p.add_argument("--squeeze-pct", type=float, default=10.0, help="Squeeze low bandwidth percentile on training (default 10)")
    p.add_argument("--squeeze-expand", type=float, default=0.5, help="Squeeze expansion factor within horizon (default 0.5)")
    p.add_argument("--spike-ret-pctl", type=float, default=99.5, help="Spike return percentile on training (default 99.5)")
    p.add_argument("--spike-vol-pctl", type=float, default=80.0, help="Spike volume-share percentile on training (default 80)")
    p.add_argument("--include-cols", help="Regex to include feature columns (applied after preset)")
    p.add_argument("--exclude-cols", help="Regex to exclude feature columns")
    # Cross-asset controls
    p.add_argument("--rank-top", type=int, help="Optional top-N by liquidity per timestamp for cross-asset ranks")
    p.add_argument("--liq-col", default="quote_volume_sum_24h_mag", help="Liquidity column for rank-top or threshold")
    p.add_argument("--liq-threshold", type=float, help="Percentile threshold for liquidity (0..100)")
    p.add_argument("--universe-fixed", action="store_true", help="Lock cross-asset universe to training set selection")
    p.add_argument("--base-missing", default="drop", choices=["drop", "ffill", "error"], help="Behavior when base symbol is missing at a timestamp for spreads")
    p.add_argument("--split", default=None, help="Ratios '0.7/0.15/0.15' or time cuts 'YYYY-mm-dd,YYYY-mm-dd' (train_end,val_end)")
    p.add_argument("--embargo", type=int, default=60, help="Embargo minutes between splits")
    p.add_argument("--cv", type=int, help="Use PurgedKFold cross-validation with given number of folds")
    p.add_argument("--cv-refit", action="store_true", help="Refit final model with n_estimators capped to mean CV best-iteration")
    p.add_argument("--select-by", default=None, choices=["ev", "f1", "mcc"], help="Threshold selection criterion (direction: f1; event: mcc; ev optional)")
    p.add_argument("--fee-bps", type=float, default=0.0, help="Fee bps for EV thresholding (if used)")
    p.add_argument("--slip-bps", type=float, default=0.0, help="Slippage bps for EV thresholding (if used)")
    p.add_argument("--calibrate", choices=["platt", "isotonic"], help="Probability calibration for binary tasks")
    p.add_argument("--perm-imp", type=int, help="Permutation importance repeats (compute on validation if set)")
    p.add_argument("--perm-imp-sample", type=int, help="Sample size for permutation importance to bound runtime")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--dry-run", action="store_true", help="Build dataset, save summaries, and exit before training")
    p.add_argument("--save-env", action="store_true", help="Save pip freeze to environment.txt in the run dir")
    p.add_argument("--outdir", default=None, help="Output directory; default models/{task}/{model}/{run_id}")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    windows = [w.strip() for w in args.wins.split(",") if w.strip()]

    spec = DatasetSpec(
        horizons=horizons,
        short_win=args.short_win,
        dense=not args.sparse,
        windows=windows,
        require_wins=None,  # default: 1h validity enforced internally
        embargo_minutes=int(args.embargo),
        cross_asset=args.cross_asset,
        base_symbol=args.base,
        calendar=bool(args.calendar),
        calendar_extended=bool(args.calendar_extended),
        rank_top=args.rank_top,
        liq_col=args.liq_col,
        liq_threshold=args.liq_threshold,
    )

    # Build features table
    # Optionally build once without cross-asset to compute fixed universe from training
    cross_asset_mode = args.cross_asset
    symbols_filter = None
    build_cross = cross_asset_mode is not None
    if build_cross and getattr(args, 'universe_fixed', False):
        spec_no_cross = DatasetSpec(**{**spec.__dict__, 'cross_asset': None})
        df0, _ = build_dense_dataset(
            args.input_ohlcvt,
            args.input_metrics,
            spec_no_cross,
            lags=int(args.lags),
            features_preset=args.features,
            short_win=args.short_win,
            windows=windows,
            include_cols_regex=args.include_cols,
            exclude_cols_regex=args.exclude_cols,
        )
        # Split to compute training universe
        train0, val0, test0 = split_time(df0, train_ratio=0.7, val_ratio=0.15, embargo_minutes=int(args.embargo))
        # Fixed symbols by median liquidity column
        liq_col = args.liq_col
        if liq_col in train0.columns:
            med = train0.groupby('symbol')[liq_col].median().sort_values(ascending=False)
            if args.rank_top:
                symbols_filter = med.head(int(args.rank_top)).index.astype(str).tolist()
            elif args.liq_threshold is not None:
                thr = np.nanpercentile(med.to_numpy(dtype=float), float(args.liq_threshold))
                symbols_filter = med[med >= thr].index.astype(str).tolist()
        # fall back: all symbols in training
        if not symbols_filter:
            symbols_filter = sorted(train0['symbol'].astype(str).unique().tolist())

    df, info = build_dense_dataset(
        args.input_ohlcvt,
        args.input_metrics,
        spec,
        lags=int(args.lags),
        features_preset=args.features,
        short_win=args.short_win,
        windows=windows,
        include_cols_regex=args.include_cols,
        exclude_cols_regex=args.exclude_cols,
        symbols_filter=symbols_filter,
    )

    # Labels per horizon
    # Quick dry-run: save row accounting and exit
    if bool(args.dry_run):
        outdir = args.outdir or os.path.join(ROOT, "models", args.task, args.model, datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(outdir, exist_ok=True)
        # Persist universe mask if present
        try:
            uni = getattr(df, "_universe_mask_df", None)
            if uni is not None and not uni.empty:
                uni.to_csv(os.path.join(outdir, "universe_mask.csv"), index=False)
        except Exception:
            pass
        try:
            ra = getattr(df, "_row_accounting", None)
            if isinstance(ra, dict):
                with open(os.path.join(outdir, "row_accounting.json"), "w", encoding="utf-8") as f:
                    json.dump(ra, f, indent=2)
        except Exception:
            pass
        print(f"Dry run complete. Outputs in {outdir}")
        return

    # Labels per horizon (event labels need training-percentiles)
    # Sanity: required short window hygiene present
    ok_col = f"window_ok_{args.short_win}"
    if ok_col not in df.columns:
        raise SystemExit(f"Missing required hygiene column '{ok_col}'. Hint: rerun metrics with --wins including {args.short_win}.")
    if args.task == "event":
        if not args.event:
            raise SystemExit("--event breakout|squeeze|spike required when --task event")
        from ml.events import EventSpec, compute_event_thresholds, compute_event_labels  # type: ignore
        # Validate required columns
        missing = []
        if "high" not in df.columns:
            missing.append("high (OHLCVT)")
        bw_col = f"bollinger_width_{args.short_win}"
        vs_col = f"vol_spike_share_{args.short_win}_pos"
        if args.event == "squeeze" and bw_col not in df.columns:
            missing.append(bw_col)
        if args.event == "spike" and vs_col not in df.columns:
            missing.append(vs_col)
        if missing:
            hint = "rerun metrics with --wins including your --short-win"
            raise SystemExit(f"Missing required columns for event '{args.event}': {missing}. Hint: {hint}")
        # Split first for percentile computation from training only
        train_df, val_df, test_df = split_time(df, train_ratio=0.7, val_ratio=0.15, embargo_minutes=int(args.embargo))
        es = EventSpec(
            event=args.event,
            short_win=args.short_win,
            breakout_bps=float(args.breakout_bps),
            squeeze_pct=float(args.squeeze_pct),
            squeeze_expand=float(args.squeeze_expand),
            spike_ret_pctl=float(args.spike_ret_pctl),
            spike_vol_pctl=float(args.spike_vol_pctl),
        )
        thresholds = compute_event_thresholds(train_df, es)
        labels = compute_event_labels(df, horizons, es, thresholds)
        event_thresholds = thresholds
    else:
        labels = compute_labels(df, horizons, task=args.task, dir_bps=int(args.dir_bps), classes=int(args.classes))

    # Split
    # Use ratio split by default with embargo
    if args.task != "event":
        train_df, val_df, test_df = split_time(df, train_ratio=0.7, val_ratio=0.15, embargo_minutes=int(args.embargo))

    # Build outputs dir
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join(ROOT, "models", args.task, args.model, run_id)
    os.makedirs(outdir, exist_ok=True)

    # Feature matrix
    feat_cols: List[str] = info.get("features", [])
    # Optionally append symbol categorical codes (skip for CatBoost which will use native string cats)
    if bool(args.symbol_cat) and args.model != "cat":
        df = df.copy()
        if args.model == "lgbm":
            df["symbol_code"] = df["symbol"].astype("category")
        else:
            df["symbol_code"] = df["symbol"].astype("category").cat.codes
        feat_cols = feat_cols + ["symbol_code"]
    # Build matrices
    if args.model == "cat":
        feat_cols_cat = [c for c in feat_cols if c != "symbol_code"]
        if "symbol" not in feat_cols_cat:
            feat_cols_cat = ["symbol"] + feat_cols_cat
        X_train = train_df[feat_cols_cat]
        X_val = val_df[feat_cols_cat]
        X_test = test_df[feat_cols_cat]
        feat_cols = feat_cols_cat
    else:
        X_train = train_df[feat_cols].to_numpy(dtype=float)
        X_val = val_df[feat_cols].to_numpy(dtype=float)
        X_test = test_df[feat_cols].to_numpy(dtype=float)

    summary_metrics = {}
    # Optional CV (PurgedKFold) metrics per horizon
    if args.cv and int(args.cv) >= 3:
        cvdir = os.path.join(outdir, "cv")
        os.makedirs(cvdir, exist_ok=True)
        cv_summary = {}
        folds_meta = {}
        cv_best_iters = {}
        for H in horizons:
            y_all = labels.get(H)
            if y_all is None:
                continue
            y_all_np = y_all.to_numpy()
            pkf = PurgedKFold(n_splits=int(args.cv), embargo_minutes=int(args.embargo))
            fold_ms = []
            fold_bounds = []
            best_list = []
            for fold_idx, (tr_idx, va_idx) in enumerate(pkf.split(df)):
                X_tr_cv = df.iloc[tr_idx][feat_cols].to_numpy(dtype=float)
                y_tr_cv = y_all_np[tr_idx]
                X_va_cv = df.iloc[va_idx][feat_cols].to_numpy(dtype=float)
                y_va_cv = y_all_np[va_idx]
                # Skip invalid
                mtr = np.isfinite(y_tr_cv)
                mva = np.isfinite(y_va_cv)
                if not mtr.any() or not mva.any():
                    continue
                mod = REGISTRY.get(args.model)
                if mod is None:
                    continue
                model_cv = mod.build_model(args.task, tuning=args.tuning)
                m_cv = mod.fit_and_score(model_cv, X_tr_cv[mtr], y_tr_cv[mtr], X_va_cv[mva], y_va_cv[mva], task=args.task)
                fold_ms.append(m_cv)
                # capture best iteration if present
                bi = None
                for attr in ("best_iteration", "best_iteration_", "get_best_iteration"):
                    try:
                        if hasattr(model_cv, attr):
                            val = getattr(model_cv, attr)
                            bi = int(val() if callable(val) else val)
                            break
                    except Exception:
                        pass
                if bi is not None:
                    best_list.append(bi)
                tmin = str(pd.to_datetime(df.iloc[va_idx]["timestamp"].min()))
                tmax = str(pd.to_datetime(df.iloc[va_idx]["timestamp"].max()))
                fold_bounds.append({"fold": fold_idx, "val_min": tmin, "val_max": tmax})
            # aggregate
            agg = {}
            for k in set().union(*fold_ms) if fold_ms else []:
                vals = [fm[k] for fm in fold_ms if k in fm]
                if vals:
                    agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            if best_list:
                agg["best_iteration"] = {"mean": float(np.mean(best_list)), "std": float(np.std(best_list))}
                cv_best_iters[H] = int(np.ceil(np.mean(best_list)))
            cv_summary[str(H)] = agg
            folds_meta[str(H)] = fold_bounds
        with open(os.path.join(cvdir, "cv_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(cv_summary, f, indent=2)
        with open(os.path.join(cvdir, "folds.json"), "w", encoding="utf-8") as f:
            json.dump(folds_meta, f, indent=2)
    else:
        cv_best_iters = {}

    # Train per horizon
    for H in horizons:
        y_all = labels.get(H)
        if y_all is None:
            continue
        # Align with splits
        y_tr = y_all.loc[train_df.index].to_numpy()
        y_va = y_all.loc[val_df.index].to_numpy()
        y_te = y_all.loc[test_df.index].to_numpy()

        # Drop rows with NaN labels (missing future bars)
        m_tr = np.isfinite(y_tr)
        m_va = np.isfinite(y_va)
        m_te = np.isfinite(y_te)
        X_tr, y_tr2 = X_train[m_tr], y_tr[m_tr]
        X_va, y_va2 = X_val[m_va], y_va[m_va]
        X_te, y_te2 = X_test[m_te], y_te[m_te]
        if X_tr.size == 0 or X_va.size == 0 or X_te.size == 0:
            print(f"H={H}: insufficient data after label alignment; skipping")
            continue

        # Build, fit and score via registry
        mod = REGISTRY.get(args.model)
        if mod is None:
            print(f"Model '{args.model}' not available; exiting.")
            raise SystemExit(0)
        # Booster overrides
        overrides = {}
        # GPU toggle
        if args.gpu and args.model in ("xgb", "lgbm", "cat"):
            if args.model == "xgb":
                overrides["tree_method"] = "gpu_hist"
            elif args.model == "lgbm":
                overrides["device_type"] = "gpu"
            elif args.model == "cat":
                overrides["task_type"] = "GPU"
        # Objective/metric by task/classes
        task_key = args.task
        if args.task == "direction":
            task_key = "direction2" if int(args.classes) == 2 else "direction3"
        if args.model == "xgb":
            if task_key in ("price", "return", "vol"):
                overrides.setdefault("objective", "reg:squarederror")
                overrides.setdefault("eval_metric", "rmse")
            elif task_key == "direction2":
                overrides.setdefault("objective", "binary:logistic")
                overrides.setdefault("eval_metric", "aucpr")
            elif task_key == "direction3":
                overrides.setdefault("objective", "multi:softprob")
                overrides.setdefault("eval_metric", "mlogloss")
                overrides.setdefault("num_class", 3)
            elif task_key == "event":
                overrides.setdefault("objective", "binary:logistic")
                overrides.setdefault("eval_metric", "aucpr")
        elif args.model == "lgbm":
            if task_key in ("price", "return", "vol"):
                overrides.setdefault("objective", "regression")
                overrides.setdefault("metric", "rmse")
            elif task_key == "direction2":
                overrides.setdefault("objective", "binary")
                overrides.setdefault("metric", "aucpr")
            elif task_key == "direction3":
                overrides.setdefault("objective", "multiclass")
                overrides.setdefault("metric", "multi_logloss")
                overrides.setdefault("num_class", 3)
            elif task_key == "event":
                overrides.setdefault("objective", "binary")
                overrides.setdefault("metric", "aucpr")
        elif args.model == "cat":
            if task_key in ("price", "return", "vol"):
                overrides.setdefault("loss_function", "RMSE")
            elif task_key == "direction2":
                overrides.setdefault("loss_function", "Logloss")
            elif task_key == "direction3":
                overrides.setdefault("loss_function", "MultiClass")
            elif task_key == "event":
                overrides.setdefault("loss_function", "Logloss")

        # Imbalance handling
        if task_key in ("direction2", "event") and args.scale_pos_weight is not None and args.model in ("xgb", "lgbm", "cat"):
            spw = None
            if str(args.scale_pos_weight).lower() == "auto":
                pos = float((y_tr2 == 1).sum())
                neg = float((y_tr2 == 0).sum())
                spw = float(neg / max(pos, 1.0)) if pos >= 0 else 1.0
            else:
                try:
                    spw = float(args.scale_pos_weight)
                except Exception:
                    spw = None
            if spw is not None:
                if args.model in ("xgb", "lgbm"):
                    overrides["scale_pos_weight"] = spw
                elif args.model == "cat":
                    overrides["class_weights"] = [1.0, spw]

        if args.model == "logreg":
            overrides["classes"] = int(args.classes)
        # pass categorical_feature indices for LGBM/Cat when symbol-cat enabled
        # Categorical feature indices for LGBM/Cat
        if args.model == "lgbm" and bool(args.symbol_cat) and "symbol_code" in feat_cols:
            overrides["categorical_feature"] = [feat_cols.index("symbol_code")]
        if args.model == "cat" and "symbol" in feat_cols:
            overrides["cat_features"] = [feat_cols.index("symbol")]
        # CV refit: cap boosters to mean best_iter and disable early-stopping by not providing eval_set in builders
        if (args.cv and int(args.cv) >= 3) and getattr(args, 'cv_refit', False) and H in cv_best_iters and args.model in ("xgb", "lgbm", "cat"):
            overrides["n_estimators"] = int(cv_best_iters[H])
        model = mod.build_model(args.task, tuning=args.tuning, **overrides)
        # Save resolved hyperparams fingerprint
        try:
            params = model.get_params(deep=False)
            import hashlib
            hp_hash = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()
        except Exception:
            params = {}
            hp_hash = None
        m = mod.fit_and_score(model, X_tr, y_tr2, X_va, y_va2, task=args.task)

        # Evaluate on test
        if args.task in ("price", "return", "vol"):
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            proba_te, y_pred = predict_out(model, X_te, task=args.task, classes=int(args.classes), best_iter=combined_metrics.get("best_iteration"))
            test_metrics = {
                "mae": float(mean_absolute_error(y_te2, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_te2, y_pred))),
                "r2": float(r2_score(y_te2, y_pred)),
            }
            # Save test predictions
            test_preds_path = os.path.join(outdir, f"H={H}", "predictions.parquet")
            pd.DataFrame({
                "timestamp": test_df["timestamp"],
                "symbol": test_df["symbol"],
                "yhat": y_pred,
            }).to_parquet(test_preds_path, index=False)
        else:
            from sklearn.metrics import f1_score, matthews_corrcoef, brier_score_loss
            y_prob, y_pred = predict_out(model, X_te, task=args.task, classes=int(args.classes), best_iter=combined_metrics.get("best_iteration"))
            test_metrics = {
                "f1_macro" if len(np.unique(y_te2)) > 2 else "f1": float(f1_score(y_te2, y_pred, average="macro" if len(np.unique(y_te2)) > 2 else "binary")),
                "mcc": float(matthews_corrcoef(y_te2, y_pred)),
            }
            if y_prob is not None and (len(np.unique(y_te2)) == 2):
                try:
                    p = y_prob
                    test_metrics["brier"] = float(brier_score_loss(y_te2, p))
                except Exception:
                    pass
            # Save test predictions
            test_preds_path = os.path.join(outdir, f"H={H}", "predictions.parquet")
            pd.DataFrame({
                "timestamp": test_df["timestamp"],
                "symbol": test_df["symbol"],
                "proba": y_prob if y_prob is not None else np.nan,
                "label": y_pred,
            }).to_parquet(test_preds_path, index=False)
            # Select decision threshold on validation (direction: F1, event: MCC) for binary (or EV if selected)
            decision_threshold = None
            if y_prob is not None and len(np.unique(y_va2)) == 2:
                try:
                    pva, _ = predict_out(model, X_va, task=args.task, classes=int(args.classes), best_iter=combined_metrics.get("best_iteration"))
                    best_score = -1.0
                    best_tau = 0.5
                    sel = (args.select_by or ("mcc" if args.task == "event" else "f1")).lower()
                    if sel == "ev":
                        fee = float(args.fee_bps) / 1e4
                        slip = float(args.slip_bps) / 1e4
                        cst = 2.0 * (fee + slip)
                        taus = np.arange(0.05, 0.951, 0.005)
                        if args.task == "direction":
                            eps = 1e-12
                            fut = df.groupby("symbol")["close"].shift(-H)
                            r_all = np.log((fut + eps) / (df["close"] + eps))
                            r_va = r_all.loc[val_df.index].to_numpy()
                            mask = np.isfinite(r_va) & np.isfinite(pva)
                            p = pva[mask]
                            rv = r_va[mask]
                            for tau in taus:
                                take = (p >= tau)
                                if not take.any():
                                    continue
                                ev = (p[take] * rv[take] - cst).mean()
                                if ev > best_score:
                                    best_score, best_tau = ev, float(tau)
                            test_metrics["threshold_val_ev"] = float(best_score)
                        else:
                            payoff = float(getattr(args, "ev_event_payoff", 1.0))
                            penalty = float(getattr(args, "ev_event_penalty", 1.0))
                            for tau in taus:
                                take = (pva >= tau)
                                if not take.any():
                                    continue
                                ev = (pva[take] * payoff - (1.0 - pva[take]) * penalty).mean()
                                if ev > best_score:
                                    best_score, best_tau = ev, float(tau)
                            test_metrics["threshold_val_ev"] = float(best_score)
                    else:
                        for tau in np.linspace(0.05, 0.95, 19):
                            pred = (pva >= tau).astype(int)
                            if args.task == "event" or sel == "mcc":
                                from sklearn.metrics import matthews_corrcoef
                                score = matthews_corrcoef(y_va2, pred)
                                key = "threshold_val_mcc"
                            else:
                                from sklearn.metrics import f1_score
                                score = f1_score(y_va2, pred)
                                key = "threshold_val_f1"
                            if score > best_score:
                                best_score, best_tau = score, float(tau)
                        test_metrics[key] = float(best_score)
                except Exception:
                    best_tau = None
        # Save artifacts under subdir per horizon
        subdir = os.path.join(outdir, f"H={H}")
        os.makedirs(subdir, exist_ok=True)
        # Merge train metrics + test metrics
        # Attach best_iteration if present
        best_iter = None
        for attr in ("best_iteration", "best_iteration_", "get_best_iteration"):
            try:
                if hasattr(model, attr):
                    val = getattr(model, attr)
                    best_iter = int(val() if callable(val) else val)
                    break
            except Exception:
                pass
        combined_metrics = {"val": m, "test": test_metrics}
        if best_iter is not None:
            combined_metrics["best_iteration"] = best_iter
        # Save artifacts (with column map)
        # Build run/column map metadata
        column_map = {
            "symbols": sorted(df["symbol"].dropna().astype(str).unique().tolist()),
            "windows": windows,
            "features_preset": args.features,
            "include_cols": args.include_cols,
            "exclude_cols": args.exclude_cols,
            "cross_asset": args.cross_asset,
            "rank_top": args.rank_top,
            "liq_col": args.liq_col,
            "liq_threshold": args.liq_threshold,
            "calendar": bool(args.calendar),
            "hyperparams_hash": hp_hash,
        }
        # Persist params.json
        try:
            with open(os.path.join(subdir, "params.json"), "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, default=str)
        except Exception:
            pass
        # include event thresholds when applicable
        if args.task == "event":
            try:
                extra = {
                    "event": args.event,
                    "breakout_bps": float(args.breakout_bps),
                    "squeeze_pct": float(args.squeeze_pct),
                    "squeeze_expand": float(args.squeeze_expand),
                    "spike_ret_pctl": float(args.spike_ret_pctl),
                    "spike_vol_pctl": float(args.spike_vol_pctl),
                }
            except Exception:
                extra = {}
            column_map.update(extra)
        # Add label spec descriptor
        if args.task == "event":
            from ml.events import LabelSpec  # type: ignore
            label_spec = LabelSpec(
                kind=str(args.event),
                horizons=tuple(horizons),
                wins=tuple(windows),
                params={
                    "breakout_bps": float(args.breakout_bps),
                    "squeeze_pct": float(args.squeeze_pct),
                    "squeeze_expand": float(args.squeeze_expand),
                    "spike_ret_pctl": float(args.spike_ret_pctl),
                    "spike_vol_pctl": float(args.spike_vol_pctl),
                },
                prior_window=str(args.short_win),
                embargo_min=int(args.embargo),
                train_cutoff=None,
            )
            column_map["label_spec"] = {
                "kind": label_spec.kind,
                "horizons": list(label_spec.horizons),
                "wins": list(label_spec.wins),
                "params": label_spec.params,
                "prior_window": label_spec.prior_window,
                "embargo_min": label_spec.embargo_min,
            }
        # Permutation importance (optional)
        if args.perm_imp and args.perm_imp > 0:
            try:
                from sklearn.inspection import permutation_importance
                Xv = X_va
                yv = y_va2
                if args.perm_imp_sample and args.perm_imp_sample > 0 and len(yv) > args.perm_imp_sample:
                    idx = np.random.RandomState(int(args.seed)).choice(len(yv), size=int(args.perm_imp_sample), replace=False)
                    Xv = Xv[idx]
                    yv = yv[idx]
                r = permutation_importance(model, Xv, yv, n_repeats=int(args.perm_imp), random_state=int(args.seed))
                pi = {feat_cols[i]: float(r.importances_mean[i]) for i in range(len(feat_cols))}
                with open(os.path.join(subdir, "permutation_importance.json"), "w", encoding="utf-8") as f:
                    json.dump(pi, f, indent=2)
            except Exception:
                pass
        # Save artifacts
        if 'best_tau' in locals() and best_tau is not None:
            column_map["decision_threshold"] = float(best_tau)
        # Persist event numeric thresholds if available
        if args.task == "event":
            try:
                column_map["label_thresholds"] = event_thresholds
            except Exception:
                pass
        # Save calibrator and per-H threshold when calibration used
        if 'calibrated' in locals() and calibrated and calibrator is not None:
            try:
                import pickle as _pkl
                with open(os.path.join(subdir, "calibrator.pkl"), "wb") as f:
                    _pkl.dump(calibrator, f)
            except Exception:
                pass
        if 'best_tau' in locals() and best_tau is not None:
            try:
                with open(os.path.join(subdir, "threshold.json"), "w", encoding="utf-8") as f:
                    json.dump({"threshold": float(best_tau)}, f, indent=2)
            except Exception:
                pass
        # Universe spec
        column_map["universe_spec"] = {
            "mode": "fixed" if bool(args.universe_fixed) else "dynamic",
            "rank_top": args.rank_top,
            "liq_pctl": args.liq_threshold,
            "base_symbol": args.base,
            "liq_col": args.liq_col,
        }
        save_artifacts(subdir, model, combined_metrics, feat_cols, column_map)
        summary_metrics[str(H)] = combined_metrics

    # Persist universe mask if present
    try:
        uni = getattr(df, "_universe_mask_df", None)
        if uni is not None and not uni.empty:
            uni_out = os.path.join(outdir, "universe_mask.csv")
            uni.to_csv(uni_out, index=False)
    except Exception:
        pass
    # Save row accounting
    try:
        ra = getattr(df, "_row_accounting", None)
        if isinstance(ra, dict):
            with open(os.path.join(outdir, "row_accounting.json"), "w", encoding="utf-8") as f:
                json.dump(ra, f, indent=2)
    except Exception:
        pass

    # Save run config and summary
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        # Data span
        try:
            data_span = {
                "train_min": str(pd.to_datetime(train_df["timestamp"].min())),
                "train_max": str(pd.to_datetime(train_df["timestamp"].max())),
                "val_min": str(pd.to_datetime(val_df["timestamp"].min())),
                "val_max": str(pd.to_datetime(val_df["timestamp"].max())),
                "test_min": str(pd.to_datetime(test_df["timestamp"].min())),
                "test_max": str(pd.to_datetime(test_df["timestamp"].max())),
            }
        except Exception:
            data_span = {}
        cfg = {
            "model": args.model,
            "task": args.task,
            "horizons": horizons,
            "features": args.features,
            "lags": int(args.lags),
            "short_win": args.short_win,
            "windows": windows,
            "dense": bool(args.dense or not args.sparse),
            "cross_asset": args.cross_asset,
            "calendar": bool(args.calendar),
            "dir_bps": int(args.dir_bps),
            "classes": int(args.classes),
            "embargo": int(args.embargo),
            "input_metrics": args.input_metrics,
            "input_ohlcvt": args.input_ohlcvt,
            "data_span": data_span,
        }
        # hash of input spec
        try:
            import hashlib
            h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
            cfg["spec_hash"] = h
        except Exception:
            pass
        json.dump(cfg, f, indent=2)
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    # Save run.json with env info
    run_info = {"cmd": " ".join(sys.argv), "python": sys.version}
    try:
        import numpy, pandas, sklearn
        run_info.update({
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "sklearn": sklearn.__version__,
        })
    except Exception:
        pass
    # Optional libraries
    try:
        import xgboost as _xgb
        run_info["xgboost"] = _xgb.__version__
    except Exception:
        pass
    try:
        import lightgbm as _lgb
        run_info["lightgbm"] = _lgb.__version__
    except Exception:
        pass
    try:
        import catboost as _cat
        run_info["catboost"] = _cat.__version__
    except Exception:
        pass
    try:
        import platform
        run_info["os"] = f"{platform.system()} {platform.release()}"
    except Exception:
        pass
    with open(os.path.join(outdir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    # Optional pip freeze
    if getattr(args, "save_env", False):
        try:
            import subprocess
            env_txt = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]) .decode()
            with open(os.path.join(outdir, "environment.txt"), "w", encoding="utf-8") as f:
                f.write(env_txt)
        except Exception:
            pass

    print(f"Run complete. Artifacts in {outdir}")


if __name__ == "__main__":
    main()
