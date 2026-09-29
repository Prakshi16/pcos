"""Step 6: Train Random Forest across all tiers and imbalance strategies.

Reads: data/processed/train.csv.
Writes: results/06_random_forest/, models/{tier}/random_forest__*.joblib.

RF skips scaling and feature selection (Section 4a step 3-4 apply to NB/LR only).
Reuses the 04/05 nested-CV / calibration pattern.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV, cross_val_predict,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import clone

from src import config
from src.utils.io import save_csv_and_md, save_json, save_model
from src.utils.metrics import compute_metrics, select_recall_threshold
from src.utils.pipelines import split_feature_types, build_pipeline
from src.utils.plotting import new_fig, save_fig

MODEL_NAME = "random_forest"
OUT = config.RESULTS_DIR / "06_random_forest"
OUT.mkdir(parents=True, exist_ok=True)
STRATEGIES = config.MODEL_STRATEGIES[MODEL_NAME]

warnings.filterwarnings("ignore", category=UserWarning)


def make_estimator(strategy):
    class_weight = "balanced" if strategy == "class_weight" else None
    return RandomForestClassifier(random_state=config.SEED, class_weight=class_weight)


def make_param_grid():
    return dict(config.RF_PARAM_GRID)


def run_nested_cv(X, y, numeric_feats, binary_feats, strategy):
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=config.SEED)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    fold_metrics = []
    for train_idx, test_idx in outer.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipe = build_pipeline(numeric_feats, binary_feats, make_estimator(strategy), strategy,
                               scale=False, select_k=False)
        grid = make_param_grid()
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1)
        gs.fit(X_tr, y_tr)

        proba = gs.predict_proba(X_te)[:, 1]
        fold_metrics.append(compute_metrics(y_te, proba, threshold=0.5))

    df = pd.DataFrame(fold_metrics)
    summary = {f"{c}_mean": df[c].mean() for c in df.columns}
    summary.update({f"{c}_std": df[c].std() for c in df.columns})
    return summary


def fit_final_and_calibrate(X, y, numeric_feats, binary_feats, strategy, tier):
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    pipe = build_pipeline(numeric_feats, binary_feats, make_estimator(strategy), strategy,
                           scale=False, select_k=False)
    grid = make_param_grid()
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1)
    gs.fit(X, y)
    best_estimator = gs.best_estimator_
    best_params = gs.best_params_

    oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    results = {}
    for calib in config.CALIBRATION_METHODS:
        if calib == "raw":
            model = clone(best_estimator)
            model.set_params(**best_params)
        else:
            base = clone(best_estimator)
            base.set_params(**best_params)
            model = CalibratedClassifierCV(estimator=base, method=calib, cv=5)

        oof_proba = cross_val_predict(model, X, y, cv=oof_cv, method="predict_proba", n_jobs=-1)[:, 1]
        oof_metrics = compute_metrics(y, oof_proba, threshold=0.5)
        threshold, achieved = select_recall_threshold(y, oof_proba, config.RECALL_TARGET)
        if not achieved:
            print(f"WARNING: {tier}/{MODEL_NAME}/{strategy}/{calib} did not reach recall "
                  f"{config.RECALL_TARGET}; using max-recall threshold {threshold:.3f}.")

        model.fit(X, y)

        results[calib] = {
            "model": model,
            "features": numeric_feats + binary_feats,
            "threshold": threshold,
            "best_params": best_params,
            "oof_brier": oof_metrics["brier"],
            "oof_auc": oof_metrics["roc_auc"],
            "oof_proba": oof_proba,
        }
    return results


def main():
    train_df = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    y_full = train_df["pcos"].values

    nested_rows = []
    calib_rows = []
    best_params_all = {}

    for tier in config.TIERS:
        features = config.TIER_FEATURES[tier]
        numeric_feats, binary_feats = split_feature_types(features)
        X = train_df[features]
        y = y_full

        best_params_all[tier] = {}

        for strategy in STRATEGIES:
            print(f"[{MODEL_NAME}] {tier} / {strategy}: running nested CV...")
            summary = run_nested_cv(X, y, numeric_feats, binary_feats, strategy)
            nested_rows.append({"tier": tier, "strategy": strategy, **summary})

            calib_results = fit_final_and_calibrate(X, y, numeric_feats, binary_feats, strategy, tier)
            best_params_all[tier][strategy] = calib_results["raw"]["best_params"]

            for calib, res in calib_results.items():
                calib_rows.append({
                    "tier": tier, "strategy": strategy, "calibration": calib,
                    "oof_brier": res["oof_brier"], "oof_auc": res["oof_auc"],
                    "threshold": res["threshold"],
                })
                save_model({
                    "model": res["model"], "features": res["features"],
                    "threshold": res["threshold"], "best_params": res["best_params"],
                    "oof_brier": res["oof_brier"], "oof_auc": res["oof_auc"],
                }, config.MODELS_DIR / tier / f"{MODEL_NAME}__{strategy}__{calib}.joblib")

        tier_calib = [r for r in calib_rows if r["tier"] == tier]
        best_strategy = min(tier_calib, key=lambda r: r["oof_brier"])["strategy"]
        fig, ax = new_fig((6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        for strategy in STRATEGIES:
            if strategy != best_strategy:
                continue
            calib_results = fit_final_and_calibrate(X, y, numeric_feats, binary_feats, strategy, tier)
            for calib, res in calib_results.items():
                frac_pos, mean_pred = calibration_curve(y, res["oof_proba"], n_bins=10, strategy="quantile")
                ax.plot(mean_pred, frac_pos, marker="o", label=calib)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"{MODEL_NAME} reliability curve ({tier}, strategy={best_strategy})")
        ax.legend()
        save_fig(fig, OUT / f"oof_calibration_curve_{tier}.png")

    nested_df = pd.DataFrame(nested_rows)
    save_csv_and_md(nested_df, OUT / "nested_cv_results")

    calib_df = pd.DataFrame(calib_rows)
    save_csv_and_md(calib_df, OUT / "oof_calibration_results")

    save_json(best_params_all, OUT / "best_params.json")

    print(f"Step 6 ({MODEL_NAME}) complete. Outputs in {OUT}.")


if __name__ == "__main__":
    main()
