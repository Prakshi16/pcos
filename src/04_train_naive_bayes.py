"""Step 4: Train Gaussian Naive Bayes across all tiers and imbalance strategies.

Reads: data/processed/train.csv.
Writes: results/04_naive_bayes/ (nested CV results, OOF calibration results,
        best params, calibration curves), models/{tier}/naive_bayes__*.joblib.

Evaluation protocol: nested CV (outer RepeatedStratifiedKFold for estimation,
inner GridSearchCV/StratifiedKFold for tuning). Final fit + 3 calibration
versions (raw/sigmoid/isotonic) on the full train set, evaluated via OOF
cross_val_predict. No test data is touched here.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
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

MODEL_NAME = "naive_bayes"
OUT = config.RESULTS_DIR / "04_naive_bayes"
OUT.mkdir(parents=True, exist_ok=True)
STRATEGIES = config.MODEL_STRATEGIES[MODEL_NAME]

warnings.filterwarnings("ignore", category=UserWarning)


def make_estimator():
    return GaussianNB()


def make_param_grid(max_k):
    k_opts = [k for k in config.NB_K_OPTIONS if k == "all" or k <= max_k]
    grid = dict(config.NB_PARAM_GRID_BASE)
    grid["select__k"] = k_opts
    return grid


def run_nested_cv(X, y, numeric_feats, binary_feats, strategy):
    """Outer RepeatedStratifiedKFold estimation with inner GridSearchCV tuning."""
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=config.SEED)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    max_k = len(numeric_feats) + len(binary_feats)

    fold_metrics = []
    for train_idx, test_idx in outer.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipe = build_pipeline(numeric_feats, binary_feats, make_estimator(), strategy,
                               scale=True, select_k=True, max_k=max_k)
        grid = make_param_grid(max_k)
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1)
        gs.fit(X_tr, y_tr)

        proba = gs.predict_proba(X_te)[:, 1]
        fold_metrics.append(compute_metrics(y_te, proba, threshold=0.5))

    df = pd.DataFrame(fold_metrics)
    summary = {f"{c}_mean": df[c].mean() for c in df.columns}
    summary.update({f"{c}_std": df[c].std() for c in df.columns})
    return summary


def fit_final_and_calibrate(X, y, numeric_feats, binary_feats, strategy, tier):
    max_k = len(numeric_feats) + len(binary_feats)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    pipe = build_pipeline(numeric_feats, binary_feats, make_estimator(), strategy,
                           scale=True, select_k=True, max_k=max_k)
    grid = make_param_grid(max_k)
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

        model.fit(X, y)  # final fit on full train

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

        # Reliability curve for the best strategy (lowest OOF brier among strategies) in this tier
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

    print(f"Step 4 ({MODEL_NAME}) complete. Outputs in {OUT}.")


if __name__ == "__main__":
    main()
