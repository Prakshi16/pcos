"""Step 10: Robustness checks using the Tier 2 selected LR and RF (retuned).

Reads: data/processed/{train,test}.csv, results/selected_models.json,
       results/09_explain/feature_importance_consensus_tier2_lab.csv,
       results/03_feature_analysis/statistical_ranking_tier2_lab.csv.
Writes: results/10_sensitivity_analysis/sensitivity_results.csv/.md + bar charts.

1. Imputation sensitivity: median vs KNNImputer vs IterativeImputer.
2. Feature-subset sensitivity: all Tier 2 features vs consensus top10 vs ANOVA top10.
3. Circularity check: Tier 3 minus Rotterdam-criterion features.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src import config
from src.utils.io import load_json, save_csv_and_md
from src.utils.metrics import compute_metrics
from src.utils.pipelines import split_feature_types, ColumnScaler
from src.utils.plotting import new_fig, save_fig

warnings.filterwarnings("ignore")

OUT = config.RESULTS_DIR / "10_sensitivity_analysis"
OUT.mkdir(parents=True, exist_ok=True)

ROTTERDAM_FEATURES = ["follicle_left", "follicle_right", "cycle_irregular", "hair_growth"]


def build_pipeline_with_imputer(numeric_feats, binary_feats, estimator, imputer_name, scale, select_k=False, k=None):
    if imputer_name == "median":
        num_imputer = SimpleImputer(strategy="median")
    elif imputer_name == "knn":
        num_imputer = KNNImputer(n_neighbors=5)
    elif imputer_name == "iterative":
        num_imputer = IterativeImputer(random_state=config.SEED)
    else:
        raise ValueError(imputer_name)

    prep = ColumnTransformer([
        ("num", num_imputer, numeric_feats),
        ("bin", SimpleImputer(strategy="most_frequent"), binary_feats),
    ])
    steps = [("prep", prep)]
    if scale:
        steps.append(("scale", ColumnScaler(len(numeric_feats), len(numeric_feats) + len(binary_feats))))
    if select_k:
        from sklearn.feature_selection import SelectKBest, f_classif
        steps.append(("select", SelectKBest(score_func=f_classif, k=k or "all")))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def tune_and_eval(pipe, param_grid, X_train, y_train, X_test, y_test):
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    gs = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=inner, n_jobs=-1)
    gs.fit(X_train, y_train)
    proba = gs.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, proba, threshold=0.5)
    return m["roc_auc"], m["brier"]


def main():
    train_df = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    y_train = train_df["pcos"].values
    y_test = test_df["pcos"].values

    tier2_features = config.TIER_FEATURES["tier2_lab"]
    numeric_feats, binary_feats = split_feature_types(tier2_features)

    results = []

    # --- 1. Imputation sensitivity (LR and RF, Tier 2, all features) ---
    for model_name in ["logistic_regression", "random_forest"]:
        for imputer_name in ["median", "knn", "iterative"]:
            if model_name == "logistic_regression":
                est = LogisticRegression(max_iter=5000, random_state=config.SEED)
                pipe = build_pipeline_with_imputer(numeric_feats, binary_feats, est, imputer_name,
                                                    scale=True, select_k=True, k=len(tier2_features))
                grid = {"clf__C": config.LR_PARAM_GRID_BASE["clf__C"]}
            else:
                est = RandomForestClassifier(random_state=config.SEED)
                pipe = build_pipeline_with_imputer(numeric_feats, binary_feats, est, imputer_name, scale=False)
                grid = {"clf__n_estimators": [300], "clf__max_depth": [None, 4, 8]}

            auc, brier = tune_and_eval(pipe, grid, train_df[tier2_features], y_train,
                                        test_df[tier2_features], y_test)
            results.append({"analysis": "imputation", "model": model_name, "variant": imputer_name,
                             "test_roc_auc": auc, "test_brier": brier})
            print(f"[imputation] {model_name}/{imputer_name}: AUC={auc:.3f}, Brier={brier:.3f}")

    # --- 2. Feature-subset sensitivity ---
    consensus_df = pd.read_csv(config.RESULTS_DIR / "09_explain" / "feature_importance_consensus_tier2_lab.csv")
    consensus_top10 = consensus_df.sort_values("avg_rank").head(10)["feature"].tolist()

    stat_df = pd.read_csv(config.RESULTS_DIR / "03_feature_analysis" / "statistical_ranking_tier2_lab.csv")
    anova_top10 = (stat_df.dropna(subset=["mutual_info"])
                   .sort_values("mutual_info", ascending=False).head(10)["feature"].tolist())

    feature_sets = {
        "all_tier2": tier2_features,
        "consensus_top10": consensus_top10,
        "anova_top10": anova_top10,
    }
    for model_name in ["logistic_regression", "random_forest"]:
        for subset_name, feats in feature_sets.items():
            num_f, bin_f = split_feature_types(feats)
            if model_name == "logistic_regression":
                est = LogisticRegression(max_iter=5000, random_state=config.SEED)
                pipe = build_pipeline_with_imputer(num_f, bin_f, est, "median", scale=True,
                                                    select_k=True, k=len(feats))
                grid = {"clf__C": config.LR_PARAM_GRID_BASE["clf__C"]}
            else:
                est = RandomForestClassifier(random_state=config.SEED)
                pipe = build_pipeline_with_imputer(num_f, bin_f, est, "median", scale=False)
                grid = {"clf__n_estimators": [300], "clf__max_depth": [None, 4, 8]}

            auc, brier = tune_and_eval(pipe, grid, train_df[feats], y_train, test_df[feats], y_test)
            results.append({"analysis": "feature_subset", "model": model_name, "variant": subset_name,
                             "test_roc_auc": auc, "test_brier": brier})
            print(f"[feature_subset] {model_name}/{subset_name}: AUC={auc:.3f}, Brier={brier:.3f}")

    # --- 3. Circularity check: Tier 3 minus Rotterdam-criterion features ---
    tier3_features = config.TIER_FEATURES["tier3_full"]
    tier3_minus_rotterdam = [f for f in tier3_features if f not in ROTTERDAM_FEATURES]
    circularity_sets = {"tier3_full": tier3_features, "tier3_minus_rotterdam": tier3_minus_rotterdam}
    for model_name in ["logistic_regression", "random_forest"]:
        for subset_name, feats in circularity_sets.items():
            num_f, bin_f = split_feature_types(feats)
            if model_name == "logistic_regression":
                est = LogisticRegression(max_iter=5000, random_state=config.SEED)
                pipe = build_pipeline_with_imputer(num_f, bin_f, est, "median", scale=True,
                                                    select_k=True, k=len(feats))
                grid = {"clf__C": config.LR_PARAM_GRID_BASE["clf__C"]}
            else:
                est = RandomForestClassifier(random_state=config.SEED)
                pipe = build_pipeline_with_imputer(num_f, bin_f, est, "median", scale=False)
                grid = {"clf__n_estimators": [300], "clf__max_depth": [None, 4, 8]}

            auc, brier = tune_and_eval(pipe, grid, train_df[feats], y_train, test_df[feats], y_test)
            results.append({"analysis": "circularity", "model": model_name, "variant": subset_name,
                             "test_roc_auc": auc, "test_brier": brier})
            print(f"[circularity] {model_name}/{subset_name}: AUC={auc:.3f}, Brier={brier:.3f}")

    results_df = pd.DataFrame(results)
    save_csv_and_md(results_df, OUT / "sensitivity_results")

    for analysis in results_df["analysis"].unique():
        sub = results_df[results_df["analysis"] == analysis]
        fig, ax = new_fig((8, 5))
        width = 0.35
        variants = sub["variant"].unique()
        models = sub["model"].unique()
        x = np.arange(len(variants))
        for i, model_name in enumerate(models):
            vals = [sub[(sub["model"] == model_name) & (sub["variant"] == v)]["test_roc_auc"].values[0]
                    for v in variants]
            ax.bar(x + i * width, vals, width, label=model_name)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(variants, rotation=20, ha="right")
        ax.set_ylabel("Test ROC-AUC")
        ax.set_title(f"Sensitivity analysis: {analysis}")
        ax.legend()
        save_fig(fig, OUT / f"sensitivity_{analysis}.png")

    print("Step 10 (10_sensitivity_analysis.py) complete.")


if __name__ == "__main__":
    main()
