"""Step 9: Interpretability (SHAP for RF/LR, a simple evidence proxy for NB).

Reads: data/processed/{train,test}.csv, results/selected_models.json,
       results/03_feature_analysis/statistical_ranking_{tier}.csv,
       models/{tier}/*.joblib.
Writes: results/09_explain/*.

For calibrated wrappers we refit the underlying (uncalibrated) pipeline with
its best params on the full train set and explain that: calibration only
rescales probabilities, SHAP explains the ranking (see Section 5, Step 9).
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src import config
from src.utils.io import load_json, load_model, save_csv_and_md
from src.utils.pipelines import split_feature_types, build_pipeline
from src.utils.plotting import save_fig

warnings.filterwarnings("ignore")

OUT = config.RESULTS_DIR / "09_explain"
OUT.mkdir(parents=True, exist_ok=True)


def refit_base_pipeline(model_name, tier, strategy, best_params, features, train_df):
    numeric_feats, binary_feats = split_feature_types(features)
    max_k = len(features)
    X = train_df[features]
    y = train_df["pcos"].values

    if model_name == "naive_bayes":
        est = GaussianNB()
        pipe = build_pipeline(numeric_feats, binary_feats, est, strategy, scale=True, select_k=True, max_k=max_k)
    elif model_name == "logistic_regression":
        class_weight = "balanced" if strategy == "class_weight" else None
        est = LogisticRegression(penalty=config.LR_FIXED["penalty"], solver=config.LR_FIXED["solver"],
                                  max_iter=config.LR_FIXED["max_iter"], class_weight=class_weight,
                                  random_state=config.SEED)
        pipe = build_pipeline(numeric_feats, binary_feats, est, strategy, scale=True, select_k=True, max_k=max_k)
    elif model_name == "random_forest":
        class_weight = "balanced" if strategy == "class_weight" else None
        est = RandomForestClassifier(random_state=config.SEED, class_weight=class_weight)
        pipe = build_pipeline(numeric_feats, binary_feats, est, strategy, scale=False, select_k=False)
    else:
        raise ValueError(model_name)

    pipe.set_params(**best_params)
    pipe.fit(X, y)
    return pipe, numeric_feats, binary_feats


def transform_features(pipe, X_df):
    """Apply every pipeline step except 'resample' (fit-only, no .transform) and 'clf'."""
    X = X_df
    for name, step in pipe.steps:
        if name in ("resample", "clf"):
            continue
        X = step.transform(X)
    return X


def explain_rf(pipe, numeric_feats, binary_feats, train_df, test_df, features, tier):
    X_test_trans = transform_features(pipe, test_df[features])
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    shap_values = explainer.shap_values(X_test_trans)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    fig = plt.figure()
    shap.summary_plot(shap_values, X_test_trans, feature_names=features, show=False)
    save_fig(fig, OUT / f"shap_summary_{tier}_random_forest.png")

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(features)[order][:15][::-1], mean_abs[order][:15][::-1], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Random Forest SHAP importance ({tier})")
    save_fig(fig, OUT / f"shap_bar_{tier}_random_forest.png")

    return dict(zip(features, mean_abs)), shap_values, X_test_trans, explainer


def explain_lr(pipe, numeric_feats, binary_feats, train_df, test_df, features, tier):
    X_train_trans = transform_features(pipe, train_df[features])
    X_test_trans = transform_features(pipe, test_df[features])

    selected_mask = None
    feat_names = features
    if "select" in pipe.named_steps:
        selected_mask = pipe.named_steps["select"].get_support()
        feat_names = list(np.array(features)[selected_mask])

    explainer = shap.LinearExplainer(pipe.named_steps["clf"], X_train_trans)
    shap_values = explainer.shap_values(X_test_trans)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig = plt.figure()
    shap.summary_plot(shap_values, X_test_trans, feature_names=feat_names, show=False)
    save_fig(fig, OUT / f"shap_summary_{tier}_logistic_regression.png")

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feat_names)[order][:15][::-1], mean_abs[order][:15][::-1], color="#C44E52")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Logistic Regression SHAP importance ({tier})")
    save_fig(fig, OUT / f"shap_bar_{tier}_logistic_regression.png")

    return dict(zip(feat_names, mean_abs))


def waterfall_examples(pipe, explainer, shap_values, X_test_trans, features, test_df, tier, y_test, proba):
    y_pred = (proba >= 0.5).astype(int)
    tp_idx = np.where((y_test == 1) & (y_pred == 1))[0]
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]
    tn_idx = np.where((y_test == 0) & (y_pred == 0))[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else base_value[0]

    for case_name, idx_arr in [("true_positive", tp_idx), ("false_negative", fn_idx), ("true_negative", tn_idx)]:
        if len(idx_arr) == 0:
            continue
        i = idx_arr[0]
        expl = shap.Explanation(values=shap_values[i], base_values=base_value,
                                 data=X_test_trans[i], feature_names=features)
        fig = plt.figure()
        shap.plots.waterfall(expl, show=False)
        save_fig(fig, OUT / f"shap_waterfall_example_{tier}_{case_name}.png")


def consensus_table(tier, rf_shap, lr_shap):
    stat_path = config.RESULTS_DIR / "03_feature_analysis" / f"statistical_ranking_{tier}.csv"
    stat_df = pd.read_csv(stat_path)

    rf_ranking = pd.Series(rf_shap).rank(ascending=False, method="min")
    lr_ranking = pd.Series(lr_shap).rank(ascending=False, method="min") if lr_shap else pd.Series(dtype=float)

    features = config.TIER_FEATURES[tier]
    rows = []
    for feat in features:
        row = {"feature": feat}
        stat_row = stat_df[stat_df["feature"] == feat]
        row["mi_rank"] = float(stat_row["mi_rank"].iloc[0]) if len(stat_row) else np.nan
        row["stat_rank"] = float(stat_row["stat_rank"].iloc[0]) if len(stat_row) else np.nan
        row["rf_shap_rank"] = float(rf_ranking.get(feat, np.nan))
        row["lr_shap_rank"] = float(lr_ranking.get(feat, np.nan)) if feat in lr_ranking else np.nan
        ranks = [v for v in [row["mi_rank"], row["stat_rank"], row["rf_shap_rank"], row["lr_shap_rank"]] if pd.notna(v)]
        row["avg_rank"] = float(np.mean(ranks)) if ranks else np.nan
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("avg_rank")
    df["consensus_top10"] = False
    df.iloc[:10, df.columns.get_loc("consensus_top10")] = True
    return df


def nb_evidence(tier, train_df):
    features = config.TIER_FEATURES[tier]
    rows = []
    for feat in features:
        pos = train_df.loc[train_df["pcos"] == 1, feat].dropna()
        neg = train_df.loc[train_df["pcos"] == 0, feat].dropna()
        if len(pos) < 2 or len(neg) < 2:
            continue
        pooled_std = np.sqrt(((len(pos) - 1) * pos.var() + (len(neg) - 1) * neg.var()) / (len(pos) + len(neg) - 2))
        evidence = abs(pos.mean() - neg.mean()) / pooled_std if pooled_std > 0 else np.nan
        rows.append({"feature": feat, "abs_standardized_mean_diff": evidence})
    return pd.DataFrame(rows).sort_values("abs_standardized_mean_diff", ascending=False)


def main():
    train_df = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    y_test = test_df["pcos"].values
    selected = load_json(config.RESULTS_DIR / "selected_models.json")

    for tier in config.TIERS:
        rf_path = config.ROOT / selected[tier]["random_forest"]
        lr_path = config.ROOT / selected[tier]["logistic_regression"]

        rf_artifact = load_model(rf_path)
        lr_artifact = load_model(lr_path)

        rf_strategy = rf_path.stem.split("__")[1]
        lr_strategy = lr_path.stem.split("__")[1]

        rf_pipe, rf_num, rf_bin = refit_base_pipeline(
            "random_forest", tier, rf_strategy, rf_artifact["best_params"], rf_artifact["features"], train_df)
        lr_pipe, lr_num, lr_bin = refit_base_pipeline(
            "logistic_regression", tier, lr_strategy, lr_artifact["best_params"], lr_artifact["features"], train_df)

        rf_shap, shap_values, X_test_trans, explainer = explain_rf(
            rf_pipe, rf_num, rf_bin, train_df, test_df, rf_artifact["features"], tier)
        lr_shap = explain_lr(lr_pipe, lr_num, lr_bin, train_df, test_df, lr_artifact["features"], tier)

        if tier in ("tier2_lab", "tier3_full"):
            rf_proba = rf_pipe.predict_proba(test_df[rf_artifact["features"]])[:, 1]
            waterfall_examples(rf_pipe, explainer, shap_values, X_test_trans,
                                rf_artifact["features"], test_df, tier, y_test, rf_proba)

        consensus_df = consensus_table(tier, rf_shap, lr_shap)
        save_csv_and_md(consensus_df, OUT / f"feature_importance_consensus_{tier}")

        nb_df = nb_evidence(tier, train_df)
        nb_df.to_csv(OUT / f"nb_feature_evidence_{tier}.csv", index=False)

        print(f"{tier}: explanation outputs written.")

    print("Step 9 (09_explain.py) complete.")


if __name__ == "__main__":
    main()
