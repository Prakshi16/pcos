"""Shared helpers for the Streamlit app: model loading, input forms, SHAP-based
explanation text, and risk banding."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.utils.io import load_json, load_model
from src.utils.bn import BNClassifier
from src.utils.pipelines import split_feature_types, build_pipeline

MODEL_LABELS = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "bayesian_network": "Bayesian Network",
}

TIER_LABELS = {
    "tier1_screening": "Screening only (no tests)",
    "tier2_lab": "+ Blood tests",
    "tier3_full": "+ Ultrasound (full)",
}

SUBHEADER_GROUPS = {
    "Basic information & vitals": ["age", "bmi", "waist_hip_ratio", "pulse_rate", "resp_rate",
                                    "bp_systolic", "bp_diastolic", "cycle_length", "marriage_years",
                                    "num_abortions", "pregnant"],
    "Symptoms & lifestyle": ["cycle_irregular", "weight_gain", "hair_growth", "skin_darkening",
                              "hair_loss", "pimples", "fast_food", "regular_exercise"],
    "Blood tests": ["hb", "fsh", "lh", "lh_fsh_ratio", "tsh", "amh", "prl", "vit_d3", "prg", "rbs"],
    "Ultrasound": ["follicle_left", "follicle_right", "follicle_size_left",
                   "follicle_size_right", "endometrium"],
}


@st.cache_resource
def load_selected_models():
    selected = load_json(config.RESULTS_DIR / "selected_models.json")
    loaded = {}
    for tier, entry in selected.items():
        loaded[tier] = {"app_default": entry["app_default"]}
        for model_name in ["naive_bayes", "logistic_regression", "random_forest", "bayesian_network"]:
            path = config.ROOT / entry[model_name]
            loaded[tier][model_name] = load_model(path)
    return loaded


@st.cache_resource
def load_input_schema():
    return load_json(config.MODELS_DIR / "input_schema.json")


@st.cache_resource
def load_test_results():
    import pandas as pd
    return pd.read_csv(config.RESULTS_DIR / "08_evaluate" / "test_results_all.csv")


@st.cache_resource
def _train_df():
    import pandas as pd
    return pd.read_csv(config.PROCESSED_DIR / "train.csv")


@st.cache_resource
def get_explainer(tier, model_name):
    """Refit the underlying (uncalibrated) pipeline with its best params on the
    full train set and build a SHAP explainer for it. Calibration only rescales
    probabilities, so this matches the model actually shown to the user
    (see Section 5, Step 9 / app_utils.explain_prediction)."""
    import shap

    selected = load_json(config.RESULTS_DIR / "selected_models.json")
    rel_path = selected[tier][model_name]
    artifact = load_model(config.ROOT / rel_path)
    features = artifact["features"]
    strategy = Path(rel_path).stem.split("__")[1]
    best_params = artifact["best_params"]

    numeric_feats, binary_feats = split_feature_types(features)
    max_k = len(features)
    train_df = _train_df()
    X = train_df[features]
    y = train_df["pcos"].values

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        class_weight = "balanced" if strategy == "class_weight" else None
        est = RandomForestClassifier(random_state=config.SEED, class_weight=class_weight)
        pipe = build_pipeline(numeric_feats, binary_feats, est, strategy, scale=False, select_k=False)
    else:  # logistic_regression
        from sklearn.linear_model import LogisticRegression
        class_weight = "balanced" if strategy == "class_weight" else None
        est = LogisticRegression(penalty=config.LR_FIXED["penalty"], solver=config.LR_FIXED["solver"],
                                  max_iter=config.LR_FIXED["max_iter"], class_weight=class_weight,
                                  random_state=config.SEED)
        pipe = build_pipeline(numeric_feats, binary_feats, est, strategy, scale=True, select_k=True, max_k=max_k)

    pipe.set_params(**best_params)
    pipe.fit(X, y)

    def transform(df):
        out = df
        for name, step in pipe.steps:
            if name in ("resample", "clf"):
                continue
            out = step.transform(out)
        return out

    X_train_trans = transform(X)
    feat_names = features
    if "select" in pipe.named_steps:
        mask = pipe.named_steps["select"].get_support()
        feat_names = list(np.array(features)[mask])

    if model_name == "random_forest":
        explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    else:
        explainer = shap.LinearExplainer(pipe.named_steps["clf"], X_train_trans)

    return pipe, explainer, feat_names, transform


def render_inputs(tier, schema, allow_unknown=False):
    """Render one widget per feature in `tier`, grouped under subheaders.
    Returns a one-row DataFrame (NaN for unknown values)."""
    tier_features = config.TIER_FEATURES[tier]
    values = {}

    for group_name, group_feats in SUBHEADER_GROUPS.items():
        feats_in_tier = [f for f in group_feats if f in tier_features and f in schema]
        if not feats_in_tier:
            continue
        st.subheader(group_name)
        for feat in feats_in_tier:
            entry = schema[feat]
            label = entry["label"]
            if entry["type"] == "numeric":
                p1, p50, p99 = entry["p1"], entry["p50"], entry["p99"]
                span = p99 - p1
                lo = p1 - 0.2 * span
                hi = p99 + 0.2 * span
                if allow_unknown:
                    known = st.checkbox(f"Known: {label}", value=True, key=f"known_{feat}")
                    if known:
                        values[feat] = st.number_input(label, min_value=float(lo), max_value=float(hi),
                                                         value=float(p50), key=feat)
                    else:
                        values[feat] = np.nan
                else:
                    values[feat] = st.number_input(label, min_value=float(lo), max_value=float(hi),
                                                     value=float(p50), key=feat)
            else:
                options = ["No", "Yes", "Unknown"] if allow_unknown else ["No", "Yes"]
                choice = st.radio(label, options, key=feat, horizontal=True)
                values[feat] = np.nan if choice == "Unknown" else (1 if choice == "Yes" else 0)

    return pd.DataFrame([values])


def explain_prediction(model_name, model_entry, x_row, tier):
    """Return top-3 SHAP-style contributors as (label, value, direction)."""
    features = model_entry.get("features")
    if model_name in ("logistic_regression", "random_forest") and features is not None:
        try:
            pipe, explainer, feat_names, transform = get_explainer(tier, model_name)
            X = x_row[features].fillna(x_row[features].median(numeric_only=True))
            X_trans = transform(X)

            sv = explainer.shap_values(X_trans)
            if isinstance(sv, list):
                sv = sv[1]
            if sv.ndim == 3:
                sv = sv[:, :, 1]

            contribs = list(zip(feat_names, sv[0]))
            contribs.sort(key=lambda t: abs(t[1]), reverse=True)
            top3 = contribs[:3]
            result = []
            for feat, val in top3:
                direction = "increased" if val > 0 else "decreased"
                raw_val = x_row[feat].iloc[0] if feat in x_row.columns else None
                result.append((config.FEATURE_LABELS.get(feat, feat), raw_val, direction))
            return result
        except Exception:
            pass

    # NB fallback (or SHAP failure): use the evidence-proxy ranking from Step 9
    try:
        nb_evidence = pd.read_csv(config.RESULTS_DIR / "09_explain" / f"nb_feature_evidence_{tier}.csv")
        top3 = nb_evidence.head(3)
        result = []
        for _, row in top3.iterrows():
            feat = row["feature"]
            raw_val = x_row[feat].iloc[0] if feat in x_row.columns else None
            result.append((config.FEATURE_LABELS.get(feat, feat), raw_val, "associated with"))
        return result
    except Exception:
        return []


def risk_band(prob, threshold):
    if prob < threshold / 2:
        return "Low", "green"
    elif prob < threshold:
        return "Moderate", "orange"
    else:
        return "High — refer for clinical evaluation", "red"
