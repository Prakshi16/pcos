"""Tiered risk screening page: pick a tier and model, fill in inputs, predict."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src import config
from app.app_utils import (
    load_selected_models, load_input_schema, load_test_results,
    render_inputs, explain_prediction, risk_band, MODEL_LABELS, TIER_LABELS,
)

st.set_page_config(page_title="Risk Screening", layout="wide")
st.warning("This is an academic screening prototype. It does not diagnose PCOS. "
           "Diagnosis requires clinical evaluation, ultrasound, and exclusion of other conditions.")
st.title("Risk Screening")

models = load_selected_models()
schema = load_input_schema()
test_results = load_test_results()

tier_choice = st.radio("Select tier", list(TIER_LABELS.values()))
tier = [k for k, v in TIER_LABELS.items() if v == tier_choice][0]

model_options = ["naive_bayes", "logistic_regression", "random_forest"]
default_model = models[tier]["app_default"]
default_idx = model_options.index(default_model) if default_model in model_options else 0
model_choice_label = st.selectbox("Select model", [MODEL_LABELS[m] for m in model_options],
                                   index=default_idx)
model_name = [k for k, v in MODEL_LABELS.items() if v == model_choice_label][0]

x_row = render_inputs(tier, schema, allow_unknown=False)

if st.button("Predict risk"):
    entry = models[tier][model_name]
    features = entry["features"]
    proba = entry["model"].predict_proba(x_row[features])[:, 1][0]
    threshold = entry["threshold"]

    st.metric("Estimated PCOS risk", f"{proba * 100:.1f}%")
    band, color = risk_band(proba, threshold)
    st.markdown(f"**Risk band:** :{color}[{band}]  \n**Screening threshold:** {threshold:.2f}")

    contributors = explain_prediction(model_name, entry, x_row, tier)
    if contributors:
        st.subheader("Top contributing factors")
        for label, val, direction in contributors:
            st.write(f"- {label} = {val} **{direction}** the estimated risk.")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 2.5))
        labels = [c[0] for c in contributors]
        vals = [1 if c[2] in ("increased",) else -1 for c in contributors]
        ax.barh(labels[::-1], vals[::-1], color=["#C44E52" if v > 0 else "#4C72B0" for v in vals[::-1]])
        ax.set_xlabel("Direction of contribution")
        st.pyplot(fig)

    if band.startswith("High"):
        st.error("Consider consulting a gynaecologist/endocrinologist for confirmatory evaluation.")
    elif band == "Moderate":
        st.warning("Consider discussing these results with a healthcare provider.")

    with st.expander("How was this computed?"):
        st.write(f"Model: {MODEL_LABELS[model_name]}  |  Tier: {TIER_LABELS[tier]}  |  "
                 f"Screening threshold: {threshold:.3f}")
        row = test_results[(test_results["tier"] == tier) & (test_results["model"] == model_name)
                            & (test_results["threshold_type"] == "screening")]
        if len(row):
            r = row.iloc[0]
            st.write(f"Test ROC-AUC: {r['roc_auc']:.3f}  |  Test Brier score: {r['brier']:.3f}")
