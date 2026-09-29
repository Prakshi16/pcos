"""Model comparison page: test-set table, per-tier ROC/calibration/confusion images,
the headline tier-comparison figure, and SHAP summaries."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src import config
from app.app_utils import TIER_LABELS

st.set_page_config(page_title="Model Comparison", layout="wide")
st.warning("This is an academic screening prototype. It does not diagnose PCOS. "
           "Diagnosis requires clinical evaluation, ultrasound, and exclusion of other conditions.")
st.title("Model Comparison")

EVAL_DIR = config.RESULTS_DIR / "08_evaluate"
EXPLAIN_DIR = config.RESULTS_DIR / "09_explain"

st.subheader("Test-set results (all tiers, all models)")
md_path = EVAL_DIR / "test_results_all.md"
if md_path.exists():
    st.markdown(md_path.read_text())

st.subheader("Headline result: performance across cost tiers")
col1, col2 = st.columns(2)
with col1:
    st.image(str(EVAL_DIR / "tier_comparison.png"), caption="Test ROC-AUC by tier")
with col2:
    st.image(str(EVAL_DIR / "tier_comparison_brier.png"), caption="Test Brier score by tier")

tabs = st.tabs([TIER_LABELS[t] for t in config.TIERS])
for tab, tier in zip(tabs, config.TIERS):
    with tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            p = EVAL_DIR / f"roc_curves_{tier}.png"
            if p.exists():
                st.image(str(p), caption="ROC curves")
        with c2:
            p = EVAL_DIR / f"calibration_curves_{tier}.png"
            if p.exists():
                st.image(str(p), caption="Calibration curves")
        with c3:
            p = EVAL_DIR / f"calibration_raw_vs_calibrated_{tier}.png"
            if p.exists():
                st.image(str(p), caption="Raw vs calibrated")

        st.write("Confusion matrices (screening threshold):")
        cm_cols = st.columns(4)
        for i, model_name in enumerate(["naive_bayes", "logistic_regression", "random_forest", "bayesian_network"]):
            p = EVAL_DIR / f"confusion_matrix_{tier}_{model_name}_screening.png"
            if p.exists():
                with cm_cols[i]:
                    st.image(str(p), caption=model_name)

        st.write("SHAP summaries:")
        shap_cols = st.columns(2)
        for i, model_name in enumerate(["random_forest", "logistic_regression"]):
            p = EXPLAIN_DIR / f"shap_summary_{tier}_{model_name}.png"
            if p.exists():
                with shap_cols[i]:
                    st.image(str(p), caption=f"{model_name} SHAP summary")
