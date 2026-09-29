"""Streamlit entry point: title, description, navigation hint, disclaimer."""
import streamlit as st

st.set_page_config(page_title="PCOS Risk Screening", layout="wide")

st.title("PCOS Risk Prediction & Decision Support System")
st.write(
    "An academic prototype that estimates a woman's PCOS risk from clinical, "
    "biochemical and ultrasound data across three cost tiers, and compares "
    "Naive Bayes, Logistic Regression, Random Forest and a Bayesian Network "
    "on discrimination and calibration."
)
st.info("Use the sidebar to navigate: Risk Screening, Partial-Information (Bayesian Network), "
        "Model Comparison, and About & Limitations.")

st.warning(
    "This is an academic screening prototype. It does not diagnose PCOS. "
    "Diagnosis requires clinical evaluation, ultrasound, and exclusion of other conditions."
)
