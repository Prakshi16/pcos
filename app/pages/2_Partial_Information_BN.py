"""Partial-information Bayesian Network page: every input defaults to Unknown."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.utils.bn import BNClassifier, discretize
from app.app_utils import load_selected_models, risk_band

st.set_page_config(page_title="Partial-Information BN", layout="wide")
st.warning("This is an academic screening prototype. It does not diagnose PCOS. "
           "Diagnosis requires clinical evaluation, ultrasound, and exclusion of other conditions.")
st.title("Partial-Information Bayesian Inference")
st.write("Every field defaults to **Unknown**. The Bayesian Network reasons using only "
         "the evidence you provide, and can estimate risk even from incomplete information.")

models = load_selected_models()
bn_artifact = models["tier1_screening"]["bayesian_network"]
clf = BNClassifier(bn_artifact["model"], bn_artifact["cutoffs"])

st.subheader("Categorical evidence")
binary_labels = {
    "weight_gain": "Recent weight gain",
    "cycle_irregular": "Irregular menstrual cycle",
    "hair_growth": "Excess hair growth",
    "pimples": "Pimples/acne",
    "skin_darkening": "Skin darkening",
}
binary_evidence = {}
for var, label in binary_labels.items():
    choice = st.radio(label, ["Unknown", "No", "Yes"], key=f"bn_{var}", horizontal=True)
    binary_evidence[var] = None if choice == "Unknown" else ("yes" if choice == "Yes" else "no")

st.subheader("Numeric evidence")
numeric_config = {
    "bmi": ("BMI (kg/m^2)", 10.0, 60.0, 24.0),
    "lh_fsh_ratio": ("LH/FSH ratio", 0.0, 20.0, 1.0),
    "amh": ("AMH (ng/mL)", 0.0, 70.0, 4.0),
    "max_follicle": ("Max follicle count (either ovary)", 0.0, 40.0, 8.0),
}
numeric_raw = {}
for var, (label, lo, hi, default) in numeric_config.items():
    dont_know = st.checkbox(f"Don't know: {label}", value=True, key=f"bn_dk_{var}")
    if dont_know:
        numeric_raw[var] = np.nan
    else:
        numeric_raw[var] = st.number_input(label, min_value=lo, max_value=hi, value=default, key=f"bn_{var}")

if st.button("Estimate risk from available evidence"):
    cutoffs = bn_artifact["cutoffs"]
    evidence = dict(binary_evidence)

    bmi = numeric_raw["bmi"]
    if pd.notna(bmi):
        lo, hi = cutoffs["bmi_cuts"]
        evidence["bmi_cat"] = "normal" if bmi < lo else ("overweight" if bmi <= hi else "obese")

    ratio = numeric_raw["lh_fsh_ratio"]
    if pd.notna(ratio):
        evidence["lh_fsh_high"] = "yes" if ratio > cutoffs["lh_fsh_high_cutoff"] else "no"

    amh = numeric_raw["amh"]
    if pd.notna(amh):
        t1, t2 = cutoffs["amh_tertiles"]
        evidence["amh_level"] = "low" if amh < t1 else ("mid" if amh < t2 else "high")

    max_foll = numeric_raw["max_follicle"]
    if pd.notna(max_foll):
        evidence["follicle_high"] = "yes" if max_foll >= cutoffs["follicle_high_cutoff"] else "no"

    evidence = {k: v for k, v in evidence.items() if v is not None}

    prob = clf.predict_proba_evidence_dict(evidence)

    known_vars = set(evidence.keys())
    tier_match = "tier1_screening"
    for tier in ["tier3_full", "tier2_lab", "tier1_screening"]:
        if known_vars >= set(config.BN_TIER_EVIDENCE[tier]) or known_vars.issubset(set(config.BN_TIER_EVIDENCE[tier])):
            tier_match = tier
            break
    threshold = models[tier_match]["bayesian_network"]["thresholds"][tier_match]

    st.metric("P(PCOS)", f"{prob * 100:.1f}%")
    band, color = risk_band(prob, threshold)
    st.markdown(f"**Risk band:** :{color}[{band}]")

    st.subheader("Evidence used")
    if evidence:
        st.table(pd.DataFrame([{"variable": k, "state": v} for k, v in evidence.items()]))
    else:
        st.write("No evidence provided — showing the population prior.")

    st.subheader("What would change the estimate?")
    all_bn_vars = ["bmi_cat", "weight_gain", "cycle_irregular", "hair_growth", "pimples",
                   "skin_darkening", "lh_fsh_high", "amh_level", "follicle_high"]
    states = {
        "bmi_cat": ["normal", "overweight", "obese"],
        "weight_gain": ["no", "yes"], "cycle_irregular": ["no", "yes"],
        "hair_growth": ["no", "yes"], "pimples": ["no", "yes"], "skin_darkening": ["no", "yes"],
        "lh_fsh_high": ["no", "yes"], "amh_level": ["low", "mid", "high"], "follicle_high": ["no", "yes"],
    }
    unknown_vars = [v for v in all_bn_vars if v not in evidence]
    for var in unknown_vars:
        rows = []
        for state in states[var]:
            hyp_evidence = dict(evidence)
            hyp_evidence[var] = state
            p = clf.predict_proba_evidence_dict(hyp_evidence)
            rows.append({"state": state, "P(PCOS)": f"{p * 100:.1f}%"})
        st.write(f"If **{var}** were:")
        st.table(pd.DataFrame(rows))
