"""Static page: dataset source, tier definitions, label circularity, generalizability,
clinical-validation status, dataset link."""
import streamlit as st

st.set_page_config(page_title="About & Limitations", layout="wide")
st.title("About & Limitations")

st.header("Dataset")
st.write(
    "This prototype is trained on the Kaggle PCOS dataset (Kottarathil), collected from "
    "541 patients across 10 hospitals in Kerala, India. "
    "[Kaggle dataset link](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos)."
)

st.header("Feature tiers")
st.write(
    "- **Tier 1 (screening):** questionnaire and basic vitals only.\n"
    "- **Tier 2 (lab):** Tier 1 plus blood tests.\n"
    "- **Tier 3 (full):** Tier 2 plus ultrasound.\n\n"
    "We report how much performance is lost when moving from the full feature set down "
    "to the cheaper screening tier."
)

st.header("Label circularity")
st.write(
    "Some Tier 3 features — follicle count, cycle irregularity, and hair growth — are part "
    "of the Rotterdam diagnostic criteria used to assign the PCOS label itself. This means "
    "Tier 3 performance is inflated relative to what a truly independent predictor would "
    "achieve: the model partially learns to reproduce the diagnostic rule rather than an "
    "independent biological signal. Our sensitivity analysis (Step 10) quantifies this by "
    "measuring the performance drop when these criterion-features are removed."
)

st.header("Generalizability limits")
st.write(
    "The dataset is drawn from a single geographic population (Kerala, India) and a modest "
    "sample size (541 patients). Performance on other populations, ethnicities, age ranges, "
    "or clinical settings is unknown and would need external validation."
)

st.header("Clinical validation status")
st.error(
    "This tool is **not** clinically validated. It is an academic prototype built for a "
    "Machine Learning coursework project and must not be used for real clinical "
    "decision-making. Any risk estimate must be confirmed by a qualified clinician using "
    "history, examination, laboratory tests, and ultrasound."
)
