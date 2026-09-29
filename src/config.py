"""Central configuration: paths, seed, column mapping, tier definitions, hyperparameter grids.

Every script imports from here so that column names, ranges and grids are defined once.
"""
import sys
import types

import numpy as np
from pathlib import Path


def _install_statsmodels_stub_if_blocked():
    """pgmpy eagerly imports statsmodels (via CausalInference -> LinearEstimator)
    even though this project only uses BayesianEstimator, HillClimbSearch/BIC and
    VariableElimination. On some locked-down Windows environments, statsmodels'
    compiled Kalman-filter extensions (statsmodels.tsa.statespace) are refused by
    an Application Control policy, which would otherwise make every pgmpy import
    fail. If the real import fails, register a minimal stub so pgmpy's import
    succeeds; nothing in this project calls the stubbed functionality.
    """
    if "statsmodels.api" in sys.modules:
        return
    try:
        import statsmodels.api  # noqa: F401
        return
    except Exception:
        pass
    stub_api = types.ModuleType("statsmodels.api")
    for name in ("OLS", "GLS", "WLS"):
        setattr(stub_api, name, type(name, (), {}))
    stub_api.add_constant = lambda data, *a, **k: data
    stub_pkg = types.ModuleType("statsmodels")
    stub_pkg.api = stub_api
    sys.modules["statsmodels"] = stub_pkg
    sys.modules["statsmodels.api"] = stub_api


_install_statsmodels_stub_if_blocked()

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

RAW_XLSX = RAW_DIR / "PCOS_data_without_infertility.xlsx"
RAW_SHEET = "Full_new"

# ---------------------------------------------------------------------------
# Column mapping: source name (after whitespace normalization) -> canonical
# ---------------------------------------------------------------------------
COLUMN_MAPPING = {
    "PCOS (Y/N)": "pcos",
    "Age (yrs)": "age",
    "BMI": "bmi",
    "Weight (Kg)": "weight",
    "Height(Cm)": "height",
    "Waist:Hip Ratio": "waist_hip_ratio",
    "Waist(inch)": "waist",
    "Hip(inch)": "hip",
    "Pulse rate(bpm)": "pulse_rate",
    "RR (breaths/min)": "resp_rate",
    "BP _Systolic (mmHg)": "bp_systolic",
    "BP _Diastolic (mmHg)": "bp_diastolic",
    "Cycle(R/I)": "cycle_irregular",
    "Cycle length(days)": "cycle_length",
    "Marraige Status (Yrs)": "marriage_years",
    "Pregnant(Y/N)": "pregnant",
    "No. of aborptions": "num_abortions",
    "Weight gain(Y/N)": "weight_gain",
    "hair growth(Y/N)": "hair_growth",
    "Skin darkening (Y/N)": "skin_darkening",
    "Hair loss(Y/N)": "hair_loss",
    "Pimples(Y/N)": "pimples",
    "Fast food (Y/N)": "fast_food",
    "Reg.Exercise(Y/N)": "regular_exercise",
    "Hb(g/dl)": "hb",
    "FSH(mIU/mL)": "fsh",
    "LH(mIU/mL)": "lh",
    "TSH (mIU/L)": "tsh",
    "AMH(ng/mL)": "amh",
    "PRL(ng/mL)": "prl",
    "Vit D3 (ng/mL)": "vit_d3",
    "PRG(ng/mL)": "prg",
    "RBS(mg/dl)": "rbs",
    "Follicle No. (L)": "follicle_left",
    "Follicle No. (R)": "follicle_right",
    "Avg. F size (L) (mm)": "follicle_size_left",
    "Avg. F size (R) (mm)": "follicle_size_right",
    "Endometrium (mm)": "endometrium",
}

# Columns used only to compute derived features, then dropped.
HELPER_COLUMNS = ["weight", "height", "waist", "hip"]

# Columns dropped outright, with the reason.
DROP_COLUMNS = {
    "Sl. No": "identifier",
    "Patient File No.": "identifier",
    "Blood Group": "coded categories with no clinical rationale for PCOS",
    "I beta-HCG(mIU/mL)": "pregnancy hormone, not a PCOS marker",
    "II beta-HCG(mIU/mL)": "pregnancy hormone, not a PCOS marker",
    "FSH/LH": "replaced by the derived lh_fsh_ratio",
}
# Any "Unnamed:" column is dropped as an empty artifact (matched by prefix at load time).

BINARY_SOURCE_COLUMNS = [
    "Pregnant(Y/N)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)",
]

# Canonical binary features (post-mapping). Every other non-target feature is numeric.
BINARY_FEATURES = [
    "pregnant",
    "weight_gain",
    "hair_growth",
    "skin_darkening",
    "hair_loss",
    "pimples",
    "fast_food",
    "regular_exercise",
    "cycle_irregular",
]

TARGET = "pcos"

# ---------------------------------------------------------------------------
# Feature tiers (nested)
# ---------------------------------------------------------------------------
TIER1_FEATURES = [
    "age", "bmi", "waist_hip_ratio", "pulse_rate", "resp_rate",
    "bp_systolic", "bp_diastolic", "cycle_irregular", "cycle_length",
    "marriage_years", "pregnant", "num_abortions", "weight_gain",
    "hair_growth", "skin_darkening", "hair_loss", "pimples",
    "fast_food", "regular_exercise",
]

TIER2_ONLY_FEATURES = [
    "hb", "fsh", "lh", "lh_fsh_ratio", "tsh", "amh", "prl",
    "vit_d3", "prg", "rbs",
]

TIER3_ONLY_FEATURES = [
    "follicle_left", "follicle_right",
    "follicle_size_left", "follicle_size_right", "endometrium",
]

TIER_FEATURES = {
    "tier1_screening": list(TIER1_FEATURES),
    "tier2_lab": list(TIER1_FEATURES) + list(TIER2_ONLY_FEATURES),
    "tier3_full": list(TIER1_FEATURES) + list(TIER2_ONLY_FEATURES) + list(TIER3_ONLY_FEATURES),
}

TIERS = ["tier1_screening", "tier2_lab", "tier3_full"]

# ---------------------------------------------------------------------------
# Plausible ranges (min, max) inclusive, per canonical feature.
# Values outside become NaN (not dropped as rows).
# ---------------------------------------------------------------------------
PLAUSIBLE_RANGES = {
    "age": (12, 60),
    "bmi": (10, 60),
    "weight": (25, 150),
    "height": (120, 200),
    "waist": (20, 60),
    "hip": (20, 70),
    "waist_hip_ratio": (0.5, 1.5),
    "pulse_rate": (40, 140),
    "resp_rate": (10, 40),
    "bp_systolic": (70, 200),
    "bp_diastolic": (40, 130),
    "cycle_length": (0, 30),
    "marriage_years": (0, 40),
    "num_abortions": (0, 10),
    "hb": (5, 20),
    "fsh": (0, 200),
    "lh": (0, 200),
    "tsh": (0, 70),
    "amh": (0, 70),
    "prl": (0, 118),
    "vit_d3": (0, 150),
    "prg": (0, 50),
    "rbs": (0, 500),
    "follicle_left": (0, 40),
    "follicle_right": (0, 40),
    "follicle_size_left": (0, 50),
    "follicle_size_right": (0, 50),
    "endometrium": (0, 30),
}

# Human-readable labels + units for the app's input schema / forms.
FEATURE_LABELS = {
    "age": "Age (years)",
    "bmi": "BMI (kg/m^2)",
    "waist_hip_ratio": "Waist:Hip ratio",
    "pulse_rate": "Pulse rate (bpm)",
    "resp_rate": "Respiratory rate (breaths/min)",
    "bp_systolic": "Systolic BP (mmHg)",
    "bp_diastolic": "Diastolic BP (mmHg)",
    "cycle_irregular": "Irregular menstrual cycle",
    "cycle_length": "Cycle length (days)",
    "marriage_years": "Years married",
    "pregnant": "Currently pregnant",
    "num_abortions": "Number of prior abortions",
    "weight_gain": "Recent weight gain",
    "hair_growth": "Excess hair growth",
    "skin_darkening": "Skin darkening",
    "hair_loss": "Hair loss",
    "pimples": "Pimples/acne",
    "fast_food": "Frequent fast food intake",
    "regular_exercise": "Regular exercise",
    "hb": "Hemoglobin (g/dl)",
    "fsh": "FSH (mIU/mL)",
    "lh": "LH (mIU/mL)",
    "lh_fsh_ratio": "LH/FSH ratio",
    "tsh": "TSH (mIU/L)",
    "amh": "AMH (ng/mL)",
    "prl": "Prolactin (ng/mL)",
    "vit_d3": "Vitamin D3 (ng/mL)",
    "prg": "Progesterone (ng/mL)",
    "rbs": "Random blood sugar (mg/dl)",
    "follicle_left": "Follicle count (left ovary)",
    "follicle_right": "Follicle count (right ovary)",
    "follicle_size_left": "Avg. follicle size, left (mm)",
    "follicle_size_right": "Avg. follicle size, right (mm)",
    "endometrium": "Endometrium thickness (mm)",
}

# ---------------------------------------------------------------------------
# Bayesian Network discretization rules
# ---------------------------------------------------------------------------
BMI_CUTS = (23, 27.5)  # Asian cutoffs: <23 normal, 23-27.5 overweight, >27.5 obese
LH_FSH_HIGH_CUTOFF = 2.0
FOLLICLE_HIGH_CUTOFF = 12

BN_STRUCTURE = [
    ("bmi_cat", "pcos"),
    ("bmi_cat", "skin_darkening"),
    ("pcos", "cycle_irregular"),
    ("pcos", "hair_growth"),
    ("pcos", "pimples"),
    ("pcos", "skin_darkening"),
    ("pcos", "weight_gain"),
    ("pcos", "lh_fsh_high"),
    ("pcos", "follicle_high"),
    ("follicle_high", "amh_level"),
    ("pcos", "amh_level"),
]

BN_TIER_EVIDENCE = {
    "tier1_screening": ["bmi_cat", "weight_gain", "cycle_irregular", "hair_growth", "pimples", "skin_darkening"],
    "tier2_lab": ["bmi_cat", "weight_gain", "cycle_irregular", "hair_growth", "pimples", "skin_darkening",
                  "lh_fsh_high", "amh_level"],
    "tier3_full": ["bmi_cat", "weight_gain", "cycle_irregular", "hair_growth", "pimples", "skin_darkening",
                   "lh_fsh_high", "amh_level", "follicle_high"],
}

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
NB_PARAM_GRID_BASE = {
    "clf__var_smoothing": np.logspace(-12, -3, 10),
}
NB_K_OPTIONS = [8, 12, 16, "all"]

LR_PARAM_GRID_BASE = {
    "clf__C": [0.01, 0.1, 1, 10],
}
LR_K_OPTIONS = [8, 12, 16, "all"]
LR_FIXED = {"penalty": "l2", "solver": "lbfgs", "max_iter": 5000}

RF_PARAM_GRID = {
    "clf__n_estimators": [300],
    "clf__max_depth": [None, 4, 8],
    "clf__min_samples_leaf": [1, 3, 5],
    "clf__max_features": ["sqrt", 0.5],
}

MODEL_STRATEGIES = {
    "naive_bayes": ["none", "smote"],
    "logistic_regression": ["none", "class_weight", "smote"],
    "random_forest": ["none", "class_weight", "smote"],
}

CALIBRATION_METHODS = ["raw", "sigmoid", "isotonic"]

RECALL_TARGET = 0.90
