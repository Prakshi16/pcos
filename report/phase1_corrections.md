# Phase 1 Report — Corrections

**Purpose of this file:** `PCOS_Phase1_Report.pdf` was written as a proposal, before any code existed. The project has since been fully implemented (`PCOS_PROJECT_PLAN.md`, `src/`, `results/`, `models/`, `app/`), and several planned details turned out different in practice. This file lists every place Phase 1's text is now outdated or inaccurate, with the original wording quoted, why it's wrong, and what it should say instead.

**How to use this file:** For each item below, find the quoted "Current text" in `PCOS_Phase1_Report.pdf`, and replace/extend it per "Correction." Items are grouped by the report's own section numbers. Section 1 (Background and Justification) and most of Section 4 (Related Research — literature citations) are **not** covered here; they were reviewed and left as-is (Section 4's citations are outside the scope of this pass — see note at the end). Do not rewrite sections not listed below; they remain accurate as written.

---

## Section 2 — Goals and Scope of the Project

### 2.1 Specific Objectives — add two explicit objectives

**Current text:** The bullet list of Specific Objectives (Data sourcing, Preprocessing, Model building, Feature analysis, Evaluation and prototyping) has no objective naming cost-tiered comparison or calibration.

**Correction:** Add two new objectives to the list:
- **Cost-tiered comparison** — train and evaluate the same model families on three nested feature tiers (screening-only questionnaire/vitals; + blood tests; + ultrasound), and report how much predictive performance is lost when restricting to the cheaper tiers.
- **Calibration evaluation** — treat probability calibration (Brier score, reliability curves) as a primary evaluation criterion alongside discrimination metrics, not an afterthought.

### 2.2 Data sourcing — remove the aspirational multi-source language

**Current text:** "Data sourcing — Consolidate a reliable patient health dataset covering clinical, biochemical, and ultrasound-derived PCOS features, primarily the Kaggle PCOS dataset (Kerala hospitals, 541 patients, 44 features), supplemented where feasible by UCI or IEEE DataPort sources."

**What's wrong:** The UCI/IEEE DataPort supplementation was never pursued — the build used the Kottarathil Excel file (`PCOS_data_without_infertility.xlsx`, sheet `Full_new`) exclusively, as its sole data source.

**Correction:** "Data sourcing — Use the Kaggle PCOS dataset (Kottarathil; 541 patients, 10 hospitals in Kerala, 44 raw columns) as the single data source, loaded from the `Full_new` sheet."

---

## Section 4 — Summary of Related Research and Prior Approaches

### 4.1 Remove the "calibrated probability outputs" claim about Naive Bayes

**Current text:** "This directly supports the present project's choice of probabilistic models: even though Naive Bayes underperforms ensemble methods on raw accuracy, its calibrated probability outputs and interpretability make it more suitable for a decision-support context than a black-box high-accuracy classifier."

**What's wrong:** Naive Bayes is *not* inherently well-calibrated — its conditional-independence assumption typically makes it *overconfident* (probabilities pushed toward 0/1). Calibration is something this project evaluates and corrects for (via Platt/sigmoid and isotonic recalibration), not a property NB has by default.

**Correction:** "This directly supports the present project's choice of probabilistic models: Naive Bayes and Logistic Regression both produce probability outputs whose calibration this project explicitly evaluates and, where needed, corrects (via Platt/sigmoid and isotonic recalibration) — rather than assuming any model is well-calibrated out of the box. Naive Bayes' independence assumption in particular tends to produce overconfident probabilities before recalibration."

### 4.2 Add the four-contribution statement

**Correction — add as a new paragraph at the end of Section 4** (after "Identified Gaps This Project Addresses"):

> Unlike prior work that reports a single accuracy figure on the full feature set, this project (i) quantifies the predictive cost of restricting inputs to progressively cheaper clinical tiers, (ii) evaluates probability calibration alongside discrimination across those tiers, (iii) uses a clinically specified Bayesian Network to produce risk estimates from incomplete patient information, and (iv) explicitly audits the circularity of using Rotterdam diagnostic-criteria features (follicle count, cycle irregularity, hyperandrogenism signs) as predictors, by measuring performance with and without them.

*(The rest of Section 4 — Aggarwal et al. 93.52%, SmartScanPCOS, EAIBS-PCOS, etc. — is left as originally written; those citations were not part of this correction pass.)*

---

## Section 5 — Planned Framework / Proposed Solution Design

### 5.1 High-Level Architecture — Layer 1

**Current text:** "Layer 1 — Data Layer: raw patient health data (Kaggle PCOS dataset as primary source, supplemented by UCI/IEEE DataPort if needed) → cleaned, structured dataset."

**Correction:** "Layer 1 — Data Layer: raw patient health data (Kaggle PCOS dataset, single source) → cleaned, structured dataset, split into three nested cost tiers (screening / +lab / +ultrasound)."

### 5.2 Add a "Feature tiers" subsection

**Correction — insert directly after the High-Level Architecture list:**

> **Feature tiers.** Every model is trained on three nested feature sets: **Tier 1 (screening)** — questionnaire and basic vitals only; **Tier 2 (lab)** — Tier 1 plus blood tests; **Tier 3 (full)** — Tier 2 plus ultrasound-derived measurements. Tiers are nested (Tier 3 ⊇ Tier 2 ⊇ Tier 1), and results are reported per tier to show how much performance is retained at each cheaper, more accessible level of testing.

### 5.3 Step 1: Data Collection

**Current text:** "Supplementary sources considered: UCI Machine Learning Repository and IEEE DataPort PCOS data, to assess feasibility of combining datasets for improved generalizability (flagged as a stretch goal given differing feature sets across sources)."

**What's wrong:** This stretch goal was not pursued.

**Correction:** Remove this bullet, or replace with: "The UCI/IEEE DataPort supplementation considered in the proposal was not pursued; the final dataset is the Kottarathil Kaggle file only, used unmodified apart from the cleaning rules below."

### 5.4 Step 2: Data Cleaning and Preprocessing — four corrections

**a) Missing value handling**

**Current text:** "Missing value handling: assess missingness pattern per feature; apply mean/median imputation for continuous variables and mode imputation for categorical variables, with sensitivity checks."

**What's wrong:** The build uses **median only** for numeric features (never mean), and most-frequent (mode) for binary features — fit inside cross-validation training folds only, never on the whole dataset up front. The "sensitivity checks" happened separately and specifically later, not as part of this step.

**Correction:** "Missing value handling: median imputation for numeric features, most-frequent imputation for binary features, both fit inside each cross-validation training fold (never on the full dataset) to avoid leakage. Imputation-strategy sensitivity (median vs. KNN vs. iterative imputation) is checked separately as a robustness analysis (see Step 7)."

**b) Outlier detection**

**Current text:** "Outlier detection: use IQR or z-score methods on continuous biochemical variables (LH, FSH, AMH, TSH) to flag implausible lab values."

**What's wrong:** The build does not use a generic statistical (IQR/z-score) rule. It uses domain-specific, clinically motivated plausible-value ranges defined per feature (e.g., age 12–60, BMI 10–60, FSH/LH 0–200, AMH 0–70, follicle counts 0–40), applied to every numeric feature, not just the four named biochemical ones.

**Correction:** "Outlier detection: apply per-feature, domain-specific plausible-value ranges (e.g., age 12–60 years, BMI 10–60 kg/m², FSH/LH 0–200 mIU/mL, AMH 0–70 ng/mL, follicle counts 0–40) to every numeric feature; values outside range are set to missing (not dropped as rows) and logged for review, rather than using a generic statistical (IQR/z-score) outlier rule."

**c) Categorical encoding**

**Current text:** "Categorical encoding: one-hot or ordinal encoding for categorical symptom indicators (e.g., hirsutism, acne, hair loss)."

**What's wrong:** Every categorical feature in this dataset is a simple binary Y/N indicator (weight gain, hair growth, skin darkening, hair loss, pimples, fast food, regular exercise, pregnant, cycle irregularity) — none are multi-category, so one-hot/ordinal encoding was never needed.

**Correction:** "Categorical encoding: all symptom/lifestyle indicators in this dataset are binary (Y/N); these are mapped directly to 0/1, with no one-hot or ordinal encoding required."

**d) Class imbalance correction**

**Current text:** "Class imbalance correction: apply SMOTE or class-weighted loss functions to address the roughly 2:1 imbalance between PCOS-negative and PCOS-positive cases."

**What's wrong:** This phrasing implies picking one approach. The build instead compares three imbalance-handling strategies as separate candidates for every model — no correction, class weighting, and SMOTE-NC (the categorical-aware SMOTE variant, since several features are binary) — and picks the best per tier via cross-validation, rather than committing to one upfront.

**Correction:** "Class imbalance correction: for each model, three strategies are trained and compared as separate candidates — no correction, class-weighted loss (`class_weight='balanced'`), and SMOTE-NC (a SMOTE variant that respects binary features) — selected via cross-validation rather than fixed in advance. SMOTE alters the effective class prior during training, so its effect on probability calibration is measured explicitly, not assumed to be neutral."

**e) Train/validation/test split**

**Current text:** "Train/validation/test split: stratified split (e.g., 70/15/15) with k-fold cross-validation (5- or 8-fold) used during model selection."

**Correction:** "Train/test split: a single stratified 80/20 split (432 training patients, 109 held out for final testing), with all model selection and performance estimation done via nested cross-validation on the training set only — an outer `RepeatedStratifiedKFold` (5 splits × 3 repeats) to estimate generalization performance, and an inner `StratifiedKFold` (5 splits) for hyperparameter search. All preprocessing (imputation, scaling, resampling, feature selection) is fit within each training fold only, never on validation or test data, to prevent leakage. The held-out test set is used exactly once, for final evaluation."

### 5.5 Step 3: Feature Analysis and Selection

**Current text:** "Statistical feature selection (chi-square, ANOVA F-test, or SelectKBest) to rank features by predictive relevance."

**What's wrong:** This reads as choosing one method. The build runs all three approaches together and reports all three rankings side by side: ANOVA F-test for numeric features, chi-square for binary features, and mutual information across all features — not a single selected method.

**Correction:** "Statistical feature ranking: ANOVA F-test (numeric features), chi-square (binary features), and mutual information (all features) are computed and ranked separately, side by side, per feature tier — not used to pick a single 'best' method, but to cross-check which features are consistently important across statistical criteria."

### 5.6 Step 4: Model Development

**Current text:** "Naive Bayes (Gaussian for continuous features) as the baseline probabilistic model, valued for simplicity, speed, and calibrated probability outputs."

**What's wrong:** Same issue as §4.1 — NB is not inherently well-calibrated.

**Correction:** "Naive Bayes (Gaussian for continuous features) as the baseline probabilistic model, valued for simplicity and speed; its raw probability outputs are typically overconfident due to the independence assumption, so calibration (sigmoid/isotonic recalibration) is evaluated and applied where it helps, not assumed."

**Current text:** "Bayesian Network, built via expert-informed structure or structure-learning algorithms, to capture conditional dependencies Naive Bayes' independence assumption ignores."

**Correction:** "Bayesian Network: a clinically specified DAG over ~10 discretized variables (structure hand-built from domain knowledge, not learned), with parameters fit via Bayesian estimation (BDeu prior). A Hill-Climb/BIC structure-learning run is included only as a comparison point, not as the primary structure. The BN's key practical advantage over the other models is that it supports inference from partial/incomplete evidence — it can estimate P(PCOS | whatever is known) even when many fields are missing."

**Current text:** "Baseline comparison models: Logistic Regression, Random Forest, and optionally SVM, to contextualize probabilistic model performance against established high-accuracy approaches."

**What's wrong:** SVM was never implemented.

**Correction:** "Baseline comparison models: Logistic Regression and Random Forest, to contextualize probabilistic model performance against established high-accuracy approaches."

**Current text:** "Tooling: Python with scikit-learn, pgmpy (Bayesian Networks), pandas/NumPy, and matplotlib/seaborn for visualization."

**What's wrong:** Incomplete — omits several libraries the build actually depends on.

**Correction:** "Tooling: Python with scikit-learn, imbalanced-learn (SMOTE-NC), pgmpy (Bayesian Networks), shap (interpretability), pandas/NumPy, matplotlib/seaborn (visualization), joblib (model persistence), openpyxl (Excel loading), and Streamlit (the decision-support prototype)."

### 5.7 Step 5: Evaluation Plan

**Current text:** "Standard classification metrics: accuracy, precision, recall, F1-score, and ROC-AUC, computed on the held-out test set."

**Correction:** Add: "PR-AUC (average precision) and specificity are also reported, alongside 95% bootstrap confidence intervals (1000 resamples) for ROC-AUC, Brier score, and recall. Beyond the standard 0.5 threshold, a recall-targeted 'screening threshold' is selected per model (the highest threshold achieving out-of-fold recall ≥ 0.90, chosen because missed PCOS cases carry a higher clinical cost than false alarms in a screening context), and metrics are reported at both thresholds."

### 5.8 Step 6: Interpretability and Decision Support Prototype

**Current text:** "Generate SHAP summary plots (or feature-importance bar charts) showing which features most influence predictions overall and for individual patient cases."

**What's wrong:** SHAP is applied to Random Forest (`TreeExplainer`) and Logistic Regression (`LinearExplainer`) only. Naive Bayes gets a separate, simpler evidence proxy (per-feature standardized class-mean difference), since SHAP doesn't apply as cleanly to NB's generative structure; the Bayesian Network's explanation is its posterior probability trace over evidence, not a SHAP value.

**Correction:** "Generate SHAP summary plots (beeswarm and bar charts) for Random Forest (`TreeExplainer`) and Logistic Regression (`LinearExplainer`), showing which features most influence predictions overall and for individual patient cases. For Naive Bayes, use a simpler evidence proxy (standardized class-mean difference per feature) in place of SHAP. For the Bayesian Network, interpretability comes from showing how the posterior P(PCOS | evidence) changes as each piece of evidence is added or removed."

**Current text:** "Build a simple prototype interface (e.g., a Streamlit web app) where a user inputs patient parameters and receives a risk probability, a plain-language summary of top contributing factors, and a recommendation to seek clinical confirmation if risk exceeds a defined threshold."

**Correction:** "Build a Streamlit application with four pages: (1) **Risk Screening** — tiered risk prediction (choice of Tier 1/2/3 and NB/LR/RF/BN) with a probability, risk band, plain-language top-3-factor explanation, and quick-fill example patient profiles; (2) **Partial-Information Bayesian Inference** — BN-based risk estimation from whichever evidence fields the user knows, with a 'what would change the estimate' view over the unknown variables; (3) **Model Comparison** — test-set metrics, ROC/calibration curves, and SHAP summaries across all tiers and models; (4) **About & Limitations** — dataset provenance, label-circularity and generalizability caveats, and clinical-validation status."

### 5.9 Layer 5 (High-Level Architecture)

**Current text:** "Layer 5 — Decision Support Layer: a prototype interface (web mock-up or notebook-based dashboard) that takes patient inputs and outputs a risk probability with an explanation of contributing factors."

**Correction:** "Layer 5 — Decision Support Layer: a Streamlit application (not a notebook or static mock-up) with pages for tiered risk screening, partial-information Bayesian inference, model comparison, and limitations, loading the serialized trained models and performing inference locally with no separate backend/API."

### 5.10 Validation and Robustness Checks — add the missing circularity check

**Current text:** "Sensitivity analysis: re-run key models with different imputation strategies and feature subsets to check result stability."

**What's wrong:** This describes two of the three actual robustness checks but omits the third and most important one for this project's stated contributions: the label-circularity ablation.

**Correction:** "Sensitivity analysis (run on Tier 2, using Logistic Regression and Random Forest): (1) **imputation sensitivity** — median vs. KNN vs. iterative imputation; (2) **feature-subset sensitivity** — all Tier 2 features vs. the consensus top-10 vs. the ANOVA top-10; (3) **label-circularity check** — Tier 3 with vs. without the Rotterdam diagnostic-criteria features (follicle counts, cycle irregularity, hair growth), to directly measure how much of Tier 3's performance comes from features that partially define the label itself."

---

## Section 6 — Anticipated Results and Contributions

### 6.1 Reframe the calibration claim as a hypothesis

**Current text:** "Probabilistic models are anticipated to show a modest accuracy trade-off relative to ensemble methods but are expected to produce better-calibrated probability estimates (lower Brier score, more reliable calibration curves), which matters more than raw accuracy for a risk-communication tool."

**Correction:** "Whether probabilistic models produce better-calibrated probability estimates than ensemble methods is treated as a hypothesis to test, not an assumed outcome. In particular, Naive Bayes' raw probabilities are expected to be poorly calibrated (overconfident) before recalibration, given its independence assumption; Logistic Regression is expected to be comparatively well-calibrated even without recalibration, since it directly optimizes a probabilistic (log-loss) objective. Performance is also expected to drop moving from Tier 3 (full, including ultrasound) down to Tier 1 (screening-only), and this drop is quantified explicitly as one of the project's headline results, not treated as an incidental finding."

### 6.2 Add the label-circularity limitation

**Correction — add to "Limitations to Be Transparently Reported":**
- "Several Tier 3 features (follicle count, cycle irregularity, hyperandrogenism signs such as hair growth) are components of the Rotterdam criteria used to assign the PCOS label itself. This inflates Tier 3 performance relative to what a fully independent predictor would achieve — the model partially learns to reproduce the diagnostic rule rather than an independent biological signal. This project quantifies that inflation directly via a feature-ablation sensitivity check (Section 5, Validation and Robustness Checks)."

### 6.3 Add the contribution statement

**Correction — add near the start of Section 6, before "Expected Model Performance":** the same four-contribution paragraph given in §4.2 above (cost tiers / calibration / BN under incomplete evidence / circularity audit). Stating it once in Section 4 and once in Section 6 is fine — the plan document (`PCOS_PROJECT_PLAN.md`) allows either placement; if only adding it once, Section 6 is the more natural spot since it directly precedes the results discussion it motivates.

---

## Note on scope

This pass covers R1–R11 from `PCOS_PROJECT_PLAN.md` §7 plus the additional plan-vs-actual mismatches found by comparing the PDF text directly against the implemented pipeline (`config.py`, `src/*.py`, `results/`). It deliberately does **not** touch Section 1 (Background/Justification, still accurate) or Section 4's literature citations (Aggarwal et al., SmartScanPCOS, EAIBS-PCOS, etc.) — those were reviewed separately and left out of scope for this correction pass.
