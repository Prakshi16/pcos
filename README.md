# PCOS Risk Prediction & Decision Support System

Install dependencies with `pip install -r requirements.txt`, place the raw Kaggle
files in `data/raw/` (see `PCOS_PROJECT_PLAN.md` Section 1), run `bash run_all.sh`
to regenerate everything from scratch, then `streamlit run app/Home.py` to launch
the app.

## Pipeline (run in this order; `run_all.sh` does this for you)

- `src/01_explore.py` — exploratory data analysis on the raw Excel file; writes column dumps, missing-value and correlation reports, and per-class distribution plots to `results/01_explore/`.
- `src/02_clean_and_split.py` — cleans and renames columns, derives BMI/waist-hip ratio/LH-FSH ratio, replaces implausible values with NaN, and writes `data/processed/{clean,train,test}.csv` plus `models/input_schema.json`.
- `src/03_feature_analysis.py` — ANOVA/chi-square/mutual-information feature ranking per tier on train only; writes `results/03_feature_analysis/`.
- `src/04_train_naive_bayes.py` — nested-CV training, calibration (raw/sigmoid/isotonic) and screening-threshold selection for Gaussian Naive Bayes across all tiers/strategies; saves models to `models/{tier}/naive_bayes__*.joblib`.
- `src/05_train_logistic_regression.py` — same protocol for Logistic Regression.
- `src/06_train_random_forest.py` — same protocol for Random Forest.
- `src/07_train_bayesian_network.py` — fits the clinically specified Bayesian Network, evaluates it via OOF cross-validation, learns a Hill-Climb/BIC structure for comparison, and saves `models/bayesian_network/bn_model.joblib`.
- `src/08_evaluate.py` — first and only use of the held-out test set; selects the best NB/LR/RF/BN candidate per tier by OOF Brier, computes test metrics with bootstrap CIs, and writes `results/selected_models.json` plus all comparison figures.
- `src/09_explain.py` — SHAP explanations for RF and LR, a mean-difference evidence proxy for NB, and a cross-method feature-importance consensus table.
- `src/10_sensitivity_analysis.py` — imputation, feature-subset and label-circularity robustness checks using the Tier 2 models.

## App

`streamlit run app/Home.py` — a 4-page Streamlit prototype (Risk Screening,
Partial-Information Bayesian inference, Model Comparison, About & Limitations)
that loads the saved models and predicts locally; there is no separate API and
no training inside the app.

See `PCOS_PROJECT_PLAN.md` for the full specification, data dictionary, and the
team's remaining manual tasks (Sections 1 and 7).
