#!/usr/bin/env bash
# Runs src/01 ... src/10 in order, stopping on the first error.
set -e

cd "$(dirname "$0")"

python src/01_explore.py
python src/02_clean_and_split.py
python src/03_feature_analysis.py
python src/04_train_naive_bayes.py
python src/05_train_logistic_regression.py
python src/06_train_random_forest.py
python src/07_train_bayesian_network.py
python src/08_evaluate.py
python src/09_explain.py
python src/10_sensitivity_analysis.py

echo "All steps completed successfully. Run 'streamlit run app/Home.py' to start the app."
