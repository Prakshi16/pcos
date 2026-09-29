# Viva Q&A — one section per file (Task M15)

Each section: what the file reads, what it does, what it writes, and questions
an examiner is likely to ask with ready answers. Numbers below are from the
current run (TSH range widened to 0-70, PRL to 0-118 per M6).

---

## src/config.py
**Reads:** nothing (constants only). **Does:** defines `ROOT`, `SEED=42`, the raw
column → canonical name mapping, dropped columns, the three feature tiers
(nested), `PLAUSIBLE_RANGES`, BN discretization cutoffs/structure, and every
model's hyperparameter grid. **Writes:** nothing; imported by every other script.

- *Q: Why centralize this instead of repeating it per script?*
  A: So every script agrees on column names, the seed, and tier membership —
  a single source of truth prevents tier-3 code accidentally using a tier-1
  feature name that drifted, and makes the seed change in exactly one place.
- *Q: Why is `waist_hip_ratio` recomputed instead of using the source column?*
  A: The source `Waist:Hip Ratio` column is pre-rounded; recomputing from
  waist/hip inches preserves precision for the correlation/feature-selection
  steps.
- *Q: Why widen TSH to (0,70) and PRL to (0,118)?*
  A: The original (0,50)/(0,100) cutoffs were clinically too tight and were
  flagging legitimate high values as implausible; M6 review widened them
  against clinical reference ranges (see `report/literature_research_notes.md`).
- *Q: What's the `_install_statsmodels_stub_if_blocked()` function at the top of this file?*
  A: A Windows-only environment workaround, not a modeling decision. `pgmpy`'s
  package `__init__` files eagerly import `CausalInference` -> `LinearEstimator`,
  which imports `statsmodels.api` (for `OLS`/`GLS`/`WLS`), purely as a side
  effect of pgmpy's module structure — this project never uses that feature
  (only `BayesianEstimator`, `HillClimbSearch`/`BIC`, `VariableElimination`).
  On this machine, a Windows Application Control policy blocks the compiled
  Kalman-filter extensions inside `statsmodels.tsa.statespace` that get pulled
  in transitively, which would otherwise make every `pgmpy` import fail. The
  function tries the real import first and only installs a stub (empty
  `OLS`/`GLS`/`WLS` classes, never actually called) if that fails, so the
  fallback is a no-op everywhere else — including inside the Docker container
  (Linux, unaffected) and any machine without this policy.

## src/01_explore.py
**Reads:** `data/raw/PCOS_data_without_infertility.xlsx` (sheet `Full_new`).
**Does:** whitespace-normalizes column names, profiles missingness, class
balance, summary stats, correlations, and per-class distributions.
**Writes:** `results/01_explore/` — `raw_columns.txt`,
`unique_values_categorical.txt`, `missing_values.csv`, `class_balance.png`,
`summary_statistics.csv`, `correlation_heatmap.png`,
`high_correlation_pairs.csv`, `distributions_by_class/`.

- *Q: What did `high_correlation_pairs.csv` (|r|>0.8) show, and does it matter?*
  A: Four pairs: `Sl. No`/`Patient File No.` (both identifiers, dropped),
  `FSH`/`FSH-LH` (the ratio column is replaced by our derived `lh_fsh_ratio`),
  `Weight`/`BMI` and `Hip`/`Waist` (weight/height/waist/hip are only helper
  columns consumed into `bmi` and `waist_hip_ratio`, then dropped). None of
  the raw correlated pairs survive into the final feature set, so no
  multicollinearity action was needed beyond the derivation already planned
  (M7 confirmed).
- *Q: Why check `Cycle(R/I)` unique values specifically?*
  A: Its source coding (2=Regular, 4=Irregular) is non-obvious; verifying it
  against `unique_values_categorical.txt` before mapping to 0/1 prevents a
  silent label-flip bug (task M5).

## src/02_clean_and_split.py
**Reads:** the raw xlsx. **Does:** column mapping, `pd.to_numeric` coercion,
binary Y/N → 0/1, implausible-value → NaN per `PLAUSIBLE_RANGES`, derives
`bmi`, `waist_hip_ratio`, `lh_fsh_ratio`, drops rows only where `pcos` is
missing, stratified 80/20 split. **Writes:** `data/processed/clean.csv`,
`train.csv`, `test.csv`, `models/input_schema.json`.

- *Q: Why is imputation not done here?*
  A: To avoid leakage — imputation must be fit on training folds only, inside
  each model's pipeline (or discretization step for the BN), never on the
  full clean dataset up front.
- *Q: Why do implausible values become NaN instead of being dropped as rows?*
  A: Dropping rows would shrink an already-small dataset (541 patients) and
  bias it toward "unremarkable" patients; treating them as missing lets the
  per-fold imputer handle them the same way as genuinely missing data.
- *Q: What's in `input_schema.json` and why compute percentiles on train only?*
  A: Per-feature label, type, tier, and the 1st/50th/99th percentile — used by
  the Streamlit app to size number-input widgets. Computed on train only so
  the app's UI bounds don't leak test-set information.

## src/03_feature_analysis.py
**Reads:** `train.csv`. **Does:** median-imputes (for ranking only, not for
modeling), then scores every feature per tier by ANOVA F-test (numeric),
chi-square (binary), and mutual information. **Writes:**
`results/03_feature_analysis/statistical_ranking_{tier}.csv/.md`,
`top_features_{tier}.png`.

- *Q: Why is this exploratory only, not used to hard-select features for the models?*
  A: The classifiers (04-06) do their own `SelectKBest` inside cross-validation,
  so feature selection is tuned per fold, not fixed globally — this script's
  ranking is for reporting and for the Step-9 consensus table, not for
  filtering the pipelines.

## src/04_train_naive_bayes.py / 05_train_logistic_regression.py / 06_train_random_forest.py
**Reads:** `train.csv`. **Does:** for each tier × imbalance strategy
(none / class_weight / SMOTE-NC, model-dependent), runs nested CV
(`RepeatedStratifiedKFold` 5×3 outer, `GridSearchCV` + `StratifiedKFold(5)`
inner, `scoring="roc_auc"`) to estimate generalization, then refits
`GridSearchCV` on full train to get `best_estimator_`, wraps it in
`CalibratedClassifierCV` (sigmoid and isotonic) alongside the raw version,
computes 5-fold out-of-fold probabilities for each, and picks a screening
threshold (highest threshold with OOF recall ≥ 0.90).
**Writes:** `models/{tier}/{model}__{strategy}__{calibration}.joblib` (dict
with model, features, threshold, best_params, oof_brier, oof_auc);
`results/0{4,5,6}_{model}/nested_cv_results.csv/.md`,
`oof_calibration_results.csv/.md`, `best_params.json`,
`oof_calibration_curve_{tier}.png`.

- *Q: Why calibrate three ways (raw/sigmoid/isotonic) instead of picking one?*
  A: Calibration quality is an empirical question, not assumed — sigmoid
  (Platt) is a low-variance parametric fit, isotonic is flexible but can
  overfit on ~433 training rows; we compare OOF Brier across all three per
  tier and let Step 8 pick the actual winner instead of guessing up front.
- *Q: Why `SMOTENC` and not plain `SMOTE`?*
  A: The feature set mixes binary and continuous columns; `SMOTENC` is told
  which column indices are categorical so it doesn't synthesize
  fractional 0.37 values for a Yes/No feature.
- *Q: Why is the outer nested-CV loop separate from the final `GridSearchCV` refit?*
  A: The outer loop's job is only to *estimate* how well the tuning process
  generalizes (repeated stratified 5-fold, 15 folds total); the final refit
  on all of train is a separate, single `GridSearchCV` whose `best_estimator_`
  is what actually gets shipped and calibrated. Conflating them would leak
  the outer test folds into hyperparameter selection.
- *Q: Why recall ≥ 0.90 for the screening threshold instead of 0.5?*
  A: This is a screening tool — missing a true PCOS case (false negative) is
  costlier than a false alarm that a clinician later rules out, so the
  threshold is chosen to guarantee ≥90% sensitivity on OOF predictions.

## src/07_train_bayesian_network.py
**Reads:** `train.csv`. **Does:** discretizes 9 variables + target per
Section 7a train-only cutoffs (BMI Asian cutoffs, LH/FSH>2, follicle≥12, AMH
train tertiles), asserts the hand-specified DAG (11 edges) is acyclic, fits
CPDs with `BayesianEstimator`/BDeu (ESS=5), runs `VariableElimination` for
inference, computes 5-fold OOF metrics (refitting discretization+CPDs per
fold), fits a comparison structure via `HillClimbSearch`+`BicScore`, and
writes 3 illustrative partial-evidence examples.
**Writes:** `models/bayesian_network/bn_model.joblib` (+ discretization.json
+ per-tier thresholds), `results/07_bayesian_network/bn_structure.png`,
`bn_structure_learned.png`, `oof_results.csv`, `structure_comparison.md`,
`partial_evidence_examples.md`.

- *Q: Why can the BN answer queries when only some variables are known, but NB/LR/RF can't?*
  A: `VariableElimination` marginalizes over any variable not given as
  evidence — inference just uses whatever evidence dict is passed — whereas
  the sklearn pipelines expect a full fixed-length feature vector (missing
  values are imputed, not "queried away").
- *Q: How does the hand-specified DAG compare to the learned one?*
  A: Hand-specified: OOF AUC 0.870/0.879/0.933 (tier1/2/3), Brier
  0.131/0.129/0.108. Hill-Climb/BIC learned: AUC 0.867/0.876/0.938, Brier
  0.125/0.124/0.097 — comparable, slightly better calibrated at tier 3, but
  its edges (e.g. `weight_gain → bmi_cat`, `follicle_high → pcos` reversed
  from ours) are less clinically interpretable, which is why we keep the
  hand-specified DAG as primary and the learned one as a comparison, per
  `structure_comparison.md`.
- *Q: Why BDeu with ESS=5 rather than maximum likelihood estimation?*
  A: With ~433 training rows split across up to 3×2×2×2×2×2×2×3×2 discretized
  states, many joint-state counts would be zero or tiny; BDeu's Dirichlet
  prior (equivalent sample size 5) smooths those CPDs instead of assigning
  0/1 probabilities from sparse counts.

## src/08_evaluate.py
**Reads:** `train.csv` OOF results (for model selection) + `test.csv` (first
and only use of test data before this point). **Does:** for each tier, picks
the NB/LR/RF candidate with lowest OOF Brier (ties→ highest OOF AUC), then the
tier's `app_default` as the lowest-Brier model among NB/LR/RF/BN; computes
test-set ROC-AUC/PR-AUC/Brier/accuracy/precision/recall/specificity/F1 at both
0.5 and the screening threshold, with 1000-resample bootstrap CIs for
AUC/Brier/recall; also evaluates each model's raw (uncalibrated) version.
**Writes:** `results/selected_models.json`,
`results/08_evaluate/test_results_all.csv/.md`, `tier_comparison.png`,
`tier_comparison_brier.png`, `roc_curves_/pr_curves_/calibration_curves_/
calibration_raw_vs_calibrated_{tier}.png`, `confusion_matrix_*.png`,
`literature_comparison.md`.

- *Q: What's the headline result?*
  A: Tier 3 test ROC-AUC 0.94-0.95 across all 4 models (best: NB isotonic
  0.9547), dropping to Tier 2 ~0.89-0.90 and Tier 1 ~0.88-0.90 — a clear,
  quantified cost of removing ultrasound/lab features, our first
  contribution.
- *Q: Is Tier 3's ~90-92% accuracy suspiciously high, and what limits how we read it?*
  A: It's in the literature's reported range (85-93.5%, `literature_comparison.md`
  confirms no >5pp deviation flag), but Tier 3 includes `follicle_left/right`,
  which are themselves Rotterdam diagnostic-criterion features used to assign
  the label — this is the label-circularity issue (contribution 4), so Tier 3
  numbers are best read as an upper bound, not a fair real-world estimate.
- *Q: At the screening threshold, what trade-off do you see vs 0.5?*
  A: E.g. Tier 3 LR: recall rises from 0.833 (@0.5) to ~0.86-0.92 depending on
  model, at the cost of precision/specificity — exactly the intended
  screening trade-off (catch more true cases, accept more follow-up workups).
- *Q: Why evaluate the raw/uncalibrated version too?*
  A: To demonstrate contribution 2 concretely — showing calibration actually
  changes Brier score/reliability, not just asserting it does.

## src/09_explain.py
**Reads:** `train.csv` (LR background) + `test.csv`. **Does:** for the
selected RF (TreeExplainer) and LR (LinearExplainer) per tier, computes SHAP
values on pipeline-transformed features; for NB, computes a simple
distance-of-means/pooled-std evidence proxy; also builds a per-tier
"consensus" ranking (MI + ANOVA/chi2 + RF impurity + RF |SHAP| + LR |SHAP|,
averaged rank, `consensus_top10` flag). **Writes:**
`results/09_explain/shap_summary_/shap_bar_{tier}_{model}.png`,
`shap_waterfall_example_{tier2,tier3}_{tp,fn,tn}.png`,
`feature_importance_consensus_{tier}.csv/.md`,
`nb_feature_evidence_{tier}.csv`.

- *Q: Why explain the base fitted pipeline rather than the calibrated wrapper?*
  A: `CalibratedClassifierCV` only rescales probabilities post-hoc (Platt/
  isotonic); the feature→prediction relationship SHAP explains lives in the
  underlying RF/LR estimator, so explaining the base estimator is both
  correct and simpler (no need to unwrap calibration folds for attribution).
- *Q: Why TreeExplainer for RF but LinearExplainer for LR?*
  A: TreeExplainer computes exact SHAP values efficiently for tree ensembles
  using the tree structure; LR's decision function is already linear in the
  (scaled) inputs, so LinearExplainer with a training background gives exact
  values without the tree-specific machinery.
- *Q: What's the NB evidence proxy and why is it needed?*
  A: `|mean_pcos - mean_no_pcos| / pooled_std` per feature — Gaussian NB has
  no native SHAP support in this pipeline shape, so this standardized
  mean-difference is used as a simple, honest interpretability stand-in (also
  what the app falls back to for NB).
- *Q: This was just re-run after the TSH/PRL range widening — did anything change?*
  A: The underlying models (04-08) were already retrained under the new
  ranges; this re-run refreshes the SHAP/consensus/NB-evidence outputs so
  they're computed against the *current* models and test split rather than
  the stale pre-widening artifacts.

## src/10_sensitivity_analysis.py
**Reads:** `train.csv` + `test.csv`. **Does:** using Tier 2 LR/RF (retuned
with the same grids), runs three robustness checks: (1) median vs KNN(k=5) vs
IterativeImputer; (2) all Tier 2 features vs consensus-top-10 (from Step 9)
vs ANOVA-top-10; (3) Tier 3 full vs Tier 3 minus the four Rotterdam-criterion
features (`follicle_left/right`, `cycle_irregular`, `hair_growth`).
**Writes:** `results/10_sensitivity_analysis/sensitivity_results.csv/.md` +
one bar chart per analysis.

- *Q: What did the imputation check show?*
  A: Negligible difference — LR AUC 0.878/0.879/0.874 and RF 0.892/0.885/0.881
  across median/KNN/iterative — the results aren't an artifact of the
  imputation method chosen.
- *Q: What did the circularity check show, and why does it matter?*
  A: Removing the 4 Rotterdam-criterion features drops LR AUC from 0.942 to
  0.833 and RF from 0.945 to 0.858 (Brier roughly doubles too) — a large,
  quantified drop that directly supports contribution 4: a meaningful part of
  Tier 3's apparent performance comes from features that are themselves part
  of how the label was assigned, not independent predictive signal.
- *Q: What did the feature-subset check show?*
  A: Consensus-top-10 and ANOVA-top-10 match or slightly beat using all Tier 2
  features (e.g. LR: 0.878→0.895/0.896 AUC) — a small feature set loses
  essentially nothing, supporting that a leaner, cheaper input set is viable.

## app/ (Home.py, app_utils.py, pages/1-4)
**Reads:** `results/selected_models.json`, `models/input_schema.json`, the
`.joblib` model files, `results/08_evaluate/test_results_all.csv`,
`results/09_explain/*` for display. **Does:** Streamlit frontend+backend (no
separate API, no training in-app) — Tier/model-selectable risk screening with
SHAP/NB-proxy explanations and risk banding; a Partial-Information BN page
that accepts partial evidence and shows "what would change the estimate" for
each unknown variable; a Model Comparison page reusing Step 8/9 artifacts; an
About/Limitations page. **Writes:** nothing (stateless inference only).

- *Q: Why can leaving every field at default never crash the app?*
  A: Numeric widgets default to the train median (from `input_schema.json`)
  and bounds are the train 1st-99th percentile widened 20%; binary widgets
  default to a valid state (or "Unknown" on the BN page) — every default is a
  valid model input, and the BN page explicitly supports zero evidence
  (returns the population prior).
- *Q: Two real bugs were fixed during app testing — what were they and why did they happen?*
  A: (1) `CalibratedClassifierCV`'s wrapped base estimator isn't independently
  "fitted" in the sklearn sense SHAP's TreeExplainer/LinearExplainer check for,
  so explaining it directly raised an unfitted-estimator error — fixed by
  explaining the underlying refit base estimator instead. (2) the BN
  discretizer assumed every tier's evidence columns were present in the input
  row; Tier 1 requests (no ultrasound) were missing `follicle_left/right`,
  causing a `KeyError` — fixed by only discretizing/looking up columns that
  are actually present for the requested tier's evidence set.

---
*Generated for M15 (viva prep). If asked "why does X differ slightly from
what's in the report," check whether the report figure predates the most
recent `run_all.sh`/re-run — regenerate from `results/` if so.*
