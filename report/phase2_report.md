# Phase 2 Report

**Course:** Machine Learning II (BAI702), CMRIT — Assignment 1
**Team:** Prakshi Lakhchaura (1CR23AI083), Sannah Sharma (1CR23AI110)

**Note on references:** citations in this report point to `report/literature_research_notes.md`, which holds the full bibliography, per-claim sourcing, and verification status (§1 Bayesian Network edge citations, §2 clinical threshold citations, §3 novelty/positioning search, §5 full reference list). This report does not duplicate that bibliography.

---

## 1. Problem Statement and Solution Overview

Polycystic Ovary Syndrome (PCOS) is a common but inconsistently diagnosed endocrine disorder, with an estimated 70% of affected women undiagnosed globally, disproportionately so in low-resource settings where specialist access (endocrinologists, sonographers) is limited. Diagnosis follows the Rotterdam criteria — two of three features (oligo/anovulation, hyperandrogenism, polycystic ovarian morphology) — which is inherently a probabilistic, partial-evidence judgment rather than a fixed checklist, and expensive confirmatory tests (hormonal panels, ultrasound) are often not the first thing available to a patient or primary-care provider.

This project builds a probabilistic machine-learning decision-support system that estimates PCOS risk from a woman's clinical, biochemical, and ultrasound data, and — critically — reports how that risk estimate degrades as fewer, cheaper inputs are available. The solution has four parts, built end to end on the Kaggle PCOS dataset (Kottarathil; 541 patients, 10 hospitals in Kerala):

1. **Three nested cost tiers.** The same model families are trained on Tier 1 (questionnaire + vitals only), Tier 2 (+ blood tests), and Tier 3 (+ ultrasound), so the performance cost of a cheaper screening tool is quantified rather than assumed.
2. **Four model families per tier** — Gaussian Naive Bayes, Logistic Regression, Random Forest, and a hand-specified Bayesian Network — evaluated with nested cross-validation, nine calibrated variants each (three imbalance strategies × three calibration methods for NB/LR/RF), and a held-out test set used exactly once.
3. **A Bayesian Network for incomplete-evidence inference.** Unlike the other three models, the BN can produce P(PCOS | evidence) from whatever subset of fields is actually known, which the other models cannot do without imputing every missing field first.
4. **A Streamlit decision-support prototype** exposing all of this — tiered risk screening, partial-evidence BN inference, model comparison, and documented limitations — as a working, if academic, tool.

The rest of this report covers what was actually built and found, in the order the pipeline runs.

---

## 2. Data and Preprocessing, as Executed

**Source.** The Kottarathil Kaggle file (`PCOS_data_without_infertility.xlsx`, sheet `Full_new`), 541 patients, no supplementary datasets — the UCI/IEEE DataPort supplementation considered in Phase 1 was not pursued.

**Class balance.** 364 PCOS-negative, 177 PCOS-positive (32.7% positive) — the "roughly one-third" imbalance anticipated in Phase 1 held exactly.

**Cleaning rules** (`src/02_clean_and_split.py`):
- All non-target columns coerced to numeric; binary Y/N columns mapped to 0/1, anything else to missing.
- Implausible values set to missing (not dropped) using per-feature clinical ranges defined in `config.PLAUSIBLE_RANGES` — e.g. age 12–60, BMI 10–60, FSH/LH 0–200, AMH 0–70, follicle counts 0–40 — rather than a generic statistical (IQR/z-score) rule.
- BMI and waist:hip ratio are recomputed from weight/height and waist/hip where those are available (the source columns are rounded), then the helper columns (weight, height, waist, hip) are dropped.
- `lh_fsh_ratio` is derived as LH/FSH, replacing the source `FSH/LH` column (the two were correlated at r = 0.97 in the raw data, confirming the derived ratio is the right replacement, not a redundant addition).
- Rows are dropped only if the target itself is missing; no other row-level dropping.

**One dataset-specific quirk worth flagging explicitly:** the source `Cycle length(days)` column does not hold literal calendar-day counts — real values in the raw file range from 0 to 12 (median 5), not the 21–45+ days a menstrual cycle normally spans. This was kept as provided rather than reinterpreted, and is surfaced as a limitation (§11) and in the app's About page, since it's the kind of thing that looks like a bug in a demo if unexplained.

**Split.** A single stratified 80/20 split: 432 training patients (291 negative / 141 positive), 109 held out for testing (73 negative / 36 positive), `random_state=42`. All model selection happens via nested cross-validation on the training set only; the test set is touched exactly once, in the final evaluation (§6).

**Feature tiers** (`config.TIER_FEATURES`, nested):
- **Tier 1 (screening):** 19 features — age, BMI, waist:hip ratio, pulse/respiratory rate, blood pressure, cycle irregularity + length, marriage years, pregnancy, prior abortions, and 8 symptom/lifestyle indicators (weight gain, hair growth, skin darkening, hair loss, pimples, fast food, regular exercise).
- **Tier 2 (lab):** Tier 1 + 10 blood-test features (Hb, FSH, LH, LH/FSH ratio, TSH, AMH, prolactin, vitamin D3, progesterone, random blood sugar).
- **Tier 3 (full):** Tier 2 + 5 ultrasound features (follicle counts and sizes left/right, endometrium thickness).

---

## 3. Feature Analysis

Each tier's features were ranked three ways on the training set (median-imputed for this exploratory pass only — the actual model pipelines impute inside CV folds): ANOVA F-test (numeric features), chi-square (binary features), and mutual information (all features).

**Tier 1** top signals by mutual information: `weight_gain`, `skin_darkening`, `hair_growth`, `cycle_irregular`, `pimples`, `waist_hip_ratio`, `cycle_length`, `bmi` — all clinically expected PCOS indicators (Rotterdam hyperandrogenism/symptom signs), which is a useful validity check that the model is picking up real signal rather than noise.

**Tier 3** top signals: `follicle_right` and `follicle_left` dominate (ANOVA F ≈ 314 and 265, mutual information ranks 1–2 by a wide margin), followed by `hair_growth`, `skin_darkening`, `weight_gain`, `amh`, `cycle_length`. This is expected — follicle count is a direct Rotterdam diagnostic component — and is exactly the signal that drives the label-circularity discussion in §8 and §10.

**Multicollinearity check:** the only high-correlation pairs found (|r| > 0.8) were among *helper* columns that get dropped before modeling anyway — `Sl. No`/`Patient File No.` (identifiers, r = 1.0), `FSH` vs. the source `FSH/LH` (r = 0.97, replaced by the derived ratio), `Weight` vs. `BMI` (r = 0.90, weight dropped after BMI is computed), `Hip` vs. `Waist` (r = 0.87, both dropped after the ratio is computed). No high collinearity remains among the final modeling features.

---

## 4. Model Development and Nested Cross-Validation Results

**Pipeline** (`src/utils/pipelines.py`), per tier × strategy: median/most-frequent imputation (numeric/binary) → SMOTE-NC resampling if the strategy calls for it → `StandardScaler` on numeric features (NB/LR only; Random Forest skips scaling) → `SelectKBest` feature selection (NB/LR only, k ∈ {8, 12, 16, all}) → estimator. All steps are fit inside each cross-validation training fold only.

**Imbalance strategies compared per model** (not chosen upfront — all trained and compared): Naive Bayes {none, SMOTE}; Logistic Regression and Random Forest {none, class-weighted, SMOTE-NC}.

**Hyperparameter grids:** GaussianNB `var_smoothing` ∈ logspace(-12, -3, 10); Logistic Regression `C` ∈ {0.01, 0.1, 1, 10} (L2, lbfgs, max_iter=5000); Random Forest `n_estimators`=300, `max_depth` ∈ {None, 4, 8}, `min_samples_leaf` ∈ {1, 3, 5}, `max_features` ∈ {sqrt, 0.5}. Tuned via inner `StratifiedKFold(5)` grid search, scored on ROC-AUC.

**Nested-CV performance estimate** (outer `RepeatedStratifiedKFold`, 5 splits × 3 repeats), best strategy per tier:

| Tier | Model | Strategy | ROC-AUC | PR-AUC | Brier | Recall |
|---|---|---|---:|---:|---:|---:|
| Tier 1 | Naive Bayes | none | 0.875 ± 0.027 | 0.810 | 0.146 | 0.785 |
| Tier 1 | Logistic Regression | smote | 0.874 ± 0.032 | 0.813 | 0.131 | 0.768 |
| Tier 1 | Random Forest | class_weight | 0.881 ± 0.032 | 0.813 | 0.133 | 0.763 |
| Tier 2 | Naive Bayes | none | 0.876 ± 0.031 | 0.807 | 0.142 | 0.773 |
| Tier 2 | Logistic Regression | class_weight | 0.874 ± 0.029 | 0.812 | 0.138 | 0.780 |
| Tier 2 | Random Forest | class_weight | 0.884 ± 0.033 | 0.812 | 0.135 | 0.733 |
| Tier 3 | Naive Bayes | none | 0.955 ± 0.018 | 0.917 | 0.091 | 0.872 |
| Tier 3 | Logistic Regression | none | 0.951 ± 0.020 | 0.920 | 0.077 | 0.811 |
| Tier 3 | Random Forest | none | 0.956 ± 0.020 | 0.926 | 0.090 | 0.775 |

The headline pattern anticipated in Phase 1 held: **Tier 3 clearly outperforms Tier 1/2** (ROC-AUC jumps from the high-0.87s to the mid-0.95s the moment ultrasound features enter), while Tier 1 and Tier 2 are close to each other — blood tests alone add relatively little over the screening-only tier, which is itself a useful, non-obvious finding (it suggests the biggest single information gain is ultrasound, not labs).

**Calibration.** Out-of-fold Brier scores after recalibration (sigmoid/isotonic) improved over raw probabilities for every model/tier, confirming raw probabilities — especially Naive Bayes' — needed correction rather than being usable as-is. For example, Tier 1 Naive Bayes' OOF Brier improved from 0.147 (raw) to 0.120 (isotonic); Tier 3 Naive Bayes from 0.096 to 0.079. The final per-tier model selection (§6) picks the calibration variant with the lowest OOF Brier, not necessarily isotonic or sigmoid uniformly — both win in different tier/model combinations, which is itself evidence that neither recalibration method is universally better on this dataset size.

---

## 5. Bayesian Network

**Structure.** A hand-specified, clinically motivated DAG over 10 discretized variables (`config.BN_STRUCTURE`): BMI category (Asian cutoffs, <23/23–27.5/>27.5) and PCOS status are parents of the symptom/lab nodes (cycle irregularity, hair growth, pimples, skin darkening, weight gain, high LH:FSH, high follicle count), with follicle count additionally feeding AMH level (small-follicle AMH production). Edge-by-edge clinical justification and citations are in `report/literature_research_notes.md` §1. Parameters were fit via Bayesian estimation with a BDeu prior (equivalent sample size 5); inference uses variable elimination.

**Structure comparison.** A Hill-Climb/BIC-learned structure was fit on the same variables as a comparison point (not the primary structure). It performed marginally better on Tier 3 (OOF ROC-AUC 0.938 vs. 0.933 for the hand-specified DAG; Brier 0.097 vs. 0.108) and was essentially tied on Tiers 1–2. More interesting than the small performance gap is that the learned structure inverts some clinically expected directions — e.g. it makes `skin_darkening` a parent of `pcos` rather than a consequence of it — which is a useful discussion point: a purely data-driven structure doesn't necessarily respect known clinical causality, reinforcing the choice to hand-specify the DAG for interpretability even at a small performance cost.

**Partial-evidence inference — the core BN demonstration.** Given only BMI (obese) and irregular cycles, the BN estimates P(PCOS) = 0.758. Adding high LH:FSH raises this to 0.888. Adding a high follicle count raises it further to 0.992 — a clean, monotonic illustration of the BN updating its estimate as more evidence becomes available, which none of the other three models can do without first imputing every missing field.

---

## 6. Final Evaluation on the Held-Out Test Set

Per-tier model selection uses training-set OOF Brier only (never the test set): the app-default model per tier is whichever of NB/LR/RF/BN has the lowest OOF Brier. This selected **Naive Bayes for Tier 1**, and **Logistic Regression for both Tier 2 and Tier 3**.

**Test-set results** (n = 109; 73 negative, 36 positive), at the standard 0.5 threshold and at each model's recall-targeted screening threshold (highest threshold achieving OOF recall ≥ 0.90 on training):

| Tier | Model | ROC-AUC | Brier | Acc @0.5 | Recall @0.5 | Acc @screening | Recall @screening |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Naive Bayes (app default) | 0.879 | 0.111 | 0.872 | 0.722 | 0.670 | 0.861 |
| 1 | Logistic Regression | 0.892 | 0.114 | 0.881 | 0.750 | 0.725 | 0.889 |
| 1 | Random Forest | 0.898 | 0.112 | 0.862 | 0.722 | 0.706 | 0.917 |
| 1 | Bayesian Network | 0.885 | 0.116 | 0.862 | 0.778 | 0.642 | 0.889 |
| 2 | Naive Bayes | 0.893 | 0.114 | 0.862 | 0.750 | 0.697 | 0.917 |
| 2 | Logistic Regression (app default) | 0.891 | 0.115 | 0.853 | 0.722 | 0.761 | 0.889 |
| 2 | Random Forest | 0.893 | 0.120 | 0.853 | 0.667 | 0.642 | 0.917 |
| 2 | Bayesian Network | 0.903 | 0.115 | 0.853 | 0.806 | 0.752 | 0.889 |
| 3 | Naive Bayes | 0.955 | 0.062 | 0.917 | 0.833 | 0.872 | 0.917 |
| 3 | Logistic Regression (app default) | 0.951 | 0.057 | 0.927 | 0.833 | 0.908 | 0.861 |
| 3 | Random Forest | 0.943 | 0.071 | 0.908 | 0.778 | 0.899 | 0.889 |
| 3 | Bayesian Network | 0.952 | 0.073 | 0.917 | 0.861 | 0.844 | 0.917 |

**Worth noting explicitly:** the screening threshold targets OOF recall ≥ 0.90 on the *training* set, but on the small held-out test set (n=109, 36 positive) achieved recall sometimes lands just under 0.90 (e.g. Logistic Regression Tier 3: 0.861 test recall) — expected sampling variability at this sample size, not a bug in the threshold logic, and worth reporting with the 95% bootstrap CIs (computed with 1000 stratified resamples, seed fixed) rather than the point estimate alone.

**Comparison against literature** (`results/08_evaluate/literature_comparison.md`): our Tier 3 Random Forest reaches 89.9% test accuracy and ROC-AUC 0.943, squarely inside the 85–93% range reported by prior Kaggle-dataset studies (Aggarwal et al. 2023: 93.52%; various: 85–93%) and clearly below the ~99% figures some later studies report with aggressive feature selection — which Phase 1 already flagged as a likely overfitting artifact on this small, single-region dataset. Our result landing inside the plausible range, not at the inflated end, is itself a small piece of evidence against overfitting in our own pipeline.

---

## 7. Interpretability

SHAP (`TreeExplainer` for Random Forest, `LinearExplainer` for Logistic Regression) was computed per tier on the pipeline-transformed test features. Naive Bayes gets a simpler evidence proxy (standardized class-mean difference per feature) instead of SHAP, since SHAP doesn't map as cleanly onto NB's generative structure; the Bayesian Network's "explanation" is its posterior-probability trace over evidence (§5).

**Consensus feature importance** (`results/09_explain/feature_importance_consensus_tier*.md`) cross-checks mutual information, ANOVA/chi-square rank, RF impurity importance, and LR |SHAP| rank per feature. The Tier 1 consensus top 10 — skin darkening, weight gain, hair growth, cycle irregularity, pimples, fast food, cycle length, BMI, age, waist:hip ratio — is entirely clinically plausible symptom/lifestyle signal, with no surprises. The Tier 3 consensus top 10 is dominated by the two follicle-count features (ranked 1–2 by every method), followed by the same symptom cluster plus AMH — directly visualizing the label-circularity concern explored next.

---

## 8. Sensitivity Analysis and the Label-Circularity Audit

Run on Tier 2, Logistic Regression and Random Forest, retuned with the same grids:

**Imputation sensitivity** (median vs. KNN vs. iterative): negligible difference for either model (LR ROC-AUC 0.874–0.879 across all three; RF 0.881–0.892, with median actually best for RF). The pipeline's default median imputation is not costing meaningful performance.

**Feature-subset sensitivity** (all Tier 2 features vs. consensus top-10 vs. ANOVA top-10): both reduced 10-feature subsets matched or *beat* the full Tier 2 feature set (LR: 0.878 all vs. 0.895 consensus-top10 vs. 0.896 ANOVA-top10; RF: 0.892 all vs. 0.890 consensus vs. **0.914** ANOVA-top10) — a genuinely useful finding for a low-resource screening tool: a carefully chosen 10-feature subset loses nothing, and sometimes gains, versus the full 24-feature Tier 2 set.

**Label-circularity check — the empirical core of contribution #4.** Removing the Rotterdam-criterion features (follicle counts, cycle irregularity, hair growth) from Tier 3 drops performance sharply: Logistic Regression ROC-AUC falls from 0.942 to 0.833 (a 10.9-point drop) and Brier roughly doubles, from 0.079 to 0.151; Random Forest falls from 0.945 to 0.858 (8.8 points), Brier from 0.096 to 0.155. This directly quantifies how much of Tier 3's apparent performance comes from features that partially *define* the diagnostic label itself, rather than from independent biological signal — the concrete evidence behind the circularity limitation stated in Phase 1 §6 and the About page.

---

## 9. Decision-Support Prototype (the App)

A four-page Streamlit application (`streamlit run app/Home.py`), loading serialized models and running inference locally with no separate backend:

1. **Risk Screening** — pick a tier and model (NB/LR/RF, defaulting to the tier's lowest-OOF-Brier model), fill in that tier's inputs, and get a risk percentage, colour-coded risk band, the top-3 SHAP-driven factors in plain language, and the screening threshold used. Three quick-fill example profiles (PCOS-positive, non-PCOS, borderline) pre-populate realistic inputs for fast manual testing, and entered values persist across tier/page switches (a session-state bug found and fixed during manual walkthrough testing — Streamlit otherwise resets widget state on every rerun that doesn't re-instantiate a given widget).
2. **Partial-Information Bayesian Inference** — every BN input defaults to "Unknown"; the BN infers P(PCOS) from whichever subset is actually filled in, plus a "what would change the estimate" table showing how the estimate would shift under each possible value of each still-unknown variable.
3. **Model Comparison** — the test-set results table, ROC/calibration/confusion-matrix figures per tier, the tier-comparison headline chart, and SHAP summaries.
4. **About & Limitations** — dataset provenance, tier definitions, the label-circularity note, generalizability limits, the `cycle_length` data-scale quirk (§2), and an explicit "not clinically validated" disclaimer shown on every page.

---

## 10. Revisiting the Four Contributions

1. **Cost-tiered comparison.** Quantified directly: Tier 3 reaches ROC-AUC ≈ 0.95–0.96 across models, Tier 1/2 plateau around 0.87–0.88 — a clearly measured, non-trivial cost of restricting to cheaper inputs, extending prior same-dataset feature-subset work (`literature_research_notes.md` §3, contribution 1) by evaluating a clinically ordered nested hierarchy rather than an unordered feature-group comparison.
2. **Calibration as a primary metric.** Every model's raw probabilities were measurably improved by recalibration (§4); calibration quality, not just accuracy, drove the final per-tier model selection (§6) — extending prior work that reports Brier scores as a secondary metric by making it the selection criterion itself.
3. **Bayesian Network under incomplete evidence.** The monotonic 0.758 → 0.888 → 0.992 partial-evidence trace (§5) is the concrete demonstration that no other model in this project can replicate without imputation.
4. **Label-circularity audit.** The 8.8–10.9 ROC-AUC-point drop when Rotterdam-criterion features are removed from Tier 3 (§8) is a directly measured circularity effect, not an assumed or qualitative one.

See `literature_research_notes.md` §3 for the full novelty-positioning discussion, including which prior papers come closest to each contribution and the reworded, defensible novelty claims to use in the write-up.

---

## 11. Limitations

- **Generalizability.** Single-population dataset (Kerala, India; 541 patients, 10 hospitals); conclusions should not be assumed to transfer to other populations without external validation.
- **Not clinically validated.** An academic prototype; no clinical trial, no regulatory review. Shown as a persistent disclaimer in the app.
- **Label circularity.** Tier 3 performance is inflated by features that are themselves Rotterdam diagnostic components — quantified in §8, not just asserted.
- **LH:FSH > 2 is not a validated diagnostic cutoff.** Included as a historically used, exploratory feature; no primary source establishes it as evidence-based, and at least one comparative study found it has little diagnostic power. Full citation-backed wording: `literature_research_notes.md` §2.
- **The `cycle_length` data-scale quirk** (§2): the source column's real range (0–12) doesn't match literal calendar days; the app surfaces the data as-is with an explanatory note rather than silently reinterpreting it.
- **Small test set (n=109).** Reported with 95% bootstrap CIs where possible; single-point test metrics (especially recall at the screening threshold) can swing several points on a set this size, as seen in §6.

---

## 12. Conclusion

The project delivers what Phase 1 set out to build — a tiered, calibrated, interpretable PCOS risk-screening system, plus a Bayesian Network that reasons under incomplete evidence — and, in executing it, surfaced concrete, measured versions of claims Phase 1 could only anticipate: the actual Tier 3→Tier 1 performance cost, the actual calibration gap Naive Bayes needed corrected, and the actual magnitude of label circularity. The Streamlit prototype makes all four tiers/models and the BN's partial-evidence reasoning directly explorable, with the explicit caveats (§11) needed to present it honestly as a screening aid, not a diagnostic tool.

---

## 13. References

See `report/literature_research_notes.md` for the full bibliography and per-claim sourcing: §1 (Bayesian Network edge justifications), §2 (clinical threshold citations — BMI, LH:FSH, follicle count), §3 (novelty/positioning literature search), §5 (full reference list).
