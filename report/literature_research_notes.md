# Literature Research Notes

Consolidated record of the three literature searches run for this project
(M8, M9, M12 in `PCOS_PROJECT_PLAN.md`, Section 7). Use this file as the
single source of truth while writing the Phase 2 report — pull citations
and wording directly from here rather than re-deriving them.

## Legend

- ✅ **Solid** — real, named venue/authors; safe to cite as-is.
- ⚠️ **Verify before submission** — plausible but has incomplete metadata,
  is very recent, or comes from a non-peer-reviewed host. Confirm the paper
  exists and says what's claimed before it goes in the report.
- ❌ **Reframe, don't cite as support** — evidence contradicts or weakens
  the claim it was originally sought to support; use it to *soften* a claim
  instead.

---

## 1. Bayesian Network edge citations (M8)

Covers all 11 edges in `config.BN_STRUCTURE` (`src/config.py`).

| Edge | Citation | One-line justification | Status |
|---|---|---|---|
| `bmi_cat → pcos` | Barber & Franks, "Obesity and polycystic ovary syndrome," *Clinical Endocrinology*, 2021 | Obesity/weight gain can clinically unmask PCOS in genetically susceptible women via worsened insulin resistance and ovarian androgen production. | ✅ |
| `bmi_cat → skin_darkening` | Ahmed et al., *PLOS ONE*, 2020 | Acanthosis nigricans is positively associated with BMI/insulin resistance in mediation analyses. | ✅ |
| `pcos → cycle_irregular` | Rotterdam ESHRE/ASRM Consensus Workshop Group, *Human Reproduction* 19(1):41–47, 2004 | Oligo/anovulation (irregular cycles) is one of the three defining Rotterdam PCOS features. | ✅ (spot-check) |
| `pcos → hair_growth` | Archer et al., "Hirsutism and acne in polycystic ovary syndrome," *Best Practice & Research Clinical Obstetrics & Gynaecology*, 2004 | Ovarian/adrenal hyperandrogenism in PCOS drives androgen-dependent hirsutism. | ✅ |
| `pcos → pimples` | Same as above (Archer et al., 2004) | Hyperandrogenism stimulates sebaceous glands, causing acne; less PCOS-specific than hirsutism. | ✅ |
| `pcos → skin_darkening` | S.R.K. et al., "Acanthosis Nigricans: Pointer of Endocrine Entities," *Indian J. Endocrinol. Metab.*, 2022 | Acanthosis nigricans is a marker of hyperinsulinemia, common in PCOS especially with obesity. | ✅ |
| `pcos → weight_gain` | Barber & Franks, "Why are women with PCOS obese?," *British Medical Bulletin*, 2022 | Bidirectional: PCOS-driven insulin resistance/hyperandrogenism promotes weight gain, which in turn worsens PCOS expression. | ✅ |
| `pcos → lh_fsh_high` | Review, *Endocrinology, Diabetes & Metabolism*, 2025 (PMC) | Increased GnRH pulse frequency in PCOS causes preferential LH secretion, raising LH:FSH. | ✅ (see §2 for the specific >2 cutoff caveat) |
| `pcos → follicle_high` | Rotterdam Consensus, 2004 (same as above) | Polycystic ovarian morphology (high antral follicle count) is one of the three Rotterdam domains. | ✅ (spot-check) |
| `follicle_high → amh_level` | Pigny, Jonard, Robert & Dewailly, *JCEM*, 2006 | Serum AMH is strongly related to antral follicle count; proposed as a surrogate for it. | ✅ (spot-check) |
| `pcos → amh_level` | Dewailly et al., *Reproductive BioMedicine Online*, 2016 | AMH is ~2–3x elevated in PCOS from both more small follicles and higher per-follicle granulosa-cell AMH output. | ✅ |

**Spot-check before final submission:** Rotterdam 2003/2004 consensus and the Pigny et al. 2006 AMH/AFC paper — both are cited twice above and are foundational, so confirm exact page numbers/DOI.

**Important nuance to carry into the report:** several of these edges are not one-directional in reality (BMI↔PCOS, PCOS↔weight gain). Frame them in the report as "the dependency the BN encodes," not as proven unidirectional causation — this is exactly the kind of honest scoping a reviewer will want to see.

---

## 2. Threshold citations (M9)

**Verified via a live Perplexity search on 2026-09-29** (see prompt and full answer archived in the project chat log). All three DOIs below were confirmed at that time; the BMI and LH:FSH framing were corrected as a result.

### Asian BMI cutoffs (23 / 27.5) — `config.BMI_CUTS` — ✅ Solid citation, reframed

- **Source:** WHO Expert Consultation, "Appropriate body-mass index for Asian populations and its implications for policy and intervention strategies," *The Lancet* 363(9403):157–163, 2004. DOI: **10.1016/S0140-6736(03)15268-3**.
- **What it actually says:** BMI associated with "observed risk" varied ~22–25 kg/m² across the Asian populations studied, and "high risk" varied ~26–31 kg/m². The consultation identified **23.0, 27.5, 32.5, and 37.5 kg/m²** as *additional public-health action points* — it explicitly did **not** establish one universal cutoff for all Asian populations, and it retained the standard WHO international classifications (25/30) alongside these action points.
- **Correction vs. earlier draft:** this is not simply "the Asian BMI classification" — it's WHO-identified risk-based action points layered on top of the standard classification, and the consultation itself cautioned against treating it as one universal cutoff.
- **Report wording (use near-verbatim):** "We used 23.0 and 27.5 kg/m² as Asian-population cardiometabolic-risk action points, following the 2004 WHO Expert Consultation; these are risk-oriented action points, not a universally accepted replacement for the standard WHO overweight/obesity classifications. The operational 3-category split in our pipeline (normal <23, overweight 23–27.4, obese ≥27.5) implements these action points directly."
- **Status:** ✅ Solid and DOI-verified.

### LH:FSH > 2 — `config.LH_FSH_HIGH_CUTOFF` — ⚠️ Confirmed: no validated diagnostic cutoff exists

- **No primary authoritative source for ">2" could be found** — confirmed by live search, not just absence from our own prior notes. The figure only appears in secondary clinical summaries.
- **Secondary source that repeats ">2":** Dokras et al., "Diagnosis and Treatment of Polycystic Ovary Syndrome," *American Family Physician*, 2016 — states ratio >2 "generally indicates PCOS" but immediately qualifies "there are no exact cutoff values because many different assays are used."
- **The Rotterdam consensus itself does not include this ratio:** Rotterdam ESHRE/ASRM-Sponsored PCOS Consensus Workshop Group, *Human Reproduction* 19(1):41–47, 2004, DOI **10.1093/humrep/deh098** — notes LH is frequently elevated in PCOS but states explicitly that LH measurement is not necessary for diagnosis; the 2-of-3 diagnostic rule (oligo/anovulation, hyperandrogenism, polycystic ovaries) does not include LH:FSH at all.
- **Direct contradicting evidence:** Cho, Jayagopal, Kilpatrick, Holding & Atkin, "The LH/FSH ratio has little use in diagnosing polycystic ovarian syndrome," *Annals of Clinical Biochemistry* 43(3):217–219, 2006, DOI **10.1258/000456306776865188** — cycle-long repeated sampling, 12 PCOS vs. 11 matched controls: median ratio 1.6 (PCOS) vs 1.2 (controls), **not significant** (p=0.14); only 7.6% of PCOS samples exceeded ratio 3, vs. 15.6% of control samples.
- **On the ">3" alternative:** an early-literature ">3" threshold exists (per a PMC lab-medicine review) but is no better supported — Cho et al. tested exactly this cutoff and found it *more* common in controls than in PCOS.
- **Report wording (use near-verbatim in Limitations):** "An LH:FSH ratio >2 was included as a historically used, exploratory biochemical feature, not a validated diagnostic criterion — it is absent from the Rotterdam consensus and from the 2023 international guideline, no primary source establishes >2 as an evidence-based cutoff, and at least one cycle-long comparative study found the ratio has little diagnostic power (Cho et al., 2006)."
- **Action:** Keep the feature/threshold in the model (still informative to some degree and matches how the source dataset appears to have been characterized), but describe it in the report strictly as a dataset-specific operational threshold — never as an accepted clinical criterion.
- **Status:** ⚠️ downgraded from earlier draft — this is not merely "contested," it is confirmed unsupported by any primary diagnostic source.

### Follicle count ≥12 vs ≥20 — `config.FOLLICLE_HIGH_CUTOFF` — ✅ Both cutoffs now fully sourced with DOIs

- **Rotterdam-2003 threshold (what the code uses, =12):** Rotterdam ESHRE/ASRM-Sponsored PCOS Consensus Workshop Group, *Human Reproduction* 19(1):41–47, 2004, DOI **10.1093/humrep/deh098**. Exact wording: "Presence of 12 or more follicles in each ovary measuring 2–9 mm in diameter, and/or increased ovarian volume (>10 ml)" — one ovary meeting the definition is sufficient.
- **2023 guideline revision (=20):** International PCOS Network (Teede, Tay, Laven, Dokras, Moran, Piltonen, Costello, Boivin, Redman, Boyle, Norman, Mousa, Joham), "Recommendations from the 2023 International Evidence-based Guideline for the Assessment and Management of Polycystic Ovary Syndrome" — published concurrently in three journals (cite whichever fits your reference style):
  - *Human Reproduction* 38(9):1655–1679, 2023, DOI **10.1093/humrep/dead156**
  - *Journal of Clinical Endocrinology & Metabolism* 108(10):2447–2469, 2023, DOI **10.1210/clinem/dgad463** (the one already in our reference list, §5)
  - *Fertility and Sterility* 120(4):767–793, 2023, DOI **10.1016/j.fertnstert.2023.07.025**

  Exact wording: "Follicle number per ovary (FNPO) ≥20 in at least one ovary should be considered the threshold for PCOM in adults."
- **Why it changed:** reflects improved ultrasound resolution over the ~20 years between the two guidelines, not a redefinition of the underlying biology.
- **Report wording (Methods):** "We use the Rotterdam-2003 ovarian-morphology threshold (≥12 follicles per ovary, either ovary sufficient) rather than the 2023 revision (≥20 in at least one ovary), since the Kottarathil dataset's PCOS labels were assigned using contemporaneous (older) ultrasound criteria; the higher 2023 threshold reflects higher-resolution modern equipment and would be inconsistent with how the ground-truth labels were originally derived."
- **Report wording (Limitations):** note explicitly that this cutoff is dataset-era- and equipment-specific, and would need to be recalibrated to ≥20 for application to patients scanned with modern high-resolution transvaginal ultrasound.
- **Status:** ✅ Both citations DOI-verified; no change needed from earlier draft beyond adding DOIs.

---

## 3. Novelty / positioning search (M12)

Overall verdict from the search: **contribution 1 (cost tiers) is largely anticipated by a same-dataset paper; contribution 2 (calibration) is already established practice; contributions 3 (BN) and 4 (circularity) are partially anticipated.** Your defensible novelty is the *combination* of all four, evaluated together on this dataset — not any single piece in isolation.

**Cross-verified with a second, independent Perplexity search on 2026-09-29** (archived in the project chat log). That search found the same set of papers — confirming they're real, not hallucinated — but still could not recover full author lists for several of them directly from the retrieved pages; those gaps are noted per-paper below rather than treated as a citation risk.

| # | Contribution | Closest paper | Closeness | Reworded claim to use |
|---|---|---|---|---|
| 1 | Cost-tiered comparison | "A machine learning approach for non-invasive PCOS diagnosis from ultrasound and clinical features," *Scientific Reports*, 2025 — **same Kottarathil dataset**; compares ultrasound-only, ultrasound+biochemical, clinical+biochemical and ultrasound+clinical feature groups (not the same 3-tier ladder, but substantial overlap) | Very close | "We extend prior PCOS feature-subset comparisons by evaluating a clinically ordered, nested cost hierarchy — screening-only, screening + biochemical, screening + ultrasound — and quantifying the incremental performance benefit of each tier." |
| 1b | (secondary) | "Optimized Machine Learning for the Early Detection of Polycystic Ovary Syndrome in Women," *Sensors*, 2025 — removes test-derived features and predicts from basic symptoms/vitals (BMI, glucose, cycle length, waist-hip ratio, weight gain, hair growth) | Moderately close — non-invasive-only, but not a 3-tier ablation | "We extend symptom-based, non-invasive PCOS prediction by quantifying the performance retained — and lost — when biochemical and ultrasound information is progressively removed." |
| 2 | Calibration as primary metric | Akter, S. & Reno, S., "Explainable AI for Generalizable PCOS Diagnosis: A Geographically Validated Ensemble Learning Approach With Feature Selection," *Engineering Reports*, 2025 (reports Brier=0.042, best of the classifiers compared); also "A reliable and explainable deep learning framework for clinical prediction," *Frontiers in Artificial Intelligence*, 2026 (reports Brier, calibration error, and reliability diagrams alongside AUROC) and "Development and validation of an explainable machine learning and nomogram model for early detection and risk stratification of PCOS: a multicenter study," *Frontiers in Endocrinology*, 2025 (reports calibration curves, picks the model closest to the diagonal) | Very close — calibration is reported in all three, though not as any paper's central contribution | "We treat calibration as a primary clinical endpoint and examine how calibration changes when information is restricted to cheaper feature tiers" — this specific angle (calibration *under tiered information restriction*) was not found in any retrieved paper. |
| 3 | Hand-built clinical BN, incomplete evidence | "An Expert System to Detect Polycystic Ovary Syndrome under Uncertainty," 2016 (⚠️ academia.edu-hosted; exact peer-reviewed venue still not confirmed by either search) — web-based PCOS risk-stratification expert system explicitly framed as diagnosis under uncertainty, reports Bayesian classification performance, but the retrieved record doesn't establish whether the network is hand-specified or whether it handles arbitrary missing fields the way ours does; also "A Concept-Enhanced, Knowledge Graph-Guided Framework for Interpretable PCOS Prediction: A Case Study in Explainable Biomedical Artificial Intelligence," *Reproduction, Fertility and Development*, 2026 — combines a knowledge graph with a BN for interpretability, not a standalone hand-built clinical BN | Close conceptually, unverified in detail | "We extend Bayesian-network-based PCOS expert systems by specifying the graph and conditional probabilities clinically and evaluating posterior PCOS probabilities under systematically incomplete evidence." No retrieved paper combines a hand-specified BN + posterior P(PCOS\|evidence) + arbitrary-missingness inference + tiered evaluation + calibration — that full combination is defensible as the contribution if documented rigorously. |
| 4 | Label circularity | Zad, Z., Jiang, V.S., Wolf, A.T., Wang, T., Cheng, J.J., Paschalidis, I.Ch. & Mahalingaiah, S., "Predicting polycystic ovary syndrome with machine learning algorithms from electronic health records," *Frontiers in Endocrinology*, 2024 — defines PCOS outcomes directly from Rotterdam components (irregular menstruation, hyperandrogenism, polycystic ovarian morphology); also the same-dataset *Scientific Reports* 2025 paper, whose top predictors (follicle count, hair growth, cycle irregularity) map directly onto Rotterdam domains | Very close (defines labels from Rotterdam components, doesn't formally name/quantify the circularity) | "We extend prior PCOS prediction studies that define outcomes from Rotterdam components by explicitly auditing diagnostic circularity and quantifying performance with and without label-defining features." Neither search found a PCOS paper that formally names this "label leakage" or quantifies the accuracy inflation the way our Step 10 ablation does. |

**Clinical-definition backing for contribution 4:** "Current Guidelines for Diagnosing PCOS," NLM/PMC clinical review, 2023 (PMC10047373) confirms the modified Rotterdam criteria require 2-of-3 domains — hyperandrogenism, oligo/anovulation, polycystic ovarian morphology — after exclusion of other conditions. Cite this alongside the Rotterdam 2004 primary source when explaining *why* follicle count/cycle irregularity/hair growth are label-defining, not just correlated.

**Not directly relevant despite initial flag:** Chakraborty et al. 2013 (*PLoS ONE* 8(5):e64446) uses a Bayesian/causal model, but for recurrent pregnancy loss risk *among women already diagnosed with PCOS* — i.e. models P(RPL | PCOS, evidence), not P(PCOS | evidence). Useful only as a general "Bayesian modeling in PCOS-adjacent research" precedent, not as a direct BN-for-PCOS-diagnosis comparator.

**⚠️ Still unverified metadata (confirmed real papers, but author lists not exposed in the retrieved pages — track down via publisher/PubMed before the bibliography is finalized):**
- "An Expert System to Detect Polycystic Ovary Syndrome under Uncertainty" (2016, academia.edu) — venue still unconfirmed by two independent searches; if it turns out to closely match your BN's specific design (missing-evidence inference), you'll need to narrow the novelty claim further.
- "A machine learning approach for non-invasive PCOS diagnosis from ultrasound and clinical features," *Scientific Reports*, 2025.
- "Optimized Machine Learning for the Early Detection of Polycystic Ovary Syndrome in Women," *Sensors*, 2025.
- "A reliable and explainable deep learning framework for clinical prediction," *Frontiers in Artificial Intelligence*, 2026.
- "Development and validation of an explainable machine learning and nomogram model...," *Frontiers in Endocrinology*, 2025.
- "A Concept-Enhanced, Knowledge Graph-Guided Framework for Interpretable PCOS Prediction," *Reproduction, Fertility and Development*, 2026.
- "Current Guidelines for Diagnosing PCOS," NLM/PMC clinical review, 2023.

**✅ Solid, safe to lean on (full author lists confirmed by two independent searches):** Zad et al. 2024 (*Frontiers in Endocrinology*), Akter & Reno 2025 (*Engineering Reports*), and Chakraborty et al. 2013 (*PLoS ONE*).

---

## 4. Action items checklist

- [ ] Rewrite R10's contribution statement in `PCOS_PROJECT_PLAN.md` using the four reworded claims in §3 above (frame as "extends X," not "first to do X").
- [ ] Add the **BMI action-point reframing** wording from §2 to the Methods section (don't describe 23/27.5 as *the* Asian BMI classification).
- [ ] Add the LH:FSH Limitations wording from §2 to the report.
- [ ] Add the follicle-threshold Methods + Limitations wording from §2 to the report.
- [x] ~~Spot-check the ⚠️-flagged citations (Rotterdam 2003/2004, Pigny et al. 2006)~~ — Rotterdam 2003/2004 and the M9 threshold citations (WHO 2004, Cho et al. 2006, Rotterdam 2004, Teede et al. 2023) verified via live Perplexity search, 2026-09-29; DOIs added in §2 and §5.
- [x] ~~Cross-verify the M12 novelty-search papers~~ — a second independent Perplexity search (2026-09-29) confirmed all §3 papers are real (not hallucinated) and added Akter & Reno's full first names, the fuller title for the 2026 knowledge-graph BN paper, and direct links for every entry; author lists for entries 10, 11, 14, 15, 16, 17, 19 still couldn't be recovered from the retrieved pages — track those down from the publisher/PubMed record before the bibliography is finalized.
- [ ] Spot-check remaining metadata gaps (entries 10, 11, 14, 15, 16, 17, 19 in §5 — missing author lists; Pigny et al. 2006 — not yet searched) before final submission.
- [ ] Fold the M8 edge-justification table (§1) into the report's Bayesian Network section, one sentence per edge.
- [ ] Use the reference list below directly in the report's bibliography.

---

## 5. Full reference list

1. Barber, T.M. & Franks, S. (2021). Obesity and polycystic ovary syndrome. *Clinical Endocrinology*.
2. Ahmed, B. et al. (2020). Acanthosis nigricans as a composite marker of cardiometabolic risk... *PLOS ONE*.
3. Rotterdam ESHRE/ASRM-Sponsored PCOS Consensus Workshop Group (2004). Revised 2003 consensus on diagnostic criteria and long-term health risks related to PCOS. *Human Reproduction*, 19(1), 41–47. DOI: 10.1093/humrep/deh098.
4. Archer, J.S. et al. (2004). Hirsutism and acne in polycystic ovary syndrome. *Best Practice & Research Clinical Obstetrics & Gynaecology*.
5. S.R.K. et al. (2022). Acanthosis Nigricans: Pointer of Endocrine Entities. *Indian Journal of Endocrinology and Metabolism*.
6. Barber, T.M. & Franks, S. (2022). Why are women with polycystic ovary syndrome obese? *British Medical Bulletin*, 143(1), 4.
7. Author(s) (2025). [Physiopathology of PCOS review]. *Endocrinology, Diabetes & Metabolism*.
8. Pigny, P., Jonard, S., Robert, Y. & Dewailly, D. (2006). Serum anti-Müllerian hormone as a surrogate for antral follicle count for definition of the polycystic ovary syndrome. *Journal of Clinical Endocrinology & Metabolism*.
9. Dewailly, D. et al. (2016). The role of AMH in the pathophysiology of polycystic ovarian syndrome. *Reproductive BioMedicine Online*.
10. [Authors — not exposed by the retrieved page, confirmed real via two independent searches] (2025). A machine learning approach for non-invasive PCOS diagnosis from ultrasound and clinical features. *Scientific Reports*. https://www.nature.com/articles/s41598-025-10453-9
11. [Authors — not exposed by the retrieved page] (2025). Optimized Machine Learning for the Early Detection of Polycystic Ovary Syndrome in Women. *Sensors*. https://www.mdpi.com/1424-8220/25/4/1166
12. Zad, Z., Jiang, V.S., Wolf, A.T., Wang, T., Cheng, J.J., Paschalidis, I.Ch. & Mahalingaiah, S. (2024). Predicting polycystic ovary syndrome with machine learning algorithms from electronic health records. *Frontiers in Endocrinology*.
13. Akter, S. & Reno, S. (2025). Explainable AI for Generalizable PCOS Diagnosis: A Geographically Validated Ensemble Learning Approach With Feature Selection. *Engineering Reports*.
14. [Authors — not exposed by the retrieved page] (2026). A reliable and explainable deep learning framework for clinical prediction. *Frontiers in Artificial Intelligence*. https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1881082/full ⚠️ verify.
15. [Authors — not exposed by the retrieved page] (2025). Development and validation of an explainable machine learning and nomogram model for early detection and risk stratification of PCOS: a multicenter study. *Frontiers in Endocrinology*. https://www.frontiersin.org/articles/10.3389/fendo.2025.1719631/full
16. [Author(s) — not exposed by the retrieved page] (2016). An Expert System to Detect Polycystic Ovary Syndrome under Uncertainty. https://www.academia.edu/26943062/An_Expert_System_to_Detect_Polycystic_Ovary_Syndrome_under_Uncertainty ⚠️ verify venue — not confirmed as peer-reviewed by either search.
17. [Authors — not exposed by the retrieved page] (2026). A Concept-Enhanced, Knowledge Graph-Guided Framework for Interpretable PCOS Prediction: A Case Study in Explainable Biomedical Artificial Intelligence. *Reproduction, Fertility and Development*. https://pubmed.ncbi.nlm.nih.gov/41887676/ ⚠️ verify.
18. Chakraborty, P., Goswami, S.K., Rajani, S., Sharma, S., Kabir, S.N., Chakravarty, B. et al. (2013). Recurrent Pregnancy Loss in Polycystic Ovary Syndrome: Role of Hyperhomocysteinemia and Insulin Resistance. *PLoS ONE*, 8(5), e64446. (Bayesian/causal model of RPL risk among PCOS patients, not a PCOS-diagnosis model — see §3.)
19. [Authors — not exposed by the retrieved page] (2023). Current Guidelines for Diagnosing PCOS. NLM/PMC clinical review. https://pmc.ncbi.nlm.nih.gov/articles/PMC10047373/
20. WHO Expert Consultation (2004). Appropriate body-mass index for Asian populations and its implications for policy and intervention strategies. *The Lancet*, 363(9403), 157–163. DOI: 10.1016/S0140-6736(03)15268-3.
21. Dokras, A. et al. (2016). Diagnosis and Treatment of Polycystic Ovary Syndrome. *American Family Physician*. (Secondary summary only — repeats the LH:FSH >2 heuristic without a primary citation; see §2.)
22. Cho, L.W., Jayagopal, V., Kilpatrick, E.S., Holding, S. & Atkin, S.L. (2006). The LH/FSH ratio has little use in diagnosing polycystic ovarian syndrome. *Annals of Clinical Biochemistry*, 43(3), 217–219. DOI: 10.1258/000456306776865188.
23. Teede, H.J., Tay, C.T., Laven, J.J.E., Dokras, A., Moran, L.J., Piltonen, T.T., Costello, M.F., Boivin, J., Redman, L.M., Boyle, J.A., Norman, R.J., Mousa, A. & Joham, A.E. (2023). Recommendations From the 2023 International Evidence-based Guideline for the Assessment and Management of Polycystic Ovary Syndrome. *Journal of Clinical Endocrinology & Metabolism*, 108(10), 2447–2469. DOI: 10.1210/clinem/dgad463. (Also published concurrently in *Human Reproduction* 38(9):1655–1679, DOI 10.1093/humrep/dead156, and *Fertility and Sterility* 120(4):767–793, DOI 10.1016/j.fertnstert.2023.07.025 — cite whichever venue fits your reference style.)

*Entries with "[Authors]" had incomplete author metadata in the original search results — fill in from the publisher/PubMed record when finalizing the bibliography.*
