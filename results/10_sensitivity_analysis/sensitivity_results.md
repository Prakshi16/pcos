| analysis       | model               | variant               |   test_roc_auc |   test_brier |
|:---------------|:--------------------|:----------------------|---------------:|-------------:|
| imputation     | logistic_regression | median                |       0.877854 |    0.124352  |
| imputation     | logistic_regression | knn                   |       0.878995 |    0.124003  |
| imputation     | logistic_regression | iterative             |       0.874429 |    0.123243  |
| imputation     | random_forest       | median                |       0.892314 |    0.132203  |
| imputation     | random_forest       | knn                   |       0.884513 |    0.1316    |
| imputation     | random_forest       | iterative             |       0.881469 |    0.131306  |
| feature_subset | logistic_regression | all_tier2             |       0.877854 |    0.124352  |
| feature_subset | logistic_regression | consensus_top10       |       0.895358 |    0.113503  |
| feature_subset | logistic_regression | anova_top10           |       0.895738 |    0.12089   |
| feature_subset | random_forest       | all_tier2             |       0.892314 |    0.132203  |
| feature_subset | random_forest       | consensus_top10       |       0.88965  |    0.122912  |
| feature_subset | random_forest       | anova_top10           |       0.914384 |    0.118807  |
| circularity    | logistic_regression | tier3_full            |       0.942161 |    0.0794405 |
| circularity    | logistic_regression | tier3_minus_rotterdam |       0.832953 |    0.150576  |
| circularity    | random_forest       | tier3_full            |       0.945205 |    0.0961893 |
| circularity    | random_forest       | tier3_minus_rotterdam |       0.857686 |    0.154584  |