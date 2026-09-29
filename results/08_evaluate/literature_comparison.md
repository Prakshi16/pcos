# Literature comparison (Tier 3, Random Forest, screening threshold)

Our Tier 3 test accuracy: 89.91%. Our Tier 3 test ROC-AUC: 0.943.

| Source | Setting | Reported |
|---|---|---|
| Aggarwal et al. (2023) | Kaggle, feature selection + RF | 93.52% accuracy |
| Various | Kaggle; RF, MLP, Linear SVM | 85-93% accuracy |
| Indian J. Community Health | NB vs DT, 200 cases | NB 81% accuracy, F1 0.81 |
| Frontiers in Endocrinology (2024) | EHR data; RF, GBT, SVM, LR | AUC 0.774-0.85 |
| Some later studies | Kaggle, aggressive selection | 99.3% (likely overfitting) |

Our result falls within the range reported by prior Kaggle-based studies.
