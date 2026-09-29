# Structure comparison: hand-specified DAG vs Hill-Climb/BIC

## Hand-specified DAG (OOF, train)

| tier            |   oof_roc_auc |   oof_brier |   threshold |
|:----------------|--------------:|------------:|------------:|
| tier1_screening |      0.869525 |    0.131006 |    0.108629 |
| tier2_lab       |      0.878994 |    0.128804 |    0.127596 |
| tier3_full      |      0.933404 |    0.107571 |    0.142315 |

## Learned structure (Hill-Climb + BIC) (OOF, train)

| tier            |   oof_roc_auc |   oof_brier |
|:----------------|--------------:|------------:|
| tier1_screening |      0.86732  |    0.124916 |
| tier2_lab       |      0.875862 |    0.123658 |
| tier3_full      |      0.938461 |    0.097436 |

Learned edges: [('weight_gain', 'bmi_cat'), ('weight_gain', 'pimples'), ('skin_darkening', 'pcos'), ('skin_darkening', 'weight_gain'), ('skin_darkening', 'hair_growth'), ('pcos', 'follicle_high'), ('pcos', 'weight_gain'), ('pcos', 'hair_growth'), ('pcos', 'cycle_irregular'), ('pcos', 'amh_level'), ('pcos', 'pimples')]
