"""Step 3: Statistical feature ranking on train only (exploratory; models select independently).

Reads: data/processed/train.csv.
Writes: results/03_feature_analysis/statistical_ranking_{tier}.csv/.md,
        results/03_feature_analysis/top_features_{tier}.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif

from src import config
from src.utils.io import save_csv_and_md
from src.utils.plotting import new_fig, save_fig

OUT = config.RESULTS_DIR / "03_feature_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def rank_tier(train_df, tier):
    features = config.TIER_FEATURES[tier]
    numeric_feats = [f for f in features if f not in config.BINARY_FEATURES]
    binary_feats = [f for f in features if f in config.BINARY_FEATURES]

    X = train_df[features].copy()
    y = train_df["pcos"].values

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features, index=X.index)

    rows = []
    for feat in numeric_feats:
        f_stat, p_val = f_classif(X_imputed[[feat]], y)
        rows.append({"feature": feat, "test": "ANOVA_F", "score": f_stat[0], "p_value": p_val[0]})
    for feat in binary_feats:
        # chi2 requires non-negative values; binary 0/1 already satisfies this.
        chi_stat, p_val = chi2(X_imputed[[feat]], y)
        rows.append({"feature": feat, "test": "chi_square", "score": chi_stat[0], "p_value": p_val[0]})

    stat_df = pd.DataFrame(rows)
    stat_df["stat_rank"] = stat_df["score"].rank(ascending=False, method="min")

    mi = mutual_info_classif(X_imputed[features], y, random_state=config.SEED)
    mi_df = pd.DataFrame({"feature": features, "mutual_info": mi})
    mi_df["mi_rank"] = mi_df["mutual_info"].rank(ascending=False, method="min")

    result = stat_df.merge(mi_df, on="feature", how="outer").sort_values("mi_rank")
    return result


def main():
    train_df = pd.read_csv(config.PROCESSED_DIR / "train.csv")

    for tier in config.TIERS:
        result = rank_tier(train_df, tier)
        save_csv_and_md(result, OUT / f"statistical_ranking_{tier}")
        print(f"{tier}: ranked {len(result)} features.")

        top15 = result.sort_values("mutual_info", ascending=False).head(15)
        fig, ax = new_fig((8, 6))
        ax.barh(top15["feature"][::-1], top15["mutual_info"][::-1], color="#4C72B0")
        ax.set_xlabel("Mutual information")
        ax.set_title(f"Top 15 features by mutual information ({tier})")
        save_fig(fig, OUT / f"top_features_{tier}.png")

    print("Step 3 (03_feature_analysis.py) complete.")


if __name__ == "__main__":
    main()
