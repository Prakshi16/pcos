"""Builds the shared sklearn/imblearn pipeline used by NB, LR and RF (Section 4a).

Column order after the ColumnTransformer is always: numeric columns first,
then binary columns. This lets SMOTENC know which indices are categorical.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

from src import config


def split_feature_types(features):
    """Return (numeric_features, binary_features) preserving numeric-first order."""
    binary = [f for f in features if f in config.BINARY_FEATURES]
    numeric = [f for f in features if f not in config.BINARY_FEATURES]
    return numeric, binary


def build_preprocessor(numeric_features, binary_features):
    """ColumnTransformer: median-impute numeric, most-frequent-impute binary.

    Numeric columns come first in the output, binary columns second, so
    SMOTENC's categorical_features indices (len(numeric_features)..end) are
    known regardless of the tier.
    """
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("bin", SimpleImputer(strategy="most_frequent"), binary_features),
        ],
        remainder="drop",
    )


def build_pipeline(numeric_features, binary_features, estimator, strategy,
                    scale: bool, select_k: bool, k=None, max_k=None):
    """Assemble the full imblearn Pipeline for one (model, strategy) candidate.

    strategy: "none", "class_weight" (handled by the estimator itself, no
              resampling step here), or "smote" (adds SMOTENC after imputation).
    scale: True for NB/LR, False for RF.
    select_k: True for NB/LR (SelectKBest before the estimator), False for RF.
    """
    steps = [("prep", build_preprocessor(numeric_features, binary_features))]

    n_binary = len(binary_features)
    n_total = len(numeric_features) + n_binary
    cat_idx = list(range(n_total - n_binary, n_total))

    if strategy == "smote":
        if n_binary > 0:
            steps.append(("resample", SMOTENC(categorical_features=cat_idx, random_state=config.SEED)))
        else:
            from imblearn.over_sampling import SMOTE
            steps.append(("resample", SMOTE(random_state=config.SEED)))

    if scale:
        steps.append(("scale", ColumnScaler(len(numeric_features), n_total)))

    if select_k:
        k_eff = k if k is not None else "all"
        if k_eff != "all" and max_k is not None:
            k_eff = min(k_eff, max_k)
        steps.append(("select", SelectKBest(score_func=f_classif, k=k_eff)))

    steps.append(("clf", estimator))
    return ImbPipeline(steps)


class ColumnScaler(BaseEstimator, TransformerMixin):
    """StandardScaler applied only to the first n_numeric columns; binary columns pass through.

    Implemented as a small transformer (not sklearn's ColumnTransformer) because
    at this point in the pipeline we only have a plain numpy array (post-imputation),
    with numeric columns first and binary columns last.
    """

    def __init__(self, n_numeric, n_total):
        self.n_numeric = n_numeric
        self.n_total = n_total

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler()
        if self.n_numeric > 0:
            self.scaler_.fit(X[:, :self.n_numeric])
        return self

    def transform(self, X):
        X = X.copy()
        if self.n_numeric > 0:
            X[:, :self.n_numeric] = self.scaler_.transform(X[:, :self.n_numeric])
        return X
