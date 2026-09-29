"""Discretization and inference helpers for the clinically specified Bayesian Network (Step 7).

Discretization is always fit on train only (see fit_discretization). At inference
time, missing values are simply not passed as evidence (evidence_from_row).
"""
import numpy as np
import pandas as pd

from src import config


def fit_discretization(train_df):
    """Compute cutoffs/imputation values from a TRAIN fold only.

    Returns a dict saved to models/bayesian_network/discretization.json (or
    used transiently inside a CV fold).
    """
    amh = train_df["amh"].dropna()
    tertiles = list(np.quantile(amh, [1 / 3, 2 / 3])) if len(amh) > 0 else [10.0, 20.0]

    impute = {
        "bmi_median": float(train_df["bmi"].median()),
        "lh_fsh_ratio_median": float(train_df["lh_fsh_ratio"].median()),
        "amh_median": float(train_df["amh"].median()),
        "follicle_left_median": float(train_df["follicle_left"].median()),
        "follicle_right_median": float(train_df["follicle_right"].median()),
    }
    for b in ["weight_gain", "cycle_irregular", "hair_growth", "pimples", "skin_darkening"]:
        mode = train_df[b].mode()
        impute[f"{b}_mode"] = float(mode.iloc[0]) if len(mode) > 0 else 0.0

    return {
        "amh_tertiles": tertiles,
        "bmi_cuts": list(config.BMI_CUTS),
        "lh_fsh_high_cutoff": config.LH_FSH_HIGH_CUTOFF,
        "follicle_high_cutoff": config.FOLLICLE_HIGH_CUTOFF,
        "impute": impute,
    }


def _bmi_cat(bmi, cuts):
    lo, hi = cuts
    if pd.isna(bmi):
        return np.nan
    if bmi < lo:
        return "normal"
    if bmi <= hi:
        return "overweight"
    return "obese"


def _amh_level(amh, tertiles):
    if pd.isna(amh):
        return np.nan
    if amh < tertiles[0]:
        return "low"
    if amh < tertiles[1]:
        return "mid"
    return "high"


def _binary_state(v):
    if pd.isna(v):
        return np.nan
    return "yes" if v == 1 else "no"


def discretize(df, cutoffs, impute=True):
    """Discretize canonical columns into BN node states.

    impute=True: fill missing values with the fit-time medians/modes first
    (used for training the CPDs / OOF evaluation).
    impute=False: leave missing as NaN (used at inference time so that
    missing evidence is simply not passed to VariableElimination).
    """
    out = pd.DataFrame(index=df.index)
    imp = cutoffs["impute"]

    def col_or_nan(name):
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    bmi = col_or_nan("bmi")
    bmi = bmi.fillna(imp["bmi_median"]) if impute else bmi
    out["bmi_cat"] = bmi.apply(lambda v: _bmi_cat(v, cutoffs["bmi_cuts"]))

    for b in ["weight_gain", "cycle_irregular", "hair_growth", "pimples", "skin_darkening"]:
        col = col_or_nan(b)
        col = col.fillna(imp[f"{b}_mode"]) if impute else col
        out[b] = col.apply(_binary_state)

    ratio = col_or_nan("lh_fsh_ratio")
    ratio = ratio.fillna(imp["lh_fsh_ratio_median"]) if impute else ratio
    out["lh_fsh_high"] = ratio.apply(
        lambda v: np.nan if pd.isna(v) else ("yes" if v > cutoffs["lh_fsh_high_cutoff"] else "no")
    )

    amh = col_or_nan("amh")
    amh = amh.fillna(imp["amh_median"]) if impute else amh
    out["amh_level"] = amh.apply(lambda v: _amh_level(v, cutoffs["amh_tertiles"]))

    fl = col_or_nan("follicle_left")
    fl = fl.fillna(imp["follicle_left_median"]) if impute else fl
    fr = col_or_nan("follicle_right")
    fr = fr.fillna(imp["follicle_right_median"]) if impute else fr
    max_foll = pd.concat([fl, fr], axis=1).max(axis=1)
    out["follicle_high"] = max_foll.apply(
        lambda v: np.nan if pd.isna(v) else ("yes" if v >= cutoffs["follicle_high_cutoff"] else "no")
    )

    if "pcos" in df.columns:
        out["pcos"] = df["pcos"].apply(_binary_state)

    return out


class BNClassifier:
    """Wraps a fitted pgmpy DiscreteBayesianNetwork so it can be called like the
    other model families: predict_proba(df, evidence_vars) -> P(pcos=yes)."""

    def __init__(self, model, cutoffs):
        self.model = model
        self.cutoffs = cutoffs
        from pgmpy.inference import VariableElimination
        self.infer = VariableElimination(model)
        self._cache = {}

    def _query_yes_prob(self, evidence: dict):
        key = tuple(sorted(evidence.items()))
        if key in self._cache:
            return self._cache[key]
        if len(evidence) == 0:
            q = self.infer.query(["pcos"], show_progress=False)
        else:
            q = self.infer.query(["pcos"], evidence=evidence, show_progress=False)
        state_names = q.state_names["pcos"]
        p = float(q.values[state_names.index("yes")])
        self._cache[key] = p
        return p

    def predict_proba(self, raw_df, evidence_vars):
        disc = discretize(raw_df, self.cutoffs, impute=False)
        probs = np.full(len(raw_df), np.nan)
        for i, (idx, row) in enumerate(disc.iterrows()):
            evidence = {v: row[v] for v in evidence_vars if v in row and pd.notna(row[v])}
            probs[i] = self._query_yes_prob(evidence)
        return probs

    def predict_proba_evidence_dict(self, evidence: dict):
        """P(pcos=yes | evidence) for a hand-built evidence dict (used for the
        partial-information demo and the app's what-if page)."""
        evidence = {k: v for k, v in evidence.items() if v is not None}
        return self._query_yes_prob(evidence)
