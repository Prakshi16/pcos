"""Step 7: Clinically specified Bayesian Network + structure ablation.

Reads: data/processed/train.csv.
Writes: models/bayesian_network/{bn_model.joblib, discretization.json},
        results/07_bayesian_network/ (structure diagram, OOF metrics,
        structure comparison, partial-evidence demo).
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import config FIRST: it installs a statsmodels stub (if needed) before pgmpy
# is touched, since pgmpy.estimators/pgmpy.inference eagerly import statsmodels
# transitively (via CausalInference -> LinearEstimator) even though this project
# never uses that feature.
from src import config

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from pgmpy.estimators import BayesianEstimator, HillClimbSearch, BIC
from pgmpy.inference import VariableElimination

try:
    from pgmpy.models import DiscreteBayesianNetwork as BNModel
except ImportError:
    from pgmpy.models import BayesianNetwork as BNModel

from src.utils.io import save_json, save_model
from src.utils.bn import fit_discretization, discretize, BNClassifier
from src.utils.metrics import compute_metrics, select_recall_threshold

OUT = config.RESULTS_DIR / "07_bayesian_network"
OUT.mkdir(parents=True, exist_ok=True)
BN_NODES = ["bmi_cat", "weight_gain", "cycle_irregular", "hair_growth", "pimples",
            "skin_darkening", "lh_fsh_high", "amh_level", "follicle_high", "pcos"]

warnings.filterwarnings("ignore")


def fit_cpds(structure_edges, discretized_train):
    model = BNModel(structure_edges)
    model.add_nodes_from(BN_NODES)
    assert nx.is_directed_acyclic_graph(nx.DiGraph(model.edges())), "BN structure must be acyclic"
    est = BayesianEstimator(model, discretized_train)
    cpds = est.get_parameters(prior_type="BDeu", equivalent_sample_size=5)
    model.add_cpds(*cpds)
    return model


def draw_structure(edges, highlight_node, path, title):
    g = nx.DiGraph(edges)
    for n in BN_NODES:
        g.add_node(n)
    pos = nx.spring_layout(g, seed=config.SEED, k=1.2)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#C44E52" if n == highlight_node else "#4C72B0" for n in g.nodes()]
    nx.draw(g, pos, ax=ax, with_labels=True, node_color=colors, node_size=1800,
            font_size=8, font_color="white", arrows=True, arrowsize=15)
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def oof_evaluate(train_df, structure_edges, tiers_evidence):
    """5-fold stratified OOF probabilities per tier; cutoffs + CPDs refit per fold."""
    y = train_df["pcos"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    oof_proba = {tier: np.full(len(train_df), np.nan) for tier in tiers_evidence}

    for train_idx, test_idx in skf.split(train_df, y):
        fold_train = train_df.iloc[train_idx]
        fold_test = train_df.iloc[test_idx]

        cutoffs = fit_discretization(fold_train)
        disc_train = discretize(fold_train, cutoffs, impute=True)
        model = fit_cpds(structure_edges, disc_train)
        clf = BNClassifier(model, cutoffs)

        for tier, evidence_vars in tiers_evidence.items():
            proba = clf.predict_proba(fold_test, evidence_vars)
            oof_proba[tier][test_idx] = proba

    return oof_proba


def main():
    train_df = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    y = train_df["pcos"].values

    # --- Fit final cutoffs + CPDs on the FULL train set ---
    cutoffs = fit_discretization(train_df)
    disc_train_full = discretize(train_df, cutoffs, impute=True)
    n_imputed = {c: int(train_df[c].isna().sum()) for c in
                 ["bmi", "weight_gain", "cycle_irregular", "hair_growth", "pimples",
                  "skin_darkening", "lh_fsh_ratio", "amh", "follicle_left", "follicle_right"]}
    print("Values imputed before discretization (train):", n_imputed)

    model = fit_cpds(config.BN_STRUCTURE, disc_train_full)
    clf = BNClassifier(model, cutoffs)

    draw_structure(config.BN_STRUCTURE, "pcos", OUT / "bn_structure.png",
                    "Hand-specified clinical DAG")

    # --- OOF evaluation per tier ---
    oof_proba = oof_evaluate(train_df, config.BN_STRUCTURE, config.BN_TIER_EVIDENCE)
    oof_rows = []
    thresholds = {}
    for tier, proba in oof_proba.items():
        m = compute_metrics(y, proba, threshold=0.5)
        threshold, achieved = select_recall_threshold(y, proba, config.RECALL_TARGET)
        if not achieved:
            print(f"WARNING: BN {tier} did not reach recall {config.RECALL_TARGET}; "
                  f"using max-recall threshold {threshold:.3f}.")
        thresholds[tier] = threshold
        oof_rows.append({"tier": tier, "oof_roc_auc": m["roc_auc"], "oof_brier": m["brier"],
                          "threshold": threshold})
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(OUT / "oof_results.csv", index=False)
    print(oof_df)

    # --- Structure ablation: HillClimbSearch + BIC ---
    learned_dag = HillClimbSearch(disc_train_full).estimate(
        scoring_method=BIC(disc_train_full), show_progress=False)
    learned_edges = list(learned_dag.edges())
    draw_structure(learned_edges, "pcos", OUT / "bn_structure_learned.png",
                    "Hill-Climb / BIC learned structure")

    learned_oof = {}
    try:
        learned_model_check = BNModel(learned_edges)
        learned_model_check.add_nodes_from(BN_NODES)
        assert nx.is_directed_acyclic_graph(nx.DiGraph(learned_model_check.edges()))
        learned_oof = oof_evaluate(train_df, learned_edges, config.BN_TIER_EVIDENCE)
        learned_rows = []
        for tier, proba in learned_oof.items():
            valid = ~np.isnan(proba)
            m = compute_metrics(y[valid], proba[valid], threshold=0.5) if valid.any() else {}
            learned_rows.append({"tier": tier, "oof_roc_auc": m.get("roc_auc"), "oof_brier": m.get("brier")})
        learned_df = pd.DataFrame(learned_rows)
    except Exception as e:
        learned_df = pd.DataFrame([{"tier": t, "oof_roc_auc": None, "oof_brier": None} for t in config.BN_TIER_EVIDENCE])
        print(f"Learned-structure evaluation could not be completed for some tiers: {e}")

    with open(OUT / "structure_comparison.md", "w") as f:
        f.write("# Structure comparison: hand-specified DAG vs Hill-Climb/BIC\n\n")
        f.write("## Hand-specified DAG (OOF, train)\n\n")
        f.write(oof_df.to_markdown(index=False) + "\n\n")
        f.write("## Learned structure (Hill-Climb + BIC) (OOF, train)\n\n")
        f.write(learned_df.to_markdown(index=False) + "\n\n")
        f.write(f"Learned edges: {learned_edges}\n")

    # --- Missing-information demo ---
    examples = [
        {"bmi_cat": "obese", "cycle_irregular": "yes"},
        {"bmi_cat": "obese", "cycle_irregular": "yes", "lh_fsh_high": "yes"},
        {"bmi_cat": "obese", "cycle_irregular": "yes", "lh_fsh_high": "yes", "follicle_high": "yes"},
    ]
    with open(OUT / "partial_evidence_examples.md", "w") as f:
        f.write("# Partial-evidence inference examples\n\n")
        for i, ev in enumerate(examples, 1):
            p = clf.predict_proba_evidence_dict(ev)
            f.write(f"**Example {i}**: evidence = `{ev}`\n\nP(PCOS=yes) = {p:.3f}\n\n")
    print("Saved partial_evidence_examples.md")

    # --- Save model ---
    save_json(cutoffs, config.MODELS_DIR / "bayesian_network" / "discretization.json")
    save_model({
        "model": model,
        "cutoffs": cutoffs,
        "thresholds": thresholds,
        "tier_evidence": config.BN_TIER_EVIDENCE,
        "oof_brier": {r["tier"]: r["oof_brier"] for r in oof_rows},
        "oof_auc": {r["tier"]: r["oof_roc_auc"] for r in oof_rows},
    }, config.MODELS_DIR / "bayesian_network" / "bn_model.joblib")

    print("Step 7 (07_train_bayesian_network.py) complete.")


if __name__ == "__main__":
    main()
