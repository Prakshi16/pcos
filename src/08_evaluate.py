"""Step 8: Final evaluation on the held-out TEST set (first use of test data).

Reads: data/processed/test.csv, results/{04,05,06}_*/oof_calibration_results.csv,
       models/bayesian_network/bn_model.joblib.
Writes: results/selected_models.json (via results dir), results/08_evaluate/*.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve

from src import config
from src.utils.io import save_csv_and_md, save_json, load_model
from src.utils.metrics import compute_metrics, bootstrap_ci, bootstrap_ci_recall
from src.utils.bn import BNClassifier
from src.utils.plotting import new_fig, save_fig

warnings.filterwarnings("ignore")

OUT = config.RESULTS_DIR / "08_evaluate"
OUT.mkdir(parents=True, exist_ok=True)

SKLEARN_MODELS = ["naive_bayes", "logistic_regression", "random_forest"]
RESULT_DIRS = {
    "naive_bayes": config.RESULTS_DIR / "04_naive_bayes",
    "logistic_regression": config.RESULTS_DIR / "05_logistic_regression",
    "random_forest": config.RESULTS_DIR / "06_random_forest",
}


def select_sklearn_candidate(model_name, tier):
    df = pd.read_csv(RESULT_DIRS[model_name] / "oof_calibration_results.csv")
    df = df[df["tier"] == tier].copy()
    df = df.sort_values(["oof_brier", "oof_auc"], ascending=[True, False])
    best = df.iloc[0]
    path = config.MODELS_DIR / tier / f"{model_name}__{best['strategy']}__{best['calibration']}.joblib"
    return {
        "path": str(path.relative_to(config.ROOT)).replace("\\", "/"),
        "strategy": best["strategy"], "calibration": best["calibration"],
        "oof_brier": float(best["oof_brier"]), "oof_auc": float(best["oof_auc"]),
    }


def get_sklearn_proba(entry, test_df):
    artifact = load_model(config.ROOT / entry["path"])
    proba = artifact["model"].predict_proba(test_df[artifact["features"]])[:, 1]
    return proba, artifact["threshold"]


def get_raw_proba(entry, tier, model_name, test_df):
    raw_path = config.MODELS_DIR / tier / f"{model_name}__{entry['strategy']}__raw.joblib"
    artifact = load_model(raw_path)
    proba = artifact["model"].predict_proba(test_df[artifact["features"]])[:, 1]
    return proba


def main():
    test_df = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    y_test = test_df["pcos"].values

    bn_artifact = load_model(config.MODELS_DIR / "bayesian_network" / "bn_model.joblib")
    bn_clf = BNClassifier(bn_artifact["model"], bn_artifact["cutoffs"])

    selected = {}
    test_rows = []
    raw_vs_calib = {}  # tier -> {model: (raw_proba, calib_proba)}

    for tier in config.TIERS:
        selected[tier] = {}
        raw_vs_calib[tier] = {}

        for model_name in SKLEARN_MODELS:
            entry = select_sklearn_candidate(model_name, tier)
            selected[tier][model_name] = entry["path"]
            proba, threshold = get_sklearn_proba(entry, test_df)
            raw_proba = get_raw_proba(entry, tier, model_name, test_df)
            raw_vs_calib[tier][model_name] = (raw_proba, proba)

            for thr_name, thr_val in [("0.5", 0.5), ("screening", threshold)]:
                m = compute_metrics(y_test, proba, threshold=thr_val)
                auc_lo, auc_hi = bootstrap_ci(y_test, proba, roc_auc_score)
                brier_lo, brier_hi = bootstrap_ci(y_test, proba, brier_score_loss)
                rec_lo, rec_hi = bootstrap_ci_recall(y_test, proba, thr_val)
                test_rows.append({
                    "tier": tier, "model": model_name, "calibration": entry["calibration"],
                    "threshold_type": thr_name, "threshold": thr_val, **m,
                    "roc_auc_ci_lo": auc_lo, "roc_auc_ci_hi": auc_hi,
                    "brier_ci_lo": brier_lo, "brier_ci_hi": brier_hi,
                    "recall_ci_lo": rec_lo, "recall_ci_hi": rec_hi,
                })

                cm = confusion_matrix(y_test, (proba >= thr_val).astype(int), labels=[0, 1])
                fig, ax = new_fig((4, 4))
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                            xticklabels=["No PCOS", "PCOS"], yticklabels=["No PCOS", "PCOS"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"{model_name} ({tier}, threshold={thr_name})")
                save_fig(fig, OUT / f"confusion_matrix_{tier}_{model_name}_{thr_name}.png")

        # --- Bayesian Network ---
        evidence_vars = config.BN_TIER_EVIDENCE[tier]
        bn_proba = bn_clf.predict_proba(test_df, evidence_vars)
        bn_threshold = bn_artifact["thresholds"][tier]
        selected[tier]["bayesian_network"] = "models/bayesian_network/bn_model.joblib"

        for thr_name, thr_val in [("0.5", 0.5), ("screening", bn_threshold)]:
            m = compute_metrics(y_test, bn_proba, threshold=thr_val)
            auc_lo, auc_hi = bootstrap_ci(y_test, bn_proba, roc_auc_score)
            brier_lo, brier_hi = bootstrap_ci(y_test, bn_proba, brier_score_loss)
            rec_lo, rec_hi = bootstrap_ci_recall(y_test, bn_proba, thr_val)
            test_rows.append({
                "tier": tier, "model": "bayesian_network", "calibration": "n/a",
                "threshold_type": thr_name, "threshold": thr_val, **m,
                "roc_auc_ci_lo": auc_lo, "roc_auc_ci_hi": auc_hi,
                "brier_ci_lo": brier_lo, "brier_ci_hi": brier_hi,
                "recall_ci_lo": rec_lo, "recall_ci_hi": rec_hi,
            })
            cm = confusion_matrix(y_test, (bn_proba >= thr_val).astype(int), labels=[0, 1])
            fig, ax = new_fig((4, 4))
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["No PCOS", "PCOS"], yticklabels=["No PCOS", "PCOS"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"bayesian_network ({tier}, threshold={thr_name})")
            save_fig(fig, OUT / f"confusion_matrix_{tier}_bayesian_network_{thr_name}.png")

        # App default: lowest OOF Brier among NB/LR/RF/BN for this tier
        oof_briers = {m: select_sklearn_candidate(m, tier)["oof_brier"] for m in SKLEARN_MODELS}
        oof_briers["bayesian_network"] = bn_artifact["oof_brier"][tier]
        app_default = min(oof_briers, key=oof_briers.get)
        selected[tier]["app_default"] = app_default

        # --- Per-tier plots: ROC, PR, calibration (all 4 models) ---
        all_probas = {m: raw_vs_calib[tier][m][1] for m in SKLEARN_MODELS}
        all_probas["bayesian_network"] = bn_proba

        fig, ax = new_fig((6, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        for m, proba in all_probas.items():
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc = roc_auc_score(y_test, proba)
            ax.plot(fpr, tpr, label=f"{m} (AUC={auc:.2f})")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC curves ({tier})")
        ax.legend()
        save_fig(fig, OUT / f"roc_curves_{tier}.png")

        fig, ax = new_fig((6, 6))
        for m, proba in all_probas.items():
            prec, rec, _ = precision_recall_curve(y_test, proba)
            ax.plot(rec, prec, label=m)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall curves ({tier})")
        ax.legend()
        save_fig(fig, OUT / f"pr_curves_{tier}.png")

        fig, ax = new_fig((6, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
        for m, proba in all_probas.items():
            frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=5, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", label=m)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration curves ({tier}, test set)")
        ax.legend()
        save_fig(fig, OUT / f"calibration_curves_{tier}.png")

        fig, ax = new_fig((6, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
        for m in SKLEARN_MODELS:
            raw_p, calib_p = raw_vs_calib[tier][m]
            fr_raw, mp_raw = calibration_curve(y_test, raw_p, n_bins=5, strategy="quantile")
            fr_cal, mp_cal = calibration_curve(y_test, calib_p, n_bins=5, strategy="quantile")
            ax.plot(mp_raw, fr_raw, marker="x", linestyle="--", label=f"{m} (raw)")
            ax.plot(mp_cal, fr_cal, marker="o", label=f"{m} (calibrated)")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Raw vs calibrated ({tier}, test set)")
        ax.legend(fontsize=7)
        save_fig(fig, OUT / f"calibration_raw_vs_calibrated_{tier}.png")

    save_json(selected, config.RESULTS_DIR / "selected_models.json")

    test_df_all = pd.DataFrame(test_rows)
    save_csv_and_md(test_df_all, OUT / "test_results_all")

    # --- Headline figures: tier comparison ---
    screening_rows = test_df_all[test_df_all["threshold_type"] == "screening"]
    fig, ax = new_fig((8, 6))
    for model_name in SKLEARN_MODELS + ["bayesian_network"]:
        sub = screening_rows[screening_rows["model"] == model_name].set_index("tier").reindex(config.TIERS)
        ax.errorbar(config.TIERS, sub["roc_auc"],
                    yerr=[sub["roc_auc"] - sub["roc_auc_ci_lo"], sub["roc_auc_ci_hi"] - sub["roc_auc"]],
                    marker="o", capsize=4, label=model_name)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Tier comparison: ROC-AUC (95% bootstrap CI)")
    ax.legend()
    save_fig(fig, OUT / "tier_comparison.png")

    fig, ax = new_fig((8, 6))
    for model_name in SKLEARN_MODELS + ["bayesian_network"]:
        sub = screening_rows[screening_rows["model"] == model_name].set_index("tier").reindex(config.TIERS)
        ax.errorbar(config.TIERS, sub["brier"],
                    yerr=[sub["brier"] - sub["brier_ci_lo"], sub["brier_ci_hi"] - sub["brier"]],
                    marker="o", capsize=4, label=model_name)
    ax.set_ylabel("Test Brier score")
    ax.set_title("Tier comparison: Brier score (95% bootstrap CI)")
    ax.legend()
    save_fig(fig, OUT / "tier_comparison_brier.png")

    # --- Literature comparison ---
    literature = [
        ("Aggarwal et al. (2023)", "Kaggle, feature selection + RF", "93.52% accuracy"),
        ("Various", "Kaggle; RF, MLP, Linear SVM", "85-93% accuracy"),
        ("Indian J. Community Health", "NB vs DT, 200 cases", "NB 81% accuracy, F1 0.81"),
        ("Frontiers in Endocrinology (2024)", "EHR data; RF, GBT, SVM, LR", "AUC 0.774-0.85"),
        ("Some later studies", "Kaggle, aggressive selection", "99.3% (likely overfitting)"),
    ]
    tier3_rf = screening_rows[(screening_rows["model"] == "random_forest") & (screening_rows["tier"] == "tier3_full")]
    our_acc = float(tier3_rf["accuracy"].iloc[0]) * 100 if len(tier3_rf) else float("nan")
    our_auc = float(tier3_rf["roc_auc"].iloc[0]) if len(tier3_rf) else float("nan")
    with open(OUT / "literature_comparison.md", "w") as f:
        f.write("# Literature comparison (Tier 3, Random Forest, screening threshold)\n\n")
        f.write(f"Our Tier 3 test accuracy: {our_acc:.2f}%. Our Tier 3 test ROC-AUC: {our_auc:.3f}.\n\n")
        f.write("| Source | Setting | Reported |\n|---|---|---|\n")
        for src, setting, reported in literature:
            f.write(f"| {src} | {setting} | {reported} |\n")
        f.write("\n")
        if our_acc < 88 or our_acc > 98:
            f.write(f"**Note:** our accuracy ({our_acc:.2f}%) differs from the typical reported range "
                    "(85-93%) by more than 5 points; review for leakage or overly aggressive tuning.\n")
        else:
            f.write("Our result falls within the range reported by prior Kaggle-based studies.\n")

    # --- Console summary ---
    print("\n=== Best model per tier (by test ROC-AUC, screening threshold) ===")
    for tier in config.TIERS:
        sub = screening_rows[screening_rows["tier"] == tier]
        best_row = sub.loc[sub["roc_auc"].idxmax()]
        print(f"{tier}: {best_row['model']} (AUC={best_row['roc_auc']:.3f}, Brier={best_row['brier']:.3f})")

    tier1_auc = screening_rows[(screening_rows["tier"] == "tier1_screening")]["roc_auc"].max()
    tier3_auc = screening_rows[(screening_rows["tier"] == "tier3_full")]["roc_auc"].max()
    print(f"\nTier 3 -> Tier 1 AUC drop: {tier3_auc - tier1_auc:.3f} "
          f"(Tier3 best={tier3_auc:.3f}, Tier1 best={tier1_auc:.3f})")

    print("\nStep 8 (08_evaluate.py) complete.")


if __name__ == "__main__":
    main()
