"""Metric functions, bootstrap confidence intervals, and threshold selection."""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def compute_metrics(y_true, y_proba, threshold=0.5):
    """Compute the full metric set at a given probability threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def select_recall_threshold(y_true, y_proba, recall_target=0.90):
    """Highest threshold whose recall >= recall_target.

    If no threshold reaches the target, fall back to the threshold with the
    maximum achievable recall and log a warning to the caller via return flag.
    """
    thresholds = np.unique(y_proba)
    thresholds = np.sort(thresholds)
    best_threshold = None
    best_recall_at_target = -1
    achieved_target = False

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        r = recall_score(y_true, y_pred, zero_division=0)
        if r >= recall_target:
            if best_threshold is None or t > best_threshold:
                best_threshold = t
                achieved_target = True

    if best_threshold is not None:
        return float(best_threshold), True

    # Fallback: threshold with maximum recall (ties broken by highest threshold)
    best_t = 0.0
    best_r = -1
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        r = recall_score(y_true, y_pred, zero_division=0)
        if r > best_r or (r == best_r and t > best_t):
            best_r = r
            best_t = t
    return float(best_t), False


def bootstrap_ci(y_true, y_proba, metric_fn, n_resamples=1000, seed=42, alpha=0.05):
    """Stratified bootstrap CI for a metric_fn(y_true, y_proba) -> float."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]

    values = []
    for _ in range(n_resamples):
        samp_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        samp_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        idx = np.concatenate([samp_pos, samp_neg])
        yt, yp = y_true[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            values.append(metric_fn(yt, yp))
        except ValueError:
            continue
    values = np.array(values)
    lower = np.percentile(values, 100 * alpha / 2)
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def bootstrap_ci_recall(y_true, y_proba, threshold, n_resamples=1000, seed=42, alpha=0.05):
    def fn(yt, yp):
        return recall_score(yt, (yp >= threshold).astype(int), zero_division=0)
    return bootstrap_ci(y_true, y_proba, fn, n_resamples, seed, alpha)
