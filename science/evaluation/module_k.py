"""
QT-2.23 — Module K: Statistical Reliability Protocol

Provides:
- 5-fold stratified cross-validation
- Multi-seed experiment repetition (≥5 seeds)
- Bootstrap confidence intervals (n=1000, 95% CI)
- ROC/AUC computation
- Confusion matrix

Every headline number = mean ± std (across seeds) or point estimate
[95% CI] (bootstrapped AUC) — never a bare single number.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import StratifiedKFold


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 95.0,
    seed: int = 42,
    progress_callback: Optional[Callable] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for AUC.

    Returns (mean_auc, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []

    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
        except ValueError:
            continue

        if progress_callback and i % 200 == 0:
            progress_callback(
                stage="Bootstrap AUC",
                progress=i / n_bootstrap,
                message=f"Sample {i}/{n_bootstrap}",
            )

    if len(aucs) == 0:
        return 0.5, 0.5, 0.5

    aucs = np.array(aucs)
    lo = float(np.percentile(aucs, (100.0 - ci) / 2.0))
    hi = float(np.percentile(aucs, 100.0 - (100.0 - ci) / 2.0))
    return float(np.mean(aucs)), lo, hi


def run_kfold_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    pipeline_fn: Callable,
    n_splits: int = 5,
    seed: int = 42,
    progress_callback: Optional[Callable] = None,
) -> List[Dict[str, float]]:
    """
    5-fold stratified cross-validation.

    pipeline_fn(X_train, y_train, X_test, y_test) → dict of metrics
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if progress_callback:
            progress_callback(
                stage="Cross-validation",
                progress=fold / n_splits,
                message=f"Fold {fold+1}/{n_splits}",
            )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_result = pipeline_fn(X_train, y_train, X_test, y_test)
        results.append(fold_result)

    return results


def run_multi_seed_experiment(
    seeds: List[int],
    full_pipeline_fn: Callable,
    progress_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-seed experiment repetition.

    Returns (mean_metrics, std_metrics) across seeds.
    """
    all_metrics = []

    for i, seed in enumerate(seeds):
        if progress_callback:
            progress_callback(
                stage="Multi-seed",
                progress=i / len(seeds),
                message=f"Seed {seed} ({i+1}/{len(seeds)})",
            )

        np.random.seed(seed)
        metrics = full_pipeline_fn(seed)
        all_metrics.append(metrics)

    arr = np.array(all_metrics)
    return arr.mean(axis=0), arr.std(axis=0)


def compute_full_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    ci_level: float = 95.0,
    seed: int = 42,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Compute complete evaluation metrics with uncertainty.
    """
    # Binary predictions at threshold
    y_pred = (y_scores >= threshold).astype(int)

    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # AUC with bootstrap CI
    if progress_callback:
        progress_callback(stage="Computing AUC CI", progress=0.3, message="Bootstrap")

    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(
        y_true, y_scores, n_bootstrap, ci_level, seed, progress_callback
    )

    # Detection rate (recall/sensitivity)
    det_rate = float(recall_score(y_true, y_pred, zero_division=0))

    # False alarm rate
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Precision, recall, F1
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        "auc": {"mean": auc_mean, "ci_lower": auc_lo, "ci_upper": auc_hi},
        "detection_rate": det_rate,
        "false_alarm_rate": far,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "confusion_matrix": cm,
        "threshold": threshold,
    }
