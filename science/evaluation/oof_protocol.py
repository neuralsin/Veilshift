"""
QT-2.23 — Out-of-Fold Evaluation Protocol

Executes the strict 5-fold Stratified OOF evaluation to prevent data leakage.
See `docs/EVALUATION_PROTOCOL.md` for the full mathematical theory.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


# ============================================================
# Canonical Metric Computation
# ============================================================

@dataclass
class CanonicalMetrics:
    """Result from the single canonical metric function."""
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    detection_rate: float = 0.0  # TP / (TP + FN) = Recall = Pd
    false_alarm_rate: float = 0.0  # FP / (FP + TN)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc: float = 0.5
    roc_fpr: Optional[np.ndarray] = None
    roc_tpr: Optional[np.ndarray] = None
    roc_thresholds: Optional[np.ndarray] = None
    # Flags for undefined conditions
    recall_undefined: bool = False
    precision_undefined: bool = False
    far_undefined: bool = False


def compute_canonical_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> CanonicalMetrics:
    """
    Single canonical metric function. ALL pages and pipelines MUST use this.
    No duplicated metric formulas anywhere in the codebase.

    Parameters
    ----------
    y_true : (N,) binary labels
    y_scores : (N,) continuous scores
    threshold : decision threshold

    Returns
    -------
    CanonicalMetrics with all fields populated
    """
    y_pred = (y_scores >= threshold).astype(int)
    y_true = np.asarray(y_true, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    # Detection rate = Recall = Sensitivity = TP / (TP + FN)
    recall_undefined = (tp + fn) == 0
    detection_rate = tp / (tp + fn) if not recall_undefined else 0.0

    # False Alarm Rate = FP / (FP + TN)
    far_undefined = (fp + tn) == 0
    false_alarm_rate = fp / (fp + tn) if not far_undefined else 0.0

    # Precision = TP / (TP + FP)
    precision_undefined = (tp + fp) == 0
    precision = tp / (tp + fp) if not precision_undefined else 0.0

    recall = detection_rate

    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # ROC + AUC
    try:
        fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true, y_scores)
        auc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        fpr_arr = np.array([0.0, 1.0])
        tpr_arr = np.array([0.0, 1.0])
        thresh_arr = np.array([1.0, 0.0])
        auc = 0.5

    return CanonicalMetrics(
        tp=tp, tn=tn, fp=fp, fn=fn,
        detection_rate=detection_rate,
        false_alarm_rate=false_alarm_rate,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        roc_fpr=fpr_arr,
        roc_tpr=tpr_arr,
        roc_thresholds=thresh_arr,
        recall_undefined=recall_undefined,
        precision_undefined=precision_undefined,
        far_undefined=far_undefined,
    )


# ============================================================
# Bootstrap CI on OOF arrays
# ============================================================

def bootstrap_oof_auc_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 95.0,
    seed: int = 42,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Bootstrap AUC confidence interval on OUT-OF-FOLD predictions.
    Single-class replicates are rejected.

    Returns dict with mean, ci_lower, ci_upper, n_valid, n_rejected, method, seed.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    n_rejected = 0

    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            n_rejected += 1
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
        except ValueError:
            n_rejected += 1
            continue

        if progress_callback and i % 200 == 0:
            progress_callback(
                stage="Bootstrap OOF AUC",
                progress=i / n_bootstrap,
                message=f"Replicate {i}/{n_bootstrap}",
            )

    if len(aucs) == 0:
        return {
            "mean": 0.5, "ci_lower": 0.5, "ci_upper": 0.5,
            "n_valid": 0, "n_rejected": n_bootstrap,
            "method": "percentile", "seed": seed,
        }

    aucs = np.array(aucs)
    alpha = (100.0 - ci_level) / 2.0
    lo = float(np.percentile(aucs, alpha))
    hi = float(np.percentile(aucs, 100.0 - alpha))

    return {
        "mean": float(np.mean(aucs)),
        "ci_lower": lo,
        "ci_upper": hi,
        "n_valid": len(aucs),
        "n_rejected": n_rejected,
        "method": "percentile",
        "seed": seed,
    }


# ============================================================
# OOF Fold Result
# ============================================================

@dataclass
class OOFFoldResult:
    """Result from a single outer fold."""
    fold_idx: int = 0
    train_indices: Optional[np.ndarray] = None
    test_indices: Optional[np.ndarray] = None
    inner_train_indices: Optional[np.ndarray] = None
    inner_val_indices: Optional[np.ndarray] = None

    # Per-fold outputs
    selected_feature_indices: Optional[List[int]] = None
    selected_feature_names: Optional[List[str]] = None
    fusion_weights: Optional[Dict[str, float]] = None
    fusion_weights_array: Optional[np.ndarray] = None
    threshold: float = 0.5
    fisher_objective: float = 0.0

    # Per-fold sensor scores on outer test
    test_sensor_scores: Optional[Dict[str, np.ndarray]] = None
    test_fused_scores: Optional[np.ndarray] = None
    test_predictions: Optional[np.ndarray] = None
    test_labels: Optional[np.ndarray] = None

    # Per-fold metrics (from outer test)
    fold_metrics: Optional[CanonicalMetrics] = None


# ============================================================
# OOF Evaluation Result
# ============================================================

@dataclass
class OOFEvaluationResult:
    """
    Complete OOF evaluation result. This is the CANONICAL source of
    truth for all scientific performance claims in the UI.
    """
    # Sample-level OOF arrays (in original sample order)
    sample_indices: Optional[np.ndarray] = None
    true_labels: Optional[np.ndarray] = None
    fold_ids: Optional[np.ndarray] = None
    sensor_scores: Optional[Dict[str, np.ndarray]] = None
    fused_scores: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None

    # Per-fold details
    fold_results: List[OOFFoldResult] = field(default_factory=list)
    fold_selected_features: Optional[List[List[str]]] = None
    fold_fusion_weights: Optional[List[Dict[str, float]]] = None
    fold_thresholds: Optional[List[float]] = None
    fold_metrics: Optional[List[CanonicalMetrics]] = None

    # Aggregate metrics (from pooled OOF predictions)
    aggregate_metrics: Optional[CanonicalMetrics] = None
    roc_fpr: Optional[np.ndarray] = None
    roc_tpr: Optional[np.ndarray] = None
    roc_thresholds: Optional[np.ndarray] = None
    auc: Optional[float] = None

    # Bootstrap CI
    bootstrap_ci: Optional[Dict[str, Any]] = None

    # Feature selection stability
    feature_selection_frequency: Optional[Dict[str, int]] = None
    consensus_features: Optional[List[str]] = None

    # Fusion weight statistics
    mean_fusion_weights: Optional[Dict[str, float]] = None
    std_fusion_weights: Optional[Dict[str, float]] = None

    # Protocol metadata
    seed: int = 42
    n_splits: int = 5
    evaluation_protocol: str = "5-FOLD STRATIFIED OOF"
    n_bootstrap: int = 1000
    ci_level: float = 0.95

    def is_valid(self) -> bool:
        """Check if this is a valid OOF result (not legacy leaked)."""
        return (
            self.sample_indices is not None
            and self.true_labels is not None
            and self.fused_scores is not None
            and self.evaluation_protocol == "5-FOLD STRATIFIED OOF"
        )


# ============================================================
# Scientific Integrity Validation
# ============================================================

class ScientificIntegrityError(Exception):
    """Raised when a scientific integrity check fails."""
    pass


def validate_oof_integrity(result: OOFEvaluationResult, n_samples: int):
    """
    Runtime scientific integrity checks. Raises ScientificIntegrityError
    on any violation. Does NOT use assert (survives python -O).
    """
    if result.sample_indices is None:
        raise ScientificIntegrityError("sample_indices is None")
    if result.true_labels is None:
        raise ScientificIntegrityError("true_labels is None")
    if result.fused_scores is None:
        raise ScientificIntegrityError("fused_scores is None")

    # Every sample appears exactly once
    unique_indices = np.unique(result.sample_indices)
    if len(unique_indices) != n_samples:
        raise ScientificIntegrityError(
            f"Expected {n_samples} unique sample indices, got {len(unique_indices)}"
        )
    expected = np.arange(n_samples)
    if not np.array_equal(np.sort(result.sample_indices), expected):
        raise ScientificIntegrityError("Sample indices do not cover expected range")

    # No duplicate predictions
    if len(result.sample_indices) != n_samples:
        raise ScientificIntegrityError(
            f"len(sample_indices)={len(result.sample_indices)} != n_samples={n_samples}"
        )

    # No NaN OOF scores
    if np.any(np.isnan(result.fused_scores)):
        raise ScientificIntegrityError("NaN found in OOF fused scores")

    # Fold train/test disjointness
    for fr in result.fold_results:
        if fr.train_indices is not None and fr.test_indices is not None:
            overlap = np.intersect1d(fr.train_indices, fr.test_indices)
            if len(overlap) > 0:
                raise ScientificIntegrityError(
                    f"Fold {fr.fold_idx}: train/test overlap of {len(overlap)} samples"
                )

    # Fusion weights
    if result.fold_fusion_weights:
        for fold_idx, fw in enumerate(result.fold_fusion_weights):
            w_arr = np.array(list(fw.values()))
            if np.any(w_arr < -1e-6):
                raise ScientificIntegrityError(
                    f"Fold {fold_idx}: negative fusion weight {w_arr}"
                )
            if abs(np.sum(w_arr) - 1.0) > 0.01:
                raise ScientificIntegrityError(
                    f"Fold {fold_idx}: fusion weights sum to {np.sum(w_arr):.4f}, not 1.0"
                )

    # AUC recomputation
    if result.auc is not None:
        try:
            recomputed = roc_auc_score(result.true_labels, result.fused_scores)
            if abs(recomputed - result.auc) > 1e-6:
                raise ScientificIntegrityError(
                    f"Stored AUC {result.auc:.6f} != recomputed {recomputed:.6f}"
                )
        except ValueError:
            pass


# ============================================================
# OOF Evaluation Engine
# ============================================================

def _fuse_scores(
    sensor_scores: Dict[str, np.ndarray],
    weights: np.ndarray,
    sensor_names: List[str],
) -> np.ndarray:
    """Apply fusion weights to sensor scores."""
    fused = np.zeros(len(sensor_scores[sensor_names[0]]), dtype=float)
    for i, s in enumerate(sensor_names):
        fused += weights[i] * sensor_scores[s]
    return fused


def run_oof_evaluation(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    sensor_feature_counts: Dict[str, int],
    n_splits: int = 5,
    seed: int = 42,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    # Feature selection params
    k_target: int = 6,
    fs_alpha: float = 1.0,
    fs_beta: float = 1.0,
    fs_gamma: float = 2.0,
    # Fusion params
    lam: float = 0.5,
    n_restarts: int = 10,
    target_far: float = 0.01,
    progress_callback: Optional[Callable] = None,
) -> OOFEvaluationResult:
    """
    5-Fold Stratified Out-of-Fold Evaluation Protocol.

    For each outer fold:
      1. Split outer_train / outer_test
      2. Inner split: 80/20 from outer_train for threshold selection
      3. On inner_train: feature selection + classifier fitting + fusion optimization
      4. On inner_val: threshold selection
      5. Refit on full outer_train with selected config
      6. On outer_test: score, fuse, predict, store

    Returns OOFEvaluationResult with pooled predictions in original sample order.
    """
    from science.qubo.module_g import build_feature_selection_qubo, brute_force_feature_selection
    from science.fusion.module_h import (
        compute_scatter_matrices, optimize_fusion_continuous,
        select_threshold_neyman_pearson,
    )

    n_samples = len(labels)
    sensor_names = list(sensor_feature_counts.keys())
    n_sensors = len(sensor_names)

    # Build sensor feature slices
    sensor_slices = {}
    offset = 0
    for s in sensor_names:
        n_feat = sensor_feature_counts[s]
        sensor_slices[s] = slice(offset, offset + n_feat)
        offset += n_feat

    # Outer CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Allocate OOF arrays
    oof_fused_scores = np.full(n_samples, np.nan)
    oof_predictions = np.full(n_samples, -1, dtype=int)
    oof_fold_ids = np.full(n_samples, -1, dtype=int)
    oof_sensor_scores = {s: np.full(n_samples, np.nan) for s in sensor_names}

    fold_results = []
    all_selected_features = []
    all_fusion_weights = []
    all_thresholds = []
    all_fold_metrics = []

    if progress_callback:
        progress_callback(
            stage="OOF Evaluation",
            progress=0.0,
            message=f"Starting {n_splits}-fold stratified OOF evaluation",
        )

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(feature_matrix, labels)):
        fold_start = time.time()

        if progress_callback:
            progress_callback(
                stage="OOF Evaluation",
                progress=fold_idx / n_splits,
                message=f"Fold {fold_idx + 1}/{n_splits}",
            )

        X_outer_train = feature_matrix[train_idx]
        y_outer_train = labels[train_idx]
        X_outer_test = feature_matrix[test_idx]
        y_outer_test = labels[test_idx]

        # ----- Inner split for threshold selection -----
        inner_seed = seed * 1000 + fold_idx
        inner_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=inner_seed
        )
        inner_train_rel, inner_val_rel = next(
            inner_splitter.split(X_outer_train, y_outer_train)
        )
        # Convert to absolute indices for logging
        inner_train_abs = train_idx[inner_train_rel]
        inner_val_abs = train_idx[inner_val_rel]

        X_inner_train = X_outer_train[inner_train_rel]
        y_inner_train = y_outer_train[inner_train_rel]
        X_inner_val = X_outer_train[inner_val_rel]
        y_inner_val = y_outer_train[inner_val_rel]

        # ----- Feature selection on inner_train ONLY -----
        Q, relevance, redundancy = build_feature_selection_qubo(
            X_inner_train, y_inner_train, k_target, fs_alpha, fs_beta, fs_gamma
        )
        n_feat = X_inner_train.shape[1]
        if n_feat <= 20:
            bf_x, bf_energy, bf_indices = brute_force_feature_selection(
                Q, n_feat, k_target
            )
            selected_indices = bf_indices
        else:
            selected_indices = list(range(min(k_target, n_feat)))

        selected_names = [feature_names[i] for i in selected_indices] if feature_names else []

        # ----- Map globally-selected indices onto each sensor's own columns -----
        # QUBO selection runs over the full (radar+thermal+acoustic+cross) matrix,
        # but each sensor's classifier should only see ITS OWN selected columns.
        # If QUBO picked none of a sensor's features this fold, fall back to that
        # sensor's full slice so the classifier still has something to fit on.
        selected_set = set(selected_indices)
        sensor_local_cols = {}
        for s in sensor_names:
            sl = sensor_slices[s]
            local = [i - sl.start for i in selected_set if sl.start <= i < sl.stop]
            sensor_local_cols[s] = sorted(local) if local else list(range(sl.stop - sl.start))

        # ----- Fit sensor classifiers on inner_train ONLY, using selected features -----
        inner_sensor_scores_train = {}
        inner_sensor_scores_val = {}
        classifiers = {}

        for s in sensor_names:
            sl = sensor_slices[s]
            cols = sensor_local_cols[s]
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(X_inner_train[:, sl][:, cols], y_inner_train)
            classifiers[s] = clf
            inner_sensor_scores_train[s] = clf.predict_proba(X_inner_train[:, sl][:, cols])[:, 1]
            inner_sensor_scores_val[s] = clf.predict_proba(X_inner_val[:, sl][:, cols])[:, 1]

        # ----- Fusion optimization on inner_train scores -----
        train_scores_matrix = np.column_stack(
            [inner_sensor_scores_train[s] for s in sensor_names]
        )
        mask_h1_train = y_inner_train == 1
        mask_h0_train = y_inner_train == 0

        S_b, S_w = compute_scatter_matrices(
            train_scores_matrix[mask_h1_train],
            train_scores_matrix[mask_h0_train],
        )

        w_fold, obj_fold, _, solver_fold = optimize_fusion_continuous(
            S_b, S_w, n_sensors, n_restarts, lam, seed + fold_idx
        )

        # ----- Threshold selection on inner_val ONLY -----
        fused_inner_val = _fuse_scores(inner_sensor_scores_val, w_fold, sensor_names)
        threshold_fold, _, _ = select_threshold_neyman_pearson(
            fused_inner_val, y_inner_val, target_far
        )

        # ----- Refit classifiers on full outer_train (same selected features) -----
        for s in sensor_names:
            sl = sensor_slices[s]
            cols = sensor_local_cols[s]
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(X_outer_train[:, sl][:, cols], y_outer_train)
            classifiers[s] = clf

        # ----- Score outer_test with refitted classifiers -----
        test_sensor_scores_fold = {}
        for s in sensor_names:
            sl = sensor_slices[s]
            cols = sensor_local_cols[s]
            test_sensor_scores_fold[s] = classifiers[s].predict_proba(
                X_outer_test[:, sl][:, cols]
            )[:, 1]

        # Apply fusion weights (from inner optimization) + threshold
        fused_test = _fuse_scores(test_sensor_scores_fold, w_fold, sensor_names)
        preds_test = (fused_test >= threshold_fold).astype(int)

        # ----- Store in OOF arrays -----
        oof_fused_scores[test_idx] = fused_test
        oof_predictions[test_idx] = preds_test
        oof_fold_ids[test_idx] = fold_idx
        for s in sensor_names:
            oof_sensor_scores[s][test_idx] = test_sensor_scores_fold[s]

        # ----- Per-fold metrics -----
        fold_metrics = compute_canonical_metrics(y_outer_test, fused_test, threshold_fold)

        fold_weights_dict = {s: float(w_fold[i]) for i, s in enumerate(sensor_names)}

        fold_result = OOFFoldResult(
            fold_idx=fold_idx,
            train_indices=train_idx,
            test_indices=test_idx,
            inner_train_indices=inner_train_abs,
            inner_val_indices=inner_val_abs,
            selected_feature_indices=selected_indices,
            selected_feature_names=selected_names,
            fusion_weights=fold_weights_dict,
            fusion_weights_array=w_fold.copy(),
            threshold=threshold_fold,
            fisher_objective=obj_fold,
            test_sensor_scores=test_sensor_scores_fold,
            test_fused_scores=fused_test,
            test_predictions=preds_test,
            test_labels=y_outer_test,
            fold_metrics=fold_metrics,
        )
        fold_results.append(fold_result)
        all_selected_features.append(selected_names)
        all_fusion_weights.append(fold_weights_dict)
        all_thresholds.append(threshold_fold)
        all_fold_metrics.append(fold_metrics)

        if progress_callback:
            progress_callback(
                stage="OOF Evaluation",
                progress=(fold_idx + 0.9) / n_splits,
                message=f"Fold {fold_idx + 1} AUC={fold_metrics.auc:.4f} "
                        f"({time.time() - fold_start:.1f}s)",
            )

    # ========================================================
    # Aggregate: pooled OOF predictions
    # ========================================================
    sample_indices = np.arange(n_samples)

    # Compute aggregate metrics from pooled OOF
    aggregate_metrics = compute_canonical_metrics(
        labels, oof_fused_scores, np.median(all_thresholds)
    )

    # Bootstrap CI on OOF arrays
    if progress_callback:
        progress_callback(
            stage="Bootstrap CI",
            progress=0.95,
            message=f"{n_bootstrap} bootstrap replicates on OOF predictions",
        )

    bootstrap_seed = seed * 10 + 7  # Deterministic derivation
    bootstrap_ci = bootstrap_oof_auc_ci(
        labels, oof_fused_scores,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level * 100 if ci_level < 1.0 else ci_level,
        seed=bootstrap_seed,
        progress_callback=progress_callback,
    )

    # Feature selection stability
    feature_freq = {}
    for fold_feats in all_selected_features:
        for f in fold_feats:
            feature_freq[f] = feature_freq.get(f, 0) + 1
    consensus_threshold = max(1, int(n_splits * 0.6))  # >= 60%
    consensus_features = [
        f for f, count in feature_freq.items() if count >= consensus_threshold
    ]

    # Fusion weight statistics
    mean_weights = {}
    std_weights = {}
    for s in sensor_names:
        w_vals = [fw[s] for fw in all_fusion_weights]
        mean_weights[s] = float(np.mean(w_vals))
        std_weights[s] = float(np.std(w_vals))

    result = OOFEvaluationResult(
        sample_indices=sample_indices,
        true_labels=labels.copy(),
        fold_ids=oof_fold_ids,
        sensor_scores=oof_sensor_scores,
        fused_scores=oof_fused_scores,
        predictions=oof_predictions,
        fold_results=fold_results,
        fold_selected_features=all_selected_features,
        fold_fusion_weights=all_fusion_weights,
        fold_thresholds=all_thresholds,
        fold_metrics=all_fold_metrics,
        aggregate_metrics=aggregate_metrics,
        roc_fpr=aggregate_metrics.roc_fpr,
        roc_tpr=aggregate_metrics.roc_tpr,
        roc_thresholds=aggregate_metrics.roc_thresholds,
        auc=aggregate_metrics.auc,
        bootstrap_ci=bootstrap_ci,
        feature_selection_frequency=feature_freq,
        consensus_features=consensus_features,
        mean_fusion_weights=mean_weights,
        std_fusion_weights=std_weights,
        seed=seed,
        n_splits=n_splits,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
    )

    # Validate integrity
    validate_oof_integrity(result, n_samples)

    if progress_callback:
        progress_callback(
            stage="OOF Evaluation Complete",
            progress=1.0,
            message=f"OOF AUC={aggregate_metrics.auc:.4f} "
                    f"[{bootstrap_ci['ci_lower']:.4f}–{bootstrap_ci['ci_upper']:.4f}]",
        )

    return result


# ============================================================
# OOF Baseline Evaluation (same folds)
# ============================================================

def run_oof_baselines(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    sensor_feature_counts: Dict[str, int],
    fold_assignments: List[Tuple[np.ndarray, np.ndarray]],
    crlb_weights: np.ndarray,
    lam: float = 0.5,
    n_restarts: int = 10,
    seed: int = 42,
    target_far: float = 0.01,
    progress_callback: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """
    Run all baseline methods using the SAME outer fold assignments.
    Each produces OOF predictions. Primary metrics from pooled OOF.
    """
    from science.fusion.module_h import (
        compute_scatter_matrices, optimize_fusion_continuous,
        select_threshold_neyman_pearson,
    )
    from science.baselines.module_f import (
        kalman_fuse, bayesian_fuse, optimize_weights_de,
    )

    sensor_names = list(sensor_feature_counts.keys())
    n_sensors = len(sensor_names)
    n_samples = len(labels)

    # Build sensor slices
    sensor_slices = {}
    offset = 0
    for s in sensor_names:
        n_feat = sensor_feature_counts[s]
        sensor_slices[s] = slice(offset, offset + n_feat)
        offset += n_feat

    n_splits = len(fold_assignments)

    # Define baseline methods
    baseline_configs = [
        {"name": "CRLB-Weighted", "method": "crlb"},
        {"name": "Kalman Fusion", "method": "kalman"},
        {"name": "Bayesian Sequential", "method": "bayesian"},
        {"name": "SLSQP Optimized", "method": "slsqp"},
        {"name": "Differential Evolution", "method": "de"},
    ]

    results = []

    for bi, bc in enumerate(baseline_configs):
        if progress_callback:
            progress_callback(
                stage=f"Baseline {bi + 1}/5: {bc['name']}",
                progress=bi / 5,
                message=f"OOF evaluation",
            )

        oof_scores = np.full(n_samples, np.nan)
        fold_weights_list = []
        fold_thresholds = []
        t_start = time.time()

        for fold_idx, (train_idx, test_idx) in enumerate(fold_assignments):
            X_train = feature_matrix[train_idx]
            y_train = labels[train_idx]
            X_test = feature_matrix[test_idx]

            # Train sensor classifiers on train fold
            train_sensor_scores = {}
            test_sensor_scores = {}
            for s in sensor_names:
                sl = sensor_slices[s]
                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(X_train[:, sl], y_train)
                train_sensor_scores[s] = clf.predict_proba(X_train[:, sl])[:, 1]
                test_sensor_scores[s] = clf.predict_proba(X_test[:, sl])[:, 1]

            # Get fusion weights for this baseline method on train data
            train_scores_matrix = np.column_stack(
                [train_sensor_scores[s] for s in sensor_names]
            )
            mask_h1 = y_train == 1
            mask_h0 = y_train == 0

            if bc["method"] == "crlb":
                w_fold = crlb_weights.copy()

            elif bc["method"] == "kalman":
                noise_vars = [float(np.var(train_sensor_scores[s])) for s in sensor_names]
                w_raw = np.array([1.0 / max(v, 1e-12) for v in noise_vars])
                w_fold = w_raw / w_raw.sum()

            elif bc["method"] == "bayesian":
                # Bayesian has implicit equal weights; fuse differently
                fused_train = bayesian_fuse(
                    0.0, [train_sensor_scores[s] for s in sensor_names]
                )
                fused_test_fold = bayesian_fuse(
                    0.0, [test_sensor_scores[s] for s in sensor_names]
                )
                # Select threshold on train
                thr, _, _ = select_threshold_neyman_pearson(
                    fused_train, y_train, target_far
                )
                oof_scores[test_idx] = fused_test_fold
                fold_weights_list.append({s: 1.0 / n_sensors for s in sensor_names})
                fold_thresholds.append(thr)
                continue

            elif bc["method"] == "slsqp":
                S_b, S_w = compute_scatter_matrices(
                    train_scores_matrix[mask_h1], train_scores_matrix[mask_h0]
                )
                w_fold, _, _, _ = optimize_fusion_continuous(
                    S_b, S_w, n_sensors, n_restarts, lam, seed + fold_idx
                )

            elif bc["method"] == "de":
                S_b, S_w = compute_scatter_matrices(
                    train_scores_matrix[mask_h1], train_scores_matrix[mask_h0]
                )
                w_fold, _, _ = optimize_weights_de(
                    S_b, S_w, n_sensors, lam, seed + fold_idx
                )

            else:
                w_fold = np.ones(n_sensors) / n_sensors

            # Fuse and threshold
            fused_train = _fuse_scores(train_sensor_scores, w_fold, sensor_names)
            thr, _, _ = select_threshold_neyman_pearson(fused_train, y_train, target_far)

            fused_test_fold = _fuse_scores(test_sensor_scores, w_fold, sensor_names)
            oof_scores[test_idx] = fused_test_fold

            fold_weights_list.append({s: float(w_fold[i]) for i, s in enumerate(sensor_names)})
            fold_thresholds.append(thr)

        t_total = time.time() - t_start

        # Compute aggregate metrics from pooled OOF
        median_thr = float(np.median(fold_thresholds)) if fold_thresholds else 0.5
        metrics = compute_canonical_metrics(labels, oof_scores, median_thr)

        # Mean weights
        mean_w = {}
        for s in sensor_names:
            vals = [fw[s] for fw in fold_weights_list]
            mean_w[s] = float(np.mean(vals))

        results.append({
            "method_name": bc["name"],
            "weights": mean_w,
            "objective_value": None,
            "solve_time_s": t_total,
            "auc": metrics.auc,
            "fused_scores": oof_scores,
            "threshold": median_thr,
            "oof_metrics": metrics,
            "fold_weights": fold_weights_list,
            "fold_thresholds": fold_thresholds,
        })

    if progress_callback:
        progress_callback(
            stage="All baselines complete",
            progress=1.0,
            message=f"5 methods evaluated with OOF protocol",
        )

    return results
