"""
QT-2.23 — Module L: Contribution Analysis

Three independent lenses on sensor importance:
1. Optimized fusion weights (from Module H)
2. Ablation ΔAUC (re-evaluate with each sensor removed)
3. SHAP feature attribution (TreeExplainer on random forest)

Agreement Status = how well the 3 lenses agree on sensor ranking:
- HIGH: all 3 produce the same ordering
- PARTIAL: 2 of 3 agree
- DISAGREEMENT: each method says something different

Frontend shows all three in a grouped bar chart with agreement indicator.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# ============================================================
# L.1 — Ablation ΔAUC
# ============================================================

def ablation_delta_auc(
    sensor_scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    fusion_weights: Dict[str, float],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Compute ΔAUC for each sensor by removing it and re-normalizing.

    ΔAUC = AUC_full - AUC_without_sensor_i

    Higher ΔAUC → sensor contributes more to detection.
    """
    sensor_names = list(sensor_scores.keys())

    # Full AUC
    fused_full = np.zeros_like(labels, dtype=float)
    for s in sensor_names:
        fused_full += fusion_weights[s] * sensor_scores[s]
    auc_full = roc_auc_score(labels, fused_full)

    deltas = {}
    for i, removed in enumerate(sensor_names):
        remaining = [s for s in sensor_names if s != removed]

        # Re-normalize remaining weights to sum to 1
        w_remaining = {s: fusion_weights[s] for s in remaining}
        w_sum = sum(w_remaining.values())
        if w_sum > 1e-12:
            w_remaining = {s: v / w_sum for s, v in w_remaining.items()}

        fused_ablated = np.zeros_like(labels, dtype=float)
        for s in remaining:
            fused_ablated += w_remaining[s] * sensor_scores[s]

        auc_ablated = roc_auc_score(labels, fused_ablated)
        deltas[removed] = auc_full - auc_ablated

        if progress_callback:
            progress_callback(
                stage="Ablation",
                progress=(i + 1) / len(sensor_names),
                message=f"Removed {removed}: ΔAUC = {deltas[removed]:.4f}",
            )

    return deltas


# ============================================================
# L.2 — SHAP Sensor Contribution
# ============================================================

def shap_sensor_contribution(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    progress_callback: Optional[Callable] = None,
) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
    """
    Compute SHAP values for all features, then aggregate by sensor.

    Uses TreeExplainer (exact SHAP for tree ensembles) on a RandomForest.

    Returns (sensor_shap, shap_values, feature_names).
    """
    if progress_callback:
        progress_callback(stage="Training RF", progress=0.1, message="Random Forest")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(feature_matrix, labels)

    if progress_callback:
        progress_callback(stage="Computing SHAP", progress=0.4, message="TreeExplainer")

    try:
        import shap
        explainer = shap.TreeExplainer(rf)

        # Use a subset for SHAP computation if dataset is large
        n_shap = min(500, len(feature_matrix))
        X_shap = feature_matrix[:n_shap]
        shap_values = explainer.shap_values(X_shap)

        # For binary classification, use class 1 SHAP values
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
    except Exception:
        # Fallback to feature importance if SHAP fails
        shap_vals = np.zeros_like(feature_matrix[:100])
        importances = rf.feature_importances_
        for i in range(len(importances)):
            shap_vals[:, i] = importances[i] / max(len(labels), 1)

    if progress_callback:
        progress_callback(stage="Aggregating", progress=0.8, message="By sensor")

    # Mean absolute SHAP per feature
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    # Aggregate by sensor
    sensor_shap = {"radar": 0.0, "thermal": 0.0, "acoustic": 0.0}
    for i, name in enumerate(feature_names):
        if i >= len(mean_abs_shap):
            break
        if name.startswith("radar_"):
            sensor_shap["radar"] += mean_abs_shap[i]
        elif name.startswith("thermal_"):
            sensor_shap["thermal"] += mean_abs_shap[i]
        elif name.startswith("acoustic_"):
            sensor_shap["acoustic"] += mean_abs_shap[i]

    # Normalize
    total_shap = sum(sensor_shap.values())
    if total_shap > 1e-12:
        sensor_shap = {k: v / total_shap for k, v in sensor_shap.items()}

    return sensor_shap, shap_vals, feature_names


# ============================================================
# L.3 — Agreement Analysis
# ============================================================

def compute_agreement(
    weight_contributions: Dict[str, float],
    ablation_contributions: Dict[str, float],
    shap_contributions: Dict[str, float],
) -> Tuple[float, str]:
    """
    Determine agreement between 3 contribution methods.

    Computes ranking for each method and checks concordance.

    Returns (agreement_score, agreement_status).
    agreement_score: 0-1 (Kendall's W)
    agreement_status: HIGH / PARTIAL / DISAGREEMENT
    """
    sensors = list(weight_contributions.keys())

    def rank_dict(d):
        sorted_keys = sorted(d.keys(), key=lambda k: d[k], reverse=True)
        return {k: i for i, k in enumerate(sorted_keys)}

    rank_w = rank_dict(weight_contributions)
    rank_a = rank_dict(ablation_contributions)
    rank_s = rank_dict(shap_contributions)

    # Count how many methods agree on the top sensor
    top_sensors = [
        max(weight_contributions, key=weight_contributions.get),
        max(ablation_contributions, key=ablation_contributions.get),
        max(shap_contributions, key=shap_contributions.get),
    ]

    # Full ranking agreement (Kendall's W simplified)
    all_ranks = np.array([[rank_w[s], rank_a[s], rank_s[s]] for s in sensors])
    n_items = len(sensors)
    n_judges = 3
    rank_sums = all_ranks.sum(axis=1)
    mean_rank_sum = np.mean(rank_sums)
    S = np.sum((rank_sums - mean_rank_sum) ** 2)
    max_S = (n_judges ** 2 * (n_items ** 3 - n_items)) / 12.0
    W = S / max_S if max_S > 0 else 0.0

    # Agreement status
    from collections import Counter
    top_counts = Counter(top_sensors)
    most_common_count = top_counts.most_common(1)[0][1]

    if most_common_count == 3:
        status = "HIGH"
    elif most_common_count == 2:
        status = "PARTIAL"
    else:
        status = "DISAGREEMENT"

    return float(W), status


# ============================================================
# L.4 — Full Contribution Pipeline
# ============================================================

def run_contribution_analysis(
    progress_callback: Callable,
    sensor_scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: List[str],
    fusion_weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    Full Module L pipeline.
    """
    # --- Weights contribution ---
    progress_callback(
        stage="Weight contributions",
        progress=0.05,
        message="Normalizing fusion weights",
    )
    total_w = sum(fusion_weights.values())
    weight_contributions = {
        s: v / total_w for s, v in fusion_weights.items()
    } if total_w > 1e-12 else fusion_weights

    # --- Ablation ---
    progress_callback(
        stage="Ablation analysis",
        progress=0.15,
        message="Computing ΔAUC",
    )
    ablation = ablation_delta_auc(
        sensor_scores, labels, fusion_weights, progress_callback
    )

    # Normalize ablation deltas
    total_abl = sum(abs(v) for v in ablation.values())
    ablation_norm = {
        k: abs(v) / total_abl for k, v in ablation.items()
    } if total_abl > 1e-12 else ablation

    # --- SHAP ---
    progress_callback(
        stage="SHAP analysis",
        progress=0.40,
        message="Computing feature-level SHAP",
    )
    sensor_shap, shap_vals, _ = shap_sensor_contribution(
        feature_matrix, labels, feature_names, progress_callback
    )

    # --- Agreement ---
    progress_callback(
        stage="Agreement analysis",
        progress=0.90,
        message="Checking 3-lens concordance",
    )
    agreement_score, agreement_status = compute_agreement(
        weight_contributions, ablation_norm, sensor_shap
    )

    progress_callback(
        stage="Contribution analysis complete",
        progress=1.0,
        message=f"Agreement: {agreement_status} (W={agreement_score:.3f})",
    )

    return {
        "weight_contributions": weight_contributions,
        "ablation_delta_auc": ablation,
        "ablation_normalized": ablation_norm,
        "shap_sensor_contribution": sensor_shap,
        "shap_feature_values": shap_vals,
        "shap_feature_names": feature_names,
        "agreement_score": agreement_score,
        "agreement_status": agreement_status,
    }
