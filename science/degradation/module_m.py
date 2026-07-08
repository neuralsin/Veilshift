"""
QT-2.23 — Module M: Degradation Sweep + Dynamic Trust Migration

The hero demonstration module. Shows:
1. What happens to detection performance when a sensor degrades
2. How adaptive fusion (re-optimized weights) retains more detection
   performance than static fusion (frozen weights)
3. Trust migration: weight shifts from degraded to healthy sensors
4. [Section 3.4] Live contribution recomputation at every severity step

This is the single most important visual inference in the application:
"As a sensor degrades, its optimized trust weight changes, the
contribution hierarchy changes, and we measure whether adaptive
fusion retains more detection performance than static fusion."
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from science.fusion.module_h import (
    compute_scatter_matrices,
    optimize_fusion_continuous,
    select_threshold_neyman_pearson,
)
from science.evaluation.module_k import compute_full_metrics


# ============================================================
# M.1 — Degradation Application
# ============================================================

def apply_degradation(
    sensor_scores: Dict[str, np.ndarray],
    sensor_to_degrade: str,
    severity: float,
    degradation_type: str = "noise_injection",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Apply degradation to a specific sensor's scores.

    severity: 0.0 (no degradation) to 1.0 (fully degraded)

    Degradation types:
    - noise_injection: add Gaussian noise proportional to severity
    - signal_suppression: reduce signal, increasing noise floor
    - sensor_removal: replace with random uniform (severity → 1.0)
    """
    rng = np.random.default_rng(seed)
    degraded = {}

    for s, scores in sensor_scores.items():
        if s == sensor_to_degrade:
            if degradation_type == "sensor_removal" and severity >= 0.99:
                # Full removal: replace with uninformative random scores
                degraded[s] = rng.uniform(0.3, 0.7, len(scores))
            elif degradation_type == "noise_injection":
                # Inject noise proportional to severity
                noise_std = severity * np.std(scores)
                noise = rng.normal(0, max(noise_std, 1e-6), len(scores))
                degraded[s] = np.clip(scores + noise, 0.0, 1.0)
            elif degradation_type == "signal_suppression":
                # Compress dynamic range toward 0.5
                center = np.mean(scores)
                degraded[s] = center + (1.0 - severity) * (scores - center)
                degraded[s] = np.clip(degraded[s], 0.0, 1.0)
            else:
                # Default: noise injection
                noise_std = severity * np.std(scores)
                noise = rng.normal(0, max(noise_std, 1e-6), len(scores))
                degraded[s] = np.clip(scores + noise, 0.0, 1.0)
        else:
            degraded[s] = scores.copy()

    return degraded


# ============================================================
# M.1b — Live Contribution Recomputation (Section 3.4)
# ============================================================

def contribution_at_severity(
    degraded_scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    adaptive_weights: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    Recompute contribution metrics at a single degradation severity.

    Returns three independent contribution lenses:
    - weight_contributions: normalized adaptive fusion weights
    - ablation_contributions: ΔAUC when each sensor is removed
    - trust_delta: how much each sensor's weight changed from uniform

    This is the Section 3.4 "closing the loop" addition:
    contribution analysis recomputed live during degradation sweeps.
    """
    from sklearn.metrics import roc_auc_score

    sensor_names = list(degraded_scores.keys())

    # --- Lens 1: Weight magnitudes (trivial but informative) ---
    total_w = sum(adaptive_weights.values())
    weight_contributions = {
        s: adaptive_weights[s] / total_w if total_w > 1e-12 else 1.0 / len(sensor_names)
        for s in sensor_names
    }

    # --- Lens 2: Ablation ΔAUC ---
    # Full fused score
    fused_full = np.zeros_like(labels, dtype=float)
    for s in sensor_names:
        fused_full += adaptive_weights[s] * degraded_scores[s]

    try:
        auc_full = roc_auc_score(labels, fused_full)
    except ValueError:
        auc_full = 0.5

    ablation_contributions = {}
    for removed in sensor_names:
        remaining = [s for s in sensor_names if s != removed]
        w_remaining = {s: adaptive_weights[s] for s in remaining}
        w_sum = sum(w_remaining.values())
        if w_sum > 1e-12:
            w_remaining = {s: v / w_sum for s, v in w_remaining.items()}

        fused_ablated = np.zeros_like(labels, dtype=float)
        for s in remaining:
            fused_ablated += w_remaining[s] * degraded_scores[s]

        try:
            auc_ablated = roc_auc_score(labels, fused_ablated)
        except ValueError:
            auc_ablated = 0.5

        ablation_contributions[removed] = auc_full - auc_ablated

    # Normalize ablation to proportions
    total_abl = sum(abs(v) for v in ablation_contributions.values())
    ablation_normalized = {
        s: abs(v) / total_abl if total_abl > 1e-12 else 1.0 / len(sensor_names)
        for s, v in ablation_contributions.items()
    }

    return {
        "weight_contributions": weight_contributions,
        "ablation_delta_auc": ablation_contributions,
        "ablation_normalized": ablation_normalized,
    }


# ============================================================
# M.2 — Single Degradation Step (with live contribution)
# ============================================================

def degradation_step(
    sensor_scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    static_weights: Dict[str, float],
    sensor_to_degrade: str,
    severity: float,
    degradation_type: str = "noise_injection",
    lam: float = 0.5,
    n_restarts: int = 10,
    seed: int = 42,
    compute_contributions: bool = True,
) -> Dict[str, Any]:
    """
    Run one degradation step:
    1. Apply degradation at given severity
    2. Compute static fusion (frozen pre-degradation weights)
    3. Re-optimize adaptive fusion weights
    4. Compute detection metrics for both
    5. [Section 3.4] Recompute contribution analysis at this severity

    Returns dict with all comparison data including live contributions.
    """
    sensor_names = list(sensor_scores.keys())

    # Apply degradation
    degraded_scores = apply_degradation(
        sensor_scores, sensor_to_degrade, severity, degradation_type, seed
    )

    # --- Static fusion (frozen weights) ---
    fused_static = np.zeros_like(labels, dtype=float)
    for s in sensor_names:
        fused_static += static_weights[s] * degraded_scores[s]

    # --- Adaptive fusion (re-optimize) ---
    scores_matrix = np.column_stack([degraded_scores[s] for s in sensor_names])
    mask_h1 = labels == 1
    mask_h0 = labels == 0

    S_b, S_w = compute_scatter_matrices(scores_matrix[mask_h1], scores_matrix[mask_h0])
    w_adaptive, obj, _, solver = optimize_fusion_continuous(
        S_b, S_w, len(sensor_names), n_restarts, lam, seed
    )

    adaptive_weights_dict = {s: float(w_adaptive[i]) for i, s in enumerate(sensor_names)}

    fused_adaptive = np.zeros_like(labels, dtype=float)
    for i, s in enumerate(sensor_names):
        fused_adaptive += w_adaptive[i] * degraded_scores[s]

    # --- Metrics ---
    threshold_static, dr_static, far_static = select_threshold_neyman_pearson(
        fused_static, labels, 0.01
    )
    threshold_adaptive, dr_adaptive, far_adaptive = select_threshold_neyman_pearson(
        fused_adaptive, labels, 0.01
    )

    from sklearn.metrics import roc_auc_score
    try:
        auc_static = roc_auc_score(labels, fused_static)
    except ValueError:
        auc_static = 0.5
    try:
        auc_adaptive = roc_auc_score(labels, fused_adaptive)
    except ValueError:
        auc_adaptive = 0.5

    result = {
        "severity": severity,
        "static_detection_rate": dr_static,
        "adaptive_detection_rate": dr_adaptive,
        "static_auc": auc_static,
        "adaptive_auc": auc_adaptive,
        "adaptive_weights": adaptive_weights_dict,
        "static_weights": static_weights,
        "fisher_objective": obj,
    }

    # --- Section 3.4: Live contribution recomputation ---
    if compute_contributions:
        contributions = contribution_at_severity(
            degraded_scores, labels, adaptive_weights_dict
        )
        result["contributions"] = contributions

    return result


# ============================================================
# M.3 — Full Degradation Sweep (with trust migration)
# ============================================================

def run_degradation_sweep(
    progress_callback: Callable,
    sensor_scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    static_weights: Dict[str, float],
    sensor_to_degrade: str = "radar",
    degradation_type: str = "noise_injection",
    n_steps: int = 5,
    severity_min: float = 0.0,
    severity_max: float = 1.0,
    lam: float = 0.5,
    n_restarts: int = 10,
    seed: int = 42,
    compute_contributions: bool = True,
) -> Dict[str, Any]:
    """
    Full Module M pipeline: sweep severity from min to max,
    compare static vs adaptive fusion at each step.

    When compute_contributions=True (default), also recomputes
    contribution analysis at every severity step for the
    Section 3.4 trust migration visualization.
    """
    sensor_names = list(sensor_scores.keys())
    severities = np.linspace(severity_min, severity_max, n_steps)

    all_static_dr = []
    all_adaptive_dr = []
    all_static_auc = []
    all_adaptive_auc = []
    all_weights = {s: [] for s in sensor_names}
    all_fisher_objectives = []

    # Section 3.4: contribution tracking across severity
    all_weight_contributions = {s: [] for s in sensor_names}
    all_ablation_contributions = {s: [] for s in sensor_names}

    for step, sev in enumerate(severities):
        progress_callback(
            stage="Degradation sweep",
            progress=step / n_steps,
            message=f"Step {step+1}/{n_steps}, severity={sev:.2f}",
        )

        result = degradation_step(
            sensor_scores, labels, static_weights,
            sensor_to_degrade, sev, degradation_type,
            lam, n_restarts, seed + step,
            compute_contributions=compute_contributions,
        )

        all_static_dr.append(result["static_detection_rate"])
        all_adaptive_dr.append(result["adaptive_detection_rate"])
        all_static_auc.append(result["static_auc"])
        all_adaptive_auc.append(result["adaptive_auc"])
        all_fisher_objectives.append(result.get("fisher_objective", 0.0))

        for s in sensor_names:
            all_weights[s].append(result["adaptive_weights"][s])

        # Collect contribution vectors (Section 3.4)
        if compute_contributions and "contributions" in result:
            contrib = result["contributions"]
            for s in sensor_names:
                all_weight_contributions[s].append(
                    contrib["weight_contributions"].get(s, 0.0)
                )
                all_ablation_contributions[s].append(
                    contrib["ablation_normalized"].get(s, 0.0)
                )

    progress_callback(
        stage="Degradation sweep complete",
        progress=1.0,
        message=f"{n_steps} severity levels evaluated"
                + (" with trust migration" if compute_contributions else ""),
    )

    sweep_result = {
        "sensor": sensor_to_degrade,
        "degradation_type": degradation_type,
        "severity_values": severities,
        "static_detection_retention": np.array(all_static_dr),
        "adaptive_detection_retention": np.array(all_adaptive_dr),
        "static_auc": np.array(all_static_auc),
        "adaptive_auc": np.array(all_adaptive_auc),
        "weights_by_severity": {s: np.array(v) for s, v in all_weights.items()},
        "fisher_objectives": np.array(all_fisher_objectives),
    }

    # Section 3.4: trust migration data
    if compute_contributions:
        sweep_result["trust_migration"] = {
            "weight_contributions": {
                s: np.array(v) for s, v in all_weight_contributions.items()
            },
            "ablation_contributions": {
                s: np.array(v) for s, v in all_ablation_contributions.items()
            },
        }

    return sweep_result
