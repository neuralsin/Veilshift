"""
QT-2.23 — Module F: Classical Baseline Detectors

Four genuine classical fusion baselines + Differential Evolution:
1. CRLB-weighted average (closed form, no optimizer)
2. Kalman filter fusion
3. Bayesian sequential fusion
4. SLSQP-optimized weights (same objective as Module H — the critical baseline)
5. Differential Evolution (population-based global search)

Baseline 4 is the most important: it solves the EXACT SAME Rayleigh-LDA
objective that the quantum solver solves, making the comparison scientifically valid.
Run with ≥10 random restarts per Appendix B.
"""

from __future__ import annotations
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ============================================================
# F.1 — Per-sensor detector (shared by all baselines)
# ============================================================

def train_sensor_detector(X_features: np.ndarray, y_labels: np.ndarray) -> LogisticRegression:
    """
    Train a simple logistic regression per sensor.
    Kept deliberately simple — innovation lives in the fusion layer.
    """
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_features, y_labels)
    return clf


def sensor_score(clf: LogisticRegression, X_features: np.ndarray) -> np.ndarray:
    """Get probability scores from a trained sensor detector."""
    return clf.predict_proba(X_features)[:, 1]


# ============================================================
# F.2 — Baseline 1: CRLB-Weighted Fusion
# ============================================================

def fuse_crlb_weighted(
    sensor_scores: Dict[str, np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """
    Simple weighted average fusion using CRLB-derived weights.
    No optimization — pure closed-form from Fisher Information.
    """
    sensors = list(sensor_scores.keys())
    fused = np.zeros_like(sensor_scores[sensors[0]])
    for i, s in enumerate(sensors):
        fused += weights[i] * sensor_scores[s]
    return fused


# ============================================================
# F.3 — Baseline 2: Kalman Filter Fusion
# ============================================================

def kalman_fuse(
    observations: List[np.ndarray],
    observation_noise_vars: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kalman fusion: inverse-variance weighted combination.
    Returns (fused_scores, fused_variance).
    """
    weights = np.array([1.0 / max(v, 1e-12) for v in observation_noise_vars])
    total = np.sum(weights)
    norm_weights = weights / total

    fused = np.zeros_like(observations[0])
    for i, obs in enumerate(observations):
        fused += norm_weights[i] * obs

    fused_var = 1.0 / total
    return fused, np.full_like(fused, fused_var)


# ============================================================
# F.4 — Baseline 3: Bayesian Sequential Fusion
# ============================================================

def bayesian_fuse(
    prior_log_odds: float,
    sensor_scores: List[np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Bayesian sequential fusion using log-odds update.

    Converts each sensor's probability score to a likelihood ratio
    and accumulates evidence sequentially.
    """
    n = len(sensor_scores[0])
    log_odds = np.full(n, prior_log_odds)

    for scores in sensor_scores:
        # Clip to avoid log(0)
        s = np.clip(scores, eps, 1.0 - eps)
        lr = s / (1.0 - s)
        log_odds += np.log(lr)

    return 1.0 / (1.0 + np.exp(-log_odds))


# ============================================================
# F.5 — Baseline 4: Classically-Optimized Weights (SLSQP)
# ============================================================

def fisher_objective(w: np.ndarray, S_b: np.ndarray, S_w: np.ndarray, lam: float = 0.5) -> float:
    """
    Fisher / Rayleigh-LDA objective: J'(w) = w^T(S_b - λ·S_w)w.
    Negate because scipy minimizes.
    """
    w = np.asarray(w, dtype=float)
    return -(w @ S_b @ w - lam * (w @ S_w @ w))


def optimize_weights_slsqp(
    S_b: np.ndarray,
    S_w: np.ndarray,
    n_sensors: int = 3,
    n_restarts: int = 10,
    lam: float = 0.5,
    seed: int = 42,
    progress_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Multi-restart SLSQP optimization of the Rayleigh-LDA objective.

    This is the CRITICAL baseline — same objective as Module H.
    ≥10 restarts per Appendix B to ensure fair comparison.

    Returns (best_weights, best_objective, solve_time_s).
    """
    rng = np.random.default_rng(seed)
    best_result, best_val = None, np.inf
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_sensors

    start_time = time.time()

    for restart in range(n_restarts):
        w0 = rng.dirichlet(np.ones(n_sensors))
        result = minimize(
            fisher_objective, w0,
            args=(S_b, S_w, lam),
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={'maxiter': 200},
        )
        if result.fun < best_val:
            best_val, best_result = result.fun, result

        if progress_callback:
            progress_callback(
                stage="SLSQP optimization",
                progress=(restart + 1) / n_restarts,
                message=f"Restart {restart+1}/{n_restarts}, objective={-best_val:.4f}",
            )

    solve_time = time.time() - start_time

    w = best_result.x
    w = np.clip(w, 0, None)
    w = w / (np.sum(w) + 1e-12)

    return w, -best_val, solve_time


# ============================================================
# F.6 — Baseline 5: Differential Evolution
# ============================================================

def optimize_weights_de(
    S_b: np.ndarray,
    S_w: np.ndarray,
    n_sensors: int = 3,
    lam: float = 0.5,
    seed: int = 42,
    progress_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Differential Evolution: population-based global classical search.
    Closest classical analog to what the quantum solver does.

    Returns (best_weights, best_objective, solve_time_s).
    """
    bounds = [(0.0, 1.0)] * n_sensors

    def objective(w):
        w = np.asarray(w)
        w = w / (np.sum(w) + 1e-12)  # Project onto simplex
        return fisher_objective(w, S_b, S_w, lam)

    start_time = time.time()

    result = differential_evolution(
        objective, bounds,
        seed=seed,
        maxiter=500,
        tol=1e-8,
        callback=lambda xk, convergence: (
            progress_callback(
                stage="Differential Evolution",
                progress=min(convergence, 0.99),
                message=f"Convergence={convergence:.4f}",
            ) if progress_callback else None
        ),
    )

    solve_time = time.time() - start_time

    w = result.x / (np.sum(result.x) + 1e-12)

    return w, -result.fun, solve_time


# ============================================================
# F.7 — Run All Baselines
# ============================================================

def run_all_baselines(
    progress_callback: Callable,
    sensor_scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    S_b: np.ndarray,
    S_w: np.ndarray,
    crlb_weights: np.ndarray,
    lam: float = 0.5,
    n_restarts: int = 10,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Run all 5 classical baselines and return results for comparison.
    """
    sensor_names = list(sensor_scores.keys())
    score_arrays = [sensor_scores[s] for s in sensor_names]
    results = []

    # --- Baseline 1: CRLB-Weighted ---
    progress_callback(
        stage="Baseline 1: CRLB-Weighted",
        progress=0.0,
        message="Computing...",
    )
    t0 = time.time()
    fused_crlb = fuse_crlb_weighted(sensor_scores, crlb_weights)
    t_crlb = time.time() - t0
    auc_crlb = roc_auc_score(labels, fused_crlb)
    results.append({
        "method_name": "CRLB-Weighted",
        "weights": {s: float(crlb_weights[i]) for i, s in enumerate(sensor_names)},
        "objective_value": None,
        "solve_time_s": t_crlb,
        "auc": auc_crlb,
        "fused_scores": fused_crlb,
    })

    # --- Baseline 2: Kalman Fusion ---
    progress_callback(
        stage="Baseline 2: Kalman",
        progress=0.15,
        message="Inverse-variance weighting",
    )
    t0 = time.time()
    noise_vars = [float(np.var(s)) for s in score_arrays]
    fused_kalman, _ = kalman_fuse(score_arrays, noise_vars)
    t_kalman = time.time() - t0
    auc_kalman = roc_auc_score(labels, fused_kalman)
    kalman_weights_raw = np.array([1.0 / max(v, 1e-12) for v in noise_vars])
    kalman_weights = kalman_weights_raw / kalman_weights_raw.sum()
    results.append({
        "method_name": "Kalman Fusion",
        "weights": {s: float(kalman_weights[i]) for i, s in enumerate(sensor_names)},
        "objective_value": None,
        "solve_time_s": t_kalman,
        "auc": auc_kalman,
        "fused_scores": fused_kalman,
    })

    # --- Baseline 3: Bayesian Sequential ---
    progress_callback(
        stage="Baseline 3: Bayesian",
        progress=0.30,
        message="Log-odds sequential update",
    )
    t0 = time.time()
    fused_bayes = bayesian_fuse(0.0, score_arrays)
    t_bayes = time.time() - t0
    auc_bayes = roc_auc_score(labels, fused_bayes)
    results.append({
        "method_name": "Bayesian Sequential",
        "weights": {s: 1.0 / len(sensor_names) for s in sensor_names},  # Implicit equal weighting
        "objective_value": None,
        "solve_time_s": t_bayes,
        "auc": auc_bayes,
        "fused_scores": fused_bayes,
    })

    # --- Baseline 4: SLSQP (CRITICAL apples-to-apples) ---
    progress_callback(
        stage="Baseline 4: SLSQP",
        progress=0.40,
        message=f"{n_restarts} restarts, same objective as solver",
    )
    w_slsqp, obj_slsqp, t_slsqp = optimize_weights_slsqp(
        S_b, S_w, len(sensor_names), n_restarts, lam, seed, progress_callback
    )
    fused_slsqp = np.zeros_like(score_arrays[0])
    for i, s in enumerate(sensor_names):
        fused_slsqp += w_slsqp[i] * sensor_scores[s]
    auc_slsqp = roc_auc_score(labels, fused_slsqp)
    results.append({
        "method_name": "SLSQP Optimized",
        "weights": {s: float(w_slsqp[i]) for i, s in enumerate(sensor_names)},
        "objective_value": obj_slsqp,
        "solve_time_s": t_slsqp,
        "auc": auc_slsqp,
        "fused_scores": fused_slsqp,
    })

    # --- Baseline 5: Differential Evolution ---
    progress_callback(
        stage="Baseline 5: Differential Evolution",
        progress=0.70,
        message="Population-based global search",
    )
    w_de, obj_de, t_de = optimize_weights_de(
        S_b, S_w, len(sensor_names), lam, seed, progress_callback
    )
    fused_de = np.zeros_like(score_arrays[0])
    for i, s in enumerate(sensor_names):
        fused_de += w_de[i] * sensor_scores[s]
    auc_de = roc_auc_score(labels, fused_de)
    results.append({
        "method_name": "Differential Evolution",
        "weights": {s: float(w_de[i]) for i, s in enumerate(sensor_names)},
        "objective_value": obj_de,
        "solve_time_s": t_de,
        "auc": auc_de,
        "fused_scores": fused_de,
    })

    progress_callback(
        stage="All baselines complete",
        progress=1.0,
        message=f"5 methods evaluated",
    )

    return results
