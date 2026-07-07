"""
QT-2.23 — Module H: Fusion Weight Optimization (Rayleigh-LDA)

Optimizes fusion weights using SLSQP and selects Neyman-Pearson threshold.
See `docs/SCIENCE_MODULES.md` for mathematical theory.
"""

from __future__ import annotations
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import roc_curve


# ============================================================
# H.1 — Scatter Matrices (standard LDA objects)
# ============================================================

def compute_scatter_matrices(
    sensor_scores_class1: np.ndarray,
    sensor_scores_class0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute between-class (S_b) and within-class (S_w) scatter matrices.

    S_b = (μ₁ - μ₀)(μ₁ - μ₀)ᵀ  (between-class)
    S_w = Σ₁ + Σ₀               (within-class)

    Parameters
    ----------
    sensor_scores_class1 : (N1, n_sensors) scores for target-present class
    sensor_scores_class0 : (N0, n_sensors) scores for target-absent class
    """
    mu1 = sensor_scores_class1.mean(axis=0)
    mu0 = sensor_scores_class0.mean(axis=0)
    diff = (mu1 - mu0).reshape(-1, 1)
    S_b = diff @ diff.T

    # Within-class scatter (sum of class covariances)
    if sensor_scores_class1.shape[0] > 1:
        S_w1 = np.cov(sensor_scores_class1.T)
    else:
        S_w1 = np.zeros((len(mu1), len(mu1)))

    if sensor_scores_class0.shape[0] > 1:
        S_w0 = np.cov(sensor_scores_class0.T)
    else:
        S_w0 = np.zeros((len(mu0), len(mu0)))

    # Handle 1D case (single sensor)
    if S_w1.ndim == 0:
        S_w1 = np.array([[S_w1]])
    if S_w0.ndim == 0:
        S_w0 = np.array([[S_w0]])

    S_w = S_w1 + S_w0

    return S_b, S_w


# ============================================================
# H.2 — Continuous Optimization (Primary Path)
# ============================================================

def fisher_objective(w: np.ndarray, S_b: np.ndarray, S_w: np.ndarray, lam: float = 0.5) -> float:
    """J'(w) = w^T(S_b - λ·S_w)w. Negate for minimization."""
    w = np.asarray(w, dtype=float)
    return -(w @ S_b @ w - lam * (w @ S_w @ w))


def optimize_fusion_continuous(
    S_b: np.ndarray,
    S_w: np.ndarray,
    n_sensors: int = 3,
    n_restarts: int = 10,
    lam: float = 0.5,
    seed: int = 42,
    progress_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, float, float, str]:
    """
    Continuous optimization of Rayleigh-LDA objective via SLSQP.

    Returns (weights, objective, solve_time_s, solver_name).
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
                message=f"Restart {restart+1}/{n_restarts}, J'={-best_val:.6f}",
            )

    solve_time = time.time() - start_time

    w = best_result.x
    w = np.clip(w, 0, None)
    w_sum = np.sum(w)
    if w_sum > 1e-12:
        w = w / w_sum

    return w, -best_val, solve_time, "Backup Solver · SLSQP (10 restarts)"


# ============================================================
# H.3 — Binary QUBO Fallback (Fixed-Point Encoding)
# ============================================================

def weights_from_bits(bits: np.ndarray, n_sensors: int, bits_per_weight: int) -> np.ndarray:
    """
    Decode binary vector to continuous weights.
    wᵢ = Σ_{p=1}^{b} 2^{-(p+1)} · x_{i,p}
    """
    w = np.zeros(n_sensors)
    for i in range(n_sensors):
        for p in range(bits_per_weight):
            w[i] += bits[i * bits_per_weight + p] * (2.0 ** -(p + 1))
    return w


def build_weight_qubo(
    S_b: np.ndarray,
    S_w: np.ndarray,
    lam: float,
    n_sensors: int = 3,
    bits_per_weight: int = 4,
    simplex_penalty: float = 5.0,
) -> np.ndarray:
    """
    Build QUBO matrix for binary-encoded fusion weights.

    Encodes wᵢ = Σ 2^{-p} x_{i,p} and converts the Rayleigh-LDA
    quadratic form into a binary optimization problem.

    Returns Q matrix of shape (n_sensors * bits_per_weight, n_sensors * bits_per_weight).
    """
    n_vars = n_sensors * bits_per_weight
    Q = np.zeros((n_vars, n_vars))

    bit_value = lambda p: 2.0 ** -(p + 1)
    M = S_b - lam * S_w

    # Objective terms
    for i in range(n_sensors):
        for p in range(bits_per_weight):
            idx_ip = i * bits_per_weight + p
            for j in range(n_sensors):
                for q in range(bits_per_weight):
                    idx_jq = j * bits_per_weight + q
                    Q[idx_ip, idx_jq] += M[i, j] * bit_value(p) * bit_value(q)

    # Negate: maximize J' becomes minimize -J' for the solver
    Q = -Q

    # Add simplex constraint penalty: penalty * (Σwᵢ - 1)²
    Qc = Q.copy()
    for i in range(n_sensors):
        for p in range(bits_per_weight):
            idx_ip = i * bits_per_weight + p
            # Diagonal: penalty * (bv² - 2*bv) using x²=x
            Qc[idx_ip, idx_ip] += simplex_penalty * (bit_value(p)**2 - 2.0 * bit_value(p))
            for j in range(n_sensors):
                for q in range(bits_per_weight):
                    if (i, p) != (j, q):
                        idx_jq = j * bits_per_weight + q
                        Qc[idx_ip, idx_jq] += simplex_penalty * bit_value(p) * bit_value(q)

    return Qc


def solve_weight_qubo(
    Q: np.ndarray,
    n_sensors: int = 3,
    bits_per_weight: int = 4,
    num_reads: int = 1000,
    progress_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, float, float, str, Dict[str, Any]]:
    """
    Solve the binary-encoded weight QUBO via neal SA.

    Returns (weights, objective, solve_time_s, solver_name, metadata).
    """
    import dimod
    import neal

    n_vars = Q.shape[0]

    if progress_callback:
        progress_callback(
            stage="Building weight BQM",
            progress=0.1,
            message=f"{n_vars} binary variables ({n_sensors}×{bits_per_weight} bits)",
        )

    linear = {i: float(Q[i, i]) for i in range(n_vars)}
    quadratic = {}
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            val = float(Q[i, j] + Q[j, i])
            if abs(val) > 1e-15:
                quadratic[(i, j)] = val

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

    if progress_callback:
        progress_callback(stage="Sampling", progress=0.4, message=f"{num_reads} reads")

    start_time = time.time()
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads, seed=42)
    solve_time = time.time() - start_time

    best = response.first
    bits = np.array([best.sample.get(i, 0) for i in range(n_vars)])
    w = weights_from_bits(bits, n_sensors, bits_per_weight)

    # Normalize
    w_sum = np.sum(w)
    if w_sum > 1e-12:
        w = w / w_sum

    metadata = {
        "solver": "neal.SimulatedAnnealingSampler",
        "num_reads": num_reads,
        "solve_time_s": solve_time,
        "n_vars": n_vars,
        "energy": float(best.energy),
        "bits": bits.tolist(),
    }

    return w, float(best.energy), solve_time, "Backup Solver · Simulated Annealing (4-bit QUBO)", metadata


# ============================================================
# H.4 — Neyman-Pearson Threshold Selection (stays classical)
# ============================================================

def select_threshold_neyman_pearson(
    fused_scores: np.ndarray,
    labels: np.ndarray,
    target_far: float = 0.01,
) -> Tuple[float, float, float]:
    """
    Classical 1-D threshold sweep to select operating point.

    Not pushed into QUBO because it's a single continuous scalar —
    not combinatorial. Knowing where quantum methods don't apply
    is more impressive than applying them everywhere.

    Returns (threshold, detection_rate, false_alarm_rate).
    """
    fpr, tpr, thresholds = roc_curve(labels, fused_scores)
    
    valid_indices = np.where(fpr <= target_far)[0]
    if len(valid_indices) == 0:
        # Fallback if no point satisfies the FAR constraint
        idx = np.argmin(fpr)
    else:
        # Maximize Pd among valid operating points
        best_tpr = np.max(tpr[valid_indices])
        # Find indices that achieve best TPR within valid FPR bounds
        best_indices = valid_indices[tpr[valid_indices] == best_tpr]
        # Tie-break: select highest FPR (lowest threshold)
        idx = best_indices[-1]
        
    return float(thresholds[idx]), float(tpr[idx]), float(fpr[idx])


# ============================================================
# H.5 — Full Fusion Optimization Pipeline
# ============================================================

def run_fusion_optimization(
    progress_callback: Callable,
    sensor_scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    lam: float = 0.5,
    n_restarts: int = 10,
    seed: int = 42,
    bits_per_weight: int = 4,
    simplex_penalty: float = 5.0,
    target_far: float = 0.01,
    run_binary: bool = True,
) -> Dict[str, Any]:
    """
    Full Module H pipeline:
    1. Compute scatter matrices
    2. Run continuous SLSQP optimization (primary)
    3. Run binary QUBO fallback (parallel)
    4. Select detection threshold (Neyman-Pearson)
    """
    sensor_names = list(sensor_scores.keys())
    n_sensors = len(sensor_names)

    # Build score arrays
    scores_matrix = np.column_stack([sensor_scores[s] for s in sensor_names])
    mask_h1 = labels == 1
    mask_h0 = labels == 0

    # --- Scatter matrices ---
    progress_callback(
        stage="Computing scatter matrices",
        progress=0.05,
        message="Between-class and within-class scatter",
    )
    S_b, S_w = compute_scatter_matrices(
        scores_matrix[mask_h1], scores_matrix[mask_h0]
    )

    # --- Continuous optimization (primary path) ---
    progress_callback(
        stage="Continuous optimization",
        progress=0.10,
        message=f"SLSQP with {n_restarts} restarts",
    )
    w_cont, obj_cont, t_cont, solver_cont = optimize_fusion_continuous(
        S_b, S_w, n_sensors, n_restarts, lam, seed, progress_callback
    )

    # --- Binary QUBO fallback (parallel path) ---
    w_binary, obj_binary, t_binary, solver_binary, binary_meta = None, None, None, None, None
    if run_binary:
        progress_callback(
            stage="Binary QUBO fallback",
            progress=0.60,
            message=f"{n_sensors}×{bits_per_weight} = {n_sensors*bits_per_weight} binary vars",
        )
        Q_weight = build_weight_qubo(S_b, S_w, lam, n_sensors, bits_per_weight, simplex_penalty)
        w_binary, obj_binary, t_binary, solver_binary, binary_meta = solve_weight_qubo(
            Q_weight, n_sensors, bits_per_weight, progress_callback=progress_callback
        )

    # Use the continuous result as primary (better precision than 4-bit binary)
    w_final = w_cont
    solver_used = solver_cont
    obj_final = obj_cont
    solve_time = t_cont

    # --- Compute fused scores ---
    progress_callback(
        stage="Computing fused scores",
        progress=0.85,
        message=f"w = [{', '.join(f'{v:.3f}' for v in w_final)}]",
    )
    fused_scores = np.zeros_like(labels, dtype=float)
    for i, s in enumerate(sensor_names):
        fused_scores += w_final[i] * sensor_scores[s]

    # Score distributions for visualization
    sensor_scores_h0 = {s: sensor_scores[s][mask_h0] for s in sensor_names}
    sensor_scores_h1 = {s: sensor_scores[s][mask_h1] for s in sensor_names}
    fused_h0 = fused_scores[mask_h0]
    fused_h1 = fused_scores[mask_h1]

    # --- Neyman-Pearson threshold ---
    progress_callback(
        stage="Threshold selection",
        progress=0.92,
        message=f"Target FAR = {target_far}",
    )
    threshold, det_rate, far = select_threshold_neyman_pearson(
        fused_scores, labels, target_far
    )

    progress_callback(
        stage="Fusion optimization complete",
        progress=1.0,
        message=f"J'={obj_final:.4f}, τ={threshold:.4f}, Pd={det_rate:.3f}",
    )

    return {
        "weights": {s: float(w_final[i]) for i, s in enumerate(sensor_names)},
        "weights_array": w_final,
        "fused_scores": fused_scores,
        "threshold": threshold,
        "fisher_objective": obj_final,
        "S_b": S_b,
        "S_w": S_w,
        "solver": solver_used,
        "solve_time_s": solve_time,
        "optimization_mode": "Continuous · SLSQP",
        "sensor_scores_h0": sensor_scores_h0,
        "sensor_scores_h1": sensor_scores_h1,
        "fused_scores_h0": fused_h0,
        "fused_scores_h1": fused_h1,
        # Binary fallback results (kept in parallel)
        "binary_weights": {s: float(w_binary[i]) for i, s in enumerate(sensor_names)} if w_binary is not None else None,
        "binary_objective": obj_binary,
        "binary_solve_time": t_binary,
        "binary_solver": solver_binary,
        "binary_metadata": binary_meta,
    }
