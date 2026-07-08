"""
QT-2.23 — Module G: QUBO Feature Selection (MID-mRMR)

Constructs the QUBO matrix for feature selection.
See `docs/SCIENCE_MODULES.md` for the full mathematical formulation.
"""

from __future__ import annotations
import time
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# G.1 — MID-QUBO Construction
# ============================================================

def build_feature_selection_qubo(
    features_matrix: np.ndarray,
    labels: np.ndarray,
    k_target: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the QUBO matrix for MID-mRMR feature selection.

    Q_ii = -α·Relᵢ + γ·(1 - 2k)
    Q_ij = β·Redᵢⱼ + 2γ      (i ≠ j)

    Uses xᵢ² = xᵢ for binary variables when expanding the cardinality penalty.

    Parameters
    ----------
    features_matrix : (N, n_features) feature matrix
    labels : (N,) binary labels
    k_target : int - desired number of selected features
    alpha, beta, gamma : float - QUBO coefficients (Appendix B defaults)

    Returns
    -------
    (Q, relevance, redundancy) - Q matrix, relevance scores, redundancy matrix
    """
    n = features_matrix.shape[1]

    # Compute relevance: |corr(featureᵢ, label)|
    relevance = np.zeros(n)
    for i in range(n):
        feat_std = np.std(features_matrix[:, i])
        if feat_std < 1e-12:
            relevance[i] = 0.0
        else:
            corr = np.corrcoef(features_matrix[:, i], labels)[0, 1]
            relevance[i] = abs(corr) if not np.isnan(corr) else 0.0

    # Compute redundancy: |corr(featureᵢ, featureⱼ)|
    # Handle constant features gracefully
    redundancy = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            si = np.std(features_matrix[:, i])
            sj = np.std(features_matrix[:, j])
            if si < 1e-12 or sj < 1e-12:
                redundancy[i, j] = redundancy[j, i] = 0.0
            else:
                corr = np.corrcoef(features_matrix[:, i], features_matrix[:, j])[0, 1]
                val = abs(corr) if not np.isnan(corr) else 0.0
                redundancy[i, j] = redundancy[j, i] = val

    # Build Q matrix
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -alpha * relevance[i] + gamma * (1 - 2 * k_target)
        for j in range(i + 1, n):
            Q[i, j] = beta * redundancy[i, j] + 2 * gamma
            Q[j, i] = Q[i, j]

    return Q, relevance, redundancy


# ============================================================
# G.2 — Brute-Force Validator
# ============================================================

def brute_force_feature_selection(
    Q: np.ndarray,
    n_features: int,
    k_target: int,
    progress_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, float, List[int]]:
    """
    Exhaustive brute-force search over all C(n, k) subsets.

    Cheap at n ≤ 16 features. MUST run before trusting any solver
    on the same problem — a mismatch means the QUBO formulation
    or solver call has a bug.

    Returns (best_x, best_H, best_indices).
    """
    best_H = np.inf
    best_x = None
    best_indices = []

    total = 0
    for _ in combinations(range(n_features), k_target):
        total += 1

    count = 0
    for combo in combinations(range(n_features), k_target):
        x = np.zeros(n_features)
        x[list(combo)] = 1
        H = float(x @ Q @ x)
        if H < best_H:
            best_H = H
            best_x = x.copy()
            best_indices = list(combo)

        count += 1
        if progress_callback and count % max(1, total // 20) == 0:
            progress_callback(
                stage="Brute force search",
                progress=count / total,
                message=f"Subset {count}/{total}, best H={best_H:.4f}",
            )

    return best_x, best_H, best_indices


# ============================================================
# G.3 — Solver Interface (uses Module I)
# ============================================================

def solve_feature_qubo(
    Q: np.ndarray,
    num_reads: int = 1000,
    progress_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Solve the feature selection QUBO using neal (Simulated Annealing).

    Returns (best_x, best_energy, solver_metadata).
    """
    import dimod
    import neal

    if progress_callback:
        progress_callback(
            stage="Building BQM",
            progress=0.1,
            message=f"Q matrix {Q.shape[0]}×{Q.shape[1]}",
        )

    # Build BQM from Q matrix
    n = Q.shape[0]
    linear = {i: float(Q[i, i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Q[i, j]) > 1e-15:
                quadratic[(i, j)] = float(Q[i, j] + Q[j, i])  # Symmetric

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

    if progress_callback:
        progress_callback(
            stage="Sampling",
            progress=0.3,
            message=f"{num_reads} reads",
        )

    start_time = time.time()
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads, seed=42)
    solve_time = time.time() - start_time

    if progress_callback:
        progress_callback(
            stage="Processing results",
            progress=0.9,
            message=f"Solve time: {solve_time:.3f}s",
        )

    best = response.first
    best_x = np.array([best.sample.get(i, 0) for i in range(n)])
    best_energy = float(best.energy)

    metadata = {
        "solver": "neal.SimulatedAnnealingSampler",
        "num_reads": num_reads,
        "solve_time_s": solve_time,
        "num_variables": n,
        "energy": best_energy,
    }

    return best_x, best_energy, metadata


# ============================================================
# G.4 — Full Feature Selection Pipeline
# ============================================================

def run_feature_selection(
    progress_callback: Callable,
    features_matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    k_target: int = 6,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 2.0,
    num_reads: int = 1000,
    run_brute_force: bool = True,
) -> Dict[str, Any]:
    """
    Full Module G pipeline: build QUBO, solve, validate against brute force.
    """
    n_features = features_matrix.shape[1]

    # --- Build QUBO ---
    progress_callback(
        stage="Building QUBO",
        progress=0.05,
        message=f"MID-mRMR, α={alpha}, β={beta}, γ={gamma}, k={k_target}",
    )

    Q, relevance, redundancy = build_feature_selection_qubo(
        features_matrix, labels, k_target, alpha, beta, gamma
    )

    # --- Brute-force validation (≤16 features) ---
    bf_x, bf_energy, bf_indices = None, None, None
    if run_brute_force and n_features <= 20:
        progress_callback(
            stage="Brute force validation",
            progress=0.15,
            message=f"C({n_features},{k_target}) subsets",
        )
        bf_x, bf_energy, bf_indices = brute_force_feature_selection(
            Q, n_features, k_target, progress_callback
        )

    # --- Solve with neal SA ---
    progress_callback(
        stage="Solving QUBO",
        progress=0.50,
        message="Simulated Annealing",
    )
    solver_x, solver_energy, solver_metadata = solve_feature_qubo(
        Q, num_reads, progress_callback
    )

    # --- Compare results ---
    selected_indices = [i for i in range(n_features) if solver_x[i] > 0.5]
    selected_features = [feature_names[i] for i in selected_indices] if feature_names else []

    bf_selected_names = [feature_names[i] for i in bf_indices] if bf_indices and feature_names else []

    subset_match = None
    if bf_indices is not None:
        subset_match = set(selected_indices) == set(bf_indices)

    progress_callback(
        stage="Feature selection complete",
        progress=1.0,
        message=f"Selected {len(selected_indices)}/{n_features}, match={subset_match}",
    )

    return {
        "feature_names": feature_names,
        "relevance": relevance,
        "redundancy_matrix": redundancy,
        "Q_matrix": Q,
        "selected_indices": selected_indices,
        "selected_features": selected_features,
        "objective_value": solver_energy,
        "brute_force_objective": bf_energy,
        "brute_force_selected": bf_indices,
        "brute_force_selected_features": bf_selected_names,
        "subset_match": subset_match,
        "solver": solver_metadata.get("solver", "neal"),
        "solver_metadata": solver_metadata,
    }


# ============================================================
# G.5 — [STRETCH] MIQ via Accelerated Dinkelbach (Section G.2)
# ============================================================

def solve_mrmr_dinkelbach(
    relevance: np.ndarray,
    redundancy_matrix: np.ndarray,
    n_features: int,
    k_target: int,
    max_iter: int = 15,
    tol: float = 1e-4,
    damping: float = 0.5,
    num_reads: int = 1000,
    gamma: float = 2.0,
    progress_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
    """
    Dinkelbach iteration for MIQ (Mutual Information Quotient) mRMR.

    Optimizes the ratio Relevance / Redundancy via iterative
    parametric QUBO sub-problems:

        Q_k = diag(relevance) - λ_k * redundancy_matrix

    with damped λ updates to mitigate non-monotonic convergence
    under heuristic (non-exact) QUBO solvers.

    Per the build manual (Section 2.2 / G.2):
    - Dinkelbach converges superlinearly ONLY with exact sub-problem solving
    - With heuristic solvers (SA), damping is mandatory
    - λ convergence MUST be empirically verified, not assumed from theory

    Parameters
    ----------
    relevance : (n_features,) array of feature relevance scores
    redundancy_matrix : (n_features, n_features) pairwise redundancy
    n_features : number of features
    k_target : desired number of selected features
    max_iter : maximum Dinkelbach iterations
    tol : convergence tolerance for |λ_{k+1} - λ_k|
    damping : blend factor for λ update (0 = no update, 1 = full jump)
    num_reads : SA solver reads per iteration
    gamma : cardinality penalty strength
    progress_callback : optional progress reporter

    Returns
    -------
    (best_x, final_lambda, convergence_history)
    convergence_history is a list of dicts tracking each iteration
    """
    import dimod
    import neal

    lam = 0.0
    x_opt = None
    convergence_history = []

    for k in range(max_iter):
        if progress_callback:
            progress_callback(
                stage="Dinkelbach MIQ",
                progress=k / max_iter,
                message=f"Iteration {k+1}/{max_iter}, λ={lam:.6f}",
            )

        # Build parametric QUBO: maximize (relevance - λ * redundancy)
        # subject to cardinality constraint
        Q = np.zeros((n_features, n_features))
        for i in range(n_features):
            Q[i, i] = -relevance[i] + gamma * (1 - 2 * k_target)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                Q[i, j] = lam * redundancy_matrix[i, j] + 2 * gamma
                Q[j, i] = Q[i, j]

        # Solve sub-problem with SA
        linear = {i: float(Q[i, i]) for i in range(n_features)}
        quadratic = {}
        for i in range(n_features):
            for j in range(i + 1, n_features):
                val = float(Q[i, j] + Q[j, i])
                if abs(val) > 1e-15:
                    quadratic[(i, j)] = val

        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample(bqm, num_reads=num_reads, seed=42 + k)

        best = response.first
        x_opt = np.array([best.sample.get(i, 0) for i in range(n_features)])

        # Compute numerator and denominator of the MIQ ratio
        numerator = float(x_opt @ relevance)
        denominator = float(x_opt @ redundancy_matrix @ x_opt)

        iteration_info = {
            "iteration": k,
            "lambda": lam,
            "numerator": numerator,
            "denominator": denominator,
            "energy": float(best.energy),
            "n_selected": int(np.sum(x_opt)),
        }
        convergence_history.append(iteration_info)

        if denominator <= 1e-9:
            # Degenerate: no redundancy among selected features
            break

        lam_unconstrained = numerator / denominator

        # Damped update (per manual Section G.2):
        # Blend with previous λ to prevent oscillation under heuristic solver
        lam_next = damping * lam_unconstrained + (1 - damping) * lam

        if abs(lam_next - lam) < tol:
            lam = lam_next
            convergence_history[-1]["converged"] = True
            break

        lam = lam_next

    if progress_callback:
        progress_callback(
            stage="Dinkelbach MIQ complete",
            progress=1.0,
            message=f"Converged at λ={lam:.6f} after {len(convergence_history)} iterations",
        )

    return x_opt, lam, convergence_history


def run_feature_selection_dinkelbach(
    progress_callback: Callable,
    features_matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    k_target: int = 6,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 2.0,
    max_iter: int = 15,
    tol: float = 1e-4,
    damping: float = 0.5,
    num_reads: int = 1000,
    run_brute_force: bool = True,
) -> Dict[str, Any]:
    """
    Full MIQ-Dinkelbach feature selection pipeline (Section G.2 stretch).

    Runs the Dinkelbach iteration, validates against brute force,
    and returns the selected features with convergence diagnostics.
    """
    n_features = features_matrix.shape[1]

    # Compute relevance and redundancy (same as MID)
    progress_callback(
        stage="Computing relevance/redundancy",
        progress=0.05,
        message=f"MIQ-Dinkelbach, damping={damping}, max_iter={max_iter}",
    )

    relevance = np.zeros(n_features)
    for i in range(n_features):
        feat_std = np.std(features_matrix[:, i])
        if feat_std < 1e-12:
            relevance[i] = 0.0
        else:
            corr = np.corrcoef(features_matrix[:, i], labels)[0, 1]
            relevance[i] = abs(corr) if not np.isnan(corr) else 0.0

    redundancy = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            si = np.std(features_matrix[:, i])
            sj = np.std(features_matrix[:, j])
            if si < 1e-12 or sj < 1e-12:
                redundancy[i, j] = redundancy[j, i] = 0.0
            else:
                corr = np.corrcoef(features_matrix[:, i], features_matrix[:, j])[0, 1]
                val = abs(corr) if not np.isnan(corr) else 0.0
                redundancy[i, j] = redundancy[j, i] = val

    # Brute-force validation (same MID QUBO for reference)
    bf_x, bf_energy, bf_indices = None, None, None
    if run_brute_force and n_features <= 20:
        progress_callback(
            stage="Brute force validation (MID reference)",
            progress=0.10,
            message=f"C({n_features},{k_target}) subsets",
        )
        Q_mid, _, _ = build_feature_selection_qubo(
            features_matrix, labels, k_target, alpha, beta, gamma
        )
        bf_x, bf_energy, bf_indices = brute_force_feature_selection(
            Q_mid, n_features, k_target, progress_callback
        )

    # Run Dinkelbach iteration
    progress_callback(
        stage="Dinkelbach iteration",
        progress=0.30,
        message="Starting MIQ optimization",
    )
    solver_x, final_lambda, convergence_history = solve_mrmr_dinkelbach(
        relevance, redundancy, n_features, k_target,
        max_iter, tol, damping, num_reads, gamma,
        progress_callback,
    )

    selected_indices = [i for i in range(n_features) if solver_x[i] > 0.5]
    selected_features = [feature_names[i] for i in selected_indices] if feature_names else []
    bf_selected_names = [feature_names[i] for i in bf_indices] if bf_indices and feature_names else []

    subset_match = None
    if bf_indices is not None:
        subset_match = set(selected_indices) == set(bf_indices)

    progress_callback(
        stage="MIQ feature selection complete",
        progress=1.0,
        message=f"Selected {len(selected_indices)}/{n_features}, "
                f"λ={final_lambda:.4f}, match_MID={subset_match}",
    )

    return {
        "method": "MIQ-Dinkelbach",
        "feature_names": feature_names,
        "relevance": relevance,
        "redundancy_matrix": redundancy,
        "selected_indices": selected_indices,
        "selected_features": selected_features,
        "final_lambda": final_lambda,
        "convergence_history": convergence_history,
        "n_iterations": len(convergence_history),
        "converged": any(h.get("converged", False) for h in convergence_history),
        "damping": damping,
        # MID brute-force reference
        "brute_force_selected": bf_indices,
        "brute_force_selected_features": bf_selected_names,
        "subset_match_vs_mid": subset_match,
    }

