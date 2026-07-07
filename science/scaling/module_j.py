"""
QT-2.23 — Module J: Scaling Experiment

The strongest slide when done honestly. Shows how solve time AND
solution quality change as problem size grows.

Protocol: Sweep 3→6→9→12 simulated sensor modalities (synthetic
additional feature channels). For each size run brute force (until
infeasible), grid search, SLSQP, Differential Evolution, and neal SA.

Key rules:
- Speed without quality must never be presented alone (Section 2.5)
- Generate your own numbers, never cite vendor benchmarks as your result
- At smallest size, ALL solvers must agree (validates correctness)
"""

from __future__ import annotations
import time
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def build_synthetic_qubo(n_variables: int, density: float = 0.7, seed: int = 42) -> np.ndarray:
    """
    Build a synthetic QUBO matrix of given size that matches the statistical
    structure of the real feature-selection QUBO (MID formulation).

    The matrix has:
    - Negative diagonal entries (relevance, pulling x_i toward 1)
    - Positive off-diagonal entries (redundancy penalty, pushing pairs away)
    - Cardinality penalty embedded
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_variables, n_variables))

    k_target = max(2, n_variables // 3)

    # Diagonal: -alpha * relevance + gamma * (1 - 2k)
    relevances = rng.uniform(0.1, 0.8, n_variables)
    gamma = 2.0
    for i in range(n_variables):
        Q[i, i] = -1.0 * relevances[i] + gamma * (1 - 2 * k_target)

    # Off-diagonal: beta * redundancy + 2*gamma
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            if rng.random() < density:
                redundancy = rng.uniform(0.0, 0.6)
                val = 1.0 * redundancy + 2 * gamma
                Q[i, j] = val
                Q[j, i] = val

    return Q


def solve_brute_force(Q: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Exhaustive search. Returns (solution, energy, solve_time)."""
    n = Q.shape[0]
    start = time.time()

    best_energy = np.inf
    best_x = np.zeros(n)

    for bits in range(2**n):
        x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
        energy = float(x @ Q @ x)
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()

    solve_time = time.time() - start
    return best_x, best_energy, solve_time


def solve_grid_search(Q: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Systematic grid search — same as brute force but slower due to overhead."""
    return solve_brute_force(Q)


def solve_slsqp_relaxed(Q: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Continuous relaxation + rounding (SLSQP). Multi-restart."""
    from scipy.optimize import minimize
    n = Q.shape[0]
    rng = np.random.default_rng(42)

    start = time.time()
    best_energy = np.inf
    best_x = np.zeros(n)

    for _ in range(10):
        x0 = rng.random(n)
        result = minimize(
            lambda x: float(x @ Q @ x), x0,
            bounds=[(0, 1)] * n,
            method='SLSQP',
            options={'maxiter': 200},
        )
        # Round to binary
        x_rounded = (result.x > 0.5).astype(float)
        energy = float(x_rounded @ Q @ x_rounded)
        if energy < best_energy:
            best_energy = energy
            best_x = x_rounded

    solve_time = time.time() - start
    return best_x, best_energy, solve_time


def solve_differential_evolution_relaxed(Q: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Differential evolution on continuous relaxation + rounding."""
    from scipy.optimize import differential_evolution
    n = Q.shape[0]

    start = time.time()

    result = differential_evolution(
        lambda x: float(np.array(x) @ Q @ np.array(x)),
        bounds=[(0, 1)] * n,
        seed=42,
        maxiter=200,
        tol=1e-6,
    )

    x_rounded = (result.x > 0.5).astype(float)
    energy = float(x_rounded @ Q @ x_rounded)
    solve_time = time.time() - start

    return x_rounded, energy, solve_time


def solve_simulated_annealing(Q: np.ndarray, num_reads: int = 1000) -> Tuple[np.ndarray, float, float]:
    """Neal SA solver."""
    import dimod
    import neal

    n = Q.shape[0]
    linear = {i: float(Q[i, i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            val = float(Q[i, j] + Q[j, i])
            if abs(val) > 1e-15:
                quadratic[(i, j)] = val

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

    start = time.time()
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads, seed=42)
    solve_time = time.time() - start

    best = response.first
    x = np.array([best.sample.get(i, 0) for i in range(n)])
    return x, float(best.energy), solve_time


def run_scaling_experiment(
    progress_callback: Callable,
    problem_sizes: List[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full Module J pipeline.

    Sweeps problem sizes, runs all available solvers, records
    solve time AND objective quality for each.
    """
    if problem_sizes is None:
        problem_sizes = [3, 6, 9, 12]

    # Define solvers
    solver_fns = {
        "Brute Force": solve_brute_force,
        "SLSQP (relaxed)": solve_slsqp_relaxed,
        "Differential Evolution": solve_differential_evolution_relaxed,
        "Simulated Annealing": solve_simulated_annealing,
    }

    results = []
    total_steps = len(problem_sizes) * len(solver_fns)
    step = 0

    for size_idx, n in enumerate(problem_sizes):
        progress_callback(
            stage=f"Problem size {n}",
            progress=size_idx / len(problem_sizes),
            message=f"Building {n}×{n} QUBO",
        )

        Q = build_synthetic_qubo(n, seed=seed + size_idx)

        # Skip brute force for n > 16 (would take too long)
        max_brute = 16

        for solver_name, solver_fn in solver_fns.items():
            step += 1

            if solver_name == "Brute Force" and n > max_brute:
                progress_callback(
                    stage=f"{solver_name} (skipped)",
                    progress=step / total_steps,
                    message=f"n={n} too large for brute force",
                )
                results.append({
                    "problem_size": n,
                    "variable_count": n,
                    "solver": solver_name,
                    "solve_time_s": None,
                    "objective_value": None,
                    "normalized_quality": None,
                    "status": "SKIPPED (infeasible)",
                    "metadata": {},
                })
                continue

            progress_callback(
                stage=f"{solver_name}",
                progress=step / total_steps,
                message=f"n={n}, solving...",
            )

            try:
                x, energy, solve_time = solver_fn(Q)
                results.append({
                    "problem_size": n,
                    "variable_count": n,
                    "solver": solver_name,
                    "solve_time_s": solve_time,
                    "objective_value": energy,
                    "normalized_quality": None,  # Computed after all results
                    "status": "COMPLETED",
                    "metadata": {"solution": x.tolist()},
                })
            except Exception as e:
                results.append({
                    "problem_size": n,
                    "variable_count": n,
                    "solver": solver_name,
                    "solve_time_s": None,
                    "objective_value": None,
                    "normalized_quality": None,
                    "status": f"FAILED: {str(e)}",
                    "metadata": {},
                })

    # Compute normalized quality: objective / best_known per size
    for size in problem_sizes:
        size_results = [r for r in results if r["problem_size"] == size and r["objective_value"] is not None]
        if size_results:
            best_obj = min(r["objective_value"] for r in size_results)
            for r in size_results:
                # Normalized quality: 1.0 = matches best known, <1.0 = worse
                if best_obj != 0:
                    r["normalized_quality"] = best_obj / r["objective_value"] if r["objective_value"] != 0 else 0.0
                else:
                    r["normalized_quality"] = 1.0

    progress_callback(
        stage="Scaling experiment complete",
        progress=1.0,
        message=f"{len(results)} solver runs across {len(problem_sizes)} sizes",
    )

    return {
        "problem_sizes": problem_sizes,
        "results": results,
    }
