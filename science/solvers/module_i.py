"""
QT-2.23 — Module I: Solver Adapter

Single interface wrapping all available solvers:
- neal.SimulatedAnnealingSampler (QUBO problems)
- scipy.optimize SLSQP (continuous fusion weights)
- scipy.optimize differential_evolution (continuous fusion weights)

BQPhy/QuantumNow is NOT available. The adapter interface is designed
so it could accept a real BQPhy backend later (one-line swap), but
the only solvers that actually run are the backup paths above.

The Solver page always shows "Backup Solver · Simulated Annealing"
computed from which solver actually executed the last run, NOT a
static string.
"""

from __future__ import annotations
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class SolverResult:
    """Result from any solver call."""

    def __init__(
        self,
        solution: np.ndarray,
        objective: float,
        solve_time_s: float,
        solver_name: str,
        metadata: Dict[str, Any],
    ):
        self.solution = solution
        self.objective = objective
        self.solve_time_s = solve_time_s
        self.solver_name = solver_name
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solution": self.solution.tolist(),
            "objective": self.objective,
            "solve_time_s": self.solve_time_s,
            "solver_name": self.solver_name,
            "metadata": self.metadata,
        }


class SolverAdapter:
    """
    Unified solver interface. Currently backup-only.

    Usage:
        adapter = SolverAdapter()
        result = adapter.solve_qubo(Q, num_reads=1000)
        result = adapter.solve_continuous(objective_fn, bounds, constraints)
    """

    def __init__(self):
        self._last_solver_used: Optional[str] = None
        self._solve_history: List[Dict[str, Any]] = []

    @property
    def backend_status(self) -> str:
        """Returns honest status for Solver page."""
        return "UNAVAILABLE — Backup Active"

    @property
    def last_solver_used(self) -> str:
        return self._last_solver_used or "No solver executed yet"

    @property
    def solver_label(self) -> str:
        """Dynamic label based on what actually ran."""
        if self._last_solver_used:
            return self._last_solver_used
        return "Backup Solver · Simulated Annealing / SLSQP"

    def solve_qubo(
        self,
        Q: np.ndarray,
        num_reads: int = 1000,
        progress_callback: Optional[Callable] = None,
    ) -> SolverResult:
        """
        Solve a QUBO problem using neal SimulatedAnnealingSampler.
        """
        import dimod
        import neal

        n = Q.shape[0]

        if progress_callback:
            progress_callback(stage="Building BQM", progress=0.1, message=f"{n} variables")

        linear = {i: float(Q[i, i]) for i in range(n)}
        quadratic = {}
        for i in range(n):
            for j in range(i + 1, n):
                val = float(Q[i, j] + Q[j, i])
                if abs(val) > 1e-15:
                    quadratic[(i, j)] = val

        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

        if progress_callback:
            progress_callback(stage="Sampling", progress=0.3, message=f"{num_reads} reads")

        start = time.time()
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample(bqm, num_reads=num_reads, seed=42)
        solve_time = time.time() - start

        best = response.first
        solution = np.array([best.sample.get(i, 0) for i in range(n)])
        energy = float(best.energy)

        solver_name = "Backup Solver · Simulated Annealing"
        self._last_solver_used = solver_name

        metadata = {
            "solver": "neal.SimulatedAnnealingSampler",
            "num_reads": num_reads,
            "solve_time_s": solve_time,
            "n_variables": n,
            "non_zero_couplings": len(quadratic),
            "matrix_density": len(quadratic) / (n * (n - 1) / 2) if n > 1 else 0,
        }

        record = {
            "timestamp": time.time(),
            "solver": solver_name,
            "problem_type": "QUBO",
            "n_variables": n,
            "solve_time_s": solve_time,
            "objective": energy,
            "status": "COMPLETED",
        }
        self._solve_history.append(record)

        if progress_callback:
            progress_callback(stage="Complete", progress=1.0, message=f"E={energy:.4f}")

        return SolverResult(solution, energy, solve_time, solver_name, metadata)

    def solve_continuous(
        self,
        objective_fn: Callable,
        bounds: List[Tuple[float, float]],
        constraints: Optional[Dict] = None,
        n_restarts: int = 10,
        seed: int = 42,
        progress_callback: Optional[Callable] = None,
    ) -> SolverResult:
        """
        Solve a continuous optimization problem via SLSQP.
        """
        from scipy.optimize import minimize

        rng = np.random.default_rng(seed)
        n = len(bounds)
        best_result, best_val = None, np.inf

        start = time.time()

        for restart in range(n_restarts):
            w0 = rng.dirichlet(np.ones(n))
            result = minimize(
                objective_fn, w0,
                bounds=bounds,
                constraints=constraints or {},
                method='SLSQP',
                options={'maxiter': 200},
            )
            if result.fun < best_val:
                best_val, best_result = result.fun, result

            if progress_callback:
                progress_callback(
                    stage="SLSQP",
                    progress=(restart + 1) / n_restarts,
                    message=f"Restart {restart+1}/{n_restarts}",
                )

        solve_time = time.time() - start

        solver_name = "Backup Solver · SLSQP"
        self._last_solver_used = solver_name

        record = {
            "timestamp": time.time(),
            "solver": solver_name,
            "problem_type": "Continuous",
            "n_variables": n,
            "solve_time_s": solve_time,
            "objective": -best_val,
            "status": "COMPLETED",
        }
        self._solve_history.append(record)

        return SolverResult(best_result.x, -best_val, solve_time, solver_name, {
            "solver": "scipy.optimize.SLSQP",
            "n_restarts": n_restarts,
            "solve_time_s": solve_time,
        })

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._solve_history)


# Global singleton
solver_adapter = SolverAdapter()
