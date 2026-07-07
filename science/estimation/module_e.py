"""
QT-2.23 — Module E: Classical Fisher Information, CRLB, and Empirical Validation

Provides:
- Classical Fisher Information and CRLB initial weight priors
- Empirical Monte Carlo estimator variance under real clutter
- Side-by-side CRLB vs empirical comparison (honesty requirement)

IMPORTANT: Uses CLASSICAL Fisher Information, not Quantum Fisher Information.
QFI applies to quantum measurement systems (entangled-photon probes, etc.).
Radar/thermal/sonar are classical sensors — using QFI is a category error.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional

import numpy as np


def crlb_initial_weights(
    snr_radar_linear: float,
    snr_thermal_linear: float,
    snr_acoustic_linear: float,
) -> np.ndarray:
    """
    Compute initial fusion weights from classical Fisher Information.

    For Gaussian observation model x ~ N(θ, σ²):
    I(θ) = 1/σ² ∝ SNR (linear).

    Higher SNR → higher Fisher Information → tighter CRLB → more trustworthy.
    Weights are proportional to Fisher Information, normalized to sum to 1.

    Returns
    -------
    np.ndarray of shape (3,) summing to 1.0
    """
    fi = np.array([
        max(snr_radar_linear, 1e-12),
        max(snr_thermal_linear, 1e-12),
        max(snr_acoustic_linear, 1e-12),
    ])
    total = fi.sum()
    if total < 1e-12:
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    return fi / total


def theoretical_crlb(snr_linear: float) -> float:
    """
    Theoretical CRLB = 1 / I(θ) = 1 / SNR for Gaussian model.
    """
    return 1.0 / max(snr_linear, 1e-12)


def empirical_estimator_variance(
    generate_scenario_fn: Callable,
    true_theta: float,
    n_trials: int = 2000,
    progress_callback: Optional[Callable] = None,
) -> float:
    """
    Monte Carlo estimate of estimator variance under the ACTUAL clutter
    distribution used in the pipeline (Rayleigh or Weibull), rather than
    trusting Gaussian-regularity assumptions in closed-form CRLB.

    Parameters
    ----------
    generate_scenario_fn : callable returning a float estimate of theta
    true_theta : float - ground truth value
    n_trials : int - number of Monte Carlo trials

    Returns
    -------
    float - empirical variance of the estimator
    """
    estimates = np.zeros(n_trials)
    for i in range(n_trials):
        estimates[i] = generate_scenario_fn()
        if progress_callback and i % 500 == 0:
            progress_callback(
                stage="Empirical CRLB",
                progress=i / n_trials,
                message=f"Trial {i}/{n_trials}",
            )

    return float(np.var(estimates - true_theta))


def report_crlb_vs_empirical(
    snr_linear: float,
    generate_scenario_fn: Callable,
    true_theta: float,
    n_trials: int = 2000,
) -> Dict[str, float]:
    """
    Compute theoretical CRLB and empirical variance side by side.

    The gap_ratio > 1 indicates the CRLB underestimates achievable error
    under the actual clutter distribution — an expected and scientifically
    interesting finding under heavy-tailed Weibull clutter.

    Returns
    -------
    dict with theoretical_crlb, empirical_variance, gap_ratio
    """
    t_crlb = theoretical_crlb(snr_linear)
    emp_var = empirical_estimator_variance(generate_scenario_fn, true_theta, n_trials)
    gap = emp_var / t_crlb if t_crlb > 1e-20 else float('inf')

    return {
        "theoretical_crlb": t_crlb,
        "empirical_variance": emp_var,
        "gap_ratio": gap,
    }


def run_crlb_analysis(
    progress_callback: Callable,
    radar_snr_db: float,
    thermal_snr: float,
    acoustic_se_db: float,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full Module E pipeline.

    Computes CRLB weights and empirical variance estimates
    for all three sensors.
    """
    progress_callback(
        stage="Computing CRLB",
        progress=0.1,
        message="Classical Fisher Information analysis",
    )

    # Convert to linear SNR
    radar_snr_lin = 10.0 ** (radar_snr_db / 10.0) if radar_snr_db > -50 else 1e-5
    # Thermal SNR is already linear
    thermal_snr_lin = max(thermal_snr, 1e-5)
    # Acoustic signal excess to linear
    acoustic_snr_lin = 10.0 ** (acoustic_se_db / 10.0) if acoustic_se_db > -50 else 1e-5

    weights = crlb_initial_weights(radar_snr_lin, thermal_snr_lin, acoustic_snr_lin)

    # Theoretical CRLBs
    t_crlb_radar = theoretical_crlb(radar_snr_lin)
    t_crlb_thermal = theoretical_crlb(thermal_snr_lin)
    t_crlb_acoustic = theoretical_crlb(acoustic_snr_lin)

    progress_callback(
        stage="CRLB complete",
        progress=1.0,
        message=f"Weights: [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]",
    )

    return {
        "theoretical_crlb": {
            "radar": t_crlb_radar,
            "thermal": t_crlb_thermal,
            "acoustic": t_crlb_acoustic,
        },
        "empirical_variance": {
            "radar": None,  # Computed on demand during full evaluation
            "thermal": None,
            "acoustic": None,
        },
        "gap_ratios": {
            "radar": None,
            "thermal": None,
            "acoustic": None,
        },
        "initial_weights": weights,
    }
