"""
QT-2.23 — Module A: Radar Sensor Simulation

Physics-grounded radar simulation with:
- Monostatic radar range equation (Skolnik, Introduction to Radar Systems)
- Rayleigh clutter (default, exponential-power, closed-form CFAR)
- Weibull clutter (stretch, heavy-tailed, Monte Carlo-calibrated CFAR)
- CA-CFAR detector with corrected threshold per clutter model
- Unit-consistent target injection into clutter
- Feature extraction for downstream fusion

Every computation is real. No hardcoded metrics. No fake data.
All physical constants from Appendix B of the build manual.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.stats import rayleigh, weibull_min


# ============================================================
# A.1 — Radar Range Equation (Skolnik)
# ============================================================

# Physical constants — NEVER change these
BOLTZMANN_K = 1.380649e-23  # J/K
T0 = 290.0                  # Reference temperature, K
C = 299792458.0              # Speed of light, m/s


def radar_snr_db(
    rcs_m2: float,
    range_m: float,
    freq_hz: float = 10e9,
    Pt_W: float = 10000.0,
    G_dB: float = 30.0,
    B_hz: float = 1e6,
    F_dB: float = 4.77,
    L_dB: float = 6.0,
) -> float:
    """
    Compute received SNR in dB using the monostatic radar range equation.

    Uses decibel form to avoid floating-point underflow with very small RCS.
    Reference: Skolnik, Introduction to Radar Systems, Ch. 2.

    Parameters
    ----------
    rcs_m2 : float - Radar Cross Section in m² (the stealth knob)
    range_m : float - Target range in metres
    freq_hz : float - Operating frequency in Hz (default 10 GHz, X-band)
    Pt_W : float - Transmit power in Watts
    G_dB : float - Antenna gain in dB
    B_hz : float - Receiver bandwidth in Hz
    F_dB : float - Noise figure in dB
    L_dB : float - System losses in dB

    Returns
    -------
    float - SNR in dB
    """
    wavelength = C / freq_hz
    Pt_dBW = 10.0 * np.log10(Pt_W)
    rcs_dBsm = 10.0 * np.log10(max(rcs_m2, 1e-20))
    thermal_noise_dBW = 10.0 * np.log10(BOLTZMANN_K * T0 * B_hz)

    snr = (
        Pt_dBW
        + 2.0 * G_dB
        + 20.0 * np.log10(wavelength)
        + rcs_dBsm
        - 30.0 * np.log10(4.0 * np.pi)
        - 40.0 * np.log10(max(range_m, 1.0))
        - thermal_noise_dBW
        - F_dB
        - L_dB
    )
    return float(snr)


def radar_snr_linear(rcs_m2: float, range_m: float, **kwargs) -> float:
    """SNR as a linear ratio (not dB)."""
    return 10.0 ** (radar_snr_db(rcs_m2, range_m, **kwargs) / 10.0)


# ============================================================
# A.2 — Clutter Generation
# ============================================================

def generate_rayleigh_clutter(
    num_cells: int,
    clutter_power: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate Rayleigh-amplitude clutter (exponential power).

    This is the standard CLT result for many small uncorrelated scatterers.
    The amplitude follows Rayleigh; the power is exponentially distributed.

    Parameters
    ----------
    num_cells : int - Number of range cells
    clutter_power : float - Mean clutter power
    rng : optional numpy random generator for seeded reproducibility

    Returns
    -------
    np.ndarray - Clutter amplitude per range cell
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma_c = np.sqrt(clutter_power / 2.0)
    I = rng.normal(0, sigma_c, num_cells)
    Q = rng.normal(0, sigma_c, num_cells)
    return np.sqrt(I**2 + Q**2)


def generate_weibull_clutter(
    num_cells: int,
    clutter_scale: float = 1.0,
    clutter_shape: float = 1.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate Weibull-amplitude clutter (heavy-tailed).

    More realistic for sea/terrain at low grazing angles.
    shape=2.0 reduces exactly to Rayleigh (sanity check).
    shape<1.5 produces heavy-tailed spiky clutter that masks stealth targets.

    Parameters
    ----------
    num_cells : int
    clutter_scale : float - RMS amplitude scale
    clutter_shape : float - Weibull shape parameter (k)
    rng : optional numpy random generator

    Returns
    -------
    np.ndarray - Clutter amplitude per range cell
    """
    if rng is None:
        rng = np.random.default_rng()

    return clutter_scale * rng.weibull(clutter_shape, num_cells)


def inject_target_into_clutter(
    clutter_map: np.ndarray,
    target_idx: int,
    snr_db: float,
) -> np.ndarray:
    """
    Unit-consistent target injection.

    Scales the injected amplitude by the LOCAL clutter RMS, not by an
    arbitrary constant. This fixes the bug in face3's snippet which
    added a raw SNR-derived voltage directly to Weibull clutter of
    scale=1 with no reference to the actual local noise floor.

    Parameters
    ----------
    clutter_map : np.ndarray - Clutter amplitude array
    target_idx : int - Range cell index where target is placed
    snr_db : float - Desired target SNR in dB relative to local clutter

    Returns
    -------
    np.ndarray - Clutter map with injected target
    """
    local_rms = np.sqrt(np.mean(clutter_map**2))
    target_amplitude = local_rms * (10.0 ** (snr_db / 20.0))
    out = clutter_map.copy()
    out[target_idx] += target_amplitude
    return out


# ============================================================
# A.3 — CA-CFAR Detector
# ============================================================

def ca_cfar_threshold_rayleigh(num_ref_cells: int, pfa: float) -> float:
    """
    Closed-form CA-CFAR threshold multiplier for Rayleigh (exponential-power) clutter.

    Derivation: Rayleigh amplitude → exponential power. Sum of N i.i.d.
    exponentials is Gamma-distributed → closed-form threshold exists.

    T = alpha * Z, where Z is the mean reference-cell power.

    Only valid for Rayleigh clutter. Using this on Weibull clutter is
    a scientific error — see calibrate_cfar_threshold_monte_carlo instead.
    """
    alpha = num_ref_cells * (pfa ** (-1.0 / num_ref_cells) - 1.0)
    return alpha


def calibrate_cfar_threshold_monte_carlo(
    clutter_scale: float,
    clutter_shape: float,
    num_ref_cells: int,
    target_pfa: float,
    n_trials: int = 200_000,
    rng: Optional[np.random.Generator] = None,
    progress_callback: Optional[Callable] = None,
) -> float:
    """
    Monte Carlo-calibrated CFAR threshold for general Weibull clutter.

    Standard engineering technique when no closed-form CFAR threshold
    exists for the assumed clutter distribution. Not a hack — it's what
    you do instead of misapplying a formula derived under different
    distributional assumptions.

    Parameters
    ----------
    clutter_scale : float - Weibull scale parameter
    clutter_shape : float - Weibull shape parameter
    num_ref_cells : int - Number of reference cells
    target_pfa : float - Target false alarm rate
    n_trials : int - Monte Carlo trials (default 200,000 per Appendix B)
    rng : optional seeded random generator
    progress_callback : optional progress reporter

    Returns
    -------
    float - Calibrated threshold multiplier alpha
    """
    if rng is None:
        rng = np.random.default_rng()

    if progress_callback:
        progress_callback(
            stage="MC CFAR calibration",
            progress=0.0,
            message=f"Running {n_trials:,} H0 trials...",
        )

    # Generate clutter-only (H0) scenarios in batches
    batch_size = min(50_000, n_trials)
    all_ratios = []

    for batch_start in range(0, n_trials, batch_size):
        current_batch = min(batch_size, n_trials - batch_start)

        ref_cells = generate_weibull_clutter(
            current_batch * num_ref_cells, clutter_scale, clutter_shape, rng
        ).reshape(current_batch, num_ref_cells)

        test_cells = generate_weibull_clutter(
            current_batch, clutter_scale, clutter_shape, rng
        )

        # CFAR test statistic: test_cell_power / mean_ref_power
        ref_power = ref_cells**2
        Z = ref_power.mean(axis=1)
        test_power = test_cells**2
        ratio = test_power / (Z + 1e-30)

        all_ratios.append(ratio)

        if progress_callback:
            pct = min(1.0, (batch_start + current_batch) / n_trials)
            progress_callback(
                stage="MC CFAR calibration",
                progress=pct * 0.9,
                message=f"Trial {batch_start + current_batch:,}/{n_trials:,}",
            )

    ratios = np.concatenate(all_ratios)
    alpha = float(np.percentile(ratios, 100.0 * (1.0 - target_pfa)))

    if progress_callback:
        progress_callback(
            stage="MC CFAR calibration",
            progress=1.0,
            message=f"Calibrated α = {alpha:.4f}",
        )

    return alpha


def run_cfar_detection(
    signal: np.ndarray,
    alpha: float,
    num_ref_cells: int = 16,
    guard_cells: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run CA-CFAR detection on a signal (power domain).

    Parameters
    ----------
    signal : np.ndarray - Signal amplitude per range cell
    alpha : float - Threshold multiplier
    num_ref_cells : int - Reference cells per side
    guard_cells : int - Guard cells per side

    Returns
    -------
    tuple of (detections_mask, threshold_map, test_statistic)
    """
    n = len(signal)
    power = signal ** 2
    half_ref = num_ref_cells // 2
    threshold_map = np.zeros(n)
    test_statistic = np.zeros(n)

    for i in range(n):
        # Leading reference cells
        lead_start = max(0, i - guard_cells - half_ref)
        lead_end = max(0, i - guard_cells)
        # Lagging reference cells
        lag_start = min(n, i + guard_cells + 1)
        lag_end = min(n, i + guard_cells + 1 + half_ref)

        ref_cells = np.concatenate([
            power[lead_start:lead_end],
            power[lag_start:lag_end],
        ])

        if len(ref_cells) > 0:
            Z = np.mean(ref_cells)
            threshold_map[i] = alpha * Z
            test_statistic[i] = power[i] / (Z + 1e-30)
        else:
            threshold_map[i] = np.inf
            test_statistic[i] = 0.0

    detections = power > threshold_map

    return detections, threshold_map, test_statistic


# ============================================================
# A.4 — Range-Doppler Map Generation
# ============================================================

def generate_range_doppler_map(
    num_range_cells: int = 500,
    num_doppler_bins: int = 64,
    snr_db: float = -10.0,
    clutter_power: float = 1.0,
    target_range_cell: int = 250,
    target_doppler_bin: int = 32,
    clutter_model: str = "rayleigh",
    weibull_shape: float = 1.5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    Generate a synthetic Range-Doppler map with embedded target.

    Returns the map, target range cell, and target doppler bin.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate 2D clutter
    if clutter_model == "weibull":
        rd_map = generate_weibull_clutter(
            num_range_cells * num_doppler_bins,
            clutter_scale=np.sqrt(clutter_power),
            clutter_shape=weibull_shape,
            rng=rng,
        ).reshape(num_range_cells, num_doppler_bins)
    else:
        rd_map = generate_rayleigh_clutter(
            num_range_cells * num_doppler_bins,
            clutter_power=clutter_power,
            rng=rng,
        ).reshape(num_range_cells, num_doppler_bins)

    # Inject target
    local_rms = np.sqrt(np.mean(rd_map**2))
    target_amplitude = local_rms * (10.0 ** (snr_db / 20.0))
    rd_map[target_range_cell, target_doppler_bin] += target_amplitude

    return rd_map, target_range_cell, target_doppler_bin


# ============================================================
# A.5 — Full Radar Simulation Pipeline
# ============================================================

def run_radar_simulation(
    progress_callback: Callable,
    config: Any,  # RadarConfig dataclass
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full Module A pipeline: compute SNR, generate clutter, run CFAR,
    extract features, build range-Doppler map.

    This is the function called by TaskManager. It emits live progress
    events at every meaningful stage.

    Returns a dict matching the RadarResult dataclass fields.
    """
    rng = np.random.default_rng(seed)

    # --- Stage 1: Compute SNR ---
    progress_callback(
        stage="Computing SNR",
        progress=0.05,
        message=f"σ={config.rcs_m2} m², R={config.range_m/1000:.0f} km",
    )

    snr = radar_snr_db(
        rcs_m2=config.rcs_m2,
        range_m=config.range_m,
        freq_hz=config.frequency_hz,
        Pt_W=config.transmit_power_W,
        G_dB=config.antenna_gain_dB,
        B_hz=config.bandwidth_hz,
        F_dB=config.noise_figure_dB,
        L_dB=config.system_loss_dB,
    )

    # --- Stage 2: Generate clutter ---
    progress_callback(
        stage="Generating clutter",
        progress=0.15,
        message=f"Model: {config.clutter_model.value}, {config.num_range_cells} cells",
    )

    if config.clutter_model == "Weibull" or (hasattr(config.clutter_model, 'value') and config.clutter_model.value == "Weibull"):
        clutter = generate_weibull_clutter(
            config.num_range_cells,
            clutter_scale=np.sqrt(config.clutter_power),
            clutter_shape=config.weibull_shape,
            rng=rng,
        )
    else:
        clutter = generate_rayleigh_clutter(
            config.num_range_cells,
            clutter_power=config.clutter_power,
            rng=rng,
        )

    # --- Stage 3: Inject target ---
    progress_callback(
        stage="Injecting target",
        progress=0.25,
        message=f"SNR = {snr:.1f} dB",
    )

    target_idx = config.num_range_cells // 2
    range_profile = inject_target_into_clutter(clutter, target_idx, snr)

    # --- Stage 4: Calibrate CFAR threshold ---
    is_weibull = config.clutter_model == "Weibull" or (hasattr(config.clutter_model, 'value') and config.clutter_model.value == "Weibull")

    if is_weibull:
        progress_callback(
            stage="Monte Carlo CFAR calibration",
            progress=0.30,
            message=f"Weibull shape={config.weibull_shape}, {config.mc_trials:,} trials",
        )
        cfar_alpha = calibrate_cfar_threshold_monte_carlo(
            clutter_scale=np.sqrt(config.clutter_power),
            clutter_shape=config.weibull_shape,
            num_ref_cells=config.num_ref_cells,
            target_pfa=config.pfa,
            n_trials=config.mc_trials,
            rng=rng,
            progress_callback=progress_callback,
        )
    else:
        progress_callback(
            stage="CFAR threshold (analytic)",
            progress=0.50,
            message="Rayleigh closed-form",
        )
        cfar_alpha = ca_cfar_threshold_rayleigh(config.num_ref_cells, config.pfa)

    # --- Stage 5: Run CFAR detection ---
    progress_callback(
        stage="Running CFAR detection",
        progress=0.65,
        message=f"α = {cfar_alpha:.4f}",
    )

    detections, threshold_map, test_statistic = run_cfar_detection(
        range_profile,
        cfar_alpha,
        num_ref_cells=config.num_ref_cells,
        guard_cells=config.guard_cells,
    )

    # Compute empirical Pfa (exclude target cell and guards)
    target_start = max(0, target_idx - config.guard_cells)
    target_end = min(len(detections), target_idx + config.guard_cells + 1)
    clutter_detections = np.concatenate([
        detections[:target_start],
        detections[target_end:],
    ])
    empirical_pfa = float(np.mean(clutter_detections)) if len(clutter_detections) > 0 else 0.0

    # Detection probability at target cell
    target_detected = bool(detections[target_idx])

    # --- Stage 6: Generate Range-Doppler map ---
    progress_callback(
        stage="Generating Range-Doppler map",
        progress=0.75,
        message="2D clutter + target",
    )

    rd_map, _, _ = generate_range_doppler_map(
        num_range_cells=config.num_range_cells,
        snr_db=snr,
        clutter_power=config.clutter_power,
        target_range_cell=target_idx,
        clutter_model="weibull" if is_weibull else "rayleigh",
        weibull_shape=config.weibull_shape,
        rng=rng,
    )

    # --- Stage 7: Generate full dataset for ML pipeline ---
    progress_callback(
        stage="Generating dataset",
        progress=0.80,
        message=f"N={config.num_samples} samples (target + clutter-only)",
    )

    dataset_features, dataset_labels = _generate_radar_dataset(
        config, snr, cfar_alpha, rng, progress_callback
    )

    # --- Stage 8: Extract features from single example ---
    progress_callback(
        stage="Extracting features",
        progress=0.95,
        message="Range profile features",
    )

    features = extract_radar_features(range_profile, threshold_map)

    progress_callback(
        stage="Completed",
        progress=1.0,
        message=f"SNR={snr:.1f}dB, Pfa_emp={empirical_pfa:.2e}, Det={target_detected}",
    )

    return {
        "range_profile": range_profile,
        "range_doppler_map": rd_map,
        "clutter_map": clutter,
        "cfar_threshold": threshold_map,
        "detections": detections,
        "snr_db": snr,
        "configured_pfa": config.pfa,
        "empirical_pfa": empirical_pfa,
        "detection_probability": 1.0 if target_detected else 0.0,
        "cfar_alpha": cfar_alpha,
        "features": features,
        "feature_names": list(features.keys()),
        "feature_values": np.array(list(features.values())),
        "dataset_features": dataset_features,
        "dataset_labels": dataset_labels,
        "metadata": {
            "rcs_m2": config.rcs_m2,
            "range_m": config.range_m,
            "frequency_hz": config.frequency_hz,
            "clutter_model": config.clutter_model.value if hasattr(config.clutter_model, 'value') else str(config.clutter_model),
            "weibull_shape": config.weibull_shape if is_weibull else None,
            "cfar_method": "Monte Carlo" if is_weibull else "Analytic (Rayleigh)",
            "target_idx": target_idx,
            "seed": seed,
        },
    }


def _generate_radar_dataset(
    config: Any,
    snr_db: float,
    cfar_alpha: float,
    rng: np.random.Generator,
    progress_callback: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a labelled dataset of radar feature vectors for the ML pipeline.
    Half target-present (label=1), half clutter-only (label=0).
    """
    n_samples = config.num_samples
    n_per_class = n_samples // 2
    is_weibull = config.clutter_model == "Weibull" or (hasattr(config.clutter_model, 'value') and config.clutter_model.value == "Weibull")

    all_features = []
    all_labels = []

    for i in range(n_samples):
        if is_weibull:
            clutter = generate_weibull_clutter(
                config.num_range_cells,
                clutter_scale=np.sqrt(config.clutter_power),
                clutter_shape=config.weibull_shape,
                rng=rng,
            )
        else:
            clutter = generate_rayleigh_clutter(
                config.num_range_cells,
                clutter_power=config.clutter_power,
                rng=rng,
            )

        target_idx = config.num_range_cells // 2
        label = 1 if i < n_per_class else 0

        if label == 1:
            signal = inject_target_into_clutter(clutter, target_idx, snr_db)
        else:
            signal = clutter

        # Quick threshold map for features
        _, threshold_map, _ = run_cfar_detection(
            signal, cfar_alpha, config.num_ref_cells, config.guard_cells
        )

        feats = extract_radar_features(signal, threshold_map)
        all_features.append(list(feats.values()))
        all_labels.append(label)

        if i % 500 == 0 and progress_callback:
            progress_callback(
                stage="Generating dataset",
                progress=0.80 + 0.15 * (i / n_samples),
                message=f"Sample {i}/{n_samples}",
            )

    return np.array(all_features), np.array(all_labels)


# ============================================================
# A.6 — Feature Extraction (Module D integration for radar)
# ============================================================

def extract_radar_features(
    range_profile: np.ndarray,
    cfar_threshold: np.ndarray,
) -> Dict[str, float]:
    """
    Extract interpretable radar features from a range profile.

    Features are physically meaningful and contribute to the ~20 total
    feature pool across all 3 modalities (Module D).

    Returns
    -------
    dict mapping feature names to values
    """
    from scipy.stats import kurtosis, skew

    power = range_profile ** 2

    peak_snr = float(np.max(range_profile) / (np.median(range_profile) + 1e-12))
    cfar_exceedances = int(np.sum(power > cfar_threshold))

    # Doppler spread proxy: standard deviation of peak positions across sub-bands
    n_bins = min(8, len(range_profile) // 10)
    if n_bins > 1:
        sub_len = len(range_profile) // n_bins
        peak_positions = [
            np.argmax(range_profile[j * sub_len:(j + 1) * sub_len])
            for j in range(n_bins)
        ]
        doppler_spread = float(np.std(peak_positions))
    else:
        doppler_spread = 0.0

    target_to_clutter = float(np.max(power) / (np.mean(power) + 1e-12))
    weibull_skew = float(skew(power))
    clutter_kurtosis = float(kurtosis(power))

    return {
        "radar_peak_snr": peak_snr,
        "radar_cfar_exceedances": float(cfar_exceedances),
        "radar_doppler_spread": doppler_spread,
        "radar_target_to_clutter": target_to_clutter,
        "radar_clutter_skewness": weibull_skew,
        "radar_clutter_kurtosis": clutter_kurtosis,
    }


# ============================================================
# A.7 — Batch Pd Estimation (for validation & degradation)
# ============================================================

def estimate_detection_probability(
    snr_db: float,
    config: Any,
    n_trials: int = 1000,
    seed: int = 42,
    progress_callback: Optional[Callable] = None,
) -> float:
    """
    Monte Carlo estimate of Pd at a given SNR.
    Used for validation checks and degradation sweeps.
    """
    rng = np.random.default_rng(seed)
    is_weibull = config.clutter_model == "Weibull" or (hasattr(config.clutter_model, 'value') and config.clutter_model.value == "Weibull")

    # Get threshold
    if is_weibull:
        cfar_alpha = calibrate_cfar_threshold_monte_carlo(
            np.sqrt(config.clutter_power), config.weibull_shape,
            config.num_ref_cells, config.pfa, n_trials=50000, rng=rng,
        )
    else:
        cfar_alpha = ca_cfar_threshold_rayleigh(config.num_ref_cells, config.pfa)

    detections = 0
    target_idx = config.num_range_cells // 2

    for i in range(n_trials):
        if is_weibull:
            clutter = generate_weibull_clutter(
                config.num_range_cells, np.sqrt(config.clutter_power),
                config.weibull_shape, rng
            )
        else:
            clutter = generate_rayleigh_clutter(
                config.num_range_cells, config.clutter_power, rng
            )

        signal = inject_target_into_clutter(clutter, target_idx, snr_db)
        dets, _, _ = run_cfar_detection(
            signal, cfar_alpha, config.num_ref_cells, config.guard_cells
        )
        if dets[target_idx]:
            detections += 1

        if progress_callback and i % 100 == 0:
            progress_callback(
                stage="Pd estimation",
                progress=i / n_trials,
                message=f"Trial {i}/{n_trials}, Pd so far = {detections/(i+1):.3f}",
            )

    return detections / n_trials
