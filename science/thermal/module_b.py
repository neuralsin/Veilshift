"""
QT-2.23 — Module B: Thermal/Infrared Sensor Simulation

Physics-grounded thermal sensor simulation with:
- Stefan-Boltzmann radiant exitance
- Emissivity-suppressed target contrast
- NETD-derived noise floor
- Corrected 1/f spatial noise generator (DC-bin-only patch)
- Power spectrum sanity check
- Feature extraction for downstream fusion

Reference: Stefan-Boltzmann law (Planck integral over all wavelengths).
All parameters from Appendix B of the build manual.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.stats import kurtosis


# Physical constant — Stefan-Boltzmann
SIGMA_SB = 5.670374e-8  # W/(m²·K⁴)


# ============================================================
# B.1 — Stefan-Boltzmann Physics
# ============================================================

def radiant_exitance(emissivity: float, temp_K: float) -> float:
    """
    Stefan-Boltzmann radiant exitance M = ε·σ_SB·T⁴ [W/m²].
    """
    return emissivity * SIGMA_SB * temp_K**4


def thermal_contrast(
    target_eps: float, target_T: float,
    bg_eps: float, bg_T: float,
) -> float:
    """
    ΔM = M_target - M_background [W/m²].
    Stealth targets suppress this via low-ε coatings and temperature management.
    """
    return radiant_exitance(target_eps, target_T) - radiant_exitance(bg_eps, bg_T)


def thermal_snr(
    delta_M: float,
    netd_K: float,
    eps_avg: float = 0.7,
    T_ref: float = 290.0,
) -> float:
    """
    Thermal SNR = ΔM / noise_floor.

    noise_floor derived from NETD via local derivative dM/dT.
    NETD is the actual spec IR camera manufacturers publish:
    50 mK (uncooled microbolometer), <20 mK (cooled).
    """
    dMdT = 4.0 * eps_avg * SIGMA_SB * T_ref**3
    noise_floor = dMdT * netd_K
    if noise_floor < 1e-20:
        return 0.0
    return delta_M / noise_floor


# ============================================================
# B.2 — 1/f Spatial Noise Generator (Corrected)
# ============================================================

def generate_1f_noise(
    shape: Tuple[int, int],
    alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate 1/f (pink) spatial noise.

    Real microbolometer arrays show 1/f spatial noise, not pure white Gaussian.

    CORRECTED version: face3's original overwrote the ENTIRE frequency grid
    with scalar 1.0 (freqs_sq = 1.0), which silently disabled all frequency
    shaping and produced plain white noise while claiming to be 1/f.

    This version patches ONLY the DC bin (freqs_sq[0,0] = 1e-6), preserving
    the frequency-dependent shaping for all other bins.

    Sanity check: plot log-log PSD — a straight downward-sloping line
    confirms 1/f shaping. A flat line means the bug is back.
    """
    if rng is None:
        rng = np.random.default_rng()

    white = rng.normal(0, 1, shape)
    f_noise = np.fft.fft2(white)

    fx = np.fft.fftfreq(shape[0])[:, None]
    fy = np.fft.fftfreq(shape[1])[None, :]
    freqs_sq = fx**2 + fy**2

    # Patch ONLY the DC bin, not the whole array
    freqs_sq[0, 0] = 1e-6

    f_noise = f_noise / (freqs_sq ** (alpha / 2.0))
    result = np.real(np.fft.ifft2(f_noise))

    # Normalize to unit variance
    std = np.std(result)
    if std > 1e-12:
        result = result / std

    return result


def generate_white_noise(
    shape: Tuple[int, int],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate white Gaussian spatial noise."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(0, 1, shape)


def compute_noise_psd(noise_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the radially-averaged power spectral density of a 2D noise field.

    Returns (frequencies, psd_values, fitted_slope).
    The slope should be approximately -alpha for 1/f noise.
    """
    psd_2d = np.abs(np.fft.fft2(noise_field))**2
    ny, nx = noise_field.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)
    freq_mag = np.sqrt(FX**2 + FY**2)

    # Radial binning
    n_bins = min(ny, nx) // 2
    freq_bins = np.linspace(0.01, 0.5, n_bins)
    psd_radial = np.zeros(n_bins - 1)
    freq_centers = np.zeros(n_bins - 1)

    for i in range(n_bins - 1):
        mask = (freq_mag >= freq_bins[i]) & (freq_mag < freq_bins[i + 1])
        if np.any(mask):
            psd_radial[i] = np.mean(psd_2d[mask])
            freq_centers[i] = (freq_bins[i] + freq_bins[i + 1]) / 2

    # Remove zeros
    valid = psd_radial > 0
    freq_centers = freq_centers[valid]
    psd_radial = psd_radial[valid]

    # Fit slope in log-log space
    if len(freq_centers) > 2:
        log_f = np.log10(freq_centers)
        log_p = np.log10(psd_radial)
        slope, _ = np.polyfit(log_f, log_p, 1)
    else:
        slope = 0.0

    return freq_centers, psd_radial, float(slope)


# ============================================================
# B.3 — Thermal Frame Generation
# ============================================================

def generate_thermal_frame(
    target_T: float,
    target_eps: float,
    bg_T: float,
    bg_eps: float,
    netd_K: float,
    frame_size: Tuple[int, int] = (64, 64),
    noise_model: str = "1/f",
    noise_alpha: float = 1.0,
    target_size: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic thermal frame with background, target, and noise.

    Returns (thermal_frame, contrast_map, noise_field).
    """
    if rng is None:
        rng = np.random.default_rng()

    ny, nx = frame_size

    # Background exitance field
    bg_exitance = radiant_exitance(bg_eps, bg_T)
    frame = np.full((ny, nx), bg_exitance)

    # Target region (centered square)
    target_exitance = radiant_exitance(target_eps, target_T)
    cy, cx = ny // 2, nx // 2
    half = target_size // 2
    y1, y2 = max(0, cy - half), min(ny, cy + half)
    x1, x2 = max(0, cx - half), min(nx, cx + half)
    frame[y1:y2, x1:x2] = target_exitance

    # Contrast map (before noise)
    contrast_map = frame - bg_exitance

    # Generate noise (strictly independent of target properties)
    dMdT = 4.0 * bg_eps * SIGMA_SB * bg_T**3
    noise_amplitude = dMdT * netd_K

    if noise_model == "1/f":
        noise_field = generate_1f_noise(frame_size, alpha=noise_alpha, rng=rng)
    else:
        noise_field = generate_white_noise(frame_size, rng=rng)

    noise_field_scaled = noise_field * noise_amplitude

    # Add noise to frame
    thermal_frame = frame + noise_field_scaled

    return thermal_frame, contrast_map, noise_field


# ============================================================
# B.4 — Feature Extraction (Module D integration for thermal)
# ============================================================

def extract_thermal_features(thermal_frame: np.ndarray) -> Dict[str, float]:
    """
    Extract interpretable thermal features from a thermal frame.
    """
    peak_contrast = float(np.max(thermal_frame) - np.median(thermal_frame))
    gy, gx = np.gradient(thermal_frame)
    edge_energy = float(np.mean(gx**2 + gy**2))
    threshold = np.median(thermal_frame) + 2.0 * np.std(thermal_frame)
    hot_spot_area = float(np.sum(thermal_frame > threshold))
    contrast_to_noise = float(peak_contrast / (np.std(thermal_frame) + 1e-12))
    frame_kurtosis = float(kurtosis(thermal_frame.flatten()))

    return {
        "thermal_peak_contrast": peak_contrast,
        "thermal_edge_energy": edge_energy,
        "thermal_hot_spot_area": hot_spot_area,
        "thermal_contrast_to_noise": contrast_to_noise,
        "thermal_kurtosis": frame_kurtosis,
    }


# ============================================================
# B.5 — Full Thermal Simulation Pipeline
# ============================================================

def run_thermal_simulation(
    progress_callback: Callable,
    config: Any,  # ThermalConfig dataclass
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full Module B pipeline. Called by TaskManager.
    Emits live progress events at every meaningful stage.
    """
    rng = np.random.default_rng(seed)

    # --- Stage 1: Compute physics ---
    progress_callback(
        stage="Computing exitance",
        progress=0.05,
        message=f"T_target={config.target_temperature_K}K, ε={config.target_emissivity}",
    )

    target_exit = radiant_exitance(config.target_emissivity, config.target_temperature_K)
    bg_exit = radiant_exitance(config.bg_emissivity, config.bg_temperature_K)
    delta_M = thermal_contrast(
        config.target_emissivity, config.target_temperature_K,
        config.bg_emissivity, config.bg_temperature_K,
    )
    snr_val = thermal_snr(delta_M, config.netd_K)

    # --- Stage 2: Generate frame ---
    progress_callback(
        stage="Generating thermal frame",
        progress=0.15,
        message=f"Frame {config.frame_size[0]}×{config.frame_size[1]}, noise={config.noise_model.value if hasattr(config.noise_model, 'value') else config.noise_model}",
    )

    noise_model_str = config.noise_model.value if hasattr(config.noise_model, 'value') else str(config.noise_model)
    thermal_frame, contrast_map, noise_field = generate_thermal_frame(
        target_T=config.target_temperature_K,
        target_eps=config.target_emissivity,
        bg_T=config.bg_temperature_K,
        bg_eps=config.bg_emissivity,
        netd_K=config.netd_K,
        frame_size=config.frame_size,
        noise_model=noise_model_str,
        noise_alpha=config.noise_alpha,
        rng=rng,
    )

    # --- Stage 3: PSD sanity check ---
    progress_callback(
        stage="Computing noise PSD",
        progress=0.30,
        message="Power spectrum analysis",
    )

    psd_freqs, psd_power, psd_slope = compute_noise_psd(noise_field)

    # --- Stage 4: Extract features from single example ---
    progress_callback(
        stage="Extracting features",
        progress=0.40,
        message="Thermal frame features",
    )

    features = extract_thermal_features(thermal_frame)

    # --- Stage 5: Generate full dataset ---
    progress_callback(
        stage="Generating dataset",
        progress=0.45,
        message=f"N={config.num_samples} samples",
    )

    dataset_features, dataset_labels = _generate_thermal_dataset(
        config, rng, progress_callback
    )

    progress_callback(
        stage="Completed",
        progress=1.0,
        message=f"ΔM={delta_M:.2f} W/m², SNR={snr_val:.2f}, slope={psd_slope:.2f}",
    )

    return {
        "thermal_frame": thermal_frame,
        "contrast_map": contrast_map,
        "noise_field": noise_field,
        "noise_psd_freqs": psd_freqs,
        "noise_psd_power": psd_power,
        "noise_psd_slope": psd_slope,
        "target_exitance": target_exit,
        "background_exitance": bg_exit,
        "delta_M": delta_M,
        "thermal_snr": snr_val,
        "features": features,
        "feature_names": list(features.keys()),
        "feature_values": np.array(list(features.values())),
        "dataset_features": dataset_features,
        "dataset_labels": dataset_labels,
        "metadata": {
            "target_T": config.target_temperature_K,
            "target_eps": config.target_emissivity,
            "bg_T": config.bg_temperature_K,
            "bg_eps": config.bg_emissivity,
            "netd_K": config.netd_K,
            "noise_model": noise_model_str,
            "noise_alpha": config.noise_alpha,
            "psd_slope": psd_slope,
            "seed": seed,
        },
    }


def _generate_thermal_dataset(
    config: Any,
    rng: np.random.Generator,
    progress_callback: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a labelled dataset of thermal feature vectors."""
    n_samples = config.num_samples
    n_per_class = n_samples // 2
    noise_model_str = config.noise_model.value if hasattr(config.noise_model, 'value') else str(config.noise_model)

    all_features = []
    all_labels = []

    for i in range(n_samples):
        label = 1 if i < n_per_class else 0

        if label == 1:
            t_T = config.target_temperature_K
            t_eps = config.target_emissivity
        else:
            # Clutter-only: target same as background
            t_T = config.bg_temperature_K
            t_eps = config.bg_emissivity

        frame, _, _ = generate_thermal_frame(
            target_T=t_T,
            target_eps=t_eps,
            bg_T=config.bg_temperature_K,
            bg_eps=config.bg_emissivity,
            netd_K=config.netd_K,
            frame_size=config.frame_size,
            noise_model=noise_model_str,
            noise_alpha=config.noise_alpha,
            rng=rng,
        )

        feats = extract_thermal_features(frame)
        all_features.append(list(feats.values()))
        all_labels.append(label)

        if i % 500 == 0 and progress_callback:
            progress_callback(
                stage="Generating dataset",
                progress=0.45 + 0.50 * (i / n_samples),
                message=f"Sample {i}/{n_samples}",
            )

    return np.array(all_features), np.array(all_labels)
