"""
QT-2.23 — Module C: Acoustic/Sonar Sensor Simulation

Physics-grounded passive sonar simulation with:
- Passive sonar equation: SE = SL - TL - (NL - DI) - DT
- Spherical spreading + absorption transmission loss
- Knudsen/Wenz ambient noise model
- LOFAR spectrogram generation via STFT
- Tonal line analysis
- Feature extraction for downstream fusion

Reference: Sonar equation standard (Urick, Principles of Underwater Sound).
SL is the stealth knob: documented quieting technology produces 30-50 dB
SL reduction from conventional to modern quiet vessels (open-literature fact).
All parameters from Appendix B of the build manual.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.signal import stft, get_window
from scipy.stats import kurtosis


# ============================================================
# C.1 — Passive Sonar Equation
# ============================================================

def transmission_loss(range_m: float, absorption_coef: float = 0.005) -> float:
    """
    Spherical spreading + absorption transmission loss.
    TL = 20·log10(R) + α·R [dB]

    Valid for short-to-moderate range. For longer ranges, see
    transmission_loss_extended (cylindrical spreading beyond 1 km).
    """
    return 20.0 * np.log10(max(range_m, 1.0)) + absorption_coef * range_m


def transmission_loss_extended(range_m: float, absorption_coef: float = 0.005) -> float:
    """
    Extended TL model: spherical near-field, cylindrical beyond 1 km.
    More realistic for longer-range scenarios.
    """
    if range_m < 1000.0:
        return 20.0 * np.log10(max(range_m, 1.0)) + absorption_coef * range_m
    return (
        20.0 * np.log10(1000.0)
        + 10.0 * np.log10(range_m / 1000.0)
        + absorption_coef * range_m
    )


def knudsen_noise_level(freq_hz: float, sea_state: int = 3) -> float:
    """
    Knudsen/Wenz ambient noise model.

    Wind/sea-state-dependent baseline with standard -17 dB/decade
    spectral falloff above ~1 kHz.

    Returns noise level in dB re 1 µPa.
    """
    base_level = 50.0 + 5.0 * sea_state
    return base_level - 17.0 * np.log10(max(freq_hz, 1.0) / 1000.0)


def signal_excess(
    SL_dB: float,
    range_m: float,
    NL_dB: float,
    DI_dB: float,
    DT_dB: float = 0.0,
    absorption_coef: float = 0.005,
) -> float:
    """
    Signal Excess = SL - TL - (NL - DI) - DT [dB].
    Detection when SE ≥ 0.
    """
    TL = transmission_loss(range_m, absorption_coef)
    return SL_dB - TL - (NL_dB - DI_dB) - DT_dB


# ============================================================
# C.2 — LOFAR Spectrogram Generation
# ============================================================

def generate_lofar_spectrogram(
    signal_excess_db: float,
    duration_s: float = 60.0,
    fs: float = 2000.0,
    tonal_freqs_hz: Tuple[float, ...] = (50.0, 120.0, 240.0),
    nperseg: int = 1024,
    noverlap: int = 512,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Generate a LOFAR (Low-Frequency Analysis and Recording) spectrogram.

    Real passive sonar processes raw hydrophone time series into LOFAR
    via STFT, then hunts for narrowband tonal lines against broadband
    ambient noise.

    Parameters
    ----------
    signal_excess_db : float - Signal excess in dB (can be negative)
    duration_s : float - Duration of recording in seconds
    fs : float - Sampling frequency in Hz
    tonal_freqs_hz : tuple - Target tonal frequencies
    nperseg : int - STFT window length
    noverlap : int - STFT overlap

    Returns
    -------
    dict with time series, LOFAR image, frequencies, times, and metadata
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = int(duration_s * fs)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    # Ambient noise (broadband)
    noise = rng.normal(0, 1, n_samples)

    # Target tonal components
    amplitude = 10.0 ** (signal_excess_db / 20.0)
    target = np.zeros(n_samples)
    for freq in tonal_freqs_hz:
        # Each tonal has slight frequency jitter (realistic)
        phase = rng.uniform(0, 2 * np.pi)
        target += amplitude * np.sin(2 * np.pi * freq * t + phase)

    raw_audio = noise + target

    # STFT → LOFAR spectrogram
    window = get_window('hann', nperseg)
    f, t_spec, Zxx = stft(raw_audio, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)

    # Convert to dB power
    lofar_image = 10.0 * np.log10(np.abs(Zxx)**2 + 1e-12)

    return {
        "time_series": raw_audio,
        "sample_rate": int(fs),
        "lofar_image": lofar_image,
        "frequencies": f,
        "times": t_spec,
        "signal_excess_db": signal_excess_db,
        "tonal_freqs_hz": list(tonal_freqs_hz),
    }


# ============================================================
# C.3 — Tonal Analysis
# ============================================================

def analyze_tonals(
    lofar_image: np.ndarray,
    frequencies: np.ndarray,
    expected_tonals: Tuple[float, ...] = (50.0, 120.0, 240.0),
    bandwidth_hz: float = 10.0,
) -> Dict[str, Any]:
    """
    Analyze tonal lines in the LOFAR spectrogram.

    For each expected tonal frequency, compute the mean power in a
    narrow band around it and compare to the broadband noise floor.
    """
    # Mean spectrum across time
    mean_spectrum = np.mean(lofar_image, axis=1)

    tonal_results = {}
    for freq in expected_tonals:
        freq_mask = (frequencies >= freq - bandwidth_hz / 2) & (frequencies <= freq + bandwidth_hz / 2)
        if np.any(freq_mask):
            tonal_power = float(np.mean(mean_spectrum[freq_mask]))
        else:
            tonal_power = float(np.min(mean_spectrum))

        # Broadband noise: everything outside tonal bands
        noise_mask = np.ones(len(frequencies), dtype=bool)
        for ef in expected_tonals:
            noise_mask &= ~((frequencies >= ef - bandwidth_hz) & (frequencies <= ef + bandwidth_hz))
        broadband_power = float(np.mean(mean_spectrum[noise_mask])) if np.any(noise_mask) else -60.0

        tonal_results[f"{freq:.0f}Hz"] = {
            "tonal_power_dB": tonal_power,
            "broadband_power_dB": broadband_power,
            "excess_dB": tonal_power - broadband_power,
        }

    return tonal_results


# ============================================================
# C.4 — Feature Extraction (Module D integration for acoustic)
# ============================================================

def extract_acoustic_features(
    lofar_image: np.ndarray,
    frequencies: np.ndarray,
    tonal_band: Tuple[float, float] = (40.0, 150.0),
    fs: float = 2000.0,
) -> Dict[str, float]:
    """
    Extract interpretable acoustic features from LOFAR spectrogram.
    """
    band_mask = (frequencies > tonal_band[0]) & (frequencies < tonal_band[1])
    tonal_power = float(np.sum(lofar_image[band_mask, :])) if np.any(band_mask) else 0.0
    broadband_power = float(np.sum(lofar_image[~band_mask, :])) if np.any(~band_mask) else 1.0

    spectral_kurt = float(kurtosis(lofar_image.flatten()))
    tonal_to_broadband = tonal_power / (abs(broadband_power) + 1e-12)

    # Peak frequency bin (across time-averaged spectrum)
    mean_spectrum = np.mean(lofar_image, axis=1)
    peak_freq_idx = np.argmax(mean_spectrum)
    peak_freq = float(frequencies[peak_freq_idx]) if len(frequencies) > peak_freq_idx else 0.0

    # Temporal variability of tonal band
    if np.any(band_mask) and lofar_image.shape[1] > 1:
        tonal_temporal = lofar_image[band_mask, :]
        temporal_std = float(np.mean(np.std(tonal_temporal, axis=1)))
    else:
        temporal_std = 0.0

    return {
        "acoustic_tonal_power": tonal_power,
        "acoustic_broadband_power": broadband_power,
        "acoustic_spectral_kurtosis": spectral_kurt,
        "acoustic_tonal_to_broadband": tonal_to_broadband,
        "acoustic_peak_frequency": peak_freq,
        "acoustic_temporal_variability": temporal_std,
    }


# ============================================================
# C.5 — Full Acoustic Simulation Pipeline
# ============================================================

def run_acoustic_simulation(
    progress_callback: Callable,
    config: Any,  # AcousticConfig dataclass
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full Module C pipeline. Called by TaskManager.
    """
    rng = np.random.default_rng(seed)

    # --- Stage 1: Compute signal excess ---
    progress_callback(
        stage="Computing signal excess",
        progress=0.05,
        message=f"SL={config.source_level_dB} dB, R={config.range_m} m",
    )

    NL = knudsen_noise_level(config.tonal_freqs_hz[0], config.sea_state)
    TL = transmission_loss(config.range_m, config.absorption_coef)
    SE = signal_excess(
        config.source_level_dB,
        config.range_m,
        NL,
        config.directivity_index_dB,
        config.detection_threshold_dB,
        config.absorption_coef,
    )

    # --- Stage 2: Generate LOFAR spectrogram ---
    progress_callback(
        stage="Generating LOFAR spectrogram",
        progress=0.15,
        message=f"SE={SE:.1f} dB, duration={config.duration_s}s",
    )

    lofar_data = generate_lofar_spectrogram(
        signal_excess_db=SE,
        duration_s=config.duration_s,
        fs=config.sampling_freq_hz,
        tonal_freqs_hz=config.tonal_freqs_hz,
        nperseg=config.stft_nperseg,
        noverlap=config.stft_noverlap,
        rng=rng,
    )

    # --- Stage 3: Tonal analysis ---
    progress_callback(
        stage="Analyzing tonals",
        progress=0.30,
        message=f"Tonals at {config.tonal_freqs_hz} Hz",
    )

    tonal_analysis = analyze_tonals(
        lofar_data["lofar_image"],
        lofar_data["frequencies"],
        config.tonal_freqs_hz,
    )

    # --- Stage 4: Extract features ---
    progress_callback(
        stage="Extracting features",
        progress=0.40,
        message="Acoustic features",
    )

    features = extract_acoustic_features(
        lofar_data["lofar_image"],
        lofar_data["frequencies"],
        fs=config.sampling_freq_hz,
    )

    # --- Stage 5: Generate full dataset ---
    progress_callback(
        stage="Generating dataset",
        progress=0.45,
        message=f"N={config.num_samples} samples",
    )

    dataset_features, dataset_labels = _generate_acoustic_dataset(
        config, rng, progress_callback
    )

    progress_callback(
        stage="Completed",
        progress=1.0,
        message=f"SE={SE:.1f}dB, TL={TL:.1f}dB, NL={NL:.1f}dB",
    )

    return {
        "time_series": lofar_data["time_series"],
        "sample_rate": lofar_data["sample_rate"],
        "lofar_image": lofar_data["lofar_image"],
        "lofar_frequencies": lofar_data["frequencies"],
        "lofar_times": lofar_data["times"],
        "signal_excess_db": SE,
        "transmission_loss_db": TL,
        "noise_level_db": NL,
        "tonal_analysis": tonal_analysis,
        "features": features,
        "feature_names": list(features.keys()),
        "feature_values": np.array(list(features.values())),
        "dataset_features": dataset_features,
        "dataset_labels": dataset_labels,
        "metadata": {
            "source_level_dB": config.source_level_dB,
            "range_m": config.range_m,
            "sea_state": config.sea_state,
            "absorption_coef": config.absorption_coef,
            "tonal_freqs_hz": list(config.tonal_freqs_hz),
            "seed": seed,
        },
    }


def _generate_acoustic_dataset(
    config: Any,
    rng: np.random.Generator,
    progress_callback: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a labelled dataset of acoustic feature vectors."""
    n_samples = config.num_samples
    n_per_class = n_samples // 2

    all_features = []
    all_labels = []

    NL = knudsen_noise_level(config.tonal_freqs_hz[0], config.sea_state)

    for i in range(n_samples):
        label = 1 if i < n_per_class else 0

        if label == 1:
            SL = config.source_level_dB
        else:
            # Background only: very low SL so no tonal components
            SL = NL - 20.0  # Effectively no target signal

        SE = signal_excess(
            SL, config.range_m, NL,
            config.directivity_index_dB,
            config.detection_threshold_dB,
            config.absorption_coef,
        )

        # Short LOFAR for dataset generation (reduce duration for speed)
        lofar_data = generate_lofar_spectrogram(
            signal_excess_db=SE,
            duration_s=5.0,  # Short clips for dataset efficiency
            fs=config.sampling_freq_hz,
            tonal_freqs_hz=config.tonal_freqs_hz,
            nperseg=config.stft_nperseg,
            noverlap=config.stft_noverlap,
            rng=rng,
        )

        feats = extract_acoustic_features(
            lofar_data["lofar_image"],
            lofar_data["frequencies"],
            fs=config.sampling_freq_hz,
        )
        all_features.append(list(feats.values()))
        all_labels.append(label)

        if i % 200 == 0 and progress_callback:
            progress_callback(
                stage="Generating dataset",
                progress=0.45 + 0.50 * (i / n_samples),
                message=f"Sample {i}/{n_samples}",
            )

    return np.array(all_features), np.array(all_labels)
