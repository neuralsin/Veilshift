"""
QT-2.23 — Module D: Feature Extraction Pool

Aggregates features from all 3 sensor modalities into a unified feature
matrix for downstream QUBO feature selection (Module G) and fusion (Module H).

~20 interpretable features across radar (6), thermal (5), acoustic (6),
plus cross-modal features (3) = 20 total.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# Canonical feature names — the complete pool
RADAR_FEATURES = [
    "radar_peak_snr",
    "radar_cfar_exceedances",
    "radar_doppler_spread",
    "radar_target_to_clutter",
    "radar_clutter_skewness",
    "radar_clutter_kurtosis",
]

THERMAL_FEATURES = [
    "thermal_peak_contrast",
    "thermal_edge_energy",
    "thermal_hot_spot_area",
    "thermal_contrast_to_noise",
    "thermal_kurtosis",
]

ACOUSTIC_FEATURES = [
    "acoustic_tonal_power",
    "acoustic_broadband_power",
    "acoustic_spectral_kurtosis",
    "acoustic_tonal_to_broadband",
    "acoustic_peak_frequency",
    "acoustic_temporal_variability",
]

CROSS_MODAL_FEATURES = [
    "cross_radar_thermal_agreement",
    "cross_thermal_acoustic_agreement",
    "cross_max_sensor_score_spread",
]

ALL_FEATURE_NAMES = RADAR_FEATURES + THERMAL_FEATURES + ACOUSTIC_FEATURES + CROSS_MODAL_FEATURES

SENSOR_FEATURE_MAP = {
    "radar": RADAR_FEATURES,
    "thermal": THERMAL_FEATURES,
    "acoustic": ACOUSTIC_FEATURES,
    "cross": CROSS_MODAL_FEATURES,
}


def compute_cross_modal_features(
    radar_feats: np.ndarray,
    thermal_feats: np.ndarray,
    acoustic_feats: np.ndarray,
) -> np.ndarray:
    """
    Compute cross-modal features that capture inter-sensor agreement.

    These are interpretable features representing whether multiple sensors
    see similar evidence levels — useful for fusion because high agreement
    across sensors is stronger evidence than any single sensor.
    """
    # Normalize each sensor's features to [0, 1] for comparison
    def norm_max(arr):
        mx = np.max(np.abs(arr))
        return arr / mx if mx > 1e-12 else arr

    r_norm = norm_max(radar_feats)
    t_norm = norm_max(thermal_feats)
    a_norm = norm_max(acoustic_feats)

    # Cross-sensor agreement: correlation between sensor feature magnitudes
    r_mean = float(np.mean(np.abs(r_norm)))
    t_mean = float(np.mean(np.abs(t_norm)))
    a_mean = float(np.mean(np.abs(a_norm)))

    radar_thermal_agreement = 1.0 - abs(r_mean - t_mean)
    thermal_acoustic_agreement = 1.0 - abs(t_mean - a_mean)
    max_spread = max(r_mean, t_mean, a_mean) - min(r_mean, t_mean, a_mean)

    return np.array([radar_thermal_agreement, thermal_acoustic_agreement, max_spread])


def build_feature_matrix(
    radar_dataset: np.ndarray,
    thermal_dataset: np.ndarray,
    acoustic_dataset: np.ndarray,
    progress_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Combine per-sensor feature arrays into a unified feature matrix.

    Parameters
    ----------
    radar_dataset : (N, n_radar_features)
    thermal_dataset : (N, n_thermal_features)
    acoustic_dataset : (N, n_acoustic_features)

    Returns
    -------
    (feature_matrix, feature_names) where feature_matrix is (N, ~20)
    """
    N = min(len(radar_dataset), len(thermal_dataset), len(acoustic_dataset))
    radar_dataset = radar_dataset[:N]
    thermal_dataset = thermal_dataset[:N]
    acoustic_dataset = acoustic_dataset[:N]

    if progress_callback:
        progress_callback(
            stage="Building feature matrix",
            progress=0.1,
            message=f"Combining {N} samples across 3 modalities",
        )

    # Compute cross-modal features for each sample
    cross_features = np.zeros((N, 3))
    for i in range(N):
        cross_features[i] = compute_cross_modal_features(
            radar_dataset[i], thermal_dataset[i], acoustic_dataset[i]
        )
        if progress_callback and i % 1000 == 0:
            progress_callback(
                stage="Cross-modal features",
                progress=0.1 + 0.8 * (i / N),
                message=f"Sample {i}/{N}",
            )

    # Concatenate all features
    feature_matrix = np.hstack([
        radar_dataset, thermal_dataset, acoustic_dataset, cross_features
    ])

    # Build feature names list
    feature_names = (
        RADAR_FEATURES[:radar_dataset.shape[1]]
        + THERMAL_FEATURES[:thermal_dataset.shape[1]]
        + ACOUSTIC_FEATURES[:acoustic_dataset.shape[1]]
        + CROSS_MODAL_FEATURES
    )

    if progress_callback:
        progress_callback(
            stage="Feature matrix complete",
            progress=1.0,
            message=f"{feature_matrix.shape[1]} features × {N} samples",
        )

    return feature_matrix, feature_names


def get_feature_sensor_mapping(feature_names: List[str]) -> Dict[str, str]:
    """Map each feature name to its source sensor for visualization."""
    mapping = {}
    for name in feature_names:
        if name.startswith("radar_"):
            mapping[name] = "radar"
        elif name.startswith("thermal_"):
            mapping[name] = "thermal"
        elif name.startswith("acoustic_"):
            mapping[name] = "acoustic"
        elif name.startswith("cross_"):
            mapping[name] = "cross"
        else:
            mapping[name] = "unknown"
    return mapping
