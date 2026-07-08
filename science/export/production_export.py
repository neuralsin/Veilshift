"""
QT-2.23 — Production Export Module

Generates a finalized, serialized model artifact that captures:
1. The selected features (from QUBO feature selection)
2. The trained per-sensor classifiers
3. The optimized fusion weights (from Rayleigh-LDA)
4. The locked Neyman-Pearson decision threshold
5. Full OOF evaluation metrics with bootstrap CI
6. All sensor configurations used for generation

This module produces a JSON manifest (human-readable, auditable)
and an optional .pkl file for the classifier objects.

The exported model can be loaded in milliseconds by a deployment
system without re-running the full generation/training pipeline.
"""

from __future__ import annotations
import json
import os
import time
import pickle
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import asdict

import numpy as np

from app.state import (
    RadarConfig, ThermalConfig, AcousticConfig,
    TargetRegime,
)
from science.radar.module_a import run_radar_simulation
from science.thermal.module_b import run_thermal_simulation
from science.acoustic.module_c import run_acoustic_simulation
from science.evaluation.oof_protocol import run_oof_evaluation


# ============================================================
# JSON Encoder for numpy types
# ============================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ============================================================
# Production Export Pipeline
# ============================================================

def export_production_model(
    output_dir: str = "production_export",
    regime: str = "STEALTH",
    seed: int = 42,
    num_samples: int = 6000,
    n_folds: int = 5,
    n_bootstrap: int = 1000,
    k_target: int = 6,
    target_far: float = 0.01,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Full production export pipeline.

    1. Configure sensor parameters based on regime
    2. Generate full synthetic dataset
    3. Run complete OOF evaluation
    4. Extract and lock all model parameters
    5. Serialize to JSON manifest + pkl

    Parameters
    ----------
    output_dir : directory to write export files
    regime : "STEALTH" or "CONVENTIONAL"
    seed : random seed for reproducibility
    num_samples : samples per sensor generator
    n_folds : OOF cross-validation folds
    n_bootstrap : bootstrap replicates for CI
    k_target : number of features to select
    target_far : Neyman-Pearson false alarm rate constraint
    progress_callback : optional (stage, progress, message) reporter

    Returns
    -------
    Dict containing the full manifest data
    """
    def cb(stage="", progress=0.0, message=""):
        if progress_callback:
            progress_callback(stage=stage, progress=progress, message=message)
        else:
            # Safe print for Windows console (cp1252 can't handle σ, etc.)
            safe_msg = f"[EXPORT] {stage} {progress*100:.0f}% {message}"
            print(safe_msg.encode("ascii", errors="replace").decode("ascii"))

    start_time = time.time()
    timestamp = datetime.now(timezone.utc).isoformat()

    # ---- Step 1: Configure regime ----
    cb(stage="Configuring regime", progress=0.0, message=f"Regime: {regime}")

    if regime.upper() == "CONVENTIONAL":
        radar_cfg = RadarConfig(
            rcs_m2=10.0, range_m=10000.0,
            target_regime=TargetRegime.CONVENTIONAL,
            num_samples=num_samples,
        )
        thermal_cfg = ThermalConfig(
            target_temperature_K=400.0, target_emissivity=0.9,
            target_regime=TargetRegime.CONVENTIONAL,
            num_samples=num_samples,
        )
        acoustic_cfg = AcousticConfig(
            source_level_dB=160.0, range_m=500.0,
            target_regime=TargetRegime.CONVENTIONAL,
            num_samples=num_samples,
        )
    else:
        # STEALTH defaults (from Appendix B, with corrected emissivity)
        radar_cfg = RadarConfig(num_samples=num_samples)
        thermal_cfg = ThermalConfig(num_samples=num_samples)
        acoustic_cfg = AcousticConfig(num_samples=num_samples)

    # ---- Step 2: Generate full dataset ----
    cb(stage="Generating radar data", progress=0.05, message=f"N={num_samples}")
    radar_data = run_radar_simulation(cb, radar_cfg, seed=seed)

    cb(stage="Generating thermal data", progress=0.15, message=f"N={num_samples}")
    thermal_data = run_thermal_simulation(cb, thermal_cfg, seed=seed)

    cb(stage="Generating acoustic data", progress=0.25, message=f"N={num_samples}")
    acoustic_data = run_acoustic_simulation(cb, acoustic_cfg, seed=seed)

    # Assemble feature matrix
    feature_names = (
        radar_data["feature_names"]
        + thermal_data["feature_names"]
        + acoustic_data["feature_names"]
    )
    X = np.hstack([
        radar_data["dataset_features"],
        thermal_data["dataset_features"],
        acoustic_data["dataset_features"],
    ])
    y = radar_data["dataset_labels"]

    sensor_feature_counts = {
        "Radar": len(radar_data["feature_names"]),
        "Thermal": len(thermal_data["feature_names"]),
        "Acoustic": len(acoustic_data["feature_names"]),
    }

    cb(stage="Dataset assembled", progress=0.30,
       message=f"X: {X.shape}, features: {len(feature_names)}")

    # ---- Step 3: Run full OOF evaluation ----
    cb(stage="Running OOF evaluation", progress=0.35,
       message=f"{n_folds}-fold stratified, {n_bootstrap} bootstrap")

    oof_result = run_oof_evaluation(
        X, y, feature_names, sensor_feature_counts,
        n_splits=n_folds, seed=seed, n_bootstrap=n_bootstrap,
        k_target=k_target, target_far=target_far,
        progress_callback=cb,
    )

    # ---- Step 4: Extract locked parameters ----
    cb(stage="Extracting model parameters", progress=0.85, message="Locking weights and threshold")

    # Consensus features (appeared in >= 60% of folds)
    consensus_features = oof_result.consensus_features or []

    # Mean fusion weights across folds
    mean_weights = oof_result.mean_fusion_weights or {}
    std_weights = oof_result.std_fusion_weights or {}

    # Median threshold across folds
    fold_thresholds = oof_result.fold_thresholds or []
    locked_threshold = float(np.median(fold_thresholds)) if fold_thresholds else 0.5

    # Per-fold details for transparency
    fold_details = []
    for fr in oof_result.fold_results:
        fold_details.append({
            "fold_idx": fr.fold_idx,
            "selected_features": fr.selected_feature_names,
            "fusion_weights": fr.fusion_weights,
            "threshold": fr.threshold,
            "fisher_objective": fr.fisher_objective,
            "auc": fr.fold_metrics.auc if fr.fold_metrics else None,
            "detection_rate": fr.fold_metrics.detection_rate if fr.fold_metrics else None,
            "false_alarm_rate": fr.fold_metrics.false_alarm_rate if fr.fold_metrics else None,
        })

    # ---- Step 5: Build manifest ----
    cb(stage="Building manifest", progress=0.90, message="Serializing")

    manifest = {
        "version": "QT-2.23-PRODUCTION",
        "timestamp": timestamp,
        "regime": regime.upper(),
        "seed": seed,
        "n_samples": num_samples,

        # Locked model parameters
        "selected_features": consensus_features,
        "feature_selection_frequency": oof_result.feature_selection_frequency,
        "fusion_weights": mean_weights,
        "fusion_weights_std": std_weights,
        "decision_threshold": locked_threshold,
        "target_far": target_far,

        # Evaluation metrics
        "oof_auc": oof_result.auc,
        "bootstrap_ci": {
            "mean": oof_result.bootstrap_ci.get("mean") if oof_result.bootstrap_ci else None,
            "ci_lower": oof_result.bootstrap_ci.get("ci_lower") if oof_result.bootstrap_ci else None,
            "ci_upper": oof_result.bootstrap_ci.get("ci_upper") if oof_result.bootstrap_ci else None,
        },
        "aggregate_detection_rate": oof_result.aggregate_metrics.detection_rate if oof_result.aggregate_metrics else None,
        "aggregate_far": oof_result.aggregate_metrics.false_alarm_rate if oof_result.aggregate_metrics else None,

        # Per-fold transparency
        "n_folds": n_folds,
        "fold_details": fold_details,
        "fold_thresholds": fold_thresholds,

        # Sensor configurations (for reproducibility)
        "sensor_configs": {
            "radar": {
                "rcs_m2": radar_cfg.rcs_m2,
                "range_m": radar_cfg.range_m,
                "transmit_power_W": radar_cfg.transmit_power_W,
                "frequency_hz": radar_cfg.frequency_hz,
                "target_regime": radar_cfg.target_regime.value,
            },
            "thermal": {
                "target_temperature_K": thermal_cfg.target_temperature_K,
                "target_emissivity": thermal_cfg.target_emissivity,
                "bg_temperature_K": thermal_cfg.bg_temperature_K,
                "bg_emissivity": thermal_cfg.bg_emissivity,
                "target_regime": thermal_cfg.target_regime.value,
            },
            "acoustic": {
                "source_level_dB": acoustic_cfg.source_level_dB,
                "range_m": acoustic_cfg.range_m,
                "sea_state": acoustic_cfg.sea_state,
                "target_regime": acoustic_cfg.target_regime.value,
            },
        },

        # Feature metadata
        "all_feature_names": feature_names,
        "sensor_feature_counts": sensor_feature_counts,
        "n_total_features": len(feature_names),

        # Timing
        "export_time_s": time.time() - start_time,
    }

    # ---- Step 6: Write to disk ----
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "production_model.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)

    # Save OOF arrays for potential downstream use
    arrays_path = os.path.join(output_dir, "oof_arrays.npz")
    np.savez_compressed(
        arrays_path,
        fused_scores=oof_result.fused_scores,
        true_labels=oof_result.true_labels,
        predictions=oof_result.predictions,
        fold_ids=oof_result.fold_ids,
    )

    cb(stage="Export complete", progress=1.0,
       message=f"Manifest: {manifest_path} | "
               f"AUC: {oof_result.auc:.4f} | "
               f"Threshold: {locked_threshold:.4f} | "
               f"Time: {time.time() - start_time:.1f}s")

    manifest["_paths"] = {
        "manifest": manifest_path,
        "arrays": arrays_path,
    }

    return manifest


def load_production_model(manifest_path: str) -> Dict[str, Any]:
    """
    Load a previously exported production model manifest.

    Returns the full manifest dict with all locked parameters.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Loaded production model: {manifest['version']}")
    print(f"  Regime: {manifest['regime']}")
    print(f"  Seed: {manifest['seed']}")
    print(f"  OOF AUC: {manifest['oof_auc']:.4f}")
    print(f"  Fusion weights: {manifest['fusion_weights']}")
    print(f"  Decision threshold: {manifest['decision_threshold']:.4f}")
    print(f"  Selected features: {manifest['selected_features']}")

    return manifest
