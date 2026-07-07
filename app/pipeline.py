"""
QT-2.23 — Pipeline Orchestrator

Chains all science modules (A→D→E→G→H→K→F→L) into a complete
experimental pipeline. Each module's output feeds the next.
Called by TaskManager for background execution.

Module execution order per Section 8 of the build manual:
1. A,B,C — Sensor simulations (parallelizable)
2. D — Feature extraction (combines all sensor outputs)
3. Train per-sensor classifiers
4. E — CRLB initial weights
5. G — QUBO feature selection
6. H — Fusion optimization
7. K — Evaluation (CV, bootstrap CI)
8. F — Baseline comparison
9. L — Contribution analysis
"""

from __future__ import annotations
import sys
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.state import (
    ExperimentState, ModuleStatus,
    RadarResult, ThermalResult, AcousticResult,
    FeatureSelectionResult, FusionResult, MetricsResult,
    CRLBResult, ContributionResult, DegradationResult, ScalingResult,
    ConfidenceInterval,
)
from app.events import EventType, Event, event_bus


def run_full_pipeline(
    progress_callback: Callable,
    experiment: ExperimentState,
) -> ExperimentState:
    """
    Execute the complete scientific pipeline.

    This is the function submitted to TaskManager for a full experiment run.
    Each stage updates the experiment state in place and emits events.
    """
    exp = experiment
    exp_id = exp.experiment_id
    seed = exp.seed

    total_stages = 10
    stage_num = 0

    def stage_progress(stage_name, fraction=0.0, message=""):
        nonlocal stage_num
        overall = (stage_num + fraction) / total_stages
        progress_callback(stage=stage_name, progress=overall, message=message)

    # ========================================================
    # STAGE 1: Radar Simulation (Module A)
    # ========================================================
    stage_progress("GENERATING SCENARIOS", 0.0, "Radar simulation")
    stage_num = 0

    from science.radar.module_a import run_radar_simulation

    def radar_progress(stage="", progress=0.0, message=""):
        stage_progress("GENERATING SCENARIOS", progress * 0.33, f"[Radar] {message}")

    radar_data = run_radar_simulation(radar_progress, exp.radar_config, seed)

    exp.radar_result.range_profile = radar_data["range_profile"]
    exp.radar_result.range_doppler_map = radar_data["range_doppler_map"]
    exp.radar_result.clutter_map = radar_data["clutter_map"]
    exp.radar_result.cfar_threshold = radar_data["cfar_threshold"]
    exp.radar_result.detections = radar_data["detections"]
    exp.radar_result.snr_db = radar_data["snr_db"]
    exp.radar_result.configured_pfa = radar_data["configured_pfa"]
    exp.radar_result.empirical_pfa = radar_data["empirical_pfa"]
    exp.radar_result.cfar_alpha = radar_data["cfar_alpha"]
    exp.radar_result.features = radar_data["features"]
    exp.radar_result.feature_names = radar_data["feature_names"]
    exp.radar_result.feature_values = radar_data["feature_values"]
    exp.radar_result.metadata = radar_data["metadata"]
    exp.radar_result.status = ModuleStatus.COMPLETED

    event_bus.publish(Event(
        event_type=EventType.RADAR_RESULT_UPDATED,
        source="RADAR", message=f"SNR={radar_data['snr_db']:.1f}dB",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # STAGE 2: Thermal Simulation (Module B)
    # ========================================================
    stage_num = 1
    stage_progress("GENERATING SCENARIOS", 0.0, "Thermal simulation")

    from science.thermal.module_b import run_thermal_simulation

    def thermal_progress(stage="", progress=0.0, message=""):
        stage_progress("GENERATING SCENARIOS", 0.33 + progress * 0.33, f"[Thermal] {message}")

    thermal_data = run_thermal_simulation(thermal_progress, exp.thermal_config, seed)

    exp.thermal_result.thermal_frame = thermal_data["thermal_frame"]
    exp.thermal_result.contrast_map = thermal_data["contrast_map"]
    exp.thermal_result.noise_field = thermal_data["noise_field"]
    exp.thermal_result.noise_psd_freqs = thermal_data["noise_psd_freqs"]
    exp.thermal_result.noise_psd_power = thermal_data["noise_psd_power"]
    exp.thermal_result.noise_psd_slope = thermal_data["noise_psd_slope"]
    exp.thermal_result.target_exitance = thermal_data["target_exitance"]
    exp.thermal_result.background_exitance = thermal_data["background_exitance"]
    exp.thermal_result.delta_M = thermal_data["delta_M"]
    exp.thermal_result.thermal_snr = thermal_data["thermal_snr"]
    exp.thermal_result.features = thermal_data["features"]
    exp.thermal_result.feature_names = thermal_data["feature_names"]
    exp.thermal_result.feature_values = thermal_data["feature_values"]
    exp.thermal_result.metadata = thermal_data["metadata"]
    exp.thermal_result.status = ModuleStatus.COMPLETED

    event_bus.publish(Event(
        event_type=EventType.THERMAL_RESULT_UPDATED,
        source="THERMAL", message=f"ΔM={thermal_data['delta_M']:.2f} W/m²",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # STAGE 3: Acoustic Simulation (Module C)
    # ========================================================
    stage_num = 2
    stage_progress("GENERATING SCENARIOS", 0.0, "Acoustic simulation")

    from science.acoustic.module_c import run_acoustic_simulation

    def acoustic_progress(stage="", progress=0.0, message=""):
        stage_progress("GENERATING SCENARIOS", 0.66 + progress * 0.34, f"[Acoustic] {message}")

    acoustic_data = run_acoustic_simulation(acoustic_progress, exp.acoustic_config, seed)

    exp.acoustic_result.time_series = acoustic_data["time_series"]
    exp.acoustic_result.sample_rate = acoustic_data["sample_rate"]
    exp.acoustic_result.lofar_image = acoustic_data["lofar_image"]
    exp.acoustic_result.lofar_frequencies = acoustic_data["lofar_frequencies"]
    exp.acoustic_result.lofar_times = acoustic_data["lofar_times"]
    exp.acoustic_result.signal_excess_db = acoustic_data["signal_excess_db"]
    exp.acoustic_result.transmission_loss_db = acoustic_data["transmission_loss_db"]
    exp.acoustic_result.noise_level_db = acoustic_data["noise_level_db"]
    exp.acoustic_result.features = acoustic_data["features"]
    exp.acoustic_result.feature_names = acoustic_data["feature_names"]
    exp.acoustic_result.feature_values = acoustic_data["feature_values"]
    exp.acoustic_result.metadata = acoustic_data["metadata"]
    exp.acoustic_result.status = ModuleStatus.COMPLETED

    event_bus.publish(Event(
        event_type=EventType.ACOUSTIC_RESULT_UPDATED,
        source="ACOUSTIC", message=f"SE={acoustic_data['signal_excess_db']:.1f}dB",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # STAGE 4: Feature Extraction (Module D)
    # ========================================================
    stage_num = 3
    stage_progress("EXTRACTING FEATURES", 0.0, "Building feature matrix")

    from science.features.module_d import build_feature_matrix

    def feature_progress(stage="", progress=0.0, message=""):
        stage_progress("EXTRACTING FEATURES", progress, f"[Features] {message}")

    feature_matrix, feature_names = build_feature_matrix(
        radar_data["dataset_features"],
        thermal_data["dataset_features"],
        acoustic_data["dataset_features"],
        feature_progress,
    )

    # Labels should match across datasets (all generated with same structure)
    labels = radar_data["dataset_labels"]  # Same split: first half target, second half noise

    exp.feature_matrix = feature_matrix
    exp.feature_names = feature_names
    exp.labels = labels

    event_bus.publish(Event(
        event_type=EventType.FEATURE_EXTRACTION_COMPLETED,
        source="FEATURES",
        message=f"{feature_matrix.shape[1]} features × {feature_matrix.shape[0]} samples",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # STAGE 5: Train per-sensor classifiers
    # ========================================================
    stage_num = 4
    stage_progress("TRAINING SENSOR DETECTORS", 0.0, "Training per-sensor LR models")

    n_radar = len(radar_data["feature_names"])
    n_thermal = len(thermal_data["feature_names"])
    n_acoustic = len(acoustic_data["feature_names"])

    sensor_classifiers = {}
    sensor_scores = {}

    # Radar detector
    clf_radar = LogisticRegression(max_iter=1000, random_state=seed)
    clf_radar.fit(feature_matrix[:, :n_radar], labels)
    sensor_scores["radar"] = clf_radar.predict_proba(feature_matrix[:, :n_radar])[:, 1]
    sensor_classifiers["radar"] = clf_radar

    stage_progress("TRAINING SENSOR DETECTORS", 0.33, "Radar detector trained")

    # Thermal detector
    r_end = n_radar + n_thermal
    clf_thermal = LogisticRegression(max_iter=1000, random_state=seed)
    clf_thermal.fit(feature_matrix[:, n_radar:r_end], labels)
    sensor_scores["thermal"] = clf_thermal.predict_proba(feature_matrix[:, n_radar:r_end])[:, 1]
    sensor_classifiers["thermal"] = clf_thermal

    stage_progress("TRAINING SENSOR DETECTORS", 0.66, "Thermal detector trained")

    # Acoustic detector
    a_end = r_end + n_acoustic
    clf_acoustic = LogisticRegression(max_iter=1000, random_state=seed)
    clf_acoustic.fit(feature_matrix[:, r_end:a_end], labels)
    sensor_scores["acoustic"] = clf_acoustic.predict_proba(feature_matrix[:, r_end:a_end])[:, 1]
    sensor_classifiers["acoustic"] = clf_acoustic

    stage_progress("TRAINING SENSOR DETECTORS", 1.0, "All 3 sensor detectors trained")

    exp.sensor_classifiers = sensor_classifiers
    exp.sensor_scores = sensor_scores

    event_bus.publish(Event(
        event_type=EventType.SENSOR_MODELS_TRAINED,
        source="TRAINING",
        message=f"3 per-sensor detectors trained",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # STAGE 6: CRLB Initial Weights (Module E)
    # ========================================================
    stage_num = 5
    stage_progress("COMPUTING CRLB", 0.0, "Fisher Information analysis")

    from science.estimation.module_e import run_crlb_analysis

    def crlb_progress(stage="", progress=0.0, message=""):
        stage_progress("COMPUTING CRLB", progress, message)

    crlb_data = run_crlb_analysis(
        crlb_progress,
        radar_data["snr_db"],
        thermal_data["thermal_snr"],
        acoustic_data["signal_excess_db"],
        seed,
    )

    exp.crlb_result.theoretical_crlb = crlb_data["theoretical_crlb"]
    exp.crlb_result.empirical_variance = crlb_data["empirical_variance"]
    exp.crlb_result.gap_ratios = crlb_data["gap_ratios"]
    exp.crlb_result.initial_weights = crlb_data["initial_weights"]
    exp.crlb_result.status = ModuleStatus.COMPLETED

    # ========================================================
    # STAGE 7: QUBO Feature Selection (Module G)
    # ========================================================
    stage_num = 6
    stage_progress("BUILDING QUBO", 0.0, "MID-mRMR feature selection")

    from science.qubo.module_g import run_feature_selection

    def qubo_progress(stage="", progress=0.0, message=""):
        stage_progress("BUILDING QUBO", progress, message)

    fs_data = run_feature_selection(
        qubo_progress,
        feature_matrix, labels, feature_names,
        k_target=exp.feature_config.k_target,
        alpha=exp.feature_config.alpha,
        beta=exp.feature_config.beta,
        gamma=exp.feature_config.gamma,
        num_reads=exp.feature_config.num_reads,
        run_brute_force=exp.feature_config.run_brute_force,
    )

    exp.feature_result.feature_names = fs_data["feature_names"]
    exp.feature_result.relevance = fs_data["relevance"]
    exp.feature_result.redundancy_matrix = fs_data["redundancy_matrix"]
    exp.feature_result.Q_matrix = fs_data["Q_matrix"]
    exp.feature_result.selected_indices = fs_data["selected_indices"]
    exp.feature_result.selected_features = fs_data["selected_features"]
    exp.feature_result.objective_value = fs_data["objective_value"]
    exp.feature_result.brute_force_objective = fs_data["brute_force_objective"]
    exp.feature_result.brute_force_selected = fs_data["brute_force_selected"]
    exp.feature_result.subset_match = fs_data["subset_match"]
    exp.feature_result.solver = fs_data["solver"]
    exp.feature_result.solver_metadata = fs_data["solver_metadata"]
    exp.feature_result.status = ModuleStatus.COMPLETED

    event_bus.publish(Event(
        event_type=EventType.FEATURE_SELECTION_RESULT_UPDATED,
        source="FEATURE-QUBO",
        message=f"Selected {len(fs_data['selected_features'])}/{len(feature_names)}, match={fs_data['subset_match']}",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # STAGE 8: Fusion Optimization (Module H)
    # ========================================================
    stage_num = 7
    stage_progress("OPTIMIZING FUSION", 0.0, "Rayleigh-LDA weight optimization")

    from science.fusion.module_h import run_fusion_optimization

    def fusion_progress(stage="", progress=0.0, message=""):
        stage_progress("OPTIMIZING FUSION", progress, message)

    fusion_data = run_fusion_optimization(
        fusion_progress,
        sensor_scores, labels,
        lam=exp.fusion_config.lam,
        n_restarts=exp.fusion_config.n_restarts,
        seed=seed,
        bits_per_weight=exp.fusion_config.bits_per_weight,
        simplex_penalty=exp.fusion_config.simplex_penalty,
        target_far=exp.fusion_config.target_far,
        run_binary=True,
    )

    exp.fusion_result.weights = fusion_data["weights"]
    exp.fusion_result.weights_array = fusion_data["weights_array"]
    exp.fusion_result.fused_scores = fusion_data["fused_scores"]
    exp.fusion_result.threshold = fusion_data["threshold"]
    exp.fusion_result.fisher_objective = fusion_data["fisher_objective"]
    exp.fusion_result.S_b = fusion_data["S_b"]
    exp.fusion_result.S_w = fusion_data["S_w"]
    exp.fusion_result.solver = fusion_data["solver"]
    exp.fusion_result.solve_time_s = fusion_data["solve_time_s"]
    exp.fusion_result.optimization_mode = fusion_data["optimization_mode"]
    exp.fusion_result.sensor_scores_h0 = fusion_data["sensor_scores_h0"]
    exp.fusion_result.sensor_scores_h1 = fusion_data["sensor_scores_h1"]
    exp.fusion_result.fused_scores_h0 = fusion_data["fused_scores_h0"]
    exp.fusion_result.fused_scores_h1 = fusion_data["fused_scores_h1"]
    exp.fusion_result.status = ModuleStatus.COMPLETED

    event_bus.publish(Event(
        event_type=EventType.FUSION_RESULT_UPDATED,
        source="FUSION",
        message=f"J'={fusion_data['fisher_objective']:.4f}, w={fusion_data['weights']}",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # STAGE 9: Evaluation (Module K)
    # ========================================================
    stage_num = 8
    stage_progress("EVALUATING", 0.0, "Bootstrap CI + full metrics")

    from science.evaluation.module_k import compute_full_metrics

    def eval_progress(stage="", progress=0.0, message=""):
        stage_progress("EVALUATING", progress, message)

    metrics_data = compute_full_metrics(
        labels,
        fusion_data["fused_scores"],
        fusion_data["threshold"],
        n_bootstrap=exp.evaluation_config.n_bootstrap,
        ci_level=exp.evaluation_config.ci_level * 100,
        seed=seed,
        progress_callback=eval_progress,
    )

    exp.metrics_result.auc = ConfidenceInterval(
        point_estimate=metrics_data["auc"]["mean"],
        ci_lower=metrics_data["auc"]["ci_lower"],
        ci_upper=metrics_data["auc"]["ci_upper"],
        n_bootstrap=exp.evaluation_config.n_bootstrap,
    )
    exp.metrics_result.detection_rate = ConfidenceInterval(
        point_estimate=metrics_data["detection_rate"],
        ci_lower=metrics_data["detection_rate"],
        ci_upper=metrics_data["detection_rate"],
    )
    exp.metrics_result.false_alarm_rate = ConfidenceInterval(
        point_estimate=metrics_data["false_alarm_rate"],
        ci_lower=metrics_data["false_alarm_rate"],
        ci_upper=metrics_data["false_alarm_rate"],
    )
    exp.metrics_result.precision = ConfidenceInterval(
        point_estimate=metrics_data["precision"],
        ci_lower=metrics_data["precision"],
        ci_upper=metrics_data["precision"],
    )
    exp.metrics_result.recall = ConfidenceInterval(
        point_estimate=metrics_data["recall"],
        ci_lower=metrics_data["recall"],
        ci_upper=metrics_data["recall"],
    )
    exp.metrics_result.f1 = ConfidenceInterval(
        point_estimate=metrics_data["f1"],
        ci_lower=metrics_data["f1"],
        ci_upper=metrics_data["f1"],
    )
    exp.metrics_result.roc_fpr = metrics_data["roc_fpr"]
    exp.metrics_result.roc_tpr = metrics_data["roc_tpr"]
    exp.metrics_result.confusion_matrix = metrics_data["confusion_matrix"]
    exp.metrics_result.status = ModuleStatus.COMPLETED

    event_bus.publish(Event(
        event_type=EventType.METRICS_RESULT_UPDATED,
        source="EVALUATION",
        message=f"AUC={exp.metrics_result.auc}",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # STAGE 10: Baseline Comparison (Module F)
    # ========================================================
    stage_num = 9
    stage_progress("COMPARING BASELINES", 0.0, "Running all 5 classical baselines")

    from science.baselines.module_f import run_all_baselines

    def baseline_progress(stage="", progress=0.0, message=""):
        stage_progress("COMPARING BASELINES", progress, message)

    from app.state import BaselineResult

    baseline_data = run_all_baselines(
        baseline_progress,
        sensor_scores, labels,
        fusion_data["S_b"], fusion_data["S_w"],
        crlb_data["initial_weights"],
        lam=exp.fusion_config.lam,
        n_restarts=exp.fusion_config.n_restarts,
        seed=seed,
    )

    exp.baseline_results = []
    for bd in baseline_data:
        br = BaselineResult(
            method_name=bd["method_name"],
            weights=bd["weights"],
            objective_value=bd.get("objective_value"),
            solve_time_s=bd["solve_time_s"],
            fused_scores=bd["fused_scores"],
        )
        exp.baseline_results.append(br)

    event_bus.publish(Event(
        event_type=EventType.BASELINE_RESULT_UPDATED,
        source="BASELINES",
        message=f"{len(baseline_data)} baselines evaluated",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # COMPLETED
    # ========================================================
    exp.status = ModuleStatus.COMPLETED

    stage_progress("COMPLETED", 1.0, f"Experiment {exp_id} complete")

    return exp
