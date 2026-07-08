"""
QT-2.23 — Pipeline Orchestrator (v2 — OOF Evaluation)

Chains all science modules into a complete experimental pipeline.
See `docs/ARCHITECTURE.md` for a full breakdown of the orchestration stages.
"""

from __future__ import annotations
import sys
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
    Execute the complete scientific pipeline with OOF evaluation.

    Architecture:
      Stages 1-4: Sensor simulation + feature extraction (unchanged physics)
      Stage 5:    OOF evaluation (5-fold stratified, leakage-free)
      Stage 6:    CRLB initial weights (label-independent, clean)
      Stage 7:    Active model — full-data feature selection (for display)
      Stage 8:    Active model — full-data fusion (for interactive use)
      Stage 9:    Populate MetricsResult from OOF (primary UI metrics)
      Stage 10:   OOF baseline comparison
    """
    exp = experiment
    exp_id = exp.experiment_id
    seed = exp.seed

    total_stages = 11
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

    labels = radar_data["dataset_labels"]

    exp.feature_matrix = feature_matrix
    exp.feature_names = feature_names
    exp.labels = labels

    n_radar = len(radar_data["feature_names"])
    n_thermal = len(thermal_data["feature_names"])
    n_acoustic = len(acoustic_data["feature_names"])

    sensor_feature_counts = {
        "radar": n_radar,
        "thermal": n_thermal,
        "acoustic": n_acoustic,
    }

    event_bus.publish(Event(
        event_type=EventType.FEATURE_EXTRACTION_COMPLETED,
        source="FEATURES",
        message=f"{feature_matrix.shape[1]} features × {feature_matrix.shape[0]} samples",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # STAGE 5: OOF EVALUATION (leakage-free)
    # ========================================================
    stage_num = 4
    stage_progress("OOF EVALUATION", 0.0, "5-fold stratified out-of-fold evaluation")

    from science.evaluation.oof_protocol import run_oof_evaluation

    def oof_progress(stage="", progress=0.0, message=""):
        stage_progress("OOF EVALUATION", progress, f"[OOF] {message}")

    oof_result = run_oof_evaluation(
        feature_matrix=feature_matrix,
        labels=labels,
        feature_names=feature_names,
        sensor_feature_counts=sensor_feature_counts,
        n_splits=exp.evaluation_config.n_folds,
        seed=seed,
        n_bootstrap=exp.evaluation_config.n_bootstrap,
        ci_level=exp.evaluation_config.ci_level,
        k_target=exp.feature_config.k_target,
        fs_alpha=exp.feature_config.alpha,
        fs_beta=exp.feature_config.beta,
        fs_gamma=exp.feature_config.gamma,
        lam=exp.fusion_config.lam,
        n_restarts=exp.fusion_config.n_restarts,
        target_far=exp.fusion_config.target_far,
        progress_callback=oof_progress,
    )

    exp.oof_result = oof_result

    # Store evaluation fusion weights (mean ± SD across folds)
    exp.evaluation_fusion_weights = oof_result.mean_fusion_weights
    exp.evaluation_fusion_weights_std = oof_result.std_fusion_weights

    event_bus.publish(Event(
        event_type=EventType.METRICS_RESULT_UPDATED,
        source="OOF-EVALUATION",
        message=f"OOF AUC={oof_result.auc:.4f} "
                f"[{oof_result.bootstrap_ci['ci_lower']:.4f}–"
                f"{oof_result.bootstrap_ci['ci_upper']:.4f}]",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # STAGE 6: CRLB Initial Weights (Module E) — label-independent
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
    # STAGE 7: Active Model — Full-data Feature Selection (Module G)
    # This is for display/interactive use, NOT for evaluation claims.
    # ========================================================
    stage_num = 6
    stage_progress("BUILDING QUBO (Active Model)", 0.0, "MID-mRMR feature selection")

    from science.qubo.module_g import run_feature_selection, run_feature_selection_dinkelbach
    from app.state import FeatureSelectionMethod

    def qubo_progress(stage="", progress=0.0, message=""):
        stage_progress("BUILDING QUBO (Active Model)", progress, message)

    if exp.feature_config.method == FeatureSelectionMethod.MIQ_DINKELBACH:
        fs_data = run_feature_selection_dinkelbach(
            qubo_progress,
            feature_matrix, labels, feature_names,
            k_target=exp.feature_config.k_target,
            alpha=exp.feature_config.alpha,
            beta=exp.feature_config.beta,
            gamma=exp.feature_config.gamma,
            max_iter=exp.feature_config.dinkelbach_max_iter,
            tol=exp.feature_config.dinkelbach_tol,
            damping=exp.feature_config.dinkelbach_damping,
            num_reads=exp.feature_config.num_reads,
            run_brute_force=exp.feature_config.run_brute_force,
        )
    else:
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

    exp.feature_result.method = fs_data.get("method", "MID (Default)")
    exp.feature_result.feature_names = fs_data["feature_names"]
    exp.feature_result.relevance = fs_data["relevance"]
    exp.feature_result.redundancy_matrix = fs_data["redundancy_matrix"]
    exp.feature_result.Q_matrix = fs_data.get("Q_matrix")
    exp.feature_result.selected_indices = fs_data["selected_indices"]
    exp.feature_result.selected_features = fs_data["selected_features"]
    exp.feature_result.objective_value = fs_data.get("objective_value")
    exp.feature_result.brute_force_objective = fs_data.get("brute_force_objective")
    exp.feature_result.brute_force_selected = fs_data.get("brute_force_selected")
    
    # Check match vs MID brute force regardless of method
    exp.feature_result.subset_match = fs_data.get("subset_match_vs_mid") if "subset_match_vs_mid" in fs_data else fs_data.get("subset_match")
    
    exp.feature_result.solver = fs_data.get("solver")
    exp.feature_result.solver_metadata = fs_data.get("solver_metadata")
    exp.feature_result.final_lambda = fs_data.get("final_lambda")
    exp.feature_result.convergence_history = fs_data.get("convergence_history")
    exp.feature_result.status = ModuleStatus.COMPLETED

    match_val = fs_data.get("subset_match_vs_mid") if "subset_match_vs_mid" in fs_data else fs_data.get("subset_match")
    event_bus.publish(Event(
        event_type=EventType.FEATURE_SELECTION_RESULT_UPDATED,
        source="FEATURE-QUBO",
        message=f"Selected {len(fs_data['selected_features'])}/{len(feature_names)}, match={match_val}",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # STAGE 8: Active Model — Full-data Fusion (Module H)
    # For interactive sensor analysis, NOT for evaluation claims.
    # ========================================================
    stage_num = 7
    stage_progress("OPTIMIZING FUSION (Active Model)", 0.0, "Full-data Rayleigh-LDA")

    # Train per-sensor classifiers on ALL data (active model)
    sensor_classifiers = {}
    sensor_scores = {}
    r_end = n_radar + n_thermal
    a_end = r_end + n_acoustic

    clf_radar = LogisticRegression(max_iter=1000, random_state=seed)
    clf_radar.fit(feature_matrix[:, :n_radar], labels)
    sensor_scores["radar"] = clf_radar.predict_proba(feature_matrix[:, :n_radar])[:, 1]
    sensor_classifiers["radar"] = clf_radar

    clf_thermal = LogisticRegression(max_iter=1000, random_state=seed)
    clf_thermal.fit(feature_matrix[:, n_radar:r_end], labels)
    sensor_scores["thermal"] = clf_thermal.predict_proba(feature_matrix[:, n_radar:r_end])[:, 1]
    sensor_classifiers["thermal"] = clf_thermal

    clf_acoustic = LogisticRegression(max_iter=1000, random_state=seed)
    clf_acoustic.fit(feature_matrix[:, r_end:a_end], labels)
    sensor_scores["acoustic"] = clf_acoustic.predict_proba(feature_matrix[:, r_end:a_end])[:, 1]
    sensor_classifiers["acoustic"] = clf_acoustic

    exp.sensor_classifiers = sensor_classifiers
    exp.sensor_scores = sensor_scores

    from science.fusion.module_h import run_fusion_optimization

    def fusion_progress(stage="", progress=0.0, message=""):
        stage_progress("OPTIMIZING FUSION (Active Model)", progress, message)

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
    exp.active_model_fusion_weights = fusion_data["weights"]

    event_bus.publish(Event(
        event_type=EventType.FUSION_RESULT_UPDATED,
        source="FUSION",
        message=f"Active model J'={fusion_data['fisher_objective']:.4f}",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # STAGE 9: Populate MetricsResult from OOF predictions
    # Primary UI metrics source — NOT from in-sample.
    # ========================================================
    stage_num = 8
    stage_progress("POPULATING METRICS", 0.0, "Writing OOF metrics to MetricsResult")

    exp.metrics_result.auc = ConfidenceInterval(
        point_estimate=oof_result.auc,
        ci_lower=oof_result.bootstrap_ci["ci_lower"],
        ci_upper=oof_result.bootstrap_ci["ci_upper"],
        n_bootstrap=exp.evaluation_config.n_bootstrap,
    )
    exp.metrics_result.detection_rate = ConfidenceInterval(
        point_estimate=oof_result.aggregate_metrics.detection_rate,
        ci_lower=oof_result.aggregate_metrics.detection_rate,
        ci_upper=oof_result.aggregate_metrics.detection_rate,
    )
    exp.metrics_result.false_alarm_rate = ConfidenceInterval(
        point_estimate=oof_result.aggregate_metrics.false_alarm_rate,
        ci_lower=oof_result.aggregate_metrics.false_alarm_rate,
        ci_upper=oof_result.aggregate_metrics.false_alarm_rate,
    )
    exp.metrics_result.precision = ConfidenceInterval(
        point_estimate=oof_result.aggregate_metrics.precision,
        ci_lower=oof_result.aggregate_metrics.precision,
        ci_upper=oof_result.aggregate_metrics.precision,
    )
    exp.metrics_result.recall = ConfidenceInterval(
        point_estimate=oof_result.aggregate_metrics.recall,
        ci_lower=oof_result.aggregate_metrics.recall,
        ci_upper=oof_result.aggregate_metrics.recall,
    )
    exp.metrics_result.f1 = ConfidenceInterval(
        point_estimate=oof_result.aggregate_metrics.f1,
        ci_lower=oof_result.aggregate_metrics.f1,
        ci_upper=oof_result.aggregate_metrics.f1,
    )
    exp.metrics_result.roc_fpr = oof_result.roc_fpr
    exp.metrics_result.roc_tpr = oof_result.roc_tpr
    exp.metrics_result.confusion_matrix = np.array([
        [oof_result.aggregate_metrics.tn, oof_result.aggregate_metrics.fp],
        [oof_result.aggregate_metrics.fn, oof_result.aggregate_metrics.tp],
    ])
    exp.metrics_result.status = ModuleStatus.COMPLETED

    event_bus.publish(Event(
        event_type=EventType.METRICS_RESULT_UPDATED,
        source="EVALUATION",
        message=f"OOF AUC={exp.metrics_result.auc}",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # STAGE 10: OOF Baseline Comparison
    # All baselines use the SAME outer fold assignments.
    # ========================================================
    stage_num = 9
    stage_progress("COMPARING BASELINES (OOF)", 0.0, "Running all 5 classical baselines with OOF")

    from science.evaluation.oof_protocol import run_oof_baselines

    # Reconstruct fold assignments from the OOF result
    skf = StratifiedKFold(
        n_splits=exp.evaluation_config.n_folds,
        shuffle=True,
        random_state=seed,
    )
    fold_assignments = list(skf.split(feature_matrix, labels))

    def baseline_progress(stage="", progress=0.0, message=""):
        stage_progress("COMPARING BASELINES (OOF)", progress, message)

    from app.state import BaselineResult

    baseline_data = run_oof_baselines(
        feature_matrix=feature_matrix,
        labels=labels,
        sensor_feature_counts=sensor_feature_counts,
        fold_assignments=fold_assignments,
        crlb_weights=crlb_data["initial_weights"],
        lam=exp.fusion_config.lam,
        n_restarts=exp.fusion_config.n_restarts,
        seed=seed,
        target_far=exp.fusion_config.target_far,
        progress_callback=baseline_progress,
    )

    exp.baseline_results = []
    for bd in baseline_data:
        br = BaselineResult(
            method_name=bd["method_name"],
            weights=bd["weights"],
            objective_value=bd.get("objective_value"),
            solve_time_s=bd["solve_time_s"],
            fused_scores=bd["fused_scores"],
            auc=bd["auc"],
            threshold=bd.get("threshold"),
        )
        exp.baseline_results.append(br)

    event_bus.publish(Event(
        event_type=EventType.BASELINE_RESULT_UPDATED,
        source="BASELINES",
        message=f"{len(baseline_data)} baselines evaluated with OOF protocol",
        data={"experiment_id": exp_id},
    ))

    # ========================================================
    # STAGE 11: Persistent Model — Train/Update & Save
    # ========================================================
    stage_num = 10
    stage_progress("SAVING MODEL", 0.0, "Training persistent detection model")

    from science.model.detection_model import DetectionModel

    # Model save path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    model_path = os.path.join(models_dir, "qt223_detection_model.pkl")

    # Check if a model already exists for incremental update
    existing_model = None
    if os.path.exists(model_path):
        try:
            existing_model = DetectionModel.load(model_path)
            stage_progress("SAVING MODEL", 0.2, f"Loaded existing model ({existing_model.total_samples_seen} samples seen)")
        except Exception:
            existing_model = None

    if existing_model is not None and existing_model.is_trained:
        # Incremental update with new data
        stage_progress("SAVING MODEL", 0.4, "Incremental update with new data")
        update_result = existing_model.update(
            X_new=feature_matrix,
            y_new=labels,
            regime=exp.target_regime.value,
            seed=seed,
        )

        # Update fusion weights from this run's optimization
        if exp.fusion_result.weights_array is not None:
            existing_model.fusion_weights = exp.fusion_result.weights_array.copy()
            existing_model.fusion_weights_dict = exp.fusion_result.weights.copy() if exp.fusion_result.weights else None
        if exp.fusion_result.threshold is not None:
            existing_model.threshold = exp.fusion_result.threshold

        model = existing_model

        stage_progress("SAVING MODEL", 0.6,
                        f"Updated model — {model.total_samples_seen} total samples, "
                        f"{sum(1 for r in model.training_history if r.incremental)} updates")
    else:
        # Fresh training
        stage_progress("SAVING MODEL", 0.4, "Training new model from scratch")
        model = DetectionModel()

        metrics_dict = {
            "auc": oof_result.auc if oof_result.auc else 0.0,
            "f1": oof_result.aggregate_metrics.f1 if oof_result.aggregate_metrics else 0.0,
            "precision": oof_result.aggregate_metrics.precision if oof_result.aggregate_metrics else 0.0,
            "recall": oof_result.aggregate_metrics.recall if oof_result.aggregate_metrics else 0.0,
            "far": oof_result.aggregate_metrics.false_alarm_rate if oof_result.aggregate_metrics else 0.0,
        }

        model.train(
            X=feature_matrix,
            y=labels,
            feature_names=feature_names,
            sensor_feature_counts=sensor_feature_counts,
            fusion_weights=exp.fusion_result.weights_array if exp.fusion_result.weights_array is not None else np.array([1/3, 1/3, 1/3]),
            threshold=exp.fusion_result.threshold if exp.fusion_result.threshold is not None else 0.5,
            regime=exp.target_regime.value,
            seed=seed,
            selected_indices=exp.feature_result.selected_indices,
            fisher_objective=exp.fusion_result.fisher_objective if exp.fusion_result.fisher_objective else 0.0,
            metrics=metrics_dict,
        )

        stage_progress("SAVING MODEL", 0.6, f"Model trained — {model.total_samples_seen} samples")

    # Save model to disk
    stage_progress("SAVING MODEL", 0.8, f"Saving to {model_path}")
    saved_path = model.save(model_path)

    # Store model reference on experiment
    exp.detection_model = model
    exp.detection_model_path = saved_path

    event_bus.publish(Event(
        event_type=EventType.MODULE_COMPLETED,
        source="MODEL",
        message=f"Model saved: {model.total_samples_seen} samples, "
                f"{len(model.training_history)} runs "
                f"({sum(1 for r in model.training_history if r.incremental)} incremental)",
        data={"experiment_id": exp_id, "model_path": saved_path},
    ))

    # ========================================================
    # COMPLETED
    # ========================================================
    exp.status = ModuleStatus.COMPLETED

    stage_progress("COMPLETED", 1.0, f"Experiment {exp_id} complete — OOF AUC={oof_result.auc:.4f}")

    return exp

