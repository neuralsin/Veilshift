"""
QT-2.23 — Application State Models

Typed dataclasses for every result, configuration, and state object.
These are the single source of truth for what data exists in the application.
Serialization: metadata → JSON, arrays → NPZ, tabular → CSV.
Figures are always regenerated from result data, never serialized as source of truth.
"""

from __future__ import annotations

SCHEMA_VERSION = 2  # Incremented from implicit v1 when OOF evaluation was added
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# ENUMS
# ============================================================

class ModuleStatus(Enum):
    """Status of a backend module / computation stage."""
    NOT_CONFIGURED = "NOT_CONFIGURED"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    WARNING = "WARNING"
    FAILED = "FAILED"


class TargetRegime(Enum):
    CONVENTIONAL = "Conventional"
    STEALTH = "Low Observability"
    CUSTOM = "Custom"


class ClutterModel(Enum):
    RAYLEIGH = "Rayleigh"
    WEIBULL = "Weibull"


class NoiseModel(Enum):
    WHITE = "White"
    ONE_OVER_F = "1/f"


class OptimizationMode(Enum):
    CONTINUOUS_SLSQP = "Continuous · SLSQP"
    CONTINUOUS_DE = "Continuous · Differential Evolution"
    BINARY_QUBO = "Binary QUBO · 4-bit"
    BACKUP_SA = "Classical · Simulated Annealing"


class SolverBackend(Enum):
    NEAL_SA = "Simulated Annealing Backend"
    SCIPY_SLSQP = "Classical · SLSQP"
    SCIPY_DE = "Classical · Differential Evolution"


class DegradationType(Enum):
    RADAR_JAMMING = "Interference / Jamming Noise"
    RADAR_CLUTTER = "Clutter Increase"
    RADAR_REMOVAL = "Sensor Removal"
    THERMAL_NETD = "NETD Increase"
    THERMAL_NOISE = "Spatial Noise Increase"
    THERMAL_REMOVAL = "Sensor Removal"
    ACOUSTIC_MASKING = "Ambient Masking Noise"
    ACOUSTIC_SUPPRESSION = "Source Suppression"
    ACOUSTIC_REMOVAL = "Sensor Removal"


class TaskType(Enum):
    RADAR_SIMULATION = auto()
    THERMAL_SIMULATION = auto()
    ACOUSTIC_SIMULATION = auto()
    FEATURE_EXTRACTION = auto()
    SENSOR_MODEL_TRAINING = auto()
    QUBO_BUILD = auto()
    FEATURE_OPTIMIZATION = auto()
    FUSION_OPTIMIZATION = auto()
    BASELINE_COMPARISON = auto()
    EVALUATION = auto()
    CONTRIBUTION_ANALYSIS = auto()
    DEGRADATION_SWEEP = auto()
    SCALING_EXPERIMENT = auto()
    EXPERIMENT_EXPORT = auto()
    FULL_PIPELINE = auto()


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

@dataclass
class ConfidenceInterval:
    """Bootstrap confidence interval for a metric."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    n_bootstrap: int = 1000
    n_folds: int = 5
    n_seeds: int = 5

    def __str__(self) -> str:
        return f"{self.point_estimate:.3f} [{self.ci_lower:.3f}–{self.ci_upper:.3f}]"


# ============================================================
# SENSOR CONFIGURATIONS
# ============================================================

@dataclass
class RadarConfig:
    """Module A configuration — all from Appendix B defaults."""
    transmit_power_W: float = 10000.0
    antenna_gain_dB: float = 30.0
    frequency_hz: float = 10e9
    bandwidth_hz: float = 1e6
    noise_figure_dB: float = 4.77
    system_loss_dB: float = 6.0
    range_m: float = 50000.0
    target_regime: TargetRegime = TargetRegime.STEALTH
    rcs_m2: float = 0.01
    clutter_model: ClutterModel = ClutterModel.RAYLEIGH
    clutter_power: float = 1.0
    weibull_shape: float = 1.5
    pfa: float = 1e-4
    num_ref_cells: int = 16
    guard_cells: int = 4
    mc_trials: int = 200_000
    num_range_cells: int = 500
    num_samples: int = 6000


@dataclass
class ThermalConfig:
    """Module B configuration — Appendix B defaults."""
    bg_temperature_K: float = 290.0
    bg_emissivity: float = 0.95
    target_temperature_K: float = 310.0
    target_emissivity: float = 0.3
    target_regime: TargetRegime = TargetRegime.STEALTH
    netd_K: float = 0.05
    noise_model: NoiseModel = NoiseModel.ONE_OVER_F
    noise_alpha: float = 1.0
    frame_size: Tuple[int, int] = (64, 64)
    num_samples: int = 6000


@dataclass
class AcousticConfig:
    """Module C configuration — Appendix B defaults."""
    source_level_dB: float = 110.0
    target_regime: TargetRegime = TargetRegime.STEALTH
    range_m: float = 5000.0
    sea_state: int = 3
    directivity_index_dB: float = 10.0
    detection_threshold_dB: float = 0.0
    absorption_coef: float = 0.005
    duration_s: float = 60.0
    sampling_freq_hz: float = 2000.0
    tonal_freqs_hz: Tuple[float, ...] = (50.0, 120.0, 240.0)
    stft_nperseg: int = 1024
    stft_noverlap: int = 512
    num_samples: int = 6000


# ============================================================
# OPTIMIZATION CONFIGURATIONS
# ============================================================

@dataclass
class FeatureSelectionConfig:
    """Module G configuration — Appendix B defaults."""
    alpha: float = 1.0       # Relevance weight
    beta: float = 1.0        # Redundancy penalty
    gamma: float = 2.0       # Cardinality penalty
    k_target: int = 6        # Target number of features
    num_reads: int = 1000    # neal sampler reads
    run_brute_force: bool = True  # Always validate


@dataclass
class FusionConfig:
    """Module H configuration — Appendix B defaults."""
    lam: float = 0.5         # Lagrangian trade-off
    optimization_mode: OptimizationMode = OptimizationMode.CONTINUOUS_SLSQP
    n_restarts: int = 10     # SLSQP random restarts
    seed: int = 42
    # Binary QUBO fallback
    bits_per_weight: int = 4
    simplex_penalty: float = 5.0
    # Neyman-Pearson
    target_far: float = 0.01


@dataclass
class EvaluationConfig:
    """Module K configuration — Appendix B defaults."""
    n_folds: int = 5
    n_seeds: int = 5
    n_bootstrap: int = 1000
    ci_level: float = 0.95
    base_seed: int = 42


@dataclass
class DegradationConfig:
    """Module M configuration."""
    sensor: str = "radar"
    degradation_type: DegradationType = DegradationType.RADAR_JAMMING
    severity_steps: int = 5
    severity_min: float = 0.0
    severity_max: float = 1.0


@dataclass
class ScalingConfig:
    """Module J configuration."""
    problem_sizes: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    solvers: List[str] = field(default_factory=lambda: [
        "Brute Force", "Grid Search", "SLSQP", "Differential Evolution",
        "Simulated Annealing"
    ])


# ============================================================
# RESULT DATACLASSES — one per science module
# ============================================================

@dataclass
class RadarResult:
    """Module A output."""
    range_profile: Optional[np.ndarray] = None
    range_doppler_map: Optional[np.ndarray] = None
    clutter_map: Optional[np.ndarray] = None
    cfar_threshold: Optional[np.ndarray] = None
    detections: Optional[np.ndarray] = None
    snr_db: Optional[float] = None
    configured_pfa: Optional[float] = None
    empirical_pfa: Optional[float] = None
    detection_probability: Optional[float] = None
    cfar_alpha: Optional[float] = None
    features: Optional[Dict[str, float]] = None
    feature_names: Optional[List[str]] = None
    feature_values: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    status: ModuleStatus = ModuleStatus.NOT_CONFIGURED


@dataclass
class ThermalResult:
    """Module B output."""
    thermal_frame: Optional[np.ndarray] = None
    contrast_map: Optional[np.ndarray] = None
    noise_field: Optional[np.ndarray] = None
    noise_psd_freqs: Optional[np.ndarray] = None
    noise_psd_power: Optional[np.ndarray] = None
    noise_psd_slope: Optional[float] = None
    target_exitance: Optional[float] = None
    background_exitance: Optional[float] = None
    delta_M: Optional[float] = None
    thermal_snr: Optional[float] = None
    features: Optional[Dict[str, float]] = None
    feature_names: Optional[List[str]] = None
    feature_values: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    status: ModuleStatus = ModuleStatus.NOT_CONFIGURED


@dataclass
class AcousticResult:
    """Module C output."""
    time_series: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    lofar_image: Optional[np.ndarray] = None
    lofar_frequencies: Optional[np.ndarray] = None
    lofar_times: Optional[np.ndarray] = None
    signal_excess_db: Optional[float] = None
    transmission_loss_db: Optional[float] = None
    noise_level_db: Optional[float] = None
    tonal_power: Optional[float] = None
    broadband_power: Optional[float] = None
    features: Optional[Dict[str, float]] = None
    feature_names: Optional[List[str]] = None
    feature_values: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    status: ModuleStatus = ModuleStatus.NOT_CONFIGURED


@dataclass
class FeatureSelectionResult:
    """Module G output."""
    feature_names: Optional[List[str]] = None
    relevance: Optional[np.ndarray] = None
    redundancy_matrix: Optional[np.ndarray] = None
    Q_matrix: Optional[np.ndarray] = None
    selected_indices: Optional[List[int]] = None
    selected_features: Optional[List[str]] = None
    objective_value: Optional[float] = None
    brute_force_objective: Optional[float] = None
    brute_force_selected: Optional[List[int]] = None
    subset_match: Optional[bool] = None
    solver: Optional[str] = None
    solver_metadata: Optional[Dict[str, Any]] = None
    status: ModuleStatus = ModuleStatus.NOT_CONFIGURED


@dataclass
class FusionResult:
    """Module H output."""
    weights: Optional[Dict[str, float]] = None
    weights_array: Optional[np.ndarray] = None
    fused_scores: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    fisher_objective: Optional[float] = None
    S_b: Optional[np.ndarray] = None
    S_w: Optional[np.ndarray] = None
    solver: Optional[str] = None
    solve_time_s: Optional[float] = None
    optimization_mode: Optional[str] = None
    # Score distributions for visualization
    sensor_scores_h0: Optional[Dict[str, np.ndarray]] = None
    sensor_scores_h1: Optional[Dict[str, np.ndarray]] = None
    fused_scores_h0: Optional[np.ndarray] = None
    fused_scores_h1: Optional[np.ndarray] = None
    status: ModuleStatus = ModuleStatus.NOT_CONFIGURED


@dataclass
class MetricsResult:
    """Module K output — evaluation metrics with uncertainty."""
    detection_rate: Optional[ConfidenceInterval] = None
    false_alarm_rate: Optional[ConfidenceInterval] = None
    precision: Optional[ConfidenceInterval] = None
    recall: Optional[ConfidenceInterval] = None
    f1: Optional[ConfidenceInterval] = None
    auc: Optional[ConfidenceInterval] = None
    roc_fpr: Optional[np.ndarray] = None
    roc_tpr: Optional[np.ndarray] = None
    confusion_matrix: Optional[np.ndarray] = None
    cv_fold_aucs: Optional[np.ndarray] = None
    seed_aucs: Optional[np.ndarray] = None
    status: ModuleStatus = ModuleStatus.NOT_CONFIGURED


@dataclass
class BaselineResult:
    """Module F output — one per baseline method."""
    method_name: str = ""
    weights: Optional[Dict[str, float]] = None
    objective_value: Optional[float] = None
    solve_time_s: Optional[float] = None
    auc: Optional[float] = None
    metrics: Optional[MetricsResult] = None
    fused_scores: Optional[np.ndarray] = None
    threshold: Optional[float] = None


@dataclass
class ContributionResult:
    """Module L output."""
    weight_contributions: Optional[Dict[str, float]] = None
    ablation_delta_auc: Optional[Dict[str, float]] = None
    ablation_normalized: Optional[Dict[str, float]] = None
    shap_sensor_contribution: Optional[Dict[str, float]] = None
    shap_feature_values: Optional[np.ndarray] = None
    shap_feature_names: Optional[List[str]] = None
    agreement_score: Optional[float] = None
    agreement_status: Optional[str] = None  # HIGH / PARTIAL / DISAGREEMENT
    status: ModuleStatus = ModuleStatus.NOT_CONFIGURED


@dataclass
class DegradationResult:
    """Module M output."""
    sensor: Optional[str] = None
    degradation_type: Optional[str] = None
    severity_values: Optional[np.ndarray] = None
    physical_parameter_values: Optional[np.ndarray] = None
    static_detection_retention: Optional[np.ndarray] = None
    adaptive_detection_retention: Optional[np.ndarray] = None
    static_auc: Optional[np.ndarray] = None
    adaptive_auc: Optional[np.ndarray] = None
    static_ci_lower: Optional[np.ndarray] = None
    static_ci_upper: Optional[np.ndarray] = None
    adaptive_ci_lower: Optional[np.ndarray] = None
    adaptive_ci_upper: Optional[np.ndarray] = None
    weights_by_severity: Optional[Dict[str, np.ndarray]] = None
    contributions_by_severity: Optional[Dict] = None
    status: ModuleStatus = ModuleStatus.NOT_CONFIGURED


@dataclass
class ScalingResult:
    """Module J output."""
    problem_sizes: Optional[List[int]] = None
    results: Optional[List[Dict[str, Any]]] = None
    status: ModuleStatus = ModuleStatus.NOT_CONFIGURED


@dataclass
class CRLBResult:
    """Module E output."""
    theoretical_crlb: Optional[Dict[str, float]] = None
    empirical_variance: Optional[Dict[str, float]] = None
    gap_ratios: Optional[Dict[str, float]] = None
    initial_weights: Optional[np.ndarray] = None
    status: ModuleStatus = ModuleStatus.NOT_CONFIGURED


# ============================================================
# EXPERIMENT STATE — the central experiment object
# ============================================================

@dataclass
class ExperimentState:
    """
    Complete experiment state. Every UI element traces to a field here.
    Serialization: metadata → JSON, arrays → NPZ, tabular → CSV.

    SCHEMA v2: Added OOF evaluation result. Legacy experiments with
    schema_version=1 contain leaked in-sample metrics and must NOT
    be displayed as valid OOF evaluations.
    """
    # Identity
    experiment_id: str = field(default_factory=lambda: f"EXP-{uuid.uuid4().hex[:6].upper()}")
    experiment_name: str = "Untitled Experiment"
    timestamp: float = field(default_factory=time.time)
    seed: int = 42
    schema_version: int = SCHEMA_VERSION

    # Evaluation protocol
    evaluation_protocol: str = "5-FOLD STRATIFIED OOF"

    # Scenario
    target_regime: TargetRegime = TargetRegime.STEALTH

    # Configurations
    radar_config: RadarConfig = field(default_factory=RadarConfig)
    thermal_config: ThermalConfig = field(default_factory=ThermalConfig)
    acoustic_config: AcousticConfig = field(default_factory=AcousticConfig)
    feature_config: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    fusion_config: FusionConfig = field(default_factory=FusionConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    degradation_config: DegradationConfig = field(default_factory=DegradationConfig)
    scaling_config: ScalingConfig = field(default_factory=ScalingConfig)

    # Results — one per module
    radar_result: RadarResult = field(default_factory=RadarResult)
    thermal_result: ThermalResult = field(default_factory=ThermalResult)
    acoustic_result: AcousticResult = field(default_factory=AcousticResult)
    crlb_result: CRLBResult = field(default_factory=CRLBResult)
    feature_result: FeatureSelectionResult = field(default_factory=FeatureSelectionResult)
    fusion_result: FusionResult = field(default_factory=FusionResult)
    metrics_result: MetricsResult = field(default_factory=MetricsResult)
    baseline_results: List[BaselineResult] = field(default_factory=list)
    contribution_result: ContributionResult = field(default_factory=ContributionResult)
    degradation_result: DegradationResult = field(default_factory=DegradationResult)
    scaling_result: ScalingResult = field(default_factory=ScalingResult)

    # OOF Evaluation Result — canonical source of scientific metrics
    # This is populated by the OOF protocol and is the ONLY valid source
    # of performance metrics for the UI.
    oof_result: Optional[Any] = None  # Type: OOFEvaluationResult (from oof_protocol)

    # Combined feature/label arrays built by Module D
    feature_matrix: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    labels: Optional[np.ndarray] = None

    # Active model — trained on ALL data for interactive use.
    # These are NOT used for scientific performance claims.
    sensor_classifiers: Optional[Dict[str, Any]] = None
    sensor_scores: Optional[Dict[str, np.ndarray]] = None
    active_model_fusion_weights: Optional[Dict[str, float]] = None

    # Evaluation fusion weights (mean ± SD across OOF folds)
    evaluation_fusion_weights: Optional[Dict[str, float]] = None
    evaluation_fusion_weights_std: Optional[Dict[str, float]] = None

    # Overall status
    status: ModuleStatus = ModuleStatus.NOT_CONFIGURED

    @property
    def active_modalities(self) -> int:
        count = 0
        if self.radar_result.status == ModuleStatus.COMPLETED:
            count += 1
        if self.thermal_result.status == ModuleStatus.COMPLETED:
            count += 1
        if self.acoustic_result.status == ModuleStatus.COMPLETED:
            count += 1
        return count

    @property
    def solver_label(self) -> str:
        """Returns the honest solver label based on what actually ran."""
        if self.fusion_result.solver:
            return self.fusion_result.solver
        return SolverBackend.NEAL_SA.value

    @property
    def is_legacy_evaluation(self) -> bool:
        """True if this experiment has leaked in-sample metrics (schema v1)."""
        return getattr(self, 'schema_version', 1) < SCHEMA_VERSION

    @property
    def has_valid_oof(self) -> bool:
        """True if OOF evaluation result exists and is valid."""
        return (
            self.oof_result is not None
            and hasattr(self.oof_result, 'is_valid')
            and self.oof_result.is_valid()
        )


# ============================================================
# APPLICATION STATE — top-level UI + app state
# ============================================================

@dataclass
class UIState:
    """UI-specific state (not serialized with experiments)."""
    current_page: str = "Mission Control"
    presentation_mode: bool = False
    presentation_stage: int = 0
    sidebar_collapsed: bool = False
    inspector_visible: bool = True
    chart_dpi: int = 100
    animation_enabled: bool = True


@dataclass
class ApplicationState:
    """Top-level application state combining UI and experiment."""
    ui: UIState = field(default_factory=UIState)
    current_experiment: ExperimentState = field(default_factory=ExperimentState)
    experiments: List[ExperimentState] = field(default_factory=list)
    solver_backend: SolverBackend = SolverBackend.NEAL_SA

    # Settings
    default_seed: int = 42
    default_sample_count: int = 6000
    default_export_dir: str = "exports"
    debug_mode: bool = False
