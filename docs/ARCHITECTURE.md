# System Architecture

Veilshift (QT-2.23) is architected around a strict linear pipeline orchestrated by `app/pipeline.py`. The architecture enforces a strict separation between state management, physics simulation, machine learning, and user interface.

## Pipeline Orchestrator (`app/pipeline.py`)

The pipeline runs completely autonomously once triggered by the UI. It chains 11 strict stages:

1. **Radar Simulation** - Generates physical range-doppler maps, clutter profiles, and target detections.
2. **Thermal Simulation** - Computes thermal signatures and background noise profiles.
3. **Acoustic Simulation** - Generates time-series acoustic frequencies and spectrograms.
4. **Data Aggregation** - Normalizes and combines the features from all three simulators into a single data matrix.
5. **Out-Of-Fold (OOF) Evaluation** - The absolute core of the scientific integrity. Executes a strict 5-fold cross-validation loop to calculate true, out-of-sample metrics (AUC, Pd, FAR). See `EVALUATION_PROTOCOL.md`.
6. **CRLB Initial Weights** - Computes Cramér-Rao Lower Bound to establish theoretical optimal sensor weights prior to any label-aware learning.
7. **Active Model (Feature Selection)** - Runs a QUBO selection across the *entire* dataset. This model is purely for visualization/interactive use and its metrics are quarantined from the evaluation tab.
8. **Active Model (Fusion)** - Optimizes sensor weights across the full dataset.
9. **Metrics Population** - Extracts the canonical metrics strictly from the OOF results (Stage 5) and prepares them for the UI.
10. **OOF Baselines** - Re-runs the OOF protocol using single-sensor data (Radar only, Thermal only, Acoustic only) using identical fold indices to prove fusion superiority.
11. **Confidence Intervals** - Runs a fast bootstrap over the OOF predictions to calculate the reliability bounds (95% CI).

## State Management (`app/state.py`)

The `ExperimentState` class is the single source of truth for the entire application.

It is heavily structured using Python `dataclasses` and strictly divides `active_state` (full-data models) from `evaluation_state` (the strict OOF metrics). This schema separation is what physically prevents data leakage from ruining the UI's display.

### Key Rules of State
- UI bindings for primary metrics (AUC, FAR, Pd) MUST always read from `ExperimentState.metrics_result`.
- UI bindings for baselines MUST always read from `ExperimentState.baseline_metrics`.
- The `ExperimentState` is serialized and passed between background tasks and the UI thread to ensure Thread Safety.
