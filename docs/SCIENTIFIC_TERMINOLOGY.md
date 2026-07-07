# VEILSHIFT SCIENTIFIC TERMINOLOGY CONSTITUTION

This document defines the canonical scientific terminology used across the VEILSHIFT (QT-2.23) application.
All UI components, log messages, and operator-facing inferences MUST adhere to these definitions.

## Core Terminology

### 1. Target Condition
**CANONICAL TERM**: `TARGET CONDITION: LOW OBSERVABILITY`
- **DEFINITION**: A target exhibiting a low signature profile (e.g., low radar cross-section, low thermal contrast).
- **UNDERLYING STATE**: Configured via the `TargetRegime` enumeration.
- **ALLOWED**: "Low Observability", "Low-RCS Target", "Low-SNR Detection Regime"
- **FORBIDDEN**: "Stealth", "Threat", "Hostile", "Bogey" (except when quoting the official challenge name "QT-2.23 Sensor Fusion and Stealth Detection").

### 2. Detection Outcome
**CANONICAL TERM**: `DETECTION THRESHOLD EXCEEDED` or `TARGET DETECTED`
- **DEFINITION**: When the fused model probability strictly exceeds the Neyman-Pearson threshold that guarantees the configured False Alarm Rate.
- **ALLOWED**: "Target Detected", "Detection Positive"
- **FORBIDDEN**: "Target Acquired", "Sensor Lock", "Signature Acquired"

### 3. Fusion Coefficient
**CANONICAL TERM**: `MEAN OOF FUSION WEIGHT`
- **DEFINITION**: A non-negative normalized coefficient applied to a modality classifier output during linear score fusion.
- **UNDERLYING COMPUTATION**: Rayleigh discriminant objective optimized via solver (e.g., Simulated Annealing backend). Computed as the mean across the outer cross-validation folds.
- **ALLOWED**: "Fusion Weight", "Mean OOF Fusion Weight", "Modality Weight", "Adaptive Fusion Weight Shift"
- **FORBIDDEN**: "Trust", "Sensor Trust", "Sensor Confidence", "Trust Distribution", "Trust Migration", "Dominant Sensor"

### 4. Classifier Score
**CANONICAL TERM**: `MODEL PROBABILITY ESTIMATE`
- **DEFINITION**: The raw probability output (predict_proba) from the underlying Logistic Regression classifiers before fusion.
- **ALLOWED**: "Model Probability Estimate", "Class Posterior Estimate"
- **FORBIDDEN**: "Confidence", "Sensor Confidence", "Certainty", "Score"

### 5. Final Output Score
**CANONICAL TERM**: `FUSED DETECTION SCORE`
- **DEFINITION**: The linear combination of the sensor model probability estimates using the optimized fusion weights.
- **ALLOWED**: "Fused Detection Score", "Fused Class Score"
- **FORBIDDEN**: "Confidence", "Score", "System Confidence"

### 6. Uncertainty
**CANONICAL TERM**: `95% BOOTSTRAP CI` or `FOLD SD`
- **DEFINITION**: Standard uncertainty metrics. Bootstrapped confidence intervals are computed over 1000 OOF predictions. Standard Deviation is computed across the 5 outer folds.
- **ALLOWED**: "[0.88 - 0.94] (95% CI)", "± 0.05 (Fold SD)"
- **FORBIDDEN**: Displaying "0.000 ± 0.000" for an un-run experiment. Use "—" or "NOT COMPUTED". Do not use naked "±" without defining whether it is SD, SE, or CI.

### 7. Feature Selection and Solvers
**CANONICAL TERM**: `QUBO FORMULATION` and `SIMULATED ANNEALING BACKEND`
- **DEFINITION**: The Quadratic Unconstrained Binary Optimization formulation for feature selection. Solved classically via `dimod` Simulated Annealing if true quantum hardware is unavailable.
- **ALLOWED**: "QUBO Feature Selection", "Simulated Annealing Backend", "Classical Fallback Active"
- **FORBIDDEN**: "Quantum Optimized", "Quantum Result", "Quantum Solver", "Quantum Advantage" (unless a real Quantum Processing Unit was used). "Backup Active".

### 8. System Status
**CANONICAL TERM**: Explicit technical statuses separating operator view from tracebacks.
- **ALLOWED**: "QUBO SAMPLER UNAVAILABLE", "PIPELINE FAILED"
- **FORBIDDEN**: "FAILED: No module named 'dimod'" as a primary operator-facing status.

### 9. Empty States
**CANONICAL TERM**: `—` or `NOT COMPUTED`
- **DEFINITION**: Used for any metric or value that has not yet been computed or is currently unavailable.
- **FORBIDDEN**: "0", "0.000", "0.0%" when implying an un-run experiment.

### 10. Ranking
**CANONICAL TERM**: `AUC RANK` or `FUSION WEIGHT RANK`
- **DEFINITION**: The explicit sorted order of a specific computed metric.
- **ALLOWED**: "Fusion Weight Order", "SHAP Importance Rank"
- **FORBIDDEN**: "Priority", "Rank" (when used alone)
