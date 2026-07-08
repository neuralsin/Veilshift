- [x] Phase A: Environment and Dependency Validation
- [x] Phase B: Generator Autopsy (generator_probe.py)
- [x] Phase C: Feature Extraction Autopsy (feature_probe.py)
- [x] Phase D: Label Proxy and Lineage Attack (lineage_probe.py)
- [x] Phase E: OOF Structural Validation (oof_probe.py)
- [x] Phase F: Model / Training Path Isolation (model_probe.py)
- [x] Phase G: Negative Controls (control_datasets.py)
- [x] Phase H: Positive Controls (control_datasets.py)
- [x] Phase I: Modality Isolation (modality_probe.py)
- [x] Phase J/K: AUC Deep Dives (metric_probe.py)
- [x] Phase L: Fusion Optimizer Autopsy (fusion_probe.py)
- [x] Phase M: Feature-Selection Autopsy (feature_probe.py)
- [x] Phase N: Threshold Autopsy (threshold_probe.py)
- [x] Phase O: Primary Metric Reconstruction (metric_probe.py)
- [x] Phase P: Bootstrap Lineage (lineage_probe.py)
- [x] Phase Q: Baseline Fairness (baseline_probe.py)
- [x] Phase R: Contribution Analysis (contribution_probe.py)
- [x] Phase S: Degradation Lab (degradation_probe.py)
- [x] Phase T: Solver Attribution (solver_probe.py)
- [x] Phase U: Multi-seed (scaling_probe.py)
- [x] Phase V: Adversarial Scenario Matrix (scaling_probe.py)
- [x] Phase W: Metamorphic Tests (dataset_probe.py)
- [x] Phase X: Scientific Output Finiteness (finiteness_probe.py)
- [x] Phase Y: Full Pipeline Execution (full_pipeline_probe.py)
- [x] Phase Z: Cycle Verdict (cycle_report.py)
- [x] Integrate all phases into autonomous_scientific_validator.py
- [x] Run the validator and fix discovered defects until 10 clean cycles.

# Task Tracker — Production Hardening & Stretch Goals

## 1. Dynamic Trust Migration (Section 3.4)
- [x] Add `contribution_at_severity()` to `module_m.py`
- [x] Extend `degradation_step()` with contribution data
- [x] Extend `run_degradation_sweep()` to collect contribution vectors
- [x] Update `ui/pages/degradation.py` with trust migration chart

## 2. Dinkelbach-MIQ Feature Selection (G.2)
- [x] Add `solve_mrmr_dinkelbach()` to `module_g.py`
- [x] Add `run_feature_selection_dinkelbach()` wrapper
- [x] Add `FeatureSelectionMethod` enum to `state.py`
- [x] Update `ui/pages/feature_space.py` with MIQ toggle and λ-convergence plot

## 3. Production Export Module
- [x] Create `science/export/__init__.py`
- [x] Create `science/export/production_export.py`
- [x] Wire export into UI (button + progress + display)

## 4. AUC Hardening & Explicit Display
- [x] Add CONVENTIONAL/STEALTH presets to `state.py`
- [x] Add regime toggle to `mission_control.py`
- [x] Add explicit weight/threshold/feature display panels
- [x] Update `fusion.py` page with scatter matrix + weight comparison

## 5. Verification
- [x] Test STEALTH export produces AUC ~0.50
- [x] Test CONVENTIONAL export produces AUC ~1.0
- [x] Test Dinkelbach λ convergence
- [x] Test trust migration data structure
- [x] Full UI smoke test
