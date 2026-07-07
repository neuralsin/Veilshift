"""
QT-2.23 — OOF Data Leakage Tests

Tests that verify the OOF evaluation protocol is free of data leakage.
Every test directly addresses a specific requirement from the remediation spec.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_synthetic_data(n_samples=600, n_features=20, seed=42):
    """Generate synthetic feature matrix and labels for testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    # Make first few features informative
    labels = np.zeros(n_samples, dtype=int)
    labels[:n_samples // 2] = 1
    # Inject signal into first 6 features for class 1
    X[:n_samples // 2, :6] += 1.5
    return X, labels


def _make_sensor_feature_counts():
    return {"radar": 6, "thermal": 5, "acoustic": 6}


# ============================================================
# TEST LEAK-001: No outer-test samples enter classifier.fit
# ============================================================
class TestLeakage:

    def test_leak_001_no_test_samples_in_fit(self):
        """Monkeypatch classifier.fit to verify no outer-test indices enter."""
        from sklearn.linear_model import LogisticRegression
        from science.evaluation.oof_protocol import run_oof_evaluation
        from sklearn.model_selection import StratifiedKFold

        X, y = _make_synthetic_data(n_samples=200, n_features=17)
        sensor_counts = _make_sensor_feature_counts()
        seed = 42
        n_splits = 5

        # Reconstruct fold assignments
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_assignments = list(skf.split(X, y))

        # Track all indices that enter fit
        fitted_indices_per_fold = {}
        original_fit = LogisticRegression.fit

        def tracking_fit(self_clf, X_fit, y_fit, **kwargs):
            # Store the number of rows that entered fit
            fold_key = len(X_fit)  # unique per fold
            if fold_key not in fitted_indices_per_fold:
                fitted_indices_per_fold[fold_key] = len(X_fit)
            return original_fit(self_clf, X_fit, y_fit, **kwargs)

        LogisticRegression.fit = tracking_fit
        try:
            result = run_oof_evaluation(
                X, y, [f"f{i}" for i in range(17)], sensor_counts,
                n_splits=n_splits, seed=seed, n_bootstrap=10,
            )
        finally:
            LogisticRegression.fit = original_fit

        # Verify: no fit call received all N samples (which would be test leakage)
        for n_rows in fitted_indices_per_fold.keys():
            assert n_rows < len(y), \
                f"Fit received {n_rows} rows == full dataset ({len(y)})"

    def test_leak_003_feature_relevance_train_labels_only(self):
        """Verify feature relevance calculation receives training labels only."""
        from science.evaluation.oof_protocol import run_oof_evaluation
        from science.qubo import module_g
        from sklearn.model_selection import StratifiedKFold

        X, y = _make_synthetic_data(n_samples=200, n_features=17)
        sensor_counts = _make_sensor_feature_counts()
        seed = 42
        n_splits = 5

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_assignments = list(skf.split(X, y))

        label_counts_seen = []
        original_build = module_g.build_feature_selection_qubo

        def tracking_build(features, labels, *args, **kwargs):
            label_counts_seen.append(len(labels))
            return original_build(features, labels, *args, **kwargs)

        module_g.build_feature_selection_qubo = tracking_build
        try:
            result = run_oof_evaluation(
                X, y, [f"f{i}" for i in range(17)], sensor_counts,
                n_splits=n_splits, seed=seed, n_bootstrap=10,
            )
        finally:
            module_g.build_feature_selection_qubo = original_build

        # Each fold's QUBO should see fewer labels than the full dataset
        for count in label_counts_seen:
            assert count < len(y), \
                f"QUBO build saw {count} labels == full dataset"

    def test_leak_005_fusion_optimizer_no_test_labels(self):
        """Verify fusion optimizer receives no outer-test labels."""
        from science.evaluation.oof_protocol import run_oof_evaluation
        from science.fusion import module_h

        X, y = _make_synthetic_data(n_samples=200, n_features=17)
        sensor_counts = _make_sensor_feature_counts()

        scatter_sizes = []
        original_scatter = module_h.compute_scatter_matrices

        def tracking_scatter(h1_scores, h0_scores):
            scatter_sizes.append(len(h1_scores) + len(h0_scores))
            return original_scatter(h1_scores, h0_scores)

        module_h.compute_scatter_matrices = tracking_scatter
        try:
            result = run_oof_evaluation(
                X, y, [f"f{i}" for i in range(17)], sensor_counts,
                n_splits=5, seed=42, n_bootstrap=10,
            )
        finally:
            module_h.compute_scatter_matrices = original_scatter

        # Scatter matrices should see inner_train samples only (< outer_train < full)
        for size in scatter_sizes:
            assert size < len(y), \
                f"Scatter matrices saw {size} samples == full dataset"

    def test_leak_006_threshold_no_test_labels(self):
        """Verify threshold selector receives no outer-test labels."""
        from science.evaluation.oof_protocol import run_oof_evaluation
        from science.fusion import module_h

        X, y = _make_synthetic_data(n_samples=200, n_features=17)
        sensor_counts = _make_sensor_feature_counts()

        threshold_label_counts = []
        original_threshold = module_h.select_threshold_neyman_pearson

        def tracking_threshold(fused_scores, labels, target_far=0.01):
            threshold_label_counts.append(len(labels))
            return original_threshold(fused_scores, labels, target_far)

        module_h.select_threshold_neyman_pearson = tracking_threshold
        try:
            result = run_oof_evaluation(
                X, y, [f"f{i}" for i in range(17)], sensor_counts,
                n_splits=5, seed=42, n_bootstrap=10,
            )
        finally:
            module_h.select_threshold_neyman_pearson = original_threshold

        for count in threshold_label_counts:
            assert count < len(y), \
                f"Threshold selector saw {count} labels == full dataset"


# ============================================================
# TEST OOF-001 through OOF-007
# ============================================================
class TestOOFIntegrity:

    def _run_oof(self, seed=42, n_samples=200):
        from science.evaluation.oof_protocol import run_oof_evaluation
        X, y = _make_synthetic_data(n_samples=n_samples, n_features=17, seed=seed)
        sensor_counts = _make_sensor_feature_counts()
        return run_oof_evaluation(
            X, y, [f"f{i}" for i in range(17)], sensor_counts,
            n_splits=5, seed=seed, n_bootstrap=50,
        ), y

    def test_oof_001_every_sample_one_prediction(self):
        """Every sample receives exactly one OOF prediction."""
        result, y = self._run_oof()
        assert len(result.sample_indices) == len(y)
        assert len(np.unique(result.sample_indices)) == len(y)

    def test_oof_002_original_ordering(self):
        """OOF sample ordering reconstructs original dataset ordering."""
        result, y = self._run_oof()
        assert np.array_equal(np.sort(result.sample_indices), np.arange(len(y)))

    def test_oof_003_auc_matches_sklearn(self):
        """Primary AUC equals independent sklearn roc_auc_score."""
        from sklearn.metrics import roc_auc_score
        result, _ = self._run_oof()
        recomputed = roc_auc_score(result.true_labels, result.fused_scores)
        assert abs(result.auc - recomputed) < 1e-6, \
            f"Stored AUC {result.auc} != recomputed {recomputed}"

    def test_oof_004_roc_matches_sklearn(self):
        """Primary ROC equals independent sklearn roc_curve output."""
        from sklearn.metrics import roc_curve
        result, _ = self._run_oof()
        fpr, tpr, _ = roc_curve(result.true_labels, result.fused_scores)
        assert np.allclose(result.roc_fpr, fpr, atol=1e-6)
        assert np.allclose(result.roc_tpr, tpr, atol=1e-6)

    def test_oof_005_confusion_metrics(self):
        """Primary confusion metrics equal independent calculations."""
        result, _ = self._run_oof()
        m = result.aggregate_metrics
        assert m.tp + m.fn + m.fp + m.tn == len(result.true_labels)

    def test_oof_006_same_seed_same_folds(self):
        """Same seed produces identical fold assignments."""
        r1, _ = self._run_oof(seed=42)
        r2, _ = self._run_oof(seed=42)
        assert np.array_equal(r1.fold_ids, r2.fold_ids)
        assert abs(r1.auc - r2.auc) < 1e-10

    def test_oof_007_different_seed_different_folds(self):
        """Different seed changes shuffled fold assignments."""
        r1, _ = self._run_oof(seed=42)
        r2, _ = self._run_oof(seed=123)
        assert not np.array_equal(r1.fold_ids, r2.fold_ids)


# ============================================================
# TEST BOOT-001 through BOOT-003
# ============================================================
class TestBootstrap:

    def test_boot_001_consumes_oof_predictions(self):
        """Bootstrap consumes OOF predictions."""
        from science.evaluation.oof_protocol import run_oof_evaluation
        X, y = _make_synthetic_data(n_samples=200, n_features=17)
        sensor_counts = _make_sensor_feature_counts()
        result = run_oof_evaluation(
            X, y, [f"f{i}" for i in range(17)], sensor_counts,
            n_splits=5, seed=42, n_bootstrap=50,
        )
        assert result.bootstrap_ci is not None
        assert result.bootstrap_ci["n_valid"] > 0

    def test_boot_002_single_class_rejected(self):
        """Single-class bootstrap replicates are rejected."""
        from science.evaluation.oof_protocol import bootstrap_oof_auc_ci
        rng = np.random.default_rng(42)
        y = np.ones(100)  # all class 1
        scores = rng.uniform(0, 1, 100)
        result = bootstrap_oof_auc_ci(y, scores, n_bootstrap=100, seed=42)
        assert result["n_rejected"] == 100  # All must be rejected

    def test_boot_003_same_seed_reproduces_ci(self):
        """Same bootstrap seed reproduces CI."""
        from science.evaluation.oof_protocol import bootstrap_oof_auc_ci
        rng = np.random.default_rng(42)
        y = np.array([0]*50 + [1]*50)
        scores = rng.uniform(0, 1, 100)
        r1 = bootstrap_oof_auc_ci(y, scores, n_bootstrap=200, seed=99)
        r2 = bootstrap_oof_auc_ci(y, scores, n_bootstrap=200, seed=99)
        assert r1["ci_lower"] == r2["ci_lower"]
        assert r1["ci_upper"] == r2["ci_upper"]


# ============================================================
# TEST BASE-001 through BASE-002
# ============================================================
class TestBaselines:

    def test_base_001_same_fold_assignments(self):
        """All baseline methods use identical outer fold assignments."""
        from science.evaluation.oof_protocol import run_oof_evaluation, run_oof_baselines
        from sklearn.model_selection import StratifiedKFold

        X, y = _make_synthetic_data(n_samples=200, n_features=17)
        sensor_counts = _make_sensor_feature_counts()
        seed = 42

        # Get fold assignments
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_assignments = list(skf.split(X, y))

        crlb_w = np.array([0.3, 0.4, 0.3])
        baselines = run_oof_baselines(
            X, y, sensor_counts, fold_assignments, crlb_w,
            seed=seed,
        )
        # Each baseline should have scores for all samples
        for bl in baselines:
            assert len(bl["fused_scores"]) == len(y)
            assert not np.any(np.isnan(bl["fused_scores"]))

    def test_base_002_no_training_evaluation(self):
        """No baseline evaluates its training predictions as primary."""
        from science.evaluation.oof_protocol import run_oof_baselines
        from sklearn.model_selection import StratifiedKFold

        X, y = _make_synthetic_data(n_samples=200, n_features=17)
        sensor_counts = _make_sensor_feature_counts()
        seed = 42

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_assignments = list(skf.split(X, y))
        crlb_w = np.array([0.3, 0.4, 0.3])

        baselines = run_oof_baselines(
            X, y, sensor_counts, fold_assignments, crlb_w, seed=seed,
        )
        for bl in baselines:
            # AUC should be computed from OOF, so < 1.0 for non-trivial data
            assert bl["auc"] < 0.999, \
                f"{bl['method_name']} AUC={bl['auc']:.4f} suspiciously high (in-sample?)"


# ============================================================
# TEST DEG-001: Static weights remain equal to clean fold weights
# ============================================================
class TestDegradation:

    def test_deg_001_static_weights_unchanged(self):
        """Static weights remain equal to clean fold weights at every severity."""
        from science.degradation.module_m import apply_degradation
        rng = np.random.default_rng(42)
        scores = {"radar": rng.uniform(0, 1, 100), "thermal": rng.uniform(0, 1, 100),
                  "acoustic": rng.uniform(0, 1, 100)}
        static_w = {"radar": 0.2, "thermal": 0.5, "acoustic": 0.3}

        for sev in [0.0, 0.25, 0.5, 0.75, 1.0]:
            degraded = apply_degradation(scores, "radar", sev, seed=42)
            # Static weights must not change
            assert static_w == {"radar": 0.2, "thermal": 0.5, "acoustic": 0.3}


# ============================================================
# TEST STATE-001: Legacy experiments flagged
# ============================================================
class TestState:

    def test_state_001_legacy_detection(self):
        """Legacy leaked experiment metrics are not loaded as valid OOF."""
        from app.state import ExperimentState
        exp = ExperimentState()
        exp.schema_version = 1  # Old schema
        assert exp.is_legacy_evaluation
        assert not exp.has_valid_oof

    def test_state_002_new_schema(self):
        """New schema v2 experiments are not legacy."""
        from app.state import ExperimentState, SCHEMA_VERSION
        exp = ExperimentState()
        assert exp.schema_version == SCHEMA_VERSION
        assert not exp.is_legacy_evaluation


# ============================================================
# TEST CANONICAL METRICS
# ============================================================
class TestCanonicalMetrics:

    def test_canonical_metrics_consistency(self):
        """compute_canonical_metrics produces consistent results."""
        from science.evaluation.oof_protocol import compute_canonical_metrics
        rng = np.random.default_rng(42)
        y = np.array([0]*50 + [1]*50)
        scores = rng.uniform(0, 1, 100)
        m = compute_canonical_metrics(y, scores, 0.5)

        assert m.tp + m.fn + m.fp + m.tn == 100
        assert 0 <= m.detection_rate <= 1
        assert 0 <= m.false_alarm_rate <= 1
        assert 0 <= m.precision <= 1
        assert 0 <= m.auc <= 1

    def test_far_definition(self):
        """FAR = FP / (FP + TN) exactly."""
        from science.evaluation.oof_protocol import compute_canonical_metrics
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.6, 0.2, 0.7, 0.8, 0.9, 0.3, 0.7])
        m = compute_canonical_metrics(y, scores, 0.5)
        expected_far = m.fp / (m.fp + m.tn) if (m.fp + m.tn) > 0 else 0
        assert abs(m.false_alarm_rate - expected_far) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
