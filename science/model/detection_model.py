"""
QT-2.23 — Persistent Detection Model

A serializable, incrementally-trainable multisensor fusion detection model.
This is the project's core deliverable: a real model artifact that can be
saved to disk, loaded in milliseconds, updated with new data, and used
for inference on new sensor observations.

Architecture:
  - 3 per-sensor LogisticRegression classifiers (radar, thermal, acoustic)
  - QUBO-selected feature subset
  - Rayleigh-LDA fusion weights
  - Neyman-Pearson detection threshold
  - Full training history and provenance metadata

Usage:
  model = DetectionModel.load("models/qt223_model.pkl")
  prediction = model.predict(new_feature_vector)
  model.update(new_X, new_y)
  model.save("models/qt223_model.pkl")
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# Training History Entry
# ============================================================

@dataclass
class TrainingRecord:
    """One entry in the model's training history."""
    timestamp: float = 0.0
    regime: str = "UNKNOWN"
    n_samples: int = 0
    n_features: int = 0
    auc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    false_alarm_rate: float = 0.0
    fusion_weights: Optional[Dict[str, float]] = None
    threshold: float = 0.5
    seed: int = 42
    incremental: bool = False  # True if this was an update, not a fresh train


# ============================================================
# Detection Model
# ============================================================

class DetectionModel:
    """
    Persistent multisensor fusion detection model.

    This wraps the full detection pipeline into a single serializable object:
      - Per-sensor classifiers (LogisticRegression with warm_start for incremental learning)
      - Feature scaler (StandardScaler for numerical stability)
      - QUBO-selected feature indices
      - Rayleigh-LDA fusion weights
      - Neyman-Pearson threshold
      - Full training provenance

    The model supports:
      - save(path) / load(path): pickle serialization
      - predict(X): full inference (features → sensor scores → fusion → detection)
      - predict_proba(X): probability scores before thresholding
      - update(X, y): incremental training with new data (warm_start)
      - export_manifest(): JSON-serializable summary for documentation
    """

    VERSION = "2.0"

    def __init__(self):
        # Per-sensor classifiers
        self.sensor_classifiers: Dict[str, LogisticRegression] = {}
        self.sensor_names: List[str] = ["radar", "thermal", "acoustic"]

        # Feature configuration
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.selected_feature_indices: Optional[List[int]] = None
        self.sensor_feature_slices: Dict[str, slice] = {}

        # Fusion parameters
        self.fusion_weights: Optional[np.ndarray] = None
        self.fusion_weights_dict: Optional[Dict[str, float]] = None
        self.threshold: float = 0.5
        self.fisher_objective: float = 0.0

        # Training state
        self.is_trained: bool = False
        self.total_samples_seen: int = 0
        self.training_history: List[TrainingRecord] = []

        # Metadata
        self.created_at: float = time.time()
        self.last_updated_at: float = time.time()
        self.regime: str = "UNKNOWN"
        self.model_id: str = ""

    # ============================================================
    # Training
    # ============================================================

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        sensor_feature_counts: Dict[str, int],
        fusion_weights: np.ndarray,
        threshold: float,
        regime: str = "UNKNOWN",
        seed: int = 42,
        selected_indices: Optional[List[int]] = None,
        fisher_objective: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Train the detection model from scratch on the provided data.

        Parameters
        ----------
        X : (N, F) feature matrix
        y : (N,) binary labels
        feature_names : list of feature names
        sensor_feature_counts : {"radar": n_r, "thermal": n_t, "acoustic": n_a}
        fusion_weights : (3,) array of sensor fusion weights
        threshold : Neyman-Pearson detection threshold
        regime : target regime string
        seed : random seed
        selected_indices : QUBO-selected feature indices (optional)
        fisher_objective : Rayleigh quotient value
        metrics : dict with auc, f1, precision, recall, far (optional)

        Returns
        -------
        Dict with training summary
        """
        self.feature_names = feature_names
        self.selected_feature_indices = selected_indices
        self.regime = regime
        self.fisher_objective = fisher_objective

        # Build sensor slices
        offset = 0
        for s in self.sensor_names:
            n_feat = sensor_feature_counts.get(s, 0)
            self.sensor_feature_slices[s] = slice(offset, offset + n_feat)
            offset += n_feat

        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train per-sensor classifiers
        self.sensor_classifiers = {}
        sensor_aucs = {}
        for s in self.sensor_names:
            sl = self.sensor_feature_slices[s]
            clf = LogisticRegression(
                max_iter=2000, random_state=seed, warm_start=True,
                C=1.0, solver='lbfgs',
            )
            clf.fit(X_scaled[:, sl], y)
            self.sensor_classifiers[s] = clf

            # Per-sensor AUC
            try:
                scores = clf.predict_proba(X_scaled[:, sl])[:, 1]
                sensor_aucs[s] = float(roc_auc_score(y, scores))
            except Exception:
                sensor_aucs[s] = 0.5

        # Store fusion parameters
        self.fusion_weights = fusion_weights.copy()
        self.fusion_weights_dict = {
            s: float(fusion_weights[i]) for i, s in enumerate(self.sensor_names)
        }
        self.threshold = threshold

        # Update state
        self.is_trained = True
        self.total_samples_seen = len(y)
        self.last_updated_at = time.time()
        self.model_id = f"QT223-{int(self.created_at)}"

        # Record training
        record = TrainingRecord(
            timestamp=time.time(),
            regime=regime,
            n_samples=len(y),
            n_features=X.shape[1],
            auc=metrics.get("auc", 0.0) if metrics else 0.0,
            f1=metrics.get("f1", 0.0) if metrics else 0.0,
            precision=metrics.get("precision", 0.0) if metrics else 0.0,
            recall=metrics.get("recall", 0.0) if metrics else 0.0,
            false_alarm_rate=metrics.get("far", 0.0) if metrics else 0.0,
            fusion_weights=self.fusion_weights_dict.copy(),
            threshold=threshold,
            seed=seed,
            incremental=False,
        )
        self.training_history.append(record)

        return {
            "model_id": self.model_id,
            "n_samples": len(y),
            "sensor_aucs": sensor_aucs,
            "fusion_weights": self.fusion_weights_dict,
            "threshold": threshold,
            "fisher_objective": fisher_objective,
        }

    # ============================================================
    # Incremental Update
    # ============================================================

    def update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        regime: str = "UNKNOWN",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Incrementally update the model with new data.

        Uses warm_start=True on the existing classifiers, so the model
        starts from its previous weights and refines them with new data.
        The scaler is updated with partial_fit.

        Parameters
        ----------
        X_new : (N, F) new feature matrix
        y_new : (N,) new binary labels

        Returns
        -------
        Dict with update summary
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before updating. Call train() first.")

        # Update scaler
        self.scaler.partial_fit(X_new)
        X_scaled = self.scaler.transform(X_new)

        # Incrementally train each sensor classifier (warm_start continues from previous weights)
        sensor_aucs = {}
        for s in self.sensor_names:
            sl = self.sensor_feature_slices[s]
            clf = self.sensor_classifiers[s]
            clf.fit(X_scaled[:, sl], y_new)  # warm_start=True means it starts from previous coefs

            try:
                scores = clf.predict_proba(X_scaled[:, sl])[:, 1]
                sensor_aucs[s] = float(roc_auc_score(y_new, scores))
            except Exception:
                sensor_aucs[s] = 0.5

        # Update state
        self.total_samples_seen += len(y_new)
        self.last_updated_at = time.time()

        # Compute fused AUC on new data
        fused_scores = self._compute_fused_scores(X_scaled)
        try:
            fused_auc = float(roc_auc_score(y_new, fused_scores))
        except Exception:
            fused_auc = 0.5

        # Record
        record = TrainingRecord(
            timestamp=time.time(),
            regime=regime,
            n_samples=len(y_new),
            n_features=X_new.shape[1],
            auc=fused_auc,
            fusion_weights=self.fusion_weights_dict.copy() if self.fusion_weights_dict else None,
            threshold=self.threshold,
            seed=seed,
            incremental=True,
        )
        self.training_history.append(record)

        return {
            "model_id": self.model_id,
            "n_new_samples": len(y_new),
            "total_samples_seen": self.total_samples_seen,
            "sensor_aucs": sensor_aucs,
            "fused_auc": fused_auc,
            "n_updates": sum(1 for r in self.training_history if r.incremental),
        }

    # ============================================================
    # Inference
    # ============================================================

    def _compute_fused_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute fused detection scores from scaled features."""
        sensor_scores = {}
        for s in self.sensor_names:
            sl = self.sensor_feature_slices[s]
            clf = self.sensor_classifiers[s]
            sensor_scores[s] = clf.predict_proba(X_scaled[:, sl])[:, 1]

        fused = np.zeros(X_scaled.shape[0])
        for i, s in enumerate(self.sensor_names):
            fused += self.fusion_weights[i] * sensor_scores[s]

        return fused

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute fused detection probability scores.

        Parameters
        ----------
        X : (N, F) raw feature matrix (will be scaled internally)

        Returns
        -------
        (N,) fused probability scores
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        X_scaled = self.scaler.transform(X)
        return self._compute_fused_scores(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Binary detection prediction.

        Parameters
        ----------
        X : (N, F) raw feature matrix

        Returns
        -------
        (N,) binary predictions (0 = no target, 1 = target detected)
        """
        scores = self.predict_proba(X)
        return (scores >= self.threshold).astype(int)

    def predict_detailed(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Full inference with per-sensor breakdown.

        Returns dict with fused_scores, predictions, and per-sensor scores.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        X_scaled = self.scaler.transform(X)

        sensor_scores = {}
        for s in self.sensor_names:
            sl = self.sensor_feature_slices[s]
            sensor_scores[s] = self.sensor_classifiers[s].predict_proba(X_scaled[:, sl])[:, 1]

        fused = np.zeros(X_scaled.shape[0])
        for i, s in enumerate(self.sensor_names):
            fused += self.fusion_weights[i] * sensor_scores[s]

        predictions = (fused >= self.threshold).astype(int)

        return {
            "fused_scores": fused,
            "predictions": predictions,
            "sensor_scores": sensor_scores,
            "fusion_weights": self.fusion_weights_dict,
            "threshold": self.threshold,
        }

    # ============================================================
    # Serialization
    # ============================================================

    def save(self, path: str) -> str:
        """
        Save the complete model to disk as a pickle file.

        Also writes a companion JSON manifest for human-readable inspection.

        Parameters
        ----------
        path : file path (e.g., "models/qt223_model.pkl")

        Returns
        -------
        Absolute path to saved file
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # Save pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save companion JSON manifest
        manifest_path = path.replace('.pkl', '_manifest.json')
        manifest = self.export_manifest()
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return os.path.abspath(path)

    @classmethod
    def load(cls, path: str) -> 'DetectionModel':
        """
        Load a saved model from disk.

        Parameters
        ----------
        path : file path to .pkl file

        Returns
        -------
        DetectionModel instance
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)

        if not isinstance(model, cls):
            raise TypeError(f"Expected DetectionModel, got {type(model)}")

        return model

    def export_manifest(self) -> Dict[str, Any]:
        """
        Export a JSON-serializable manifest describing the model.

        This is the human-readable companion to the pickle file.
        """
        history_records = []
        for r in self.training_history:
            history_records.append({
                "timestamp": r.timestamp,
                "regime": r.regime,
                "n_samples": r.n_samples,
                "auc": r.auc,
                "f1": r.f1,
                "precision": r.precision,
                "recall": r.recall,
                "false_alarm_rate": r.false_alarm_rate,
                "fusion_weights": r.fusion_weights,
                "threshold": r.threshold,
                "incremental": r.incremental,
            })

        return {
            "model_id": self.model_id,
            "version": self.VERSION,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
            "is_trained": self.is_trained,
            "total_samples_seen": self.total_samples_seen,
            "regime": self.regime,
            "sensor_names": self.sensor_names,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "selected_feature_indices": self.selected_feature_indices,
            "fusion_weights": self.fusion_weights_dict,
            "threshold": self.threshold,
            "fisher_objective": self.fisher_objective,
            "n_training_runs": len(self.training_history),
            "n_incremental_updates": sum(1 for r in self.training_history if r.incremental),
            "training_history": history_records,
            "sensor_classifiers": {
                s: {
                    "type": type(clf).__name__,
                    "n_coefs": clf.coef_.shape[1] if hasattr(clf, 'coef_') else 0,
                    "intercept": float(clf.intercept_[0]) if hasattr(clf, 'intercept_') else 0.0,
                }
                for s, clf in self.sensor_classifiers.items()
            } if self.sensor_classifiers else {},
        }

    def summary(self) -> str:
        """Human-readable model summary string."""
        if not self.is_trained:
            return "DetectionModel [UNTRAINED]"

        lines = [
            f"DetectionModel {self.model_id}",
            f"  Regime: {self.regime}",
            f"  Trained on: {self.total_samples_seen} samples",
            f"  Features: {len(self.feature_names) if self.feature_names else 0}",
            f"  Fusion weights: {self.fusion_weights_dict}",
            f"  Threshold: {self.threshold:.4f}",
            f"  Fisher J': {self.fisher_objective:.4f}",
            f"  Training runs: {len(self.training_history)}",
            f"  Incremental updates: {sum(1 for r in self.training_history if r.incremental)}",
        ]

        if self.training_history:
            last = self.training_history[-1]
            lines.append(f"  Last AUC: {last.auc:.4f}")
            lines.append(f"  Last F1: {last.f1:.4f}")

        return "\n".join(lines)
