"""
QT-2.23 — Mission Control Page

Landing page per Stitch screen "Mission Control - Quantum Sensor Fusion (Revised)".
Layout: 4 metric cards → 3 sensor cards → Fusion node → Trust distribution → Inference.

Scientific question: "What is each sensor seeing and what does fusion currently trust?"
"""

from __future__ import annotations
from typing import Any, Optional

import customtkinter as ctk
import numpy as np

from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import MetricCard, SensorCard, SectionFrame, EmptyState, ActionButton
from app.state import ExperimentState, ModuleStatus


class MissionControlPage(ctk.CTkScrollableFrame):
    """Mission Control — landing page."""

    def __init__(self, master, app_shell):
        super().__init__(
            master, fg_color=Colors.BG_DARKEST,
            scrollbar_button_color=Colors.BORDER_SUBTLE,
        )
        self._app = app_shell
        self._build_ui()

    def _build_ui(self):
        """Build the Mission Control layout."""
        self.grid_columnconfigure(0, weight=1)

        # Container for max-width content
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=Spacing.PAGE_PADDING, pady=Spacing.PAGE_PADDING)
        container.grid_columnconfigure(0, weight=1)

        # ROW 1: 4 Metric cards
        metrics_row = ctk.CTkFrame(container, fg_color="transparent")
        metrics_row.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        for i in range(4):
            metrics_row.grid_columnconfigure(i, weight=1)

        self._target_regime = MetricCard(
            metrics_row, label="Target Regime", value="STEALTH",
            accent_color=Colors.TEXT_SECONDARY,
        )
        self._target_regime.grid(row=0, column=0, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._modalities = MetricCard(
            metrics_row, label="Active Modalities", value="0 / 3",
        )
        self._modalities.grid(row=0, column=1, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._fused_auc = MetricCard(
            metrics_row, label="Fused AUC", value="—", ci_text="",
        )
        self._fused_auc.grid(row=0, column=2, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._far = MetricCard(
            metrics_row, label="False Alarm Rate", value="—",
        )
        self._far.grid(row=0, column=3, sticky="nsew")

        # ROW 2: 3 Sensor evidence cards
        sensors_row = ctk.CTkFrame(container, fg_color="transparent")
        sensors_row.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        for i in range(3):
            sensors_row.grid_columnconfigure(i, weight=1)

        self._radar_card = SensorCard(
            sensors_row, sensor_name="Radar", sensor_color=Colors.RADAR,
            status="Pending",
        )
        self._radar_card.grid(row=0, column=0, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._thermal_card = SensorCard(
            sensors_row, sensor_name="Thermal / IR", sensor_color=Colors.THERMAL,
            status="Pending",
        )
        self._thermal_card.grid(row=0, column=1, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._acoustic_card = SensorCard(
            sensors_row, sensor_name="Acoustic / Sonar", sensor_color=Colors.ACOUSTIC,
            status="Pending",
        )
        self._acoustic_card.grid(row=0, column=2, sticky="nsew")

        # ROW 3: Trust Distribution + System Inference
        bottom_row = ctk.CTkFrame(container, fg_color="transparent")
        bottom_row.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        bottom_row.grid_columnconfigure(0, weight=2)
        bottom_row.grid_columnconfigure(1, weight=1)

        # Trust Distribution
        trust_section = SectionFrame(bottom_row, title="Trust Distribution")
        trust_section.grid(row=0, column=0, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._trust_bars = {}
        for sensor, color in [("Radar", Colors.RADAR), ("Thermal", Colors.THERMAL), ("Acoustic", Colors.ACOUSTIC)]:
            bar_frame = ctk.CTkFrame(trust_section.content, fg_color="transparent")
            bar_frame.pack(fill="x", pady=Spacing.XS)

            ctk.CTkLabel(
                bar_frame, text=sensor.upper(),
                font=(Typography.UI_FONT, 11, "bold"),
                text_color=Colors.TEXT_SECONDARY, width=80, anchor="w",
            ).pack(side="left")

            bar = ctk.CTkProgressBar(
                bar_frame, fg_color=Colors.BG_DARKEST,
                progress_color=color, height=12, corner_radius=3, width=200,
            )
            bar.pack(side="left", fill="x", expand=True, padx=Spacing.SM)
            bar.set(0.33)

            val_label = ctk.CTkLabel(
                bar_frame, text="0.333",
                font=(Typography.MONO_FONT, 12),
                text_color=color, width=50, anchor="e",
            )
            val_label.pack(side="right")

            self._trust_bars[sensor.lower()] = (bar, val_label)

        # System Inference
        inference_section = SectionFrame(bottom_row, title="System Inference")
        inference_section.grid(row=0, column=1, sticky="nsew")

        self._inference_text = ctk.CTkTextbox(
            inference_section.content,
            font=(Typography.UI_FONT, 12),
            text_color=Colors.TEXT_SECONDARY,
            fg_color="transparent",
            height=100,
            wrap="word",
            state="disabled",
        )
        self._inference_text.pack(fill="both", expand=True)

        # ROW 4: Action buttons
        actions_row = ctk.CTkFrame(container, fg_color="transparent")
        actions_row.pack(fill="x", pady=(0, Spacing.GRID_GAP))

        self._run_btn = ActionButton(
            actions_row, text="RUN FULL PIPELINE",
            command=self._run_pipeline, primary=True,
        )
        self._run_btn.pack(side="left", padx=(0, Spacing.SM))

        ActionButton(
            actions_row, text="NEW EXPERIMENT",
            command=self._new_experiment, primary=False,
        ).pack(side="left", padx=(0, Spacing.SM))

        # Empty state overlay
        self._empty_state = EmptyState(
            container,
            title="NO ACTIVE EXPERIMENT",
            message="Configure a scenario or load a previous experiment to begin.",
            buttons=[
                ("RUN FULL PIPELINE", self._run_pipeline),
            ],
        )

    def refresh(self, exp: ExperimentState):
        """Refresh all Mission Control data from experiment state."""
        # Target regime
        self._target_regime.update_value(exp.target_regime.value.upper())

        # Active modalities
        active = exp.active_modalities
        self._modalities.update_value(f"{active} / 3")

        # Fused AUC
        if exp.metrics_result.auc:
            auc = exp.metrics_result.auc
            self._fused_auc.update_value(
                f"{auc.point_estimate:.3f}",
                f"[95% CI {auc.ci_lower:.3f}–{auc.ci_upper:.3f}]",
            )

        # FAR
        if exp.metrics_result.false_alarm_rate:
            far = exp.metrics_result.false_alarm_rate.point_estimate
            self._far.update_value(f"{far*100:.2f}%")

        # Trust distribution — prefer OOF evaluation weights
        weights_to_show = exp.evaluation_fusion_weights or (
            exp.fusion_result.weights if exp.fusion_result.weights else None
        )
        if weights_to_show:
            for sensor, (bar, label) in self._trust_bars.items():
                w = weights_to_show.get(sensor, 0.333)
                bar.set(w)
                # Show ± SD if evaluation weights available
                if exp.evaluation_fusion_weights_std and sensor in exp.evaluation_fusion_weights_std:
                    sd = exp.evaluation_fusion_weights_std[sensor]
                    label.configure(text=f"{w:.3f}±{sd:.3f}")
                else:
                    label.configure(text=f"{w:.3f}")

        # Inference text
        self._update_inference(exp)

    def _update_inference(self, exp: ExperimentState):
        """Build deterministic inference text from actual data."""
        self._inference_text.configure(state="normal")
        self._inference_text.delete("0.0", "end")

        if exp.status != ModuleStatus.COMPLETED:
            self._inference_text.insert("0.0", "Run the pipeline to generate system inference.")
        else:
            lines = []
            if exp.fusion_result.weights:
                w = exp.fusion_result.weights
                ranked = sorted(w.items(), key=lambda x: x[1], reverse=True)
                lines.append(f"Sensor trust ranking: {ranked[0][0]} ({ranked[0][1]:.3f}) > "
                           f"{ranked[1][0]} ({ranked[1][1]:.3f}) > "
                           f"{ranked[2][0]} ({ranked[2][1]:.3f}).")

            if exp.radar_result.snr_db is not None:
                lines.append(f"Radar SNR: {exp.radar_result.snr_db:.1f} dB "
                           f"({'strong' if exp.radar_result.snr_db > 0 else 'weak'})")

            if exp.has_valid_oof:
                lines.append(f"Evaluation: 5-Fold Stratified OOF")
                lines.append(f"Fused AUC (OOF): {exp.metrics_result.auc}")
            elif exp.metrics_result.auc:
                lines.append(f"⚠ LEGACY IN-SAMPLE AUC: {exp.metrics_result.auc}")

            if exp.fusion_result.solver:
                lines.append(f"Solver: {exp.fusion_result.solver}")

            self._inference_text.insert("0.0", "\n".join(lines))

        self._inference_text.configure(state="disabled")

    def _run_pipeline(self):
        self._run_btn.set_running(True)
        self._app._run_pipeline()

    def _new_experiment(self):
        self._app._new_experiment()
