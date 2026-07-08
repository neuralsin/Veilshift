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
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import MetricCard, SensorCard, SectionFrame, EmptyState, ActionButton
from app.state import ExperimentState, ModuleStatus, TargetRegime


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

        # TOP BAR: Target Regime presets selection (regime-aware AUC hardening)
        top_bar = ctk.CTkFrame(container, fg_color=Colors.BG_CARD, corner_radius=Radii.DEFAULT,
                               border_color=Colors.BORDER_SUBTLE, border_width=1)
        top_bar.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        top_bar_inner = ctk.CTkFrame(top_bar, fg_color="transparent")
        top_bar_inner.pack(fill="x", padx=Spacing.LG, pady=Spacing.LG)
        
        self._regime_label = ctk.CTkLabel(
            top_bar_inner, text="OBSERVABILITY LEVEL:", 
            font=(Typography.UI_FONT, 11, "bold"),
            text_color=Colors.TEXT_SECONDARY
        )
        self._regime_label.pack(side="left", padx=(0, Spacing.SM))

        self._regime_value_label = ctk.CTkLabel(
            top_bar_inner, text="STEALTH (0%)",
            font=(Typography.UI_FONT, 16, "bold"),
            text_color=Colors.RADAR
        )
        self._regime_value_label.pack(side="left")

        # ROW 1: 4 Metric cards
        metrics_row = ctk.CTkFrame(container, fg_color="transparent")
        metrics_row.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        for i in range(4):
            metrics_row.grid_columnconfigure(i, weight=1)

        self._target_regime = MetricCard(
            metrics_row, label="Target Condition", value="—",
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

        # Modality Weights
        trust_section = SectionFrame(bottom_row, title="Mean OOF Fusion Weights")
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
            bar.set(0.0)

            val_label = ctk.CTkLabel(
                bar_frame, text="—",
                font=(Typography.MONO_FONT, 12),
                text_color=color, width=90, anchor="e",
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

        self._export_btn = ActionButton(
            actions_row, text="EXPORT PRODUCTION MODEL",
            command=self._export_model, primary=False,
        )
        self._export_btn.pack(side="left", padx=(0, Spacing.SM))

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

        # Sync regime label with current regime
        regime_map = {
            TargetRegime.STEALTH: "STEALTH (0%)",
            TargetRegime.NEAR_BACKGROUND: "NEAR-BACKGROUND (25%)",
            TargetRegime.LOW_OBSERVABILITY: "LOW OBSERVABILITY (50%)",
            TargetRegime.REDUCED_SIGNATURE: "REDUCED SIGNATURE (75%)",
            TargetRegime.CONVENTIONAL: "CONVENTIONAL (100%)",
        }
        label_text = regime_map.get(exp.target_regime, exp.target_regime.value.upper())
        self._regime_value_label.configure(text=label_text)

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

        # Fusion weights
        weights_to_show = exp.evaluation_fusion_weights or (
            exp.fusion_result.weights if exp.fusion_result.weights else None
        )
        if weights_to_show:
            for sensor, (bar, label) in self._trust_bars.items():
                w = weights_to_show.get(sensor, 0.0)
                bar.set(w)
                # Show ± SD if evaluation weights available
                if exp.evaluation_fusion_weights_std and sensor in exp.evaluation_fusion_weights_std:
                    sd = exp.evaluation_fusion_weights_std[sensor]
                    label.configure(text=f"{w:.3f} ±{sd:.3f} SD")
                else:
                    label.configure(text=f"{w:.3f}")
        else:
            for sensor, (bar, label) in self._trust_bars.items():
                bar.set(0.0)
                label.configure(text="—")

        # Update Sensor Cards
        if exp.status in [ModuleStatus.COMPLETED, ModuleStatus.FAILED]:
            self._run_btn.set_running(False)
            
        if exp.status == ModuleStatus.COMPLETED:
            from sklearn.metrics import roc_auc_score

            # Use OOF weights as the single source of truth for the dashboard
            weights = exp.evaluation_fusion_weights or exp.fusion_result.weights or {}
            
            # Sort weights to find rank
            w_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            ranks = {k: str(i+1) for i, (k, v) in enumerate(w_items)}

            def calculate_oof_auc(sensor_name):
                if exp.oof_result and exp.oof_result.sensor_scores and sensor_name in exp.oof_result.sensor_scores:
                    try:
                        return f"{roc_auc_score(exp.oof_result.true_labels, exp.oof_result.sensor_scores[sensor_name]):.3f}"
                    except ValueError:
                        return "0.500"
                # Fallback to active model probabilities if OOF is missing
                if exp.sensor_scores and sensor_name in exp.sensor_scores and exp.labels is not None:
                    try:
                        return f"{roc_auc_score(exp.labels, exp.sensor_scores[sensor_name]):.3f}"
                    except ValueError:
                        return "0.500"
                return "—"
            
            # Radar
            radar_score = calculate_oof_auc("radar")
            radar_weight = f"{weights.get('radar', 0.0):.3f}" if weights else "—"
            radar_snr = f"{exp.radar_result.snr_db:.1f}" if exp.radar_result.snr_db is not None else "—"
            self._radar_card.update_metrics(
                status="Online", score=radar_score, weight=radar_weight, 
                rank=ranks.get('radar', "—"), snr=radar_snr
            )
            
            # Thermal
            thermal_score = calculate_oof_auc("thermal")
            thermal_weight = f"{weights.get('thermal', 0.0):.3f}" if weights else "—"
            thermal_snr = f"{exp.thermal_result.thermal_snr:.1f}" if exp.thermal_result.thermal_snr is not None else "—"
            self._thermal_card.update_metrics(
                status="Online", score=thermal_score, weight=thermal_weight, 
                rank=ranks.get('thermal', "—"), snr=thermal_snr
            )
            
            # Acoustic
            acoustic_score = calculate_oof_auc("acoustic")
            acoustic_weight = f"{weights.get('acoustic', 0.0):.3f}" if weights else "—"
            acoustic_snr = f"{exp.acoustic_result.signal_excess_db:.1f}" if exp.acoustic_result.signal_excess_db is not None else "—"
            self._acoustic_card.update_metrics(
                status="Online", score=acoustic_score, weight=acoustic_weight, 
                rank=ranks.get('acoustic', "—"), snr=acoustic_snr
            )
            
            self._draw_sparklines(exp)
        else:
            self._radar_card.update_metrics(status="Pending", score="—", weight="—", rank="—", snr="—")
            self._thermal_card.update_metrics(status="Pending", score="—", weight="—", rank="—", snr="—")
            self._acoustic_card.update_metrics(status="Pending", score="—", weight="—", rank="—", snr="—")
            self._draw_sparklines(exp)

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
            if exp.evaluation_fusion_weights:
                w = exp.evaluation_fusion_weights
                ranked = sorted(w.items(), key=lambda x: x[1], reverse=True)
                lines.append(f"Fusion weight order: {ranked[0][0]} ({ranked[0][1]:.3f}) > "
                           f"{ranked[1][0]} ({ranked[1][1]:.3f}) > "
                           f"{ranked[2][0]} ({ranked[2][1]:.3f}).")
            elif exp.fusion_result.weights:
                w = exp.fusion_result.weights
                ranked = sorted(w.items(), key=lambda x: x[1], reverse=True)
                lines.append(f"Fusion weight order (Legacy): {ranked[0][0]} ({ranked[0][1]:.3f}) > "
                           f"{ranked[1][0]} ({ranked[1][1]:.3f}) > "
                           f"{ranked[2][0]} ({ranked[2][1]:.3f}).")

            if exp.radar_result.snr_db is not None:
                lines.append(f"Radar SNR: {exp.radar_result.snr_db:.1f} dB "
                           f"({'strong' if exp.radar_result.snr_db > 0 else 'weak'})")

            if exp.has_valid_oof:
                lines.append(f"Evaluation: {exp.evaluation_config.n_folds}-Fold Stratified OOF")
                lines.append(f"Fused AUC (OOF): {exp.metrics_result.auc}")
            elif exp.metrics_result.auc:
                lines.append(f"⚠ LEGACY IN-SAMPLE AUC: {exp.metrics_result.auc}")

            if hasattr(exp, 'detection_model') and exp.detection_model is not None:
                lines.append("")
                lines.append(f"--- PERSISTENT MODEL ---")
                lines.append(f"Model ID: {exp.detection_model.model_id}")
                lines.append(f"Total samples trained: {exp.detection_model.total_samples_seen}")
                incremental = sum(1 for r in exp.detection_model.training_history if r.incremental)
                lines.append(f"Updates: {len(exp.detection_model.training_history)} ({incremental} incremental)")
                lines.append(f"Saved to: {exp.detection_model_path}")

            if exp.fusion_result.solver:
                lines.append(f"Solver: {exp.fusion_result.solver}")

            self._inference_text.insert("0.0", "\n".join(lines))

        self._inference_text.configure(state="disabled")

    def _draw_sparklines(self, exp: ExperimentState):
        """Render simple matplotlib sparklines inside the sensor cards."""
        for card, sensor in [(self._radar_card, "radar"), (self._thermal_card, "thermal"), (self._acoustic_card, "acoustic")]:
            for w in card._viz_frame.winfo_children():
                w.destroy()
            
            if exp.status != ModuleStatus.COMPLETED:
                continue
                
            fig, ax = plt.subplots(figsize=(3, 1.2), facecolor=Colors.BG_DARKEST)
            ax.set_facecolor(Colors.BG_DARKEST)
            
            if sensor == "radar" and exp.radar_result.range_profile is not None:
                x = np.arange(len(exp.radar_result.range_profile))
                ax.plot(x, 20 * np.log10(exp.radar_result.range_profile + 1e-12), color=Colors.RADAR, linewidth=1)
                ax.margins(0)
            elif sensor == "thermal" and exp.thermal_result.thermal_frame is not None:
                ax.imshow(exp.thermal_result.thermal_frame, aspect="auto", cmap="inferno")
            elif sensor == "acoustic" and exp.acoustic_result.time_series is not None:
                ts = exp.acoustic_result.time_series[:2000]
                ax.plot(ts, color=Colors.ACOUSTIC, linewidth=0.5)
                ax.margins(0)
                
            ax.axis('off')
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            canvas = FigureCanvasTkAgg(fig, master=card._viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            plt.close(fig)

    def _export_model(self):
        from app.state import TaskType, TargetRegime
        from science.export.production_export import export_production_model
        
        exp = self._app.app_state.current_experiment
        regime_str = "CONVENTIONAL" if exp.target_regime == TargetRegime.CONVENTIONAL else "STEALTH"
        
        self._export_btn.set_running(True)
        
        def run_export(progress_callback, *args):
            # Run the production export module with exact parameters
            manifest = export_production_model(
                output_dir="production_export",
                regime=regime_str,
                seed=exp.seed,
                num_samples=exp.radar_config.num_samples,
                n_folds=exp.evaluation_config.n_folds,
                n_bootstrap=exp.evaluation_config.n_bootstrap,
                k_target=exp.feature_config.k_target,
                target_far=exp.fusion_config.target_far,
                progress_callback=progress_callback,
            )
            return manifest
            
        def on_export_done(result_future):
            self._export_btn.set_running(False)
            
        self._app.task_manager.submit(
            TaskType.EXPERIMENT_EXPORT,
            exp.experiment_id,
            run_export,
            callback=on_export_done,
        )

    def _run_pipeline(self):
        self._run_btn.set_running(True)
        self._app._run_pipeline()

    def _new_experiment(self):
        self._app._new_experiment()
