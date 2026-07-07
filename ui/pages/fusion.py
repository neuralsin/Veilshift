"""QT-2.23 — Fusion Optimization Page"""
from __future__ import annotations
import customtkinter as ctk
import numpy as np
from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import SectionFrame, MetricCard, ActionButton, EmptyState
from app.state import ExperimentState, ModuleStatus


class FusionPage(ctk.CTkScrollableFrame):
    def __init__(self, master, app_shell):
        super().__init__(master, fg_color=Colors.BG_DARKEST, scrollbar_button_color=Colors.BORDER_SUBTLE)
        self._app = app_shell
        self._build_ui()

    def _build_ui(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=Spacing.PAGE_PADDING, pady=Spacing.PAGE_PADDING)

        # Metrics row
        metrics = ctk.CTkFrame(container, fg_color="transparent")
        metrics.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        for i in range(4): metrics.grid_columnconfigure(i, weight=1)

        self._objective = MetricCard(metrics, label="Fisher Objective J'(w)", value="—")
        self._objective.grid(row=0, column=0, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._det_rate = MetricCard(metrics, label="Detection Rate", value="—", accent_color=Colors.SUCCESS)
        self._det_rate.grid(row=0, column=1, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._far_card = MetricCard(metrics, label="False Alarm Rate", value="—")
        self._far_card.grid(row=0, column=2, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._solver_label = MetricCard(metrics, label="Solver", value="—")
        self._solver_label.grid(row=0, column=3, sticky="nsew")

        # Weight bars
        weights_section = SectionFrame(container, title="Optimized Fusion Weights")
        weights_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))

        self._weight_bars = {}
        for sensor, color in [("radar", Colors.RADAR), ("thermal", Colors.THERMAL), ("acoustic", Colors.ACOUSTIC)]:
            bar_frame = ctk.CTkFrame(weights_section.content, fg_color="transparent")
            bar_frame.pack(fill="x", pady=Spacing.XS)

            ctk.CTkLabel(bar_frame, text=sensor.upper(), font=(Typography.UI_FONT, 12, "bold"),
                         text_color=color, width=100, anchor="w").pack(side="left")

            bar = ctk.CTkProgressBar(bar_frame, fg_color=Colors.BG_DARKEST, progress_color=color,
                                      height=16, corner_radius=4, width=300)
            bar.pack(side="left", fill="x", expand=True, padx=Spacing.SM)
            bar.set(0.33)

            val = ctk.CTkLabel(bar_frame, text="0.333", font=(Typography.MONO_FONT, 14, "bold"),
                               text_color=color, width=60, anchor="e")
            val.pack(side="right")

            self._weight_bars[sensor] = (bar, val)

        # Charts row
        charts = ctk.CTkFrame(container, fg_color="transparent")
        charts.pack(fill="both", expand=True, pady=(0, Spacing.GRID_GAP))
        charts.grid_columnconfigure(0, weight=1)
        charts.grid_columnconfigure(1, weight=1)

        # ROC
        self._roc_section = SectionFrame(charts, title="ROC Curve")
        self._roc_section.grid(row=0, column=0, sticky="nsew", padx=(0, Spacing.GRID_GAP))
        self._roc_chart = ctk.CTkFrame(self._roc_section.content, fg_color=Colors.BG_DARKEST, height=300)
        self._roc_chart.pack(fill="both", expand=True)

        # Score distributions
        self._dist_section = SectionFrame(charts, title="Fused Detection Score Distributions (H₀ vs H₁)")
        self._dist_section.grid(row=0, column=1, sticky="nsew")
        self._dist_chart = ctk.CTkFrame(self._dist_section.content, fg_color=Colors.BG_DARKEST, height=300)
        self._dist_chart.pack(fill="both", expand=True)

    def refresh(self, exp: ExperimentState):
        fr = exp.fusion_result
        mr = exp.metrics_result
        if fr.status != ModuleStatus.COMPLETED:
            return

        self._objective.update_value(f"{fr.fisher_objective:.4f}" if fr.fisher_objective else "—")
        self._solver_label.update_value(fr.solver or "—")

        if mr.detection_rate:
            self._det_rate.update_value(f"{mr.detection_rate.point_estimate*100:.1f}%")
        if mr.false_alarm_rate:
            self._far_card.update_value(f"{mr.false_alarm_rate.point_estimate*100:.2f}%")

        # Prefer OOF evaluation weights for display
        eval_w = exp.evaluation_fusion_weights
        eval_sd = exp.evaluation_fusion_weights_std
        if eval_w:
            for sensor, (bar, val) in self._weight_bars.items():
                w = eval_w.get(sensor, 0.333)
                bar.set(w)
                if eval_sd and sensor in eval_sd:
                    val.configure(text=f"{w:.3f} ±{eval_sd[sensor]:.3f} SD")
                else:
                    val.configure(text=f"{w:.3f}")
        elif fr.weights:
            for sensor, (bar, val) in self._weight_bars.items():
                w = fr.weights.get(sensor, 0.333)
                bar.set(w)
                val.configure(text=f"{w:.3f}")

        self._render_charts(exp)

    def _render_charts(self, exp):
        mr = exp.metrics_result
        fr = exp.fusion_result
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from ui.theme.tokens import ChartStyle

            # ROC curve
            if mr.roc_fpr is not None:
                for w in self._roc_chart.winfo_children(): w.destroy()
                fig, ax = plt.subplots(figsize=(5, 4), facecolor=ChartStyle.FIGURE_FACECOLOR)
                ax.set_facecolor(ChartStyle.AXES_FACECOLOR)
                ax.plot(mr.roc_fpr, mr.roc_tpr, color=Colors.RADAR, linewidth=2,
                        label=f"Fused (AUC={mr.auc.point_estimate:.3f})" if mr.auc else "Fused")
                ax.plot([0, 1], [0, 1], ":", color=ChartStyle.GRID_COLOR, linewidth=1, label="Random")
                if fr.threshold:
                    ax.axhline(y=mr.detection_rate.point_estimate if mr.detection_rate else 0,
                              color=Colors.SUCCESS, linewidth=0.8, linestyle="--", alpha=0.6, label="Operating point")
                ax.set_xlabel("False Positive Rate", color=ChartStyle.LABEL_COLOR)
                ax.set_ylabel("True Positive Rate", color=ChartStyle.LABEL_COLOR)
                ax.set_title("Receiver Operating Characteristic", color=ChartStyle.TEXT_COLOR)
                ax.legend(fontsize=8, facecolor=ChartStyle.AXES_FACECOLOR, edgecolor=ChartStyle.GRID_COLOR,
                          labelcolor=ChartStyle.TEXT_COLOR)
                ax.grid(True, color=ChartStyle.GRID_COLOR, alpha=ChartStyle.GRID_ALPHA)
                ax.tick_params(colors=ChartStyle.TICK_COLOR)
                for s in ax.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
                fig.tight_layout()
                canvas = FigureCanvasTkAgg(fig, master=self._roc_chart)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)

            # Score distributions
            if fr.fused_scores_h0 is not None:
                for w in self._dist_chart.winfo_children(): w.destroy()
                fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor=ChartStyle.FIGURE_FACECOLOR)
                ax2.set_facecolor(ChartStyle.AXES_FACECOLOR)
                ax2.hist(fr.fused_scores_h0, bins=50, alpha=0.6, color=Colors.TEXT_MUTED, label="H₀ (no target)", density=True)
                ax2.hist(fr.fused_scores_h1, bins=50, alpha=0.6, color=Colors.RADAR, label="H₁ (target)", density=True)
                if fr.threshold:
                    ax2.axvline(x=fr.threshold, color=Colors.CRITICAL, linewidth=1.5, linestyle="--", label=f"τ = {fr.threshold:.3f}")
                ax2.set_xlabel("Fused Detection Score", color=ChartStyle.LABEL_COLOR)
                ax2.set_ylabel("Density", color=ChartStyle.LABEL_COLOR)
                ax2.set_title("H₀ vs H₁ Fused Detection Score Distributions", color=ChartStyle.TEXT_COLOR)
                ax2.legend(fontsize=8, facecolor=ChartStyle.AXES_FACECOLOR, edgecolor=ChartStyle.GRID_COLOR,
                           labelcolor=ChartStyle.TEXT_COLOR)
                ax2.tick_params(colors=ChartStyle.TICK_COLOR)
                for s in ax2.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
                fig2.tight_layout()
                canvas2 = FigureCanvasTkAgg(fig2, master=self._dist_chart)
                canvas2.draw()
                canvas2.get_tk_widget().pack(fill="both", expand=True)
        except ImportError:
            pass
