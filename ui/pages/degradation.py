"""QT-2.23 — Degradation Lab Page (the hero demonstration)"""
from __future__ import annotations
import customtkinter as ctk
import numpy as np
from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import SectionFrame, MetricCard, ActionButton, EmptyState, ParameterField
from app.state import ExperimentState, ModuleStatus


class DegradationPage(ctk.CTkScrollableFrame):
    def __init__(self, master, app_shell):
        super().__init__(master, fg_color=Colors.BG_DARKEST, scrollbar_button_color=Colors.BORDER_SUBTLE)
        self._app = app_shell
        self._build_ui()

    def _build_ui(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=Spacing.PAGE_PADDING, pady=Spacing.PAGE_PADDING)

        # Controls
        controls = ctk.CTkFrame(container, fg_color=Colors.BG_CARD, corner_radius=Radii.DEFAULT,
                                border_color=Colors.BORDER_SUBTLE, border_width=1)
        controls.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        controls_inner = ctk.CTkFrame(controls, fg_color="transparent")
        controls_inner.pack(fill="x", padx=Spacing.LG, pady=Spacing.LG)

        ctk.CTkLabel(controls_inner, text="TARGET SENSOR:", font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(side="left", padx=(0, Spacing.SM))

        self._sensor_select = ctk.CTkOptionMenu(
            controls_inner, values=["radar", "thermal", "acoustic"],
            font=(Typography.UI_FONT, 11), fg_color=Colors.BG_DARKEST, button_color=Colors.BG_ELEVATED,
            dropdown_fg_color=Colors.BG_CARD, text_color=Colors.TEXT_PRIMARY, width=120, height=Spacing.INPUT_HEIGHT,
        )
        self._sensor_select.pack(side="left", padx=(0, Spacing.LG))
        self._sensor_select.set("radar")

        ctk.CTkLabel(controls_inner, text="TYPE:", font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(side="left", padx=(0, Spacing.SM))

        self._type_select = ctk.CTkOptionMenu(
            controls_inner, values=["noise_injection", "signal_suppression", "sensor_removal"],
            font=(Typography.UI_FONT, 11), fg_color=Colors.BG_DARKEST, button_color=Colors.BG_ELEVATED,
            dropdown_fg_color=Colors.BG_CARD, text_color=Colors.TEXT_PRIMARY, width=150, height=Spacing.INPUT_HEIGHT,
        )
        self._type_select.pack(side="left", padx=(0, Spacing.LG))

        ctk.CTkLabel(controls_inner, text="STEPS:", font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(side="left", padx=(0, Spacing.SM))

        self._steps = ParameterField(controls_inner, label="", default="5")
        self._steps.pack(side="left", padx=(0, Spacing.LG))

        self._run_btn = ActionButton(controls_inner, text="RUN SWEEP", command=self._run_sweep, primary=True)
        self._run_btn.pack(side="right")

        # Charts row
        charts = ctk.CTkFrame(container, fg_color="transparent")
        charts.pack(fill="both", expand=True, pady=(0, Spacing.GRID_GAP))
        charts.grid_columnconfigure(0, weight=1)
        charts.grid_columnconfigure(1, weight=1)

        # Detection retention
        self._retention_section = SectionFrame(charts, title="Detection Retention: Static vs Adaptive")
        self._retention_section.grid(row=0, column=0, sticky="nsew", padx=(0, Spacing.GRID_GAP))
        self._retention_chart = ctk.CTkFrame(self._retention_section.content, fg_color=Colors.BG_DARKEST, height=350)
        self._retention_chart.pack(fill="both", expand=True)

        # Trust migration
        self._trust_section = SectionFrame(charts, title="Trust Weight Migration")
        self._trust_section.grid(row=0, column=1, sticky="nsew")
        self._trust_chart = ctk.CTkFrame(self._trust_section.content, fg_color=Colors.BG_DARKEST, height=350)
        self._trust_chart.pack(fill="both", expand=True)

        # AUC comparison
        self._auc_section = SectionFrame(container, title="AUC Trajectory")
        self._auc_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        self._auc_chart = ctk.CTkFrame(self._auc_section.content, fg_color=Colors.BG_DARKEST, height=300)
        self._auc_chart.pack(fill="both", expand=True)

    def refresh(self, exp: ExperimentState):
        dr = exp.degradation_result
        if dr.status != ModuleStatus.COMPLETED:
            return
        self._run_btn.set_running(False)
        self._render_charts(exp)

    def _render_charts(self, exp):
        dr = exp.degradation_result
        if dr.severity_values is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from ui.theme.tokens import ChartStyle

            # Detection retention
            for w in self._retention_chart.winfo_children(): w.destroy()
            fig, ax = plt.subplots(figsize=(6, 4), facecolor=ChartStyle.FIGURE_FACECOLOR)
            ax.set_facecolor(ChartStyle.AXES_FACECOLOR)
            ax.plot(dr.severity_values, dr.static_detection_retention, 'o-',
                    color=Colors.TEXT_MUTED, linewidth=2, markersize=5, label="Static Fusion")
            ax.plot(dr.severity_values, dr.adaptive_detection_retention, 's-',
                    color=Colors.RADAR, linewidth=2, markersize=5, label="Adaptive Fusion")
            ax.fill_between(dr.severity_values, dr.static_detection_retention, dr.adaptive_detection_retention,
                           alpha=0.15, color=Colors.RADAR)
            ax.set_xlabel("Degradation Severity", color=ChartStyle.LABEL_COLOR)
            ax.set_ylabel("Detection Rate", color=ChartStyle.LABEL_COLOR)
            ax.set_title(f"Detection Retention · {dr.sensor} degradation", color=ChartStyle.TEXT_COLOR)
            ax.legend(fontsize=9, facecolor=ChartStyle.AXES_FACECOLOR, edgecolor=ChartStyle.GRID_COLOR,
                      labelcolor=ChartStyle.TEXT_COLOR)
            ax.grid(True, color=ChartStyle.GRID_COLOR, alpha=ChartStyle.GRID_ALPHA)
            ax.tick_params(colors=ChartStyle.TICK_COLOR)
            for s in ax.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self._retention_chart)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            # Trust migration
            if dr.weights_by_severity:
                for w in self._trust_chart.winfo_children(): w.destroy()
                fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor=ChartStyle.FIGURE_FACECOLOR)
                ax2.set_facecolor(ChartStyle.AXES_FACECOLOR)
                sensor_colors = {"radar": Colors.RADAR, "thermal": Colors.THERMAL, "acoustic": Colors.ACOUSTIC}
                for sensor, weights in dr.weights_by_severity.items():
                    ax2.plot(dr.severity_values, weights, 'o-', color=sensor_colors.get(sensor, Colors.TEXT_MUTED),
                            linewidth=2, markersize=5, label=sensor.title())
                ax2.set_xlabel("Degradation Severity", color=ChartStyle.LABEL_COLOR)
                ax2.set_ylabel("Trust Weight", color=ChartStyle.LABEL_COLOR)
                ax2.set_title("Trust Weight Migration", color=ChartStyle.TEXT_COLOR)
                ax2.legend(fontsize=9, facecolor=ChartStyle.AXES_FACECOLOR, edgecolor=ChartStyle.GRID_COLOR,
                           labelcolor=ChartStyle.TEXT_COLOR)
                ax2.grid(True, color=ChartStyle.GRID_COLOR, alpha=ChartStyle.GRID_ALPHA)
                ax2.tick_params(colors=ChartStyle.TICK_COLOR)
                for s in ax2.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
                fig2.tight_layout()
                canvas2 = FigureCanvasTkAgg(fig2, master=self._trust_chart)
                canvas2.draw()
                canvas2.get_tk_widget().pack(fill="both", expand=True)

            # AUC trajectory
            if dr.static_auc is not None:
                for w in self._auc_chart.winfo_children(): w.destroy()
                fig3, ax3 = plt.subplots(figsize=(10, 3), facecolor=ChartStyle.FIGURE_FACECOLOR)
                ax3.set_facecolor(ChartStyle.AXES_FACECOLOR)
                ax3.plot(dr.severity_values, dr.static_auc, 'o--', color=Colors.TEXT_MUTED,
                        linewidth=1.5, label="Static AUC")
                ax3.plot(dr.severity_values, dr.adaptive_auc, 's-', color=Colors.RADAR,
                        linewidth=2, label="Adaptive AUC")
                ax3.set_xlabel("Degradation Severity", color=ChartStyle.LABEL_COLOR)
                ax3.set_ylabel("AUC", color=ChartStyle.LABEL_COLOR)
                ax3.legend(fontsize=9, facecolor=ChartStyle.AXES_FACECOLOR, edgecolor=ChartStyle.GRID_COLOR,
                           labelcolor=ChartStyle.TEXT_COLOR)
                ax3.grid(True, color=ChartStyle.GRID_COLOR, alpha=ChartStyle.GRID_ALPHA)
                ax3.tick_params(colors=ChartStyle.TICK_COLOR)
                for s in ax3.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
                fig3.tight_layout()
                canvas3 = FigureCanvasTkAgg(fig3, master=self._auc_chart)
                canvas3.draw()
                canvas3.get_tk_widget().pack(fill="both", expand=True)
        except ImportError:
            pass

    def _run_sweep(self):
        self._run_btn.set_running(True)
        exp = self._app.app_state.current_experiment
        if exp.fusion_result.status != ModuleStatus.COMPLETED:
            self._run_btn.set_running(False)
            return

        from science.degradation.module_m import run_degradation_sweep
        from app.state import TaskType

        sensor = self._sensor_select.get()
        deg_type = self._type_select.get()
        n_steps = int(self._steps.value or "5")

        def run_fn(progress_callback, *args):
            result = run_degradation_sweep(
                progress_callback, exp.sensor_scores, exp.labels,
                exp.fusion_result.weights, sensor, deg_type, n_steps,
                seed=exp.seed,
            )
            dr = exp.degradation_result
            dr.sensor = result["sensor"]
            dr.degradation_type = result["degradation_type"]
            dr.severity_values = result["severity_values"]
            dr.static_detection_retention = result["static_detection_retention"]
            dr.adaptive_detection_retention = result["adaptive_detection_retention"]
            dr.static_auc = result["static_auc"]
            dr.adaptive_auc = result["adaptive_auc"]
            dr.weights_by_severity = result["weights_by_severity"]
            dr.status = ModuleStatus.COMPLETED
            return result

        self._app.task_manager.submit(TaskType.DEGRADATION_SWEEP, exp.experiment_id, run_fn)
