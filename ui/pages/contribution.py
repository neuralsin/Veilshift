"""QT-2.23 — Contribution Analysis Page"""
from __future__ import annotations
import customtkinter as ctk
import numpy as np
from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import SectionFrame, MetricCard, ActionButton, EmptyState
from app.state import ExperimentState, ModuleStatus


class ContributionPage(ctk.CTkScrollableFrame):
    def __init__(self, master, app_shell):
        super().__init__(master, fg_color=Colors.BG_DARKEST, scrollbar_button_color=Colors.BORDER_SUBTLE)
        self._app = app_shell
        self._build_ui()

    def _build_ui(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=Spacing.PAGE_PADDING, pady=Spacing.PAGE_PADDING)

        # Agreement status
        metrics = ctk.CTkFrame(container, fg_color="transparent")
        metrics.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        for i in range(3): metrics.grid_columnconfigure(i, weight=1)

        self._agreement_card = MetricCard(metrics, label="Agreement Status", value="—")
        self._agreement_card.grid(row=0, column=0, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._kendall_card = MetricCard(metrics, label="Kendall's W", value="—")
        self._kendall_card.grid(row=0, column=1, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._run_card = MetricCard(metrics, label="Methods", value="3")
        self._run_card.grid(row=0, column=2, sticky="nsew")

        # Grouped bar chart
        self._grouped_section = SectionFrame(container, title="Three-Lens Contribution Comparison")
        self._grouped_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        self._grouped_chart = ctk.CTkFrame(self._grouped_section.content, fg_color=Colors.BG_DARKEST, height=350)
        self._grouped_chart.pack(fill="both", expand=True)

        # Run button
        self._run_btn = ActionButton(container, text="RUN CONTRIBUTION ANALYSIS",
                                      command=self._run_analysis, primary=True)
        self._run_btn.pack(anchor="w", pady=(0, Spacing.GRID_GAP))

        # Detail tables
        self._detail_section = SectionFrame(container, title="Per-Method Detail")
        self._detail_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        self._detail_text = ctk.CTkTextbox(self._detail_section.content, font=(Typography.MONO_FONT, 11),
                                            text_color=Colors.TEXT_PRIMARY, fg_color=Colors.BG_DARKEST,
                                            height=150, state="disabled")
        self._detail_text.pack(fill="both", expand=True)

    def refresh(self, exp: ExperimentState):
        cr = exp.contribution_result
        if cr.status != ModuleStatus.COMPLETED:
            return

        self._agreement_card.update_value(cr.agreement_status or "—")
        self._kendall_card.update_value(f"{cr.agreement_score:.3f}" if cr.agreement_score else "—")

        # Detail table
        self._detail_text.configure(state="normal")
        self._detail_text.delete("0.0", "end")
        lines = [f"{'Sensor':<12} {'Weights':>10} {'Ablation':>10} {'SHAP':>10}"]
        lines.append("─" * 44)
        for sensor in ["radar", "thermal", "acoustic"]:
            w = cr.weight_contributions.get(sensor, 0) if cr.weight_contributions else 0
            a = cr.ablation_normalized.get(sensor, 0) if cr.ablation_normalized else 0
            s = cr.shap_sensor_contribution.get(sensor, 0) if cr.shap_sensor_contribution else 0
            lines.append(f"{sensor:<12} {w:>10.3f} {a:>10.3f} {s:>10.3f}")
        self._detail_text.insert("0.0", "\n".join(lines))
        self._detail_text.configure(state="disabled")

        self._render_charts(exp)

    def _render_charts(self, exp):
        cr = exp.contribution_result
        if not cr.weight_contributions:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from ui.theme.tokens import ChartStyle

            for w in self._grouped_chart.winfo_children(): w.destroy()
            fig, ax = plt.subplots(figsize=(8, 4), facecolor=ChartStyle.FIGURE_FACECOLOR)
            ax.set_facecolor(ChartStyle.AXES_FACECOLOR)

            sensors = ["radar", "thermal", "acoustic"]
            methods = ["Weights", "Ablation", "SHAP"]
            x = np.arange(len(sensors))
            width = 0.25

            data = [
                [cr.weight_contributions.get(s, 0) for s in sensors],
                [cr.ablation_normalized.get(s, 0) for s in sensors] if cr.ablation_normalized else [0]*3,
                [cr.shap_sensor_contribution.get(s, 0) for s in sensors] if cr.shap_sensor_contribution else [0]*3,
            ]
            method_colors = ["#6B7280", "#9CA3AF", "#D1D5DB"]

            for i, (method, vals, color) in enumerate(zip(methods, data, method_colors)):
                ax.bar(x + i*width, vals, width, label=method, color=color, edgecolor=ChartStyle.AXES_EDGECOLOR)

            # Sensor color indicators on x-axis
            sensor_colors = [Colors.RADAR, Colors.THERMAL, Colors.ACOUSTIC]
            ax.set_xticks(x + width)
            ax.set_xticklabels([s.upper() for s in sensors], fontsize=10, color=ChartStyle.TEXT_COLOR)
            for tick, color in zip(ax.get_xticklabels(), sensor_colors):
                tick.set_color(color)

            ax.set_ylabel("Normalized Contribution", color=ChartStyle.LABEL_COLOR)
            ax.set_title(f"Three-Lens Contribution · Agreement: {cr.agreement_status or '—'}",
                        color=ChartStyle.TEXT_COLOR, fontsize=ChartStyle.TITLE_SIZE)
            ax.legend(fontsize=8, facecolor=ChartStyle.AXES_FACECOLOR, edgecolor=ChartStyle.GRID_COLOR,
                      labelcolor=ChartStyle.TEXT_COLOR)
            ax.grid(True, axis="y", color=ChartStyle.GRID_COLOR, alpha=ChartStyle.GRID_ALPHA)
            ax.tick_params(colors=ChartStyle.TICK_COLOR)
            for s in ax.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self._grouped_chart)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        except ImportError:
            pass

    def _run_analysis(self):
        self._run_btn.set_running(True)
        exp = self._app.app_state.current_experiment
        if exp.fusion_result.status != ModuleStatus.COMPLETED:
            self._run_btn.set_running(False)
            return

        from science.contribution.module_l import run_contribution_analysis
        from app.tasks import TaskManager
        from app.state import TaskType

        def run_fn(progress_callback, *args):
            result = run_contribution_analysis(
                progress_callback,
                exp.sensor_scores, exp.labels,
                exp.feature_matrix, exp.feature_names,
                exp.fusion_result.weights,
            )
            cr = exp.contribution_result
            cr.weight_contributions = result["weight_contributions"]
            cr.ablation_delta_auc = result["ablation_delta_auc"]
            cr.ablation_normalized = result["ablation_normalized"]
            cr.shap_sensor_contribution = result["shap_sensor_contribution"]
            cr.agreement_score = result["agreement_score"]
            cr.agreement_status = result["agreement_status"]
            cr.status = ModuleStatus.COMPLETED
            return result

        self._app.task_manager.submit(TaskType.CONTRIBUTION_ANALYSIS, exp.experiment_id, run_fn)
