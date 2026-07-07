"""QT-2.23 — Baselines Comparison Page"""
from __future__ import annotations
import customtkinter as ctk
import numpy as np
from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import SectionFrame, MetricCard, EmptyState
from app.state import ExperimentState, ModuleStatus


BASELINE_COLORS = ["#6B7280", "#9CA3AF", "#D1D5DB", "#4B5563", "#374151"]


class BaselinesPage(ctk.CTkScrollableFrame):
    def __init__(self, master, app_shell):
        super().__init__(master, fg_color=Colors.BG_DARKEST, scrollbar_button_color=Colors.BORDER_SUBTLE)
        self._app = app_shell
        self._build_ui()

    def _build_ui(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=Spacing.PAGE_PADDING, pady=Spacing.PAGE_PADDING)

        # Comparison table section
        self._table_section = SectionFrame(container, title="Baseline Comparison Table")
        self._table_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        self._table = ctk.CTkTextbox(self._table_section.content, font=(Typography.MONO_FONT, 11),
                                      text_color=Colors.TEXT_PRIMARY, fg_color=Colors.BG_DARKEST, height=200, state="disabled")
        self._table.pack(fill="both", expand=True)

        # Chart row
        charts = ctk.CTkFrame(container, fg_color="transparent")
        charts.pack(fill="both", expand=True, pady=(0, Spacing.GRID_GAP))
        charts.grid_columnconfigure(0, weight=1)
        charts.grid_columnconfigure(1, weight=1)

        self._auc_section = SectionFrame(charts, title="AUC Comparison")
        self._auc_section.grid(row=0, column=0, sticky="nsew", padx=(0, Spacing.GRID_GAP))
        self._auc_chart = ctk.CTkFrame(self._auc_section.content, fg_color=Colors.BG_DARKEST, height=300)
        self._auc_chart.pack(fill="both", expand=True)

        self._time_section = SectionFrame(charts, title="Solve Time Comparison")
        self._time_section.grid(row=0, column=1, sticky="nsew")
        self._time_chart = ctk.CTkFrame(self._time_section.content, fg_color=Colors.BG_DARKEST, height=300)
        self._time_chart.pack(fill="both", expand=True)

    def refresh(self, exp: ExperimentState):
        if not exp.baseline_results:
            return

        # Build table
        self._table.configure(state="normal")
        self._table.delete("0.0", "end")
        header = f"{'Method':<25} {'AUC(OOF)':>8} {'Time (s)':>10} {'Obj':>10}"
        self._table.insert("0.0", header + "\n" + "─" * 55 + "\n")
        for br in exp.baseline_results:
            auc_str = f"{br.auc:.4f}" if br.auc else "—"
            time_str = f"{br.solve_time_s:.4f}" if br.solve_time_s else "—"
            obj_str = f"{br.objective_value:.4f}" if br.objective_value else "—"
            self._table.insert("end", f"{br.method_name:<25} {auc_str:>8} {time_str:>10} {obj_str:>10}\n")

        # Add optimized row
        if exp.fusion_result.status == ModuleStatus.COMPLETED:
            auc = exp.metrics_result.auc.point_estimate if exp.metrics_result.auc else 0
            t = exp.fusion_result.solve_time_s or 0
            obj = exp.fusion_result.fisher_objective or 0
            self._table.insert("end", "─" * 55 + "\n")
            self._table.insert("end", f"{'► Optimized (OOF)':<25} {auc:>8.4f} {t:>10.4f} {obj:>10.4f}\n")
        self._table.configure(state="disabled")

        self._render_charts(exp)

    def _render_charts(self, exp):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from ui.theme.tokens import ChartStyle
            from sklearn.metrics import roc_auc_score

            baselines = exp.baseline_results
            names = [b.method_name for b in baselines]
            aucs = []
            for b in baselines:
                if b.auc is not None:
                    aucs.append(b.auc)
                elif b.fused_scores is not None:
                    aucs.append(roc_auc_score(exp.labels, b.fused_scores))
                else:
                    aucs.append(0.5)

            times = [b.solve_time_s if b.solve_time_s else 0 for b in baselines]

            # Add optimized
            if exp.fusion_result.status == ModuleStatus.COMPLETED:
                names.append("Optimized")
                aucs.append(exp.metrics_result.auc.point_estimate if exp.metrics_result.auc else 0.5)
                times.append(exp.fusion_result.solve_time_s or 0)

            colors = BASELINE_COLORS[:len(baselines)] + [Colors.RADAR]

            # AUC bar chart
            for w in self._auc_chart.winfo_children(): w.destroy()
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=ChartStyle.FIGURE_FACECOLOR)
            ax.set_facecolor(ChartStyle.AXES_FACECOLOR)
            bars = ax.barh(range(len(names)), aucs, color=colors)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8, color=ChartStyle.TEXT_COLOR)
            ax.set_xlabel("AUC", color=ChartStyle.LABEL_COLOR)
            ax.set_xlim(0.4, 1.0)
            ax.tick_params(colors=ChartStyle.TICK_COLOR)
            ax.grid(True, axis="x", color=ChartStyle.GRID_COLOR, alpha=ChartStyle.GRID_ALPHA)
            for s in ax.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self._auc_chart)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            # Time bar chart
            for w in self._time_chart.winfo_children(): w.destroy()
            fig2, ax2 = plt.subplots(figsize=(6, 3.5), facecolor=ChartStyle.FIGURE_FACECOLOR)
            ax2.set_facecolor(ChartStyle.AXES_FACECOLOR)
            ax2.barh(range(len(names)), times, color=colors)
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names, fontsize=8, color=ChartStyle.TEXT_COLOR)
            ax2.set_xlabel("Solve Time (s)", color=ChartStyle.LABEL_COLOR)
            ax2.tick_params(colors=ChartStyle.TICK_COLOR)
            ax2.grid(True, axis="x", color=ChartStyle.GRID_COLOR, alpha=ChartStyle.GRID_ALPHA)
            for s in ax2.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
            fig2.tight_layout()
            canvas2 = FigureCanvasTkAgg(fig2, master=self._time_chart)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill="both", expand=True)

        except ImportError:
            pass
