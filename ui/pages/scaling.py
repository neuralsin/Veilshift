"""QT-2.23 — Scaling Experiment Page"""
from __future__ import annotations
import customtkinter as ctk
import numpy as np
from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import SectionFrame, MetricCard, ActionButton, EmptyState
from app.state import ExperimentState, ModuleStatus


class ScalingPage(ctk.CTkScrollableFrame):
    def __init__(self, master, app_shell):
        super().__init__(master, fg_color=Colors.BG_DARKEST, scrollbar_button_color=Colors.BORDER_SUBTLE)
        self._app = app_shell
        self._build_ui()

    def _build_ui(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=Spacing.PAGE_PADDING, pady=Spacing.PAGE_PADDING)

        # Action
        self._run_btn = ActionButton(container, text="RUN SCALING EXPERIMENT",
                                      command=self._run_scaling, primary=True)
        self._run_btn.pack(anchor="w", pady=(0, Spacing.GRID_GAP))

        # Charts
        charts = ctk.CTkFrame(container, fg_color="transparent")
        charts.pack(fill="both", expand=True, pady=(0, Spacing.GRID_GAP))
        charts.grid_columnconfigure(0, weight=1)
        charts.grid_columnconfigure(1, weight=1)

        self._time_section = SectionFrame(charts, title="Solve Time vs Problem Size (log scale)")
        self._time_section.grid(row=0, column=0, sticky="nsew", padx=(0, Spacing.GRID_GAP))
        self._time_chart = ctk.CTkFrame(self._time_section.content, fg_color=Colors.BG_DARKEST, height=350)
        self._time_chart.pack(fill="both", expand=True)

        self._quality_section = SectionFrame(charts, title="Solution Quality vs Problem Size")
        self._quality_section.grid(row=0, column=1, sticky="nsew")
        self._quality_chart = ctk.CTkFrame(self._quality_section.content, fg_color=Colors.BG_DARKEST, height=350)
        self._quality_chart.pack(fill="both", expand=True)

        # Results table
        self._table_section = SectionFrame(container, title="Raw Results")
        self._table_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        self._table = ctk.CTkTextbox(self._table_section.content, font=(Typography.MONO_FONT, 11),
                                      text_color=Colors.TEXT_PRIMARY, fg_color=Colors.BG_DARKEST, height=200, state="disabled")
        self._table.pack(fill="both", expand=True)

    def refresh(self, exp: ExperimentState):
        sr = exp.scaling_result
        if sr.status != ModuleStatus.COMPLETED:
            return
        self._run_btn.set_running(False)

        # Table
        self._table.configure(state="normal")
        self._table.delete("0.0", "end")
        header = f"{'Size':>6} {'Solver':<25} {'Time (s)':>10} {'Objective':>12} {'Quality':>8}"
        self._table.insert("0.0", header + "\n" + "─" * 63 + "\n")
        if sr.results:
            for r in sr.results:
                t = f"{r['solve_time_s']:.4f}" if r['solve_time_s'] else "SKIP"
                o = f"{r['objective_value']:.4f}" if r['objective_value'] is not None else "—"
                q = f"{r['normalized_quality']:.3f}" if r['normalized_quality'] is not None else "—"
                self._table.insert("end", f"{r['problem_size']:>6} {r['solver']:<25} {t:>10} {o:>12} {q:>8}\n")
        self._table.configure(state="disabled")

        self._render_charts(exp)

    def _render_charts(self, exp):
        sr = exp.scaling_result
        if not sr.results:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from ui.theme.tokens import ChartStyle

            solvers = list(set(r["solver"] for r in sr.results))
            solver_colors = {s: c for s, c in zip(solvers, [Colors.RADAR, Colors.THERMAL, Colors.ACOUSTIC,
                                                             Colors.TEXT_MUTED, "#9CA3AF"])}

            # Time plot (log y)
            for w in self._time_chart.winfo_children(): w.destroy()
            fig, ax = plt.subplots(figsize=(6, 4), facecolor=ChartStyle.FIGURE_FACECOLOR)
            ax.set_facecolor(ChartStyle.AXES_FACECOLOR)
            for solver in solvers:
                data = [r for r in sr.results if r["solver"] == solver and r["solve_time_s"] is not None]
                if data:
                    sizes = [d["problem_size"] for d in data]
                    times = [d["solve_time_s"] for d in data]
                    ax.semilogy(sizes, times, 'o-', color=solver_colors[solver], linewidth=2, markersize=6, label=solver)
            ax.set_xlabel("Problem Size (variables)", color=ChartStyle.LABEL_COLOR)
            ax.set_ylabel("Solve Time (s, log)", color=ChartStyle.LABEL_COLOR)
            ax.set_title("Scaling: Solve Time", color=ChartStyle.TEXT_COLOR)
            ax.legend(fontsize=7, facecolor=ChartStyle.AXES_FACECOLOR, edgecolor=ChartStyle.GRID_COLOR,
                      labelcolor=ChartStyle.TEXT_COLOR)
            ax.grid(True, color=ChartStyle.GRID_COLOR, alpha=ChartStyle.GRID_ALPHA)
            ax.tick_params(colors=ChartStyle.TICK_COLOR)
            for s in ax.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self._time_chart)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            # Quality plot
            for w in self._quality_chart.winfo_children(): w.destroy()
            fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor=ChartStyle.FIGURE_FACECOLOR)
            ax2.set_facecolor(ChartStyle.AXES_FACECOLOR)
            for solver in solvers:
                data = [r for r in sr.results if r["solver"] == solver and r["normalized_quality"] is not None]
                if data:
                    sizes = [d["problem_size"] for d in data]
                    quality = [d["normalized_quality"] for d in data]
                    ax2.plot(sizes, quality, 's-', color=solver_colors[solver], linewidth=2, markersize=6, label=solver)
            ax2.set_xlabel("Problem Size (variables)", color=ChartStyle.LABEL_COLOR)
            ax2.set_ylabel("Solution Quality (1.0 = optimal)", color=ChartStyle.LABEL_COLOR)
            ax2.set_title("Scaling: Solution Quality", color=ChartStyle.TEXT_COLOR)
            ax2.legend(fontsize=7, facecolor=ChartStyle.AXES_FACECOLOR, edgecolor=ChartStyle.GRID_COLOR,
                       labelcolor=ChartStyle.TEXT_COLOR)
            ax2.grid(True, color=ChartStyle.GRID_COLOR, alpha=ChartStyle.GRID_ALPHA)
            ax2.tick_params(colors=ChartStyle.TICK_COLOR)
            for s in ax2.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
            fig2.tight_layout()
            canvas2 = FigureCanvasTkAgg(fig2, master=self._quality_chart)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill="both", expand=True)
        except ImportError:
            pass

    def _run_scaling(self):
        self._run_btn.set_running(True)
        from science.scaling.module_j import run_scaling_experiment
        from app.state import TaskType

        exp = self._app.app_state.current_experiment

        def run_fn(progress_callback, *args):
            result = run_scaling_experiment(progress_callback, seed=exp.seed)
            sr = exp.scaling_result
            sr.problem_sizes = result["problem_sizes"]
            sr.results = result["results"]
            sr.status = ModuleStatus.COMPLETED
            return result

        self._app.task_manager.submit(TaskType.SCALING_EXPERIMENT, exp.experiment_id, run_fn)
