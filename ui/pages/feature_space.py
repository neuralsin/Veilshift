"""QT-2.23 — Feature Space Page (QUBO Feature Selection)"""
from __future__ import annotations
import customtkinter as ctk
import numpy as np
from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import SectionFrame, MetricCard, ActionButton, EmptyState
from app.state import ExperimentState, ModuleStatus


class FeatureSpacePage(ctk.CTkScrollableFrame):
    def __init__(self, master, app_shell):
        super().__init__(master, fg_color=Colors.BG_DARKEST, scrollbar_button_color=Colors.BORDER_SUBTLE)
        self._app = app_shell
        self._build_ui()

    def _build_ui(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=Spacing.PAGE_PADDING, pady=Spacing.PAGE_PADDING)

        # Metric cards row
        metrics = ctk.CTkFrame(container, fg_color="transparent")
        metrics.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        for i in range(4): metrics.grid_columnconfigure(i, weight=1)

        self._total_feat = MetricCard(metrics, label="Total Features", value="—")
        self._total_feat.grid(row=0, column=0, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._selected = MetricCard(metrics, label="Selected", value="—", accent_color=Colors.RADAR)
        self._selected.grid(row=0, column=1, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._bf_match = MetricCard(metrics, label="Brute Force Match", value="—")
        self._bf_match.grid(row=0, column=2, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._solver_card = MetricCard(metrics, label="Solver", value="—")
        self._solver_card.grid(row=0, column=3, sticky="nsew")

        # Two-column layout: charts
        charts = ctk.CTkFrame(container, fg_color="transparent")
        charts.pack(fill="both", expand=True, pady=(0, Spacing.GRID_GAP))
        charts.grid_columnconfigure(0, weight=1)
        charts.grid_columnconfigure(1, weight=1)

        # Relevance chart
        self._relevance_section = SectionFrame(charts, title="Feature Relevance (|corr with label|)")
        self._relevance_section.grid(row=0, column=0, sticky="nsew", padx=(0, Spacing.GRID_GAP))
        self._relevance_chart = ctk.CTkFrame(self._relevance_section.content, fg_color=Colors.BG_DARKEST, height=300)
        self._relevance_chart.pack(fill="both", expand=True)

        # Redundancy matrix
        self._redundancy_section = SectionFrame(charts, title="Redundancy Matrix")
        self._redundancy_section.grid(row=0, column=1, sticky="nsew")
        self._redundancy_chart = ctk.CTkFrame(self._redundancy_section.content, fg_color=Colors.BG_DARKEST, height=300)
        self._redundancy_chart.pack(fill="both", expand=True)

        # QUBO matrix
        self._qubo_section = SectionFrame(container, title="QUBO Matrix Q")
        self._qubo_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        self._qubo_chart = ctk.CTkFrame(self._qubo_section.content, fg_color=Colors.BG_DARKEST, height=300)
        self._qubo_chart.pack(fill="both", expand=True)

        # Selected features list
        self._selected_section = SectionFrame(container, title="Selected Feature Subset")
        self._selected_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        self._feature_list = ctk.CTkTextbox(
            self._selected_section.content, font=(Typography.MONO_FONT, 12),
            text_color=Colors.TEXT_PRIMARY, fg_color=Colors.BG_DARKEST, height=120, state="disabled",
        )
        self._feature_list.pack(fill="both", expand=True)

    def refresh(self, exp: ExperimentState):
        r = exp.feature_result
        if r.status != ModuleStatus.COMPLETED:
            return

        self._total_feat.update_value(str(len(r.feature_names)) if r.feature_names else "—")
        self._selected.update_value(str(len(r.selected_indices)) if r.selected_indices else "—")

        match_str = "—"
        if r.subset_match is not None:
            match_str = "✓ MATCH" if r.subset_match else "✗ MISMATCH"
        self._bf_match.update_value(match_str)

        self._solver_card.update_value(r.solver or "—")

        # Feature list
        self._feature_list.configure(state="normal")
        self._feature_list.delete("0.0", "end")
        if r.selected_features:
            lines = [f"  [{i+1}] {name}" for i, name in enumerate(r.selected_features)]
            if r.objective_value is not None:
                lines.append(f"\n  Solver objective:     {r.objective_value:.6f}")
            if r.brute_force_objective is not None:
                lines.append(f"  Brute force objective: {r.brute_force_objective:.6f}")
            self._feature_list.insert("0.0", "\n".join(lines))
        self._feature_list.configure(state="disabled")

        self._render_charts(exp)

    def _render_charts(self, exp):
        r = exp.feature_result
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from ui.theme.tokens import ChartStyle

            # Relevance bar chart
            if r.relevance is not None and r.feature_names:
                for w in self._relevance_chart.winfo_children(): w.destroy()
                fig, ax = plt.subplots(figsize=(6, 3), facecolor=ChartStyle.FIGURE_FACECOLOR)
                ax.set_facecolor(ChartStyle.AXES_FACECOLOR)

                n = len(r.relevance)
                colors = []
                for name in r.feature_names:
                    if name.startswith("radar"): colors.append(Colors.RADAR)
                    elif name.startswith("thermal"): colors.append(Colors.THERMAL)
                    elif name.startswith("acoustic"): colors.append(Colors.ACOUSTIC)
                    else: colors.append(Colors.TEXT_MUTED)

                selected_set = set(r.selected_indices) if r.selected_indices else set()
                edge_colors = [Colors.SUCCESS if i in selected_set else "none" for i in range(n)]

                ax.barh(range(n), r.relevance, color=colors, edgecolor=edge_colors, linewidth=2)
                ax.set_yticks(range(n))
                ax.set_yticklabels([n[:15] for n in r.feature_names], fontsize=7, color=ChartStyle.TEXT_COLOR)
                ax.set_xlabel("Relevance", color=ChartStyle.LABEL_COLOR)
                ax.tick_params(colors=ChartStyle.TICK_COLOR)
                ax.invert_yaxis()
                for s in ax.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
                fig.tight_layout()
                canvas = FigureCanvasTkAgg(fig, master=self._relevance_chart)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)

            # Redundancy heatmap
            if r.redundancy_matrix is not None:
                for w in self._redundancy_chart.winfo_children(): w.destroy()
                fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor=ChartStyle.FIGURE_FACECOLOR)
                ax2.set_facecolor(ChartStyle.AXES_FACECOLOR)
                im = ax2.imshow(r.redundancy_matrix, cmap="magma", aspect="auto", vmin=0, vmax=1)
                ax2.set_title("Redundancy", color=ChartStyle.TEXT_COLOR)
                ax2.tick_params(colors=ChartStyle.TICK_COLOR)
                fig2.colorbar(im, ax=ax2)
                for s in ax2.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
                fig2.tight_layout()
                canvas2 = FigureCanvasTkAgg(fig2, master=self._redundancy_chart)
                canvas2.draw()
                canvas2.get_tk_widget().pack(fill="both", expand=True)

        except ImportError:
            pass
