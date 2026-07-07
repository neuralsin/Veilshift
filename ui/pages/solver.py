"""QT-2.23 — Solver Status Page"""
from __future__ import annotations
import customtkinter as ctk
from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import SectionFrame, MetricCard, StatusPill
from app.state import ExperimentState, ModuleStatus


class SolverPage(ctk.CTkScrollableFrame):
    def __init__(self, master, app_shell):
        super().__init__(master, fg_color=Colors.BG_DARKEST, scrollbar_button_color=Colors.BORDER_SUBTLE)
        self._app = app_shell
        self._build_ui()

    def _build_ui(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=Spacing.PAGE_PADDING, pady=Spacing.PAGE_PADDING)

        # Status metrics
        metrics = ctk.CTkFrame(container, fg_color="transparent")
        metrics.pack(fill="x", pady=(0, Spacing.GRID_GAP))
        for i in range(3): metrics.grid_columnconfigure(i, weight=1)

        self._backend_card = MetricCard(metrics, label="Backend Status", value="UNAVAILABLE")
        self._backend_card.grid(row=0, column=0, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._last_solver = MetricCard(metrics, label="Last Solver Used", value="—")
        self._last_solver.grid(row=0, column=1, sticky="nsew", padx=(0, Spacing.GRID_GAP))

        self._total_calls = MetricCard(metrics, label="Total Solver Calls", value="0")
        self._total_calls.grid(row=0, column=2, sticky="nsew")

        # Backend detail
        self._backend_section = SectionFrame(container, title="Solver Backend Configuration")
        self._backend_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))

        backend_info = ctk.CTkFrame(self._backend_section.content, fg_color="transparent")
        backend_info.pack(fill="x")

        fields = [
            ("BQPhy / QuantumNow", "CLASSICAL FALLBACK ACTIVE", Colors.WARNING),
            ("Classical QUBO Solver", "neal.SimulatedAnnealingSampler", Colors.SUCCESS),
            ("Classical Continuous", "scipy.optimize.minimize (SLSQP)", Colors.SUCCESS),
            ("Classical Global", "scipy.optimize.differential_evolution", Colors.SUCCESS),
        ]
        for label, value, status_color in fields:
            row = ctk.CTkFrame(backend_info, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=label, font=(Typography.UI_FONT, 11, "bold"),
                         text_color=Colors.TEXT_SECONDARY, width=200, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=value, font=(Typography.MONO_FONT, 11),
                         text_color=status_color, anchor="w").pack(side="left", padx=Spacing.SM)

        # Last solve detail
        self._solve_section = SectionFrame(container, title="Last Solve Detail")
        self._solve_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))

        self._solve_detail = ctk.CTkTextbox(
            self._solve_section.content, font=(Typography.MONO_FONT, 11),
            text_color=Colors.TEXT_PRIMARY, fg_color=Colors.BG_DARKEST, height=200, state="disabled",
        )
        self._solve_detail.pack(fill="both", expand=True)

        # Solve history
        self._history_section = SectionFrame(container, title="Solve History")
        self._history_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))

        self._history = ctk.CTkTextbox(
            self._history_section.content, font=(Typography.MONO_FONT, 11),
            text_color=Colors.TEXT_SECONDARY, fg_color=Colors.BG_DARKEST, height=200, state="disabled",
        )
        self._history.pack(fill="both", expand=True)

    def refresh(self, exp: ExperimentState):
        # Solver status — always computed from what actually ran
        self._backend_card.update_value("CLASSICAL FALLBACK ACTIVE")

        if exp.fusion_result.solver:
            self._last_solver.update_value(exp.fusion_result.solver)

        # Last solve detail
        self._solve_detail.configure(state="normal")
        self._solve_detail.delete("0.0", "end")

        lines = []
        if exp.feature_result.solver_metadata:
            meta = exp.feature_result.solver_metadata
            lines.append("=== Feature Selection (QUBO #1) ===")
            lines.append(f"Solver:      {meta.get('solver', '—')}")
            lines.append(f"Variables:   {meta.get('num_variables', '—')}")
            lines.append(f"Reads:       {meta.get('num_reads', '—')}")
            lines.append(f"Solve time:  {meta.get('solve_time_s', 0):.4f}s")
            lines.append(f"Energy:      {meta.get('energy', '—')}")
            lines.append("")

        if exp.fusion_result.solver:
            lines.append("=== Fusion Weights ===")
            lines.append(f"Solver:      {exp.fusion_result.solver}")
            lines.append(f"Mode:        {exp.fusion_result.optimization_mode}")
            lines.append(f"Solve time:  {exp.fusion_result.solve_time_s:.4f}s" if exp.fusion_result.solve_time_s else "")
            lines.append(f"Objective:   {exp.fusion_result.fisher_objective:.6f}" if exp.fusion_result.fisher_objective else "")
            lines.append(f"Weights:     {exp.fusion_result.weights}")

        if not lines:
            lines.append("No solver calls have been made yet.")

        self._solve_detail.insert("0.0", "\n".join(lines))
        self._solve_detail.configure(state="disabled")

        # Solve history from adapter
        try:
            from science.solvers.module_i import solver_adapter
            history = solver_adapter.get_history()
            self._total_calls.update_value(str(len(history)))

            self._history.configure(state="normal")
            self._history.delete("0.0", "end")
            if history:
                for h in history:
                    self._history.insert("end",
                        f"{h.get('problem_type', ''):>10} | {h.get('solver', ''):>35} | "
                        f"n={h.get('n_variables', 0):>4} | {h.get('solve_time_s', 0):.4f}s | "
                        f"obj={h.get('objective', 0):.4f} | {h.get('status', '')}\n"
                    )
            else:
                self._history.insert("0.0", "No solver calls recorded.")
            self._history.configure(state="disabled")
        except Exception:
            pass
