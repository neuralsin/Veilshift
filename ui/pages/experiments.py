"""QT-2.23 — Experiments Management Page"""
from __future__ import annotations
import time
import customtkinter as ctk
from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import SectionFrame, ActionButton, EmptyState
from app.state import ExperimentState, ModuleStatus


class ExperimentsPage(ctk.CTkScrollableFrame):
    def __init__(self, master, app_shell):
        super().__init__(master, fg_color=Colors.BG_DARKEST, scrollbar_button_color=Colors.BORDER_SUBTLE)
        self._app = app_shell
        self._build_ui()

    def _build_ui(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=Spacing.PAGE_PADDING, pady=Spacing.PAGE_PADDING)

        # Actions
        actions = ctk.CTkFrame(container, fg_color="transparent")
        actions.pack(fill="x", pady=(0, Spacing.GRID_GAP))

        ActionButton(actions, text="NEW EXPERIMENT", command=self._new_exp, primary=True).pack(side="left", padx=(0, Spacing.SM))
        ActionButton(actions, text="DUPLICATE CURRENT", command=self._duplicate, primary=False).pack(side="left", padx=(0, Spacing.SM))
        ActionButton(actions, text="EXPORT REPORT", command=self._export, primary=False).pack(side="left")

        # Current experiment info
        self._current_section = SectionFrame(container, title="Current Experiment")
        self._current_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))

        self._current_info = ctk.CTkTextbox(
            self._current_section.content, font=(Typography.MONO_FONT, 11),
            text_color=Colors.TEXT_PRIMARY, fg_color=Colors.BG_DARKEST, height=180, state="disabled",
        )
        self._current_info.pack(fill="both", expand=True)

        # Experiment history
        self._history_section = SectionFrame(container, title="Experiment History")
        self._history_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))

        self._history_list = ctk.CTkTextbox(
            self._history_section.content, font=(Typography.MONO_FONT, 11),
            text_color=Colors.TEXT_SECONDARY, fg_color=Colors.BG_DARKEST, height=200, state="disabled",
        )
        self._history_list.pack(fill="both", expand=True)

    def refresh(self, exp: ExperimentState):
        self._current_info.configure(state="normal")
        self._current_info.delete("0.0", "end")

        lines = [
            f"ID:           {exp.experiment_id}",
            f"Name:         {exp.experiment_name}",
            f"Seed:         {exp.seed}",
            f"Created:      {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp.timestamp))}",
            f"Target:       {exp.target_regime.value}",
            f"Status:       {exp.status.value}",
            f"",
            f"--- SENSOR STATUS ---",
            f"Radar:        {exp.radar_result.status.value}",
            f"Thermal:      {exp.thermal_result.status.value}",
            f"Acoustic:     {exp.acoustic_result.status.value}",
            f"",
            f"--- OPTIMIZATION ---",
            f"QUBO (G):     {exp.feature_result.status.value}",
            f"Fusion (H):   {exp.fusion_result.status.value}",
            f"Evaluation:   {exp.metrics_result.status.value}",
            f"",
            f"--- ANALYSIS ---",
            f"Contribution: {exp.contribution_result.status.value}",
            f"Degradation:  {exp.degradation_result.status.value}",
            f"Scaling:      {exp.scaling_result.status.value}",
        ]

        if exp.metrics_result.auc:
            lines.append(f"\n--- HEADLINE RESULT ---")
            lines.append(f"AUC:          {exp.metrics_result.auc}")
        if exp.fusion_result.solver:
            lines.append(f"Solver:       {exp.fusion_result.solver}")

        self._current_info.insert("0.0", "\n".join(lines))
        self._current_info.configure(state="disabled")

        # History
        self._history_list.configure(state="normal")
        self._history_list.delete("0.0", "end")
        exps = self._app.app_state.experiments
        if exps:
            for e in exps:
                ts = time.strftime('%H:%M:%S', time.localtime(e.timestamp))
                self._history_list.insert("end", f"{e.experiment_id}  {ts}  {e.status.value}\n")
        else:
            self._history_list.insert("0.0", "No previous experiments.")
        self._history_list.configure(state="disabled")

    def _new_exp(self):
        old = self._app.app_state.current_experiment
        self._app.app_state.experiments.append(old)
        self._app._new_experiment()
        self.refresh(self._app.app_state.current_experiment)

    def _duplicate(self):
        import copy
        exp = self._app.app_state.current_experiment
        new_exp = copy.deepcopy(exp)
        new_exp.experiment_id = f"EXP-{__import__('uuid').uuid4().hex[:6].upper()}"
        new_exp.timestamp = time.time()
        new_exp.experiment_name = f"{exp.experiment_name} (copy)"
        self._app.app_state.current_experiment = new_exp
        self.refresh(new_exp)

    def _export(self):
        pass  # Export functionality
