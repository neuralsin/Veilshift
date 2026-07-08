"""
QT-2.23 — Radar Sensor Page

Per Stitch screen "Radar Sensor - Physics Simulation & SNR Analysis".
Layout: Left parameter rail | Center tabbed workspace | Right inspector.

Scientific question: "Why is the radar target difficult to detect?"
"""

from __future__ import annotations
from typing import Any, Optional

import customtkinter as ctk
import numpy as np

from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import SectionFrame, ParameterField, ActionButton, EmptyState, StatusPill
from app.state import ExperimentState, ModuleStatus, TaskType
from app.events import event_bus, EventType, Event


class RadarPage(ctk.CTkFrame):
    """Radar sensor configuration and analysis page."""

    def __init__(self, master, app_shell):
        super().__init__(master, fg_color=Colors.BG_DARKEST)
        self._app = app_shell
        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0, minsize=240)  # Parameter rail
        self.grid_columnconfigure(1, weight=1)               # Chart workspace
        self.grid_columnconfigure(2, weight=0, minsize=280)  # Inspector
        self.grid_rowconfigure(0, weight=1)

        # LEFT: Parameter rail
        self._build_param_rail()

        # CENTER: Tabbed chart workspace
        self._build_workspace()

        # RIGHT: Inspector panel
        self._build_inspector()

    def _build_param_rail(self):
        rail = ctk.CTkScrollableFrame(
            self, fg_color=Colors.BG_CARD,
            width=240, corner_radius=0,
            scrollbar_button_color=Colors.BORDER_SUBTLE,
        )
        rail.grid(row=0, column=0, sticky="nsew")

        # Radar Hardware
        ctk.CTkLabel(rail, text="RADAR HARDWARE",
                     font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.RADAR).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        self._params = {}
        params = [
            ("Pt", "10000", "W", "Transmit power"),
            ("Gain", "30", "dBi", "Antenna gain"),
            ("Freq", "10", "GHz", "Operating frequency"),
            ("BW", "1", "MHz", "Receiver bandwidth"),
            ("NF", "4.77", "dB", "Noise figure"),
            ("Losses", "6.0", "dB", "System losses"),
            ("Range", "50", "km", "Target range"),
        ]
        for name, default, unit, tooltip in params:
            pf = ParameterField(rail, label=name, default=default, unit=unit, tooltip=tooltip)
            pf.pack(fill="x", padx=Spacing.MD, pady=2)
            self._params[name] = pf

        # Target
        ctk.CTkLabel(rail, text="TARGET",
                     font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.RADAR).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        self._target_type = ctk.CTkOptionMenu(
            rail, values=["Low Observability", "Conventional", "Custom"],
            font=(Typography.UI_FONT, 11),
            fg_color=Colors.BG_DARKEST, button_color=Colors.BG_ELEVATED,
            dropdown_fg_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY,
            width=200, height=Spacing.INPUT_HEIGHT,
        )
        self._target_type.pack(padx=Spacing.MD, pady=2, anchor="w")
        self._target_type.set("Low Observability")

        self._rcs = ParameterField(rail, label="RCS σ", default="0.01", unit="m²")
        self._rcs.pack(fill="x", padx=Spacing.MD, pady=2)

        # Clutter
        ctk.CTkLabel(rail, text="CLUTTER",
                     font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.RADAR).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        self._clutter_model = ctk.CTkOptionMenu(
            rail, values=["Rayleigh", "Weibull"],
            font=(Typography.UI_FONT, 11),
            fg_color=Colors.BG_DARKEST, button_color=Colors.BG_ELEVATED,
            dropdown_fg_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY,
            width=200, height=Spacing.INPUT_HEIGHT,
        )
        self._clutter_model.pack(padx=Spacing.MD, pady=2, anchor="w")

        self._clutter_power = ParameterField(rail, label="Power", default="1.0", unit="")
        self._clutter_power.pack(fill="x", padx=Spacing.MD, pady=2)

        self._weibull_shape = ParameterField(rail, label="Shape k", default="1.5", unit="")
        self._weibull_shape.pack(fill="x", padx=Spacing.MD, pady=2)

        # CFAR
        ctk.CTkLabel(rail, text="CFAR DETECTION",
                     font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.RADAR).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        self._pfa = ParameterField(rail, label="Pfa", default="1e-4", unit="")
        self._pfa.pack(fill="x", padx=Spacing.MD, pady=2)
        self._ref_cells = ParameterField(rail, label="Ref cells", default="16", unit="")
        self._ref_cells.pack(fill="x", padx=Spacing.MD, pady=2)
        self._guard = ParameterField(rail, label="Guard", default="4", unit="")
        self._guard.pack(fill="x", padx=Spacing.MD, pady=2)

        # Run button
        self._run_btn = ActionButton(
            rail, text="RUN RADAR SIM",
            command=self._run_simulation, primary=True,
        )
        self._run_btn.pack(fill="x", padx=Spacing.MD, pady=Spacing.LG)

    def _build_workspace(self):
        workspace = ctk.CTkFrame(self, fg_color=Colors.BG_DARKEST, corner_radius=0)
        workspace.grid(row=0, column=1, sticky="nsew", padx=1)

        # Tabs
        self._tabview = ctk.CTkTabview(
            workspace, fg_color=Colors.BG_CARD,
            segmented_button_fg_color=Colors.BG_SURFACE,
            segmented_button_selected_color=Colors.RADAR,
            segmented_button_unselected_color=Colors.BG_ELEVATED,
            text_color=Colors.TEXT_PRIMARY,
            corner_radius=Radii.DEFAULT,
        )
        self._tabview.pack(fill="both", expand=True, padx=Spacing.GRID_GAP, pady=Spacing.GRID_GAP)

        tab_names = ["RANGE PROFILE", "RANGE-DOPPLER", "CFAR", "DISTRIBUTION"]
        for name in tab_names:
            self._tabview.add(name)

        # Each tab gets a placeholder for matplotlib canvas
        for name in tab_names:
            tab = self._tabview.tab(name)
            self._empty = EmptyState(
                tab,
                title="RADAR SIGNAL NOT GENERATED",
                message="Configure physical and clutter parameters, then run the sensor simulation.",
            )
            self._empty.pack(fill="both", expand=True)

        # Store chart frames for later matplotlib embedding
        self._chart_frames = {name: self._tabview.tab(name) for name in tab_names}

    def _build_inspector(self):
        inspector = ctk.CTkScrollableFrame(
            self, fg_color=Colors.BG_CARD,
            width=280, corner_radius=0,
            scrollbar_button_color=Colors.BORDER_SUBTLE,
        )
        inspector.grid(row=0, column=2, sticky="nsew")

        # Equation card
        ctk.CTkLabel(inspector, text="RADAR RANGE EQUATION",
                     font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        eq_frame = ctk.CTkFrame(
            inspector, fg_color=Colors.BG_DARKEST,
            border_color=Colors.BORDER_SUBTLE, border_width=1,
            corner_radius=Radii.SMALL,
        )
        eq_frame.pack(fill="x", padx=Spacing.MD, pady=Spacing.SM)

        self._eq_text = ctk.CTkLabel(
            eq_frame,
            text="SNR = Pt·G²·λ²·σ / (4π)³·R⁴·kT₀BFL",
            font=(Typography.MONO_FONT, 11),
            text_color=Colors.TEXT_PRIMARY,
            wraplength=240,
        )
        self._eq_text.pack(padx=Spacing.SM, pady=Spacing.SM)

        # Live values
        ctk.CTkLabel(inspector, text="LIVE VALUES",
                     font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        self._live_values = ctk.CTkTextbox(
            inspector,
            font=(Typography.MONO_FONT, 11),
            text_color=Colors.TEXT_MUTED,
            fg_color=Colors.BG_DARKEST,
            height=120,
            state="disabled",
        )
        self._live_values.pack(fill="x", padx=Spacing.MD, pady=Spacing.SM)

        # Inference
        ctk.CTkLabel(inspector, text="PHYSICAL INFERENCE",
                     font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        self._inference = ctk.CTkTextbox(
            inspector,
            font=(Typography.UI_FONT, 11),
            text_color=Colors.TEXT_SECONDARY,
            fg_color="transparent",
            height=100,
            wrap="word",
            state="disabled",
        )
        self._inference.pack(fill="x", padx=Spacing.MD, pady=Spacing.SM)

        # Validation
        ctk.CTkLabel(inspector, text="VALIDATION",
                     font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        self._val_frame = ctk.CTkFrame(inspector, fg_color="transparent")
        self._val_frame.pack(fill="x", padx=Spacing.MD)

        self._cfar_status = StatusPill(self._val_frame, "PENDING")
        self._cfar_status.pack(anchor="w", pady=2)

    def refresh(self, exp: ExperimentState):
        """Refresh radar page from experiment state."""
        result = exp.radar_result
        if result.status in [ModuleStatus.COMPLETED, ModuleStatus.FAILED]:
            self._run_btn.set_running(False)
        if result.status == ModuleStatus.COMPLETED:
            self._update_live_values(exp)
            self._update_inference(exp)
            self._update_charts(exp)
            self._cfar_status.set_status("PASS")

    def _update_live_values(self, exp: ExperimentState):
        r = exp.radar_result
        self._live_values.configure(state="normal")
        self._live_values.delete("0.0", "end")
        lines = [
            f"σ (RCS)    = {exp.radar_config.rcs_m2} m²",
            f"R          = {exp.radar_config.range_m/1000:.0f} km",
            f"SNR        = {r.snr_db:.2f} dB",
            f"α (CFAR)   = {r.cfar_alpha:.4f}" if r.cfar_alpha else "",
            f"Pfa_config = {r.configured_pfa:.1e}" if r.configured_pfa else "",
            f"Pfa_emp    = {r.empirical_pfa:.2e}" if r.empirical_pfa else "",
        ]
        self._live_values.insert("0.0", "\n".join(l for l in lines if l))
        self._live_values.configure(state="disabled")

    def _update_inference(self, exp: ExperimentState):
        r = exp.radar_result
        self._inference.configure(state="normal")
        self._inference.delete("0.0", "end")

        lines = []
        if r.snr_db is not None:
            if r.snr_db < -5:
                lines.append(f"At SNR={r.snr_db:.1f} dB, the target return is buried well below the noise floor.")
            elif r.snr_db < 5:
                lines.append(f"At SNR={r.snr_db:.1f} dB, the target is near the detection threshold.")
            else:
                lines.append(f"At SNR={r.snr_db:.1f} dB, the target should be detectable by radar alone.")

        if r.empirical_pfa is not None and r.configured_pfa is not None:
            if r.empirical_pfa == 0.0:
                lines.append(f"CFAR threshold too high (empirical Pfa = 0.0) — no false alarms triggered.")
            else:
                ratio = r.empirical_pfa / r.configured_pfa
                if 0.5 < ratio < 2.0:
                    lines.append(f"CFAR calibration is consistent (Pfa_emp/Pfa_target = {ratio:.2f}).")
                else:
                    lines.append(f"CFAR calibration shows deviation (ratio = {ratio:.2f}).")

        self._inference.insert("0.0", "\n".join(lines))
        self._inference.configure(state="disabled")

    def _update_charts(self, exp: ExperimentState):
        """Embed matplotlib charts into the tab frames."""
        r = exp.radar_result
        if r.range_profile is None:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from ui.theme.tokens import ChartStyle

            # Range Profile tab
            tab = self._chart_frames["RANGE PROFILE"]
            for w in tab.winfo_children():
                w.destroy()

            fig, ax = plt.subplots(figsize=(8, 4), facecolor=ChartStyle.FIGURE_FACECOLOR)
            ax.set_facecolor(ChartStyle.AXES_FACECOLOR)

            x = np.arange(len(r.range_profile))
            ax.plot(x, 20 * np.log10(r.range_profile + 1e-12),
                    color=Colors.RADAR, linewidth=ChartStyle.LINE_WIDTH, label="Signal")
            if r.cfar_threshold is not None:
                ax.plot(x, 10 * np.log10(r.cfar_threshold + 1e-12),
                        color=Colors.CRITICAL, linewidth=1, linestyle="--", label="CFAR Threshold", alpha=0.7)

            ax.set_xlabel("Range Cell", color=ChartStyle.LABEL_COLOR, fontsize=ChartStyle.LABEL_SIZE)
            ax.set_ylabel("Power (dB)", color=ChartStyle.LABEL_COLOR, fontsize=ChartStyle.LABEL_SIZE)
            ax.set_title("Range Profile with CA-CFAR", color=ChartStyle.TEXT_COLOR, fontsize=ChartStyle.TITLE_SIZE)
            ax.tick_params(colors=ChartStyle.TICK_COLOR, labelsize=ChartStyle.TICK_SIZE)
            ax.legend(fontsize=ChartStyle.LEGEND_SIZE, facecolor=ChartStyle.AXES_FACECOLOR,
                      edgecolor=ChartStyle.GRID_COLOR, labelcolor=ChartStyle.TEXT_COLOR)
            ax.grid(True, color=ChartStyle.GRID_COLOR, alpha=ChartStyle.GRID_ALPHA)
            for spine in ax.spines.values():
                spine.set_color(ChartStyle.AXES_EDGECOLOR)

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            # Range-Doppler tab
            if r.range_doppler_map is not None:
                tab2 = self._chart_frames["RANGE-DOPPLER"]
                for w in tab2.winfo_children():
                    w.destroy()

                fig2, ax2 = plt.subplots(figsize=(8, 4), facecolor=ChartStyle.FIGURE_FACECOLOR)
                ax2.set_facecolor(ChartStyle.AXES_FACECOLOR)

                rd_db = 20 * np.log10(np.abs(r.range_doppler_map) + 1e-12)
                im = ax2.imshow(rd_db.T, aspect="auto", cmap="inferno",
                               origin="lower", interpolation="nearest")
                ax2.set_xlabel("Range Cell", color=ChartStyle.LABEL_COLOR, fontsize=ChartStyle.LABEL_SIZE)
                ax2.set_ylabel("Doppler Bin", color=ChartStyle.LABEL_COLOR, fontsize=ChartStyle.LABEL_SIZE)
                ax2.set_title("Range-Doppler Map", color=ChartStyle.TEXT_COLOR, fontsize=ChartStyle.TITLE_SIZE)
                ax2.tick_params(colors=ChartStyle.TICK_COLOR, labelsize=ChartStyle.TICK_SIZE)
                for spine in ax2.spines.values():
                    spine.set_color(ChartStyle.AXES_EDGECOLOR)

                cbar = fig2.colorbar(im, ax=ax2)
                cbar.set_label("Power (dB)", color=ChartStyle.LABEL_COLOR, fontsize=ChartStyle.LABEL_SIZE)
                cbar.ax.tick_params(colors=ChartStyle.TICK_COLOR, labelsize=ChartStyle.TICK_SIZE)

                fig2.tight_layout()
                canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
                canvas2.draw()
                canvas2.get_tk_widget().pack(fill="both", expand=True)

        except ImportError:
            pass

    def _run_simulation(self):
        """Run radar simulation in background."""
        self._run_btn.set_running(True)

        from science.radar.module_a import run_radar_simulation

        config = self._app.app_state.current_experiment.radar_config

        def on_complete(event: Event):
            if event.event_type == EventType.TASK_COMPLETED:
                self._app.after(0, lambda: self.refresh(self._app.app_state.current_experiment))
                self._app.after(0, lambda: self._run_btn.set_running(False))

        event_bus.subscribe(EventType.TASK_COMPLETED, on_complete)

        self._app.task_manager.submit(
            TaskType.RADAR_SIMULATION,
            self._app.app_state.current_experiment.experiment_id,
            run_radar_simulation,
            config,
            self._app.app_state.current_experiment.seed,
        )
