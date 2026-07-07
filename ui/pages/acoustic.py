"""QT-2.23 — Acoustic / Sonar Sensor Page"""
from __future__ import annotations
import customtkinter as ctk
import numpy as np
from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import SectionFrame, ParameterField, ActionButton, EmptyState
from app.state import ExperimentState, ModuleStatus


class AcousticPage(ctk.CTkFrame):
    def __init__(self, master, app_shell):
        super().__init__(master, fg_color=Colors.BG_DARKEST)
        self._app = app_shell
        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0, minsize=240)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=280)
        self.grid_rowconfigure(0, weight=1)

        # LEFT: Parameters
        rail = ctk.CTkScrollableFrame(self, fg_color=Colors.BG_CARD, width=240, corner_radius=0,
                                       scrollbar_button_color=Colors.BORDER_SUBTLE)
        rail.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(rail, text="SONAR EQUATION", font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.ACOUSTIC).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        self._params = {}
        for name, default, unit in [
            ("SL", "110", "dB"), ("Range", "5000", "m"),
            ("DI", "20", "dB"), ("DT", "0", "dB"),
            ("Sea St", "3", ""), ("α_abs", "0.005", "dB/m"),
            ("Duration", "60", "s"), ("fs", "2000", "Hz"),
        ]:
            pf = ParameterField(rail, label=name, default=default, unit=unit)
            pf.pack(fill="x", padx=Spacing.MD, pady=2)
            self._params[name] = pf

        ctk.CTkLabel(rail, text="TONAL FREQUENCIES", font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.ACOUSTIC).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))
        self._tonals = ParameterField(rail, label="Freqs", default="50,120,240", unit="Hz")
        self._tonals.pack(fill="x", padx=Spacing.MD, pady=2)

        self._run_btn = ActionButton(rail, text="RUN ACOUSTIC SIM", command=self._run_sim, primary=True)
        self._run_btn.pack(fill="x", padx=Spacing.MD, pady=Spacing.LG)

        # CENTER: Tabs
        workspace = ctk.CTkFrame(self, fg_color=Colors.BG_DARKEST, corner_radius=0)
        workspace.grid(row=0, column=1, sticky="nsew", padx=1)

        self._tabview = ctk.CTkTabview(workspace, fg_color=Colors.BG_CARD,
                                        segmented_button_fg_color=Colors.BG_SURFACE,
                                        segmented_button_selected_color=Colors.ACOUSTIC,
                                        segmented_button_unselected_color=Colors.BG_ELEVATED)
        self._tabview.pack(fill="both", expand=True, padx=Spacing.GRID_GAP, pady=Spacing.GRID_GAP)

        for name in ["LOFAR", "WAVEFORM", "TONALS"]:
            self._tabview.add(name)
            EmptyState(self._tabview.tab(name), title="NO ACOUSTIC DATA",
                       message="Run the acoustic simulation.").pack(fill="both", expand=True)

        self._chart_frames = {n: self._tabview.tab(n) for n in ["LOFAR", "WAVEFORM", "TONALS"]}

        # RIGHT: Inspector
        inspector = ctk.CTkScrollableFrame(self, fg_color=Colors.BG_CARD, width=280, corner_radius=0,
                                            scrollbar_button_color=Colors.BORDER_SUBTLE)
        inspector.grid(row=0, column=2, sticky="nsew")

        ctk.CTkLabel(inspector, text="PASSIVE SONAR EQUATION", font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        eq_frame = ctk.CTkFrame(inspector, fg_color=Colors.BG_DARKEST, border_color=Colors.BORDER_SUBTLE,
                                border_width=1, corner_radius=Radii.SMALL)
        eq_frame.pack(fill="x", padx=Spacing.MD, pady=Spacing.SM)
        ctk.CTkLabel(eq_frame, text="SE = SL - TL - (NL - DI) - DT", font=(Typography.MONO_FONT, 11),
                     text_color=Colors.TEXT_PRIMARY).pack(padx=Spacing.SM, pady=Spacing.SM)

        ctk.CTkLabel(inspector, text="LIVE VALUES", font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        self._live_values = ctk.CTkTextbox(inspector, font=(Typography.MONO_FONT, 11), text_color=Colors.TEXT_MUTED,
                                            fg_color=Colors.BG_DARKEST, height=120, state="disabled")
        self._live_values.pack(fill="x", padx=Spacing.MD, pady=Spacing.SM)

        self._inference = ctk.CTkTextbox(inspector, font=(Typography.UI_FONT, 11), text_color=Colors.TEXT_SECONDARY,
                                          fg_color="transparent", height=100, wrap="word", state="disabled")
        self._inference.pack(fill="x", padx=Spacing.MD, pady=Spacing.SM)

    def refresh(self, exp: ExperimentState):
        r = exp.acoustic_result
        if r.status != ModuleStatus.COMPLETED:
            return
        self._run_btn.set_running(False)

        self._live_values.configure(state="normal")
        self._live_values.delete("0.0", "end")
        self._live_values.insert("0.0", "\n".join([
            f"SE    = {r.signal_excess_db:.2f} dB" if r.signal_excess_db is not None else "",
            f"TL    = {r.transmission_loss_db:.2f} dB" if r.transmission_loss_db is not None else "",
            f"NL    = {r.noise_level_db:.2f} dB" if r.noise_level_db is not None else "",
        ]))
        self._live_values.configure(state="disabled")

        self._render_charts(exp)

    def _render_charts(self, exp):
        r = exp.acoustic_result
        if r.lofar_image is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from ui.theme.tokens import ChartStyle

            tab = self._chart_frames["LOFAR"]
            for w in tab.winfo_children(): w.destroy()
            fig, ax = plt.subplots(figsize=(8, 4), facecolor=ChartStyle.FIGURE_FACECOLOR)
            ax.set_facecolor(ChartStyle.AXES_FACECOLOR)
            ax.pcolormesh(r.lofar_times, r.lofar_frequencies, r.lofar_image.real, cmap="viridis", shading="auto")
            ax.set_xlabel("Time (s)", color=ChartStyle.LABEL_COLOR)
            ax.set_ylabel("Frequency (Hz)", color=ChartStyle.LABEL_COLOR)
            ax.set_title("LOFAR Spectrogram", color=ChartStyle.TEXT_COLOR)
            ax.tick_params(colors=ChartStyle.TICK_COLOR)
            for s in ax.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        except ImportError:
            pass

    def _run_sim(self):
        self._run_btn.set_running(True)
        self._app._run_pipeline()
