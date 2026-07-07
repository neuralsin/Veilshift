"""QT-2.23 — Thermal / IR Sensor Page"""
from __future__ import annotations
import customtkinter as ctk
import numpy as np
from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import SectionFrame, ParameterField, ActionButton, EmptyState, StatusPill
from app.state import ExperimentState, ModuleStatus


class ThermalPage(ctk.CTkFrame):
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

        ctk.CTkLabel(rail, text="THERMAL PHYSICS", font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.THERMAL).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        self._params = {}
        for name, default, unit in [
            ("T_bg", "290", "K"), ("ε_bg", "0.95", ""),
            ("T_tgt", "310", "K"), ("ε_tgt", "0.30", ""),
            ("NETD", "0.05", "K"), ("α (1/f)", "1.0", ""),
            ("Grid", "64", "px"),
        ]:
            pf = ParameterField(rail, label=name, default=default, unit=unit)
            pf.pack(fill="x", padx=Spacing.MD, pady=2)
            self._params[name] = pf

        self._run_btn = ActionButton(rail, text="RUN THERMAL SIM", command=self._run_sim, primary=True)
        self._run_btn.pack(fill="x", padx=Spacing.MD, pady=Spacing.LG)

        # CENTER: Tabs
        workspace = ctk.CTkFrame(self, fg_color=Colors.BG_DARKEST, corner_radius=0)
        workspace.grid(row=0, column=1, sticky="nsew", padx=1)

        self._tabview = ctk.CTkTabview(workspace, fg_color=Colors.BG_CARD,
                                        segmented_button_fg_color=Colors.BG_SURFACE,
                                        segmented_button_selected_color=Colors.THERMAL,
                                        segmented_button_unselected_color=Colors.BG_ELEVATED)
        self._tabview.pack(fill="both", expand=True, padx=Spacing.GRID_GAP, pady=Spacing.GRID_GAP)

        for name in ["THERMAL FRAME", "CONTRAST MAP", "1/f PSD", "NOISE FIELD"]:
            self._tabview.add(name)
            EmptyState(self._tabview.tab(name), title="NO THERMAL DATA",
                       message="Run the thermal simulation to generate frames.").pack(fill="both", expand=True)

        self._chart_frames = {n: self._tabview.tab(n) for n in ["THERMAL FRAME", "CONTRAST MAP", "1/f PSD", "NOISE FIELD"]}

        # RIGHT: Inspector
        inspector = ctk.CTkScrollableFrame(self, fg_color=Colors.BG_CARD, width=280, corner_radius=0,
                                            scrollbar_button_color=Colors.BORDER_SUBTLE)
        inspector.grid(row=0, column=2, sticky="nsew")

        ctk.CTkLabel(inspector, text="STEFAN-BOLTZMANN", font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        eq_frame = ctk.CTkFrame(inspector, fg_color=Colors.BG_DARKEST, border_color=Colors.BORDER_SUBTLE,
                                border_width=1, corner_radius=Radii.SMALL)
        eq_frame.pack(fill="x", padx=Spacing.MD, pady=Spacing.SM)
        ctk.CTkLabel(eq_frame, text="M = ε·σ_SB·T⁴", font=(Typography.MONO_FONT, 11),
                     text_color=Colors.TEXT_PRIMARY).pack(padx=Spacing.SM, pady=Spacing.SM)

        ctk.CTkLabel(inspector, text="LIVE VALUES", font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        self._live_values = ctk.CTkTextbox(inspector, font=(Typography.MONO_FONT, 11), text_color=Colors.TEXT_MUTED,
                                            fg_color=Colors.BG_DARKEST, height=120, state="disabled")
        self._live_values.pack(fill="x", padx=Spacing.MD, pady=Spacing.SM)

        ctk.CTkLabel(inspector, text="PHYSICAL INFERENCE", font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(fill="x", padx=Spacing.MD, pady=(Spacing.LG, Spacing.SM))

        self._inference = ctk.CTkTextbox(inspector, font=(Typography.UI_FONT, 11), text_color=Colors.TEXT_SECONDARY,
                                          fg_color="transparent", height=100, wrap="word", state="disabled")
        self._inference.pack(fill="x", padx=Spacing.MD, pady=Spacing.SM)

    def refresh(self, exp: ExperimentState):
        r = exp.thermal_result
        if r.status != ModuleStatus.COMPLETED:
            return
        self._run_btn.set_running(False)

        self._live_values.configure(state="normal")
        self._live_values.delete("0.0", "end")
        self._live_values.insert("0.0", "\n".join([
            f"M_target  = {r.target_exitance:.2f} W/m²" if r.target_exitance else "",
            f"M_bg      = {r.background_exitance:.2f} W/m²" if r.background_exitance else "",
            f"ΔM        = {r.delta_M:.4f} W/m²" if r.delta_M else "",
            f"SNR_th    = {r.thermal_snr:.3f}" if r.thermal_snr else "",
            f"PSD slope = {r.noise_psd_slope:.2f}" if r.noise_psd_slope else "",
        ]))
        self._live_values.configure(state="disabled")

        # Inference
        self._inference.configure(state="normal")
        self._inference.delete("0.0", "end")
        lines = []
        if r.delta_M and r.delta_M < 1.0:
            lines.append(f"Contrast is {r.delta_M:.4f} W/m² — near the noise floor.")
        if r.noise_psd_slope and abs(r.noise_psd_slope - (-1.0)) < 0.3:
            lines.append(f"1/f noise validated: PSD slope = {r.noise_psd_slope:.2f} (expected ≈ -1.0)")
        self._inference.insert("0.0", "\n".join(lines))
        self._inference.configure(state="disabled")

        self._render_charts(exp)

    def _render_charts(self, exp):
        r = exp.thermal_result
        if r.thermal_frame is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from ui.theme.tokens import ChartStyle

            # Thermal frame
            tab = self._chart_frames["THERMAL FRAME"]
            for w in tab.winfo_children(): w.destroy()
            fig, ax = plt.subplots(figsize=(6, 5), facecolor=ChartStyle.FIGURE_FACECOLOR)
            ax.set_facecolor(ChartStyle.AXES_FACECOLOR)
            im = ax.imshow(r.thermal_frame, cmap="inferno", aspect="equal")
            ax.set_title("Thermal Frame (K)", color=ChartStyle.TEXT_COLOR, fontsize=ChartStyle.TITLE_SIZE)
            ax.tick_params(colors=ChartStyle.TICK_COLOR)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Temperature (K)", color=ChartStyle.LABEL_COLOR)
            cbar.ax.tick_params(colors=ChartStyle.TICK_COLOR)
            for s in ax.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            # 1/f PSD
            if r.noise_psd_freqs is not None and r.noise_psd_power is not None:
                tab3 = self._chart_frames["1/f PSD"]
                for w in tab3.winfo_children(): w.destroy()
                fig3, ax3 = plt.subplots(figsize=(6, 4), facecolor=ChartStyle.FIGURE_FACECOLOR)
                ax3.set_facecolor(ChartStyle.AXES_FACECOLOR)
                ax3.loglog(r.noise_psd_freqs, r.noise_psd_power, color=Colors.THERMAL, linewidth=1.5)
                ax3.set_xlabel("Frequency", color=ChartStyle.LABEL_COLOR)
                ax3.set_ylabel("Power", color=ChartStyle.LABEL_COLOR)
                ax3.set_title("Noise PSD (1/f validation)", color=ChartStyle.TEXT_COLOR)
                ax3.grid(True, color=ChartStyle.GRID_COLOR, alpha=ChartStyle.GRID_ALPHA)
                ax3.tick_params(colors=ChartStyle.TICK_COLOR)
                for s in ax3.spines.values(): s.set_color(ChartStyle.AXES_EDGECOLOR)
                fig3.tight_layout()
                canvas3 = FigureCanvasTkAgg(fig3, master=tab3)
                canvas3.draw()
                canvas3.get_tk_widget().pack(fill="both", expand=True)
        except ImportError:
            pass

    def _run_sim(self):
        self._run_btn.set_running(True)
        self._app._run_pipeline()
