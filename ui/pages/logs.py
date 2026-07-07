"""QT-2.23 — Event Logs Page"""
from __future__ import annotations
import customtkinter as ctk
from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import SectionFrame, ActionButton
from app.state import ExperimentState
from app.events import event_bus


class LogsPage(ctk.CTkFrame):
    def __init__(self, master, app_shell):
        super().__init__(master, fg_color=Colors.BG_DARKEST)
        self._app = app_shell
        self._build_ui()

    def _build_ui(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=Spacing.PAGE_PADDING, pady=Spacing.PAGE_PADDING)

        # Header
        header = ctk.CTkFrame(container, fg_color="transparent")
        header.pack(fill="x", pady=(0, Spacing.GRID_GAP))

        ctk.CTkLabel(header, text=f"EVENT LOG",
                     font=(Typography.UI_FONT, 14, "bold"),
                     text_color=Colors.TEXT_PRIMARY).pack(side="left")

        ActionButton(header, text="CLEAR", command=self._clear, primary=False).pack(side="right")
        ActionButton(header, text="REFRESH", command=lambda: self.refresh(self._app.app_state.current_experiment),
                     primary=False).pack(side="right", padx=Spacing.SM)

        # Log viewer
        self._log_text = ctk.CTkTextbox(
            container, font=(Typography.MONO_FONT, 10),
            text_color=Colors.TEXT_SECONDARY, fg_color=Colors.BG_CARD,
            border_color=Colors.BORDER_SUBTLE, border_width=1,
            corner_radius=Radii.DEFAULT,
            wrap="none",
            state="disabled",
        )
        self._log_text.pack(fill="both", expand=True)

        self._log_text.tag_config("INFO", foreground=Colors.TEXT_SECONDARY)
        self._log_text.tag_config("WARNING", foreground=Colors.WARNING)
        self._log_text.tag_config("ERROR", foreground=Colors.CRITICAL)
        self._log_text.tag_config("SUCCESS", foreground=Colors.SUCCESS)
        self._log_text.tag_config("RADAR", foreground=Colors.RADAR)
        self._log_text.tag_config("THERMAL", foreground=Colors.THERMAL)
        self._log_text.tag_config("ACOUSTIC", foreground=Colors.ACOUSTIC)

    def refresh(self, exp: ExperimentState):
        log_entries = event_bus.get_log_entries(limit=500)

        self._log_text.configure(state="normal")
        self._log_text.delete("0.0", "end")

        for entry in log_entries:
            ts = entry.time_str
            source = entry.module
            event_type = entry.level
            message = entry.message

            line = f"[{ts}] [{source:>12}] {event_type}: {message}\n"

            # Color tag based on source
            tag = "INFO"
            if "RADAR" in source:
                tag = "RADAR"
            elif "THERMAL" in source:
                tag = "THERMAL"
            elif "ACOUSTIC" in source:
                tag = "ACOUSTIC"
            elif "FAIL" in event_type or "ERROR" in event_type:
                tag = "ERROR"
            elif "COMPLETED" in event_type or "SUCCESS" in message.upper():
                tag = "SUCCESS"
            elif "WARNING" in event_type:
                tag = "WARNING"

            self._log_text.insert("end", line, tag)

        self._log_text.configure(state="disabled")
        self._log_text.see("end")

    def _clear(self):
        event_bus.clear_logs()
        self.refresh(self._app.app_state.current_experiment)
