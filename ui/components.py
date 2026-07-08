"""
QT-2.23 — UI Components

Reusable CustomTkinter components used across all 14 pages.
All visual constants come from ui.theme.tokens — never hardcoded here.
"""

from __future__ import annotations
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import customtkinter as ctk
from ui.theme.tokens import Colors, Typography, Spacing, Radii


# ============================================================
# METRIC CARD — shows a primary metric with optional CI
# ============================================================

class MetricCard(ctk.CTkFrame):
    """
    Card showing a single headline metric.
    Supports: label, value, CI, top accent border.
    """

    def __init__(
        self, master,
        label: str = "",
        value: str = "—",
        ci_text: str = "",
        accent_color: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            master,
            fg_color=Colors.BG_CARD,
            border_color=Colors.BORDER_SUBTLE,
            border_width=1,
            corner_radius=Radii.DEFAULT,
            height=Spacing.METRIC_CARD_HEIGHT,
            **kwargs,
        )

        self.grid_propagate(False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Top accent line
        if accent_color:
            accent = ctk.CTkFrame(self, fg_color=accent_color, height=2, corner_radius=0)
            accent.grid(row=0, column=0, sticky="new", padx=0, pady=0)

        # Label
        self._label = ctk.CTkLabel(
            self, text=label.upper(),
            font=(Typography.UI_FONT, 11, "bold"),
            text_color=Colors.TEXT_SECONDARY,
            anchor="w",
        )
        self._label.grid(row=0, column=0, sticky="nw", padx=Spacing.LG, pady=(Spacing.LG, 2))

        # Value + CI container
        val_frame = ctk.CTkFrame(self, fg_color="transparent")
        val_frame.grid(row=1, column=0, sticky="sw", padx=Spacing.LG, pady=(0, Spacing.LG))

        font_size = 28 if len(value) <= 10 else 18
        self._value = ctk.CTkLabel(
            val_frame, text=value,
            font=(Typography.MONO_FONT, font_size, "bold"),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        self._value.pack(side="left")

        if ci_text:
            self._ci = ctk.CTkLabel(
                val_frame, text=ci_text,
                font=(Typography.UI_FONT, 11),
                text_color=Colors.TEXT_MUTED,
                anchor="w",
            )
            self._ci.pack(side="left", padx=(Spacing.SM, 0))
        else:
            self._ci = None

    def update_value(self, value: str, ci_text: str = ""):
        font_size = 28 if len(value) <= 10 else 18
        self._value.configure(text=value, font=(Typography.MONO_FONT, font_size, "bold"))
        if self._ci and ci_text:
            self._ci.configure(text=ci_text)


# ============================================================
# SENSOR CARD — compact sensor evidence card
# ============================================================

class SensorCard(ctk.CTkFrame):
    """
    Compact sensor card showing mini-viz, score, weight, rank, SNR.
    Used on Mission Control page.
    """

    def __init__(
        self, master,
        sensor_name: str = "Radar",
        sensor_color: str = Colors.RADAR,
        status: str = "Online",
        score: str = "—",
        weight: str = "—",
        rank: str = "—",
        snr: str = "—",
        **kwargs,
    ):
        super().__init__(
            master,
            fg_color=Colors.BG_CARD,
            border_color=Colors.BORDER_SUBTLE,
            border_width=1,
            corner_radius=Radii.DEFAULT,
            **kwargs,
        )

        self.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color=Colors.BG_SURFACE, corner_radius=0, height=36)
        header.grid(row=0, column=0, sticky="new", padx=0, pady=0)
        header.grid_columnconfigure(0, weight=1)
        header.grid_propagate(False)

        title = ctk.CTkLabel(
            header, text=sensor_name,
            font=(Typography.UI_FONT, 14, "bold"),
            text_color=Colors.TEXT_PRIMARY, anchor="w",
        )
        title.grid(row=0, column=0, sticky="w", padx=Spacing.LG, pady=Spacing.SM)

        # Status pill
        status_colors = {
            "Online": Colors.SUCCESS,
            "Degraded": Colors.WARNING,
            "Failed": Colors.CRITICAL,
            "Pending": Colors.TEXT_MUTED,
        }
        status_color = status_colors.get(status, Colors.TEXT_MUTED)
        self._status = ctk.CTkLabel(
            header, text=status.upper(),
            font=(Typography.UI_FONT, 11, "bold"),
            text_color=status_color, anchor="e",
        )
        self._status.grid(row=0, column=1, sticky="e", padx=Spacing.LG, pady=Spacing.SM)

        # Viz placeholder (will be replaced with matplotlib canvas)
        viz_frame = ctk.CTkFrame(
            self, fg_color=Colors.BG_DARKEST,
            border_color=Colors.BORDER_SUBTLE, border_width=1,
            corner_radius=Radii.SMALL, height=120,
        )
        viz_frame.grid(row=1, column=0, columnspan=2, sticky="new",
                       padx=Spacing.LG, pady=(Spacing.LG, Spacing.SM))
        viz_frame.grid_propagate(False)
        self._viz_frame = viz_frame

        # Metrics row
        metrics = ctk.CTkFrame(self, fg_color="transparent")
        metrics.grid(row=2, column=0, columnspan=2, sticky="sew",
                     padx=Spacing.LG, pady=(Spacing.SM, Spacing.LG))

        self._metric_labels = {}
        for col, (lbl, val, color) in enumerate([
            ("OOF AUC", score, Colors.TEXT_PRIMARY),
            ("Weight", weight, sensor_color),
            ("Rank", rank, Colors.TEXT_PRIMARY),
            ("SNR dB", snr, Colors.WARNING if snr.startswith("-") else Colors.TEXT_PRIMARY),
        ]):
            metrics.grid_columnconfigure(col, weight=1)
            f = ctk.CTkFrame(metrics, fg_color="transparent")
            f.grid(row=0, column=col, sticky="w")
            ctk.CTkLabel(f, text=lbl, font=(Typography.UI_FONT, 11),
                         text_color=Colors.TEXT_SECONDARY).pack(anchor="w")
            l = ctk.CTkLabel(f, text=val, font=(Typography.MONO_FONT, 14),
                             text_color=color)
            l.pack(anchor="w")
            self._metric_labels[lbl] = l

    def update_metrics(self, status: Optional[str] = None, score: Optional[str] = None, 
                       weight: Optional[str] = None, rank: Optional[str] = None, snr: Optional[str] = None):
        """Update the metrics displayed on the card."""
        if status is not None:
            status_colors = {
                "Online": Colors.SUCCESS,
                "Degraded": Colors.WARNING,
                "Failed": Colors.CRITICAL,
                "Pending": Colors.TEXT_MUTED,
            }
            # Also handle upper case values for matching
            color = Colors.SUCCESS if status.upper() in ["ONLINE", "COMPLETED", "PASS"] else \
                    Colors.WARNING if status.upper() in ["DEGRADED", "CHECK", "RUNNING"] else \
                    Colors.CRITICAL if status.upper() in ["FAILED", "ERROR"] else \
                    Colors.TEXT_MUTED
            self._status.configure(text=status.upper(), text_color=color)
        
        if score is not None and "OOF AUC" in self._metric_labels:
            self._metric_labels["OOF AUC"].configure(text=score)
        
        if weight is not None and "Weight" in self._metric_labels:
            self._metric_labels["Weight"].configure(text=weight)
            
        if rank is not None and "Rank" in self._metric_labels:
            self._metric_labels["Rank"].configure(text=rank)
            
        if snr is not None and "SNR dB" in self._metric_labels:
            color = Colors.WARNING if snr.startswith("-") else Colors.TEXT_PRIMARY
            self._metric_labels["SNR dB"].configure(text=snr, text_color=color)

# ============================================================
# STATUS PILL — inline status indicator
# ============================================================

class StatusPill(ctk.CTkLabel):
    """Small status pill: PASS / CHECK / FAILED / RUNNING etc."""

    STATUS_COLORS = {
        "PASS": Colors.SUCCESS,
        "COMPLETED": Colors.SUCCESS,
        "ONLINE": Colors.SUCCESS,
        "CHECK": Colors.WARNING,
        "RUNNING": Colors.WARNING,
        "DEGRADED": Colors.WARNING,
        "FAILED": Colors.CRITICAL,
        "ERROR": Colors.CRITICAL,
        "PENDING": Colors.TEXT_MUTED,
        "NOT_CONFIGURED": Colors.TEXT_MUTED,
    }

    def __init__(self, master, status: str = "PENDING", **kwargs):
        color = self.STATUS_COLORS.get(status.upper(), Colors.TEXT_MUTED)
        super().__init__(
            master,
            text=status.upper(),
            font=(Typography.UI_FONT, 11, "bold"),
            text_color=color,
            **kwargs,
        )

    def set_status(self, status: str):
        color = self.STATUS_COLORS.get(status.upper(), Colors.TEXT_MUTED)
        self.configure(text=status.upper(), text_color=color)


# ============================================================
# SECTION FRAME — titled card/section container
# ============================================================

class SectionFrame(ctk.CTkFrame):
    """Card container with a section heading."""

    def __init__(self, master, title: str = "", **kwargs):
        super().__init__(
            master,
            fg_color=Colors.BG_CARD,
            border_color=Colors.BORDER_SUBTLE,
            border_width=1,
            corner_radius=Radii.DEFAULT,
            **kwargs,
        )

        if title:
            header = ctk.CTkFrame(self, fg_color=Colors.BG_SURFACE, corner_radius=0, height=36)
            header.pack(fill="x", side="top")
            header.pack_propagate(False)
            self.title_label = ctk.CTkLabel(
                header, text=title,
                font=(Typography.UI_FONT, 14, "bold"),
                text_color=Colors.TEXT_PRIMARY, anchor="w",
            )
            self.title_label.pack(side="left", padx=Spacing.LG, pady=Spacing.SM)
        else:
            self.title_label = None

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=Spacing.LG, pady=Spacing.LG)

    def set_title(self, title: str):
        if self.title_label:
            self.title_label.configure(text=title)


# ============================================================
# PARAMETER GROUP — label + entry field for sensor config pages
# ============================================================

class ParameterField(ctk.CTkFrame):
    """Single parameter input: label, entry, unit label."""

    def __init__(
        self, master,
        label: str = "",
        default: str = "",
        unit: str = "",
        tooltip: str = "",
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self, text=label,
            font=(Typography.UI_FONT, 11),
            text_color=Colors.TEXT_SECONDARY, anchor="w",
            width=120,
        ).grid(row=0, column=0, sticky="w", padx=(0, Spacing.SM))

        self._entry = ctk.CTkEntry(
            self, height=Spacing.INPUT_HEIGHT,
            font=(Typography.MONO_FONT, 12),
            fg_color=Colors.BG_DARKEST,
            border_color=Colors.BORDER_SUBTLE,
            text_color=Colors.TEXT_PRIMARY,
            corner_radius=Radii.SMALL,
        )
        self._entry.grid(row=0, column=1, sticky="ew")
        self._entry.insert(0, default)

        if unit:
            ctk.CTkLabel(
                self, text=unit,
                font=(Typography.UI_FONT, 11),
                text_color=Colors.TEXT_MUTED, anchor="w",
                width=40,
            ).grid(row=0, column=2, sticky="w", padx=(Spacing.XS, 0))

    @property
    def value(self) -> str:
        return self._entry.get()

    def set_value(self, val: str):
        self._entry.delete(0, "end")
        self._entry.insert(0, val)


# ============================================================
# ACTION BUTTON — primary/secondary with running state
# ============================================================

class ActionButton(ctk.CTkButton):
    """Button with running state timer."""

    def __init__(
        self, master,
        text: str = "RUN",
        command: Optional[Callable] = None,
        primary: bool = True,
        **kwargs,
    ):
        fg = Colors.RADAR if primary else Colors.BG_ELEVATED
        text_color = Colors.BG_DARKEST if primary else Colors.TEXT_PRIMARY
        hover = "#2DA8E0" if primary else Colors.BORDER_HOVER

        super().__init__(
            master, text=text,
            font=(Typography.UI_FONT, 11, "bold"),
            fg_color=fg, text_color=text_color,
            hover_color=hover,
            height=Spacing.BUTTON_HEIGHT,
            corner_radius=Radii.SMALL,
            command=command,
            **kwargs,
        )

        self._original_text = text
        self._running = False
        self._start_time = 0.0

    def set_running(self, running: bool = True):
        self._running = running
        if running:
            self._start_time = time.time()
            self.configure(state="disabled")
            self._update_timer()
        else:
            self.configure(text=self._original_text, state="normal")

    def _update_timer(self):
        if not self._running:
            return
        elapsed = time.time() - self._start_time
        self.configure(text=f"{self._original_text.split()[0]}ING · {elapsed:.1f} s")
        self.after(100, self._update_timer)


# ============================================================
# EMPTY STATE — displayed when no data is available
# ============================================================

class EmptyState(ctk.CTkFrame):
    """Empty state with title, message, and optional action buttons."""

    def __init__(
        self, master,
        title: str = "NO DATA",
        message: str = "",
        buttons: Optional[List[Tuple[str, Callable]]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)

        ctk.CTkLabel(
            self, text=title,
            font=(Typography.UI_FONT, 16, "bold"),
            text_color=Colors.TEXT_MUTED,
        ).pack(pady=(40, 8))

        if message:
            ctk.CTkLabel(
                self, text=message,
                font=(Typography.UI_FONT, 12),
                text_color=Colors.TEXT_MUTED,
                wraplength=400,
            ).pack(pady=(0, 20))

        if buttons:
            btn_frame = ctk.CTkFrame(self, fg_color="transparent")
            btn_frame.pack()
            for text, cmd in buttons:
                ActionButton(btn_frame, text=text, command=cmd, primary=False).pack(
                    side="left", padx=Spacing.SM,
                )
