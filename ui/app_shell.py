"""
QT-2.23 — Application Shell

Main CTk window with:
- Left sidebar (210px, nav groups from Frontend manual Section 2)
- Top context bar (56px, page title + subtitle + actions)
- Main workspace (page content, scrollable)
- Bottom status strip (32px, solver status + running task)

Layout per Stitch System Overview screen: grid-based, row/column
weights correctly set, sidebar fixed, workspace fills remaining space.
"""

from __future__ import annotations
import sys
import os
import pickle
from typing import Any, Callable, Dict, Optional

import customtkinter as ctk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.theme.tokens import Colors, Typography, Spacing, Radii, NAV_GROUPS, NAV_ICONS
from app.state import ApplicationState, ExperimentState, ModuleStatus, TaskType
from app.events import EventBus, EventType, Event, event_bus
from app.tasks import TaskManager


class AppShell(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Quantum Core — QT-2.23")
        self.geometry("1920x1080")
        self.minsize(1366, 768)
        self.configure(fg_color=Colors.BG_DARKEST)

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Application state
        self.app_state = ApplicationState()
        self.task_manager = TaskManager(event_bus)

        # Page registry
        self._pages: Dict[str, ctk.CTkFrame] = {}
        self._current_page: Optional[ctk.CTkFrame] = None
        self._nav_buttons: Dict[str, ctk.CTkFrame] = {}

        # Build shell
        self._build_sidebar()
        self._build_topbar()
        self._build_workspace()
        self._build_statusbar()

        # Subscribe to events
        event_bus.subscribe(EventType.TASK_STARTED, self._on_task_event)
        event_bus.subscribe(EventType.TASK_PROGRESS, self._on_task_event)
        event_bus.subscribe(EventType.TASK_STAGE_CHANGED, self._on_task_event)
        event_bus.subscribe(EventType.TASK_COMPLETED, self._on_task_event)
        event_bus.subscribe(EventType.TASK_FAILED, self._on_task_event)

        # Register all pages (lazy import)
        self._register_pages()

        # Load persistent state if available
        self._load_state()

        # Navigate to System Overview
        self.navigate_to("System Overview")

        # Keyboard shortcuts
        self.bind("<F11>", lambda e: self._toggle_presentation_mode())
        self.bind("<Control-n>", lambda e: self._new_experiment())
        self.bind("<Control-r>", lambda e: self._run_pipeline())

    # ============================================================
    # SIDEBAR
    # ============================================================

    def _build_sidebar(self):
        """Left navigation sidebar — 210px, per Stitch screen."""
        self._sidebar = ctk.CTkFrame(
            self, fg_color=Colors.BG_DARKEST,
            width=Spacing.SIDEBAR_WIDTH,
            corner_radius=0,
        )
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)

        # Logo / App name
        logo_frame = ctk.CTkFrame(self._sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", padx=Spacing.LG, pady=(20, Spacing.LG))

        ctk.CTkLabel(
            logo_frame, text="⬡",
            font=(Typography.UI_FONT, 24),
            text_color=Colors.RADAR,
        ).pack(side="left", padx=(0, Spacing.SM))

        name_frame = ctk.CTkFrame(logo_frame, fg_color="transparent")
        name_frame.pack(side="left")
        ctk.CTkLabel(
            name_frame, text="Quantum Core",
            font=(Typography.UI_FONT, 14, "bold"),
            text_color=Colors.TEXT_PRIMARY, anchor="w",
        ).pack(anchor="w")

        # Navigation groups
        nav_scroll = ctk.CTkScrollableFrame(
            self._sidebar, fg_color="transparent",
            scrollbar_button_color=Colors.BORDER_SUBTLE,
        )
        nav_scroll.pack(fill="both", expand=True, padx=0, pady=0)

        for group_name, pages in NAV_GROUPS.items():
            # Group label
            ctk.CTkLabel(
                nav_scroll, text=group_name,
                font=(Typography.UI_FONT, 10, "bold"),
                text_color=Colors.TEXT_MUTED, anchor="w",
            ).pack(fill="x", padx=Spacing.LG, pady=(Spacing.MD, Spacing.XS))

            for page_name in pages:
                self._create_nav_button(nav_scroll, page_name)

        # Bottom: Documentation link
        bottom = ctk.CTkFrame(self._sidebar, fg_color="transparent")
        bottom.pack(fill="x", side="bottom", padx=0, pady=Spacing.SM)

        sep = ctk.CTkFrame(bottom, fg_color=Colors.BORDER_SUBTLE, height=1)
        sep.pack(fill="x", padx=Spacing.LG, pady=(0, Spacing.SM))

    def _create_nav_button(self, parent, page_name: str):
        """Create a single nav item button."""
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent", height=Spacing.NAV_ITEM_HEIGHT)
        btn_frame.pack(fill="x", padx=Spacing.XS, pady=1)
        btn_frame.pack_propagate(False)

        icon = NAV_ICONS.get(page_name, "•")

        btn = ctk.CTkButton(
            btn_frame,
            text=f"  {icon}  {page_name.upper()}",
            font=(Typography.UI_FONT, 11, "bold"),
            text_color=Colors.TEXT_SECONDARY,
            fg_color="transparent",
            hover_color=Colors.BG_ELEVATED,
            anchor="w",
            height=Spacing.NAV_ITEM_HEIGHT,
            corner_radius=Radii.SMALL,
            command=lambda p=page_name: self.navigate_to(p),
        )
        btn.pack(fill="both", expand=True)

        self._nav_buttons[page_name] = btn

    # ============================================================
    # TOPBAR
    # ============================================================

    def _build_topbar(self):
        """Top context bar — 56px, page title + actions."""
        self._topbar = ctk.CTkFrame(
            self, fg_color=Colors.BG_SURFACE,
            height=Spacing.TOPBAR_HEIGHT,
            corner_radius=0,
        )
        self._topbar.pack(fill="x", side="top")
        self._topbar.pack_propagate(False)

        # Title area
        title_frame = ctk.CTkFrame(self._topbar, fg_color="transparent")
        title_frame.pack(side="left", padx=Spacing.XL, fill="y")

        self._page_title = ctk.CTkLabel(
            title_frame, text="System Overview",
            font=(Typography.UI_FONT, 22, "bold"),
            text_color=Colors.TEXT_PRIMARY, anchor="w",
        )
        self._page_title.pack(anchor="sw", pady=(8, 0))

        self._page_subtitle = ctk.CTkLabel(
            title_frame, text="Which sensors contain evidence, and what does fusion currently trust?",
            font=(Typography.UI_FONT, 11),
            text_color=Colors.TEXT_SECONDARY, anchor="w",
        )
        self._page_subtitle.pack(anchor="nw")

        # Right actions
        actions = ctk.CTkFrame(self._topbar, fg_color="transparent")
        actions.pack(side="right", padx=Spacing.XL, fill="y")

        self._experiment_label = ctk.CTkLabel(
            actions,
            text="QTMReg_109-Wandering Wight",
            font=(Typography.MONO_FONT, 11),
            text_color=Colors.TEXT_MUTED,
        )
        self._experiment_label.pack(side="right", pady=Spacing.MD)

    # ============================================================
    # WORKSPACE
    # ============================================================

    def _build_workspace(self):
        """Main content area — fills remaining space."""
        self._workspace = ctk.CTkFrame(
            self, fg_color=Colors.BG_DARKEST, corner_radius=0,
        )
        self._workspace.pack(fill="both", expand=True, side="top")

    # ============================================================
    # STATUS BAR
    # ============================================================

    def _build_statusbar(self):
        """Bottom status strip — 32px, solver status + task."""
        self._statusbar = ctk.CTkFrame(
            self, fg_color=Colors.BG_SURFACE,
            height=Spacing.STATUSBAR_HEIGHT,
            corner_radius=0,
        )
        self._statusbar.pack(fill="x", side="bottom")
        self._statusbar.pack_propagate(False)

        # Solver status (left)
        self._solver_status = ctk.CTkLabel(
            self._statusbar,
            text="Quantum solver (QUBO/dimod): unavailable — classical simulated annealing fallback active.",
            font=(Typography.MONO_FONT, 10),
            text_color=Colors.TEXT_MUTED, anchor="w",
        )
        self._solver_status.pack(side="left", padx=Spacing.LG)

        # Task status (right)
        self._task_status = ctk.CTkLabel(
            self._statusbar,
            text="IDLE",
            font=(Typography.MONO_FONT, 10),
            text_color=Colors.TEXT_MUTED, anchor="e",
        )
        self._task_status.pack(side="right", padx=Spacing.LG)

        # Progress bar (center)
        self._progress = ctk.CTkProgressBar(
            self._statusbar,
            fg_color=Colors.BG_CARD,
            progress_color=Colors.RADAR,
            height=4, corner_radius=2,
            width=300,
        )
        self._progress.pack(side="right", padx=Spacing.LG, pady=Spacing.SM)
        self._progress.set(0)

    # ============================================================
    # PAGE TITLES — per Frontend manual Section 29
    # ============================================================

    PAGE_SUBTITLES = {
        "System Overview": "Which sensors contain evidence, and what does fusion currently trust?",
        "Radar": "Why is the radar target difficult to detect?",
        "Thermal / IR": "How much radiative contrast survives stealth suppression?",
        "Acoustic / Sonar": "Are target tonal components visible above ambient noise?",
        "Feature Space": "Which evidence is useful without being redundant?",
        "Fusion Optimization": "Which sensor trust distribution maximizes target/background separation?",
        "Baselines": "Does optimized fusion actually outperform credible classical alternatives?",
        "Contribution": "Which sensor materially changes detection performance?",
        "Degradation Lab": "When one sensor becomes unreliable, does trust move and does detection survive?",
        "Scaling": "How does optimizer speed and solution quality change as the problem grows?",
        "Experiments": "Are we comparing equivalent experimental conditions?",
        "Solver": "What optimization problem actually ran and what did the solver return?",
        "Logs": "Scientific application event log",
        "Settings": "Application and experiment configuration",
    }

    # ============================================================
    # NAVIGATION
    # ============================================================

    def navigate_to(self, page_name: str):
        """Switch to a page. Creates the page lazily if needed."""
        # Update nav button states
        for name, btn in self._nav_buttons.items():
            if name == page_name:
                btn.configure(
                    text_color=Colors.RADAR,
                    fg_color=Colors.BG_ELEVATED,
                )
            else:
                btn.configure(
                    text_color=Colors.TEXT_SECONDARY,
                    fg_color="transparent",
                )

        # Update topbar
        self._page_title.configure(text=page_name)
        subtitle = self.PAGE_SUBTITLES.get(page_name, "")
        self._page_subtitle.configure(text=subtitle)

        # Hide current page
        if self._current_page:
            self._current_page.pack_forget()

        # Show or create new page
        if page_name not in self._pages:
            self._pages[page_name] = self._create_page(page_name)

        page = self._pages[page_name]
        page.pack(in_=self._workspace, fill="both", expand=True)
        self._current_page = page

        # Update state
        self.app_state.ui.current_page = page_name

        # Refresh page data if it has a refresh method
        if hasattr(page, 'refresh'):
            page.refresh(self.app_state.current_experiment)

    def _register_pages(self):
        """Pre-register page factories. Pages are created lazily."""
        pass  # Pages created on first navigate_to

    def _create_page(self, page_name: str) -> ctk.CTkFrame:
        """Create a page by name. Imports lazily to avoid circular deps."""
        from ui.pages import (
            MissionControlPage, RadarPage, ThermalPage, AcousticPage,
            FeatureSpacePage, FusionPage, BaselinesPage, ContributionPage,
            DegradationPage, ScalingPage, ExperimentsPage, SolverPage,
            LogsPage, SettingsPage,
        )

        page_map = {
            "System Overview": MissionControlPage,
            "Radar": RadarPage,
            "Thermal / IR": ThermalPage,
            "Acoustic / Sonar": AcousticPage,
            "Feature Space": FeatureSpacePage,
            "Fusion Optimization": FusionPage,
            "Baselines": BaselinesPage,
            "Contribution": ContributionPage,
            "Degradation Lab": DegradationPage,
            "Scaling": ScalingPage,
            "Experiments": ExperimentsPage,
            "Solver": SolverPage,
            "Logs": LogsPage,
            "Settings": SettingsPage,
        }

        PageClass = page_map.get(page_name)
        if PageClass:
            return PageClass(self._workspace, self)
        else:
            # Fallback empty page
            from ui.components import EmptyState
            frame = ctk.CTkFrame(self._workspace, fg_color=Colors.BG_DARKEST)
            EmptyState(frame, title=f"{page_name}", message="Page not implemented yet").pack(
                fill="both", expand=True
            )
            return frame

    # ============================================================
    # EVENT HANDLERS
    # ============================================================

    def _on_task_event(self, event: Event):
        """Handle task lifecycle events — marshal to main thread."""
        self.after(0, self._update_statusbar, event)

    def _update_statusbar(self, event: Event):
        """Update status bar from main thread."""
        if event.event_type == EventType.TASK_STARTED:
            self._task_status.configure(
                text=f"RUNNING: {event.task_type or ''}",
                text_color=Colors.WARNING,
            )
        elif event.event_type == EventType.TASK_PROGRESS:
            if event.progress is not None:
                self._progress.set(event.progress)
            if event.message:
                self._task_status.configure(text=event.message)
        elif event.event_type == EventType.TASK_STAGE_CHANGED:
            self._task_status.configure(
                text=f"{event.stage}: {event.message or ''}",
                text_color=Colors.RADAR,
            )
            if event.progress is not None:
                self._progress.set(event.progress)
        elif event.event_type == EventType.TASK_COMPLETED:
            self._task_status.configure(text="COMPLETED", text_color=Colors.SUCCESS)
            self._progress.set(1.0)

            # Refresh current page
            if self._current_page and hasattr(self._current_page, 'refresh'):
                self._current_page.refresh(self.app_state.current_experiment)
        elif event.event_type == EventType.TASK_FAILED:
            self._task_status.configure(
                text=f"FAILED: {event.error or ''}",
                text_color=Colors.CRITICAL,
            )

    # ============================================================
    # ACTIONS
    # ============================================================

    def propagate_experiment_state(self):
        """Force refresh on all loaded pages to synchronize visual fields with the state."""
        for page in self._pages.values():
            if hasattr(page, 'refresh'):
                page.refresh(self.app_state.current_experiment)

    def _new_experiment(self):
        """Create a new experiment."""
        self.app_state.current_experiment = ExperimentState()
        self._experiment_label.configure(
            text="QTMReg_109-Wandering Wight"
        )
        # Refresh current page
        self.propagate_experiment_state()

    def _run_pipeline(self):
        """Run the full scientific pipeline in background."""
        from app.pipeline import run_full_pipeline

        self.task_manager.submit(
            TaskType.FULL_PIPELINE,
            self.app_state.current_experiment.experiment_id,
            run_full_pipeline,
            self.app_state.current_experiment,
        )

    def _toggle_presentation_mode(self):
        """Toggle presentation mode."""
        self.app_state.ui.presentation_mode = not self.app_state.ui.presentation_mode
        if self.app_state.ui.presentation_mode:
            self._sidebar.pack_forget()
            self.attributes("-fullscreen", True)
        else:
            self._sidebar.pack(side="left", fill="y", before=self._topbar)
            self.attributes("-fullscreen", False)

    def _load_state(self):
        """Load experiment state from disk so settings persist across reboots."""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".qt223_settings.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.app_state.current_experiment = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load saved settings: {e}")

    def _save_state(self):
        """Save experiment state to disk so settings persist."""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".qt223_settings.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump(self.app_state.current_experiment, f)
        except Exception as e:
            print(f"Warning: Could not save settings: {e}")

    def on_closing(self):
        self._save_state()
        self.task_manager.shutdown()
        self.destroy()
        os._exit(0)
