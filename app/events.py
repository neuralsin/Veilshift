"""
QT-2.23 — Event Bus

Thread-safe event bus for decoupling science modules from UI updates.
Events are published by TaskManager workers and consumed by UI pages
via root.after() marshalling. Also drives the structured logger and
the in-app Logs page.
"""

from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("qt223.events")


class EventType(Enum):
    """All event types in the application."""

    # Task lifecycle
    TASK_STARTED = auto()
    TASK_STAGE_CHANGED = auto()
    TASK_PROGRESS = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()

    # Module result updates — one per science module
    RADAR_RESULT_UPDATED = auto()
    THERMAL_RESULT_UPDATED = auto()
    ACOUSTIC_RESULT_UPDATED = auto()
    FEATURE_EXTRACTION_COMPLETED = auto()
    CRLB_RESULT_UPDATED = auto()
    FEATURE_SELECTION_RESULT_UPDATED = auto()
    FUSION_RESULT_UPDATED = auto()
    METRICS_RESULT_UPDATED = auto()
    BASELINE_RESULT_UPDATED = auto()
    CONTRIBUTION_RESULT_UPDATED = auto()
    DEGRADATION_RESULT_UPDATED = auto()
    SCALING_RESULT_UPDATED = auto()

    # Sensor model training
    SENSOR_MODELS_TRAINED = auto()

    # UI navigation / state
    PAGE_CHANGED = auto()
    EXPERIMENT_LOADED = auto()
    EXPERIMENT_SAVED = auto()
    EXPERIMENT_CREATED = auto()

    # Settings
    SETTINGS_CHANGED = auto()

    # Presentation mode
    PRESENTATION_STAGE_CHANGED = auto()


@dataclass
class Event:
    """An event payload published through the event bus."""
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Module or component that emitted it

    # Task-specific fields (set when event_type is TASK_*)
    task_type: Optional[str] = None
    stage: Optional[str] = None
    progress: Optional[float] = None  # 0.0 to 1.0
    message: Optional[str] = None
    elapsed_s: Optional[float] = None
    error: Optional[str] = None


@dataclass
class LogEntry:
    """Structured log entry for the in-app Logs page."""
    timestamp: float
    module: str
    level: str  # INFO, WARNING, ERROR, PASS, CHECK, FAILED
    message: str
    experiment_id: str = ""

    @property
    def time_str(self) -> str:
        t = time.localtime(self.timestamp)
        ms = int((self.timestamp % 1) * 1000)
        return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ms:03d}"


class EventBus:
    """
    Thread-safe publish/subscribe event bus.

    Subscribers register callbacks for specific EventType values.
    Publishers emit events from any thread. The bus dispatches
    synchronously on the publishing thread — UI subscribers must
    use root.after() internally to marshal to the Tkinter main thread.
    """

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()
        self._log_entries: List[LogEntry] = []
        self._log_lock = threading.Lock()

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Register a callback for a specific event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Remove a callback registration."""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    cb for cb in self._subscribers[event_type] if cb != callback
                ]

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers of its type.
        Also logs the event to the structured log.
        Thread-safe.
        """
        # Structured logging to stdout
        self._log_to_console(event)

        # Store in log entries for the Logs page
        self._store_log_entry(event)

        # Dispatch to subscribers
        with self._lock:
            subscribers = list(self._subscribers.get(event.event_type, []))

        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event handler error for {event.event_type}: {e}")

    def _log_to_console(self, event: Event) -> None:
        """Structured one-line console log per the live progress requirement."""
        parts = [f"[{event.source or 'APP'}]"]

        if event.stage:
            parts.append(event.stage)

        if event.message:
            parts.append(event.message)

        if event.progress is not None:
            pct = int(event.progress * 100)
            parts.append(f"{pct}%")

        if event.elapsed_s is not None:
            parts.append(f"{event.elapsed_s:.1f}s")

        if event.error:
            parts.append(f"ERROR: {event.error}")

        log_line = " ".join(parts)

        if event.event_type == EventType.TASK_FAILED:
            logger.error(log_line)
        elif event.event_type == EventType.TASK_COMPLETED:
            logger.info(log_line)
        else:
            logger.debug(log_line)

        # Also print to stdout for terminal visibility
        print(log_line)

    def _store_log_entry(self, event: Event) -> None:
        """Store event as a LogEntry for the in-app Logs page."""
        level = "INFO"
        if event.event_type == EventType.TASK_FAILED:
            level = "ERROR"
        elif event.event_type == EventType.TASK_STAGE_CHANGED:
            level = "INFO"
        elif event.event_type == EventType.TASK_COMPLETED:
            level = "PASS"

        entry = LogEntry(
            timestamp=event.timestamp,
            module=event.source or "APP",
            level=level,
            message=event.message or event.stage or str(event.event_type.name),
            experiment_id=event.data.get("experiment_id", ""),
        )

        with self._log_lock:
            self._log_entries.append(entry)
            # Keep at most 10000 entries
            if len(self._log_entries) > 10000:
                self._log_entries = self._log_entries[-5000:]

    def get_log_entries(
        self,
        module_filter: Optional[str] = None,
        level_filter: Optional[str] = None,
        limit: int = 500,
    ) -> List[LogEntry]:
        """Retrieve log entries for the Logs page, with optional filtering."""
        with self._log_lock:
            entries = list(self._log_entries)

        if module_filter and module_filter != "ALL":
            entries = [e for e in entries if e.module == module_filter]

        if level_filter and level_filter != "ALL":
            entries = [e for e in entries if e.level == level_filter]

        return entries[-limit:]

    def clear_logs(self) -> None:
        with self._log_lock:
            self._log_entries.clear()


# Global singleton event bus
event_bus = EventBus()
