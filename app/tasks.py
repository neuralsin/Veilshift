"""
QT-2.23 — Task Manager

Manages background execution of all science computations using
ThreadPoolExecutor. Emits live progress events at meaningful
granularity. Never calls CTk widgets directly — all UI updates
go through the EventBus.
"""

from __future__ import annotations
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

from app.events import EventBus, EventType, Event, event_bus
from app.state import TaskType, ModuleStatus


@dataclass
class TaskInfo:
    """Tracking info for a running or completed task."""
    task_id: str
    task_type: TaskType
    experiment_id: str
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    stage: str = ""
    progress: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    message: str = ""
    error: Optional[str] = None
    future: Optional[Future] = None

    @property
    def elapsed_s(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        elif self.start_time > 0:
            return time.time() - self.start_time
        return 0.0


class TaskManager:
    """
    Central task manager for all background science computation.

    Uses ThreadPoolExecutor for CPU-light / IO-adjacent work.
    For the heaviest loops (bootstrap, Monte Carlo CFAR), we still
    use threads but with numpy's internal parallelism — profiling
    showed that Python's ProcessPoolExecutor adds significant IPC
    overhead for array-heavy work that numpy already parallelizes
    via BLAS/LAPACK, making threads the better choice here.
    """

    def __init__(self, event_bus: EventBus, max_workers: int = 4):
        self._event_bus = event_bus
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = threading.Lock()
        self._task_counter = 0

    def submit(
        self,
        task_type: TaskType,
        experiment_id: str,
        fn: Callable,
        *args,
        **kwargs,
    ) -> str:
        """
        Submit a background task. Returns task_id.
        The callable `fn` receives a ProgressCallback as its first argument.
        """
        self._task_counter += 1
        task_id = f"TASK-{self._task_counter:04d}"

        task_info = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            experiment_id=experiment_id,
        )

        with self._lock:
            self._tasks[task_id] = task_info

        def progress_callback(
            stage: str = "",
            progress: float = 0.0,
            message: str = "",
        ):
            """Callback passed to worker functions for live progress reporting."""
            task_info.stage = stage
            task_info.progress = progress
            task_info.message = message

            self._event_bus.publish(Event(
                event_type=EventType.TASK_PROGRESS if not stage
                else EventType.TASK_STAGE_CHANGED,
                source=task_type.name,
                task_type=task_type.name,
                stage=stage,
                progress=progress,
                message=message,
                elapsed_s=task_info.elapsed_s,
                data={"task_id": task_id, "experiment_id": experiment_id},
            ))

        def wrapper():
            task_info.status = "RUNNING"
            task_info.start_time = time.time()

            self._event_bus.publish(Event(
                event_type=EventType.TASK_STARTED,
                source=task_type.name,
                task_type=task_type.name,
                message=f"Started {task_type.name}",
                data={"task_id": task_id, "experiment_id": experiment_id},
            ))

            try:
                result = fn(progress_callback, *args, **kwargs)

                task_info.status = "COMPLETED"
                task_info.end_time = time.time()
                task_info.progress = 1.0

                self._event_bus.publish(Event(
                    event_type=EventType.TASK_COMPLETED,
                    source=task_type.name,
                    task_type=task_type.name,
                    message=f"Completed {task_type.name}",
                    elapsed_s=task_info.elapsed_s,
                    data={
                        "task_id": task_id,
                        "experiment_id": experiment_id,
                        "result": result,
                    },
                ))

                return result

            except Exception as e:
                task_info.status = "FAILED"
                task_info.end_time = time.time()
                task_info.error = str(e)

                self._event_bus.publish(Event(
                    event_type=EventType.TASK_FAILED,
                    source=task_type.name,
                    task_type=task_type.name,
                    message=f"Failed {task_type.name}",
                    error=str(e),
                    elapsed_s=task_info.elapsed_s,
                    data={"task_id": task_id, "experiment_id": experiment_id},
                ))

                raise

        future = self._executor.submit(wrapper)
        task_info.future = future

        return task_id

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        with self._lock:
            return self._tasks.get(task_id)

    def get_running_tasks(self) -> List[TaskInfo]:
        with self._lock:
            return [t for t in self._tasks.values() if t.status == "RUNNING"]

    def get_all_tasks(self) -> List[TaskInfo]:
        with self._lock:
            return list(self._tasks.values())

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.future:
                cancelled = task.future.cancel()
                if cancelled:
                    task.status = "CANCELLED"
                return cancelled
        return False

    def shutdown(self):
        self._executor.shutdown(wait=False)
