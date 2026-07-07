"""Fire-and-forget task helper that doesn't swallow exceptions.

`asyncio.create_task(coro)` on its own drops any exception the coroutine raises:
the task is never awaited, so the error surfaces nowhere and the failure is
silent. In a live pilot that means a background write (memory update, purchase
verification, enrichment) can fail with zero trace.

`spawn()` wraps create_task and attaches a done-callback that logs any exception
(other than normal cancellation). Same fire-and-forget ergonomics, but failures
become visible in the logs.

Usage:  spawn(_update_fan_memory(...), name="update_fan_memory")
"""
import asyncio


def _log_task_result(task: "asyncio.Task") -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        name = task.get_name()
        print(f"[TASK ERROR] background task '{name}' failed: {exc!r}")


def spawn(coro, name: str | None = None) -> "asyncio.Task":
    """Schedule a coroutine as a background task, logging any exception it raises."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(_log_task_result)
    return task