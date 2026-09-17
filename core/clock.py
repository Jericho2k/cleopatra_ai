"""The time this process believes it is, and the one way to move it.

WHY THIS EXISTS
---------------
The continuation brief asks the evaluation harness for "a clock injected
through expiry, scheduling and continuity". Without one,
``Disturbance.days_since_previous`` was a number the harness added up and
nothing read: ``services/trajectory_eval.py`` called ``advance_clock`` only
when a caller supplied one, and no caller did. So the trajectory labelled
"he comes back a day later, then a week later" ran its four turns back to
back, and every expiry window, every scheduled action and every continuity
thread saw one uninterrupted session.

An evaluation that cannot move time cannot test a return after a week, cannot
make a queued follow-up become due, and cannot expire anything. Those are three
of the rows §5 asks for.

WHAT IT IS NOT
--------------
Not a general time abstraction, and not something to migrate the whole codebase
onto. There are over a hundred ``datetime.now`` calls here and most of them
want the wall clock: a log line, a health check timestamp, a
``recorded_at``. Rewriting those would be a large change with no test behind
it. Three surfaces are wired to this deliberately, because they are the three
the brief names and the three an evaluation has to be able to move:

  * **continuity** — ``services/conversation_continuity`` thread expiry;
  * **scheduling** — claiming a due ``scheduled_actions`` row;
  * anything reading either of those.

Everything else still reads the wall clock, and that is the correct default.

THE SAFETY PROPERTY
-------------------
An offset clock in production would be a serious incident. Scheduled actions
would fire early or never, threads would expire in the wrong order, and
follow-ups would go out at the wrong time to real customers.

So ``advance`` refuses unless BOTH hold:

  * ``APP_ENV`` is not ``production``; and
  * ``EVAL_CLOCK_ENABLED`` is explicitly on.

Two conditions rather than one, and neither is the default. A deployment that
somehow sets the flag still cannot move time, and a test environment that
forgets the flag gets a refusal rather than a silently real clock.

Refusing LOUDLY matters as much as refusing. A harness that silently declined
to advance is exactly the failure this replaces, so ``advance`` raises and the
caller reports the run as uncovered rather than as a week having passed.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone

#: Set to ``1``/``true``/``yes`` to allow the clock to be moved at all.
EVAL_CLOCK_FLAG = "EVAL_CLOCK_ENABLED"

_LOCK = threading.Lock()
_OFFSET = timedelta(0)


class ClockNotMovable(RuntimeError):
    """Something tried to move time where that is not allowed."""


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def movable() -> bool:
    """Whether this process is allowed to simulate the passage of time."""
    if str(os.environ.get("APP_ENV") or "").strip().lower() == "production":
        return False
    return _truthy(os.environ.get(EVAL_CLOCK_FLAG))


def now() -> datetime:
    """Now, plus whatever an evaluation has advanced.

    Identical to ``datetime.now(timezone.utc)`` in every deployment that has
    not explicitly enabled the eval clock, which is all of them by default.
    """
    if not _OFFSET:
        return datetime.now(timezone.utc)
    return datetime.now(timezone.utc) + _OFFSET


def offset() -> timedelta:
    """How far ahead of the wall clock this process currently is."""
    return _OFFSET


def offset_seconds() -> float:
    return _OFFSET.total_seconds()


def advance(days: float) -> None:
    """Move time forward by ``days`` simulated days.

    Raises ``ClockNotMovable`` rather than doing nothing, because an
    evaluation that quietly failed to advance would report a week as having
    passed when it did not — which is the bug this module exists to remove,
    reproduced one level up.
    """
    global _OFFSET
    if days <= 0:
        return
    if not movable():
        raise ClockNotMovable(
            "the evaluation clock is not enabled in this process. It needs "
            f"APP_ENV != production and {EVAL_CLOCK_FLAG}=1, and it is "
            "deliberately awkward to turn on: an offset clock against real "
            "customers would fire scheduled work at the wrong time."
        )
    with _LOCK:
        _OFFSET += timedelta(days=float(days))


def reset() -> None:
    """Return to the wall clock.

    Always allowed, and always safe: moving BACK to real time cannot make a
    deployment do anything early. Scenario isolation depends on this — the
    brief asks for fresh state between independent scenarios, and a clock
    carried over from the last one is state.
    """
    global _OFFSET
    with _LOCK:
        _OFFSET = timedelta(0)


def simulated_now_for_sql() -> str | None:
    """The simulated ``now`` to hand to SQL, or None to let the database decide.

    None whenever the clock has not been moved, so a normal deployment sends
    exactly what it sent before and the database uses its own ``now()``. The
    parameter only appears when an evaluation is actually simulating time.
    """
    if not _OFFSET:
        return None
    return now().isoformat()
