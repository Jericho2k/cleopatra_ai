"""Confirmation and hysteresis for the database health signal.

The dashboard kept showing "Cleopatra cannot reach its database. Messages are
not being processed." while every ordinary endpoint immediately before and after
worked and ``/health`` answered 200. One PostgREST connection recycled under a
probe was enough: the probe failed once, ``evaluate`` called that fatal, and the
banner then sat there until the next sixty-second poll.

A single failed probe is not an outage. A sustained inability to reach the
database is. Telling them apart needs state across probes, and that state has to
be deterministic — health that depends on wall-clock luck cannot be tested, and
health that cannot be tested is health nobody trusts.

The ladder
----------
=====================  ==================================================
observation            verdict
=====================  ==================================================
success                ``HEALTHY`` immediately, whatever came before
1 failure              ``UNCONFIRMED`` — recorded, no operator notice
``UNSTABLE_AFTER``+    ``UNSTABLE`` — degraded, "retrying", never fatal
``UNAVAILABLE_AFTER``+ ``UNAVAILABLE`` — fatal, but only once the failures
  failures AND         have also persisted for ``SUSTAINED_SECONDS``, so a
  sustained            burst of fast probes cannot manufacture an outage
=====================  ==================================================

Both thresholds must be crossed for the fatal verdict. Counting alone would let
a dashboard polling in a tight loop declare an outage in under a second;
duration alone would let two failures an hour apart do it.

Recovery is immediate and unconditional: the first success resets everything.
There is no "cooling off" period, because a database that just answered is a
database that works, and making an operator wait out a timer for a banner to
clear is exactly the complaint this module answers.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from enum import Enum


class DatabaseHealthState(str, Enum):
    HEALTHY = "healthy"
    UNCONFIRMED = "unconfirmed"
    UNSTABLE = "unstable"
    UNAVAILABLE = "unavailable"


# Consecutive failed probes before the operator is told anything at all.
DEFAULT_UNSTABLE_AFTER_FAILURES = 2
# Consecutive failed probes before the deployment is called unable to work.
DEFAULT_UNAVAILABLE_AFTER_FAILURES = 4
# ...and how long those failures must have been going on for.
DEFAULT_SUSTAINED_SECONDS = 30.0


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def unstable_after_failures() -> int:
    return _env_int("HEALTH_DB_UNSTABLE_AFTER_FAILURES", DEFAULT_UNSTABLE_AFTER_FAILURES)


def unavailable_after_failures() -> int:
    return max(
        unstable_after_failures(),
        _env_int(
            "HEALTH_DB_UNAVAILABLE_AFTER_FAILURES",
            DEFAULT_UNAVAILABLE_AFTER_FAILURES,
        ),
    )


def sustained_seconds() -> float:
    return _env_float("HEALTH_DB_SUSTAINED_SECONDS", DEFAULT_SUSTAINED_SECONDS)


@dataclass(frozen=True)
class DatabaseHealthVerdict:
    """What the current run of probe results means, and why."""

    state: DatabaseHealthState
    consecutive_failures: int
    failing_for_seconds: float
    error: str | None

    @property
    def fatal(self) -> bool:
        return self.state is DatabaseHealthState.UNAVAILABLE

    @property
    def degraded(self) -> bool:
        return self.state is DatabaseHealthState.UNSTABLE

    @property
    def reason(self) -> str | None:
        """The machine-readable reason string the health document publishes."""
        error = self.error or "unknown"
        if self.state is DatabaseHealthState.UNAVAILABLE:
            return f"database_unavailable:{error}"
        if self.state is DatabaseHealthState.UNSTABLE:
            return f"database_unstable:{error}"
        if self.state is DatabaseHealthState.UNCONFIRMED:
            # Published so an operator reading the raw document can see the
            # blip, and deliberately NOT mapped to a banner by the dashboard.
            return f"database_probe_failed_unconfirmed:{error}"
        return None

    def to_context(self) -> dict:
        return {
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "failing_for_seconds": round(self.failing_for_seconds, 3),
            "error": self.error,
            "fatal": self.fatal,
            "thresholds": {
                "unstable_after_failures": unstable_after_failures(),
                "unavailable_after_failures": unavailable_after_failures(),
                "sustained_seconds": sustained_seconds(),
            },
        }


HEALTHY_VERDICT = DatabaseHealthVerdict(
    state=DatabaseHealthState.HEALTHY,
    consecutive_failures=0,
    failing_for_seconds=0.0,
    error=None,
)


class DatabaseHealthTracker:
    """Consecutive-failure state for one process. Thread-safe and injectable.

    ``clock`` is a parameter rather than a module import so the ladder can be
    driven exactly in tests; production passes nothing and gets ``monotonic``.
    """

    def __init__(self, clock=None) -> None:
        import time

        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._failures = 0
        self._first_failure_at: float | None = None
        self._error: str | None = None

    def observe(self, *, reachable: bool, error: str | None = None) -> DatabaseHealthVerdict:
        """Record one probe result and return the current verdict."""
        with self._lock:
            if reachable:
                self._failures = 0
                self._first_failure_at = None
                self._error = None
                return HEALTHY_VERDICT

            now = self._clock()
            if self._first_failure_at is None:
                self._first_failure_at = now
            self._failures += 1
            self._error = error or self._error or "unknown"
            failing_for = max(0.0, now - self._first_failure_at)

            if (
                self._failures >= unavailable_after_failures()
                and failing_for >= sustained_seconds()
            ):
                state = DatabaseHealthState.UNAVAILABLE
            elif self._failures >= unstable_after_failures():
                state = DatabaseHealthState.UNSTABLE
            else:
                state = DatabaseHealthState.UNCONFIRMED

            return DatabaseHealthVerdict(
                state=state,
                consecutive_failures=self._failures,
                failing_for_seconds=failing_for,
                error=self._error,
            )

    def snapshot(self) -> DatabaseHealthVerdict:
        """The current verdict without recording a new observation."""
        with self._lock:
            if self._failures == 0 or self._first_failure_at is None:
                return HEALTHY_VERDICT
            failing_for = max(0.0, self._clock() - self._first_failure_at)
            if (
                self._failures >= unavailable_after_failures()
                and failing_for >= sustained_seconds()
            ):
                state = DatabaseHealthState.UNAVAILABLE
            elif self._failures >= unstable_after_failures():
                state = DatabaseHealthState.UNSTABLE
            else:
                state = DatabaseHealthState.UNCONFIRMED
            return DatabaseHealthVerdict(
                state=state,
                consecutive_failures=self._failures,
                failing_for_seconds=failing_for,
                error=self._error,
            )

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._first_failure_at = None
            self._error = None


# The process-wide tracker the health document uses.
DATABASE_HEALTH = DatabaseHealthTracker()
