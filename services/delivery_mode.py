"""Whether deliberate human-like waiting is served durably or executed now.

Production always uses ``DURABLE``: a composition pause and every inter-bubble
pause become rows in the existing scheduled-action queue, so nine seconds of
"typing" costs a row rather than a worker slot.

``IMMEDIATE`` exists for the owner-only simulator and the offline evaluators,
where the wait is the one thing that is not interesting. It removes DURATION
only. The sequence is still planned, still persisted, still bound to the
conversation generation, and still revalidated at every send boundary — a
simulated turn that skipped those checks would be measuring a system nobody
runs.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar

DURABLE = "durable"
IMMEDIATE = "immediate"

_MODE: ContextVar[str] = ContextVar("cleopatra_delivery_mode", default=DURABLE)


def current_mode() -> str:
    return _MODE.get()


def is_immediate() -> bool:
    return _MODE.get() == IMMEDIATE


@contextlib.contextmanager
def immediate_delivery_scope():
    """Execute planned timing at once, keeping every correctness check."""
    token = _MODE.set(IMMEDIATE)
    try:
        yield
    finally:
        _MODE.reset(token)


@contextlib.contextmanager
def durable_delivery_scope():
    token = _MODE.set(DURABLE)
    try:
        yield
    finally:
        _MODE.reset(token)
