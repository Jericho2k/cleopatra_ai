"""Whether this deployment may talk to the managed API Fansly provider at all.

Two independent reasons exist for refusing a remote call, and they are kept
apart because they mean different things to an operator:

``APIFANSLY_ENABLED=false``
    A deliberate, deployment-wide decision: we are between pilots and are not
    paying for provider credits. Everything that needs the remote platform is
    intentionally offline. This is a *configuration* state, not a fault, so it
    must never be reported as an outage, and it must never produce a repeating
    stack trace from a background loop.

an active simulation scope
    One request is running the real Full Auto pipeline against a ``test_`` fan
    with delivery transport simulated. No remote call may escape that turn —
    before, during, or after generation — no matter which code path the pipeline
    happens to take. The scope is a :mod:`contextvars` value rather than a
    parameter threaded through twenty call sites, so it propagates into tasks
    created by the turn (``asyncio.create_task`` and ``asyncio.to_thread`` both
    copy the context) and cannot be lost by a caller that forgot to pass a flag.

The check itself is installed in the transport (``services.apifansly.headers``,
``request`` and ``download_media``), so it is a property of the deployment
rather than a list of call sites somebody remembered to patch. Callers still
suppress their own work cleanly on top of it — the transport guard is the
backstop that makes "zero remote calls" a fact instead of an audit.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


# Absent means enabled: every deployment that predates this switch keeps its
# current live behaviour after upgrading without setting anything.
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}

# Reason codes. These are contract with the dashboard and with tests; they are
# deliberately stable strings rather than prose.
REASON_DISABLED = "apifansly_disabled"
REASON_SIMULATION = "apifansly_simulation"

_SIMULATION: ContextVar[bool] = ContextVar("apifansly_simulation", default=False)


class ApiFanslyDisabledError(RuntimeError):
    """A remote API Fansly call was refused before it reached the network.

    ``reason`` distinguishes the deployment-wide switch from a simulated turn so
    a caller can log the right thing, and so an endpoint can return a stable
    machine-readable code to the dashboard.

    Subclasses ``RuntimeError`` on purpose: the existing callers that already
    catch ``(httpx.HTTPError, RuntimeError)`` around a provider call degrade
    gracefully instead of crashing a background loop, even where this module has
    not taught them about the switch explicitly.
    """

    def __init__(self, message: str, *, reason: str = REASON_DISABLED) -> None:
        super().__init__(message)
        self.reason = reason


def apifansly_enabled() -> bool:
    """Whether the remote API Fansly connector is turned on for this deployment.

    The single source of truth. Nothing else in the codebase may read
    ``APIFANSLY_ENABLED`` directly.
    """
    raw = str(os.environ.get("APIFANSLY_ENABLED", "")).strip().lower()
    if not raw:
        return True
    return raw not in _FALSE_VALUES


def simulation_active() -> bool:
    """Whether the current task is inside an owner-only Full Auto simulation."""
    return bool(_SIMULATION.get())


@contextmanager
def simulation_scope() -> Iterator[None]:
    """Refuse every remote API Fansly call for the duration of this block.

    Applies to the current task and to anything it spawns, because both
    ``asyncio.create_task`` and ``asyncio.to_thread`` copy the active context.
    """
    token = _SIMULATION.set(True)
    try:
        yield
    finally:
        _SIMULATION.reset(token)


def refusal_reason() -> str | None:
    """The reason a remote call would be refused right now, or None."""
    if simulation_active():
        return REASON_SIMULATION
    if not apifansly_enabled():
        return REASON_DISABLED
    return None


def require_apifansly_available(operation: str = "API Fansly request") -> None:
    """Raise unless a remote API Fansly call is permitted right now."""
    reason = refusal_reason()
    if reason is None:
        return
    if reason == REASON_SIMULATION:
        raise ApiFanslyDisabledError(
            f"{operation} blocked: local Full Auto simulation must not reach "
            "the remote platform",
            reason=REASON_SIMULATION,
        )
    raise ApiFanslyDisabledError(
        f"{operation} blocked: the API Fansly connector is disabled by "
        "configuration (APIFANSLY_ENABLED=false)",
        reason=REASON_DISABLED,
    )


def describe_apifansly() -> str:
    """One line for the startup log, so the resolved state is never a guess."""
    if apifansly_enabled():
        return "[APIFANSLY] connector enabled"
    return "[APIFANSLY] connector disabled by configuration"
