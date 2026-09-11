"""Owner-only local Full Auto simulation: who may use it, and on which fans.

The simulator runs the *real* Full Auto pipeline with delivery transport and
human-like waiting replaced by local persistence. That makes it a privileged
capability, not a product feature: it writes creator messages that the ordinary
dashboard renders, and it exercises commercial state transitions. It must
therefore be invisible and inaccessible to every ordinary agency tenant.

Access requires ALL of:

1. ``AUTO_SIMULATION_ENABLED=true``;
2. an authenticated Supabase dashboard user (``request.state.dashboard_user_id``);
3. that user's UUID listed in ``AUTO_SIMULATION_ALLOWED_USER_IDS``;
4. the normal creator tenancy check (``core.tenancy``);
5. the fan belonging to that creator;
6. ``fans.platform_fan_id`` starting with ``test_``.

Requirements 1-3 live here. 4-6 are enforced by the route, reusing the existing
tenancy helpers rather than inventing a parallel authorization model.

Two deliberate choices:

*No development bypass.* Unlike ``core.auth``, an unset allowlist is not relaxed
under ``APP_ENV=development``. The relaxed branch in ``core.tenancy`` exists so
local work is not blocked by an unconfigured tenancy table; there is no
equivalent need here, and a dev bypass would be one misread variable away from
handing the simulator to a production tenant.

*Failures are indistinguishable.* Every rejection is the same 404 the tenancy
layer raises, so a caller cannot probe which of the six conditions it failed,
and cannot learn that another tenant's creator or fan exists.
"""
from __future__ import annotations

import os

from fastapi import HTTPException, Request, status

from core.auth import dashboard_user_id


_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}

# The prefix that marks a fan as safe to simulate against. A real Fansly fan
# must never become eligible merely because somebody knows its UUID.
TEST_FAN_PREFIX = "test_"


def not_found() -> HTTPException:
    """The single rejection used by every simulation check.

    Identical to ``core.tenancy._forbidden`` on purpose: a non-allowlisted
    caller must not be able to tell "you are not allowed" from "that does not
    exist", in either direction.
    """
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Resource not found",
    )


def simulation_enabled() -> bool:
    """Whether the deployment has switched the simulator on at all."""
    return str(os.environ.get("AUTO_SIMULATION_ENABLED", "")).strip().lower() in _TRUE_VALUES


def allowed_simulation_user_ids() -> frozenset[str]:
    """Supabase auth UUIDs permitted to simulate. Empty means nobody.

    Parsed rather than compared as a raw string so whitespace, a trailing comma
    or a pasted newline cannot silently deny a correctly configured owner —
    and so a blank entry can never match a blank user id.
    """
    raw = str(os.environ.get("AUTO_SIMULATION_ALLOWED_USER_IDS", ""))
    return frozenset(
        entry.strip().lower()
        for entry in raw.replace("\n", ",").split(",")
        if entry.strip()
    )


def user_may_simulate(user_id: str | None) -> bool:
    """Requirements 1-3, with no side effects, so the capability endpoint and
    the mutation endpoint can never disagree about who is allowed."""
    if not simulation_enabled():
        return False
    if not user_id:
        return False
    return str(user_id).strip().lower() in allowed_simulation_user_ids()


def request_may_simulate(request: Request) -> bool:
    """Requirements 1-3 for the caller of this request."""
    return user_may_simulate(dashboard_user_id(request))


async def require_simulation_user(request: Request) -> str:
    """FastAPI dependency for requirements 1-3. Returns the allowed user id.

    Deliberately does NOT reveal which condition failed, and never logs or
    returns the allowlist.
    """
    user_id = dashboard_user_id(request)
    if not user_may_simulate(user_id):
        raise not_found()
    return str(user_id)


def is_simulatable_fan(platform_fan_id: object) -> bool:
    """Requirement 6 — the server-side test-fan boundary."""
    return str(platform_fan_id or "").startswith(TEST_FAN_PREFIX)
