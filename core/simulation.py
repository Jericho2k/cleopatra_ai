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

import json
import os
from collections import OrderedDict

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


# ---------------------------------------------------------------------------
# The simulator's event marker
# ---------------------------------------------------------------------------
#
# The simulator persists its fan message with an ordinary INSERT, which is
# exactly what the production Supabase database webhook on ``messages`` fires
# on. That webhook POSTs /generate-suggestions, so one simulated fan turn used
# to run the whole ordinary inbound pipeline — situation analysis, commercial
# state, price learning, the conversation director — a second time, alongside
# the Full Auto turn the simulator itself drives. The simulator's results were
# therefore measuring two overlapping passes, not one.
#
# The row is marked instead of being recognised by shape. Nothing here keys off
# the ``test_`` platform-fan prefix: that prefix says a fan is *simulatable*,
# not that a particular message came from the simulator, and an operator typing
# into a test fan's real chat must still be processed normally. An explicit
# marker on the event is the only thing that means "the simulator already owns
# this one".
#
# ``media_context`` is existing JSON metadata on ``messages``, so no migration
# is involved.

SIMULATION_SOURCE = "owner_auto_simulator"


def simulation_message_marker() -> dict:
    """The ``media_context`` the simulator stamps on the fan message it writes."""
    return {"simulation": True, "simulation_source": SIMULATION_SOURCE}


def is_simulation_message(media_context: object) -> bool:
    """Whether one message row was written by the owner-only simulator.

    Accepts whatever the Supabase webhook happens to deliver for a ``jsonb``
    column — a decoded mapping, the raw JSON text, or JSON text that was itself
    double-encoded by a relay — because a transport detail must not decide
    whether a simulated turn is processed twice.

    Fails closed in the direction that matters: anything it cannot positively
    identify as a simulator event is an ordinary production message and is
    processed normally.
    """
    # Two unwraps, not one. A payload that reaches us as "\"{...}\"" — a jsonb
    # value serialised by one hop and then string-encoded by the next — decodes
    # to a string on the first pass, and returning False there is precisely the
    # silent failure that let a simulated turn be processed twice.
    for _ in range(2):
        if not isinstance(media_context, str):
            break
        try:
            media_context = json.loads(media_context)
        except (ValueError, TypeError):
            return False
    if not isinstance(media_context, dict):
        return False
    if media_context.get("simulation") is not True:
        return False
    return str(media_context.get("simulation_source") or "") == SIMULATION_SOURCE


# ---------------------------------------------------------------------------
# Ownership when the payload cannot answer
# ---------------------------------------------------------------------------
#
# The marker above is written into the same INSERT the database webhook fires
# on, so in principle the webhook record always carries it. In practice the
# record is a JSON document built by somebody else's trigger and shipped over
# somebody else's transport, and Railway kept showing one simulated INSERT
# followed by a full ordinary pipeline pass. Rather than keep guessing which
# hop drops or reshapes ``media_context``, ownership is settled by two things
# that do not depend on the payload's shape at all:
#
# 1. An in-process registry of the message ids the simulator wrote. Same
#    process, zero cost, and immune to every transport question.
# 2. The database row itself, re-read by id. Authoritative by construction,
#    used only when the payload did not positively answer, so the ordinary
#    production path costs no extra read.
#
# Both are additive. Neither can make an ordinary fan message be skipped: they
# can only recognise a row the simulator actually wrote.

_SIMULATION_OWNED_MESSAGE_IDS: "OrderedDict[str, None]" = OrderedDict()

# Bounded so a long-lived process cannot accumulate ids forever. Generous
# relative to how many turns one owner runs in a sitting, and the database
# fallback still covers anything evicted.
_OWNED_ID_LIMIT = 512


def mark_simulation_owned_message(message_id: object) -> None:
    """Record that the simulator wrote this row and owns its processing."""
    key = str(message_id or "").strip()
    if not key:
        return
    _SIMULATION_OWNED_MESSAGE_IDS.pop(key, None)
    _SIMULATION_OWNED_MESSAGE_IDS[key] = None
    while len(_SIMULATION_OWNED_MESSAGE_IDS) > _OWNED_ID_LIMIT:
        _SIMULATION_OWNED_MESSAGE_IDS.popitem(last=False)


def is_simulation_owned_message_id(message_id: object) -> bool:
    """Whether this process wrote that row as a simulator event."""
    return str(message_id or "").strip() in _SIMULATION_OWNED_MESSAGE_IDS


def reset_simulation_owned_message_ids() -> None:
    """Test-support only."""
    _SIMULATION_OWNED_MESSAGE_IDS.clear()


def payload_media_context(record: object) -> tuple[object, bool]:
    """Extract ``media_context`` from a webhook record.

    Returns ``(value, present)``. ``present`` is False when the key is absent
    entirely, which is the case that must NOT be read as "not a simulation" —
    it is the case where the payload simply did not say, and the database has
    to be asked instead.
    """
    if not isinstance(record, dict):
        return None, False
    for key in ("media_context", "mediaContext"):
        if key in record:
            return record[key], True
    return None, False


async def message_row_is_simulation_owned(message_id: object) -> bool:
    """Ask the database whether that row carries the simulator's marker.

    The authoritative answer, used when the webhook payload did not carry one.
    Never raises: a read failure means "not positively identified", which routes
    the message through the ordinary pipeline exactly as before.
    """
    key = str(message_id or "").strip()
    if not key:
        return False
    try:
        import asyncio

        from core.supabase import get_supabase

        def _read() -> object:
            response = (
                get_supabase().table("messages")
                .select("media_context")
                .eq("id", key)
                .limit(1)
                .execute()
            )
            rows = response.data or []
            return rows[0].get("media_context") if rows else None

        return is_simulation_message(await asyncio.to_thread(_read))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WEBHOOK] simulation ownership read failed id={key}: {exc}")
        return False
