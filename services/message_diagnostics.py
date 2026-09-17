"""Owner-only diagnostics, kept out of the row a browser can read.

THE BOUNDARY THIS ENFORCES
--------------------------
``services.ai_stack_visibility`` states the rule: an agency operator is told
which AI stack answered and never what it routes to. It enforces that rule on
the HTTP responses ``main.py`` serves, which was correct and was not the whole
surface. The dashboard also reads the table directly::

    app/simulator/page.tsx:205   supabase.from('messages').select('*')

That is the operator's own JWT against Supabase, and
``db/browser_least_privilege_v1.sql`` grants ``authenticated`` SELECT on every
column of ``messages``. So ``media_context.reply_provenance.writer.actual`` —
the provider and model that actually served the reply, plus the whole fallback
ladder in ``.attempts`` — was already in an agency operator's browser, along
with ``media_context.ai_stack.provider`` and ``.model``. Redacting one more
response would not have changed that, and neither would hiding a panel.

So the fix is where the bytes live. Anything owner-only is split off before the
row is written and stored in ``public.message_diagnostics``, which
``db/owner_only_diagnostics_v1.sql`` registers as owner-only: no policy and no
grant for ``authenticated``, and both discovery migrations skip it.

WHY THE SPLIT IS AN ALLOWLIST
-----------------------------
The same reasoning as ``ai_stack_visibility``: a denylist protects the keys
somebody remembered to name. ``reply_provenance`` is the proof — it was added
next to ``ai_stack``, under the rule that already governed ``ai_stack``, and
was not covered by it.

Unknown keys are therefore treated as owner-only. They are MOVED rather than
dropped: a key this module has not been told about is diverted into the
diagnostics record, logged, and readable through the owner trace endpoint. A
new product key that genuinely belongs in the browser's copy is one entry in
``PUBLIC_MEDIA_CONTEXT_KEYS`` away, and ``tests/test_owner_only_diagnostics.py``
fails until someone makes that call deliberately. Nothing is ever destroyed by
being unrecognised.

WHERE IT RUNS
-------------
``db.queries.save_message_result`` — the single chokepoint every message write
goes through, including ``services.ppv_persistence.save_ppv_message_receipt``,
which routes through it. Splitting per call site is how ``reply_provenance``
escaped the first time.

FAILING SOFT
------------
The same rule ``reply_provenance`` follows: a recorder that failed must never
stop a reply. The split itself is total and pure, so the sensitive keys leave
the row whether or not the diagnostics write succeeds. Losing the diagnostics
row costs a diagnostic; failing the send costs a customer conversation. Only
one of those is recoverable, and it is not the conversation.
"""

from __future__ import annotations

import asyncio
from typing import Any

from core.supabase import get_supabase
from services.ai_stack_visibility import (
    PUBLIC_MARKER_KEYS,
    PUBLIC_MEDIA_CONTEXT_KEYS,
)

#: The table the split half lands in. Registered owner-only in
#: db/owner_only_diagnostics_v1.sql.
DIAGNOSTICS_TABLE = "message_diagnostics"

#: Which top-level keys the browser's copy of the row may keep. One definition,
#: in services/ai_stack_visibility.py, shared by the storage boundary here and
#: the response boundary there — two copies of an allowlist is two allowlists.
__all__ = [
    "DIAGNOSTICS_TABLE",
    "PUBLIC_MEDIA_CONTEXT_KEYS",
    "diagnostics_row",
    "read_diagnostics",
    "record_diagnostics",
    "split_media_context",
    "unrecognised_keys",
]


def split_media_context(
    media_context: Any,
) -> tuple[Any, dict[str, Any]]:
    """Separate one message's metadata into (browser-readable, owner-only).

    Total and pure. Anything it does not understand — a non-dict, a missing
    value — is returned unchanged with nothing to record, because this runs in
    the send path and must not be able to raise.
    """
    if not isinstance(media_context, dict):
        return media_context, {}

    public: dict[str, Any] = {}
    owner_only: dict[str, Any] = {}

    for key, value in media_context.items():
        if key not in PUBLIC_MEDIA_CONTEXT_KEYS:
            owner_only[key] = value
            continue
        if key != "ai_stack":
            public[key] = value
            continue
        # The stack marker is public in name only: `profile` identifies the
        # product-level choice, and route/prompt_version/provider/model are the
        # supply chain behind it.
        if not isinstance(value, dict):
            public[key] = value
            continue
        visible = {k: v for k, v in value.items() if k in PUBLIC_MARKER_KEYS}
        routing = {k: v for k, v in value.items() if k not in PUBLIC_MARKER_KEYS}
        if visible:
            public[key] = visible
        if routing:
            owner_only.setdefault("ai_stack", {}).update(routing)

    return public, owner_only


def unrecognised_keys(media_context: Any) -> list[str]:
    """Top-level keys the allowlist has not been told about.

    Diverting these is correct — an unreviewed key must not reach a browser —
    but it is also how a product field silently stops rendering, so the caller
    logs it rather than letting it pass unremarked.
    """
    if not isinstance(media_context, dict):
        return []
    known = PUBLIC_MEDIA_CONTEXT_KEYS | {"reply_provenance"}
    return sorted(key for key in media_context if key not in known)


def diagnostics_row(
    *,
    message_id: str,
    creator_id: str,
    fan_id: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """One ``message_diagnostics`` row, with the turn denormalised out.

    ``turn_id`` and ``part`` are lifted out of the provenance record so the
    bubbles of a single turn can be found without scanning jsonb. They are read
    defensively: a record assembled by an older build, or one carrying only an
    ``ai_stack`` marker, has neither.
    """
    provenance = record.get("reply_provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    turn_id = provenance.get("turn_id")
    part = provenance.get("part")
    return {
        "message_id": str(message_id),
        "creator_id": str(creator_id),
        "fan_id": str(fan_id),
        "turn_id": str(turn_id) if turn_id else None,
        "part": int(part) if isinstance(part, int) else 0,
        "record": record,
    }


async def record_diagnostics(
    *,
    message_id: str | None,
    creator_id: str,
    fan_id: str,
    record: dict[str, Any],
) -> bool:
    """Persist the owner-only half. Returns whether it landed.

    Never raises. A message with no id (an upsert whose response was lost) has
    nothing to attach to, and is reported as not recorded rather than guessed
    at: attaching a trace to the wrong message is worse than having none.
    """
    if not message_id or not record:
        return False
    row = diagnostics_row(
        message_id=message_id,
        creator_id=creator_id,
        fan_id=fan_id,
        record=record,
    )

    def _write() -> None:
        get_supabase().table(DIAGNOSTICS_TABLE).upsert(
            row, on_conflict="message_id"
        ).execute()

    try:
        await asyncio.to_thread(_write)
        return True
    except Exception as exc:  # pragma: no cover - a recorder never blocks a send
        print(f"[DIAGNOSTICS] could not record message={message_id}: {type(exc).__name__}")
        return False


async def read_diagnostics(message_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Owner-only records for these messages, keyed by message id.

    Reads through the service role, so it is the CALLER's job to have
    established that the requester is the platform owner. ``main.py`` does that
    with ``core.simulation.request_is_platform_operator`` — the identity
    allowlist, the same check that gates ``operator_diagnostics``.
    """
    if not message_ids:
        return {}
    wanted = [str(mid) for mid in message_ids if mid]

    def _read() -> list[dict[str, Any]]:
        result = (
            get_supabase().table(DIAGNOSTICS_TABLE)
            .select("message_id, creator_id, fan_id, turn_id, part, record, recorded_at")
            .in_("message_id", wanted)
            .execute()
        )
        return list(result.data or [])

    try:
        rows = await asyncio.to_thread(_read)
    except Exception as exc:
        print(f"[DIAGNOSTICS] read failed: {type(exc).__name__}")
        return {}
    return {str(row["message_id"]): row for row in rows if row.get("message_id")}
