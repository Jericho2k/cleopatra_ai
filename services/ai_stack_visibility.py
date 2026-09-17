"""What each tier is told about the AI stack, and what it is not.

THE RULE
--------
An agency operator chooses an AI Stack Profile in the Simulator. To do that it
needs two things: a stable identifier to send back, and a name to show in a
dropdown. That is the entire product-level fact::

    {"id": "cleo_v3", "name": "Cleo V3"}

It does NOT need, and must not receive:

* provider names (openrouter, together, anthropic);
* model identifiers (Kimi, Qwen, GLM, Llama, Claude, anything);
* fallback chains — which model catches a failure of which other one;
* internal stage routing — that a safety turn is written by a different target
  than an ordinary one, or that those stages exist;
* prompt versions and generation configuration (reasoning, temperature,
  max tokens, output contract).

Those are the platform's supply chain, its cost structure and its unshipped
comparisons. They are diagnostics for the platform owner, who is the person who
changes them, and they are not part of what the product sells.

WHY IT IS ENFORCED HERE AND NOT IN THE DASHBOARD
------------------------------------------------
A dashboard that simply does not render a field has still received it: the
value is in the HTTP response, in the browser's memory and in anyone's network
tab. So the redaction is applied to the RESPONSE, by the routes in main.py, on
every surface that carries stack detail — the profile registry, and the AI
stack marker persisted on simulated creator messages.

WHO IS THE OWNER
----------------
``core.simulation.request_is_platform_operator`` — the identity allowlist
behind ``AUTO_SIMULATION_ALLOWED_USER_IDS``, the same one that already gates
``operator_diagnostics``. Deliberately the IDENTITY check rather than
``request_is_simulation_owner``: switching the simulator off must not change
who the platform owner is, and stack routing is diagnostics rather than a
simulator feature.

FAILING CLOSED
--------------
Every helper here rewrites a document down to a known allowlist of keys rather
than deleting the keys it currently knows to be sensitive. A stage field added
to the registry later, or a new key added to the message marker, is therefore
invisible to an agency by default instead of visible until somebody remembers.
"""

from __future__ import annotations

from typing import Any

from ai.stack_profiles import describe_profiles, profile_directory


# The only key of the persisted ``media_context.ai_stack`` marker that names a
# product-level fact. ``route``, ``prompt_version``, ``provider`` and ``model``
# are all internal routing; see services.suggestions.message_ai_stack_metadata.
PUBLIC_MARKER_KEYS: frozenset[str] = frozenset({"profile"})


# The top-level ``media_context`` keys an agency may see. Each is the agency's
# own operational record of its own fan — what was sold, what was attached,
# that a turn was simulated, that access was repaired. None names a provider,
# a model or a route.
#
# An allowlist rather than a denylist, for the reason this whole module exists,
# and ``reply_provenance`` is the proof it was needed: it was added next to
# ``ai_stack``, under the rule that already governed ``ai_stack``, and was not
# covered by the denial that named ``ai_stack`` alone.
#
# services/message_diagnostics.py applies the same set at the STORAGE boundary,
# which is the one that actually holds — the dashboard reads
# ``messages.media_context`` straight out of Supabase, so a response-level
# redaction alone never saw those reads. This copy stays because a deployment
# whose migration has not run yet still has the old rows.
PUBLIC_MEDIA_CONTEXT_KEYS: frozenset[str] = frozenset(
    {
        "ppv",
        "attachments",
        "ai_stack",
        "simulation",
        "simulation_source",
        "scheduled",
        "content_access_repair",
        "delivery_status",
    }
)


def registry_view(*, diagnostics: bool) -> list[dict[str, Any]]:
    """The profile registry as this caller may see it.

    Same profiles, same order, same ids either way: an agency picks from the
    identical registry and is simply not told what each entry routes to.
    """
    return describe_profiles() if diagnostics else profile_directory()


def public_ai_stack_marker(marker: Any) -> dict[str, Any] | None:
    """One message's stack marker reduced to the profile that answered.

    Returns ``None`` when there is nothing left to report, so a redacted marker
    is absent rather than an empty object a client might render as "unknown
    stack".
    """
    if not isinstance(marker, dict):
        return None
    reduced = {
        key: value for key, value in marker.items() if key in PUBLIC_MARKER_KEYS
    }
    return reduced or None


def public_media_context(media_context: Any) -> Any:
    """A message's ``media_context`` reduced to what an agency may see.

    Rewrites down to ``PUBLIC_MEDIA_CONTEXT_KEYS`` rather than deleting the
    keys known to be sensitive. It previously did the opposite — it redacted
    ``ai_stack`` and returned every sibling untouched — and
    ``media_context.reply_provenance`` was added as one of those siblings,
    carrying ``writer.actual.provider``, ``writer.actual.model`` and the whole
    ``writer.attempts`` fallback ladder straight through it.

    A dropped key is a dropped key, not an emptied one: a client must not be
    able to tell a redacted document from one that never had the field.
    """
    if not isinstance(media_context, dict):
        return media_context
    public: dict[str, Any] = {}
    for key, value in media_context.items():
        if key not in PUBLIC_MEDIA_CONTEXT_KEYS:
            continue
        if key != "ai_stack":
            public[key] = value
            continue
        marker = public_ai_stack_marker(value)
        if marker is not None:
            public["ai_stack"] = marker
    return public


def public_message_rows(rows: Any) -> list[dict[str, Any]]:
    """Simulated creator messages with every stack marker redacted."""
    if not isinstance(rows, list):
        return []
    public: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "media_context" not in row:
            public.append(row)
            continue
        public.append(
            {**row, "media_context": public_media_context(row.get("media_context"))}
        )
    return public


def public_model_health(health: Any) -> dict[str, Any]:
    """Provider availability with the supply chain removed.

    The dashboard's health banner needs to know that AI replies are degraded,
    unavailable or misconfigured, and when that was last checked. It does not
    need to know WHICH provider or model is failing, and ``detail`` is prose
    that names them — so the caller gets a generic sentence for the status it
    already has, and the per-model list is emptied rather than summarised.

    Emptied rather than dropped so the field's type never changes: a client
    iterating ``models`` keeps working and simply finds nothing.
    """
    if not isinstance(health, dict):
        return {}
    status = str(health.get("status") or "unknown")
    return {
        "status": status,
        "checked_at": health.get("checked_at"),
        "detail": _GENERIC_HEALTH_DETAIL.get(
            status, _GENERIC_HEALTH_DETAIL["unknown"]
        ),
        "models": [],
    }


# One sentence per status, saying what an operator can act on and nothing about
# what is behind it. Keyed by the statuses services.model_availability sets.
_GENERIC_HEALTH_DETAIL: dict[str, str] = {
    "healthy": "AI replies are available.",
    "degraded": (
        "AI replies are degraded. A configured writer is failing; the "
        "deployment's fallback is carrying the traffic it can."
    ),
    "unavailable": (
        "AI replies are unavailable. No configured writer can be reached."
    ),
    "misconfigured": (
        "AI replies are not configured correctly and may not be generated. "
        "This needs the platform operator."
    ),
    "check_failed": (
        "AI reply availability could not be verified. Existing fallback "
        "behaviour remains active."
    ),
    "unknown": "AI reply availability has not been checked yet.",
}
