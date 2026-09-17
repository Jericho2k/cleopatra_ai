"""What an operator can see, and change, about what a conversation remembers.

WHY THIS EXISTS
---------------
The brief asks for "practical operator views for saved facts and threads with
source/correction/resolution controls", and until now there were none. The
continuity tables were written and read entirely by the machine: an operator
could see a frozen conversation and a transcript, and had no way at all to see
that the system was carrying an obligation, where it came from, or that it was
wrong.

That gap has a specific cost. These records are fed to a model on every turn.
A wrong one — a question recorded from a message that did not ask it, a
correction extracted the wrong way round — keeps being fed to it, and shows up
as the model behaving strangely for reasons nobody can trace. Without a view,
the only fix available to an operator is to stop using the conversation.

SOURCE IS PART OF THE VIEW, NOT A DETAIL
----------------------------------------
Every record carries where it came from: whether somebody said it or it was
read out of the conversation, how confident the extraction was, which turn
produced it, and when it was first and last seen. An operator deciding whether
a remembered obligation is real needs that more than they need the summary —
"he asked about Chicago" read out of an ambiguous message and the same line
stated outright are different claims, and only one of them is worth acting on.

WHAT AN OPERATOR MAY DO TO IT
-----------------------------
Resolve one, or correct one. Both go through the existing lifecycle —
``resolve_thread`` and ``supersede_thread`` — rather than editing rows, so an
operator's change carries the same provenance as the machine's and the audit
trail stays intact. A corrected thread supersedes rather than overwrites: the
obsolete version is kept and stops being carried, which is the whole point of
supersession and the reason the operator's correction behaves exactly like the
customer's.

Deliberately no delete. A record an operator disagrees with is resolved or
corrected, both of which say what happened; a deleted one leaves the next
reader wondering whether it was ever there.
"""

from __future__ import annotations

from typing import Any

from models.conversation_continuity import (
    EvidenceType,
    OpenThread,
    ResolvedBy,
    ThreadStatus,
)


def thread_view(thread: OpenThread) -> dict[str, Any]:
    """One carried obligation, as an operator needs to read it."""
    return {
        "id": thread.id,
        "kind": thread.kind.value,
        "raised_by": thread.raised_by.value,
        "summary": thread.summary,
        "resolution_condition": thread.resolution_condition,
        "status": thread.status.value,
        # Source, which is the half an operator actually judges on.
        "evidence_type": thread.evidence_type.value,
        "confidence": thread.confidence,
        "source_turn_id": thread.source_turn_id,
        "source_message_fingerprint": thread.source_message_fingerprint,
        "first_seen_at": _iso(thread.first_seen_at),
        "last_seen_at": _iso(thread.last_seen_at),
        "expires_at": _iso(thread.expires_at),
        # Said outright rather than inferred from evidence_type, because the
        # question an operator is asking is "should I trust this", and making
        # them derive the answer from an enum is how they stop reading it.
        "was_read_rather_than_said": thread.evidence_type is EvidenceType.INFERRED,
        # Whose move it is. A question he asked and one she asked are different
        # obligations, and answering the wrong one is a named failure.
        "waiting_on_us": thread.is_obligation_on_us,
    }


def episode_view(episode: Any) -> dict[str, Any]:
    """One earlier stretch of conversation."""
    return {
        "id": getattr(episode, "id", ""),
        "summary": episode.summary,
        "ended_with": episode.ended_with.value,
        "first_message_at": _iso(episode.first_message_at),
        "last_message_at": _iso(episode.last_message_at),
        "message_count": episode.message_count,
        "evidence_type": episode.evidence_type.value,
        # An episode is always read rather than said, and the model that holds
        # it says so. Restated here so a client never has to know that.
        "was_read_rather_than_said": True,
    }


def fact_view(row: Any) -> dict[str, Any]:
    """One saved fact about the customer, with where it came from.

    Takes a row rather than a model because fan_facts predates the continuity
    models and is read as dicts throughout.
    """
    if not isinstance(row, dict):
        return {}
    return {
        "id": str(row.get("id") or ""),
        "key": str(row.get("fact_key") or ""),
        "value": str(row.get("fact_value") or ""),
        "status": str(row.get("status") or ""),
        "confidence": row.get("confidence"),
        "source_message_id": str(row.get("source_message_id") or ""),
        "updated_at": str(row.get("updated_at") or ""),
    }


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None and hasattr(value, "isoformat") else None


async def load_memory(creator_id: str, fan_id: str) -> dict[str, Any]:
    """Everything this conversation remembers, for one operator to read.

    Reads the same retrieval a turn does, so what an operator sees is what the
    model gets. A separate query here would eventually disagree with the one
    that matters, and an operator debugging a strange reply against a view that
    does not match its evidence is worse off than one with no view at all.
    """
    import asyncio

    from services.conversation_continuity import (
        open_threads_for,
        recent_episodes_for,
    )

    threads, episodes = await asyncio.gather(
        open_threads_for(creator_id, fan_id),
        recent_episodes_for(creator_id, fan_id),
    )
    return {
        "creator_id": str(creator_id),
        "fan_id": str(fan_id),
        "open_threads": [thread_view(thread) for thread in threads],
        "episodes": [episode_view(episode) for episode in episodes],
        # Stated rather than left to an empty list, for the same reason the
        # access panel states it: "nothing is being carried" and "we could not
        # read it" lead an operator to different actions.
        "carrying_anything": bool(threads),
    }


async def resolve_as_operator(
    thread_id: str, *, note: str = "", cancelled: bool = False
) -> bool:
    """Close one obligation because an operator says so.

    ``cancelled`` is the difference between "this was dealt with" and "this was
    never real". An operator correcting a bad extraction is doing the second,
    and recording it as the first would teach anyone reading the history that
    the system handled something it in fact invented.
    """
    from services.conversation_continuity import resolve_thread

    resolved = await resolve_thread(
        thread_id,
        status=ThreadStatus.CANCELLED if cancelled else ThreadStatus.FULFILLED,
        resolved_by=ResolvedBy.OPERATOR,
        note=note,
    )
    return bool(resolved)


async def correct_as_operator(
    thread: OpenThread, *, summary: str, note: str = ""
) -> OpenThread | None:
    """Replace one obligation with what it should have said.

    Supersession, not an edit: the obsolete version is kept and stops being
    carried. An operator's correction therefore behaves exactly like the
    customer's, which is what makes the audit trail readable — every version of
    a record was displaced by something, and it is always possible to say what.

    Recorded as OPERATOR evidence: a human stated this, which is a stronger
    claim than anything read out of a message, and a later reader deciding how
    much to rely on it needs to know which it was.
    """
    from services.conversation_continuity import supersede_thread

    replacement = thread.model_copy(
        update={
            "summary": summary,
            "id": "",
            "status": ThreadStatus.OPEN,
            "evidence_type": EvidenceType.OPERATOR,
            "confidence": 1.0,
            "resolution_note": note,
        }
    )
    return await supersede_thread(thread.id, replacement)
