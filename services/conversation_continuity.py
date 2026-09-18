"""Keeping and closing the obligations a conversation is carrying.

``docs/autonomy_architecture_review.md`` §4 asks for a layer that nothing in
this codebase had: open threads that "persist until fulfilled, cancelled,
superseded, or expired", with provenance and correction semantics, scoped by
creator and customer. §1 says why, in the terms a test can check:

* *A question deferred across 30+ turns, two unrelated topics intervene* —
  caught only if the question outlives the recent-message window.
* *Preference correction, old preference appears in retrieved history* — caught
  only if a correction supersedes rather than sitting beside what it corrects.
* *Multiple open threads, customer returns to the earlier of two subjects* —
  caught only if both were kept, not just the newest.
* *Explicit decline or goodbye, previously queued follow-up becomes due* —
  caught only if a thread can be cancelled by something the customer said.

Design rules this module holds:

**Recording is idempotent.** A question mentioned across four turns is one
obligation, not four. ``dedupe_key`` is derived from the conversation, the kind
and the subject, so re-recording refreshes ``last_seen_at`` instead of piling up
duplicates that would each claim a share of the prompt.

**Resolution is against a stated condition.** A thread records what would close
it when it is raised, so closing it later is a test rather than a judgement made
by whoever happens to be looking.

**Correction supersedes, and keeps the link.** Nothing is deleted. A superseded
thread points at the one that replaced it, so "he changed his mind about X" is
answerable, and the old preference cannot be reasserted from history as though
it were still current.

**Nothing here may stop a reply.** Every function swallows its own failures and
returns an empty or unchanged result. Continuity is an input to a turn, never a
precondition for one — the same rule ``services/reply_provenance.py`` follows,
and for the same reason: a memory layer that can take the conversation down is
worse than no memory layer.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timedelta
from typing import Any

from core import clock
from core.supabase import get_supabase
from models.conversation_continuity import (
    ConversationEpisode,
    OpenThread,
    ResolvedBy,
    ThreadKind,
    ThreadParty,
    ThreadStatus,
)

THREADS_TABLE = "conversation_open_threads"
EPISODES_TABLE = "conversation_episodes"

#: How many open threads a turn may be given. The review asks to "reserve
#: context space for unresolved obligations"; this is the reservation's size.
#: Small on purpose — a reply that tries to service eight open threads at once
#: is an interrogation, and review §1 lists forced questioning as a failure.
DEFAULT_THREAD_LIMIT = 6

#: How many past episodes a turn may be given.
DEFAULT_EPISODE_LIMIT = 4

#: Default lifetime for a thread whose caller does not set one. Deliberately
#: long: an unanswered question does not stop mattering because a week passed,
#: and review §1 tests a return "after one day and one week". A caller with a
#: genuinely short-lived obligation passes its own.
DEFAULT_THREAD_TTL = timedelta(days=45)


def _now() -> datetime:
    """Now, as the evaluation clock sees it.

    Thread expiry and "is this obligation still current" are two of the three
    surfaces core/clock.py exists for: a trajectory testing a return after a
    week cannot test anything if a week never passes. Identical to the wall
    clock in every deployment that has not explicitly enabled the eval clock,
    which is all of them by default.
    """
    return clock.now()


_SUBJECT_NOISE = re.compile(r"[^a-z0-9 ]+")


def subject_key(text: object) -> str:
    """A stable key for "the same topic", robust to rewording.

    Lowercased, punctuation stripped, the first few significant words kept. It
    is not semantic matching and does not pretend to be: two genuinely different
    phrasings of one question will produce two threads, and the prompt will show
    both. That is a visible, harmless duplicate; the alternative — collapsing
    two different obligations onto one key — silently loses one of them.
    """
    words = _SUBJECT_NOISE.sub(" ", str(text or "").lower()).split()
    significant = [word for word in words if len(word) > 2][:6]
    return "-".join(significant) or "unspecified"


def dedupe_key_for(kind: ThreadKind, raised_by: ThreadParty, summary: str) -> str:
    """One obligation, however many turns mention it."""
    digest = hashlib.sha256(
        f"{kind.value}|{raised_by.value}|{subject_key(summary)}".encode("utf-8")
    ).hexdigest()
    return f"{kind.value}:{digest[:16]}"


def episode_key_for(first_message_at: datetime, last_message_at: datetime) -> str:
    """One episode per source range, so re-summarising the same stretch is idempotent."""
    digest = hashlib.sha256(
        f"{first_message_at.isoformat()}|{last_message_at.isoformat()}".encode("utf-8")
    ).hexdigest()
    return f"episode:{digest[:16]}"


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


async def record_open_thread(thread: OpenThread) -> OpenThread | None:
    """Keep one unfinished thing, or refresh it if it is already known.

    Returns the thread as it now stands, or ``None`` when it could not be
    recorded. A caller must not treat ``None`` as a reason to stop: losing a
    thread costs continuity on a later turn, never this one.
    """
    if not (thread.creator_id and thread.fan_id and thread.summary):
        return None
    key = dedupe_key_for(thread.kind, thread.raised_by, thread.summary)
    expires_at = thread.expires_at or (_now() + DEFAULT_THREAD_TTL)
    payload = thread.model_copy(update={"expires_at": expires_at}).to_row(
        dedupe_key=key
    )

    def _write() -> dict[str, Any] | None:
        db = get_supabase()
        existing = (
            db.table(THREADS_TABLE)
            .select("*")
            .eq("creator_id", thread.creator_id)
            .eq("fan_id", thread.fan_id)
            .eq("dedupe_key", key)
            .limit(1)
            .execute()
        )
        rows = list(existing.data or [])
        if rows:
            current = rows[0]
            if str(current.get("status") or "") != ThreadStatus.OPEN.value:
                # Already closed. Mentioning a resolved obligation again does
                # not reopen it: that is how a fulfilled promise comes back as
                # an outstanding one and gets kept twice.
                return current
            updated = (
                db.table(THREADS_TABLE)
                .update(
                    {
                        "last_seen_at": _now().isoformat(),
                        "updated_at": _now().isoformat(),
                        "expires_at": expires_at.isoformat(),
                    }
                )
                .eq("id", current["id"])
                .execute()
            )
            return (list(updated.data or []) or [current])[0]
        inserted = db.table(THREADS_TABLE).insert(payload).execute()
        return (list(inserted.data or []) or [None])[0]

    try:
        row = await asyncio.to_thread(_write)
    except Exception as exc:
        print(f"[CONTINUITY] could not record thread fan={thread.fan_id}: {exc}")
        return None
    return OpenThread.from_row(row) if row else None


async def record_episode(episode: ConversationEpisode) -> ConversationEpisode | None:
    """Summarise a completed stretch of conversation, once per source range."""
    if not (episode.creator_id and episode.fan_id and episode.summary):
        return None
    key = episode_key_for(episode.first_message_at, episode.last_message_at)
    payload = episode.to_row(dedupe_key=key)

    def _write() -> dict[str, Any] | None:
        db = get_supabase()
        existing = (
            db.table(EPISODES_TABLE)
            .select("*")
            .eq("creator_id", episode.creator_id)
            .eq("fan_id", episode.fan_id)
            .eq("dedupe_key", key)
            .limit(1)
            .execute()
        )
        rows = list(existing.data or [])
        if rows:
            return rows[0]
        inserted = db.table(EPISODES_TABLE).insert(payload).execute()
        return (list(inserted.data or []) or [None])[0]

    try:
        row = await asyncio.to_thread(_write)
    except Exception as exc:
        print(f"[CONTINUITY] could not record episode fan={episode.fan_id}: {exc}")
        return None
    return ConversationEpisode.from_row(row) if row else None


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------


async def resolve_thread(
    thread_id: str,
    *,
    status: ThreadStatus,
    resolved_by: ResolvedBy,
    note: str = "",
) -> bool:
    """Close one thread. Returns whether this call closed it.

    Guarded on ``status = 'open'``, the same predicate discipline as
    ``services/ppv_delivery_ledger.py``: two workers noticing the same answer
    must not both claim to have closed it, and a closed thread must not be
    reopened and re-closed with a different reason.
    """
    if status == ThreadStatus.OPEN:
        raise ValueError("resolve_thread cannot set a thread back to open")
    now = _now().isoformat()

    def _close() -> bool:
        result = (
            get_supabase()
            .table(THREADS_TABLE)
            .update(
                {
                    "status": status.value,
                    "resolved_at": now,
                    "resolved_by": resolved_by.value,
                    "resolution_note": note[:400],
                    "updated_at": now,
                }
            )
            .eq("id", thread_id)
            .eq("status", ThreadStatus.OPEN.value)
            .execute()
        )
        return bool(result.data)

    try:
        return await asyncio.to_thread(_close)
    except Exception as exc:
        print(f"[CONTINUITY] could not resolve thread {thread_id}: {exc}")
        return False


async def resolve_referenced_thread(
    thread_id: str,
    *,
    creator_id: str,
    fan_id: str,
    status: ThreadStatus,
    resolved_by: ResolvedBy,
    note: str = "",
) -> bool:
    """Close an analyzer-referenced thread inside one conversation only.

    An opaque id is necessary but not sufficient authority.  The creator, fan
    and current OPEN status are predicates on the same update, so a stale,
    cross-customer or cross-tenant reference changes no row.
    """
    if status == ThreadStatus.OPEN:
        raise ValueError("resolve_referenced_thread cannot set a thread back to open")
    if not (thread_id and creator_id and fan_id):
        return False
    now = _now().isoformat()

    def _close() -> bool:
        result = (
            get_supabase()
            .table(THREADS_TABLE)
            .update(
                {
                    "status": status.value,
                    "resolved_at": now,
                    "resolved_by": resolved_by.value,
                    "resolution_note": note[:400],
                    "updated_at": now,
                }
            )
            .eq("id", thread_id)
            .eq("creator_id", creator_id)
            .eq("fan_id", fan_id)
            .eq("status", ThreadStatus.OPEN.value)
            .execute()
        )
        return bool(result.data)

    try:
        return await asyncio.to_thread(_close)
    except Exception as exc:
        print(f"[CONTINUITY] could not resolve referenced thread {thread_id}: {exc}")
        return False


async def supersede_thread(old_thread_id: str, new_thread: OpenThread) -> OpenThread | None:
    """Record a correction: the new thread replaces the old one.

    §4: *A later correction should supersede an old preference, not create two
    simultaneously authoritative facts.* The order matters — the replacement is
    written first, so a failure between the two steps leaves the old thread open
    rather than leaving the conversation with neither.
    """
    replacement = await record_open_thread(new_thread)
    if replacement is None or not replacement.id:
        return None

    now = _now().isoformat()

    def _supersede() -> bool:
        result = (
            get_supabase()
            .table(THREADS_TABLE)
            .update(
                {
                    "status": ThreadStatus.SUPERSEDED.value,
                    "resolved_at": now,
                    "resolved_by": ResolvedBy.SUPERSESSION.value,
                    "superseded_by": replacement.id,
                    "updated_at": now,
                }
            )
            .eq("id", old_thread_id)
            .eq("status", ThreadStatus.OPEN.value)
            .execute()
        )
        return bool(result.data)

    try:
        await asyncio.to_thread(_supersede)
    except Exception as exc:
        # The replacement exists and the old thread is still open, so the
        # conversation now shows both. Visible and wrong beats invisible and
        # wrong: an operator can see two threads, not a silently lost one.
        print(
            f"[CONTINUITY] recorded the correction but could not supersede "
            f"{old_thread_id}: {exc}"
        )
    return replacement


async def supersede_referenced_thread(
    old_thread_id: str,
    new_thread: OpenThread,
    *,
    creator_id: str,
    fan_id: str,
    note: str = "",
) -> OpenThread | None:
    """Supersede a named OPEN row, guarded by its conversation scope.

    The replacement is never allowed to inherit scope from model output.  It is
    forced to the caller's creator/fan pair, and the old-row update repeats the
    same predicates.  If the reference is stale or belongs elsewhere, nothing
    is recorded and nothing is closed.
    """
    if not (old_thread_id and creator_id and fan_id):
        return None

    def _exists() -> bool:
        result = (
            get_supabase()
            .table(THREADS_TABLE)
            .select("id")
            .eq("id", old_thread_id)
            .eq("creator_id", creator_id)
            .eq("fan_id", fan_id)
            .eq("status", ThreadStatus.OPEN.value)
            .limit(1)
            .execute()
        )
        return bool(result.data)

    try:
        exists = await asyncio.to_thread(_exists)
    except Exception as exc:
        print(f"[CONTINUITY] could not validate thread {old_thread_id}: {exc}")
        return None
    if not exists:
        return None

    replacement = await record_open_thread(
        new_thread.model_copy(
            update={"creator_id": creator_id, "fan_id": fan_id, "id": ""}
        )
    )
    if replacement is None or not replacement.id:
        return None

    now = _now().isoformat()

    def _supersede() -> bool:
        result = (
            get_supabase()
            .table(THREADS_TABLE)
            .update(
                {
                    "status": ThreadStatus.SUPERSEDED.value,
                    "resolved_at": now,
                    "resolved_by": ResolvedBy.SUPERSESSION.value,
                    "resolution_note": note[:400],
                    "superseded_by": replacement.id,
                    "updated_at": now,
                }
            )
            .eq("id", old_thread_id)
            .eq("creator_id", creator_id)
            .eq("fan_id", fan_id)
            .eq("status", ThreadStatus.OPEN.value)
            .execute()
        )
        return bool(result.data)

    try:
        changed = await asyncio.to_thread(_supersede)
    except Exception as exc:
        print(f"[CONTINUITY] could not supersede referenced thread {old_thread_id}: {exc}")
        return None
    return replacement if changed else None


async def expire_due_threads(
    creator_id: str, fan_id: str, *, now: datetime | None = None
) -> int:
    """Close threads whose time is up. Returns how many.

    Called on the read path rather than by a sweeper: an expired obligation only
    matters when somebody is about to build a prompt from it, and a background
    job whose whole purpose is one UPDATE is a thing to operate for no gain.
    """
    moment = (now or _now()).isoformat()

    def _expire() -> int:
        result = (
            get_supabase()
            .table(THREADS_TABLE)
            .update(
                {
                    "status": ThreadStatus.EXPIRED.value,
                    "resolved_at": moment,
                    "resolved_by": ResolvedBy.EXPIRY.value,
                    "updated_at": moment,
                }
            )
            .eq("creator_id", creator_id)
            .eq("fan_id", fan_id)
            .eq("status", ThreadStatus.OPEN.value)
            .lt("expires_at", moment)
            .execute()
        )
        return len(list(result.data or []))

    try:
        return await asyncio.to_thread(_expire)
    except Exception as exc:
        print(f"[CONTINUITY] expiry sweep failed fan={fan_id}: {exc}")
        return 0


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


async def open_threads_for(
    creator_id: str,
    fan_id: str,
    *,
    limit: int = DEFAULT_THREAD_LIMIT,
    expire_first: bool = True,
) -> list[OpenThread]:
    """What this conversation is still carrying, most recently seen first.

    Scoped by creator AND fan, as §4 requires: the same customer talking to two
    creators has two sets of obligations, and one must never be answered with
    the other's.
    """
    if not (creator_id and fan_id):
        return []
    if expire_first:
        await expire_due_threads(creator_id, fan_id)

    def _read() -> list[dict[str, Any]]:
        result = (
            get_supabase()
            .table(THREADS_TABLE)
            .select("*")
            .eq("creator_id", creator_id)
            .eq("fan_id", fan_id)
            .eq("status", ThreadStatus.OPEN.value)
            .order("last_seen_at", desc=True)
            .limit(max(1, int(limit)))
            .execute()
        )
        return list(result.data or [])

    try:
        rows = await asyncio.to_thread(_read)
    except Exception as exc:
        print(f"[CONTINUITY] could not read threads fan={fan_id}: {exc}")
        return []
    return [OpenThread.from_row(row) for row in rows]


async def recent_episodes_for(
    creator_id: str, fan_id: str, *, limit: int = DEFAULT_EPISODE_LIMIT
) -> list[ConversationEpisode]:
    """What previous stretches of this conversation were about, newest first."""
    if not (creator_id and fan_id):
        return []

    def _read() -> list[dict[str, Any]]:
        result = (
            get_supabase()
            .table(EPISODES_TABLE)
            .select("*")
            .eq("creator_id", creator_id)
            .eq("fan_id", fan_id)
            .order("last_message_at", desc=True)
            .limit(max(1, int(limit)))
            .execute()
        )
        return list(result.data or [])

    try:
        rows = await asyncio.to_thread(_read)
    except Exception as exc:
        print(f"[CONTINUITY] could not read episodes fan={fan_id}: {exc}")
        return []
    return [ConversationEpisode.from_row(row) for row in rows]


def rank_threads(threads: list[OpenThread]) -> list[OpenThread]:
    """Order threads by what a good operator would deal with first.

    Not by recency. Review §1: *a request to fix access outranks a new
    commercial suggestion*, and *the customer returns to the earlier of two
    subjects* is a failure precisely because the newest was treated as the only
    context. So: unresolved complaints first, then what he is waiting on us for,
    then everything else — and within each group the OLDEST first, because an
    obligation that has been outstanding longer is the one more likely to have
    been forgotten.
    """

    def sort_key(thread: OpenThread) -> tuple:
        if thread.kind == ThreadKind.COMPLAINT:
            group = 0
        elif thread.is_obligation_on_us:
            group = 1
        elif thread.kind == ThreadKind.CORRECTION:
            # A correction is not an obligation, but reasserting something the
            # customer already corrected is its own failure mode, so it stays
            # ahead of ordinary deferred chat.
            group = 2
        else:
            group = 3
        seen = thread.first_seen_at or _now()
        return (group, seen)

    return sorted(threads, key=sort_key)


def summarize_threads(threads: list[OpenThread], *, limit: int = DEFAULT_THREAD_LIMIT) -> list[str]:
    """The lines a prompt gets: ranked, capped, and free of internal vocabulary."""
    return [thread.render() for thread in rank_threads(threads)[: max(0, int(limit))]]


def summarize_thread_references(
    threads: list[OpenThread], *, limit: int = DEFAULT_THREAD_LIMIT
) -> list[str]:
    """Analyzer-only lines with the opaque ids it may return.

    Writers keep receiving ``summarize_threads`` without ids.  Identifiers are
    control data for lifecycle proposals, not prose for a reply to imitate.
    """
    return [
        f"[thread_id={thread.id}] {thread.render()}"
        for thread in rank_threads(threads)[: max(0, int(limit))]
        if thread.id
    ]
