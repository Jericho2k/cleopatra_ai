"""Closing a stretch of conversation once it is over.

THE GAP
-------
``conversation_episodes`` had no producer at all. ``record_episode`` existed,
was tested, and was called by nothing — so "we talked about this before" was a
table shape rather than something the system could ever say.

WHEN A STRETCH IS OVER
----------------------
When he comes back after a gap. That is the only moment the previous stretch
is knowably complete: while a conversation is running, any turn might continue
it, and summarising a live exchange produces a summary that is immediately
wrong.

So this runs at the START of a turn, about the turns BEFORE it, and only when
the gap is long enough that the previous stretch has plainly ended.

WHY THE SUMMARY IS NOT WRITTEN BY A MODEL
-----------------------------------------
It would be an invention. Asked to summarise a stretch of conversation, a model
produces fluent prose about what it thinks happened, and this record is then
read back to it turns later as though it were evidence — a belief the system
formed about itself, laundered into a fact. The review's objection to the
baseline was exactly this shape: a claim with nothing behind it.

The summary is assembled instead from the obligations that stretch actually
raised, which are rows with source turn ids and fingerprints behind them. When
it raised none, the episode says what it factually was — how long, how many
messages, how it ended — and claims nothing about content. A dull true record
beats an interesting invented one, and this one is read back to a model.

``ended_with`` is a reading of who spoke last, not a judgement about why:
* the creator spoke last and he did not come back — he went quiet;
* he spoke last and nobody answered before the gap — it was cut short.

AND NEVER MONEY
---------------
``ConversationEpisode`` carries no amount, no price and no purchase flag, and
the model says why: an episode is never proof of payment. The summary is built
from thread subjects, and ``services/continuity_extraction.py`` has already
refused any of those that mentioned money — so the exclusion holds here by
construction rather than by a second rule that could drift from the first.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Sequence

from core import clock
from models.conversation_continuity import (
    ConversationEpisode,
    EpisodeEnding,
    EvidenceType,
)

#: How long a silence has to be before the stretch before it counts as closed.
#:
#: Six hours: long enough that an ordinary pause mid-conversation — a meal, a
#: shift, a night's sleep in another timezone — does not chop one exchange into
#: three episodes, and short enough that "he came back the next day" reliably
#: closes the previous day.
EPISODE_GAP = timedelta(hours=6)

#: Below this, the stretch is not worth a record. Two messages and a gap is
#: somebody saying "hey" and losing interest, and an episode per abandoned
#: hello would bury the ones that mean something.
MIN_EPISODE_MESSAGES = 4

#: The summary is read in a context packet beside a transcript. A long one is a
#: second transcript, which is the thing this table exists not to be.
MAX_SUBJECTS = 3


def _at(message: Any) -> datetime | None:
    raw = getattr(message, "sent_at", None)
    if raw is None and isinstance(message, dict):
        raw = message.get("sent_at")
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=clock.now().tzinfo)
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=clock.now().tzinfo)


def _role(message: Any) -> str:
    raw = getattr(message, "role", None)
    if raw is None and isinstance(message, dict):
        raw = message.get("role")
    return str(raw or "").strip().lower()


def find_closed_stretch(history: Sequence[Any], *, gap: timedelta = EPISODE_GAP):
    """The messages before the most recent long silence, or None.

    Returns the trailing stretch that ENDED at that silence — not everything
    before it. An older episode was already closed by its own gap, and
    re-summarising it would produce a second record of the same conversation
    with a different shape.
    """
    timed = [(message, _at(message)) for message in history or []]
    timed = [(message, at) for message, at in timed if at is not None]
    if len(timed) < MIN_EPISODE_MESSAGES + 1:
        return None

    split = None
    for index in range(len(timed) - 1, 0, -1):
        if timed[index][1] - timed[index - 1][1] >= gap:
            split = index
            break
    if split is None:
        return None

    stretch = timed[:split]
    # The stretch before the gap may itself contain an earlier gap; take only
    # the part after it, which is the episode this silence actually closed.
    for index in range(len(stretch) - 1, 0, -1):
        if stretch[index][1] - stretch[index - 1][1] >= gap:
            stretch = stretch[index:]
            break
    if len(stretch) < MIN_EPISODE_MESSAGES:
        return None
    return stretch


def _ending(stretch) -> EpisodeEnding:
    """How it ended, read off who spoke last."""
    last_role = _role(stretch[-1][0])
    if last_role == "creator":
        # She answered and he did not come back.
        return EpisodeEnding.WENT_QUIET
    if last_role == "fan":
        # He spoke and nothing followed before the silence.
        return EpisodeEnding.INTERRUPTED
    return EpisodeEnding.UNKNOWN


def _summary(stretch, subjects: Sequence[str]) -> str:
    """What it was about, from records, or what it factually was.

    Never invented prose. ``subjects`` are the summaries of obligations that
    stretch raised — rows with source turn ids behind them — and when there are
    none the episode describes itself instead of guessing at content.
    """
    kept = [str(subject).strip() for subject in subjects if str(subject).strip()]
    if kept:
        return "; ".join(kept[:MAX_SUBJECTS])

    span_hours = (stretch[-1][1] - stretch[0][1]).total_seconds() / 3600
    length = "a short exchange" if span_hours < 1 else f"about {span_hours:.0f} hours"
    return f"{length}, {len(stretch)} messages, nothing left open"


def build_episode(
    stretch,
    *,
    creator_id: str,
    fan_id: str,
    subjects: Sequence[str] = (),
) -> ConversationEpisode:
    """One closed stretch, as a record.

    ``evidence_type`` is INFERRED and stays INFERRED: this was read out of the
    conversation rather than said by anybody, and a later reader deciding how
    much to rely on it needs that distinction more than it needs a confident
    label.
    """
    return ConversationEpisode(
        creator_id=str(creator_id),
        fan_id=str(fan_id),
        summary=_summary(stretch, subjects),
        ended_with=_ending(stretch),
        first_message_at=stretch[0][1],
        last_message_at=stretch[-1][1],
        message_count=len(stretch),
        evidence_type=EvidenceType.INFERRED,
    )


async def close_finished_episode(
    *,
    creator_id: str,
    fan_id: str,
    history: Sequence[Any],
    subjects: Sequence[str] = (),
) -> ConversationEpisode | None:
    """Record the stretch this turn is returning after, if there is one.

    Never raises. An episode is memory, and a recorder that can stop a reply is
    a recorder that can cost a customer their answer —
    ``services/reply_provenance.py`` takes the same position for the same
    reason.

    Idempotent through ``record_episode``, which dedupes on the stretch's first
    and last timestamps: the same turn processed twice, or a webhook
    redelivered, writes one episode.
    """
    from services.conversation_continuity import record_episode

    try:
        stretch = find_closed_stretch(history)
        if stretch is None:
            return None
        episode = build_episode(
            stretch, creator_id=creator_id, fan_id=fan_id, subjects=subjects
        )
        return await record_episode(episode)
    except Exception as exc:  # pragma: no cover - memory never blocks a reply
        print(f"[CONTINUITY] could not close episode fan={fan_id}: {type(exc).__name__}")
        return None
