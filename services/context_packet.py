"""One budgeted view of a conversation, shared by everything that reads one.

``docs/autonomy_architecture_review.md`` finding D, confirmed:

    ``db/queries.py::get_conversation_history`` defaults to the latest 40
    messages. ``ai/situation_analyzer.py::build_analyzer_prompt`` renders only
    the last 12; ``ai/prompt_builder.py::build_prompt`` renders the last 16.
    These are message bubbles, not complete conversational turns. Multipart
    replies consume the window faster. [...] This creates an evidence
    asymmetry: a decision-driving classifier can lack context that exists
    elsewhere in the system.

Three separate problems in one paragraph, and this module answers each.

**Bubbles are not turns.** A reply split into three bubbles spent three
sixteenths of the writer's window saying one thing. Grouping consecutive
messages from the same speaker into a turn makes the budget mean what it
appears to mean, and makes it stable: the same exchange costs the same whether
the writer happened to send it as one message or four.

**The analyzer saw less than the writer.** The classifier that decides what the
turn DOES had a narrower view than the model that writes it. Both now build from
here, with the same turn budget, so an interpretation can no longer be made on
evidence the writer has and the analyzer does not.

**Small talk evicted unfinished business.** §4: *reserve context space for
unresolved obligations; do not drop them merely because small talk filled a
16-bubble window.* Obligations are a separate allowance here, taken before the
transcript is measured. A conversation can push a question out of the recent
window; it cannot push it out of the packet.

This module is pure. No database, no clock, no I/O — it takes the history, the
open threads and the episodes it is given and returns a view of them. That is
what lets the evaluation harness build the identical packet the live path builds
(§5 requires candidates to be compared on the same evidence), and what makes
every rule here testable without a fixture.
"""

from __future__ import annotations

import hashlib

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

#: How many complete turns a reply is written from.
#:
#: Sixteen bubbles was the old writer window. Twelve turns is at least as much
#: conversation in every case and considerably more when either side sends
#: multipart replies, which this product does by design
#: (services/message_shape.py). Measured in turns so the number means something.
DEFAULT_TURN_BUDGET = 12

#: The ceiling on rendered transcript size, in characters.
#:
#: A turn budget alone is not a budget: one pasted wall of text can be longer
#: than a hundred ordinary turns. Oldest turns are dropped first when this
#: binds, because the newest are what the reply is answering.
DEFAULT_TRANSCRIPT_CHARS = 6000

#: The reservation for unresolved obligations, taken before the transcript is
#: measured. Small deliberately: review §1 lists "forced commercial pivots,
#: repetitive questioning" as failures, and a reply that services six
#: obligations at once is an interrogation.
DEFAULT_THREAD_BUDGET = 6

#: How many earlier episodes are worth recalling. Enough to answer "we talked
#: about this before"; not a second transcript.
DEFAULT_EPISODE_BUDGET = 4


@dataclass(frozen=True)
class ContextBudget:
    """What one reader of the conversation is allowed to see."""

    turns: int = DEFAULT_TURN_BUDGET
    transcript_chars: int = DEFAULT_TRANSCRIPT_CHARS
    threads: int = DEFAULT_THREAD_BUDGET
    episodes: int = DEFAULT_EPISODE_BUDGET


#: The budget the writer and the analyzer now share. Finding D is the reason it
#: is one object and not two: the asymmetry was the bug.
STANDARD_BUDGET = ContextBudget()


@dataclass(frozen=True)
class ConversationTurn:
    """Everything one speaker said before the other replied."""

    speaker: str
    bubbles: tuple[str, ...]
    at: datetime | None = None

    @property
    def text(self) -> str:
        return " ".join(self.bubbles)

    def render(self, *, fan_name: str = "Fan", creator_name: str = "You") -> str:
        """One line, with a multipart turn kept visibly multipart.

        The separator matters: a reply that arrived as three bubbles reads
        differently from one long sentence, and flattening it would teach the
        writer a rhythm the conversation does not have.
        """
        who = fan_name if self.speaker == "fan" else creator_name
        return f"{who}: {' | '.join(self.bubbles)}"


@dataclass(frozen=True)
class ContextPacket:
    """The conversation as one reader is allowed to see it.

    ``dropped_turns`` and ``dropped_threads`` are carried rather than discarded
    so a reply's provenance record can say what the turn was NOT shown. "The
    model did not mention the thing he asked about" and "the model was never
    told about it" are different failures with different fixes, and without
    these counts they look identical afterwards.
    """

    turns: tuple[ConversationTurn, ...] = ()
    open_threads: tuple[str, ...] = ()
    episodes: tuple[str, ...] = ()
    dropped_turns: int = 0
    dropped_threads: int = 0
    dropped_episodes: int = 0
    #: Turns shortened to fit, and how many characters went. A turn that was
    #: cut is not the same as one that was dropped and not the same as one that
    #: arrived whole, and a reply attributed to "the model had the message"
    #: needs to know which of the three happened.
    truncated_turns: int = 0
    truncated_chars: int = 0
    budget: ContextBudget = field(default_factory=ContextBudget)

    @property
    def message_count(self) -> int:
        """How many original bubbles the rendered turns came from."""
        return sum(len(turn.bubbles) for turn in self.turns)

    def render_transcript(
        self, *, fan_name: str = "Fan", creator_name: str = "You"
    ) -> str:
        lines = [
            turn.render(fan_name=fan_name, creator_name=creator_name)
            for turn in self.turns
        ]
        return "\n".join(line for line in lines if line.strip())

    def render_continuity(self) -> str:
        """The unresolved-obligations block, or empty when there is nothing.

        Phrased as what is outstanding, not as instructions. §4 is explicit that
        a decision object "should not prescribe a mandatory emotional ladder or
        a fixed sentence shape", and the same restraint applies here: these are
        facts about the conversation, and what to do about them is the reply's
        business.
        """
        sections: list[str] = []
        if self.open_threads:
            sections.append(
                "STILL OPEN IN THIS CONVERSATION (unfinished, not instructions):\n"
                + "\n".join(f"- {line}" for line in self.open_threads)
            )
        if self.episodes:
            sections.append(
                "EARLIER CONVERSATIONS (what they were about):\n"
                + "\n".join(f"- {line}" for line in self.episodes)
            )
        return "\n\n".join(sections)

    def content_digest(self) -> str:
        """A hash of what this packet actually contains.

        The counts below cannot tell two packets apart: a conversation about
        Chicago and a conversation about Boston are both "1 turn, 1 message, 0
        threads, 0 episodes", so two provenance records could agree in every
        field while the model saw entirely different evidence. That makes the
        replay comparison §5 asks for — hold the evidence fixed, change the
        thing under test — unverifiable, because nothing could confirm the
        evidence was held fixed.

        A digest is not content. It stores no message text, cannot be reversed
        into any, and keeps the same discipline ``services/reply_provenance.py``
        keeps for the same reason. What it adds is the ability to say "these two
        replies were produced from the same input" and be right.
        """
        material = "\n".join(
            [
                self.render_transcript(),
                "\u0000threads\u0000",
                *self.open_threads,
                "\u0000episodes\u0000",
                *self.episodes,
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def fingerprint(self) -> dict[str, Any]:
        """What went into this packet, for a reply's provenance record.

        Counts, budgets and a content digest — never content itself.
        """
        return {
            "turns": len(self.turns),
            "messages": self.message_count,
            "open_threads": len(self.open_threads),
            "episodes": len(self.episodes),
            "dropped_turns": self.dropped_turns,
            "dropped_threads": self.dropped_threads,
            "dropped_episodes": self.dropped_episodes,
            "truncated_turns": self.truncated_turns,
            "truncated_chars": self.truncated_chars,
            "turn_budget": self.budget.turns,
            "transcript_char_budget": self.budget.transcript_chars,
            # The one field that distinguishes two packets with identical
            # counts, which is most of them.
            "content_digest": self.content_digest(),
        }


def group_into_turns(history: Sequence[Any]) -> list[ConversationTurn]:
    """Collapse consecutive messages from one speaker into a single turn.

    This is the fix for "these are message bubbles, not complete conversational
    turns". Takes anything with ``role`` and ``content`` — the ``Message`` model,
    a row dict, a stub in a test — because the evaluation harness builds turns
    from replayed records that are not ``Message`` instances.

    Empty content is skipped rather than producing an empty bubble: a blank
    message is not a thing anybody said.
    """
    turns: list[ConversationTurn] = []
    for message in history or []:
        role = str(getattr(message, "role", None) or (
            message.get("role") if isinstance(message, dict) else ""
        ) or "").strip().lower()
        raw = getattr(message, "content", None)
        if raw is None and isinstance(message, dict):
            raw = message.get("content")
        content = str(raw or "").strip()
        if not content:
            continue
        speaker = "fan" if role == "fan" else "creator"
        at = getattr(message, "sent_at", None)
        if at is None and isinstance(message, dict):
            at = message.get("sent_at")

        if turns and turns[-1].speaker == speaker:
            previous = turns[-1]
            turns[-1] = ConversationTurn(
                speaker=speaker,
                bubbles=previous.bubbles + (content,),
                # The turn is stamped with when it FINISHED, which is what
                # "how long ago did he last say something" actually means.
                at=at or previous.at,
            )
        else:
            turns.append(
                ConversationTurn(speaker=speaker, bubbles=(content,), at=at)
            )
    return turns


def build_context_packet(
    history: Sequence[Any],
    *,
    open_threads: Sequence[str] = (),
    episodes: Sequence[str] = (),
    budget: ContextBudget = STANDARD_BUDGET,
) -> ContextPacket:
    """Assemble what one reader of this conversation may see.

    The order is the point. Obligations and episodes are allocated FIRST, from
    their own allowances, so no amount of recent chatter can evict them. The
    transcript then takes what the turn budget and the character ceiling allow,
    newest first, because the newest turns are what the reply is answering.

    Nothing here decides anything or calls anything. Given the same inputs it
    returns the same packet, which is what lets §5's replay comparison hold the
    evidence fixed while the thing being compared changes.
    """
    kept_threads = tuple(
        line for line in list(open_threads)[: max(0, budget.threads)] if str(line).strip()
    )
    dropped_threads = max(0, len(list(open_threads)) - len(kept_threads))

    kept_episodes = tuple(
        line for line in list(episodes)[: max(0, budget.episodes)] if str(line).strip()
    )
    dropped_episodes = max(0, len(list(episodes)) - len(kept_episodes))

    all_turns = group_into_turns(history)
    if budget.turns <= 0:
        return ContextPacket(
            turns=(),
            open_threads=kept_threads,
            episodes=kept_episodes,
            dropped_turns=len(all_turns),
            dropped_threads=dropped_threads,
            dropped_episodes=dropped_episodes,
            budget=budget,
        )

    windowed = all_turns[-budget.turns :]

    # The character ceiling, applied newest-first. One pasted wall of text must
    # not be able to push out the exchange it was pasted into.
    ceiling = max(0, budget.transcript_chars)
    kept: list[ConversationTurn] = []
    used = 0
    truncated_turns = 0
    truncated_chars = 0
    for turn in reversed(windowed):
        cost = len(turn.render()) + 1
        if not kept and cost > ceiling:
            # The newest turn does not fit on its own. It used to be admitted
            # whole anyway — the guard was `if kept and ...` — so a single
            # 10,000-character message rendered a 10,005-character transcript
            # against a 6,000 budget, AND took the exchange it was pasted into
            # down with it, since nothing else could follow.
            #
            # Neither extreme is right: dropping it loses the message the reply
            # is answering, and keeping it whole makes the budget a suggestion.
            # So it is shortened, and the shortening is recorded.
            shortened = _shorten(turn, ceiling)
            if shortened is None:
                break
            truncated_turns += 1
            truncated_chars += len(turn.text) - len(shortened.text)
            kept.append(shortened)
            used += len(shortened.render()) + 1
            continue
        if kept and used + cost > ceiling:
            break
        kept.append(turn)
        used += cost
    kept.reverse()

    return ContextPacket(
        turns=tuple(kept),
        open_threads=kept_threads,
        episodes=kept_episodes,
        dropped_turns=max(0, len(all_turns) - len(kept)),
        dropped_threads=dropped_threads,
        dropped_episodes=dropped_episodes,
        truncated_turns=truncated_turns,
        truncated_chars=truncated_chars,
        budget=budget,
    )


#: Marks where a turn was shortened, and by how much. Visible on purpose: a
#: reader — human or model — must be able to tell an abridged message from a
#: complete one, or it will answer the abridgement as though it were the whole.
ELISION = "[… {count} characters omitted …]"


def _shorten(turn: ConversationTurn, ceiling: int) -> ConversationTurn | None:
    """One turn cut to fit, keeping both ends.

    Head AND tail, not just the head. A long message very often puts its point
    at the end — "…and anyway, can you send me the other one?" — and a head-only
    cut would reliably discard exactly the part the reply has to answer.

    Returns None when the ceiling cannot fit even the scaffolding, in which case
    the caller drops the turn rather than emitting a marker with no message
    around it.
    """
    prefix = len(turn.render()) - len(turn.text)
    room = ceiling - prefix - len(ELISION.format(count=999999)) - 1
    if room < 40:
        return None

    head = room * 2 // 3
    tail = room - head
    text = turn.text
    removed = len(text) - head - tail
    if removed <= 0:
        return turn
    cut = f"{text[:head]} {ELISION.format(count=removed)} {text[-tail:]}"
    return ConversationTurn(speaker=turn.speaker, bubbles=(cut,), at=turn.at)
