"""Whether to say anything after a purchase, and never the same thing.

WHAT THIS REPLACES
------------------
``services/suggestions.py`` held seven lines::

    "let me know what you think 🙈", "dying to know your reaction 😏", ...

picked with ``random.choice`` at SCHEDULE time and frozen into the queued
action's payload. Every customer, after every purchase, got one of seven
sentences, chosen before anything was known about how the next minute would go.

Freezing it also bypassed the writer. ``services/proactive.py`` generates from
a goal and the conversation — unless ``_delivery.text`` is already set, which
short-circuits generation entirely. So the one proactive message that follows
money changing hands was the only one that never saw the conversation.

WHAT "CONTEXT-AWARE OR SILENCE" MEANS HERE
------------------------------------------
Mostly silence. The nudge exists to catch a customer who bought and then said
nothing, and almost every reason to skip it is a reason a person would skip it:

* he already replied — he is engaged, and "don't leave me hanging" sent to
  somebody who did not leave anyone hanging reads as not having been listened
  to;
* he asked not to be followed up — a request that outranks a nudge;
* the creator's operator turned Auto off, or took the conversation over.

That last one was not checked at all. ``POST_PURCHASE_REACTION`` was explicitly
exempted from the worker's auto-mode gate, so a customer whose creator had
switched automation OFF still received an automated message about their
purchase. Operator takeover is exactly what that switch means, and money having
just changed hands is the worst moment to ignore it.

WHY THE DECISION IS AT EXECUTE TIME
-----------------------------------
Because everything it depends on can change in the thirty seconds between the
purchase and the nudge — which is most of the window. He can reply, an operator
can take over, a hold can be raised. A decision made at schedule time is a
decision made before any of that.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

#: The goal handed to the writer when a nudge IS eligible. A goal, not a line:
#: ``services/proactive.py`` builds the message from this plus the actual
#: conversation, which is the path every other proactive action already takes
#: and the one a frozen string skipped.
REACTION_GOAL = (
    "He just bought something and has not said anything since. Acknowledge it "
    "warmly in your own voice and leave the door open for him to react, in one "
    "short message. Do not ask a question he has already answered, do not "
    "state a price, do not offer anything else, and do not send media."
)


@dataclass(frozen=True)
class Eligibility:
    """Whether to send, and the reason either way.

    The reason is recorded on the action rather than dropped, so "it stayed
    quiet" can be told from "it never ran" — which, for a proactive message,
    is the difference between the product working and the queue being broken.
    """

    send: bool
    reason: str

    @property
    def silent(self) -> bool:
        return not self.send


def _sent_at(message: Any) -> datetime | None:
    raw = getattr(message, "sent_at", None)
    if raw is None and isinstance(message, dict):
        raw = message.get("sent_at")
    if isinstance(raw, datetime):
        return raw
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _role(message: Any) -> str:
    raw = getattr(message, "role", None)
    if raw is None and isinstance(message, dict):
        raw = message.get("role")
    return str(raw or "").strip().lower()


def he_replied_since(history: Sequence[Any], purchased_at: datetime | None) -> bool:
    """Whether the customer has said anything since the purchase.

    A message with no timestamp is not counted. Guessing that an undated
    message came after the purchase would suppress a nudge on no evidence,
    which is the quieter failure but still a failure.
    """
    if purchased_at is None:
        return False
    for message in history or []:
        if _role(message) != "fan":
            continue
        at = _sent_at(message)
        if at is None:
            continue
        if at.tzinfo is None:
            at = at.replace(tzinfo=purchased_at.tzinfo)
        if at > purchased_at:
            return True
    return False


def asked_for_no_follow_up(open_threads: Sequence[Any]) -> bool:
    """Whether an open obligation says to leave him alone.

    Read off the continuity records rather than re-read out of the transcript:
    the obligation is already stored with its source, and a second reading
    here could disagree with the one the rest of the system uses.
    """
    for thread in open_threads or []:
        kind = getattr(getattr(thread, "kind", None), "value", "")
        summary = str(getattr(thread, "summary", "") or "").lower()
        if kind == "deferred_topic" and (
            "no follow" in summary
            or "not message" in summary
            or "don't message" in summary
            or "leave him" in summary
            or "stop messaging" in summary
        ):
            return True
    return False


def decide(
    *,
    history: Sequence[Any],
    open_threads: Sequence[Any] = (),
    purchased_at: datetime | None = None,
    auto_mode: bool = True,
    frozen_for_review: bool = False,
) -> Eligibility:
    """Whether to nudge after this purchase.

    Ordered so the most important reason is the one reported: an operator
    having taken the conversation over outranks everything, and a customer who
    already replied outranks the nudge's own purpose.
    """
    if frozen_for_review:
        return Eligibility(False, "the conversation is on hold for a human")
    if not auto_mode:
        # The check that was missing entirely. Auto off means an operator has
        # the conversation, and money having just changed hands is the worst
        # moment to message over the top of them.
        return Eligibility(False, "auto mode is off for this customer")
    if he_replied_since(history, purchased_at):
        return Eligibility(False, "he already replied after buying")
    if asked_for_no_follow_up(open_threads):
        return Eligibility(False, "he asked not to be followed up")
    return Eligibility(True, "he bought and has not said anything since")
