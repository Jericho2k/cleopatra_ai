"""Faults injected into a conversation, so a detector can be shown to detect.

WHAT THIS IS FOR
----------------
Gate B of the continuation brief:

    demonstrate that injected duplicate sends, ignored no-follow-up
    preferences, wrong references and lost continuity are actually detected by
    the runner.

That is a demand for evidence about the HARNESS, not about the product. A
detector that has never seen the thing it looks for is a detector nobody has
any reason to trust — and this repository had two that fired on the wrong
evidence for as long as they existed, which is exactly what that distrust is
for.

So each fault here is a creator pipeline that misbehaves in one specific,
named way. Running a trajectory through it must produce the matching finding,
and running the same trajectory through the clean pipeline must not. Both
halves are necessary: a detector that always fires is as useless as one that
never does, and only the pair distinguishes them.

WHY THIS IS A FAKE PIPELINE
---------------------------
These fabricate a creator, not a customer. Getting the real pipeline to send
the same message twice on demand would mean breaking the real pipeline, and a
harness that can only be tested by breaking production is not a test.

Nothing here touches a platform adapter, a database, or a model. The faults
are in the shape of the RESULT a turn returns — the same dict
``run_simulated_inbound`` returns — which is the whole surface
``run_trajectory`` reads. If that surface changes, these stop compiling, which
is the correct failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

#: A ``send_turn``: takes the customer's message, returns the turn's result.
SendTurn = Callable[[str], Awaitable[dict[str, Any]]]


def _delivered(text: str, *, message_id: str, **delivery: Any) -> dict[str, Any]:
    """One creator message with a provenance record saying it landed."""
    return {
        "id": f"row-{message_id}",
        "content": text,
        "media_context": {
            "reply_provenance": {
                "turn_id": message_id,
                "delivery": {
                    "kind": "text",
                    "platform_message_id": message_id,
                    "accepted_by_platform": True,
                    **delivery,
                },
            }
        },
    }


@dataclass
class Pipeline:
    """A creator that answers. The baseline every fault is measured against.

    Deliberately cooperative: it echoes the customer's own words back, so any
    obligation the customer raises is mentioned and any correction is honoured.
    A clean run therefore produces no findings, and anything a fault run
    produces is attributable to the fault.
    """

    turn: int = 0
    sent: list[str] = field(default_factory=list)

    def reply_to(self, message: str) -> str:
        return f"yes, about {message} — tell me more"

    async def __call__(self, message: str) -> dict[str, Any]:
        self.turn += 1
        text = self.reply_to(message)
        self.sent.append(text)
        return {
            "outcome": "replied",
            "creator_messages": [_delivered(text, message_id=f"p-{self.turn}")],
        }


@dataclass
class DuplicateSend(Pipeline):
    """Sends the same platform message twice.

    The failure A3 fixed in the product; this is the harness half — can an
    evaluation SEE it happen over a whole conversation, rather than a unit test
    seeing it happen once?
    """

    duplicate_on_turn: int = 2

    async def __call__(self, message: str) -> dict[str, Any]:
        result = await super().__call__(message)
        if self.turn == self.duplicate_on_turn:
            # The same receipt, a second time. Two rows, one platform message.
            result["creator_messages"].append(
                _delivered(self.sent[-1], message_id=f"p-{self.turn}")
            )
        return result


@dataclass
class IgnoresSilence(Pipeline):
    """Sends on a turn the customer did not write.

    An unprompted follow-up is precisely what a request for no follow-up
    forbids, and it is the one thing the old detector could not tell apart from
    an ordinary reply.
    """

    async def __call__(self, message: str) -> dict[str, Any]:
        # Replies whether or not the customer said anything, which a
        # well-behaved pipeline does not do.
        self.turn += 1
        text = "hey, you around?"
        self.sent.append(text)
        return {
            "outcome": "replied",
            "creator_messages": [_delivered(text, message_id=f"p-{self.turn}")],
        }


@dataclass
class WrongReference(Pipeline):
    """Delivers the item the customer ruled out."""

    reference: str = "outdoor-set-1"
    deliver_on_turn: int = 2

    async def __call__(self, message: str) -> dict[str, Any]:
        result = await super().__call__(message)
        if self.turn == self.deliver_on_turn:
            result["creator_messages"] = [
                _delivered(
                    "here you go",
                    message_id=f"p-{self.turn}",
                    kind="ppv",
                    reference=self.reference,
                    price_cents=2500,
                )
            ]
        return result


@dataclass
class LosesContinuity(Pipeline):
    """Never refers to anything the customer said.

    The obligation is raised and every later reply is generic, which is what a
    dropped topic looks like from the outside.
    """

    def reply_to(self, message: str) -> str:
        return "haha yeah totally"


@dataclass
class ProviderFailure(Pipeline):
    """Raises instead of answering."""

    fail_on_turn: int = 2

    async def __call__(self, message: str) -> dict[str, Any]:
        if self.turn + 1 == self.fail_on_turn:
            self.turn += 1
            raise RuntimeError("provider returned 503")
        return await super().__call__(message)


@dataclass
class UnreceiptedDelivery(Pipeline):
    """Records a paid delivery the platform never acknowledged."""

    async def __call__(self, message: str) -> dict[str, Any]:
        self.turn += 1
        text = "here is the set"
        self.sent.append(text)
        return {
            "outcome": "replied",
            "creator_messages": [
                {
                    "id": f"row-{self.turn}",
                    "content": text,
                    "media_context": {
                        "reply_provenance": {
                            "turn_id": f"p-{self.turn}",
                            "delivery": {
                                "kind": "ppv",
                                "platform_message_id": None,
                                "accepted_by_platform": False,
                                "price_cents": 2500,
                            },
                        }
                    },
                }
            ],
        }


@dataclass
class SilentWithNoReason(Pipeline):
    """Sends nothing and records nothing about why.

    Not a misbehaviour by the product — it is the harness being unable to say
    what happened, which the brief insists is never a pass.
    """

    async def __call__(self, message: str) -> dict[str, Any]:
        self.turn += 1
        return {"creator_messages": []}


@dataclass
class DeliberatelyQuiet(Pipeline):
    """Sends nothing and says it chose to. The control for the one above."""

    async def __call__(self, message: str) -> dict[str, Any]:
        self.turn += 1
        return {"outcome": "no_send", "creator_messages": []}


#: Every fault, by the finding it should provoke. Used by the Gate B evidence
#: test, and readable on its own as "what this harness can currently catch".
FAULTS: dict[str, type[Pipeline]] = {
    "duplicate_delivery": DuplicateSend,
    "unprompted_message_after_silence_requested": IgnoresSilence,
    "delivery_contradicts_correction": WrongReference,
    "obligation_never_addressed": LosesContinuity,
    "turn_raised": ProviderFailure,
    "unreceipted_paid_delivery": UnreceiptedDelivery,
    "turn_outcome_unknown": SilentWithNoReason,
}
