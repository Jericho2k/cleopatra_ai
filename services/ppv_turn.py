"""Deterministic PPV delivery for one turn.

WHY THIS EXISTS
---------------
Delivery used to be a string the writer had to serialise correctly:
``prompt_builder`` asked it to end its message with ``[PPV:media_id:price]``, and
``suggestions`` parsed that back out. When the model wrote "sending it now",
"here's the first piece", "here it is" and no tag — which it did — the fan got
three delivery claims and no media, and nothing in the pipeline noticed, because
as far as the sender was concerned this was an ordinary text turn.

That is a bad boundary. Which media, at which price, of which type, is
deterministic commercial state that the backend already knows before the writer
runs. So the backend decides it here, the writer is told only that it is
attached, and the sender attaches it. The LLM cannot cause a delivery and cannot
prevent one; the worst a bad generation can now do is produce bad copy.

THE ONE-MESSAGE RULE
--------------------
When a delivery is planned, the whole reply goes out as the PPV message's own
text, in one message. That is how the platform sends paid media — content and
attachment together — and it is what makes "the text cannot claim delivery if
delivery failed" true by construction rather than by careful ordering: there is
no separate text message to have already left.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from services.session_lifecycle import normalize_session


# Anything the writer emits that looks like the old delivery tag. It is stripped
# rather than honoured: the writer has no say in what is sent.
PPV_TAG_RE = re.compile(r"\[PPV:[^\]]*\]?", re.IGNORECASE)


@dataclass(frozen=True)
class PpvStepDelivery:
    """Exactly what this turn will attach, decided before the writer runs."""

    media_ids: list[str]
    media_id: str
    price_cents: int
    set_id: str | None
    step_index: int
    asset_type: str
    description: str

    @property
    def price(self) -> float:
        return round(self.price_cents / 100, 2)

    def writer_context(self) -> dict[str, Any]:
        """What the WRITER is told. No ids, no tag, no serialisation duty."""
        return {
            "attached": True,
            "asset_type": self.asset_type,
            "media_count": len(self.media_ids),
            "price_cents": self.price_cents,
            "description": self.description,
        }

    def media_context(self, *, payment_reference: str, source: str = "auto") -> dict:
        return {
            "ppv": {
                "media_ids": list(self.media_ids),
                "media_id": self.media_id,
                "price": self.price,
                "price_cents": self.price_cents,
                "access_type": "ppv",
                "set_id": self.set_id,
                "step_index": self.step_index,
                "payment_reference": payment_reference,
                "source": source,
            }
        }


def _decision_action(decision: Any) -> str:
    if decision is None:
        return ""
    if isinstance(decision, dict):
        return str(decision.get("action") or "")
    action = getattr(decision, "action", None)
    return str(getattr(action, "value", action) or "")


def plan_ppv_step_delivery(
    *,
    decision: Any,
    active_session: dict | None,
    legacy_send_now: bool = False,
) -> PpvStepDelivery | None:
    """The locked PPV this turn will send, or None.

    Every field comes from the persisted session plan, which is the only thing
    that ever authorised a price or a media id. Nothing is read from the writer,
    and nothing is inferred from the fan's message.

    ``legacy_send_now`` is the pre-commercial-engine path (COMMERCIAL_LAYER_ENABLED
    off). It used to be expressed as prompt text — "🚨 TIME TO SEND" plus a tag
    for the model to copy — so whether a fan received his media depended on a
    model reproducing an id correctly. The trigger is unchanged; only who acts on
    it moved, from the writer to here.
    """
    if _decision_action(decision) != "SEND_NEXT_PPV_STEP" and not legacy_send_now:
        return None

    session = normalize_session(active_session)
    if not session or session.get("status") != "active":
        return None
    if session.get("awaiting_purchase_index") is not None:
        # Something is already locked and unpaid. Never send a second.
        return None

    plan = session.get("plan") or []
    try:
        index = int(session.get("current_index", 0) or 0)
    except (TypeError, ValueError):
        return None
    if index < 0 or index >= len(plan):
        return None

    step = plan[index]
    if step.get("sent") or step.get("purchased"):
        return None

    media_ids = [str(value) for value in (step.get("media_ids") or []) if value]
    if not media_ids and step.get("media_id"):
        media_ids = [str(step["media_id"])]
    if not media_ids:
        return None

    price_cents = step.get("price_cents")
    if price_cents is None:
        try:
            price_cents = int(round(float(step.get("price") or 0) * 100))
        except (TypeError, ValueError):
            price_cents = 0
    price_cents = int(price_cents or 0)
    if price_cents <= 0:
        return None

    return PpvStepDelivery(
        media_ids=media_ids,
        media_id=media_ids[0],
        price_cents=price_cents,
        set_id=step.get("set_id"),
        step_index=index,
        asset_type=str(step.get("asset_type") or "photo_set"),
        description=str(step.get("description") or ""),
    )


def strip_ppv_tags(reply: str) -> tuple[str, bool]:
    """Remove any delivery tag the writer emitted. Returns ``(text, stripped)``.

    A writer that still writes one is not obeyed and is not punished: the tag is
    simply not a control surface any more. The boolean exists so the caller can
    say so in the log once, rather than silently accepting output that thinks it
    is driving delivery.
    """
    text = str(reply or "")
    if not PPV_TAG_RE.search(text):
        return text.strip(), False
    cleaned = PPV_TAG_RE.sub(" ", text)
    cleaned = re.sub(r"\s+\|", " |", cleaned)
    cleaned = " | ".join(
        part.strip() for part in cleaned.split("|") if part.strip()
    )
    return cleaned.strip(), True
