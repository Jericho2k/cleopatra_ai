"""Send a proactive, goal-directed message in the creator's voice.

Used by the scheduled-actions worker (payday re-engagement today; unfinished
sessions and other lifecycle actions later).

The caller supplies a GOAL — a decided commercial action — and this module only
expresses it. The model does not get to decide whether to sell here.
"""
import asyncio
from datetime import datetime, timezone
from typing import Any

from ai.generator import generate_replies
from ai.prompt_builder import build_prompt
from db.queries import (
    get_conversation_history,
    get_creator_persona,
    get_fan_by_id,
    get_fan_session,
    get_sent_ppv,
    save_message,
)
from models.schemas import ConversationContext, Persona, StageType
from core.supabase import get_supabase
from db.commercial_queries import update_action_payload
from services.apifansly import list_chat_messages


def _platform_time(value: object) -> datetime | None:
    try:
        timestamp = float(value or 0)
    except (TypeError, ValueError):
        return None
    if timestamp <= 0:
        return None
    if timestamp > 1e12:
        timestamp /= 1000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


async def _reconcile_ambiguous_delivery(
    *,
    account_id: str,
    group_id: str,
    creator_platform_id: str,
    text: str,
    started_at: datetime,
) -> str | None:
    """Find a message accepted before a worker crash without resending it."""
    normalized = " ".join(text.split())
    cursor: str | None = None
    for _ in range(5):
        messages, _, cursor = await list_chat_messages(
            account_id,
            group_id,
            cursor=cursor,
            limit=10,
        )
        oldest: datetime | None = None
        for message in messages:
            created_at = _platform_time(message.get("createdAt"))
            if created_at and (oldest is None or created_at < oldest):
                oldest = created_at
            sender_id = str(message.get("senderId") or "")
            if creator_platform_id and sender_id != creator_platform_id:
                continue
            if " ".join(str(message.get("content") or "").split()) != normalized:
                continue
            if created_at and created_at < started_at:
                continue
            message_id = str(message.get("id") or "").strip()
            if message_id:
                return message_id
        if not cursor or (oldest and oldest < started_at):
            break
    return None


async def send_proactive_message(
    creator_id: str,
    fan_id: str,
    goal: str,
    *,
    action_id: str | None = None,
    action_payload: dict[str, Any] | None = None,
) -> bool:
    """Generate one message toward `goal` and deliver it. Returns True if sent."""
    fan = await get_fan_by_id(fan_id)
    if not fan:
        return False

    persona = await get_creator_persona(creator_id) or Persona()
    history = await get_conversation_history(fan_id, limit=20)

    try:
        from db.queries import get_creator_legend
        legend = await get_creator_legend(creator_id)
    except Exception:
        legend = {}

    ctx = ConversationContext(
        fan_message="",          # proactive: he hasn't just said anything
        conversation_history=history,
        fan_profile=fan,
        creator_persona=persona,
        similar_exchanges=[],
        conversation_stage=StageType.RE_ENGAGEMENT
        if hasattr(StageType, "RE_ENGAGEMENT") else StageType.WARMING_UP,
        creator_name="",
        ppv_offers=[],
        sent_ppv=await get_sent_ppv(fan_id),
        active_session=await get_fan_session(fan_id),
        creator_legend=legend,
        situation={
            "fan_mood": "unknown",
            "strategic_move": "re_engage",
            "tone": "playful",
            "purchase_signal": "none",
            "crisis_signal": "none",
        },
    )

    payload = dict(action_payload or {})
    delivery = dict(payload.get("_delivery") or {})
    text = str(delivery.get("text") or "").strip()
    if not text:
        prompt = build_prompt(ctx)
        # Append the decided goal to the user turn. This is the ONLY thing the model
        # is meant to accomplish — it does not choose to sell or send media.
        prompt[1]["content"] += (
            f"\n\nPROACTIVE MESSAGE — YOUR GOAL FOR THIS MESSAGE:\n{goal}\n"
            "Write ONE short message (two at most). Do NOT include any [PPV:...] tag. "
            "Do not mention a price. This must feel like you thought of him, not like a "
            "scheduled campaign."
        )

        replies = await generate_replies(prompt, persona)
        if not replies:
            print(f"[PROACTIVE] generation failed for fan={fan_id} — sending nothing")
            return False
        text = replies[0].strip()
        if not text:
            return False

    # Strip any PPV tag the model may have emitted anyway — proactive messages
    # never carry media.
    import re
    text = re.sub(r"\[PPV:[^\]]*\]", "", text).strip()
    if not text:
        return False

    group_id = getattr(fan, "fansly_group_id", None)
    creator_row = await asyncio.to_thread(
        lambda: get_supabase().table("creators")
        .select("apifansly_account_id, fansly_account_id")
        .eq("id", creator_id)
        .single()
        .execute()
    )
    apifansly_account_id = (creator_row.data or {}).get("apifansly_account_id")
    creator_platform_id = str((creator_row.data or {}).get("fansly_account_id") or "")
    local_test_delivery = str(getattr(fan, "platform_fan_id", "") or "").startswith("test_")
    if (not group_id or not apifansly_account_id) and not local_test_delivery:
        print(f"[PROACTIVE SEND ERROR] fan={fan_id}: no live delivery route")
        return False

    confirmed_message_id = str(delivery.get("platform_message_id") or "").strip()
    started_at_raw = delivery.get("started_at")
    started_at = None
    if started_at_raw:
        try:
            started_at = datetime.fromisoformat(
                str(started_at_raw).replace("Z", "+00:00")
            ).astimezone(timezone.utc)
        except (TypeError, ValueError):
            started_at = None

    if action_id and not local_test_delivery and not confirmed_message_id and started_at:
        try:
            confirmed_message_id = await _reconcile_ambiguous_delivery(
                account_id=str(apifansly_account_id),
                group_id=str(group_id),
                creator_platform_id=creator_platform_id,
                text=text,
                started_at=started_at,
            ) or ""
            if confirmed_message_id:
                print(
                    f"[PROACTIVE RECONCILED] fan={fan_id} "
                    f"message={confirmed_message_id}"
                )
        except Exception as exc:
            # A failed read is not proof that the original send failed. Let the
            # worker retry later rather than risk an immediate duplicate.
            print(f"[PROACTIVE RECONCILE ERROR] fan={fan_id}: {exc}")
            return False

    if action_id and not confirmed_message_id:
        delivery.update({
            "text": text,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "attempts": int(delivery.get("attempts") or 0) + 1,
        })
        payload["_delivery"] = delivery
        await update_action_payload(action_id, payload)

    platform_message_id = confirmed_message_id
    try:
        if local_test_delivery:
            platform_message_id = f"local-test:{action_id or fan_id}"
            print(f"[PROACTIVE TEST DELIVERY] fan={fan_id} accepted=true")
        elif not platform_message_id:
            from main import send_fansly_message

            platform_message_id = await send_fansly_message(
                str(apifansly_account_id), str(group_id), text
            )
            if not platform_message_id:
                raise RuntimeError("platform rejected proactive delivery")
    except Exception as e:
        print(f"[PROACTIVE SEND ERROR] fan={fan_id}: {e}")
        return False

    journal_error: Exception | None = None
    if action_id:
        try:
            delivery.update({
                "text": text,
                "platform_message_id": platform_message_id,
                "confirmed_at": datetime.now(timezone.utc).isoformat(),
            })
            payload["_delivery"] = delivery
            await update_action_payload(action_id, payload)
        except Exception as exc:
            journal_error = exc

    # Persist only after the platform accepted delivery. The platform ID makes
    # this write idempotent when a stale worker reconciles an earlier acceptance.
    try:
        await save_message(
            fan_id,
            creator_id,
            "creator",
            text,
            was_ai_suggested=True,
            fansly_message_id=platform_message_id,
        )
    except Exception as exc:
        from db.queries import freeze_fan_for_review

        await freeze_fan_for_review(fan_id, "proactive_sent_but_not_persisted")
        print(f"[PROACTIVE PERSIST ERROR] fan={fan_id}: {exc}")
        return True

    if journal_error is not None:
        from db.queries import freeze_fan_for_review

        await freeze_fan_for_review(fan_id, "proactive_sent_but_journal_not_persisted")
        print(f"[PROACTIVE JOURNAL ERROR] fan={fan_id}: {journal_error}")
        return True

    print(f"[PROACTIVE] fan={fan_id} sent: {text[:60]}")
    return True
