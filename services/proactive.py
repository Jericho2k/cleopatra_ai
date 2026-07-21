"""Send a proactive, goal-directed message in the creator's voice.

Used by the scheduled-actions worker (payday re-engagement today; unfinished
sessions and other lifecycle actions later).

The caller supplies a GOAL — a decided commercial action — and this module only
expresses it. The model does not get to decide whether to sell here.
"""
import asyncio

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


async def send_proactive_message(creator_id: str, fan_id: str, goal: str) -> bool:
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
        .select("apifansly_account_id")
        .eq("id", creator_id)
        .single()
        .execute()
    )
    apifansly_account_id = (creator_row.data or {}).get("apifansly_account_id")
    local_test_delivery = str(getattr(fan, "platform_fan_id", "") or "").startswith("test_")
    if (not group_id or not apifansly_account_id) and not local_test_delivery:
        print(f"[PROACTIVE SEND ERROR] fan={fan_id}: no live delivery route")
        return False

    try:
        if local_test_delivery:
            print(f"[PROACTIVE TEST DELIVERY] fan={fan_id} accepted=true")
        else:
            from main import send_fansly_message

            delivered = await send_fansly_message(apifansly_account_id, str(group_id), text)
            if not delivered:
                raise RuntimeError("platform rejected proactive delivery")
    except Exception as e:
        print(f"[PROACTIVE SEND ERROR] fan={fan_id}: {e}")
        return False

    # Persist only after the platform accepted delivery. If the platform send
    # succeeded but the local write failed, freeze automation rather than retrying
    # and risking a duplicate proactive message.
    try:
        await save_message(fan_id, creator_id, "creator", text)
    except Exception as exc:
        from db.queries import freeze_fan_for_review

        await freeze_fan_for_review(fan_id, "proactive_sent_but_not_persisted")
        print(f"[PROACTIVE PERSIST ERROR] fan={fan_id}: {exc}")
        return True

    print(f"[PROACTIVE] fan={fan_id} sent: {text[:60]}")
    return True
