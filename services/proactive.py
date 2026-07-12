"""Send a proactive, goal-directed message in the creator's voice.

Used by the scheduled-actions worker (payday re-engagement today; unfinished
sessions and other lifecycle actions later).

The caller supplies a GOAL — a decided commercial action — and this module only
expresses it. The model does not get to decide whether to sell here.
"""
import os

import httpx

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

    await save_message(fan_id, creator_id, "creator", text)

    group_id = getattr(fan, "fansly_group_id", None)
    apifansly_account_id = os.environ.get("APIFANSLY_ACCOUNT_ID")
    if group_id and apifansly_account_id:
        try:
            from main import send_fansly_message
            await send_fansly_message(apifansly_account_id, str(group_id), text)
        except Exception as e:
            print(f"[PROACTIVE SEND ERROR] fan={fan_id}: {e}")
            return False

    print(f"[PROACTIVE] fan={fan_id} sent: {text[:60]}")
    return True