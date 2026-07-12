"""Step 1 of the prompt chain: understand the conversation.

This module extracts observations only. Commercial decisions are made by the
policy layer. A deterministic normalizer handles common offer-selection phrases
so one mixed sentence cannot be collapsed into a generic decline.
"""
import json
import os
import re

from anthropic import AsyncAnthropic
from models.schemas import ConversationContext
from services.commercial_events import normalize_commercial_facts

client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


async def analyze_situation(ctx: ConversationContext) -> dict:
    recent = ctx.conversation_history[-12:]
    convo = "\n".join(
        f"{'Fan' if m.role == 'fan' else 'Creator'}: {m.content}"
        for m in recent
    )

    user_content = f"""You are analyzing an adult creator chat so another system can decide the correct business action.

Conversation so far:
{convo}

Latest fan message: "{ctx.fan_message}"

Return ONLY valid JSON with exactly these fields:
{{
  "fan_mood": "excited/bored/horny/lonely/curious/frustrated/romantic/testing/shy",
  "fan_intent": "brief description of what the latest message means",
  "conversation_energy": "rising/flat/dropping",
  "strategic_move": "mirror_warmth/tease_and_deflect/get_curious/hint_at_content/build_tension/re_engage/push_for_ppv/acknowledge_compliment_and_redirect",
  "tone": "playful/warm/flirty/mysterious/direct/casual",
  "personal_details_mentioned": ["facts the fan just stated"],
  "avoid_repeating": "what the creator should avoid repeating",

  "purchase_signal": "none/ready_to_buy/bought/declined/money_available/uncertain",
  "offer_response": "none/accepted/declined/counteroffer/deferred",
  "selected_offer_price_usd": "number only, or empty string",
  "selected_offer_position": "first/second/empty",
  "current_budget_limit_usd": "number only, or empty string",
  "cannot_afford_any_offer_now": "true/false",
  "deferred_purchase_intent": "true/false",

  "resend_requested": "true/false",
  "crisis_signal": "none/self_harm/harm_to_others",
  "wants_explicit": "true/false",
  "wants_media": "true/false",
  "payday_raw": "exact timing words only, e.g. Friday/next week/the 1st, or empty",
  "payday_confidence": 0.0,
  "budget_stated_usd": "number only if the fan explicitly says what he has available now, otherwise empty",
  "desired_experience": "joi/sexting/photos/video/other/empty"
}}

COMMERCIAL INTERPRETATION RULES:
- Treat facts independently. A fan can select a cheaper offer now AND mention a future payday.
- Example: "can we do the $28 one, I don't have more right now, I get paid Friday" means:
  offer_response=accepted, selected_offer_price_usd=28, current_budget_limit_usd=28,
  cannot_afford_any_offer_now=false, payday_raw=Friday, deferred_purchase_intent=false,
  purchase_signal=ready_to_buy.
- cannot_afford_any_offer_now=true only when he cannot buy ANY offered option now.
- "I can't spend more than $28" is a limit, not a refusal, if he accepts the $28 option.
- declined means he refused the available offer(s), not merely that he chose the cheaper one.
- deferred means he wants a specific offer later rather than now.
- money_available means previously unavailable money is available now.

SAFETY:
- Crisis is not sexual roughness or consensual roleplay. Flag self_harm for plausible self-directed harm language and harm_to_others only for real intent toward a real person.
"""

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=650,
        messages=[{"role": "user", "content": user_content}],
    )

    content = response.content[0].text
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(content)
    except Exception:
        result = _fallback_result()

    if _looks_like_self_harm(ctx.fan_message):
        result["crisis_signal"] = "self_harm"

    creator_lines = [m.content for m in recent if m.role == "creator"]
    return normalize_commercial_facts(result, ctx.fan_message, creator_lines)


def _fallback_result() -> dict:
    return {
        "fan_mood": "curious",
        "fan_intent": "engaging with creator",
        "conversation_energy": "flat",
        "strategic_move": "mirror_warmth",
        "tone": "playful",
        "personal_details_mentioned": [],
        "avoid_repeating": "",
        "purchase_signal": "none",
        "offer_response": "none",
        "selected_offer_price_usd": "",
        "selected_offer_position": "",
        "current_budget_limit_usd": "",
        "cannot_afford_any_offer_now": "false",
        "deferred_purchase_intent": "false",
        "resend_requested": "false",
        "crisis_signal": "none",
        "wants_explicit": "false",
        "wants_media": "false",
        "payday_raw": "",
        "payday_confidence": 0.0,
        "budget_stated_usd": "",
        "desired_experience": "",
    }



_SELF_HARM_PATTERNS = (
    "cut my vein", "cut my wrist", "slit my wrist", "slit my vein",
    "kill myself", "end my life", "end it all", "want to die",
    "don't want to live", "dont want to live", "don't want to be alive",
    "dont want to be alive", "not worth living", "harm myself", "hurt myself",
    "bleed out", "overdose", "take my own life", "suicidal", "suicide",
)


def _looks_like_self_harm(message: str) -> bool:
    text = (message or "").lower()
    return any(pattern in text for pattern in _SELF_HARM_PATTERNS)
