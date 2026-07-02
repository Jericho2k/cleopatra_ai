"""
Step 1 of the prompt chain.
Analyzes the conversation situation before generating replies.
"""

import json
import os

from anthropic import AsyncAnthropic
from models.schemas import ConversationContext

client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


async def analyze_situation(ctx: ConversationContext) -> dict:
    """Analyze fan mood, intent and best strategic move."""

    recent = ctx.conversation_history[-10:]
    convo = "\n".join([
        f"{'Fan' if m.role == 'fan' else 'Creator'}: {m.content}"
        for m in recent
    ])

    user_content = f"""You are analyzing an OnlyFans chat to help the creator respond perfectly.

Conversation so far:
{convo}

Latest fan message: "{ctx.fan_message}"

Analyze and return ONLY valid JSON:
{{
  "fan_mood": "one of: excited/bored/horny/lonely/curious/frustrated/romantic/testing/shy",
  "fan_intent": "what does this message signal — is he complimenting, escalating, testing, opening up, pulling back?",
  "conversation_energy": "rising/flat/dropping",
  "strategic_move": "what should the creator do RIGHT NOW — choose one: mirror_warmth/tease_and_deflect/get_curious/hint_at_content/build_tension/re_engage/push_for_ppv/acknowledge_compliment_and_redirect. NEVER use tease_and_deflect or get_curious if the fan is asking about the creator's identity — use mirror_warmth instead.",
  "tone": "what tone should the reply have — playful/warm/flirty/mysterious/direct/casual",
  "personal_details_mentioned": ["any names, locations, jobs, interests mentioned by fan"],
  "avoid_repeating": "flag if the creator has already used the same line recently",
  "purchase_signal": "none | ready_to_buy | bought | declined | uncertain — none=no purchase context. ready_to_buy=fan just said yes/I want it/send it after a price was mentioned but no PPV sent yet. bought=positive reaction after PPV was already sent. declined=mentions price/can't afford/maybe later. uncertain=unclear",
  "resend_requested": "true | false — did the fan indicate they cannot see content that was sent, it did not arrive, or they are asking for it to be sent again? Look at the full conversation context, not just the latest message.",
  "crisis_signal": "none | self_harm | harm_to_others — DEFAULT none, but err toward flagging self_harm when the language plausibly references self-harm even if it might be hyperbole. This is NOT about sexual roughness or kink: choking, rough, degradation, being held by the throat, or any violent/rough fantasy DIRECTED AT THE CREATOR as sexual roleplay is always none. Flag self_harm if the fan references hurting himself, cutting, bleeding, his veins/wrists, ending his life, not wanting to be alive, or similar self-directed harm — including phrasing like 'I'd cut my veins', 'I'd die for you', 'I can't go on' — because we respond with care rather than gambling that it was a joke. Flag harm_to_others only if the fan states real intent to harm a specific real person (not sexual fantasy about the creator). When a message mixes flirtation with self-harm phrasing, still flag self_harm."
}}"""

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": user_content}],
    )

    content = response.content[0].text
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(content)
    except Exception:
        result = {
            "fan_mood": "curious",
            "fan_intent": "engaging with creator",
            "conversation_energy": "flat",
            "strategic_move": "mirror_warmth",
            "tone": "playful",
            "personal_details_mentioned": [],
            "avoid_repeating": "",
            "purchase_signal": "none",
            "resend_requested": "false",
            "crisis_signal": "none",
        }

    # Deterministic safety backstop: the crisis rail must not depend solely on one
    # probabilistic classification. If the latest message contains unambiguous
    # self-harm phrasing, force the flag even if the classifier rationalized it as
    # flirtation. Kept intentionally narrow to avoid catching kink/roleplay.
    if _looks_like_self_harm(ctx.fan_message):
        result["crisis_signal"] = "self_harm"

    return result


# Unambiguous self-harm phrases. Deliberately conservative: these read as genuine
# self-harm regardless of surrounding flirtation, and are not kink/roleplay terms.
_SELF_HARM_PATTERNS = (
    "cut my vein", "cut my wrist", "slit my wrist", "slit my vein",
    "kill myself", "end my life", "end it all", "want to die",
    "don't want to live", "dont want to live", "don't want to be alive",
    "dont want to be alive", "not worth living", "harm myself", "hurt myself",
    "bleed out", "overdose", "take my own life", "suicidal", "suicide",
)


def _looks_like_self_harm(message: str) -> bool:
    text = (message or "").lower()
    return any(p in text for p in _SELF_HARM_PATTERNS)