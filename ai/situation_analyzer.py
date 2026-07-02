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
  "crisis_signal": "none | self_harm | harm_to_others — DEFAULT none. This is NOT about sexual roughness or kink: choking, rough, degradation, or violent fantasy DIRECTED AT THE CREATOR as sexual roleplay is none. Only flag self_harm if the fan expresses genuine intent to hurt or kill himself or that he doesn't want to be alive. Only flag harm_to_others if the fan states real intent to harm a specific real person (not sexual fantasy about the creator). When unsure, use none."
}}"""

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": user_content}],
    )

    content = response.content[0].text
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(content)
    except Exception:
        return {
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