"""
Step 1 of the prompt chain.
Analyzes the conversation situation before generating replies.
"""

import json

from ai.generator import client
from core.config import PRIMARY_MODEL
from models.schemas import ConversationContext


async def analyze_situation(ctx: ConversationContext) -> dict:
    """Analyze fan mood, intent and best strategic move."""

    recent = ctx.conversation_history[-10:]
    convo = "\n".join([
        f"{'Fan' if m.role == 'fan' else 'Creator'}: {m.content}"
        for m in recent
    ])

    response = await client.chat.completions.create(
        model=PRIMARY_MODEL,
        messages=[{
            "role": "user",
            "content": f"""You are analyzing an OnlyFans chat to help the creator respond perfectly.

Conversation so far:
{convo}

Latest fan message: "{ctx.fan_message}"

Analyze and return ONLY valid JSON:
{{
  "fan_mood": "one of: excited/bored/horny/lonely/curious/frustrated/romantic/testing/shy",
  "fan_intent": "what does this message signal — is he complimenting, escalating, testing, opening up, pulling back?",
  "conversation_energy": "rising/flat/dropping",
  "strategic_move": "what should the creator do RIGHT NOW — choose one: mirror_warmth/tease_and_deflect/ask_personal_question/hint_at_content/build_tension/re_engage/push_for_ppv/acknowledge_compliment_and_redirect",
  "tone": "what tone should the reply have — playful/warm/flirty/mysterious/direct/casual",
  "personal_details_mentioned": ["any names, locations, jobs, interests mentioned by fan"],
  "avoid_repeating": "flag if the creator has already used the same line recently"
}}"""
        }],
        temperature=0.2,
        max_tokens=200,
    )

    content = response.choices[0].message.content or ""
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
        }
