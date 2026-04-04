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
            "content": f"""Analyze this OnlyFans chat conversation.

Conversation:
{convo}

Latest fan message: "{ctx.fan_message}"

Return ONLY valid JSON:
{{
  "fan_mood": "one word: excited/bored/horny/lonely/curious/frustrated/romantic/testing",
  "fan_intent": "what does this message actually want or signal in 1 sentence",
  "conversation_energy": "rising/flat/dropping",
  "strategic_move": "best move for the creator right now in 1 sentence — build connection/tease/upsell/re-engage/playful deflect/mirror energy",
  "personal_details_mentioned": ["any names, locations, jobs, interests the fan mentioned"]
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
            "strategic_move": "build connection with warmth and a question",
            "personal_details_mentioned": [],
        }
