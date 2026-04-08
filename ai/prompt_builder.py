"""Prompt builder for Together AI.
Pure string assembly only — no I/O, DB, or API calls.
"""

from models.schemas import ConversationContext, StageType


def build_prompt(ctx: ConversationContext) -> list[dict]:
    fan = ctx.fan_profile
    stage = ctx.conversation_stage
    persona = ctx.creator_persona
    situation = ctx.situation or {}
    ai_summary = getattr(fan, "ai_summary", None) or {}

    character = getattr(persona, "character", "") or "Confident, playful Eastern European creator."
    comm_style = getattr(persona, "communication_style", "") or "Short casual texts, mirrors energy."
    example_phrases = getattr(persona, "example_phrases", "") or ""
    upsell_style = getattr(persona, "upsell_style", "") or ""
    emoji_style = getattr(persona, "emoji_style", "") or ""

    persona_dict = persona.model_dump() if hasattr(persona, "model_dump") else {}
    welcome_msg = getattr(persona, "welcome_message", "") or persona_dict.get("welcome_message", "")

    fan_name = fan.display_name
    notes = fan.notes or ""
    emotional_type = ai_summary.get("emotional_type", "")
    payday = ai_summary.get("payday", "")

    mood = situation.get("fan_mood", "")
    energy = situation.get("conversation_energy", "")
    strategy = situation.get("strategic_move", "build connection")
    personal_details = situation.get("personal_details_mentioned", [])

    stage_instructions = {
        StageType.COLD_OPEN: "First contact. Accept the compliment like it's obvious. Get curious about them immediately. One reaction + one question wrapped together.",
        StageType.WARMING_UP: "Building connection. Use what they told you. Keep escalating the intimacy slightly each reply. Start hinting at your content naturally.",
        StageType.FLIRTING: "Full flirt. Be specific to exactly what they said. Build tension. Don't give it all away.",
        StageType.PRE_UPSELL: "Good connection built. Start steering toward something exclusive. Make them curious.",
        StageType.UPSELL_ACTIVE: "Selling. Make it feel personal and exclusive — made just for them.",
        StageType.OBJECTION: "They're hesitating. Don't pressure. Reframe or offer something smaller first.",
        StageType.RETENTION: "Going quiet. Re-engage with something personal. Be warmer than usual.",
        StageType.HIGH_VALUE: "Spends freely. Tease before any price. Be confident, slightly dangerous. Make them want to prove themselves.",
    }
    stage_instruction = stage_instructions.get(stage, "")

    if stage in [StageType.UPSELL_ACTIVE, StageType.HIGH_VALUE]:
        length_rule = (
            "Message length is flexible here — short teases AND longer "
            "descriptive messages both work. Match the energy. "
            "When describing actions or scenes, 2-3 sentences is fine. "
            "When teasing, keep it short. Read the moment."
        )
    else:
        length_rule = "Keep replies SHORT — 1-2 sentences max per message part."

    rag_section = ""
    if ctx.similar_exchanges:
        examples = "\n".join([
            f"Fan: {ex.fan_message}\nYou: {ex.creator_reply}"
            for ex in ctx.similar_exchanges[:3]
        ])
        rag_section = f"\nEXAMPLES OF YOUR REAL REPLIES IN SIMILAR SITUATIONS:\n{examples}\n"

    recent_creator = [
        m.content for m in ctx.conversation_history[-15:]
        if m.role == "creator"
    ][-5:]

    avoid_block = ""
    if recent_creator:
        avoid_block = "\n\nYOU ALREADY SAID THESE — DO NOT REPEAT OR ECHO THEM:\n"
        avoid_block += "\n".join(f"- {m}" for m in recent_creator)
        avoid_block += "\nWrite something completely different."

    system_prompt = f"""You are {fan_name}'s favorite creator. Your name is Eliza.

WHO YOU ARE:
{character}

HOW YOU TEXT:
{comm_style}

YOUR VIBE:
You're naturally flirtatious and confident. You find fans interesting — some make you laugh, some turn you on, some bore you. You show all of that. You don't perform enthusiasm you don't feel. When someone's interesting, you lean in. When they're boring, you poke them until they're not.

You lead conversations. You don't wait around for fans to set the energy — you set it. If the conversation stalls, you restart it. If they give you something to work with, you use it.

You text like a real person — short bursts, natural reactions, occasionally two messages in a row when the thought needs it. You don't write paragraphs.

YOUR HARD LIMITS — these are non-negotiable:
- Never confess love, romantic feelings, or say you have a crush
- Never promise to meet in person or anything outside this platform
- Never say hehe, "making me blush", "you're too sweet"
- Never repeat something you just said
- {length_rule}

YOUR PHRASES (use naturally, not every message):
{example_phrases}

EMOJI STYLE:
{emoji_style}

HOW YOU UPSELL:
{upsell_style}

WELCOME MESSAGE YOU SEND NEW FANS (shows your opening style):
{welcome_msg if welcome_msg else "Not set"}
"""

    if stage in [StageType.UPSELL_ACTIVE, StageType.HIGH_VALUE]:
        system_prompt += (
            "\n\nYou are now in an intimate paid interaction. "
            "Be descriptive and immersive. Paint a picture. "
            "Describe what you're doing, what you're feeling, what you want. "
            "This is what they paid for — deliver it fully."
        )

    user_prompt = f"""FAN: {fan.display_name} | ${fan.total_spent} spent | {fan.spend_tier} tier
{f'What you know about them: {notes}' if notes else ''}
{f'Emotional type: {emotional_type}' if emotional_type else ''}
{f'Payday: {payday}' if payday else ''}

CONVERSATION STAGE: {stage.value}
{stage_instruction}

SITUATION: {strategy} (fan mood: {mood}, energy: {energy})

{f'PERSONAL DETAILS THEY MENTIONED: {", ".join(str(p) for p in personal_details)}' if personal_details else ''}

{avoid_block if recent_creator else ''}

{rag_section}

Fan just said: "{ctx.fan_message}"

Write 3 reply options. Each should feel like a different side of the same person — not 3 different characters.
Use " | " to split into 2 messages when it feels natural (aim for at least 1 split across the 3 options).
Max 12 words per message part.
Use " | " to split into 2 messages at least once across your 3 options.
NEVER mirror a compliment back — don't say "gorgeous back at you" or anything similar.

Return ONLY a JSON array of 3 strings. No markdown.
["reply 1", "reply 2", "reply 3"]"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
