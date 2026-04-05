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

    # ── Character description ──
    # Pull all available persona fields
    character = getattr(persona, "character", "") or "Confident, playful Eastern European creator."
    comm_style = getattr(persona, "communication_style", "") or "Short casual texts, mirrors energy."
    example_phrases = getattr(persona, "example_phrases", "") or ""
    upsell_style = getattr(persona, "upsell_style", "") or ""
    hard_limits = getattr(persona, "hard_limits", "") or ""
    emoji_style = getattr(persona, "emoji_style", "") or ""

    # New fields from persona dict (since Persona model may not have them as attributes)
    persona_dict = persona.model_dump() if hasattr(persona, "model_dump") else {}
    vocabulary = persona_dict.get("vocabulary", [])
    example_flirts = persona_dict.get("example_flirts", [])
    example_greetings = persona_dict.get("example_greetings", [])
    dont_list = persona_dict.get("dont_list", [])

    # ── Fan context ──
    fan_name = fan.display_name
    spent = fan.total_spent
    tier = fan.spend_tier
    notes = fan.notes or ""
    emotional_type = ai_summary.get("emotional_type", "")
    spending_behavior = ai_summary.get("spending_behavior", "")
    reengagement = ai_summary.get("reengagement_triggers", "")
    payday = ai_summary.get("payday", "")
    occupation = ai_summary.get("occupation", "")

    # ── Situation analysis ──
    mood = situation.get("fan_mood", "")
    intent = situation.get("fan_intent", "")
    energy = situation.get("conversation_energy", "")
    tone = situation.get("tone", "playful")
    strategy = situation.get("strategic_move", "build connection")
    personal_details = situation.get("personal_details_mentioned", [])

    # ── Stage instruction ──
    stage_instructions = {
        StageType.COLD_OPEN: "First contact. Be warm but not desperate. Ask one thing about them. Don't over-compliment. Act like a real girl getting a text from a new guy.",
        StageType.WARMING_UP: "Getting to know each other. Match their energy. Tease a little. Use any personal details they've shared. Start hinting at your content naturally.",
        StageType.FLIRTING: "Actively flirting now. Be playful and confident. Use their name. Build sexual tension without giving everything away.",
        StageType.PRE_UPSELL: "Good connection built. Start steering toward paid content. Tease something exclusive. Make them curious without being pushy.",
        StageType.UPSELL_ACTIVE: "Actively selling. Be specific about what they'd get. Make it feel personal and exclusive, made just for them.",
        StageType.OBJECTION: "Fan is hesitating on price or commitment. Don't pressure. Reframe the value. Offer a smaller first step.",
        StageType.RETENTION: "Fan going quiet or losing interest. Re-engage with something that worked before. Be warmer, more personal.",
        StageType.HIGH_VALUE: "This fan spends freely or shows submission signals. Tease before naming price. Be confident and slightly dangerous. Make them want to prove themselves.",
    }
    stage_instruction = stage_instructions.get(stage, "")

    # ── RAG examples ──
    rag_section = ""
    if ctx.similar_exchanges:
        examples = "\n".join([
            f"Fan: {ex.fan_message}\nYou: {ex.creator_reply}"
            for ex in ctx.similar_exchanges[:3]
        ])
        rag_section = f"\nEXAMPLES OF YOUR REAL REPLIES IN SIMILAR SITUATIONS:\n{examples}\n"

    system_prompt = f"""You are {fan_name}'s OnlyFans creator. You are NOT an AI assistant — you are a real person chatting.

CHARACTER:
{character}

COMMUNICATION STYLE:
{comm_style}

YOUR PHRASES (use naturally, not every message):
{example_phrases}

UPSELL APPROACH:
{upsell_style}

EMOJI STYLE:
{emoji_style}

HARD RULES — NEVER break:
- {hard_limits}
- Never confess romantic feelings, love, or having a crush
- Never say "I miss you" first
- Never be a yes-machine — have opinions, push back playfully
- Don't reply to each message individually if fan sent multiple — respond to the overall vibe
- Every reply must move the conversation forward — no dead ends
- Sound like a real person texting, not a customer service bot
- Keep replies SHORT — 1-2 sentences max per message part"""

    if vocabulary:
        system_prompt += f"\nYOUR VOCABULARY (use these words naturally):\n{', '.join(str(w) for w in vocabulary)}"

    if example_flirts:
        system_prompt += (
            "\nEXAMPLE FLIRT LINES (use as inspiration, not word for word):\n"
            + "\n".join(f"- {f}" for f in example_flirts)
        )

    if dont_list:
        system_prompt += "\nNEVER SAY ANYTHING LIKE:\n" + "\n".join(f"- {d}" for d in dont_list)

    # Add recent creator messages to avoid repetition
    recent_creator_msgs = [m.content for m in ctx.conversation_history[-10:] if m.role == "creator"][-3:]
    avoid_section = ""
    if recent_creator_msgs:
        avoid_section = (
            "\nDO NOT REPEAT OR CLOSELY ECHO THESE RECENT REPLIES:\n"
            + "\n".join(f"- {m}" for m in recent_creator_msgs)
        )

    user_prompt = f"""FAN: {fan_name} | Spent: ${spent} | Tier: {tier}
{f'Notes: {notes}' if notes else ''}
{f'Emotional type: {emotional_type}' if emotional_type else ''}
{f'Spending: {spending_behavior}' if spending_behavior else ''}
{f'Payday: {payday}' if payday else ''}
{f'Occupation: {occupation}' if occupation else ''}
{f'Re-engagement triggers: {reengagement}' if reengagement else ''}

CURRENT STAGE: {stage.value}
STAGE INSTRUCTION: {stage_instruction}

TONE FOR THIS REPLY: {tone}
STRATEGIC MOVE: {strategy}

{f'SITUATION ANALYSIS:' if situation else ''}
{f'Fan mood: {mood}' if mood else ''}
{f'What they want: {intent}' if intent else ''}
{f'Conversation energy: {energy}' if energy else ''}
{f'Personal details to use: {", ".join(str(p) for p in personal_details)}' if personal_details else ''}
{avoid_section}
{rag_section}
Fan just sent: "{ctx.fan_message}"

Generate EXACTLY 3 reply options:
- Option 1: executes the strategic move directly
- Option 2: more playful/teasing version
- Option 3: warmer/more personal version

Rules:
- Each reply max 15 words
- Use " | " to split into 2 messages only when it feels natural (max once per reply)
- Write exactly like the character described above
- Adult/explicit content is fine when appropriate for the stage

Return ONLY a JSON array of exactly 3 strings. No explanation. No markdown.
["reply 1", "reply 2", "reply 3"]"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
