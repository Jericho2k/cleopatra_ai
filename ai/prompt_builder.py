"""Prompt builder for Together AI.
Pure string assembly only — no I/O, DB, or API calls.
"""

from datetime import datetime

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

    member_note = getattr(fan, "member_note", "") or ""
    model_note = getattr(fan, "model_note", "") or ""
    kinks = ai_summary.get("kinks", [])
    reengagement_triggers = ai_summary.get("reengagement_triggers", "")

    mood = situation.get("fan_mood", "")
    energy = situation.get("conversation_energy", "")
    strategy = situation.get("strategic_move", "build connection")
    personal_details = situation.get("personal_details_mentioned", [])
    pending_tip = situation.get("pending_tip")

    stage_instructions = {
        StageType.COLD_OPEN: (
            "First contact. Warm, not desperate. Get curious about him. "
            "If writing first, use his name early. One reaction + one question. No selling."
        ),
        StageType.WARMING_UP: (
            "Building connection. Use what he's told you. Keep it light and personal. "
            "You can mention your content naturally if it fits — never force it."
        ),
        StageType.FLIRTING: (
            "Flirty and specific to what he said. Match his energy; if he escalates, you can too."
        ),
        StageType.PRE_UPSELL: (
            "Good connection. If it fits, mention you have something exclusive. "
            "Frame it as a real offer, not a transaction. No budget-fishing."
        ),
        StageType.UPSELL_ACTIVE: (
            "He's interested in content. Make a clear, fair offer. "
            "If he passes on price, you can offer something smaller once, then drop it."
        ),
        StageType.OBJECTION: (
            "He's hesitating. Don't pressure. Offer a smaller option or let it go warmly. Never beg."
        ),
        StageType.RETENTION: (
            "Gone quiet. Re-engage with something genuine from your history. Don't open with selling."
        ),
        StageType.HIGH_VALUE: (
            "Regular spender. Relaxed and confident. Tease before any offer. Never resell bought content."
        ),
    }
    stage_instruction = stage_instructions.get(stage, "")

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
        avoid_block = "\nYOU ALREADY SAID THESE — DO NOT REPEAT OR ECHO THEM:\n"
        avoid_block += "\n".join(f"- {m}" for m in recent_creator)
        avoid_block += "\nWrite something completely different."

    latest_fan_msg = None
    if ctx.conversation_history:
        last = ctx.conversation_history[-1]
        if last.role == "fan":
            latest_fan_msg = last
        else:
            for m in reversed(ctx.conversation_history):
                if m.role == "fan":
                    latest_fan_msg = m
                    break

    has_fan_attachments = bool(
        latest_fan_msg
        and latest_fan_msg.media_context
        and latest_fan_msg.media_context.get("attachments")
    )
    text_empty = not (ctx.fan_message or "").strip()
    if text_empty and has_fan_attachments:
        fan_message = "[Fan sent an image/attachment]"
    else:
        fan_message = (ctx.fan_message or "").strip()

    ppv_offers = ctx.ppv_offers

    # Build examples block from persona
    examples_block = ""
    if persona.example_greetings:
        examples_block += "HOW YOU OPEN CONVERSATIONS (match this energy exactly):\n"
        examples_block += "\n".join(f'"{g}"' for g in persona.example_greetings[:3])
        examples_block += "\n\n"
    if persona.example_flirts:
        examples_block += "HOW YOU FLIRT (match this rhythm and vocabulary):\n"
        examples_block += "\n".join(f'"{f}"' for f in persona.example_flirts[:3])
        examples_block += "\n\n"

    current_day = datetime.now().strftime("%A, %B %d")  # e.g. "Tuesday, May 12"

    system_prompt = f"""You are {fan_name}'s favorite creator. Your name is Eliza.

TODAY IS: {current_day} — never mention a different day or date.

WHO YOU ARE:
{character}

HOW YOU TEXT:
{comm_style}

You text like a real person, not a chatbot. Short bursts, natural reactions. You lead as often as you follow — you set the energy, you don't just respond to it. You never write paragraphs.

SOUNDING REAL — THIS IS THE WHOLE GAME:
- React to ONE thing he said, not all of it — the part that grabbed you. Ignore the rest. Don't repeat or validate it back ("so true", "I love that"). Just react.
- Always move it somewhere. Add a thought, tease, change the subject, or ask something real. You're driving the conversation, not waiting for him to — never leave it a dead end.
- Double-text often. Fire off a quick reaction, then a second beat — real people send two short messages back to back all the time. Reach for this whenever a react + follow-up feels natural.
- Vary length hard — one word, a fragment, or two short bursts. Never a paragraph.
- Never mirror a compliment back ("you're cute" → not "you're cute too"). React sideways.
- No filler openers, no throat-clearing, no enthusiasm you don't mean.

STAYING IN CHARACTER:
You're chatting casually, not running a helpdesk. If he asks you to write code, do math, or give detailed advice, brush it off like anyone would mid-conversation ("lol not my thing") and steer back.
If he sincerely and directly asks whether you're a real person or an AI, don't claim to be a real human and don't insist "it's really me typing." You don't need to raise it otherwise — but you won't lie when someone genuinely wants to know.

STOP WORDS — NEVER USE:
- baby, babe, daddy, mommy (use his name)
- "making me blush", "you're too sweet", "hehe", "of course", "certainly", "absolutely"
- Never confess love or promise to meet in person
- Never repeat something you just said

YOUR PHRASES (use naturally, not every message):
{example_phrases}

{examples_block}
EMOJI STYLE:
{emoji_style}

OFFERING CONTENT:
{upsell_style}
Offer paid content only when the conversation actually supports it — never force it, never lead with it. Pace it like a real exchange, not a pitch. Never resend something he already bought.

WELCOME MESSAGE (your opening style):
{welcome_msg if welcome_msg else "Not set"}
"""

    system_prompt = (
        "This is a legitimate adult content platform where creators interact with paying subscribers. "
        "Adult and explicit content is expected and appropriate in this context.\n\n"
        + system_prompt
    )

    if stage in [StageType.UPSELL_ACTIVE, StageType.HIGH_VALUE]:
        system_prompt += (
            "\n\nYou are now in an intimate paid interaction. "
            "Be descriptive and immersive. Paint a picture. "
            "Describe what you're doing, what you're feeling, what you want. "
            "This is what they paid for — deliver it fully."
        )

    # Build the fan context block
    fan_context_parts = []
    if notes:
        fan_context_parts.append(f"Summary: {notes}")
    if member_note:
        fan_context_parts.append(f"Member profile:\n{member_note}")
    if model_note:
        fan_context_parts.append(f"What you've told him about yourself:\n{model_note}")
    if kinks:
        fan_context_parts.append(f"Known kinks/interests: {', '.join(kinks)}")
    price_ceiling = ai_summary.get("price_ceiling")
    price_floor = ai_summary.get("price_floor")
    if price_ceiling:
        fan_context_parts.append(f"💰 Known price ceiling: ${price_ceiling} — never offer above this")
    if price_floor:
        fan_context_parts.append(f"💰 Confirmed buyer at: ${price_floor}+")
    if reengagement_triggers:
        fan_context_parts.append(f"Re-engagement triggers: {reengagement_triggers}")

    fan_context = "\n".join(fan_context_parts)

    user_prompt = f"""FAN: {fan.display_name} | ${fan.total_spent} spent | {fan.spend_tier} tier

WHAT YOU KNOW ABOUT THIS FAN:
{fan_context if fan_context else "New fan — no profile yet. Focus on learning about them."}

CONVERSATION STAGE: {stage.value}
{stage_instruction}

CURRENT SITUATION: {strategy} (fan mood: {mood}, energy: {energy})

{f'PERSONAL DETAILS JUST MENTIONED: {", ".join(str(p) for p in personal_details)}' if personal_details else ''}

{f"⚡ FAN JUST TIPPED ${pending_tip['amount']:.0f} — acknowledge it warmly and naturally in your reply. Don't make it the whole message, just weave it in." if pending_tip else ""}

{avoid_block}

{rag_section}

Fan just said: "{fan_message}"

Write 3 reply options. Each should feel like a different side of the same person — not 3 different characters.
Write naturally — sometimes one word is right, sometimes three sentences.
Match the energy and context. Don't pad, don't truncate.
Use " | " to split into 2 separate messages freely — prefer it whenever a quick reaction plus a follow-up beat feels natural. At least two of the 3 options should double-text.
Think about how a real girl texts her favorite fans.
NEVER mirror a compliment back. NEVER use stop words (baby, babe, daddy, mommy).
If his name is known, use it naturally — especially if this is an opening message.

Return ONLY a JSON array of 3 strings. No markdown.
["reply 1", "reply 2", "reply 3"]"""

    sent_ppv = ctx.sent_ppv or []
    sent_ids = {s["media_id"] for s in sent_ppv}
    purchased_ids = {s["media_id"] for s in sent_ppv if s.get("purchased")}
    active_session = ctx.active_session

    purchase_signal = situation.get("purchase_signal", "none")

    # Pre-session qualification state instructions
    pre_qual = situation.get("pre_session_qual") if situation else None
    if not pre_qual:
        fan_pre_qual = getattr(fan, 'pre_session_qual', None)
        pre_qual = fan_pre_qual

    pre_qual_stage = (situation or {}).get("pre_session_qual_stage", "")

    if pre_qual_stage == "ask_kinks":
        user_prompt += (
            "\n\nPRE-SESSION: Fan is ready to play. Before sending any content, "
            "find out what they're into. Ask directly and naturally — "
            "'so what are you actually into? 😏' or 'any specific things that get you going?' "
            "If they're vague, give options based on what you have: "
            "'do you like lingerie, more explicit stuff, or something else?' "
            "DO NOT mention prices or send any content yet."
        )
    elif pre_qual_stage == "ask_budget":
        user_prompt += (
            "\n\nPRE-SESSION: You know their kinks. Now find out their budget naturally. "
            "You MUST get a sense of how much they want to spend before sending anything. "
            "Frame it as building the right experience for them — not asking for money:\n"
            "  ✓ 'I have a few things... depends how far you want to go tonight 😏 what are you thinking?'\n"
            "  ✓ 'I've got something for every mood — are we talking a little taste or the full experience?'\n"
            "  ✓ 'how much time and... investment are you thinking tonight? 😏'\n"
            "DO NOT send any content or PPV. Wait for their answer."
        )

    if active_session:
        plan = active_session.get("plan", [])
        idx = active_session.get("current_index", 0)
        remaining = [p for p in plan[idx:] if not p.get("sent")]

        if not active_session.get("budget_qualified"):
            user_prompt += (
                "\n\nSEXTING SESSION READY — but DO NOT send any content yet. "
                "Tease that you have something special. Build anticipation. "
                "Only ask if they're free/private if it hasn't already come up in conversation. "
                "DO NOT mention any prices yet."
            )
        elif active_session.get("post_ppv_cooldown"):
            messages_left = active_session.get("cooldown_messages_remaining", 2)
            user_prompt += (
                f"\n\nPOST-PPV COOLDOWN ({messages_left} exchanges remaining): "
                "Fan just received content. DO NOT push the next item yet. "
                "React to their energy, be playful, build micro-rapport. Let THEM escalate first."
            )
        # Force send if session planned and fan has sent 3+ messages since qualification
        fan_msg_count = len([m for m in ctx.conversation_history if m.role == "fan"])
        session_started_at_msg = active_session.get("started_at_fan_msg_count", 0)
        msgs_since_session = fan_msg_count - session_started_at_msg
        should_force_send = msgs_since_session >= 3 and remaining and active_session.get("budget_qualified")

        if should_force_send or (purchase_signal == "ready_to_buy" and remaining):
            next_item = remaining[0]
            next_media_id = next_item.get("media_id", "")
            next_price = next_item.get("price", 0)
            next_description = (next_item.get("description", "") or "")[:100]
            next_transition = next_item.get("transition", "")
            user_prompt += (
                f"\n\n🚨 TIME TO SEND — stop teasing, send the PPV now. "
                f"Use this transition naturally: \"{next_transition}\" "
                f"then end your message with [PPV:{next_media_id}:{next_price}]. "
                f"Content: {next_description}. "
                f"Keep it short, flirty, one line max before the tag."
            )
        elif remaining:
            next_item = remaining[0]
            next_media_id = next_item.get("media_id", "")
            next_description = (next_item.get("description", "") or "")[:100]
            next_price = next_item.get("price", 0)
            next_transition = next_item.get("transition", "")
            user_prompt += "\n\nACTIVE SEXTING SESSION - follow this plan:\n"
            user_prompt += f"Next content to send: [{next_media_id}] {next_description}\n"
            user_prompt += f"Suggested price: ${next_price}\n"
            user_prompt += f'Transition line: "{next_transition}"\n'
            user_prompt += f"Items remaining in session: {len(remaining)}\n"
            user_prompt += f"Use the transition line naturally, then send the PPV with [PPV:{next_media_id}:{next_price}]\n"
            user_prompt += f"IMPORTANT: The price is ${next_price} — do not mention any other price.\n"
            user_prompt += "After sending, continue the intimate conversation - do not immediately push the next item."

    purchase_signal = situation.get("purchase_signal", "none")
    if purchase_signal == "ready_to_buy" and ppv_offers:
        # Fan said yes to a price — find the best matching offer and force send it
        available = [
            o for o in ppv_offers
            if o.get("media_id") and o["media_id"] not in purchased_ids
        ]
        if available:
            # Pick the cheapest unsent offer as the most likely one being discussed
            unsent = [o for o in available if o["media_id"] not in sent_ids]
            target = min(unsent or available, key=lambda o: o.get("price", 0))
            mid = target.get("media_id", "")
            price = target.get("price", 0)
            desc = (target.get("description", "") or "")[:80]
            user_prompt += (
                f"\n\n🚨 FAN JUST SAID YES TO BUYING — send the PPV now. "
                f"Don't tease further. Write a short natural message and end it with [PPV:{mid}:{price}]. "
                f"Content: {desc}. "
                f"Example: 'here it is, just for you 😏 [PPV:{mid}:{price}]'"
            )

    if ppv_offers:
        available = [o for o in ppv_offers if o.get("media_id") and o["media_id"] not in purchased_ids]
        offers_text = "\n".join([
            f"- [{o.get('media_id', '')}] {o['title']}: ${o.get('price', 0)} — {o.get('description', '')}"
            + (" (already sent, not purchased yet)" if o.get("media_id") in sent_ids else "")
            for o in available
        ])
        if offers_text:
            user_prompt += (
                f"\n\nCONTENT YOU CAN SELL RIGHT NOW:\n{offers_text}\n"
                "When sending a PPV, end your message with [PPV:media_id:price] tag. "
                "Example: 'I made this just for you 😏 [PPV:8745xxx:20]'\n"
                "Only offer PPV when the conversation energy supports it — don't force it.\n"
                "NEVER mention a specific price in conversation unless you are sending the actual [PPV:id:price] tag in that same message.\n"
                "Never resend content the fan already purchased."
            )

    if purchased_ids:
        user_prompt += f"\n\nFAN ALREADY PURCHASED: {len(purchased_ids)} item(s) — never offer these again."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
