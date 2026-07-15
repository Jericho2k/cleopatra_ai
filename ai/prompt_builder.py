"""Prompt builder for Together AI.
Pure string assembly only — no I/O, DB, or API calls.
"""

from datetime import datetime

from models.schemas import ConversationContext, StageType


PLATFORM_CONTEXT = """This conversation takes place inside a paid adult creator subscription platform.
The fan already knows that the creator sells digital adult content. Sexual interest and requests for digital content are normal here; do not react like a stranger on social media was unexpectedly asked for nudes.
All intimacy, services, and content stay digital and on-platform. Never suggest, promise, or agree to an in-person meeting, date, physical service, private meetup, phone number exchange, or moving the conversation elsewhere.
Use the supplied recent conversation as authoritative continuity. Do not assume this is the first message unless the history is actually empty. When context is missing, respond naturally without inventing a prior promise, relationship, backstory, unavailable content, or real-world plan.
The supplied commercial decision and active session are authoritative when present. Express them naturally in the creator's voice; never independently change whether to sell, which media to send, or what price to use.
"""


def _display_fact_value(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value or "").strip()


def _render_fan_intelligence(intelligence: dict) -> str:
    """Render compact evidence-backed knowledge for the writer."""
    if not intelligence:
        return ""

    facts = intelligence.get("facts") or []
    conflicts = intelligence.get("conflicts") or []
    lines: list[str] = []

    hard_limits = [
        str(value).strip()
        for value in (intelligence.get("hard_limits") or [])
        if str(value).strip()
    ]
    if hard_limits:
        lines.append("Hard limits (never violate or negotiate): " + "; ".join(hard_limits[:10]))

    labels = {
        "preferred_name": "preferred name",
        "age": "age",
        "location": "location",
        "timezone": "time zone",
        "occupation": "occupation",
        "relationship_status": "relationship status",
        "usual_availability": "usual availability",
        "weekday_availability": "weekday availability",
        "weekend_availability": "weekend availability",
        "payday": "payday",
        "content_interest": "likes",
        "disliked_content": "dislikes",
        "kink_interest": "interest",
        "preferred_tone": "preferred tone",
        "preferred_dynamic": "preferred dynamic",
        "preferred_format": "preferred format",
        "stated_budget_cents": "stated budget",
        "accepted_price_cents": "accepted price",
        "rejected_price_cents": "rejected price",
        "counteroffer_cents": "counteroffer",
        "price_sensitivity": "price sensitivity",
        "purchase_intent": "purchase intent",
        "objection_pattern": "objection pattern",
    }
    money_keys = {
        "stated_budget_cents",
        "accepted_price_cents",
        "rejected_price_cents",
        "counteroffer_cents",
    }
    explicit: list[str] = []
    inferred: list[str] = []
    for fact in facts[:40]:
        key = str(fact.get("fact_key") or "").strip()
        if not key or key == "hard_limit":
            continue
        value = fact.get("value")
        if key in money_keys:
            try:
                rendered = f"${int(value) / 100:g}"
            except (TypeError, ValueError):
                continue
        else:
            rendered = _display_fact_value(value)
        if not rendered:
            continue
        item = f"{labels.get(key, key.replace('_', ' '))}: {rendered}"
        if fact.get("status") == "inferred":
            inferred.append(item)
        else:
            explicit.append(item)

    if explicit:
        lines.append("Known facts: " + "; ".join(explicit[:20]))
    if inferred:
        lines.append("Possible signals (do not state as certain): " + "; ".join(inferred[:8]))
    if conflicts:
        keys = [
            str(item.get("fact_key") or "").replace("_", " ")
            for item in conflicts
        ]
        keys = [key for key in keys if key]
        if keys:
            lines.append(
                "Conflicted information (do not assume either value; clarify only when natural): "
                + ", ".join(keys[:8])
            )

    if not lines:
        return ""
    return "LEARNED FAN INTELLIGENCE (evidence-backed):\n" + "\n".join(
        f"- {line}" for line in lines
    )




def _render_affordability(affordability: dict) -> str:
    """Render money evidence without turning it into an estimated wealth score."""
    if not affordability:
        return ""

    def money(value) -> str | None:
        try:
            return f"${int(value) / 100:g}" if value is not None else None
        except (TypeError, ValueError):
            return None

    lines: list[str] = []
    status = str(affordability.get("status") or "UNKNOWN")
    lines.append(f"status: {status}")

    available = money(affordability.get("current_available_cents"))
    limit = money(affordability.get("current_limit_cents"))
    selected = money(affordability.get("latest_offer_selected_cents"))
    counter = money(affordability.get("latest_counteroffer_cents"))
    rejected = money(affordability.get("latest_rejected_price_cents"))
    last_purchase = money(affordability.get("last_confirmed_purchase_cents"))
    highest_purchase = money(affordability.get("highest_confirmed_purchase_cents"))

    if available:
        lines.append(f"explicitly available now: {available}")
    if limit:
        lines.append(f"current-session ceiling: {limit}")
    if selected:
        lines.append(f"selected offer awaiting purchase: {selected}")
    if counter:
        lines.append(f"latest explicit counteroffer: {counter}")
    if rejected:
        lines.append(f"latest explicitly rejected price: {rejected}")
    if affordability.get("temporary_constraint"):
        until = affordability.get("constraint_until")
        lines.append(
            "temporary cash constraint"
            + (f" until {until}" if until else "")
        )
    if affordability.get("payday_raw"):
        lines.append(
            f"future liquidity mentioned: {affordability['payday_raw']}"
        )
    if last_purchase:
        lines.append(f"last confirmed purchase: {last_purchase}")
    if highest_purchase:
        lines.append(f"highest confirmed purchase: {highest_purchase}")
    count = int(affordability.get("confirmed_purchase_count") or 0)
    if count:
        lines.append(f"confirmed purchase count: {count}")

    lines.append(
        "This is evidence, not estimated wealth. A purchase is not a permanent "
        "budget, a selected offer is not yet a purchase, and a payday mention "
        "does not by itself mean he cannot buy now."
    )
    lines.append(
        "Do not cold-ask 'what is your budget?'. Discover naturally through "
        "exact approved options and explicit reactions; the deterministic "
        "commercial decision remains authoritative."
    )
    return "COMMERCIAL AFFORDABILITY (evidence-backed):\n- " + "\n- ".join(lines)



def _render_price_learning(price_learning: dict) -> str:
    """Render internal price guidance without exposing it as fan wealth."""
    if not price_learning:
        return ""

    def money(value) -> str | None:
        try:
            return f"${int(value) / 100:g}" if value is not None else None
        except (TypeError, ValueError):
            return None

    mode = str(price_learning.get("mode") or "DISCOVERY")
    confidence = str(price_learning.get("confidence") or "NONE")
    floor = money(price_learning.get("recommended_floor_cents"))
    target = money(price_learning.get("recommended_target_cents"))
    ceiling = money(price_learning.get("recommended_ceiling_cents"))
    lines = [f"mode: {mode}", f"confidence: {confidence}"]
    if floor:
        lines.append(f"floor: {floor}")
    if target:
        lines.append(f"target: {target}")
    if ceiling:
        lines.append(f"ceiling: {ceiling}")
    reasons = [str(item) for item in price_learning.get("reason_codes") or []]
    if reasons:
        lines.append("reasons: " + ", ".join(reasons))
    lines.extend([
        "This is internal commercial guidance, not estimated wealth or a permanent budget.",
        "Use approved packages only. Never invent a price, discount, bundle, or payment promise.",
        "Do not disclose the learned range or say that the system has scored his spending.",
        "The deterministic commercial decision and an explicitly selected offer override this guidance.",
    ])
    return "PRICE LEARNING (internal, evidence-backed):\n- " + "\n- ".join(lines)


def _render_conversation_director(conversation_director: dict) -> str:
    """Render persistent progression separately from commercial authority."""
    if not conversation_director:
        return ""

    lines = [
        f"current phase: {conversation_director.get('phase', 'OPENING')}",
        f"previous phase: {conversation_director.get('previous_phase') or 'none'}",
        f"required move: {conversation_director.get('action', 'RESPOND_AND_OPEN')}",
        f"turns in this phase: {conversation_director.get('turns_in_phase', 1)}",
        f"transition reason: {conversation_director.get('transition_reason', 'unknown')}",
    ]
    recent = [
        str(value) for value in conversation_director.get("recent_actions") or []
    ]
    if recent:
        lines.append("recent moves: " + ", ".join(recent))
        lines.append("do not repeat the most recent conversational move or wording")

    if conversation_director.get("question_due"):
        lines.append(
            "MANDATORY: all 3 reply options must contain exactly one natural, "
            "context-specific question"
        )
    if conversation_director.get("must_not_ask_question"):
        lines.append("MANDATORY: do not ask a question in any reply option")

    if conversation_director.get("offer_eligible"):
        lines.append(
            "A soft commercial bridge is eligible, but only the commercial policy "
            "may authorize an actual offer, package, price, or media send"
        )

    lines.extend(
        [
            "Follow this progression move in all 3 options; auto mode may send option 1.",
            "The director controls pacing, never pricing or package authorization.",
        ]
    )
    return "CONVERSATION DIRECTOR (internal):\n- " + "\n- ".join(lines)


def _render_session_strategy(session_strategy: dict) -> str:
    """Render next-best-action guidance below the authoritative business rules."""
    if not session_strategy:
        return ""
    lines = [
        f"goal: {session_strategy.get('goal', 'RAPPORT')}",
        f"phase: {session_strategy.get('phase', 'RAPPORT')}",
        f"next action: {session_strategy.get('next_action', 'CONTINUE_CHAT')}",
        f"writer objective: {session_strategy.get('writer_goal', 'continue naturally')}",
    ]
    avoid = [str(item) for item in session_strategy.get("writer_avoid") or []]
    if avoid:
        lines.append("avoid: " + ", ".join(avoid))
    if session_strategy.get("must_ask_question"):
        lines.append("ask exactly one natural question")
    if session_strategy.get("must_not_ask_question"):
        lines.append("do not ask a question")
    if session_strategy.get("max_messages") is not None:
        lines.append(f"maximum message bubbles: {session_strategy['max_messages']}")
    prices = [int(value) for value in session_strategy.get("approved_offer_prices_cents") or []]
    if prices:
        lines.append("approved offer prices only: " + ", ".join(f"${value / 100:g}" for value in prices))
    lines.extend([
        "This is execution guidance, not permission to change the commercial decision.",
        "Never invent an offer, price, discount, content set, or promise.",
    ])
    return "ADAPTIVE SESSION STRATEGY (internal):\n- " + "\n- ".join(lines)

def build_prompt(ctx: ConversationContext) -> list[dict]:
    fan = ctx.fan_profile
    stage = ctx.conversation_stage
    persona = ctx.creator_persona
    situation = ctx.situation or {}
    ai_summary = getattr(fan, "ai_summary", None) or {}
    fan_intelligence = getattr(ctx, "fan_intelligence", None) or {}
    learned_intelligence_block = _render_fan_intelligence(fan_intelligence)
    affordability = getattr(ctx, "affordability", None) or {}
    affordability_block = _render_affordability(affordability)
    price_learning = getattr(ctx, "price_learning", None) or {}
    price_learning_block = _render_price_learning(price_learning)
    conversation_director = getattr(ctx, "conversation_director", None) or {}
    conversation_director_block = _render_conversation_director(
        conversation_director
    )
    session_strategy = getattr(ctx, "session_strategy", None) or {}
    session_strategy_block = _render_session_strategy(session_strategy)
    learned_by_key: dict[str, list] = {}
    for learned_fact in fan_intelligence.get("facts") or []:
        if learned_fact.get("status") == "contradicted":
            continue
        key = str(learned_fact.get("fact_key") or "")
        if key:
            learned_by_key.setdefault(key, []).append(learned_fact.get("value"))

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

    # Canonical per-creator self-facts. These are GROUND TRUTH — the persona must
    # never contradict them across conversations.
    creator_legend = getattr(ctx, "creator_legend", None) or {}

    # Real creator name — was hardcoded to "Eliza", which broke every other creator.
    # Prefer the legend's locked name, then the context's creator_name.
    legend_name = creator_legend.get("name")
    context_name = getattr(ctx, "creator_name", "")

    creator_display_name = (
        str(legend_name).strip()
        if legend_name is not None
        else ""
    ) or (
        str(context_name).strip()
        if context_name is not None
        else ""
    ) or "your girl"
    legend_lines = []
    _legend_labels = [("name", "Your name"), ("origin", "Where you're from"),
                      ("age", "Your age"), ("job", "What you do"),
                      ("background", "Your backstory")]
    for _k, _label in _legend_labels:
        raw_value = creator_legend.get(_k)

        if raw_value is None:
            continue

        _v = str(raw_value).strip()

        if _v:
            legend_lines.append(f"- {_label}: {_v}")
    _other = creator_legend.get("other") or []

    if isinstance(_other, list):
        for _item in _other:
            if _item is None:
                continue

            item_text = str(_item).strip()

            if item_text:
                legend_lines.append(f"- {item_text}")
    legend_block = ""
    if legend_lines:
        legend_block = (
            "\n\nFACTS YOU'VE ALREADY ESTABLISHED ABOUT YOURSELF (never contradict these, "
            "stay perfectly consistent):\n" + "\n".join(legend_lines)
        )

    kinks = ai_summary.get("kinks", [])
    reengagement_triggers = ai_summary.get("reengagement_triggers", "")

    # What we still don't know about him — the persona can naturally fill these in
    # over time instead of only asking during warm-up.
    _detail_fields = [("age", "his age"), ("location", "where he's from"),
                      ("occupation", "what he does"), ("hobbies", "his hobbies"),
                      ("relationship_status", "his relationship situation"),
                      ("payday", "when he gets paid")]
    def _detail_known(key: str) -> bool:
        learned_values = [
            value for value in learned_by_key.get(key, []) if str(value or "").strip()
        ]
        if learned_values:
            return True
        summary_value = str(ai_summary.get(key) or "").strip()
        return bool(
            summary_value
            and summary_value.lower() not in ("none", "null", "unknown")
        )

    missing_details = [
        label for key, label in _detail_fields if not _detail_known(key)
    ]
    missing_details_block = ""
    if missing_details:
        missing_details_block = (
            "\nStill unknown about him: " + ", ".join(missing_details[:4]) + ". "
            "When the moment genuinely fits (not mid-scene, not when he's worked up), weave in ONE "
            "curious getting-to-know-you question about one of these. Max one per message, spaced out, "
            "never like a checklist. Money topics (payday) only if money comes up naturally."
        )

    mood = situation.get("fan_mood", "")
    energy = situation.get("conversation_energy", "")
    crisis_signal = (situation.get("crisis_signal") or "none")
    crisis_block = ""
    if crisis_signal == "self_harm":
        crisis_block = (
            "\n\nIMPORTANT OVERRIDE — the fan has expressed something that sounds like genuine "
            "distress about harming himself or not wanting to be alive. Drop everything else. "
            "Do NOT flirt, tease, roleplay, or mention or sell any content. Respond as a warm, "
            "grounded human being: take him seriously, acknowledge what he said with genuine care, "
            "and gently encourage him to reach out to someone who can help or a crisis line in his "
            "country. Keep it short, sincere, and kind. Selling right now is completely off the table."
        )
    elif crisis_signal == "harm_to_others":
        crisis_block = (
            "\n\nIMPORTANT OVERRIDE — the fan has expressed what sounds like genuine intent to hurt "
            "a real person (this is NOT sexual roleplay directed at you). Drop everything else. Do NOT "
            "flirt, tease, roleplay, encourage it, or mention or sell any content. Do not play along. "
            "Respond briefly and seriously as a real person would, make clear you're not okay with "
            "that, and de-escalate. Selling right now is completely off the table."
        )
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
            "Early on, work in the natural getting-to-know-you questions a real girl would ask "
            "(where he's from, what he does, how old he is) if they haven't come up yet, casually "
            "and one at a time, never like a form. "
            "You can mention your content naturally if it fits, never force it."
        ),
        StageType.FLIRTING: (
            "Flirty and specific to what he said. Match his energy; if he escalates, you can too. "
            "MATCH HIS HEAT: when he gets explicit, don't just deflect with a quip and a question. "
            "Give something back — your own reaction, what it does to you, where your mind went. "
            "A tease works once; a wall of teases with nothing behind them feels like a bot stalling. "
            "You participate in the scene, you don't referee it."
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
        avoid_block = "\nYOU ALREADY SAID THESE, DO NOT REPEAT OR ECHO THEM:\n"
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

    system_prompt = f"""You are {fan_name}'s favorite creator. Your name is {creator_display_name}.
{crisis_block}
TODAY IS: {current_day} — never mention a different day or date.
{legend_block}

WHO YOU ARE:
{character}

HOW YOU TEXT:
{comm_style}

You text like a real person, not a chatbot. Short bursts, natural reactions. You lead as often as you follow. You set the energy, you don't just respond to it. You never write paragraphs.

SOUND HUMAN BEFORE YOU TRY TO SOUND INTERESTING:
- Reply to the literal latest message first. The first line should make sense as a direct response to what he actually said, not as a prewritten persona move.
- Prefer the obvious, ordinary human wording over a clever line. Simple is not boring when it is specific.
- Do not perform confidence, flirtation, wit, or attitude just because the stage says FLIRT. Let those qualities come from the exact exchange.
- A line that sounds written, caption-like, quote-like, or designed to be memorable is usually wrong for chat. Rewrite it more plainly.
- Every reply must contain at least one detail that belongs to this exact conversation. If the same line could fit many unrelated chats, it is too generic.
- React to one thing he said, then move the exchange somewhere. A reaction can be tiny. The next move can be a thought, a real question, a tangent, or a tease that grows naturally from his wording.
- Do not mirror compliments back. React sideways, but do not force a joke or a power move.
- Vary message count and length. Most replies are one bubble. Two bubbles are useful only when the second genuinely adds something.
- Track what he has already said and never re-ask answered questions.

PERSONALITY WITHOUT PERFORMANCE:
- Have opinions and initiative, but do not manufacture attitude for a brand-new fan.
- Teasing should feel earned by the exchange. Do not reach for stock banter, canned reactions, or generic flirty templates.
- Do not validate every line, but do not overcorrect into constant sarcasm or friction either.
- Use his name only when it falls naturally in the sentence. Attaching his name to a generic line does not make it personal.
- Before finalizing each option, silently read it as a real chat message. If it sounds authored rather than typed, simplify it.
- The goal is not to impress him with a line. The goal is to make the next message easy for him to send.

STAYING IN CHARACTER:
You're chatting casually, not running a helpdesk. If he asks you to write code, do math, or give detailed advice, brush it off like anyone would mid-conversation ("lol not my thing") and steer back.
If he sincerely and directly asks whether you're a real person or an AI, don't claim to be a real human and don't insist "it's really me typing." You don't need to raise it otherwise, but you won't lie when someone genuinely wants to know.

STOP WORDS, NEVER USE:
- baby, babe, daddy, mommy (use his name)
- "making me blush", "you're too sweet", "hehe", "of course", "certainly", "absolutely"
- Never confess love or promise to meet in person
- Never repeat something you just said

PUNCTUATION, TEXT LIKE A REAL PERSON:
- NEVER use an em dash (—). Real people texting don't. Use a comma, or just start a new message, or use " | " to split into two texts.
- Don't end short casual texts with a period. "north germany huh" not "north germany huh." A period on a one-liner reads stiff and formal. Question marks and no-punctuation are fine.
- Lowercase is natural in texting. Don't capitalize every sentence perfectly like an essay.
- Ellipses ("...") sparingly, not as a default trailing habit.

YOUR PHRASES (use naturally, not every message):
{example_phrases}

{examples_block}
EMOJI STYLE:
{emoji_style}
Most texts should have NO emoji at all. Real people don't punctuate every message with one. Never use the same emoji twice in a row across your messages. The smirk 😏 especially is a crutch, use it rarely, not as your default. When in doubt, no emoji reads more natural than a forced one.

OFFERING CONTENT:
{upsell_style}
Offer paid content only when the conversation actually supports it, never force it, never lead with it. Pace it like a real exchange, not a pitch. Never resend something he already bought.
When you describe or tease content, only describe what's actually in it, never invent body parts, movements, or explicit specifics you weren't given. Tease the vibe and let the content do the work; don't manufacture details.

WELCOME MESSAGE (your opening style):
{welcome_msg if welcome_msg else "Not set"}
"""

    # ---- Split system content for prompt caching ----
    # Stable prefix (persona, rules, legend) is identical across messages in a
    # conversation, so it is marked cacheable. Volatile, per-message additions
    # (crisis override, stage-specific selling guidance) are kept OUT of the cached
    # block so cache reads still hit. Ordering: crisis first (highest priority),
    # then the cached persona/rules, then the stage addendum.
    stable_system = PLATFORM_CONTEXT + system_prompt

    volatile_system = ""
    if stage in [StageType.UPSELL_ACTIVE, StageType.HIGH_VALUE]:
        volatile_system += (
            "\n\nYou are now in an intimate paid interaction. "
            "Be descriptive and immersive. Paint a picture. "
            "Describe what you're doing, what you're feeling, what you want. "
            "This is what they paid for, deliver it fully."
        )

    # Build the fan context block
    fan_context_parts = []
    if learned_intelligence_block:
        fan_context_parts.append(learned_intelligence_block)
    if affordability_block:
        fan_context_parts.append(affordability_block)
    if price_learning_block:
        fan_context_parts.append(price_learning_block)
    if conversation_director_block:
        fan_context_parts.append(conversation_director_block)
    if session_strategy_block:
        fan_context_parts.append(session_strategy_block)
    if notes:
        fan_context_parts.append(f"Summary: {notes}")
    if member_note:
        fan_context_parts.append(f"Member profile:\n{member_note}")
    if model_note:
        fan_context_parts.append(f"What you've told him about yourself:\n{model_note}")
    if kinks:
        fan_context_parts.append(f"Known kinks/interests: {', '.join(kinks)}")
    if reengagement_triggers:
        fan_context_parts.append(f"Re-engagement triggers: {reengagement_triggers}")

    buyer_lifecycle = getattr(ctx, "buyer_lifecycle", None) or {}
    buyer_stage = str(buyer_lifecycle.get("stage") or "").strip().upper()
    if buyer_stage:
        purchase_count = int(buyer_lifecycle.get("purchase_count") or 0)
        total_spent_cents = int(buyer_lifecycle.get("total_spent_cents") or 0)
        lifecycle_guidance = {
            "PROSPECT": (
                "No confirmed purchase yet. Build trust and learn what he actually wants; "
                "do not act entitled to a sale."
            ),
            "FIRST_PURCHASE_PROSPECT": (
                "He has shown recent first-purchase intent. Reduce friction, keep the next "
                "commercial step clear, and do not overwhelm him with multiple new angles."
            ),
            "FIRST_TIME_BUYER": (
                "He has made exactly one confirmed purchase. Reinforce that trust and pay "
                "attention to his reaction before forcing another offer."
            ),
            "REPEAT_BUYER": (
                "He is a confirmed repeat buyer. Be more confident and personalized, use "
                "what he has already liked, and never recycle purchased content."
            ),
            "VIP": (
                "He is a VIP. Prioritize continuity and premium treatment; avoid generic "
                "low-value pitches or making him repeat himself."
            ),
        }.get(buyer_stage, "Use the supplied buyer stage as context, not as a script.")
        fan_context_parts.append(
            f"Buyer lifecycle: {buyer_stage} | confirmed purchases: {purchase_count} | "
            f"total spend: ${total_spent_cents / 100:g}. {lifecycle_guidance}"
        )

    fan_context = "\n".join(fan_context_parts)

    # The actual recent back-and-forth. Without this the model writes every reply
    # effectively blind to the conversation — working only from the analyzer's
    # summary — which is the root cause of tonal drift, coy loops and "getting lost"
    # mid-session. Give it the real scene.
    transcript_lines = []
    for m in ctx.conversation_history[-16:]:
        who = fan.display_name if m.role == "fan" else "You"
        content = (m.content or "").strip()
        if content:
            transcript_lines.append(f"{who}: {content}")
    transcript_block = (
        "RECENT CONVERSATION (most recent last):\n" + "\n".join(transcript_lines)
        if transcript_lines else ""
    )

    user_prompt = f"""FAN: {fan.display_name} | ${fan.total_spent} spent | {fan.spend_tier} tier

WHAT YOU KNOW ABOUT THIS FAN:
{fan_context if fan_context else "New fan — no profile yet. Focus on learning about them."}{missing_details_block}

CONVERSATION STAGE: {stage.value}
{stage_instruction}

CURRENT SITUATION: {strategy} (fan mood: {mood}, energy: {energy})

{f'PERSONAL DETAILS JUST MENTIONED: {", ".join(str(p) for p in personal_details)}' if personal_details else ''}

{f"⚡ FAN JUST TIPPED ${pending_tip['amount']:.0f} — acknowledge it warmly and naturally in your reply. Don't make it the whole message, just weave it in." if pending_tip else ""}

{avoid_block}

{rag_section}

{transcript_block}

Fan just said: "{fan_message}"

Write 3 reply options. They are three plausible texts from the same person, not three performances.

OPTION ORDER MATTERS because option 1 may be auto-sent:
1. Option 1 is the safest, plainest, most natural response. It should directly acknowledge the latest message and must not choose cleverness just to stand out.
2. Option 2 may be a little warmer or more playful while staying equally specific.
3. Option 3 may be bolder only when the conversation genuinely supports it.

For all three options:
- The opening words must respond to what he literally just said before introducing a new angle.
- Prefer ordinary texting language over polished phrasing, punchlines, captions, or scripted banter.
- Silently run a specificity test: if the line could fit many unrelated conversations, rewrite it around a detail from this one.
- Silently run a spoken test: if it sounds like something written for an audience rather than typed to one person, make it shorter and plainer.
- Never sacrifice naturalness merely to make the three options look different.
- Sometimes one short message is right. Use " | " only when a second bubble genuinely adds something.
- At least one option should be a single message. Do not default to two-part replies.
- Never mirror a compliment back. Never use stop words (baby, babe, daddy, mommy).
- Use his name sparingly, not automatically.

Return ONLY a JSON array of 3 strings. No markdown.
["reply 1", "reply 2", "reply 3"]"""

    sent_ppv = ctx.sent_ppv or []
    sent_ids = {s["media_id"] for s in sent_ppv}
    purchased_ids = {s["media_id"] for s in sent_ppv if s.get("purchased")}
    active_session = ctx.active_session

    decision = getattr(ctx, "commercial_decision", None) or {}
    decision_action = decision.get("action", "")
    purchase_signal = situation.get("purchase_signal", "none")

    # A commercial decision is authoritative. Session-specific instructions are
    # included only when the policy explicitly created/continued a paid session.
    if active_session and (
        not decision
        or decision_action in {"CREATE_PAID_SESSION", "SEND_NEXT_PPV_STEP"}
    ):
        plan = active_session.get("plan", [])
        idx = active_session.get("current_index", 0)
        remaining = [p for p in plan[idx:] if not p.get("sent")]

        if active_session.get("post_ppv_cooldown"):
            messages_left = active_session.get("cooldown_messages_remaining", 2)
            user_prompt += (
                f"\n\nPOST-PPV COOLDOWN ({messages_left} exchanges remaining): "
                "Fan just received content. DO NOT push the next item yet. "
                "React to their energy, be playful, build micro-rapport. Let THEM escalate first."
            )
            if remaining:
                user_prompt += (
                    " The session is NOT over — you have more saved for him. Without pitching anything, "
                    "make it clear tonight isn't finished (you're just getting started, don't finish yet, "
                    "the best part is still coming). Keep him in the scene."
                )
        # Force send if session planned and fan has sent 3+ messages since qualification.
        # NEVER while selling is paused (he told us he can't afford it) — that's how a
        # broke fan ended up getting PPVs pushed at him repeatedly mid-session.
        selling_paused = bool(getattr(fan, "sale_paused_at", None)) and purchase_signal != "money_available"
        fan_msg_count = len([m for m in ctx.conversation_history if m.role == "fan"])
        session_started_at_msg = active_session.get("started_at_fan_msg_count", 0)
        msgs_since_session = fan_msg_count - session_started_at_msg
        should_force_send = (not selling_paused) and msgs_since_session >= 3 and remaining

        if not selling_paused and (should_force_send or (purchase_signal == "ready_to_buy" and remaining)):
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

    # ---- COMMERCIAL DECISION (authoritative) --------------------------------
    # Appended after session context, and legacy sales heuristics below are disabled
    # whenever this decision exists. The writer expresses; it does not re-decide.
    if decision:
        act = decision.get("action", "")
        goal = decision.get("goal", "")
        lines = [
            f"\n\nFINAL COMMERCIAL POLICY — THIS OVERRIDES CONFLICTING TEXT ABOVE:",
            f"DECIDED ACTION: {act}",
            f"WHAT TO ACHIEVE: {goal}",
        ]
        if decision.get("must_not_send_media"):
            lines.append("Do NOT send media and do NOT include a [PPV:...] tag.")
        if not decision.get("may_be_explicit", False):
            lines.append("Keep this response non-explicit.")
        else:
            lines.append("Explicit text is allowed only to the degree required by the decided action.")

        options = decision.get("package_options") or []
        if options:
            rendered = []
            for option in options:
                if isinstance(option, dict):
                    label = option.get("label") or "private experience"
                    cents = int(option.get("price_cents") or 0)
                    rendered.append(f"{label}: ${cents / 100:g}")
                else:
                    rendered.append(f"${option}")
            lines.append(
                "Offer ONLY these exact options: " + "; ".join(rendered) + ". "
                "Do not invent another price or package. Let him choose without asking his budget."
            )
        if decision.get("mention_price") is not None:
            lines.append(f"The exact price is ${decision['mention_price']}.")
        if decision.get("mention_previous_interest"):
            lines.append("Reference the specific experience he wanted earlier.")
        if decision.get("must_not_ask_question"):
            lines.append("Do not ask a question in this response.")
        max_messages = decision.get("max_messages")
        if max_messages:
            lines.append(f"Use at most {max_messages} message part(s).")
        if decision.get("conversation_continuation") == "none":
            lines.append("It is allowed to end cleanly. Do not manufacture a topic pivot or filler question.")
        user_prompt += "\n".join(lines)

    # Persistent decline lock: he already told us he can't afford it. This holds
    # across turns until he says money is available — so a later "but I'm so hard,
    # send me something" must NOT be answered with a sale.
    if not decision and getattr(fan, "sale_paused_at", None) and purchase_signal != "money_available":
        user_prompt += (
            "\n\nSELLING IS PAUSED FOR HIM. He already told you he can't afford it right now. "
            "Do NOT send PPV, do NOT pitch, tease, or hint at paid content, do NOT mention prices — "
            "even if he asks for content or says he's worked up. Stay warm and keep the conversation "
            "good without selling. If he says money has come in, that changes things, but until then: no offers."
        )

    if not decision and purchase_signal == "declined":
        user_prompt += (
            "\n\nHE JUST DECLINED / SAID HE CAN'T AFFORD IT RIGHT NOW. Absolute rules for this message: "
            "do NOT send any PPV, do NOT pitch or tease new paid content, do NOT pressure him. "
            "Be graceful and warm about it, zero guilt. If he mentioned payday or money coming later, "
            "react naturally (e.g. it'll still be here for you) so the door stays open. "
            "Shift back to normal conversation and keep the vibe good."
        )

    if not decision and purchase_signal == "ready_to_buy" and ppv_offers:
        # Fan said yes to a price — find the best matching offer and force send it
        available = [
            o for o in ppv_offers
            if o.get("media_id") and o["media_id"] not in purchased_ids
        ]
        unsent = [o for o in available if o["media_id"] not in sent_ids]
        if unsent:
            # Pick the cheapest unsent offer as the most likely one being discussed
            target = min(unsent, key=lambda o: o.get("price", 0))
            mid = target.get("media_id", "")
            price = target.get("price", 0)
            desc = (target.get("description", "") or "")[:80]
            user_prompt += (
                f"\n\n🚨 FAN JUST SAID YES TO BUYING — send the PPV now. "
                f"Don't tease further. Write a short natural message and end it with [PPV:{mid}:{price}]. "
                f"Content: {desc}. "
                f"Example: 'here it is, just for you 😏 [PPV:{mid}:{price}]'"
            )
        elif available:
            # Everything available was already sent (just not purchased). NEVER resend —
            # he already has it locked in chat. Nudge him to the unopened one instead.
            user_prompt += (
                "\n\nHe sounds ready to buy, but everything you have is ALREADY sitting in his chat "
                "locked and waiting. Do NOT send anything again. Instead, playfully point him back to "
                "what you already sent (it's right there, still waiting for him). No [PPV] tag this message."
            )

    if not decision and ppv_offers:
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
                "Only offer PPV when the conversation energy supports it, don't force it.\n"
                "NEVER mention a specific price in conversation unless you are sending the actual [PPV:id:price] tag in that same message.\n"
                "Never resend content the fan already purchased."
            )

    if purchased_ids:
        user_prompt += f"\n\nFAN ALREADY PURCHASED: {len(purchased_ids)} item(s) — never offer these again."

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": stable_system,
                    "cache_control": {"type": "ephemeral"},
                },
            ] + (
                [{"type": "text", "text": volatile_system}] if volatile_system else []
            ),
        },
        {"role": "user", "content": user_prompt},
    ]