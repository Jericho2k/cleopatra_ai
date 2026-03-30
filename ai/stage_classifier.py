"""Conversation stage classifier.
Pure logic only — no I/O, DB, or API calls.
"""
from datetime import datetime, timezone
from models.schemas import Fan, Message, StageType


def classify_stage(
    conversation_history: list[Message],
    fan_profile: Fan,
) -> StageType:

    message_count = len(conversation_history)
    fan_messages = [m for m in conversation_history if m.role == "fan"]
    fan_text_recent = " ".join(m.content for m in fan_messages[-10:]).lower()
    fan_text_all = " ".join(m.content for m in fan_messages).lower()

    # ── HIGH VALUE: big spender ──
    if fan_profile.total_spent > 300:
        return StageType.HIGH_VALUE

    # ── HIGH VALUE: findom signals in ANY fan message ──
    findom_keywords = [
        "i'll pay anything", "ill pay anything", "take my money",
        "drain me", "you own me", "i belong to you", "you control me",
        "i am yours", "i obey", "yes mistress", "yes goddess",
        "just tell me what to send", "how much do you want",
        "i would pay anything", "i owe you", "i'll do anything",
        "ill do anything", "you deserve everything", "spoil me please",
        "i would honestly pay whatever",
        "pay whatever you want",
        "you deserve it all",
        "whatever you ask",
        "i'll pay whatever",
        "ill pay whatever",
        "pay anything for you",
    ]
    if any(phrase in fan_text_all for phrase in findom_keywords):
        return StageType.HIGH_VALUE

    # ── OBJECTION: recent price resistance ──
    objection_keywords = [
        "too expensive", "too much", "cant afford", "can't afford",
        "not worth", "no thanks", "cheaper", "maybe later",
        "not right now", "that's a lot", "thats a lot", "too pricey",
    ]
    if any(phrase in fan_text_recent for phrase in objection_keywords):
        return StageType.OBJECTION

    # ── UPSELL ACTIVE: fan asking about buying ──
    # Only trigger on FAN messages to avoid false positives from creator messages
    upsell_keywords = [
        "how much", "what does it cost", "how much for",
        "can i buy", "i want to buy", "would you make",
        "how do i get", "what's the price", "whats the price",
        "can i get a custom", "do you do customs", "send me a ppv",
    ]
    if any(phrase in fan_text_recent for phrase in upsell_keywords):
        return StageType.UPSELL_ACTIVE

    # ── Very early conversation ──
    if message_count <= 2:
        return StageType.COLD_OPEN

    # ── RETENTION: fan was active before but went quiet ──
    if fan_profile.last_active is not None:
        now = datetime.now(timezone.utc)
        last_active = fan_profile.last_active
        if last_active.tzinfo is None:
            last_active = last_active.replace(tzinfo=timezone.utc)
        if (now - last_active).days > 3:
            return StageType.RETENTION

    # ── PRE_UPSELL: fan showing buying intent or strong attachment ──
    # Only check fan messages, not creator messages
    pre_upsell_keywords = [
        "cant stop thinking about you", "obsessed with you",
        "you are my favorite", "you're my favorite",
        "worth it", "you deserve", "i want more",
        "what else do you have", "what else do you offer",
        "how can i see more", "is there more",
    ]
    if any(phrase in fan_text_recent for phrase in pre_upsell_keywords):
        return StageType.PRE_UPSELL

    # ── Fan has spent before — skip straight to PRE_UPSELL ──
    if fan_profile.total_spent > 50:
        return StageType.PRE_UPSELL

    # ── FLIRTING: sexual/romantic signals ──
    flirting_keywords = [
        "sexy", "so hot", "you're hot", "youre hot", "gorgeous",
        "want you", "miss you", "turn me on", "hard for you",
        "beautiful", "stunning", "perfect body", "dream girl",
        "my type", "so attractive", "so pretty", "omg ur",
    ]
    if any(phrase in fan_text_recent for phrase in flirting_keywords):
        if message_count < 12:
            return StageType.FLIRTING
        return StageType.PRE_UPSELL

    # ── Default by message count ──
    if message_count < 4:
        return StageType.WARMING_UP
    if message_count < 14:
        return StageType.FLIRTING

    return StageType.PRE_UPSELL
