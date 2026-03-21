"""Conversation stage classifier.
Pure logic only — no I/O, DB, or API calls.
"""
from datetime import datetime, timezone
from models.schemas import Fan, Message, StageType


def classify_stage(
    conversation_history: list[Message],
    fan_profile: Fan,
) -> StageType:
    if fan_profile.total_spent > 500:
        return StageType.HIGH_VALUE

    message_count = len(conversation_history)
    recent_messages = conversation_history[-10:]
    recent_text = " ".join(m.content for m in recent_messages).lower()
    all_text = " ".join(m.content for m in conversation_history).lower()

    # Findom signals — fan explicitly offers money, says they'll pay anything, calls themselves submissive
    findom_keywords = [
        "i'll pay anything", "ill pay anything", "whatever you want",
        "you deserve more", "i owe you", "take my money", "i belong to you",
        "you own me", "i obey", "yes mistress", "yes goddess", "i would do anything",
        "i would pay anything", "just tell me what to send", "how much do you want",
        "drain me", "i am yours", "you control me",
    ]
    if any(phrase in all_text for phrase in findom_keywords):
        return StageType.HIGH_VALUE

    # Objection keywords
    objection_keywords = [
        "too expensive", "too much", "cant afford", "can't afford",
        "not worth", "no thanks", "cheaper", "maybe later", "not right now",
    ]
    if any(phrase in recent_text for phrase in objection_keywords):
        return StageType.OBJECTION

    # Active upsell in progress
    upsell_keywords = [
        "ppv", "custom", "special content", "just for you",
        "exclusive", "send me", "how much", "what does it cost",
        "how much for", "would you make", "i want to buy",
    ]
    if any(phrase in recent_text for phrase in upsell_keywords):
        return StageType.UPSELL_ACTIVE

    # Very early
    if message_count <= 1:
        return StageType.COLD_OPEN

    # Retention — inactive fan
    if fan_profile.last_active is not None:
        now = datetime.now(timezone.utc)
        last_active = fan_profile.last_active
        if last_active.tzinfo is None:
            last_active = last_active.replace(tzinfo=timezone.utc)
        if (now - last_active).days > 3:
            return StageType.RETENTION

    # Strong upsell signals — fan has spent before or is showing buying intent
    pre_upsell_keywords = [
        "thinking about you", "cant stop thinking", "obsessed",
        "you are my favorite", "i would do anything", "i would pay",
        "worth it", "you deserve", "spoil you", "treat you",
        "wish i could", "i want more", "what else do you offer",
    ]
    if any(phrase in recent_text for phrase in pre_upsell_keywords):
        return StageType.PRE_UPSELL

    # Flirting
    flirting_keywords = [
        "sexy", "hot", "gorgeous", "want you", "thinking about",
        "miss you", "turn on", "hard for", "beautiful", "stunning",
        "perfect", "dream girl", "my type",
    ]
    if any(phrase in recent_text for phrase in flirting_keywords):
        if message_count < 10:
            return StageType.FLIRTING
        return StageType.PRE_UPSELL

    if message_count < 6:
        return StageType.WARMING_UP
    if message_count < 15:
        return StageType.FLIRTING

    return StageType.PRE_UPSELL
