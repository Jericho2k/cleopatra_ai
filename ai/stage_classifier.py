"""Conversation stage classifier.

Pure logic only — no I/O, DB, or API calls.
"""

from datetime import datetime, timezone

from models.schemas import Fan, Message, StageType


def classify_stage(
    conversation_history: list[Message],
    fan_profile: Fan,
) -> StageType:
    """Classify the current conversation stage based on history and fan profile."""

    # 1. High-value fans by total spend
    if fan_profile.total_spent > 500:
        return StageType.HIGH_VALUE

    message_count = len(conversation_history)

    # 2. Collect the text of the last 10 messages into one lowercase string
    recent_messages = conversation_history[-10:]
    recent_text = " ".join(m.content for m in recent_messages).lower()

    # 3. Objection keywords
    objection_keywords = [
        "too expensive",
        "too much",
        "cant afford",
        "can't afford",
        "not worth",
        "no thanks",
        "cheaper",
    ]
    if any(phrase in recent_text for phrase in objection_keywords):
        return StageType.OBJECTION

    # 4. Active upsell keywords
    upsell_keywords = [
        "ppv",
        "custom",
        "special content",
        "just for you",
        "exclusive",
    ]
    if any(phrase in recent_text for phrase in upsell_keywords):
        return StageType.UPSELL_ACTIVE

    # 5. Very early conversations
    if message_count <= 1:
        return StageType.COLD_OPEN

    # 6. Retention if fan has been inactive for more than 3 days
    if fan_profile.last_active is not None:
        now = datetime.now(timezone.utc)
        last_active = fan_profile.last_active
        if last_active.tzinfo is None:
            last_active = last_active.replace(tzinfo=timezone.utc)
        if (now - last_active).days > 3:
            return StageType.RETENTION

    # 7. Flirting keywords, behavior depends on message count
    flirting_keywords = [
        "sexy",
        "hot",
        "gorgeous",
        "want you",
        "thinking about",
        "miss you",
        "turn on",
        "hard for",
    ]
    if any(phrase in recent_text for phrase in flirting_keywords):
        if message_count < 10:
            return StageType.FLIRTING
        return StageType.PRE_UPSELL

    # 8. Early warm-up stage
    if message_count < 6:
        return StageType.WARMING_UP

    # 9. General flirting window
    if message_count < 15:
        return StageType.FLIRTING

    # 10. Default
    return StageType.PRE_UPSELL

