from ai.prompt_builder import build_prompt
from models.schemas import (
    ConversationContext,
    Fan,
    Persona,
    StageType,
)


def test_prompt_builder_accepts_numeric_creator_legend_values():
    ctx = ConversationContext(
        fan_profile=Fan(
            id="fan-1",
            creator_id="creator-1",
            platform_fan_id="platform-fan-1",
            display_name="Alex",
            total_spent=0,
            spend_tier="cold",
        ),
        creator_persona=Persona(),
        creator_name="Sophia",
        creator_legend={
            "name": "Sophia",
            "age": 23,
            "origin": "Germany",
            "other": [123, "likes travelling"],
        },
        conversation_stage=StageType.WARMING_UP,
        conversation_history=[],
        similar_exchanges=[],
        fan_message="hey",
        situation={},
        ppv_offers=[],
        sent_ppv=[],
    )

    messages = build_prompt(ctx)

    assert messages
    assert "23" in str(messages)
    assert "123" in str(messages)