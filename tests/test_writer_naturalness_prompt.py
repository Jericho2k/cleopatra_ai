from ai.prompt_builder import build_prompt
from models.schemas import ConversationContext, Fan, Persona, StageType


def _prompt_text() -> str:
    ctx = ConversationContext(
        fan_profile=Fan(
            id="fan-naturalness",
            display_name="Tony",
            total_spent=0,
            spend_tier="cold",
        ),
        creator_persona=Persona(),
        creator_name="Eliz",
        conversation_stage=StageType.WARMING_UP,
        conversation_history=[],
        similar_exchanges=[],
        fan_message="just came across your page today",
        situation={"strategic_move": "acknowledge_compliment_and_redirect"},
        ppv_offers=[],
        sent_ppv=[],
    )
    messages = build_prompt(ctx)
    chunks: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict)
            )
    return "\n".join(chunks)


def test_option_one_is_explicitly_the_plain_auto_send_default():
    prompt = _prompt_text().lower()

    assert "option 1 is the safest, plainest, most natural response" in prompt
    assert "option 1 may be auto-sent" in prompt
    assert "directly acknowledge the latest message" in prompt


def test_prompt_prefers_specific_plain_text_over_performed_banter():
    prompt = _prompt_text().lower()

    assert "prefer the obvious, ordinary human wording over a clever line" in prompt
    assert "if it sounds authored rather than typed, simplify it" in prompt
    assert "if the line could fit many unrelated conversations" in prompt
    assert "stock banter" in prompt


def test_old_stock_line_examples_are_not_seeded_into_the_prompt():
    prompt = _prompt_text().lower()

    assert '"lucky you"' not in prompt
    assert '"bold claim"' not in prompt
    assert '"you\'re trouble, aren\'t you"' not in prompt
