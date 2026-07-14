from ai.prompt_builder import build_prompt
from models.schemas import ConversationContext, Fan, Persona, StageType


def _context(stage: str) -> ConversationContext:
    return ConversationContext(
        fan_message="hey",
        conversation_history=[],
        fan_profile=Fan(id="fan-1", display_name="Alex"),
        creator_persona=Persona(),
        similar_exchanges=[],
        conversation_stage=StageType.WARMING_UP,
        buyer_lifecycle={
            "stage": stage,
            "purchase_count": 2,
            "total_spent_cents": 12_000,
            "flags": {},
        },
    )


def test_repeat_buyer_context_reaches_writer_prompt():
    prompt = build_prompt(_context("REPEAT_BUYER"))
    user_text = prompt[-1]["content"]
    assert "Buyer lifecycle: REPEAT_BUYER" in user_text
    assert "confirmed purchases: 2" in user_text


def test_first_purchase_prospect_guidance_avoids_overwhelming_offer():
    prompt = build_prompt(_context("FIRST_PURCHASE_PROSPECT"))
    user_text = prompt[-1]["content"]
    assert "reduce friction" in user_text.lower()
