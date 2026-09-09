"""The writer prompt must keep its cacheable prefix stable between turns.

Provider-side prefix caching only pays off when consecutive calls in one
conversation share a long identical prefix. These tests pin the ordering that
makes that true, so a future edit cannot quietly move a per-message value ahead
of the durable context.
"""

from __future__ import annotations

from ai.generator import flatten_message_content
from ai.prompt_builder import build_prompt
from models.schemas import (
    ConversationContext,
    Fan,
    Message,
    Persona,
    StageType,
)


def _fan():
    return Fan(
        id="fan-1",
        creator_id="creator-1",
        platform_fan_id="platform-fan-1",
        display_name="Alex",
        total_spent=0,
        spend_tier="cold",
    )


def _ctx(*, fan_message: str, history: list[Message], stage=StageType.WARMING_UP):
    return ConversationContext(
        fan_profile=_fan(),
        creator_persona=Persona(),
        creator_name="Sophia",
        creator_legend={"name": "Sophia"},
        conversation_stage=stage,
        conversation_history=history,
        similar_exchanges=[],
        fan_message=fan_message,
        situation={
            "fan_mood": "curious",
            "conversation_energy": "medium",
            "strategic_move": "build connection",
        },
        ppv_offers=[],
        sent_ppv=[],
    )


def _rendered(ctx) -> tuple[str, str]:
    messages = build_prompt(ctx)
    return (
        flatten_message_content(messages[0]["content"]),
        flatten_message_content(messages[1]["content"]),
    )


def _common_prefix(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def test_system_prompt_is_identical_across_turns_of_one_conversation():
    first_system, _ = _rendered(_ctx(fan_message="hey", history=[]))
    second_system, _ = _rendered(
        _ctx(
            fan_message="what are you up to",
            history=[Message(role="fan", content="hey")],
        )
    )

    assert first_system == second_system
    assert len(first_system) > 1000


def test_stable_system_prefix_precedes_the_volatile_stage_addendum():
    system, _ = _rendered(
        _ctx(fan_message="hey", history=[], stage=StageType.UPSELL_ACTIVE)
    )
    baseline, _ = _rendered(_ctx(fan_message="hey", history=[]))

    # The paid-interaction addendum is appended, so the whole ordinary system
    # prompt survives untouched in front of it.
    assert system.startswith(baseline)
    assert "intimate paid interaction" in system[len(baseline):]


def test_durable_fan_context_precedes_the_volatile_tail():
    _, user = _rendered(_ctx(fan_message="hey", history=[]))

    knowledge = user.index("WHAT YOU KNOW ABOUT THIS FAN:")
    stage = user.index("CONVERSATION STAGE:")
    situation = user.index("CURRENT SITUATION:")
    latest = user.index("Fan just said:")
    instruction = user.index("Write 3 reply options")

    assert knowledge < stage < situation < latest < instruction


def test_transcript_sits_before_per_message_state():
    _, user = _rendered(
        _ctx(
            fan_message="what are you up to",
            history=[
                Message(role="fan", content="hey"),
                Message(role="creator", content="hey you"),
            ],
        )
    )

    assert user.index("RECENT CONVERSATION") < user.index("CONVERSATION STAGE:")
    assert user.index("RECENT CONVERSATION") < user.index("Fan just said:")


def test_changed_stage_and_mood_no_longer_evict_the_transcript_from_the_prefix():
    """The reason the volatile block moved behind the transcript.

    The analyzer recomputes stage, mood, energy and strategy on every inbound
    message. With those values in front of the transcript, a single mood flip
    truncated the shared prefix before any conversation history, so the whole
    transcript had to be re-read uncached. With them behind it, the earlier
    transcript stays inside the prefix.
    """

    history = [
        Message(role="fan", content="hey"),
        Message(role="creator", content="hey you"),
    ]
    first_system, first_user = _rendered(
        _ctx(fan_message="what are you up to", history=history)
    )

    later = _ctx(
        fan_message="nice, and tomorrow?",
        history=[
            *history,
            Message(role="fan", content="what are you up to"),
            Message(role="creator", content="just got home, kinda lazy today"),
        ],
    )
    later.situation = {
        "fan_mood": "playful",
        "conversation_energy": "high",
        "strategic_move": "build tension",
    }
    second_system, second_user = _rendered(later)

    shared = _common_prefix(
        first_system + first_user,
        second_system + second_user,
    )

    # Everything up to and including the earlier transcript survives.
    assert shared > len(first_system)
    boundary = (first_system + first_user)[:shared]
    assert "WHAT YOU KNOW ABOUT THIS FAN:" in boundary
    assert "RECENT CONVERSATION" in boundary
    assert "Alex: hey" in boundary

    # And the per-message state is outside it, which is the point.
    assert "CURRENT SITUATION:" not in boundary
    assert "Fan just said:" not in boundary


def test_no_anthropic_cache_markers_reach_an_openai_compatible_transport():
    messages = build_prompt(_ctx(fan_message="hey", history=[]))

    # The blocks exist for the Anthropic route, which consumes them natively.
    assert isinstance(messages[0]["content"], list)
    assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    # Everything else receives plain text, never a serialized marker.
    flattened = flatten_message_content(messages[0]["content"])
    assert "cache_control" not in flattened
    assert "ephemeral" not in flattened
    assert not flattened.startswith("[{")


# --- COST-002c: durable fan context in front, recomputed state behind --------


def _enriched_ctx(*, fan_message: str, history: list[Message], floor_cents: int):
    ctx = _ctx(fan_message=fan_message, history=history)
    ctx.fan_profile.notes = "Works nights, lives in Denver."
    ctx.fan_profile.member_note = "Subscribed four months."
    ctx.affordability = {
        "status": "CONSTRAINED",
        "current_available_cents": floor_cents,
        "payday_raw": "Friday",
    }
    ctx.price_learning = {
        "mode": "REFINING",
        "confidence": "MEDIUM",
        "recommended_floor_cents": floor_cents,
    }
    ctx.conversation_director = {
        "phase": "QUALIFYING",
        "action": "BUILD_TENSION",
        "turns_in_phase": len(history),
    }
    ctx.session_strategy = {
        "goal": "SELL",
        "phase": "QUALIFYING",
        "next_action": "CONTINUE_CHAT",
    }
    return ctx


def test_recomputed_state_sits_behind_the_durable_profile_and_transcript():
    _, user = _rendered(
        _enriched_ctx(
            fan_message="tell me then",
            history=[
                Message(role="fan", content="hey"),
                Message(role="creator", content="hey you"),
            ],
            floor_cents=2800,
        )
    )

    durable = user.index("Summary: Works nights")
    transcript = user.index("RECENT CONVERSATION")
    affordability = user.index("COMMERCIAL AFFORDABILITY")
    price = user.index("PRICE LEARNING")
    director = user.index("CONVERSATION DIRECTOR")
    strategy = user.index("ADAPTIVE SESSION STRATEGY")
    stage = user.index("CONVERSATION STAGE:")
    latest = user.index("Fan just said:")

    # Durable first, transcript next, per-message state after it, and the whole
    # volatile group still ahead of stage and the newest message.
    assert durable < transcript < affordability < price < director < strategy
    assert strategy < stage < latest


def test_a_recomputed_price_band_no_longer_evicts_the_transcript():
    """The regression COST-002c describes, expressed as a prefix measurement.

    With the per-message blocks in front, changing one recomputed cents value
    truncated the shared prefix before the durable profile and before any
    conversation history. Behind the transcript, both survive.
    """

    history = [
        Message(role="fan", content="hey"),
        Message(role="creator", content="hey you"),
    ]
    first_system, first_user = _rendered(
        _enriched_ctx(fan_message="what now", history=history, floor_cents=2800)
    )
    second_system, second_user = _rendered(
        _enriched_ctx(fan_message="what now", history=history, floor_cents=3400)
    )

    shared = _common_prefix(
        first_system + first_user,
        second_system + second_user,
    )
    boundary = (first_system + first_user)[:shared]

    assert "Summary: Works nights" in boundary
    assert "RECENT CONVERSATION" in boundary
    assert "Alex: hey" in boundary
    # The changed value itself is of course outside the shared prefix.
    assert "PRICE LEARNING" not in boundary


def test_expression_calibration_stays_durable_when_nothing_recomputes_it():
    """Flags-off deployments must not lose prefix they already had.

    With no director and no session strategy the calibration block is one
    constant paragraph, so moving it behind the transcript would have pushed
    ~700 characters out of the shared prefix for the default configuration.
    """

    _, user = _rendered(
        _ctx(
            fan_message="what are you up to",
            history=[Message(role="fan", content="hey")],
        )
    )

    assert user.index("EXPRESSION CALIBRATION") < user.index("RECENT CONVERSATION")
    assert user.index("EXPRESSION CALIBRATION") < user.index("CONVERSATION STAGE:")


def test_expression_calibration_moves_behind_the_transcript_once_it_varies():
    _, user = _rendered(
        _enriched_ctx(
            fan_message="tell me then",
            history=[Message(role="fan", content="hey")],
            floor_cents=2800,
        )
    )

    assert user.index("RECENT CONVERSATION") < user.index("EXPRESSION CALIBRATION")
