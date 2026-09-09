"""Measure how much of each prompt a provider's prefix cache can actually reuse.

This reports THEORETICAL PREFIX REUSE only: it renders two consecutive turns of
one synthetic conversation and measures the longest byte-identical prefix
between them. It says nothing about what a provider really cached — that is
recorded per call in ``model_usage_events.cache_read_tokens`` by
``services/model_telemetry.py`` and must be read from the database. Never quote
a number from this script as a provider cache saving.

Two prompts are measured:

* the WRITER prompt from ``ai.prompt_builder.build_prompt``;
* the ANALYZER prompt from ``ai.situation_analyzer.build_analyzer_prompt``.

For the analyzer the interesting figure is not the turn-to-turn prefix but the
size of the static system block, because that block is identical for *every*
conversation in the deployment rather than only for consecutive turns of one.

Token figures are ``characters / 4`` estimates. Character counts are exact and
are what the prefix comparison actually uses; the token column exists only to
make the magnitudes readable next to provider billing.

Usage:
    python scripts/measure_prompt_cache.py
    python scripts/measure_prompt_cache.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.generator import flatten_message_content  # noqa: E402
from ai.prompt_builder import build_prompt  # noqa: E402
from ai.situation_analyzer import build_analyzer_prompt  # noqa: E402
from models.schemas import (  # noqa: E402
    ConversationContext,
    Fan,
    Message,
    Persona,
    StageType,
)

CHARS_PER_TOKEN = 4


def approx_tokens(text: str) -> int:
    return round(len(text) / CHARS_PER_TOKEN)


def longest_common_prefix(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def _fan() -> Fan:
    return Fan(
        id="fan-1",
        platform_fan_id="platform-fan-1",
        display_name="Alex",
        total_spent=140,
        spend_tier="warm",
        notes="Works nights, lives in Denver, into gym talk.",
        member_note="Been subscribed four months. Polite, slow replier.",
        model_note="Told him I'm studying part time and live with a roommate.",
        preferences=["gym", "shower sets", "morning texts"],
    )


def _history(turns: int) -> list[Message]:
    lines = [
        ("fan", "hey you around"),
        ("creator", "just got in, long shift"),
        ("fan", "same here honestly, what were you up to"),
        ("creator", "gym then work, i'm wrecked"),
        ("fan", "you always say that and still look unreal"),
        ("creator", "flattery is working btw"),
        ("fan", "good, that was the plan"),
        ("creator", "you're trouble"),
        ("fan", "only a little"),
        ("creator", "mhm sure"),
        ("fan", "what are you doing later"),
        ("creator", "shower then bed probably"),
        ("fan", "that's a nice image"),
        ("creator", "you have no idea"),
        ("fan", "tell me then"),
        ("creator", "maybe if you ask nicer"),
        ("fan", "please"),
        ("creator", "better"),
        ("fan", "so what do i get"),
        ("creator", "patience first"),
    ]
    return [
        Message(role=role, content=content)
        for role, content in lines[:turns]
    ]


# The per-message blocks the intelligence flags turn on. COST-002c is only
# visible when these are populated: with the flags off there is nothing volatile
# sitting in front of the durable profile to move.
def _volatile_blocks(*, turns: int) -> dict:
    return {
        "fan_intelligence": {
            "facts": [
                {"key": "location", "value": "Denver", "confidence": 0.9},
                {"key": "payday", "value": "Friday", "confidence": 0.7},
            ],
            "hard_limits": ["no phone calls"],
        },
        "affordability": {
            "status": "CONSTRAINED",
            "current_available_cents": 2800 + turns,
            "current_limit_cents": 4000,
            "payday_raw": "Friday",
            "confirmed_purchase_count": 3,
            "last_confirmed_purchase_cents": 2500,
        },
        "price_learning": {
            "mode": "REFINING",
            "confidence": "MEDIUM",
            "recommended_floor_cents": 1800 + turns,
            "recommended_ceiling_cents": 4500,
        },
        "conversation_director": {
            "phase": "QUALIFYING",
            "previous_phase": "OPENING",
            "action": "BUILD_TENSION",
            "turns_in_phase": turns,
            "transition_reason": "fan_engaged",
            "recent_actions": ["RESPOND_AND_OPEN", "BUILD_TENSION"],
            "question_due": turns % 2 == 0,
        },
        "session_strategy": {
            "goal": "SELL",
            "phase": "QUALIFYING",
            "next_action": "CONTINUE_CHAT",
            "writer_goal": f"keep momentum at turn {turns}",
            "writer_avoid": ["repeating the shower line"],
            "approved_offer_prices_cents": [2800, 4000],
        },
    }


def _ctx(
    *,
    fan_message: str,
    turns: int,
    situation: dict,
    stage: StageType = StageType.WARMING_UP,
    enriched: bool = False,
) -> ConversationContext:
    extra = _volatile_blocks(turns=turns) if enriched else {}
    return ConversationContext(
        fan_profile=_fan(),
        creator_persona=Persona(),
        creator_name="Sophia",
        creator_legend={"name": "Sophia"},
        conversation_stage=stage,
        conversation_history=_history(turns),
        similar_exchanges=[],
        fan_message=fan_message,
        situation=situation,
        ppv_offers=[],
        sent_ppv=[],
        **extra,
    )


FIRST_SITUATION = {
    "fan_mood": "curious",
    "conversation_energy": "flat",
    "strategic_move": "mirror_warmth",
}
SECOND_SITUATION = {
    "fan_mood": "playful",
    "conversation_energy": "rising",
    "strategic_move": "build_tension",
}


def _writer_turn(ctx: ConversationContext) -> tuple[str, str]:
    messages = build_prompt(ctx)
    return (
        flatten_message_content(messages[0]["content"]),
        flatten_message_content(messages[1]["content"]),
    )


def _analyzer_turn(ctx: ConversationContext) -> tuple[str, str]:
    system, user = build_analyzer_prompt(ctx)
    return flatten_message_content(system), user


def _measure(first: tuple[str, str], second: tuple[str, str]) -> dict:
    first_system, first_user = first
    second_system, second_user = second

    first_whole = first_system + first_user
    second_whole = second_system + second_user
    shared_chars = longest_common_prefix(first_whole, second_whole)

    return {
        "system_chars": len(second_system),
        "system_tokens_approx": approx_tokens(second_system),
        "user_chars": len(second_user),
        "user_tokens_approx": approx_tokens(second_user),
        "total_chars": len(second_whole),
        "total_tokens_approx": approx_tokens(second_whole),
        "shared_prefix_chars": shared_chars,
        "shared_prefix_tokens_approx": round(shared_chars / CHARS_PER_TOKEN),
        "reusable_percent": round(100 * shared_chars / max(len(second_whole), 1), 1),
        "system_fully_reused": shared_chars >= len(second_system),
    }


def collect() -> dict:
    writer_first = _writer_turn(
        _ctx(fan_message="what are you doing later", turns=12, situation=FIRST_SITUATION)
    )
    writer_second = _writer_turn(
        _ctx(fan_message="tell me then", turns=14, situation=SECOND_SITUATION)
    )

    enriched_first = _writer_turn(
        _ctx(
            fan_message="what are you doing later",
            turns=12,
            situation=FIRST_SITUATION,
            enriched=True,
        )
    )
    enriched_second = _writer_turn(
        _ctx(
            fan_message="tell me then",
            turns=14,
            situation=SECOND_SITUATION,
            enriched=True,
        )
    )

    analyzer_first = _analyzer_turn(
        _ctx(fan_message="what are you doing later", turns=12, situation=FIRST_SITUATION)
    )
    analyzer_second = _analyzer_turn(
        _ctx(fan_message="tell me then", turns=14, situation=SECOND_SITUATION)
    )

    # The analyzer's system block is conversation-independent, so it is reusable
    # across every fan in the deployment, not only across consecutive turns.
    other_conversation = _analyzer_turn(
        _ctx(fan_message="totally different fan asking about pricing", turns=4, situation=FIRST_SITUATION)
    )

    writer = _measure(writer_first, writer_second)
    writer_enriched = _measure(enriched_first, enriched_second)
    analyzer = _measure(analyzer_first, analyzer_second)
    analyzer["system_shared_across_conversations"] = (
        analyzer_second[0] == other_conversation[0]
    )
    return {
        "writer": writer,
        "writer_enriched": writer_enriched,
        "analyzer": analyzer,
    }


def _render(results: dict) -> str:
    rows = []
    rows.append("THEORETICAL PREFIX REUSE (synthetic; not a provider cache report)")
    rows.append("")
    header = (
        f"{'prompt':<18}{'system tok':>12}{'total tok':>12}"
        f"{'prefix tok':>12}{'reusable':>10}"
    )
    rows.append(header)
    rows.append("-" * len(header))
    for name in ("writer", "writer_enriched", "analyzer"):
        data = results[name]
        rows.append(
            f"{name:<18}{data['system_tokens_approx']:>12}"
            f"{data['total_tokens_approx']:>12}"
            f"{data['shared_prefix_tokens_approx']:>12}"
            f"{str(data['reusable_percent']) + '%':>10}"
        )
    rows.append("")
    rows.append(
        "analyzer system block identical across unrelated conversations: "
        + str(results["analyzer"]["system_shared_across_conversations"])
    )
    rows.append(
        "writer system block fully inside the shared prefix: "
        + str(results["writer"]["system_fully_reused"])
    )
    rows.append("")
    rows.append(
        "Token counts are characters/4 estimates. Actual provider cache reads "
        "live in model_usage_events.cache_read_tokens."
    )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()

    results = collect()
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print(_render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
