"""Historical compaction: evidence discipline, merge precedence, resumability.

Compaction reads an archive with a cheap model and writes into the SAME
fan-fact record live extraction writes to. That makes two things load-bearing:
the model's proposals must be verifiable against the actual source message, and
old evidence must not be able to overturn current knowledge.
"""

import asyncio
import json

import pytest

from models.fan_intelligence import (
    FactCategory,
    FactCertainty,
    FactStatus,
    MergeAction,
    ProposedHistoricalObservation,
    SOURCE_TYPE_HISTORICAL_MESSAGE,
    ValidatedObservation,
)
from services import fan_history_memory
from services.fan_intelligence import plan_fact_merge
from services.fan_history_memory import (
    _rows_after_cursor,
    chunk_messages,
    merge_continuity,
    parse_historical_payload,
    render_chunk,
    validate_historical_observation,
)
from tests.fake_supabase import FakeSupabase


CREATOR = "creator-1"
FAN = "fan-1"


def _row(message_id: str, role: str, content: str, sent_at: str = "2026-01-01T00:00:00+00:00"):
    return {
        "id": f"local-{message_id}",
        "fansly_message_id": message_id,
        "role": role,
        "content": content,
        "sent_at": sent_at,
    }


def _proposed(**overrides) -> ProposedHistoricalObservation:
    payload = {
        "category": FactCategory.COMMERCIAL,
        "fact_key": "payday",
        "value": "Friday",
        "certainty": FactCertainty.EXPLICIT,
        "confidence": 0.95,
        "evidence": "i get paid friday",
        "source_message_id": "m1",
    }
    payload.update(overrides)
    return ProposedHistoricalObservation(**payload)


CHUNK = {
    "m1": _row("m1", "fan", "hey i get paid friday so maybe then"),
    "m2": _row("m2", "creator", "I'm a California girl, born and raised"),
}


# --- evidence validation ----------------------------------------------------


def test_a_quoted_fan_fact_is_accepted():
    validated = validate_historical_observation(_proposed(), chunk_by_id=CHUNK)
    assert validated is not None
    assert validated.fact_key == "payday"
    assert validated.source_type == SOURCE_TYPE_HISTORICAL_MESSAGE


def test_a_fact_whose_quote_is_not_in_the_message_is_rejected():
    """An unverifiable claim is not evidence, however confident the model is."""
    assert (
        validate_historical_observation(
            _proposed(evidence="i get paid on the 15th"), chunk_by_id=CHUNK
        )
        is None
    )


def test_a_fact_attributed_to_the_creator_is_rejected():
    """The failure mode this check exists for.

    'I'm a California girl' is the CREATOR speaking. Recording it as the fan's
    location is how a backfill silently corrupts a CRM.
    """
    assert (
        validate_historical_observation(
            _proposed(
                fact_key="location",
                category=FactCategory.IDENTITY,
                value="California",
                evidence="I'm a California girl",
                source_message_id="m2",
            ),
            chunk_by_id=CHUNK,
        )
        is None
    )


def test_a_fact_naming_a_message_outside_the_chunk_is_rejected():
    assert (
        validate_historical_observation(
            _proposed(source_message_id="m9999"), chunk_by_id=CHUNK
        )
        is None
    )


def test_explicit_only_keys_still_require_explicit_wording():
    """Historical extraction inherits every live rule, not a relaxed copy."""
    assert (
        validate_historical_observation(
            _proposed(certainty=FactCertainty.STRONG_INFERENCE, confidence=0.99),
            chunk_by_id=CHUNK,
        )
        is None
    )


def test_money_is_parsed_from_the_quote_not_from_the_model_s_number():
    chunk = {"m1": _row("m1", "fan", "i could do $25 for that one")}
    validated = validate_historical_observation(
        _proposed(
            fact_key="counteroffer_cents",
            value=9_900,  # the model's claim, which is wrong
            evidence="i could do $25",
        ),
        chunk_by_id=chunk,
    )
    assert validated is not None
    assert validated.value_json == 2500


def test_a_disallowed_fact_key_is_rejected():
    assert (
        validate_historical_observation(
            _proposed(fact_key="net_worth", category=FactCategory.IDENTITY),
            chunk_by_id=CHUNK,
        )
        is None
    )


# --- merge precedence -------------------------------------------------------


def _validated(value: str = "Chicago") -> ValidatedObservation:
    return ValidatedObservation(
        category=FactCategory.IDENTITY,
        fact_key="location",
        value_json=value,
        normalized_value=json.dumps(value).casefold(),
        certainty=FactCertainty.EXPLICIT,
        confidence=0.9,
        evidence_text="i live in chicago",
        source_type=SOURCE_TYPE_HISTORICAL_MESSAGE,
    )


def _existing(status: str, value: str = "Austin") -> list[dict]:
    return [
        {
            "id": "fact-1",
            "status": status,
            "is_active": True,
            "normalized_value": json.dumps(value).casefold(),
            "confidence": 0.9,
            "confirmation_count": 2,
        }
    ]


def test_history_never_overturns_a_currently_explicit_fact():
    """A three-year-old 'I live in Chicago' is not news about where he lives.

    Live evidence in the same position declares a CONFLICT and deactivates the
    old value. Historical evidence stands down instead — the observation is
    still recorded, only the authority to overturn is withheld.
    """
    plan = plan_fact_merge(
        _existing(FactStatus.EXPLICIT.value), _validated(), historical=True
    )
    assert plan.action == MergeAction.IGNORE

    live = plan_fact_merge(_existing(FactStatus.EXPLICIT.value), _validated())
    assert live.action == MergeAction.CONFLICT


def test_history_never_overturns_a_confirmed_fact():
    plan = plan_fact_merge(
        _existing(FactStatus.CONFIRMED.value), _validated(), historical=True
    )
    assert plan.action == MergeAction.IGNORE


def test_history_may_still_replace_a_mere_guess():
    """Standing down is about strength, not about being old."""
    plan = plan_fact_merge(
        _existing(FactStatus.INFERRED.value), _validated(), historical=True
    )
    assert plan.action == MergeAction.REPLACE_INFERRED


def test_history_may_still_create_a_fact_nobody_knew():
    assert plan_fact_merge([], _validated(), historical=True).action == MergeAction.CREATE


def test_history_may_still_reinforce_the_same_value():
    same = _existing(FactStatus.EXPLICIT.value, value="Chicago")
    plan = plan_fact_merge(same, _validated("Chicago"), historical=True)
    assert plan.action == MergeAction.REINFORCE


# --- chunking and resumability ---------------------------------------------


def test_history_is_chunked_in_bounded_chronological_pieces():
    rows = [_row(f"m{index}", "fan", "x") for index in range(95)]
    chunks = chunk_messages(rows, size=40)
    assert [len(chunk) for chunk in chunks] == [40, 40, 15]
    assert chunks[0][0]["fansly_message_id"] == "m0"
    assert chunks[-1][-1]["fansly_message_id"] == "m94"


def test_the_extraction_cursor_resumes_after_the_last_processed_message():
    rows = [
        _row("m1", "fan", "a", "2026-01-01T00:00:00+00:00"),
        _row("m2", "fan", "b", "2026-01-02T00:00:00+00:00"),
        _row("m3", "fan", "c", "2026-01-03T00:00:00+00:00"),
    ]
    assert [
        row["fansly_message_id"]
        for row in _rows_after_cursor(
            rows, cursor_sent_at="2026-01-02T00:00:00+00:00", cursor_message_id="m2"
        )
    ] == ["m3"]


def test_messages_sharing_a_timestamp_are_neither_skipped_nor_repeated():
    """Chat timestamps collide constantly; a sent_at-only cursor loses messages."""
    same = "2026-01-01T00:00:00+00:00"
    rows = [
        _row("m1", "fan", "a", same),
        _row("m2", "fan", "b", same),
        _row("m3", "fan", "c", same),
    ]
    resumed = _rows_after_cursor(rows, cursor_sent_at=same, cursor_message_id="m2")
    assert [row["fansly_message_id"] for row in resumed] == ["m3"]


def test_no_cursor_means_start_at_the_beginning():
    rows = [_row("m1", "fan", "a")]
    assert _rows_after_cursor(rows, cursor_sent_at=None, cursor_message_id=None) == rows


# --- rendering --------------------------------------------------------------


def test_the_chunk_labels_every_message_with_its_id_and_speaker():
    rendered = render_chunk([CHUNK["m1"], CHUNK["m2"]])
    assert "[m1] FAN: hey i get paid friday so maybe then" in rendered
    assert "[m2] CREATOR: I'm a California girl, born and raised" in rendered


def test_an_attachment_only_message_is_described_not_dropped():
    row = _row("m5", "fan", "")
    row["media_context"] = {"attachments": [{"contentId": "x"}]}
    assert "[sent 1 attachment(s), no text]" in render_chunk([row])


def test_a_long_message_cannot_blow_out_the_chunk_budget():
    rendered = render_chunk([_row("m6", "fan", "x" * 5_000)])
    assert len(rendered) < fan_history_memory.HISTORY_MESSAGE_CHARS + 50


# --- response parsing -------------------------------------------------------


def test_json_wrapped_in_prose_is_still_read():
    envelope = parse_historical_payload(
        'Sure! ```json\n{"observations": [], "ongoing_topics": ["gym"]}\n``` done'
    )
    assert envelope.ongoing_topics == ["gym"]


def test_a_bare_list_is_accepted_as_observations():
    envelope = parse_historical_payload("[]")
    assert envelope.observations == []


def test_an_empty_response_raises_rather_than_silently_succeeding():
    with pytest.raises(ValueError):
        parse_historical_payload("   ")


# --- continuity state -------------------------------------------------------


def test_continuity_accumulates_and_stays_compact():
    from models.fan_intelligence import HistoricalExtractionEnvelope

    state = None
    for index in range(20):
        state = merge_continuity(
            state,
            HistoricalExtractionEnvelope(
                ongoing_topics=[f"topic {index}"],
                commercial_context=[f"offer {index}"],
                relationship_summary=f"summary {index}",
            ),
            chunk_last_sent_at=f"2026-01-{index + 1:02d}T00:00:00+00:00",
        )
    assert len(state["ongoing_topics"]) == fan_history_memory.MAX_ONGOING_TOPICS
    assert len(state["commercial_context"]) == fan_history_memory.MAX_COMMERCIAL_CONTEXT
    # History is read chronologically, so the newest chunk's picture wins.
    assert state["relationship_summary"] == "summary 19"
    assert state["through"] == "2026-01-20T00:00:00+00:00"


def test_continuity_does_not_repeat_a_topic():
    from models.fan_intelligence import HistoricalExtractionEnvelope

    first = merge_continuity(None, HistoricalExtractionEnvelope(ongoing_topics=["Gym"]))
    second = merge_continuity(
        first, HistoricalExtractionEnvelope(ongoing_topics=["gym", "new job"])
    )
    assert second["ongoing_topics"] == ["gym", "new job"]


def test_continuity_is_rendered_as_older_context_not_as_fact():
    from ai.prompt_builder import _render_fan_intelligence

    rendered = _render_fan_intelligence(
        {
            "facts": [],
            "history_continuity": {
                "ongoing_topics": ["his new job"],
                "relationship_summary": "Long-running rapport.",
                "history_fully_paged": True,
            },
        }
    )
    assert "Earlier ongoing topics with him: his new job" in rendered
    assert "older context, not current fact" in rendered


def test_an_absent_history_renders_nothing():
    from ai.prompt_builder import _render_fan_intelligence

    assert _render_fan_intelligence({"facts": [], "history_continuity": {}}) == ""


# --- the gate ---------------------------------------------------------------


def test_compaction_is_off_unless_both_switches_are_on(monkeypatch):
    """History cannot grant fan intelligence to a deployment that declined it."""
    monkeypatch.setenv("HISTORY_EXTRACTION_ENABLED", "true")
    monkeypatch.setenv("FAN_INTELLIGENCE_ENABLED", "false")
    assert fan_history_memory.history_extraction_enabled() is False

    monkeypatch.setenv("FAN_INTELLIGENCE_ENABLED", "true")
    assert fan_history_memory.history_extraction_enabled() is True

    monkeypatch.setenv("HISTORY_EXTRACTION_ENABLED", "false")
    assert fan_history_memory.history_extraction_enabled() is False


def test_disabled_compaction_makes_no_model_call(monkeypatch):
    monkeypatch.setenv("HISTORY_EXTRACTION_ENABLED", "false")

    async def _explode(*args, **kwargs):
        raise AssertionError("compaction must not call a model while disabled")

    monkeypatch.setattr(fan_history_memory, "complete", _explode)
    result = asyncio.run(
        fan_history_memory.compact_fan_history(creator_id=CREATOR, fan_id=FAN)
    )
    assert result["status"] == "disabled"


# --- the model choice -------------------------------------------------------


def test_historical_compaction_runs_on_the_cheap_bulk_model():
    """A background archive reader, explicitly not the conversational writer."""
    from ai.stack_profiles import (
        PROFILES,
        STAGE_HISTORY_EXTRACTION,
        STAGE_WRITER_DEFAULT,
    )

    for profile in PROFILES.values():
        spec = profile.stages[STAGE_HISTORY_EXTRACTION]
        assert spec.resolved_primary() == ("together", "zai-org/GLM-5.3-Flash")
        # Kimi is untouched as the conversational writer.
        assert profile.stages[STAGE_WRITER_DEFAULT].resolved_primary()[1] != (
            "zai-org/GLM-5.3-Flash"
        )


def test_live_fan_intelligence_extraction_is_deliberately_unchanged():
    from ai.stack_profiles import PROFILES, STAGE_FAN_INTELLIGENCE

    for profile in PROFILES.values():
        assert profile.stages[STAGE_FAN_INTELLIGENCE].resolved_primary() == (
            "together",
            "openai/gpt-oss-120b",
        )


# --- end-to-end compaction --------------------------------------------------


def test_compaction_stores_facts_and_advances_the_cursor(monkeypatch):
    fake = FakeSupabase(
        {
            "messages": [
                {
                    "id": "local-m1",
                    "fan_id": FAN,
                    "creator_id": CREATOR,
                    "role": "fan",
                    "content": "i get paid friday",
                    "fansly_message_id": "m1",
                    "sent_at": "2026-01-01T00:00:00+00:00",
                }
            ],
            "fan_history_backfill": [
                {"id": "b1", "creator_id": CREATOR, "fan_id": FAN, "status": "paused"}
            ],
            "fan_fact_observations": [],
            "fan_facts": [],
        }
    )
    for module in (
        "db.fan_history_queries",
        "db.fan_intelligence_queries",
        "services.fan_history",
        "core.pagination",
    ):
        monkeypatch.setattr(
            __import__(module, fromlist=["get_supabase"]),
            "get_supabase",
            lambda: fake,
            raising=False,
        )
    monkeypatch.setenv("FAN_INTELLIGENCE_ENABLED", "true")
    monkeypatch.setenv("HISTORY_EXTRACTION_ENABLED", "true")

    class _Result:
        text = json.dumps(
            {
                "observations": [
                    {
                        "category": "commercial",
                        "fact_key": "payday",
                        "value": "Friday",
                        "certainty": "explicit",
                        "confidence": 0.97,
                        "evidence": "i get paid friday",
                        "source_message_id": "m1",
                    },
                    {
                        # Unverifiable: rejected before it can reach a fact.
                        "category": "identity",
                        "fact_key": "occupation",
                        "value": "surgeon",
                        "certainty": "explicit",
                        "confidence": 0.99,
                        "evidence": "i am a surgeon",
                        "source_message_id": "m1",
                    },
                ],
                "ongoing_topics": ["payday timing"],
                "commercial_context": [],
                "relationship_summary": "He was waiting on payday.",
            }
        )

    async def _complete(*args, **kwargs):
        return _Result()

    async def _record(*args, **kwargs):
        return None

    monkeypatch.setattr(fan_history_memory, "complete", _complete)
    monkeypatch.setattr(fan_history_memory, "record_model_result", _record)

    result = asyncio.run(
        fan_history_memory.compact_fan_history(creator_id=CREATOR, fan_id=FAN)
    )
    assert result["proposed"] == 2
    assert result["accepted"] == 1

    facts = fake.tables["fan_facts"]
    assert [fact["fact_key"] for fact in facts] == ["payday"]
    assert facts[0]["source_type"] == SOURCE_TYPE_HISTORICAL_MESSAGE

    state = fake.tables["fan_history_backfill"][0]
    assert state["extraction_cursor_message_id"] == "m1"
    assert state["continuity"]["ongoing_topics"] == ["payday timing"]


def test_compaction_never_calls_the_platform(monkeypatch):
    """Compaction reads local rows. It cannot cost a provider credit."""
    import services.apifansly as apifansly

    for name in ("list_chat_messages", "send_message", "download_media"):
        monkeypatch.setattr(
            apifansly,
            name,
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"compaction must not call {name}")
            ),
        )
    monkeypatch.setenv("HISTORY_EXTRACTION_ENABLED", "false")
    asyncio.run(fan_history_memory.compact_fan_history(creator_id=CREATOR, fan_id=FAN))
    assert apifansly.usage_snapshot()["calls"] == 0
