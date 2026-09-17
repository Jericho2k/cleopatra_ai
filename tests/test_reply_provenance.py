"""Sprint 0 — every visible reply is attributable to what produced it.

``docs/autonomy_architecture_review.md`` §6 step 1. These tests hold the four
claims the rest of the programme depends on:

1. The running commit and the flags that are on are recorded, and no credential
   can reach the record.
2. The model that ANSWERED is recorded, not the one that was asked for
   (finding H). A fallback must be visible as a fallback.
3. Everything done to the writer's text after generation is recorded, so a reply
   the code rewrote is not attributed to the model as though it were the
   model's words.
4. A delivery is reported as accepted only when the platform returned a receipt.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from ai import generator
from ai.generation_trace import GenerationTrace
from core import build_info
from models.model_runtime import ModelResult, ModelTarget, ModelUsage
from models.schemas import Persona
from services.reply_provenance import (
    PIPELINE_ASSISTED,
    PIPELINE_AUTO,
    PROVENANCE_KEY,
    TRANSFORM_INVENTORY_REPAIR,
    TRANSFORM_OPERATOR_EDIT,
    ReplyProvenance,
    SuggestionProvenanceStore,
    merge_provenance,
    provenance_of,
)


def _target(model: str, provider: str = "together") -> ModelTarget:
    return ModelTarget(
        name=model,
        provider=provider,
        model=model,
        base_url="https://example.invalid/v1",
        api_key_env="TOGETHER_API_KEY",
    )


# --- 1: the build snapshot ---------------------------------------------------


def test_observed_flags_can_never_carry_a_credential():
    """The allowlist is the only thing standing between a jsonb column and a key."""
    for name in build_info.OBSERVED_FLAGS:
        for marker in ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL"):
            assert marker not in name, f"{name} looks like a credential"


def test_widening_the_allowlist_to_a_credential_fails_loudly():
    with pytest.raises(RuntimeError, match="must not contain credentials"):
        build_info._assert_no_secrets(("APP_ENV", "ANTHROPIC_API_KEY"))


def test_active_flags_reports_only_the_allowlist(monkeypatch):
    monkeypatch.setenv("SOME_UNRELATED_VARIABLE", "leak-me")
    flags = build_info.active_flags()
    assert set(flags) == set(build_info.OBSERVED_FLAGS)
    assert "SOME_UNRELATED_VARIABLE" not in flags


def test_unset_and_empty_are_not_the_same_flag_value(monkeypatch):
    """SEC-004 is exactly the bug where those two were conflated."""
    monkeypatch.delenv("COMMERCIAL_LAYER_ENABLED", raising=False)
    unset = build_info.active_flags()["COMMERCIAL_LAYER_ENABLED"]
    monkeypatch.setenv("COMMERCIAL_LAYER_ENABLED", "")
    empty = build_info.active_flags()["COMMERCIAL_LAYER_ENABLED"]
    assert unset == build_info.UNSET
    assert empty == ""
    assert unset != empty


def test_flags_digest_moves_when_behaviour_changes(monkeypatch):
    monkeypatch.setenv("CONVERSATION_DIRECTOR_ENABLED", "false")
    before = build_info.flags_digest()
    monkeypatch.setenv("CONVERSATION_DIRECTOR_ENABLED", "true")
    after = build_info.flags_digest()
    assert before != after, "two configurations must not share a digest"
    assert len(after) == 8


def test_build_sha_prefers_the_explicit_deployment_variable(monkeypatch):
    build_info.build_sha.cache_clear()
    monkeypatch.setenv("CLEOPATRA_BUILD_SHA", "deadbeefcafe1234")
    try:
        assert build_info.build_sha() == "deadbeefcafe1234"
        assert build_info.short_sha() == "deadbeefcafe"
    finally:
        build_info.build_sha.cache_clear()


def test_build_sha_falls_back_to_the_working_tree(monkeypatch):
    """A local run and the test suite still have to answer the question."""
    build_info.build_sha.cache_clear()
    for name in build_info._SHA_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    try:
        resolved = build_info.build_sha()
    finally:
        build_info.build_sha.cache_clear()
    # Either a real SHA from .git, or the honest admission. Never a fabrication.
    assert resolved == "unknown" or len(resolved) >= 7


def test_snapshot_on_a_message_carries_the_digest_not_the_flag_values():
    compact = build_info.build_snapshot(include_flags=False)
    assert set(compact) == {"sha", "env", "flags_digest"}
    assert "flags" in build_info.build_snapshot()


# --- 2: the model that actually answered -------------------------------------


def _run_generate(monkeypatch, *, fail_primary: bool, trace: GenerationTrace):
    primary = _target("moonshotai/kimi-k2.6", provider="openrouter")
    fallback = _target("Qwen/Qwen3.7-Plus")

    async def fake_complete(target, **kwargs):
        if fail_primary and target.model == primary.model:
            raise RuntimeError("upstream exploded")
        return ModelResult(
            text='["one", "two", "three"]',
            target=target,
            usage=ModelUsage(input_tokens=10, output_tokens=5),
            latency_ms=1,
        )

    async def noop(*_a, **_k):
        return None

    monkeypatch.setattr(generator, "complete", fake_complete)
    monkeypatch.setattr(generator, "record_model_result", noop)
    monkeypatch.setattr(generator, "record_model_failure", noop)
    monkeypatch.setattr(generator, "record_writer_recovery_outcome", noop)
    monkeypatch.setattr(generator, "_sleep", noop)

    return asyncio.run(
        generator.generate_replies(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            Persona(avg_message_length="short"),
            target_override=primary,
            fallback_target_override=fallback,
            profile_id="cleo_v3",
            trace=trace,
        )
    )


def test_trace_names_the_requested_model_when_it_answers(monkeypatch):
    trace = GenerationTrace()
    replies = _run_generate(monkeypatch, fail_primary=False, trace=trace)

    assert replies == ["one", "two", "three"]
    assert trace.succeeded
    assert trace.model == "moonshotai/kimi-k2.6"
    assert trace.requested_model == "moonshotai/kimi-k2.6"
    assert trace.served_by_requested_model is True
    assert trace.profile == "cleo_v3"


def test_trace_names_the_fallback_when_the_fallback_answers(monkeypatch):
    """Finding H: this is the case the old marker recorded as the primary."""
    trace = GenerationTrace()
    replies = _run_generate(monkeypatch, fail_primary=True, trace=trace)

    assert replies == ["one", "two", "three"]
    assert trace.requested_model == "moonshotai/kimi-k2.6"
    assert trace.model == "Qwen/Qwen3.7-Plus"
    assert trace.served_by_requested_model is False
    assert trace.role == "fallback"
    assert trace.attempts >= 2

    metadata = trace.as_metadata()
    assert metadata["requested"]["model"] == "moonshotai/kimi-k2.6"
    assert metadata["actual"]["model"] == "Qwen/Qwen3.7-Plus"
    assert metadata["served_by_requested_model"] is False


def test_trace_records_a_total_failure_rather_than_staying_silent(monkeypatch):
    primary = _target("moonshotai/kimi-k2.6", provider="openrouter")

    async def always_fails(target, **kwargs):
        raise RuntimeError("upstream exploded")

    async def noop(*_a, **_k):
        return None

    monkeypatch.setattr(generator, "complete", always_fails)
    monkeypatch.setattr(generator, "record_model_result", noop)
    monkeypatch.setattr(generator, "record_model_failure", noop)
    monkeypatch.setattr(generator, "record_writer_recovery_outcome", noop)
    monkeypatch.setattr(generator, "_sleep", noop)

    trace = GenerationTrace()
    replies = asyncio.run(
        generator.generate_replies(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            Persona(avg_message_length="short"),
            target_override=primary,
            profile_id="cleo_v3",
            trace=trace,
        )
    )

    assert replies == []
    assert trace.recorded is True
    assert trace.succeeded is False
    assert trace.failure_reason
    assert trace.as_metadata()["requested"]["model"] == "moonshotai/kimi-k2.6"


def test_generator_without_a_trace_behaves_exactly_as_before(monkeypatch):
    """The trace is an optional out-parameter, never a behaviour change."""
    primary = _target("moonshotai/kimi-k2.6", provider="openrouter")

    async def fake_complete(target, **kwargs):
        return ModelResult(
            text='["one"]',
            target=target,
            usage=ModelUsage(input_tokens=1, output_tokens=1),
            latency_ms=1,
        )

    async def noop(*_a, **_k):
        return None

    monkeypatch.setattr(generator, "complete", fake_complete)
    monkeypatch.setattr(generator, "record_model_result", noop)
    monkeypatch.setattr(generator, "record_writer_recovery_outcome", noop)

    replies = asyncio.run(
        generator.generate_replies(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            Persona(avg_message_length="short"),
            target_override=primary,
        )
    )
    assert replies == ["one"]


def test_stack_marker_reports_the_model_that_answered(monkeypatch):
    from services.suggestions import message_ai_stack_metadata

    class _Route:
        route = type("R", (), {"value": "default"})()
        prompt_version = "writer_v3"
        primary_target = _target("moonshotai/kimi-k2.6", provider="openrouter")
        fallback_target = None

    trace = GenerationTrace()
    trace.record_request(
        primary_target=_Route.primary_target,
        fallback_target=None,
        profile="cleo_v3",
        policy="persistent_primary",
        deadline_seconds=30.0,
    )
    trace.record_success(
        target=_target("Qwen/Qwen3.7-Plus"),
        role="fallback",
        attempt_index=3,
        upstream_provider="together",
        outcome="qwen_emergency_fallback",
        attempts=3,
        pinned_attempts=2,
        alternate_attempts=0,
        elapsed_ms=1200,
    )

    marker = message_ai_stack_metadata(_Route(), profile_id="cleo_v3", trace=trace)[
        "ai_stack"
    ]
    assert marker["model"] == "Qwen/Qwen3.7-Plus"
    assert marker["requested_model"] == "moonshotai/kimi-k2.6"
    assert marker["served_by_requested_model"] is False


def test_stack_marker_is_unchanged_when_no_trace_is_supplied():
    """Callers that do not run the recovery ladder keep the old marker exactly."""
    from services.suggestions import message_ai_stack_metadata

    class _Route:
        route = type("R", (), {"value": "default"})()
        prompt_version = "writer_v3"
        primary_target = _target("moonshotai/kimi-k2.6", provider="openrouter")
        fallback_target = None

    marker = message_ai_stack_metadata(_Route(), profile_id="cleo_v3")["ai_stack"]
    assert marker == {
        "profile": "cleo_v3",
        "route": "default",
        "prompt_version": "writer_v3",
        "provider": "openrouter",
        "model": "moonshotai/kimi-k2.6",
    }


# --- 3: the record itself ----------------------------------------------------


def _filled_record() -> ReplyProvenance:
    provenance = ReplyProvenance(
        creator_id="creator-1", fan_id="fan-1", mode=PIPELINE_AUTO
    )
    provenance.record_trigger(
        kind="fan_message", text="are you there?", history_position=17
    )
    provenance.record_context(
        history_messages=40,
        analyzer_window=12,
        writer_window=16,
        stack_profile="cleo_v3",
        writer_prompt_version="writer_v3",
        live_state={"creator_legend": True, "price_learning": False},
    )
    provenance.record_decision(
        source="commercial_orchestrator",
        action="tease",
        purchase_signal="none",
        resend_requested="false",
    )
    provenance.record_transform(TRANSFORM_INVENTORY_REPAIR)
    return provenance


def test_a_record_answers_every_question_step_one_asks():
    record = _filled_record().as_metadata(
        part=0, parts=2, platform_message_id="platform-99"
    )[PROVENANCE_KEY]

    assert record["trigger"]["kind"] == "fan_message"
    assert record["context"]["analyzer_window"] == 12
    assert record["context"]["writer_window"] == 16
    assert record["context"]["history_messages"] == 40
    assert record["decision"]["action"] == "tease"
    assert record["transforms"] == [TRANSFORM_INVENTORY_REPAIR]
    assert record["delivery"]["platform_message_id"] == "platform-99"
    assert record["build"]["sha"]
    assert record["build"]["flags_digest"]


def test_the_asymmetry_between_analyzer_and_writer_windows_is_on_the_record():
    """Finding D becomes a measurable property of each reply, not a code claim."""
    record = _filled_record().as_metadata()[PROVENANCE_KEY]
    assert record["context"]["analyzer_window"] < record["context"]["writer_window"]


def test_only_the_controllers_that_spoke_are_listed():
    record = _filled_record().as_metadata()[PROVENANCE_KEY]
    assert record["context"]["live_state_blocks"] == ["creator_legend"]


def test_a_record_never_stores_the_conversation():
    """It is written on every message; it must not become a second transcript."""
    record = _filled_record().as_metadata()[PROVENANCE_KEY]
    assert "are you there?" not in repr(record)
    assert record["trigger"]["text_fingerprint"] != "are you there?"
    assert record["trigger"]["text_chars"] == len("are you there?")


def test_a_delivery_with_no_receipt_is_not_reported_as_accepted():
    """A delivery claim is tied to the operation result, never to the copy."""
    record = _filled_record().as_metadata(platform_message_id=None)[PROVENANCE_KEY]
    assert record["delivery"]["accepted_by_platform"] is False
    assert record["delivery"]["platform_message_id"] is None

    accepted = _filled_record().as_metadata(platform_message_id="platform-1")[
        PROVENANCE_KEY
    ]
    assert accepted["delivery"]["accepted_by_platform"] is True


def test_the_parts_of_one_reply_share_a_turn_and_differ_only_by_part():
    provenance = _filled_record()
    first = provenance.as_metadata(part=0, parts=2, platform_message_id="a")[
        PROVENANCE_KEY
    ]
    second = provenance.as_metadata(part=1, parts=2, platform_message_id="b")[
        PROVENANCE_KEY
    ]
    assert first["turn_id"] == second["turn_id"]
    assert (first["part"], second["part"]) == (0, 1)
    assert first["delivery"]["platform_message_id"] != second[
        "delivery"
    ]["platform_message_id"]


def test_two_turns_never_share_a_turn_id():
    assert (
        _filled_record().as_metadata()[PROVENANCE_KEY]["turn_id"]
        != _filled_record().as_metadata()[PROVENANCE_KEY]["turn_id"]
    )


def test_a_transform_is_recorded_once_and_in_order():
    provenance = _filled_record()
    provenance.record_transform(TRANSFORM_OPERATOR_EDIT)
    provenance.record_transform(TRANSFORM_INVENTORY_REPAIR)
    provenance.record_transform("skipped", applied=False)
    assert provenance.transforms == [
        TRANSFORM_INVENTORY_REPAIR,
        TRANSFORM_OPERATOR_EDIT,
    ]


def test_merging_never_disturbs_existing_message_metadata():
    existing = {"ai_stack": {"profile": "cleo_v3"}, "media_ids": ["m1"]}
    merged = merge_provenance(existing, _filled_record().as_metadata())
    assert merged["ai_stack"] == {"profile": "cleo_v3"}
    assert merged["media_ids"] == ["m1"]
    assert PROVENANCE_KEY in merged


def test_reading_a_row_written_before_this_existed_is_not_an_error():
    assert provenance_of(None) == {}
    assert provenance_of({"media_ids": ["m1"]}) == {}
    assert provenance_of({PROVENANCE_KEY: "not a record"}) == {}
    assert provenance_of(merge_provenance({}, _filled_record().as_metadata()))


# --- 4: carrying an Assisted record across the operator's decision -----------


def test_a_suggestion_token_is_redeemable_exactly_once():
    store = SuggestionProvenanceStore()
    provenance = ReplyProvenance(
        creator_id="c", fan_id="f", mode=PIPELINE_ASSISTED
    )
    token = store.put(provenance)

    assert store.take(token, creator_id="c", fan_id="f") is provenance
    assert store.take(token, creator_id="c", fan_id="f") is None, (
        "one generated turn becomes at most one sent message"
    )


def test_a_token_from_another_conversation_is_refused():
    store = SuggestionProvenanceStore()
    token = store.put(
        ReplyProvenance(creator_id="c", fan_id="f", mode=PIPELINE_ASSISTED)
    )
    assert store.take(token, creator_id="c", fan_id="someone-else") is None
    assert store.take(token, creator_id="other", fan_id="f") is None


def test_a_stale_token_is_forgotten_rather_than_attached_to_a_new_reply():
    store = SuggestionProvenanceStore(ttl_seconds=60)
    token = store.put(
        ReplyProvenance(creator_id="c", fan_id="f", mode=PIPELINE_ASSISTED), now=0.0
    )
    assert store.take(token, now=61.0) is None


def test_the_store_forgets_the_oldest_turn_rather_than_every_turn():
    """core/bounded_state.py's rule: a bound that clears itself is not a bound."""
    store = SuggestionProvenanceStore(maxsize=2)
    first = store.put(ReplyProvenance(creator_id="c", fan_id="1", mode=PIPELINE_ASSISTED))
    second = store.put(ReplyProvenance(creator_id="c", fan_id="2", mode=PIPELINE_ASSISTED))
    third = store.put(ReplyProvenance(creator_id="c", fan_id="3", mode=PIPELINE_ASSISTED))

    assert store.take(first) is None
    assert store.take(second) is not None
    assert store.take(third) is not None


def test_a_missing_token_never_raises():
    store = SuggestionProvenanceStore()
    assert store.take(None) is None
    assert store.take("") is None
    assert store.take("not-a-token") is None


# --- 5: the build endpoint is the lookup for a digest ------------------------


def test_build_endpoint_returns_the_full_flag_mapping(monkeypatch):
    from fastapi.testclient import TestClient

    import main

    async def fake_user(authorization):
        return "operator-1"

    monkeypatch.setattr(main, "authenticated_dashboard_user", fake_user, raising=False)
    monkeypatch.setattr("core.auth.authenticated_dashboard_user", fake_user)

    # No lifespan: this asserts the route's contract, not the boot sequence.
    response = TestClient(main.app).get(
        "/build",
        headers={
            "X-API-Key": os.environ["DASHBOARD_API_SECRET"],
            "Authorization": "Bearer operator-1",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["flags_digest"]
    assert set(body["flags"]) == set(build_info.OBSERVED_FLAGS)
    # The response is read by operators; a key must not be reachable from it.
    assert "ANTHROPIC_API_KEY" not in body["flags"]
    assert os.environ.get("ANTHROPIC_API_KEY", "sentinel") not in str(body)


def test_the_public_health_path_exposes_a_digest_but_never_the_flags(monkeypatch):
    """/health is unauthenticated, so the values stay behind /build."""
    from fastapi.testclient import TestClient

    from services import operational_health

    import main

    operational_health.reset_cache()
    body = TestClient(main.app).get("/health").json()
    assert body["build_sha"], "the deployed commit is always identifiable"
    assert "flags" not in body, "flag values stay behind auth on /build"


# --- 6: the operator's send closes the Assisted record ----------------------


def _reply_client(monkeypatch, sent: list[dict]):
    """A ``/reply`` that reaches persistence without touching a platform."""
    from fastapi.testclient import TestClient

    import main

    async def fake_user(authorization):
        return "operator-1"

    async def fake_access(*_a, **_k):
        return None

    async def fake_send(*_a, **_k):
        return {"response": {"id": "platform-7"}}

    async def fake_save_message(*args, **kwargs):
        sent.append({"args": args, "kwargs": kwargs})
        return "message-1"

    def fake_supabase():
        class _Q:
            def select(self, *_a, **_k):
                return self

            def eq(self, *_a, **_k):
                return self

            def single(self):
                return self

            def execute(self):
                from types import SimpleNamespace

                return SimpleNamespace(
                    data={
                        "fansly_group_id": "group-1",
                        "apifansly_account_id": "account-1",
                    }
                )

        class _DB:
            def table(self, _name):
                return _Q()

        return _DB()

    monkeypatch.setattr(main, "authenticated_dashboard_user", fake_user, raising=False)
    monkeypatch.setattr("core.auth.authenticated_dashboard_user", fake_user)
    monkeypatch.setattr(main, "require_creator_fan_access", fake_access)
    monkeypatch.setattr(main, "send_apifansly_message", fake_send)
    monkeypatch.setattr(main, "save_message", fake_save_message)
    monkeypatch.setattr(main, "get_supabase", fake_supabase)
    def fake_spawn(coro, *_a, **_k):
        # Close it rather than dropping it: an un-awaited coroutine warning here
        # is noise that would hide a real one somewhere else.
        coro.close()
        return None

    monkeypatch.setattr(main, "spawn", fake_spawn)
    monkeypatch.setattr(main, "sent_message_id", lambda body: "platform-7")
    return TestClient(main.app)


def _reply_headers() -> dict[str, str]:
    return {
        "X-API-Key": os.environ["DASHBOARD_API_SECRET"],
        "Authorization": "Bearer operator-1",
    }


def test_an_operator_sent_reply_carries_the_turn_that_produced_it(monkeypatch):
    from services.reply_provenance import SUGGESTION_PROVENANCE

    provenance = ReplyProvenance(
        creator_id="creator-1", fan_id="fan-1", mode=PIPELINE_ASSISTED
    )
    provenance.record_trigger(kind="fan_message", text="hey")
    token = SUGGESTION_PROVENANCE.put(provenance)

    sent: list[dict] = []
    response = _reply_client(monkeypatch, sent).post(
        "/reply",
        headers=_reply_headers(),
        json={
            "fan_id": "fan-1",
            "creator_id": "creator-1",
            "content": "hey you",
            "was_ai_suggested": True,
            "suggestion_token": token,
            "suggestion_index": 1,
            "suggestion_edited": True,
        },
    )

    assert response.status_code == 200
    record = sent[0]["kwargs"]["media_context"][PROVENANCE_KEY]
    assert record["mode"] == PIPELINE_ASSISTED
    assert record["decision"]["chosen_index"] == "1"
    assert record["delivery"]["platform_message_id"] == "platform-7"
    assert record["delivery"]["accepted_by_platform"] is True
    # The most important one: a human rewrote it, so the model does not own it.
    assert TRANSFORM_OPERATOR_EDIT in record["transforms"]


def test_a_reply_typed_from_scratch_sends_with_no_invented_provenance(monkeypatch):
    sent: list[dict] = []
    response = _reply_client(monkeypatch, sent).post(
        "/reply",
        headers=_reply_headers(),
        json={
            "fan_id": "fan-1",
            "creator_id": "creator-1",
            "content": "typed by hand",
        },
    )

    assert response.status_code == 200
    assert sent[0]["kwargs"]["media_context"] is None


def test_an_expired_token_does_not_stop_the_operator_sending(monkeypatch):
    """Provenance is a record. Losing one must never cost a message."""
    sent: list[dict] = []
    response = _reply_client(monkeypatch, sent).post(
        "/reply",
        headers=_reply_headers(),
        json={
            "fan_id": "fan-1",
            "creator_id": "creator-1",
            "content": "still sends",
            "suggestion_token": "long-gone",
        },
    )

    assert response.status_code == 200
    assert sent[0]["kwargs"]["media_context"] is None
