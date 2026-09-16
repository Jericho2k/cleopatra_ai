"""Kimi is the real V3 writer: another Kimi host before another model.

The incident this file pins down, in full:

* V3 correctly selected ``moonshotai/kimi-k2.6``;
* Kimi was pinned to the OpenRouter upstream ``Inceptron``;
* Inceptron returned 429, repeatedly;
* the writer retried the SAME rate-limited pool on a 5s/30s/60s schedule —
  ninety-five seconds of waiting for one host to stop being busy;
* after four Inceptron attempts it fell to Qwen, which answered instantly.

Nothing ever asked a different Kimi host. So "primary: Kimi" described the
configuration and not the output, and the fan's reply came from a different
model because one provider had a bad minute.

The corrected ladder is three cache-affine Inceptron attempts (5s, 30s), then
the same model routed anywhere else eligible, and Qwen only when Kimi cannot be
served at all. These tests assert each rung, that healthy traffic never leaves
the pin, that a permanent failure skips the whole thing, and that the frozen V1
and V2 plans are untouched.

Every delay here is asserted through a fake clock. No test waits.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ai import generator, writer_recovery
from ai.generator import (
    CONTRACT_AUTO_MESSAGES,
    KIMI_ALTERNATE_ATTEMPTS,
    KIMI_ALTERNATE_WAIT_SECONDS,
    KIMI_PINNED_ATTEMPTS,
    KIMI_PINNED_WAIT_SECONDS,
    LEGACY_WRITER_RETRY_POLICY,
    PERSISTENT_PRIMARY_RETRY_POLICY,
    is_permanent_failure,
)
from ai.openrouter_routing import PROVIDER_MODE_ALTERNATE
from ai.writer_style import (
    MODE_ASSISTED,
    MODE_AUTO,
    WRITER_V1,
    WRITER_V2,
    WRITER_V3,
    persistent_primary_retries,
    uses_auto_messages_contract,
)
from models.model_runtime import ModelResult, ModelTarget, ModelUsage
from models.schemas import Persona
from services.suggestions import writer_retry_policy


def _target(provider: str, model: str, **metadata) -> ModelTarget:
    return ModelTarget(
        name=model,
        provider=provider,
        model=model,
        base_url=f"https://{provider}.example/v1",
        api_key_env="TEST_API_KEY",
        timeout_seconds=90.0,
        metadata=dict(metadata),
    )


KIMI = _target("openrouter", "moonshotai/kimi-k2.6", openrouter_providers=["Inceptron"])
QWEN = _target("together", "Qwen/Qwen3.7-Plus")

PINNED = writer_recovery.ROLE_PINNED
ALTERNATE = writer_recovery.ROLE_ALTERNATE


class _ProviderError(Exception):
    """Shaped like the SDK errors, without importing a client."""

    def __init__(self, status_code: int | None = None, message: str | None = None):
        super().__init__(message or f"HTTP {status_code}")
        if status_code is not None:
            self.status_code = status_code


def _routing_mode(target: ModelTarget) -> str:
    """Which OpenRouter routing mode this attempt would actually send."""
    return str((target.metadata or {}).get("openrouter_provider_mode") or "pinned")


@pytest.fixture
def harness(monkeypatch):
    """Records every attempt as ``(model, routing_mode)`` and every wait."""
    state: dict = {"calls": [], "sleeps": [], "responses": [], "logs": []}

    async def fake_sleep(seconds):
        state["sleeps"].append(seconds)

    async def fake_complete(target, **_kwargs):
        state["calls"].append((target.model, _routing_mode(target)))
        index = len(state["calls"]) - 1
        behaviour = state["responses"][min(index, len(state["responses"]) - 1)]
        if isinstance(behaviour, Exception):
            raise behaviour
        return ModelResult(
            text=behaviour,
            target=target,
            usage=ModelUsage(input_tokens=10, output_tokens=5),
            latency_ms=1,
            upstream_provider=(
                "Parasail" if _routing_mode(target) == PROVIDER_MODE_ALTERNATE else "Inceptron"
            ),
        )

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(generator, "_sleep", fake_sleep)
    monkeypatch.setattr(generator, "complete", fake_complete)
    monkeypatch.setattr(generator, "record_model_result", noop)
    monkeypatch.setattr(generator, "record_model_failure", noop)
    monkeypatch.setattr(generator, "record_writer_recovery_outcome", noop)
    monkeypatch.setattr(generator.random, "uniform", lambda _low, high: high)
    monkeypatch.setitem(
        generator.__builtins__ if isinstance(generator.__builtins__, dict)
        else vars(generator.__builtins__),
        "print",
        lambda *args, **_k: state["logs"].append(" ".join(map(str, args))),
    )
    return state


def _run(
    harness,
    *,
    policy=PERSISTENT_PRIMARY_RETRY_POLICY,
    fallback=QWEN,
    contract=CONTRACT_AUTO_MESSAGES,
    profile_id="cleo_v3",
    primary=KIMI,
):
    return asyncio.run(
        generator.generate_replies(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            Persona(),
            telemetry_context={"feature": "auto_reply"},
            target_override=primary,
            fallback_target_override=fallback,
            output_contract=contract,
            retry_policy=policy,
            profile_id=profile_id,
        )
    )


def _messages(*bubbles: str) -> str:
    return json.dumps({"messages": list(bubbles)})


def _models(harness) -> list[str]:
    return [model for model, _mode in harness["calls"]]


def _modes(harness) -> list[str]:
    return [mode for _model, mode in harness["calls"]]


# --- the shape of the ladder ------------------------------------------------


def test_the_configured_ladder_is_pinned_then_alternate_then_fallback():
    plan = PERSISTENT_PRIMARY_RETRY_POLICY.build_plan(KIMI, QWEN)

    assert [step.role for step in plan] == (
        [PINNED] * KIMI_PINNED_ATTEMPTS
        + [ALTERNATE] * KIMI_ALTERNATE_ATTEMPTS
        + [writer_recovery.ROLE_FALLBACK]
    )
    # Three cache-affine attempts spaced 5s and 30s, then a host change with no
    # further Inceptron wait — the 60s the incident spent is gone.
    assert KIMI_PINNED_ATTEMPTS == 3
    assert KIMI_PINNED_WAIT_SECONDS == (5.0, 30.0)
    assert [step.wait_before for step in plan[:3]] == [0.0, 5.0, 30.0]
    assert plan[3].wait_before == 0.0, "no extra wait before changing host"
    assert KIMI_ALTERNATE_WAIT_SECONDS == (0.0, 5.0)
    # Every Kimi rung is the same model. Only the routing differs.
    assert {step.target.model for step in plan[:-1]} == {KIMI.model}


def test_the_alternate_rungs_route_away_from_the_pinned_provider():
    plan = PERSISTENT_PRIMARY_RETRY_POLICY.build_plan(KIMI, QWEN)
    alternate = next(step for step in plan if step.role == ALTERNATE)

    preferences = generator.openrouter_routing.provider_preferences(
        alternate.target.metadata
    )

    assert preferences["ignore"] == ["Inceptron"]
    assert preferences["allow_fallbacks"] is True
    assert "only" not in preferences
    # Recovery is never a way around the privacy configuration.
    assert preferences["data_collection"] == "deny"


def test_healthy_traffic_never_leaves_the_pinned_provider(harness):
    """The economics. One success means one Inceptron request, cache-affine."""
    harness["responses"] = [_messages("hey you")]

    replies = _run(harness)

    assert replies == ["hey you"]
    assert harness["calls"] == [(KIMI.model, "pinned")]
    assert harness["sleeps"] == []


# --- THE INCIDENT -----------------------------------------------------------


def test_repeated_inceptron_429s_reach_another_kimi_host_and_never_qwen(harness):
    """The exact production sequence, with the outcome it should have had.

    429, wait 5, 429, wait 30, 429 — then the SAME model somewhere else, which
    answers. Qwen is never called, because Kimi was never actually unavailable.
    """
    harness["responses"] = [
        _ProviderError(429),
        _ProviderError(429),
        _ProviderError(429),
        _messages("hey you", "what are you up to"),
    ]

    replies = _run(harness)

    assert replies == ["hey you | what are you up to"]
    assert _models(harness) == [KIMI.model] * 4
    assert _modes(harness) == ["pinned", "pinned", "pinned", PROVIDER_MODE_ALTERNATE]
    assert QWEN.model not in _models(harness)
    # 35 seconds, not 95, and nothing waited before changing host.
    assert harness["sleeps"] == [5.0, 30.0]
    assert sum(harness["sleeps"]) == 35.0


def test_the_failover_and_its_success_are_both_logged(harness):
    harness["responses"] = [
        _ProviderError(429),
        _ProviderError(429),
        _ProviderError(429),
        _messages("hi"),
    ]

    _run(harness)

    failover = [l for l in harness["logs"] if l.startswith("[KIMI PROVIDER FAILOVER]")]
    assert len(failover) == 1, harness["logs"]
    assert "from=Inceptron" in failover[0]
    assert "reason=repeated_429" in failover[0]
    assert "after_pinned_attempts=3" in failover[0]

    success = [l for l in harness["logs"] if l.startswith("[KIMI PROVIDER SUCCESS]")]
    assert len(success) == 1
    assert "provider=Parasail" in success[0], "the host that actually served it"

    primary = [l for l in harness["logs"] if l.startswith("[WRITER PRIMARY]")]
    assert primary and "provider=Inceptron" in primary[0]

    retries = [l for l in harness["logs"] if l.startswith("[KIMI RETRY]")]
    assert "attempt=2" in retries[0] and "wait=5" in retries[0]
    assert "reason=rate_limited:429" in retries[0]


def test_qwen_is_reached_only_after_every_kimi_host_is_spent(harness):
    harness["responses"] = [
        _ProviderError(429),
        _ProviderError(429),
        _ProviderError(429),
        _ProviderError(503),
        _ProviderError(503),
        _messages("qwen wrote this"),
    ]

    replies = _run(harness)

    assert replies == ["qwen wrote this"]
    assert _models(harness) == [KIMI.model] * 5 + [QWEN.model]
    assert _modes(harness)[3:5] == [PROVIDER_MODE_ALTERNATE] * 2
    # 5 and 30 on the pin, then 0 and 5 between the alternate hosts.
    assert harness["sleeps"] == [5.0, 30.0, 5.0]

    fallback = [l for l in harness["logs"] if l.startswith("[WRITER FALLBACK]")]
    assert fallback and "reason=kimi_exhausted" in fallback[0]
    assert "after_primary_attempts=5" in fallback[0]


def test_a_provider_retry_after_longer_than_the_schedule_wins(harness):
    class _RateLimited(Exception):
        def __init__(self):
            super().__init__("429")
            self.status_code = 429

            class _Response:
                headers = {"retry-after": "12"}
                status_code = 429

            self.response = _Response()

    harness["responses"] = [_RateLimited(), _messages("ok now")]

    _run(harness)

    # 12s advertised beats the 5s floor for attempt 2.
    assert harness["sleeps"] == [12.0]


def test_invalid_structured_output_retries_the_same_host(harness):
    """A generation fault, not a routing one. Changing provider cannot fix it."""
    harness["responses"] = [
        "not json at all",
        json.dumps(["an array, which is the wrong shape"]),
        _messages("third time lucky"),
    ]

    replies = _run(harness)

    assert replies == ["third time lucky"]
    assert _modes(harness) == ["pinned"] * 3


def test_rejected_candidate_output_retries_kimi_under_the_persistent_policy(harness):
    harness["responses"] = [_messages("noted"), _messages("actually saying something")]

    replies = _run(harness)

    assert replies == ["actually saying something"]
    assert _models(harness) == [KIMI.model, KIMI.model]
    assert harness["sleeps"] == [KIMI_PINNED_WAIT_SECONDS[0]]


# --- failover is a recovery mechanism, not a routing strategy ---------------


def test_provider_failover_can_be_switched_off_entirely(harness, monkeypatch):
    """Back to pinned-then-fallback, for a deployment that wants exactly that."""
    monkeypatch.setenv("OPENROUTER_PROVIDER_FAILOVER", "false")
    harness["responses"] = [
        _ProviderError(429),
        _ProviderError(429),
        _ProviderError(429),
        _messages("qwen wrote this"),
    ]

    replies = _run(harness)

    assert replies == ["qwen wrote this"]
    assert _models(harness) == [KIMI.model] * 3 + [QWEN.model]
    assert PROVIDER_MODE_ALTERNATE not in _modes(harness)


def test_a_direct_provider_primary_has_no_alternate_host_to_try(harness):
    """Failover is an aggregator capability. Together has one host per model."""
    plan = PERSISTENT_PRIMARY_RETRY_POLICY.build_plan(QWEN, None)

    assert [step.role for step in plan] == [PINNED] * KIMI_PINNED_ATTEMPTS


def test_an_explicit_alternate_list_is_used_when_the_operator_sets_one(monkeypatch):
    """Discovery is the default; a named list is available and is not required."""
    monkeypatch.setenv("OPENROUTER_FALLBACK_PROVIDERS", "Parasail, Together")

    plan = PERSISTENT_PRIMARY_RETRY_POLICY.build_plan(KIMI, QWEN)
    alternate = next(step for step in plan if step.role == ALTERNATE)
    preferences = generator.openrouter_routing.provider_preferences(
        alternate.target.metadata
    )

    assert preferences["only"] == ["Parasail", "Together"]
    assert "ignore" not in preferences
    assert preferences["allow_fallbacks"] is True


# --- permanent failures skip the whole ladder -------------------------------


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 405, 422])
def test_permanent_statuses_are_recognised(status):
    assert is_permanent_failure(_ProviderError(status))


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504])
def test_retryable_statuses_are_not_permanent(status):
    assert not is_permanent_failure(_ProviderError(status))


def test_a_missing_api_key_is_permanent_even_without_a_status():
    assert is_permanent_failure(_ProviderError(message="OPENROUTER_API_KEY not configured"))


@pytest.mark.parametrize(
    "error,reason",
    [
        (_ProviderError(429), writer_recovery.REASON_RATE_LIMITED),
        (_ProviderError(503), writer_recovery.REASON_PROVIDER_5XX),
        (_ProviderError(401), writer_recovery.REASON_PERMANENT),
        (TimeoutError("request timed out"), writer_recovery.REASON_TIMEOUT),
        (ConnectionResetError("connection reset by peer"), writer_recovery.REASON_TRANSPORT),
    ],
)
def test_failure_reasons_are_classified_distinctly(error, reason):
    """Telemetry has to be able to tell a rate limit from a dead socket."""
    assert writer_recovery.classify_failure(error).reason == reason


def test_a_permanent_failure_skips_the_alternate_hosts_too(harness):
    """A rejected key or an unknown model is not repaired by changing host.

    Spending 35 seconds and two extra hosts rediscovering it is exactly the
    waste the classification exists to prevent.
    """
    harness["responses"] = [_ProviderError(401), _messages("qwen answered")]

    replies = _run(harness)

    assert replies == ["qwen answered"]
    assert _models(harness) == [KIMI.model, QWEN.model]
    assert PROVIDER_MODE_ALTERNATE not in _modes(harness)
    assert harness["sleeps"] == [], "no wait can fix a rejected key"


def test_a_permanent_failure_with_no_fallback_stops_immediately(harness):
    harness["responses"] = [_ProviderError(403)]

    replies = _run(harness, fallback=None)

    assert replies == []
    assert _models(harness) == [KIMI.model]
    assert harness["sleeps"] == []


def test_the_persistent_policy_does_not_invent_an_extra_attempt_without_a_fallback(
    harness,
):
    harness["responses"] = [_ProviderError(503)]

    replies = _run(harness, fallback=None)

    assert replies == []
    assert _models(harness) == [KIMI.model] * (
        KIMI_PINNED_ATTEMPTS + KIMI_ALTERNATE_ATTEMPTS
    )


# --- the backend-owned deadline ---------------------------------------------


def test_the_deadline_comfortably_covers_the_whole_ladder():
    plan = PERSISTENT_PRIMARY_RETRY_POLICY.build_plan(KIMI, QWEN)
    deadline = writer_recovery.plan_deadline_seconds(plan)

    # Every wait plus every attempt's own client timeout, and then some.
    minimum = sum(step.wait_before + step.target.timeout_seconds for step in plan)
    assert deadline >= minimum
    # Derived, not typed: changing the schedule moves it.
    assert deadline == pytest.approx(minimum * writer_recovery.DEADLINE_MARGIN)


def test_the_deadline_ends_the_turn_rather_than_letting_it_run_on(harness, monkeypatch):
    """A writer task may not run forever just because polling made it safe to.

    The turn stops, reports a total failure, and — because the in-flight
    request is cancelled rather than abandoned — cannot produce a reply that
    arrives after the operator was told it failed.
    """
    monkeypatch.setenv("WRITER_TURN_DEADLINE_SECONDS", "1")

    async def never_answers(_target, **_kwargs):
        harness["calls"].append((KIMI.model, "pinned"))
        await asyncio.sleep(30)

    monkeypatch.setattr(generator, "complete", never_answers)

    replies = _run(harness)

    assert replies == []
    assert len(harness["calls"]) == 1, "the deadline stopped the ladder"
    deadline_lines = [l for l in harness["logs"] if l.startswith("[WRITER DEADLINE]")]
    assert deadline_lines, harness["logs"]
    assert "cancelled=attempt_1" in deadline_lines[0]


def test_a_turn_does_not_sleep_into_an_expiry_it_can_already_see(harness, monkeypatch):
    """The 30s wait is not entered when only 10s of budget remain."""
    monkeypatch.setenv("WRITER_TURN_DEADLINE_SECONDS", "20")
    harness["responses"] = [_ProviderError(429), _ProviderError(429)]

    replies = _run(harness)

    assert replies == []
    # Attempt 1, the 5s wait, attempt 2 — and then the 30s wait is refused
    # because it cannot fit, rather than being slept and discovered afterwards.
    assert harness["sleeps"] == [5.0]
    assert len(harness["calls"]) == 2


# --- the telemetry vocabulary -----------------------------------------------


@pytest.mark.parametrize(
    "role,attempt_in_role,expected",
    [
        (PINNED, 1, writer_recovery.OUTCOME_PINNED_FIRST_TRY),
        (PINNED, 3, writer_recovery.OUTCOME_PINNED_RETRY),
        (ALTERNATE, 1, writer_recovery.OUTCOME_ALTERNATE_PROVIDER),
        (writer_recovery.ROLE_FALLBACK, 1, writer_recovery.OUTCOME_QWEN_FALLBACK),
    ],
)
def test_each_rung_maps_to_its_own_operator_outcome(role, attempt_in_role, expected):
    attempt = writer_recovery.WriterAttempt(
        index=1, role=role, target=KIMI, wait_before=0.0, attempt_in_role=attempt_in_role
    )
    assert writer_recovery.outcome_for(attempt) == expected


def test_no_successful_attempt_is_a_total_failure():
    assert writer_recovery.outcome_for(None) == writer_recovery.OUTCOME_TOTAL_FAILURE


def test_the_five_outcomes_are_the_ones_operators_query():
    assert writer_recovery.OUTCOMES == (
        "kimi_inceptron_first_try_success",
        "kimi_inceptron_retry_success",
        "kimi_alternate_provider_success",
        "qwen_emergency_fallback",
        "writer_total_failure",
    )


def test_the_recovery_outcome_is_recorded_once_per_turn(monkeypatch):
    """Into the EXISTING model telemetry, not a second store beside it."""
    from services import model_telemetry

    rows: list[dict] = []
    monkeypatch.setattr(
        model_telemetry, "_enqueue_record", lambda **kwargs: rows.append(kwargs)
    )
    monkeypatch.setenv("MODEL_TELEMETRY_ENABLED", "true")

    asyncio.run(
        model_telemetry.record_writer_recovery_outcome(
            writer_recovery.OUTCOME_ALTERNATE_PROVIDER,
            target=KIMI,
            context=model_telemetry.ModelTelemetryContext(
                feature="auto_reply", creator_id="c1", fan_id="f1"
            ),
            profile="cleo_v3",
            policy="persistent_primary",
            elapsed_ms=36_000,
            deadline_seconds=638.0,
            attempts=4,
            pinned_attempts=3,
            alternate_attempts=1,
            role=ALTERNATE,
            upstream_provider="Parasail",
        )
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["context"].feature == "writer_recovery"
    assert row["context"].creator_id == "c1"
    metadata = row["context"].metadata
    assert metadata["writer_recovery_outcome"] == "kimi_alternate_provider_success"
    assert metadata["writer_recovery_pinned_attempts"] == 3
    assert metadata["writer_recovery_alternate_attempts"] == 1
    # Latency is the whole turn, waits included — what a deadline is set
    # against, and not a provider call.
    assert row["latency_ms"] == 36_000
    assert row["upstream_provider"] == "Parasail"


# --- the frozen profiles are untouched --------------------------------------


def test_v1_and_v2_keep_the_frozen_two_attempts_then_fallback_plan(harness):
    harness["responses"] = [
        _ProviderError(503),
        _ProviderError(503),
        json.dumps(["qwen wrote this"]),
    ]

    replies = _run(
        harness,
        policy=LEGACY_WRITER_RETRY_POLICY,
        contract="candidates",
        profile_id="cleo_v2",
    )

    assert replies == ["qwen wrote this"]
    assert _models(harness) == [KIMI.model, KIMI.model, QWEN.model]
    assert PROVIDER_MODE_ALTERNATE not in _modes(harness), (
        "the frozen profiles get no provider failover: they are the baseline"
    )


def test_the_legacy_policy_still_retires_a_model_whose_output_was_rejected(harness):
    harness["responses"] = [json.dumps(["noted"]), json.dumps(["come closer"])]

    replies = _run(
        harness,
        policy=LEGACY_WRITER_RETRY_POLICY,
        contract="candidates",
        profile_id="cleo_v2",
    )

    assert replies == ["come closer"]
    assert _models(harness) == [KIMI.model, QWEN.model]
    assert harness["sleeps"] == []


@pytest.mark.parametrize("version", [WRITER_V1, WRITER_V2])
def test_only_v3_opts_into_the_persistent_plan(version):
    assert persistent_primary_retries(version) is False
    assert writer_retry_policy(version) is LEGACY_WRITER_RETRY_POLICY


def test_v3_selects_the_persistent_plan():
    assert persistent_primary_retries(WRITER_V3) is True
    assert writer_retry_policy(WRITER_V3) is PERSISTENT_PRIMARY_RETRY_POLICY


# --- retired configuration --------------------------------------------------


def test_the_old_retry_variables_are_reported_as_having_no_effect(monkeypatch, capsys):
    """Configuring nothing is worse than configuring the wrong thing."""
    monkeypatch.setenv("WRITER_PRIMARY_RETRY_WAIT_SECONDS", "5,30,60")
    monkeypatch.setattr(writer_recovery, "_warned", False)

    stale = writer_recovery.warn_about_retired_configuration()

    assert stale == ["WRITER_PRIMARY_RETRY_WAIT_SECONDS"]
    logged = capsys.readouterr().out
    assert "no longer has any effect" in logged
    assert "WRITER_KIMI_PINNED_WAIT_SECONDS" in logged


# --- the auto contract ------------------------------------------------------


def test_only_v3_auto_uses_the_messages_object():
    assert uses_auto_messages_contract(WRITER_V3, MODE_AUTO) is True
    assert uses_auto_messages_contract(WRITER_V3, MODE_ASSISTED) is False
    assert uses_auto_messages_contract(WRITER_V2, MODE_AUTO) is False
    assert uses_auto_messages_contract(WRITER_V1, MODE_AUTO) is False


def test_two_bubbles_are_one_reply_not_two_candidates(harness):
    harness["responses"] = [_messages("okay wait", "that's actually funny")]

    replies = _run(harness)

    assert replies == ["okay wait | that's actually funny"]
    assert len(replies) == 1, "a multi-bubble reply is still ONE reply"


def test_no_candidate_truncation_log_on_the_auto_path(harness):
    harness["responses"] = [_messages("one", "two", "three")]

    replies = _run(harness)

    assert replies == ["one | two | three"]
    assert not [line for line in harness["logs"] if "candidates for a" in line]
