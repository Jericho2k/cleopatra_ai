from __future__ import annotations

import json

import pytest

from models.model_runtime import ModelResult, ModelTarget, ModelUsage
from services.candidate_provider_eval import (
    WRITER_SYSTEM,
    build_evaluation_bundle,
    run_provider_comparison,
)
from services.decision_owners import REPLY_PLUS_INTENT_SYSTEM, SEMANTIC_SYSTEM
from services.decision_replay import ReplayTurn


def _result(target: ModelTarget, text: str) -> ModelResult:
    return ModelResult(
        text=text,
        target=target,
        usage=ModelUsage(input_tokens=100, output_tokens=20),
        latency_ms=12,
        raw_response_id="response-1",
    )


@pytest.mark.asyncio
async def test_real_provider_seams_compare_complete_cores_without_side_effects():
    chat = ModelTarget(name="chat", provider="together", model="chat-model")
    analyzer = ModelTarget(
        name="analyzer", provider="together", model="analyzer-model"
    )
    calls: list[str] = []

    async def complete(target, *, system, messages, max_tokens, **_kwargs):
        calls.append(system)
        if system == REPLY_PLUS_INTENT_SYSTEM:
            return _result(
                target,
                json.dumps(
                    {
                        "reply": "glad you finished it",
                        "active_needs": ["ordinary conversation"],
                        "unresolved_references": [],
                        "must_address": [],
                        "operation": "none",
                        "operation_subject": "",
                        "operation_because": "",
                        "hold": "none",
                        "hold_detail": "",
                        "confidence": 0.9,
                    }
                ),
            )
        if system == SEMANTIC_SYSTEM:
            return _result(
                target,
                json.dumps(
                    {
                        "active_needs": ["ordinary conversation"],
                        "unresolved_references": [],
                        "must_address": [],
                        "operation": "none",
                        "operation_subject": "",
                        "operation_because": "",
                        "hold": "none",
                        "hold_detail": "",
                        "confidence": 0.9,
                    }
                ),
            )
        assert system == WRITER_SYSTEM
        return _result(target, "glad you finished it")

    turns = [
        ReplayTurn(
            name="ordinary",
            history=(
                {"role": "fan", "content": "finally finished that project"},
            ),
        )
    ]
    report, provider_calls = await run_provider_comparison(
        turns,
        complete=complete,
        one_call_target=chat,
        semantic_target=analyzer,
        writer_target=chat,
    )

    assert calls == [REPLY_PLUS_INTENT_SYSTEM, SEMANTIC_SYSTEM, WRITER_SYSTEM]
    assert [call.candidate for call in provider_calls] == [
        "reply_plus_intent",
        "semantic_owner",
        "semantic_owner_writer",
    ]
    assert report.turns[0]["executions"]["reply_plus_intent"][
        "operation_permitted_in_dry_run"
    ] is False


def test_bundle_pins_inputs_outputs_and_zero_mutation_invariants(tmp_path):
    root = tmp_path / "repo"
    (root / "db").mkdir(parents=True)
    (root / "db" / "migration_order.txt").write_text(
        "first.sql\nlast.sql\n", encoding="utf-8"
    )
    scenarios = root / "scenarios.json"
    scenarios.write_text('{"scenarios": []}\n', encoding="utf-8")
    output = tmp_path / "bundle"
    target = ModelTarget(name="test", provider="together", model="test-model")

    from services.candidate_execution import ExecutionReport

    manifest = build_evaluation_bundle(
        output,
        root=root,
        scenarios_path=scenarios,
        scenario_names=["ordinary"],
        report=ExecutionReport(candidates=("left", "right")),
        calls=[],
        targets={"left": target},
        flags={"APP_ENV": "test"},
    )

    assert manifest["invariants"]["live_messages_sent"] == 0
    assert manifest["invariants"]["production_mutations"] == 0
    assert manifest["database_schema"]["last_migration"] == "last.sql"
    assert {path.name for path in output.iterdir()} == {
        "README.md",
        "manifest.json",
        "results.json",
        "disagreements.json",
    }
