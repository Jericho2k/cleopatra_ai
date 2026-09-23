"""The commands, end to end, over the real scenario suite.

The unit tests cover the pieces. This covers the thing an operator actually
types: that ``--describe`` works with no backend at all, that a complete run
over all thirteen scenarios produces the documented directory, and that the blind
review built from that directory is genuinely blind.

The backend is the recording stand-in rather than the simulator, so this suite
needs no database and no provider. What it is testing is the harness and the
artifacts, which is the part that would otherwise only ever be exercised by a
run that costs money.
"""

# The subprocess return codes are the subject of these CLI tests.
# ruff: noqa: PLW1510

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from services.ab_trajectory_eval import (
    ROLE_BASELINE,
    ROLE_CANDIDATE,
    ArmSpec,
    RunSpec,
    load_scenario_file,
    run_suite,
    select_trajectories,
    write_artifacts,
)
from services.blind_conversation_review import (
    forbidden_terms,
    leaked_terms,
    unblind_all,
)
from tests.test_ab_trajectory_eval import RecordingBackend, _fans

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "eval" / "conversational_core_scenarios.json"
CLI = ROOT / "scripts" / "run_ab_trajectory_eval.py"
REVIEW_CLI = ROOT / "scripts" / "build_conversation_blind_review.py"


def test_describe_lists_the_suite_without_a_backend():
    result = subprocess.run(
        [sys.executable, str(CLI), "--describe"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "A_ordinary_statement" in result.stdout
    assert "L_long_trajectory" in result.stdout
    assert "13 scenario(s)" in result.stdout
    # Claims and reach printed together, so a label cannot drift unnoticed.
    assert "NOT COVERED" in result.stdout or "covers:" in result.stdout


def test_running_without_a_creator_refuses_rather_than_guessing():
    result = subprocess.run(
        [sys.executable, str(CLI), "--baseline", "semantic_v2"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--creator is required" in result.stderr


def test_an_unregistered_candidate_runtime_is_refused_by_name():
    """An actually unknown runtime is rejected before any backend access."""
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--baseline",
            "semantic_v2",
            "--candidate",
            "conversational_v99",
            "--creator",
            "creator-1",
            "--baseline-fan",
            "a",
            "--candidate-fan",
            "b",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "not registered in this build" in result.stderr
    assert "run the baseline arm on its own" in result.stderr


def _full_run(tmp_path, roles=(ROLE_BASELINE, ROLE_CANDIDATE)):
    payload = load_scenario_file(SCENARIOS)
    trajectories, ids = select_trajectories(payload)
    arms = [
        ArmSpec(role=ROLE_BASELINE, core_id="semantic_v2", fan_id="fan-a"),
        ArmSpec(role=ROLE_CANDIDATE, core_id="semantic_v1", fan_id="fan-b"),
    ][: len(roles)]
    backend = RecordingBackend(fans=_fans(*[arm.fan_id for arm in arms]))
    spec = RunSpec(
        run_id="cli-test-run",
        creator_id="creator-1",
        seed=1729,
        arms=tuple(arms),
        scenario_ids=tuple(ids),
        stack_profile="cleo_v2",
    )
    runs = asyncio.run(
        run_suite(spec=spec, trajectories=trajectories, backend=backend, scenario_ids=ids)
    )
    return write_artifacts(
        spec=spec, runs=runs, trajectories=trajectories, results_root=tmp_path
    ), backend


def test_a_complete_ab_run_over_the_whole_suite_produces_the_artifacts(tmp_path):
    root, backend = _full_run(tmp_path)

    metadata = json.loads((root / "metadata.json").read_text())
    assert len(metadata["scenario_ids"]) == 13
    assert metadata["baseline_core"] == "semantic_v2"
    assert metadata["candidate_core"] == "semantic_v1"
    assert len(metadata["arm_order_per_scenario"]) == 13

    paired = json.loads((root / "paired.json").read_text())
    assert paired["paired"] is True
    assert paired["scenarios"] == 13
    assert paired["fan_inputs_identical"] is True

    # Both arms saw every scripted message, and neither saw the other's fan.
    per_fan: dict[str, list[str]] = {}
    for fan_id, message in backend.sent:
        per_fan.setdefault(fan_id, []).append(message)
    assert per_fan["fan-a"] == per_fan["fan-b"]
    assert len(per_fan["fan-a"]) == sum(
        len(pair["arms"]["baseline"]["turns"]) for pair in paired["pairs"]
    )

    metrics = json.loads((root / "metrics.json").read_text())
    assert set(metrics["baseline"]["per_scenario"]) == set(metadata["scenario_ids"])


def test_the_blind_review_command_produces_a_blind_document(tmp_path):
    root, _ = _full_run(tmp_path)
    result = subprocess.run(
        [sys.executable, str(REVIEW_CLI), str(root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    document = (root / "blind_review.md").read_text(encoding="utf-8")
    assert leaked_terms(document, forbidden_terms(["semantic_v2", "semantic_v1", "legacy"])) == []
    assert "Conversation A" in document and "Conversation B" in document

    mapping = json.loads((root / "blind_mapping.json").read_text())
    by_scenario = unblind_all(mapping)
    assert len(by_scenario) == 13
    for labels in by_scenario.values():
        assert set(labels.values()) == {"semantic_v2", "semantic_v1"}

    # And the key can be read back through the same command.
    unblinded = subprocess.run(
        [sys.executable, str(REVIEW_CLI), str(root), "--unblind"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert unblinded.returncode == 0
    assert "Conversation A = " in unblinded.stdout


def test_a_baseline_only_run_has_nothing_to_review_blindly(tmp_path):
    """Refused with a reason, rather than half a comparison.

    This is the state of the world until the runtime branch lands, so the
    message has to say what to do rather than what went wrong.
    """
    root, _ = _full_run(tmp_path, roles=(ROLE_BASELINE,))
    result = subprocess.run(
        [sys.executable, str(REVIEW_CLI), str(root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "one arm only" in result.stderr
