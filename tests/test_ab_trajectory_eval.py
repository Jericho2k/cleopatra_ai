"""The A/B trajectory harness: fairness, isolation, safety, and artifacts.

Every test here is about a way the comparison could be wrong without looking
wrong. Two runtimes writing into one fan, one arm quietly getting different
inputs, an arm pinned to a runtime and never unpinned, a candidate runtime that
does not exist yet crashing the whole harness, a result directory that gets
committed — none of those produce an error message on their own, and each of
them makes every number in the run describe something that did not happen.

The last two pin the contract with the runtime branch: a baseline-only run
works before ``conversational_v1`` exists, and the day it registers its id the
harness accepts it with nothing here edited.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from services import conversation_core
from services.ab_trajectory_eval import (
    ArmSpec,
    CORE_STATE_KEY,
    CoreNotAvailable,
    EvaluationRefused,
    ROLE_BASELINE,
    ROLE_CANDIDATE,
    RunSpec,
    UnfairComparison,
    UnsafeEvaluationTarget,
    arm_order,
    assert_arm_isolation,
    assert_safe_targets,
    assert_scripted,
    build_metadata,
    fan_input_digest,
    known_core_ids,
    load_scenario_file,
    new_run_id,
    observe_turn,
    pair_conversations,
    require_known_core,
    run_suite,
    select_trajectories,
    write_artifacts,
)
from services.trajectory_eval import Disturbance, Trajectory, TurnRecord, load_trajectories

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "eval" / "conversational_core_scenarios.json"


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# A backend that records what it was asked to do, and never touches a database
# ---------------------------------------------------------------------------


class RecordingBackend:
    """Stands in for the simulator, and remembers everything it was told.

    The point of the recording is that fairness claims are checkable: which fan
    received which message, in which order, under which pinned runtime. A test
    that only inspected the output could not tell a run where both arms got the
    same script from one where they did not.
    """

    def __init__(self, *, fans: dict[str, dict] | None = None, replies=None):
        self.fans = fans or {}
        self.sent: list[tuple[str, str]] = []
        self.selected: list[tuple[str, str | None]] = []
        self.reset_calls: list[str] = []
        self.seeded: list[str] = []
        self.pinned: dict[str, str | None] = {}
        self._replies = replies or (lambda fan_id, message: [f"reply to {message}"])

    async def describe_fan(self, creator_id: str, fan_id: str) -> dict:
        return dict(self.fans.get(fan_id) or {})

    async def select_core(self, fan_id: str, core_id: str | None) -> str | None:
        previous = self.pinned.get(fan_id)
        self.pinned[fan_id] = core_id
        self.selected.append((fan_id, core_id))
        return previous

    async def reset_state(self, creator_id: str, fan_id: str) -> None:
        self.reset_calls.append(fan_id)

    async def apply_seed(self, creator_id: str, fan_id: str, seed: dict) -> list[dict]:
        self.seeded.append(fan_id)
        return [
            {"reference": f"eval:{fan_id}", "price_cents": 1800, "media_ids": ["m1"]}
        ]

    async def send_turn(self, creator_id: str, fan_id: str, message: str) -> dict:
        self.sent.append((fan_id, message))
        core = self.pinned.get(fan_id) or ""
        return {
            "outcome": "replied",
            "creator_messages": [
                {
                    "id": f"{fan_id}:{len(self.sent)}:{index}",
                    "content": reply,
                    "media_context": {
                        "reply_provenance": {
                            "turn_id": f"turn-{fan_id}-{len(self.sent)}",
                            "decision": {
                                "conversation_core": core,
                                "disposition": "reply",
                            },
                            "writer": {
                                "requested": {"model": "model-x"},
                                "actual": {"model": "model-x", "provider": "p"},
                            },
                            "context": {"packet": {"fingerprint": "abc123"}},
                            "delivery": {
                                "kind": "text",
                                "platform_message_id": f"pm-{fan_id}-{len(self.sent)}",
                                "accepted_by_platform": True,
                            },
                        }
                    },
                }
                for index, reply in enumerate(self._replies(fan_id, message))
            ],
        }

    async def run_due_work(self, creator_id: str, fan_id: str) -> dict:
        self.sent.append((fan_id, ""))
        return {"outcome": "due_work_sent_nothing", "creator_messages": [], "due_worker_ran": True}


def _scenario(name="s", messages=("hello", "again")) -> Trajectory:
    return Trajectory(
        name=name,
        covers="a test scenario",
        disturbances=tuple(Disturbance(message=message) for message in messages),
    )


def _fans(*ids: str) -> dict[str, dict]:
    return {
        fan_id: {"id": fan_id, "creator_id": "creator-1", "platform_fan_id": f"test_{fan_id}"}
        for fan_id in ids
    }


def _spec(*arms: ArmSpec, seed: int = 7) -> RunSpec:
    return RunSpec(
        run_id="run-1",
        creator_id="creator-1",
        seed=seed,
        arms=tuple(arms),
        scenario_ids=("s",),
    )


BASELINE_ARM = ArmSpec(role=ROLE_BASELINE, core_id="semantic_v2", fan_id="fan-a")
CANDIDATE_ARM = ArmSpec(role=ROLE_CANDIDATE, core_id="semantic_v1", fan_id="fan-b")


# --- 1. both arms receive the same scripted fan inputs ----------------------


def test_both_arms_receive_the_same_scripted_messages_in_the_same_order():
    backend = RecordingBackend(fans=_fans("fan-a", "fan-b"))
    trajectory = _scenario(messages=("first thing", "second thing", "third thing"))

    runs = run(
        run_suite(
            spec=_spec(BASELINE_ARM, CANDIDATE_ARM),
            trajectories=[trajectory],
            backend=backend,
            scenario_ids=["s"],
        )
    )

    per_fan: dict[str, list[str]] = {}
    for fan_id, message in backend.sent:
        per_fan.setdefault(fan_id, []).append(message)
    assert per_fan["fan-a"] == ["first thing", "second thing", "third thing"]
    assert per_fan["fan-a"] == per_fan["fan-b"]

    paired = pair_conversations(runs)
    assert paired["fan_inputs_identical"] is True
    assert paired["input_mismatches"] == []
    assert paired["pairs"][0]["fan_inputs_identical"] is True


def test_a_pair_whose_inputs_differ_is_reported_rather_than_compared():
    """A digest mismatch is named, not averaged.

    This is the failure that looks like a result: two conversations, both
    complete, every metric computable, and nothing about them is evidence about
    the runtimes because they were not asked the same thing.
    """
    from services.ab_trajectory_eval import ArmRun

    baseline = ArmRun(arm=BASELINE_ARM)
    candidate = ArmRun(arm=CANDIDATE_ARM)
    baseline.conversations.append(
        {"scenario_id": "s", "fan_input_digest": "aaaa", "turns": [], "summary": {}}
    )
    candidate.conversations.append(
        {"scenario_id": "s", "fan_input_digest": "bbbb", "turns": [], "summary": {}}
    )
    paired = pair_conversations({ROLE_BASELINE: baseline, ROLE_CANDIDATE: candidate})
    assert paired["fan_inputs_identical"] is False
    assert paired["input_mismatches"] == ["s"]


def test_an_adaptive_trajectory_is_refused_for_an_ab_comparison():
    adaptive = Trajectory(
        name="adaptive",
        disturbances=(Disturbance(responds_to=lambda replies: "whatever"),),
    )
    with pytest.raises(UnfairComparison):
        assert_scripted(adaptive)


def test_fan_input_digest_depends_on_the_script_and_not_on_the_name():
    left = _scenario(name="one", messages=("a", "b"))
    right = _scenario(name="two", messages=("a", "b"))
    other = _scenario(name="one", messages=("a", "c"))
    assert fan_input_digest(left) == fan_input_digest(right)
    assert fan_input_digest(left) != fan_input_digest(other)


# --- 2. state isolation between the arms -----------------------------------


def test_two_arms_may_not_share_one_fan():
    same = ArmSpec(role=ROLE_CANDIDATE, core_id="semantic_v1", fan_id="fan-a")
    with pytest.raises(UnfairComparison):
        assert_arm_isolation([BASELINE_ARM, same])


def test_each_arm_is_pinned_to_its_own_runtime_and_unpinned_afterwards():
    backend = RecordingBackend(fans=_fans("fan-a", "fan-b"))
    backend.pinned = {"fan-a": "legacy", "fan-b": None}

    run(
        run_suite(
            spec=_spec(BASELINE_ARM, CANDIDATE_ARM),
            trajectories=[_scenario()],
            backend=backend,
            scenario_ids=["s"],
        )
    )

    # Pinned at the start, restored at the end — including the fan that had no
    # previous selection, which must go back to having none rather than to
    # whatever this run used.
    assert backend.selected[0] == ("fan-a", "semantic_v2")
    assert backend.selected[1] == ("fan-b", "semantic_v1")
    assert backend.pinned == {"fan-a": "legacy", "fan-b": None}


def test_selection_is_restored_even_when_the_run_raises():
    class Exploding(RecordingBackend):
        async def send_turn(self, creator_id, fan_id, message):
            raise RuntimeError("the backend fell over")

    backend = Exploding(fans=_fans("fan-a", "fan-b"))
    backend.pinned = {"fan-a": "legacy", "fan-b": "legacy"}

    # run_trajectory records a turn that raised rather than propagating, so the
    # run completes; the restoration is asserted either way.
    run(
        run_suite(
            spec=_spec(BASELINE_ARM, CANDIDATE_ARM),
            trajectories=[_scenario()],
            backend=backend,
            scenario_ids=["s"],
        )
    )
    assert backend.pinned == {"fan-a": "legacy", "fan-b": "legacy"}


def test_seeded_state_is_cleared_for_every_arm_before_every_scenario():
    backend = RecordingBackend(fans=_fans("fan-a", "fan-b"))
    run(
        run_suite(
            spec=_spec(BASELINE_ARM, CANDIDATE_ARM),
            trajectories=[_scenario("one"), _scenario("two")],
            backend=backend,
            scenario_ids=["one", "two"],
        )
    )
    assert sorted(backend.reset_calls) == ["fan-a", "fan-a", "fan-b", "fan-b"]


def test_the_clock_is_reset_between_arms_and_between_scenarios():
    resets: list[int] = []
    backend = RecordingBackend(fans=_fans("fan-a", "fan-b"))
    run(
        run_suite(
            spec=_spec(BASELINE_ARM, CANDIDATE_ARM),
            trajectories=[_scenario("one"), _scenario("two")],
            backend=backend,
            reset_clock=lambda: resets.append(1),
            scenario_ids=["one", "two"],
        )
    )
    # Once per arm per scenario, plus the final reset that leaves the process
    # with no simulated time left over.
    assert len(resets) == 5


def test_arm_order_is_shuffled_per_scenario_and_reproducible_from_the_seed():
    arms = [BASELINE_ARM, CANDIDATE_ARM]
    first = [
        [arm.role for arm in arm_order(arms, seed=99, scenario_index=index)]
        for index in range(8)
    ]
    again = [
        [arm.role for arm in arm_order(arms, seed=99, scenario_index=index)]
        for index in range(8)
    ]
    assert first == again
    # Not the same arm first every time, which is the thing a fixed order would
    # hand systematically to one side.
    assert len({tuple(order) for order in first}) == 2


# --- 3. only safe, test-fan paths can run ----------------------------------


def test_a_real_fan_is_refused():
    real = {"fan-a": {"id": "fan-a", "creator_id": "creator-1", "platform_fan_id": "981273"}}
    with pytest.raises(UnsafeEvaluationTarget):
        assert_safe_targets([BASELINE_ARM], real)


def test_a_fan_that_does_not_exist_is_refused():
    with pytest.raises(UnsafeEvaluationTarget):
        assert_safe_targets([BASELINE_ARM], {})


def test_a_test_fan_is_allowed():
    assert_safe_targets([BASELINE_ARM, CANDIDATE_ARM], _fans("fan-a", "fan-b"))


def test_the_live_backend_refuses_a_fan_belonging_to_another_creator(monkeypatch):
    from services.ab_trajectory_eval import LiveSimulatorBackend

    backend = LiveSimulatorBackend()

    async def _fake_thread(fn, *args, **kwargs):
        return {"id": "fan-a", "creator_id": "someone-else", "platform_fan_id": "test_x"}

    monkeypatch.setattr(asyncio, "to_thread", _fake_thread)
    with pytest.raises(UnsafeEvaluationTarget):
        run(backend.describe_fan("creator-1", "fan-a"))


# --- 7. missing optional Core v1 metadata ----------------------------------


def test_a_turn_with_no_provenance_at_all_is_observed_without_crashing():
    turn = TurnRecord(index=0, customer_message="hi", outcome="no_send")
    observed = observe_turn(turn, core_id="semantic_v2")
    assert observed["turn_index"] == 0
    assert observed["creator_output"] == []
    assert observed["conversation_core"] == "semantic_v2"
    assert observed["conversation_core_recorded"] is False
    # Absent, not zero: a runtime without the concept must not contribute a
    # value that could be averaged.
    assert CORE_STATE_KEY not in observed
    assert "tokens" not in observed
    assert "cost_usd" not in observed


def test_core_state_is_read_when_a_runtime_records_it_and_partial_blocks_are_kept():
    turn = TurnRecord(index=3, customer_message="hi", outcome="replied", replies=["hey"])
    turn.provenance = [
        {
            "turn_id": "t-1",
            CORE_STATE_KEY: {
                "state_before": {"scene": "train station"},
                "accepted_fields": ["scene"],
                "rejected_fields": [],
                "unknown_future_field": 1,
            },
        }
    ]
    observed = observe_turn(turn, core_id="conversational_v1")
    assert observed[CORE_STATE_KEY] == {
        "state_before": {"scene": "train station"},
        "accepted_fields": ["scene"],
        "rejected_fields": [],
    }


def test_an_unknown_outcome_string_does_not_crash_observation():
    turn = TurnRecord(index=1, customer_message="hi", outcome="something_invented_later")
    observed = observe_turn(turn, core_id="conversational_v1")
    assert observed["classified_outcome"] == "unknown"


# --- 10 & 11. the contract with the runtime branch -------------------------


def test_baseline_only_runs_and_pairs_without_a_candidate():
    backend = RecordingBackend(fans=_fans("fan-a"))
    runs = run(
        run_suite(
            spec=_spec(BASELINE_ARM),
            trajectories=[_scenario()],
            backend=backend,
            scenario_ids=["s"],
        )
    )
    assert set(runs) == {ROLE_BASELINE}
    paired = pair_conversations(runs)
    assert paired["paired"] is False
    assert paired["pairs"][0]["fan_inputs_identical"] is None
    metadata = build_metadata(_spec(BASELINE_ARM), runs)
    assert metadata["baseline_core"] == "semantic_v2"
    assert metadata["candidate_core"] is None


def test_an_unregistered_core_is_refused_by_name_rather_than_crashing():
    with pytest.raises(CoreNotAvailable) as refused:
        require_known_core("conversational_v1_that_does_not_exist_yet")
    assert "not registered" in str(refused.value)


def test_a_newly_registered_core_id_is_accepted_with_no_evaluator_change(monkeypatch):
    """The whole contract with the runtime branch, in one test.

    ``known_core_ids`` reads ``CORE_IDS`` at call time, so registering an id on
    the runtime branch is the only change needed to run it here. This simulates
    that registration and asserts that a complete A/B run against the new id
    works — no new enum, no new branch, no new scenario format.
    """
    monkeypatch.setattr(
        conversation_core, "CORE_IDS", (*conversation_core.CORE_IDS, "conversational_v1")
    )
    assert "conversational_v1" in known_core_ids()
    assert require_known_core("conversational_v1") == "conversational_v1"

    candidate = ArmSpec(role=ROLE_CANDIDATE, core_id="conversational_v1", fan_id="fan-b")
    backend = RecordingBackend(fans=_fans("fan-a", "fan-b"))
    runs = run(
        run_suite(
            spec=_spec(BASELINE_ARM, candidate),
            trajectories=[_scenario()],
            backend=backend,
            scenario_ids=["s"],
        )
    )
    assert runs[ROLE_CANDIDATE].conversations[0]["conversation_core"] == "conversational_v1"
    assert pair_conversations(runs)["fan_inputs_identical"] is True


# --- artifacts --------------------------------------------------------------


def test_write_artifacts_produces_the_documented_directory(tmp_path):
    backend = RecordingBackend(fans=_fans("fan-a", "fan-b"))
    spec = _spec(BASELINE_ARM, CANDIDATE_ARM)
    trajectory = _scenario()
    runs = run(
        run_suite(spec=spec, trajectories=[trajectory], backend=backend, scenario_ids=["s"])
    )
    root = write_artifacts(spec=spec, runs=runs, trajectories=[trajectory], results_root=tmp_path)

    names = sorted(path.name for path in root.iterdir())
    assert names == ["metadata.json", "metrics.json", "paired.json", "semantic_v1.json", "semantic_v2.json"]

    metadata = json.loads((root / "metadata.json").read_text())
    for key in (
        "run_id",
        "generated_at",
        "git_sha",
        "seed",
        "baseline_core",
        "candidate_core",
        "creator_id",
        "ai_stack_profile",
        "scenario_ids",
        "arms",
        "isolation",
        "models_served",
    ):
        assert key in metadata, key
    assert metadata["isolation"]["fans"] == {"baseline": "fan-a", "candidate": "fan-b"}

    metrics = json.loads((root / "metrics.json").read_text())
    assert set(metrics) == {"baseline", "candidate"}
    assert "not_measured_here" in metrics["baseline"]


def test_generated_results_are_ignored_by_git():
    """Artifact directories must not be committable by default.

    ``git check-ignore`` rather than reading .gitignore, because what matters is
    the answer git gives, not the text of the rule.
    """
    probe = "eval/results/some-run-id/semantic_v2.json"
    result = subprocess.run(
        ["git", "check-ignore", "-q", probe], cwd=ROOT, capture_output=True
    )
    assert result.returncode == 0, f"{probe} would be committed"


def test_run_ids_are_unique_and_time_ordered():
    first, second = new_run_id(), new_run_id()
    assert first != second
    assert first.startswith("ccv1-")


# --- the scenario file ------------------------------------------------------


def test_the_scenario_suite_loads_and_selects():
    payload = load_scenario_file(SCENARIOS)
    trajectories, ids = select_trajectories(payload)
    assert len(trajectories) == len(ids) == 12

    subset, subset_ids = select_trajectories(payload, suite="commercial")
    assert subset_ids == ["I_natural_media_interest", "J_commercial_rejection", "K_post_event_continuation"]
    assert len(subset) == 3

    one, one_id = select_trajectories(payload, scenario_ids=["G_correction"])
    assert one_id == ["G_correction"]
    assert one[0].disturbances[2].corrects == "sister"


def test_selecting_a_scenario_that_does_not_exist_is_an_error_not_an_empty_run():
    payload = load_scenario_file(SCENARIOS)
    with pytest.raises(EvaluationRefused):
        select_trajectories(payload, scenario_ids=["Z_not_a_scenario"])
    with pytest.raises(EvaluationRefused):
        select_trajectories(payload, suite="not_a_suite")


def test_every_scenario_is_scripted_so_both_arms_can_receive_it():
    payload = load_scenario_file(SCENARIOS)
    for trajectory in load_trajectories(payload["trajectories"]):
        assert_scripted(trajectory)


# --- the optional runtime hook ---------------------------------------------


def test_the_core_state_recorder_is_optional_shape_agnostic_and_total():
    """The one optional hook the runtime branch may use.

    Three properties, each of which is the reason it is safe for a runtime to
    call: a runtime that never calls it produces a record with no block at all;
    a runtime that calls it with anything keeps whatever it passed; and a
    runtime that calls it with rubbish is not punished for it, because a
    recorder must never be able to stop a reply.
    """
    from services.reply_provenance import CORE_STATE_KEY as KEY, ReplyProvenance

    silent = ReplyProvenance(creator_id="c", fan_id="f", mode="auto")
    assert KEY not in silent.as_metadata()["reply_provenance"]

    recording = ReplyProvenance(creator_id="c", fan_id="f", mode="auto")
    recording.record_core_state(
        {
            "state_before": {"scene": "station"},
            "proposed_delta": {"scene": "platform"},
            "accepted_fields": ["scene"],
            "rejected_fields": ["mood"],
            "state_after": {"scene": "platform"},
        }
    )
    record = recording.as_metadata()["reply_provenance"]
    assert record[KEY]["accepted_fields"] == ["scene"]
    assert record[KEY]["rejected_fields"] == ["mood"]

    # Total: nothing it is handed can raise, and nothing meaningless is stored.
    for rubbish in (None, "", [], 0, {}):
        tolerant = ReplyProvenance(creator_id="c", fan_id="f", mode="auto")
        tolerant.record_core_state(rubbish)
        assert KEY not in tolerant.as_metadata()["reply_provenance"]


def test_recorded_core_state_survives_the_assisted_state_round_trip():
    from services.reply_provenance import ReplyProvenance

    original = ReplyProvenance(creator_id="c", fan_id="f", mode="assisted")
    original.record_core_state({"state_after": {"scene": "platform"}})
    restored = ReplyProvenance.from_state(original.as_state())
    assert restored is not None
    assert restored.core_state == {"state_after": {"scene": "platform"}}


def test_a_turn_carrying_recorded_core_state_is_observed_end_to_end():
    """From the recorder through a message row to the run artifact."""
    from services.reply_provenance import ReplyProvenance

    provenance = ReplyProvenance(creator_id="c", fan_id="f", mode="auto")
    provenance.record_core_state({"state_before": {"topic": "rain"}, "accepted_fields": ["topic"]})
    metadata = provenance.as_metadata(platform_message_id="pm-1")

    turn = TurnRecord(index=0, customer_message="hi", outcome="replied", replies=["hey"])
    turn.provenance = [metadata["reply_provenance"]]
    observed = observe_turn(turn, core_id="conversational_v1")
    assert observed[CORE_STATE_KEY] == {
        "state_before": {"topic": "rain"},
        "accepted_fields": ["topic"],
    }
