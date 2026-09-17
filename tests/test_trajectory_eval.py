"""Sprint 4 — evaluating whole conversations, and refusing to overclaim.

``docs/autonomy_architecture_review.md`` §5. Two things it asks for that are easy
to get wrong, and that these tests pin down:

    Count critical execution failures separately; a high average prose score
    cannot cancel an unauthorized transaction.

    A scripted cooperative customer and a model grading its own text are
    insufficient substitutes for expert human review.

So: every critical finding is deterministic — read off state or provenance,
never off prose — and the harness produces no quality score at all. A test below
asserts that no score exists, because adding one later is exactly the drift the
review warns against, and it would look like an improvement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.trajectory_eval import (
    Disturbance,
    Finding,
    Severity,
    Trajectory,
    TrajectoryReport,
    TurnRecord,
    evaluate_turns,
    load_trajectories,
    render_reports,
    run_trajectory,
)

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORIES = ROOT / "eval" / "trajectories.json"


def run(coro):
    import asyncio

    return asyncio.run(coro)


def _turn(index: int, *replies: str, **overrides) -> TurnRecord:
    record = TurnRecord(
        index=index,
        customer_message=overrides.pop("customer_message", f"message {index}"),
        replies=list(replies),
    )
    for key, value in overrides.items():
        setattr(record, key, value)
    return record


def _critical(findings: list[Finding]) -> list[str]:
    return [f.kind for f in findings if f.severity == Severity.CRITICAL]


def _notable(findings: list[Finding]) -> list[str]:
    return [f.kind for f in findings if f.severity == Severity.NOTABLE]


# --- critical failures are deterministic ------------------------------------


def test_a_paid_delivery_with_no_receipt_is_a_critical_failure():
    """§4: a delivery claim is tied to the operation result."""
    turn = _turn(0, "here you go")
    turn.provenance = [
        {"delivery": {"kind": "ppv", "accepted_by_platform": False}}
    ]

    assert "unreceipted_paid_delivery" in _critical(evaluate_turns([turn], []))


def test_a_paid_delivery_with_a_receipt_is_not_a_finding():
    turn = _turn(0, "here you go")
    turn.provenance = [
        {
            "delivery": {
                "kind": "ppv",
                "accepted_by_platform": True,
                "platform_message_id": "p1",
            }
        }
    ]

    assert _critical(evaluate_turns([turn], [])) == []


def test_the_same_platform_message_recorded_twice_is_a_critical_failure():
    """The most damaging thing this product can do in front of an agency."""
    turns = []
    for index in (0, 1):
        turn = _turn(index, "here you go")
        turn.provenance = [
            {
                "delivery": {
                    "kind": "ppv",
                    "accepted_by_platform": True,
                    "platform_message_id": "same-message",
                }
            }
        ]
        turns.append(turn)

    assert "duplicate_delivery" in _critical(evaluate_turns(turns, []))


def test_two_different_deliveries_are_not_a_duplicate():
    turns = []
    for index, message_id in enumerate(("p1", "p2")):
        turn = _turn(index, "here you go")
        turn.provenance = [
            {
                "delivery": {
                    "kind": "ppv",
                    "accepted_by_platform": True,
                    "platform_message_id": message_id,
                }
            }
        ]
        turns.append(turn)

    assert _critical(evaluate_turns(turns, [])) == []


def test_a_message_after_he_asked_for_none_is_a_critical_failure():
    """§1: respect a goodbye or a request for no follow-up."""
    disturbances = [
        Disturbance(message="night, dont message me tonight", asks_for_silence=True),
        Disturbance(message=""),
    ]
    turns = [_turn(0, "night you"), _turn(1, "hey are you up")]

    assert "message_after_silence_requested" in _critical(
        evaluate_turns(turns, disturbances)
    )


def test_staying_quiet_after_a_goodbye_produces_no_finding():
    disturbances = [
        Disturbance(message="night", asks_for_silence=True),
        Disturbance(message=""),
    ]
    turns = [_turn(0, "night you"), _turn(1)]

    assert evaluate_turns(turns, disturbances) == []


def test_reasserting_something_he_corrected_is_a_critical_failure():
    """§5: preference correction; catches reasserting superseded information."""
    disturbances = [
        Disturbance(message="not the outdoor ones", corrects="outdoor"),
        Disturbance(message="so anyway"),
    ]
    turns = [_turn(0, "got it"), _turn(1, "you would love the outdoor set")]

    assert "reasserted_corrected_information" in _critical(
        evaluate_turns(turns, disturbances)
    )


def test_respecting_a_correction_produces_no_finding():
    disturbances = [
        Disturbance(message="not the outdoor ones", corrects="outdoor"),
        Disturbance(message="so anyway"),
    ]
    turns = [_turn(0, "got it, hotel ones"), _turn(1, "the hotel set then")]

    assert _critical(evaluate_turns(turns, disturbances)) == []


def test_a_turn_that_raised_is_recorded_as_a_critical_failure():
    turn = _turn(0)
    turn.error = "RuntimeError: writer stack unavailable"

    assert "turn_raised" in _critical(evaluate_turns([turn], []))


# --- notable behaviours are for a person, not a verdict ---------------------


def test_an_unanswered_obligation_is_notable_rather_than_critical():
    """Matching an obligation against prose is a keyword test.

    Strong enough to put a turn in front of a person. Not strong enough to call
    something a failure with certainty, and the harness must not pretend
    otherwise.
    """
    disturbances = [
        Disturbance(message="do you ever get to chicago", raises_obligation="chicago"),
        Disturbance(message="anyway"),
    ]
    turns = [_turn(0, "hey"), _turn(1, "how was your day")]

    findings = evaluate_turns(turns, disturbances)
    assert "obligation_never_addressed" in _notable(findings)
    assert _critical(findings) == []


def test_an_answered_obligation_produces_no_finding():
    disturbances = [
        Disturbance(message="do you ever get to chicago", raises_obligation="chicago"),
        Disturbance(message="anyway"),
    ]
    turns = [_turn(0, "hey"), _turn(1, "i did chicago once actually")]

    assert evaluate_turns(turns, disturbances) == []


def test_a_question_in_every_single_reply_is_notable():
    """§1: without forcing a question into every reply."""
    turns = [_turn(i, f"thing {i}. what about you?") for i in range(5)]

    assert "question_in_every_reply" in _notable(evaluate_turns(turns, []))


def test_a_conversation_with_some_statements_is_not_flagged():
    turns = [_turn(0, "how was it?"), _turn(1, "that sounds rough"),
             _turn(2, "what did you do?"), _turn(3, "nice.")]

    assert "question_in_every_reply" not in _notable(evaluate_turns(turns, []))


def test_the_same_reply_sent_twice_is_notable():
    line = "that sounds like it was a really long week for you honestly"
    turns = [_turn(0, line), _turn(1, "mm"), _turn(2, line)]

    assert "verbatim_repetition" in _notable(evaluate_turns(turns, []))


def test_a_short_acknowledgement_repeating_is_ordinary():
    turns = [_turn(0, "mm"), _turn(1, "mm")]

    assert evaluate_turns(turns, []) == []


# --- the report refuses to produce a verdict --------------------------------


def test_the_report_has_no_quality_score():
    """Adding one later would look like an improvement. It is the thing §5 rules out."""
    report = TrajectoryReport(trajectory="t")
    summary = report.summary()

    for forbidden in ("score", "quality", "grade", "rating", "pass", "passed"):
        assert forbidden not in summary, (
            f"{forbidden!r} in the summary would be a model or a script grading "
            "a conversation, which the review says is not a substitute for "
            "expert human review"
        )


def test_the_rendered_run_says_what_it_does_not_establish():
    report = TrajectoryReport(trajectory="t", covers="something")

    rendered = render_reports([report])

    assert "No quality score is produced" in rendered
    assert "expert review" in rendered


def test_critical_failures_are_totalled_on_their_own_line():
    report = TrajectoryReport(
        trajectory="t",
        findings=[
            Finding(Severity.CRITICAL, "duplicate_delivery", "twice", 1),
            Finding(Severity.NOTABLE, "verbatim_repetition", "twice", 2),
        ],
    )

    rendered = render_reports([report])

    assert "1 CRITICAL execution failure(s)" in rendered
    assert len(report.critical) == 1
    assert len(report.notable) == 1


def test_operator_rescues_are_counted_as_a_cost_not_a_failure():
    """PR #48 made one class of handoff the correct behaviour."""
    report = TrajectoryReport(
        trajectory="t",
        turns=[_turn(0, outcome="human_review"), _turn(1, "hey", outcome="replied")],
    )

    assert report.operator_rescues == 1
    assert report.critical == []


def test_the_tail_latency_is_reported_rather_than_averaged_away():
    """§5 asks for tail behaviour, which an average hides."""
    report = TrajectoryReport(
        trajectory="t",
        turns=[
            _turn(0, "a", latency_ms=100),
            _turn(1, "b", latency_ms=120),
            _turn(2, "c", latency_ms=9000),
        ],
    )

    latency = report.latency()
    assert latency["worst_ms"] == 9000
    assert latency["median_ms"] < 1000


def test_the_report_says_which_models_actually_answered():
    """Finding H over a whole conversation."""
    turn = _turn(0, "hey")
    turn.provenance = [
        {"writer": {"actual": {"model": "Qwen/Qwen3.7-Plus"}}},
    ]
    other = _turn(1, "hi")
    other.provenance = [{"writer": {"actual": {"model": "moonshotai/kimi-k2.6"}}}]

    report = TrajectoryReport(trajectory="t", turns=[turn, other])

    assert report.models_used() == {
        "Qwen/Qwen3.7-Plus": 1,
        "moonshotai/kimi-k2.6": 1,
    }


def test_a_reply_with_no_provenance_is_reported_as_unrecorded():
    report = TrajectoryReport(trajectory="t", turns=[_turn(0, "hey")])

    assert report.models_used() == {}


# --- running one ------------------------------------------------------------


def test_a_trajectory_drives_every_disturbance_in_order():
    seen: list[str] = []

    async def send_turn(message: str) -> dict:
        seen.append(message)
        return {"outcome": "replied", "creator_messages": [{"content": "ok"}]}

    trajectory = Trajectory(
        name="t",
        disturbances=(
            Disturbance(message="one"),
            Disturbance(message="two"),
            Disturbance(message="three"),
        ),
    )

    report = run(run_trajectory(trajectory, send_turn=send_turn))

    assert seen == ["one", "two", "three"]
    assert len(report.turns) == 3


def test_a_turn_that_raises_does_not_stop_the_trajectory():
    """§5 asks for tail behaviour during errors, which an abort cannot measure."""

    async def send_turn(message: str) -> dict:
        if message == "two":
            raise RuntimeError("writer stack unavailable")
        return {"outcome": "replied", "creator_messages": [{"content": "ok"}]}

    trajectory = Trajectory(
        name="t",
        disturbances=(
            Disturbance(message="one"),
            Disturbance(message="two"),
            Disturbance(message="three"),
        ),
    )

    report = run(run_trajectory(trajectory, send_turn=send_turn))

    assert len(report.turns) == 3
    assert report.turns[1].error
    assert report.turns[2].replies == ["ok"]
    assert "turn_raised" in _critical(report.findings)


def test_an_absence_advances_the_clock_rather_than_sleeping():
    advanced: list[float] = []

    async def send_turn(message: str) -> dict:
        return {"outcome": "replied", "creator_messages": []}

    trajectory = Trajectory(
        name="t",
        disturbances=(
            Disturbance(message="one"),
            Disturbance(message="two", days_since_previous=7),
        ),
    )

    run(
        run_trajectory(
            trajectory, send_turn=send_turn, advance_clock=advanced.append
        )
    )

    assert advanced == [7.0]


def test_an_adaptive_customer_sees_what_the_creator_said():
    """§5: adaptive trajectories expose the consequences of earlier choices."""
    trajectory = Trajectory(
        name="t",
        disturbances=(
            Disturbance(message="hey"),
            Disturbance(responds_to=lambda replies: f"you said: {replies[-1]}"),
        ),
    )
    seen: list[str] = []

    async def send_turn(message: str) -> dict:
        seen.append(message)
        return {"outcome": "replied", "creator_messages": [{"content": "hello there"}]}

    run(run_trajectory(trajectory, send_turn=send_turn))

    assert seen[1] == "you said: hello there"


def test_a_trajectory_collects_each_replys_provenance():
    async def send_turn(message: str) -> dict:
        from services.reply_provenance import PROVENANCE_KEY

        return {
            "outcome": "replied",
            "creator_messages": [
                {
                    "content": "ok",
                    "media_context": {
                        PROVENANCE_KEY: {"turn_id": "t1", "delivery": {"kind": "text"}}
                    },
                }
            ],
        }

    report = run(
        run_trajectory(
            Trajectory(name="t", disturbances=(Disturbance(message="hey"),)),
            send_turn=send_turn,
        )
    )

    assert report.turns[0].provenance[0]["turn_id"] == "t1"


# --- the shipped trajectories -----------------------------------------------


def test_every_shipped_trajectory_says_which_review_row_it_covers():
    raw = json.loads(TRAJECTORIES.read_text(encoding="utf-8"))["trajectories"]

    for trajectory in raw:
        assert trajectory.get("covers"), (
            f"{trajectory.get('name')!r} does not say what it is testing"
        )


def test_the_shipped_trajectories_cover_the_reviews_table():
    raw = json.loads(TRAJECTORIES.read_text(encoding="utf-8"))["trajectories"]
    covered = " ".join(str(t.get("covers", "")) for t in raw).lower()

    for row in (
        "ordinary conversation",
        "one day and one week",
        "deferred across 30+ turns",
        "preference correction",
        "multiple open threads",
        "content-access support",
        "goodbye",
        "ambiguous pronouns",
    ):
        assert row in covered, f"no trajectory covers {row!r}"


def test_the_shipped_trajectories_load():
    trajectories = load_trajectories(
        json.loads(TRAJECTORIES.read_text(encoding="utf-8"))["trajectories"]
    )

    assert len(trajectories) >= 8
    assert all(trajectory.disturbances for trajectory in trajectories)


def test_the_goodbye_trajectory_marks_the_turn_that_asks_for_silence():
    trajectories = load_trajectories(
        json.loads(TRAJECTORIES.read_text(encoding="utf-8"))["trajectories"]
    )
    goodbye = next(t for t in trajectories if "goodnight" in t.name)

    assert any(d.asks_for_silence for d in goodbye.disturbances)


def test_the_correction_trajectory_declares_what_was_corrected():
    trajectories = load_trajectories(
        json.loads(TRAJECTORIES.read_text(encoding="utf-8"))["trajectories"]
    )
    correction = next(t for t in trajectories if "corrects a preference" in t.name)

    assert any(d.corrects for d in correction.disturbances)


def test_no_trajectory_contains_explicit_material():
    """§5 asks for the longitudinal evaluation to be non-explicit."""
    raw = TRAJECTORIES.read_text(encoding="utf-8").lower()

    for word in ("nude", "nudes", "pussy", "cock", "cum", "fuck"):
        assert word not in raw


def test_the_file_states_that_its_customer_is_scripted():
    """A script cannot expose the consequences of the system's own choices.

    §5 asks for adaptive trajectories as well, and saying so in the file is what
    stops a green run being read as more than it is.
    """
    raw = TRAJECTORIES.read_text(encoding="utf-8")

    assert "SCRIPTED" in raw
    assert "adaptive" in raw


@pytest.mark.parametrize("required", ["proposed test design", "non-explicit"])
def test_the_file_repeats_the_reviews_own_caveat(required):
    assert required in TRAJECTORIES.read_text(encoding="utf-8")
