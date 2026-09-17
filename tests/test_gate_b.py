"""Gate B: the runner actually detects what it claims to detect.

    demonstrate that injected duplicate sends, ignored no-follow-up
    preferences, wrong references and lost continuity are actually detected by
    the runner.
                        — docs/continuation_brief_2026-09-17.md, Phase B

This is evidence about the HARNESS, not about the product. A detector that has
never seen the thing it looks for is one nobody has reason to trust, and this
repository had two firing on the wrong evidence for as long as they existed:
a reply honouring a correction was reported as reasserting it, and a question
answered in the same breath was reported as never addressed. Both passed every
test they had.

So every case here is a PAIR:

    inject the fault  -> the finding appears
    run it clean      -> the finding does not

Both halves are load-bearing. A detector that always fires is as useless as one
that never does, and only the pair tells them apart. The pairing is what the
old tests lacked.

What this does NOT establish, and the brief says so directly: a complete mock
run proves wiring. It says nothing about live model quality, and nothing here
should be read as evidence of it.
"""

from __future__ import annotations

import asyncio

import pytest

from services.trajectory_eval import (
    Disturbance,
    Severity,
    Trajectory,
    run_trajectory,
)
from services.trajectory_faults import (
    DeliberatelyQuiet,
    DuplicateSend,
    IgnoresSilence,
    LosesContinuity,
    Pipeline,
    ProviderFailure,
    SilentWithNoReason,
    UnreceiptedDelivery,
    WrongReference,
)


def _run(trajectory: Trajectory, pipeline) -> list[str]:
    report = asyncio.run(run_trajectory(trajectory, send_turn=pipeline))
    return [finding.kind for finding in report.findings]


def _kinds(trajectory: Trajectory, faulty, clean=None) -> tuple[list[str], list[str]]:
    """Findings from a faulty run and from a clean run of the same trajectory."""
    return _run(trajectory, faulty), _run(trajectory, clean or Pipeline())


ORDINARY = Trajectory(
    name="ordinary",
    disturbances=(
        Disturbance(message="hey how are you"),
        Disturbance(message="what have you been up to"),
        Disturbance(message="that sounds nice"),
    ),
)


# ===========================================================================
# The four the brief names
# ===========================================================================


def test_an_injected_duplicate_send_is_detected():
    faulty, clean = _kinds(ORDINARY, DuplicateSend(duplicate_on_turn=2))

    assert "duplicate_delivery" in faulty
    assert "duplicate_delivery" not in clean


def test_an_ignored_no_follow_up_preference_is_detected():
    """He asks for no messages; the next turn is one he did not write."""
    trajectory = Trajectory(
        name="silence",
        disturbances=(
            Disturbance(message="dont message me tonight", asks_for_silence=True),
            Disturbance(message=""),
            Disturbance(message=""),
        ),
    )

    faulty, clean = _kinds(trajectory, IgnoresSilence(), DeliberatelyQuiet())

    assert "unprompted_message_after_silence_requested" in faulty
    assert "unprompted_message_after_silence_requested" not in clean


def test_answering_him_when_he_writes_first_is_still_not_a_violation():
    """The false positive that made this detector unreadable, held down.

    A customer who asks for quiet and then comes back and writes is not being
    messaged. If this ever fires, the detector has regressed to flagging every
    turn after the request.
    """
    trajectory = Trajectory(
        name="he came back",
        disturbances=(
            Disturbance(message="dont message me for a while", asks_for_silence=True),
            Disturbance(message="ok im back, what do you have", days_since_previous=7.0),
        ),
    )

    findings = _run(trajectory, Pipeline())

    assert "unprompted_message_after_silence_requested" not in findings


def test_a_wrong_reference_is_detected():
    """Delivering the item he ruled out."""
    trajectory = Trajectory(
        name="correction",
        disturbances=(
            Disturbance(
                message="not the outdoor ones",
                corrects="outdoor",
                forbids_delivery_of=("outdoor-set-1",),
            ),
            Disturbance(message="what else is there"),
            Disturbance(message="ok"),
        ),
    )

    faulty, clean = _kinds(trajectory, WrongReference(reference="outdoor-set-1"))

    assert "delivery_contradicts_correction" in faulty
    assert "delivery_contradicts_correction" not in clean


def test_delivering_something_he_did_not_rule_out_is_not_flagged():
    """Otherwise every PPV in a corrected conversation is a critical failure."""
    trajectory = Trajectory(
        name="correction",
        disturbances=(
            Disturbance(
                message="not the outdoor ones",
                corrects="outdoor",
                forbids_delivery_of=("outdoor-set-1",),
            ),
            Disturbance(message="what else is there"),
            Disturbance(message="ok"),
        ),
    )

    findings = _run(trajectory, WrongReference(reference="hotel-set-2"))

    assert "delivery_contradicts_correction" not in findings


def test_lost_continuity_is_detected():
    """An obligation raised, and every later reply generic."""
    trajectory = Trajectory(
        name="continuity",
        disturbances=(
            Disturbance(message="do you ever get to chicago",
                        raises_obligation="chicago"),
            Disturbance(message="anyway, long week"),
            Disturbance(message="mm"),
        ),
    )

    faulty, clean = _kinds(trajectory, LosesContinuity())

    assert "obligation_never_addressed" in faulty
    assert "obligation_never_addressed" not in clean


# ===========================================================================
# The rest of the detector set, on the same terms
# ===========================================================================


def test_a_provider_failure_is_recorded_rather_than_ending_the_run():
    """§5 asks for tail behaviour during errors, which needs the run to continue."""
    report = asyncio.run(
        run_trajectory(ORDINARY, send_turn=ProviderFailure(fail_on_turn=2))
    )

    assert "turn_raised" in [f.kind for f in report.findings]
    assert len(report.turns) == 3, "the conversation continues past the failure"
    assert report.turns[2].replies, "and recovers"


def test_a_paid_delivery_the_platform_never_took_is_detected():
    faulty, clean = _kinds(ORDINARY, UnreceiptedDelivery())

    assert "unreceipted_paid_delivery" in faulty
    assert "unreceipted_paid_delivery" not in clean


def test_a_turn_that_explains_nothing_is_detected():
    """Missing evidence is unknown, and unknown is never a pass."""
    faulty, clean = _kinds(ORDINARY, SilentWithNoReason(), DeliberatelyQuiet())

    assert "turn_outcome_unknown" in faulty
    assert "turn_outcome_unknown" not in clean


def test_choosing_silence_is_not_mistaken_for_failing_silently():
    """The distinction the old single silent_turns count could not make."""
    report = asyncio.run(run_trajectory(ORDINARY, send_turn=DeliberatelyQuiet()))

    assert report.outcomes() == {"chose_silence": 3}
    assert report.failed_turns == 0
    assert report.unexplained_turns == 0


def test_a_run_that_explains_nothing_is_all_failure():
    report = asyncio.run(run_trajectory(ORDINARY, send_turn=SilentWithNoReason()))

    assert report.outcomes() == {"unknown": 3}
    assert report.failed_turns == 3
    assert report.unexplained_turns == 3


# ===========================================================================
# The clean baseline, and what it is evidence of
# ===========================================================================


def test_a_cooperative_conversation_produces_no_findings_at_all():
    """The control for every pair above.

    If this ever starts producing findings, every "not in clean" assertion
    above becomes vacuous — they would be passing because the detector is
    noisy rather than because the fault was absent.
    """
    report = asyncio.run(run_trajectory(ORDINARY, send_turn=Pipeline()))

    assert report.findings == []
    assert report.critical == []
    assert report.outcomes() == {"replied": 3}


def test_the_faults_are_told_apart_from_each_other():
    """A detector that fires on every fault is not detecting anything.

    Each injected fault must produce ITS finding and not the others', or the
    pairs above are measuring one indiscriminate alarm.
    """
    cases = {
        "duplicate_delivery": DuplicateSend(duplicate_on_turn=2),
        "turn_raised": ProviderFailure(fail_on_turn=2),
        "unreceipted_paid_delivery": UnreceiptedDelivery(),
    }

    for expected, pipeline in cases.items():
        kinds = set(_run(ORDINARY, pipeline))
        assert expected in kinds, expected
        others = set(cases) - {expected}
        assert not (kinds & others), f"{expected} also produced {kinds & others}"


def test_a_critical_finding_is_never_cancelled_by_the_rest_of_the_run():
    """§5: a high average prose score cannot cancel an unauthorized transaction.

    Two turns of perfectly ordinary conversation around one duplicate send
    still leave the duplicate critical and countable.
    """
    report = asyncio.run(
        run_trajectory(ORDINARY, send_turn=DuplicateSend(duplicate_on_turn=2))
    )

    assert [f.severity for f in report.critical] == [Severity.CRITICAL]
    assert report.summary()["critical_failures"] == 1


@pytest.mark.parametrize(
    "pipeline,expected",
    [
        (DuplicateSend(duplicate_on_turn=2), "duplicate_delivery"),
        (UnreceiptedDelivery(), "unreceipted_paid_delivery"),
        (SilentWithNoReason(), "turn_outcome_unknown"),
    ],
)
def test_each_fault_is_detected_across_a_longer_conversation(pipeline, expected):
    """Faults do not stop being visible because the conversation is long.

    A detector that only works on three turns is not one a longitudinal
    evaluation can use.
    """
    long_form = Trajectory(
        name="long",
        disturbances=tuple(
            Disturbance(message=f"turn {index}, anything new") for index in range(30)
        ),
    )

    assert expected in _run(long_form, pipeline)
