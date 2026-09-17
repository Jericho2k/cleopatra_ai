"""Sprint 3 — one statement of what a turn does, and a way to compare two.

``docs/autonomy_architecture_review.md`` finding F: several representations can
prescribe what a single conversation should do next, so nothing in the system
states what a turn is actually doing and nothing can compare two ways of
deciding it. §6.4 says to compare offline with the executor held fixed, and §4
warns that a shorter prompt must not be assumed better — the effect is tested
under replay or it is not established.

These tests hold four things:

1. the decision type cannot reacquire the prescriptive vocabulary it replaces;
2. the deterministic checks refuse what §4 says must always be refused, whoever
   produced it;
3. the projection describes today's behaviour honestly, including where today's
   behaviour decides nothing at all;
4. the replay gives every candidate the same evidence and counts critical
   failures separately from everything else.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.conversation_decision import (
    FORBIDDEN_FIELDS,
    ConversationDecision,
    HoldReason,
    OperationKind,
    ProposedOperation,
    deterministic_violations,
)
from services.context_packet import build_context_packet
from services.decision_owners import (
    CurrentStackOwner,
    SemanticDecisionOwner,
    build_semantic_prompt,
    current_stack_decision,
    parse_semantic_decision,
)
from services.decision_replay import (
    ReplayTurn,
    compare_owners_sync,
    load_turns,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "eval" / "decision_scenarios.json"


def _packet(threads=(), history=()):
    return build_context_packet(history, open_threads=threads)


# --- 1: the type cannot become what it replaces ----------------------------


def test_a_decision_carries_no_phase_tone_or_sentence_shape():
    """Finding F's failure mode is a new representation acquiring the old vocabulary."""
    present = {field.name for field in fields(ConversationDecision)}

    for forbidden in FORBIDDEN_FIELDS:
        assert forbidden not in present, (
            f"ConversationDecision.{forbidden} would make it a fifth thing "
            "prescribing how to write, which is the problem it replaces"
        )


def test_a_proposed_operation_carries_no_price_and_no_success_flag():
    """§4: neither model can authorize a charge or declare a tool succeeded."""
    present = {field.name for field in fields(ProposedOperation)}

    for forbidden in ("price", "price_cents", "amount", "media_id", "media_ids",
                      "succeeded", "delivered", "authorized"):
        assert forbidden not in present


# --- 2: the checks that always run -----------------------------------------


def _decision(**overrides) -> ConversationDecision:
    values = {"active_needs": ("ordinary conversation",), "source": "test"}
    values.update(overrides)
    return ConversationDecision(**values)


def test_an_operation_this_turn_is_not_permitted_is_refused():
    decision = _decision(
        proposed_operation=ProposedOperation(
            kind=OperationKind.DELIVER_PAID_CONTENT,
            subject="the hotel set",
            because="he said yes",
        )
    )

    problems = deterministic_violations(decision, authorized_operations=frozenset())

    assert any("not authorized" in problem for problem in problems)


def test_the_same_operation_is_accepted_when_the_executor_permits_it():
    decision = _decision(
        proposed_operation=ProposedOperation(
            kind=OperationKind.DELIVER_PAID_CONTENT,
            subject="the hotel set",
            because="he said yes",
        )
    )

    problems = deterministic_violations(
        decision,
        authorized_operations=frozenset({OperationKind.DELIVER_PAID_CONTENT}),
        known_subjects=frozenset({"the hotel set"}),
    )

    assert problems == []


def test_an_invented_subject_is_refused():
    decision = _decision(
        proposed_operation=ProposedOperation(
            kind=OperationKind.OFFER_CONTENT,
            subject="the beach set nobody has",
            because="he might like it",
        )
    )

    problems = deterministic_violations(
        decision,
        authorized_operations=frozenset({OperationKind.OFFER_CONTENT}),
        known_subjects=frozenset({"the hotel set"}),
    )

    assert any("not something this turn knows to exist" in p for p in problems)


def test_a_decision_may_never_name_a_price():
    """It has no way to know one and every way to guess."""
    decision = _decision(
        proposed_operation=ProposedOperation(
            kind=OperationKind.OFFER_CONTENT,
            subject="the hotel set",
            because="he asked how much, it is $40",
        )
    )

    problems = deterministic_violations(
        decision,
        authorized_operations=frozenset({OperationKind.OFFER_CONTENT}),
        known_subjects=frozenset({"the hotel set"}),
    )

    assert any("price comes from inventory" in problem for problem in problems)


@pytest.mark.parametrize(
    "claim",
    ["already sent", "payment went through", "i sent it earlier", "delivered it"],
)
def test_a_claim_that_something_already_happened_is_refused(claim):
    """§4: a delivery claim must be tied to the operation result."""
    decision = _decision(
        hold=HoldReason.WAITING_ON_PAYMENT,
        hold_detail=f"he says his {claim}",
    )

    problems = deterministic_violations(decision)

    assert any("only a receipt can establish that" in p for p in problems)


def test_a_hold_must_say_why():
    problems = deterministic_violations(_decision(hold=HoldReason.NEEDS_HUMAN))

    assert any("without saying why" in problem for problem in problems)


def test_an_operation_must_say_what_it_is_about():
    decision = _decision(
        proposed_operation=ProposedOperation(kind=OperationKind.OFFER_CONTENT)
    )

    problems = deterministic_violations(
        decision, authorized_operations=frozenset({OperationKind.OFFER_CONTENT})
    )

    assert any("without saying what it is about" in p for p in problems)


def test_ordinary_conversation_proposes_nothing_and_violates_nothing():
    assert deterministic_violations(_decision()) == []


# --- 3: the projection describes today, honestly ---------------------------


def test_the_projection_makes_no_model_call_and_invents_nothing():
    decision = current_stack_decision(
        _packet(),
        {
            "situation": {"purchase_signal": "ready_to_buy"},
            "commercial_decision": {
                "action": "SEND_NEXT_PPV_STEP",
                "accepted_offer_set_id": "the hotel set",
            },
        },
    )

    assert decision.source == "current_stack"
    assert decision.proposed_operation.kind == OperationKind.DELIVER_PAID_CONTENT
    assert decision.proposed_operation.subject == "the hotel set"


def test_an_access_complaint_outranks_the_commercial_layers_own_offer():
    """§4: a request to fix access outranks a new commercial suggestion."""
    decision = current_stack_decision(
        _packet(),
        {
            "situation": {"purchase_signal": "ready_to_buy", "resend_requested": "true"},
            "commercial_decision": {
                "action": "OFFER_NEXT_UNLOCK",
                "next_offer": {"label": "the new set"},
            },
        },
    )

    assert decision.active_needs[0] == "he cannot access content he paid for"
    assert decision.hold == HoldReason.NEEDS_HUMAN
    assert decision.proposed_operation.kind == OperationKind.NONE
    # The offer the commercial layer separately wanted is on the record rather
    # than silently dropped: that it exists at all, in a turn whose own reason
    # is "hand this to a human", is finding F.
    assert "the commercial layer separately wanted" in decision.hold_detail


def test_a_holding_turn_never_also_proposes_a_sale():
    """One decision states one thing."""
    for situation, commercial in (
        ({"crisis_signal": "self_harm"}, {"action": "OFFER_NEXT_UNLOCK"}),
        ({"resend_requested": "true"}, {"action": "OFFER_NEXT_UNLOCK"}),
    ):
        decision = current_stack_decision(
            _packet(), {"situation": situation, "commercial_decision": commercial}
        )
        assert decision.is_hold
        assert decision.proposed_operation.kind == OperationKind.NONE


def test_a_fabricated_analysis_produces_no_confident_reading():
    """REL-001, in this vocabulary."""
    decision = current_stack_decision(
        _packet(),
        {
            "situation": {"purchase_signal": "none"},
            "analysis_degraded": True,
            "commercial_decision": {"action": "CONTINUE_NORMAL_CHAT"},
        },
    )

    assert decision.hold == HoldReason.INSUFFICIENT_EVIDENCE
    assert decision.confidence == 0.0


def test_the_projection_reports_that_it_decides_nothing_about_obligations():
    """The finding, not a gap in the projection.

    No controller in the current stack takes an open thread as input. Filling
    must_address in from the packet would make the projection look like it
    decides something it does not, and would make the replay's missed-obligation
    count vacuous for this candidate.
    """
    decision = current_stack_decision(
        _packet(threads=("he asked: whether you ever visit Chicago",)),
        {"situation": {}, "commercial_decision": {"action": "CONTINUE_NORMAL_CHAT"}},
    )

    assert decision.must_address == ()
    assert decision.active_needs == (
        "he is waiting on something from an earlier message",
    )


def test_a_pause_is_a_hold_with_the_customers_own_reason():
    decision = current_stack_decision(
        _packet(),
        {
            "situation": {"purchase_signal": "no_money"},
            "commercial_decision": {"action": "PAUSE_UNTIL_PAYDAY"},
        },
    )

    assert decision.hold == HoldReason.WAITING_ON_CUSTOMER
    assert "money later" in decision.hold_detail


# --- the semantic candidate -------------------------------------------------


def _model_reply(text: str):
    async def complete(_target, **_kwargs):
        return SimpleNamespace(text=text)

    return complete


def test_the_semantic_owner_reads_the_same_packet_the_projection_does():
    """§5: replay gives candidates the same evidence."""
    packet = _packet(
        threads=("he asked: whether you ever visit Chicago",),
        history=[{"role": "fan", "content": "hey"}],
    )

    _system, user = build_semantic_prompt(packet, {"latest_message": "hey"})

    assert "whether you ever visit Chicago" in user
    assert "Fan: hey" in user


def test_the_semantic_owner_is_told_not_to_decide_how_to_write():
    system, _user = build_semantic_prompt(_packet(), {})

    assert "You do not write the message" in system
    assert "Do not decide tone, length" in system
    assert "Never state a price" in system


def test_a_well_formed_answer_becomes_a_typed_decision():
    decision = parse_semantic_decision(
        json.dumps(
            {
                "active_needs": ["he wants the set he asked about"],
                "unresolved_references": [],
                "must_address": ["he asked whether you visit chicago"],
                "operation": "offer_content",
                "operation_subject": "the hotel set",
                "operation_because": "he asked about it directly",
                "hold": "none",
                "hold_detail": "",
                "confidence": 0.8,
            }
        )
    )

    assert decision is not None
    assert decision.proposed_operation.kind == OperationKind.OFFER_CONTENT
    assert decision.must_address == ("he asked whether you visit chicago",)
    assert decision.confidence == 0.8


def test_an_answer_wrapped_in_a_code_fence_is_still_read():
    """A complete payload, because this test is about the fence.

    It used to pass `{"active_needs": ["chat"]}` — incomplete under the strict
    parser, which now requires the three fields that ARE the decision. Kept
    testing what it was named for rather than doubling as a "partial payloads
    are fine" test, which is the behaviour the review found wrong.
    """
    decision = parse_semantic_decision(
        '```json\n'
        '{"active_needs": ["chat"], "operation": "none", "hold": "none", '
        '"confidence": 0.7}\n'
        '```'
    )

    assert decision is not None
    assert decision.active_needs == ("chat",)


@pytest.mark.parametrize("bad", ["", "no.", "{oops", "[1, 2, 3]"])
def test_an_unparseable_answer_is_refused_rather_than_repaired(bad):
    """A decision assembled from a half-read response is REL-001 again."""
    assert parse_semantic_decision(bad) is None


def test_an_unknown_operation_is_refused_rather_than_degraded():
    """Inverted deliberately. The old behaviour was the defect.

    This test asserted that an operation the system does not know silently
    became NONE. Combined with `confidence` defaulting to 1.0 when absent,
    that made

        {"hold": "typo", "operation": "typo"}

    a decision proposing nothing, holding nothing, and claiming FULL
    confidence in that reading — a fabricated confident decision assembled
    from a response the parser could not understand.

    "Degrades to proposing nothing" also sounds safe and is not: the caller
    treats a returned decision as an answer, so the candidate is scored as
    having read the conversation when it did not answer at all. Refusing makes
    it a visible insufficient-evidence outcome instead.
    """
    assert parse_semantic_decision('{"operation": "charge_his_card"}') is None
    assert parse_semantic_decision(
        '{"operation": "charge_his_card", "hold": "none", "confidence": 0.9}'
    ) is None


def test_a_candidate_that_cannot_answer_says_so_rather_than_guessing():
    owner = SemanticDecisionOwner(_model_reply("not json at all"))

    decision = compare_owners_sync(
        [ReplayTurn(name="t", state={})], [owner]
    ).comparisons[0].decisions["semantic_owner"]

    assert decision.hold == HoldReason.INSUFFICIENT_EVIDENCE
    assert decision.confidence == 0.0


def test_a_provider_failure_is_a_hold_not_a_crash():
    async def boom(*_a, **_k):
        raise RuntimeError("provider down")

    report = compare_owners_sync(
        [ReplayTurn(name="t", state={})], [SemanticDecisionOwner(boom)]
    )
    decision = report.comparisons[0].decisions["semantic_owner"]

    assert decision.hold == HoldReason.INSUFFICIENT_EVIDENCE
    assert "could not be reached" in decision.hold_detail


# --- 4: the comparison ------------------------------------------------------


def test_every_candidate_sees_the_same_evidence():
    seen: list[str] = []

    class Recorder:
        def __init__(self, name):
            self.name = name

        async def decide(self, packet, state):
            seen.append(packet.render_transcript())
            return ConversationDecision(source=self.name)

    compare_owners_sync(
        [
            ReplayTurn(
                name="t",
                history=({"role": "fan", "content": "hey"},),
                state={},
            )
        ],
        [Recorder("a"), Recorder("b")],
    )

    assert len(seen) == 2
    assert seen[0] == seen[1], "a comparison must compare deciding, not evidence"


def test_a_critical_failure_is_counted_on_its_own():
    """§5: a prose score must never cancel an unauthorized transaction."""

    class Overreaching:
        name = "overreaching"

        async def decide(self, packet, state):
            return ConversationDecision(
                active_needs=("sell him something",),
                proposed_operation=ProposedOperation(
                    kind=OperationKind.DELIVER_PAID_CONTENT,
                    subject="anything",
                    because="why not",
                ),
                source=self.name,
            )

    report = compare_owners_sync(
        [ReplayTurn(name="t", authorized_operations=frozenset())], [Overreaching()]
    )

    assert report.critical_failures("overreaching") == 1
    assert "CRITICAL" in report.render()


def test_a_missed_obligation_is_measured_against_what_was_known_beforehand():
    class Ignores:
        name = "ignores"

        async def decide(self, packet, state):
            return ConversationDecision(source=self.name)

    report = compare_owners_sync(
        [
            ReplayTurn(
                name="t",
                open_threads=("he asked: whether you ever visit Chicago",),
            )
        ],
        [Ignores()],
    )

    assert report.missed_obligations("ignores") == 1


def test_two_candidates_that_would_do_different_things_are_reported():
    class Sells:
        name = "sells"

        async def decide(self, packet, state):
            return ConversationDecision(
                proposed_operation=ProposedOperation(
                    kind=OperationKind.OFFER_CONTENT, subject="a set", because="mood"
                ),
                source=self.name,
            )

    class Waits:
        name = "waits"

        async def decide(self, packet, state):
            return ConversationDecision(
                hold=HoldReason.RESPECT_SILENCE,
                hold_detail="he said goodnight",
                source=self.name,
            )

    report = compare_owners_sync(
        [
            ReplayTurn(
                name="t",
                authorized_operations=frozenset({OperationKind.OFFER_CONTENT}),
                known_subjects=frozenset({"a set"}),
            )
        ],
        [Sells(), Waits()],
    )

    assert report.disagreed == 1
    differences = report.comparisons[0].disagreements
    assert any("operation:" in difference for difference in differences)
    assert any("hold:" in difference for difference in differences)


def test_the_replay_changes_nothing_and_repeats_exactly():
    turns = load_turns(json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"])

    first = compare_owners_sync(turns, [CurrentStackOwner()]).summary()
    second = compare_owners_sync(turns, [CurrentStackOwner()]).summary()

    assert first == second


# --- the shipped scenarios --------------------------------------------------


def test_the_scenarios_load_and_cover_the_reviews_trajectory_table():
    """Checked against each scenario's declared `_covers`, not its title.

    A title can drift away from what a scenario tests without anybody noticing.
    `_covers` names the row of review §5 the scenario exists for, so it is the
    thing worth asserting on — and the assertion fails if a row loses its last
    scenario.
    """
    raw = json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"]
    turns = load_turns(raw)

    assert len(turns) >= 10
    covered = " ".join(str(scenario.get("_covers", "")) for scenario in raw).lower()

    for row in (
        "ordinary conversation",
        "deferred across 30+ turns",
        "multiple open threads",
        "content-access support",
        "mixed-intent",
        "preference correction",
        "goodbye",
        "crisis",
        "rel-001",
        "ambiguous pronouns",
    ):
        assert row in covered, f"no scenario declares that it covers {row!r}"

    for scenario in raw:
        assert scenario.get("_covers"), (
            f"{scenario.get('name')!r} does not say which review row it is for"
        )


def test_the_current_stack_proposes_nothing_it_is_not_permitted():
    """The always-on half of §4's verification, over the shipped scenarios."""
    turns = load_turns(json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"])

    report = compare_owners_sync(turns, [CurrentStackOwner()])

    assert report.critical_failures("current_stack") == 0, report.render()


def test_the_shipped_scenarios_show_the_obligation_gap():
    """The measurement Sprint 3 exists to produce.

    Not an assertion that the current stack is bad — it is an assertion that the
    gap is real, measured, and will move if a candidate closes it. If this ever
    reads zero because the projection started filling must_address in from the
    packet, the comparison has stopped measuring anything.
    """
    turns = load_turns(json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"])

    report = compare_owners_sync(turns, [CurrentStackOwner()])

    assert report.missed_obligations("current_stack") > 0


def test_no_scenario_contains_explicit_material():
    """§5 asks for the longitudinal evaluation to be non-explicit."""
    raw = SCENARIOS.read_text(encoding="utf-8").lower()

    for word in ("nude", "nudes", "pussy", "cock", "cum", "fuck"):
        assert word not in raw


# ===========================================================================
# Strict typed parsing
# ===========================================================================
#
#     missing required fields, invalid enums, wrong types, non-finite
#     confidence and malformed output must result in a visible
#     failed/insufficient-evidence outcome, never a fabricated confident
#     decision.
#                     — docs/continuation_brief_2026-09-17.md, Phase D
#
# The version this replaces documented itself as strict — "Returns None for
# anything unparseable. There is no partial credit and no repair pass" — and
# did the opposite for the cases that matter most. A docstring is not a
# behaviour, and these are.


def _complete(**overrides) -> str:
    payload = {"operation": "none", "hold": "none", "confidence": 0.8}
    payload.update(overrides)
    return json.dumps(payload)


def test_a_complete_answer_is_read():
    """The control. Without it every refusal below could be a broken parser."""
    from services.decision_owners import parse_semantic_decision_result

    parsed = parse_semantic_decision_result(_complete(active_needs=["chat"]))

    assert parsed.ok
    assert parsed.reason == ""
    assert parsed.decision.confidence == 0.8


@pytest.mark.parametrize("missing", ["operation", "hold", "confidence"])
def test_a_response_that_leaves_out_the_decision_is_refused(missing):
    """These three ARE the decision: what it proposes, whether it waits, and
    how far to trust either."""
    from services.decision_owners import parse_semantic_decision_result

    payload = json.loads(_complete())
    payload.pop(missing)

    parsed = parse_semantic_decision_result(json.dumps(payload))

    assert not parsed.ok
    assert missing in parsed.reason


def test_an_empty_object_is_refused():
    assert parse_semantic_decision("{}") is None


@pytest.mark.parametrize(
    "payload",
    [
        {"operation": "typo", "hold": "typo", "confidence": 1.0},
        {"operation": "none", "hold": "typo", "confidence": 1.0},
        {"operation": "typo", "hold": "none", "confidence": 1.0},
    ],
)
def test_an_invented_enum_is_refused(payload):
    """The review's exact reproduction, and its two halves separately."""
    assert parse_semantic_decision(json.dumps(payload)) is None


@pytest.mark.parametrize(
    "confidence", [float("nan"), float("inf"), float("-inf")]
)
def test_a_non_finite_confidence_is_refused(confidence):
    from services.decision_owners import parse_semantic_decision_result

    parsed = parse_semantic_decision_result(
        '{"operation":"none","hold":"none","confidence":%s}'
        % ("NaN" if confidence != confidence else ("Infinity" if confidence > 0 else "-Infinity"))
    )

    assert not parsed.ok
    assert "finite" in parsed.reason


@pytest.mark.parametrize("confidence", [1.5, -0.2, 100])
def test_a_confidence_outside_the_range_is_refused_rather_than_clamped(confidence):
    """The one choice here worth arguing about.

    Clamping 1.5 to 1.0 RAISES a malformed claim to maximum confidence — the
    failure this function exists to prevent, applied by the repair.
    """
    assert parse_semantic_decision(_complete(confidence=confidence)) is None


def test_a_boolean_confidence_is_refused():
    """bool is a subclass of int, so True would read as a confidence of 1.0."""
    assert parse_semantic_decision('{"operation":"none","hold":"none","confidence":true}') is None


@pytest.mark.parametrize(
    "payload",
    [
        {"operation": 3, "hold": "none", "confidence": 0.5},
        {"operation": "none", "hold": ["none"], "confidence": 0.5},
        {"operation": "none", "hold": "none", "confidence": "high"},
        {"operation": "none", "hold": "none", "confidence": 0.5, "active_needs": "chat"},
        {"operation": "none", "hold": "none", "confidence": 0.5, "must_address": {"a": 1}},
        {"operation": "none", "hold": "none", "confidence": 0.5, "hold_detail": 7},
    ],
)
def test_a_field_of_the_wrong_type_is_refused(payload):
    assert parse_semantic_decision(json.dumps(payload)) is None


def test_every_refusal_says_which_one_it_was():
    """A comparison that cannot say WHICH candidate failed HOW is not much of
    a comparison, and a bare None made every failure identical."""
    from services.decision_owners import parse_semantic_decision_result

    reasons = {
        parse_semantic_decision_result("no json here").reason,
        parse_semantic_decision_result('{"a": }').reason,
        parse_semantic_decision_result("{}").reason,
        parse_semantic_decision_result(_complete(operation="typo")).reason,
        parse_semantic_decision_result(_complete(hold="typo")).reason,
        parse_semantic_decision_result(_complete(confidence=5)).reason,
        parse_semantic_decision_result(_complete(confidence="high")).reason,
    }

    assert len(reasons) == 7, reasons
    assert all(reason for reason in reasons)

    # "no json here" and "{oops" DO share a reason, and correctly: neither
    # contains a JSON object, and inventing a distinction between them would
    # be precision about nothing.
    assert (
        parse_semantic_decision_result("{oops").reason
        == parse_semantic_decision_result("no json here").reason
    )


def test_the_owner_reports_the_reason_it_refused():
    """The failure has to be visible, not just counted."""
    import asyncio

    from services.decision_owners import SemanticDecisionOwner

    async def answers_badly(*_args, **_kwargs):
        return SimpleNamespace(text='{"hold": "typo", "operation": "typo"}')

    owner = SemanticDecisionOwner(answers_badly)
    decision = asyncio.run(owner.decide(build_context_packet(history=[]), {}))

    # The review's exact payload. What matters is that it is refused with a
    # stated reason and zero confidence — not WHICH reason: it is missing
    # `confidence` as well as carrying invented enums, and the missing-field
    # check runs first because a response that does not say how far to trust
    # it has not answered whatever else it got right.
    assert decision.hold is HoldReason.INSUFFICIENT_EVIDENCE
    assert decision.confidence == 0.0
    assert "did not answer usably" in decision.hold_detail
    assert "confidence" in decision.hold_detail

    async def answers_with_a_bad_enum(*_args, **_kwargs):
        return SimpleNamespace(
            text='{"hold": "none", "operation": "typo", "confidence": 0.9}'
        )

    named = asyncio.run(
        SemanticDecisionOwner(answers_with_a_bad_enum).decide(
            build_context_packet(history=[]), {}
        )
    )

    assert "typo" in named.hold_detail
    assert named.confidence == 0.0


def test_supporting_evidence_is_carried_when_the_answer_gives_it():
    """The brief asks a decision to carry the evidence it rests on."""
    decision = parse_semantic_decision(
        _complete(supporting_messages=["fp-1", "fp-2"], must_address=["chicago"])
    )

    assert decision.supporting_messages == ("fp-1", "fp-2")
    assert decision.must_address == ("chicago",)
