"""Grounding, commercial balance, and the difference between a claim and a receipt.

Three production failures, one shape: the conversation layer had no
deterministic statement of what its evidence does and does not support, so the
models filled the gap.

  * Approved vault inventory was read as proof of a feed post, and Cleopatra
    told a fan to "check my feed" for something that exists only privately.
  * After #68 removed the explicitness funnel, the system started declining
    explicit, fan-created buying opportunities — "just say the price" answered
    with "not about price".
  * "I bought it" was treated as a receipt.

Everything below is computed from what the fan actually wrote, with the message
ids it came from. Nothing here infers wealth, spending power, or a stage.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.commercial import CreatorPolicy, FanCommercialState, FanStatus, Offer
from models.live_orchestration import (
    ApprovedExecution,
    EvidenceFact,
    EvidenceSnapshot,
    TurnTrigger,
    ValidationResult,
)
from models.schemas import Fan, Persona
from services import conversation_signals, live_orchestration
from services.context_packet import ContextPacket
from ai.writer_style import MODE_AUTO


def message(role: str, text: str, message_id: str = "") -> dict:
    return {"message_id": message_id, "speaker": role, "text": text, "at": None}


def snapshot(
    *,
    latest: str = "hey",
    burst: tuple[dict, ...] = (),
    recent: tuple[dict, ...] = (),
    creator_facts: tuple[EvidenceFact, ...] = (),
    purchases: tuple[dict, ...] = (),
    next_offer: Offer | None = None,
    pending_offer: Offer | None = None,
) -> EvidenceSnapshot:
    burst = burst or (message("fan", latest, "m-latest"),)
    recent = recent or burst
    return EvidenceSnapshot(
        creator_id="creator-1",
        fan_id="fan-1",
        trigger=TurnTrigger(
            kind="fan_message", identity="m-latest", latest_message=latest
        ),
        state_revision="rev-1",
        creator_facts=creator_facts,
        latest_fan_burst=burst,
        recent_messages=recent,
        publication_evidence=conversation_signals.publication_evidence(creator_facts),
        fan_publication_references=conversation_signals.fan_publication_references(
            burst, recent
        ),
        commercial_opportunity={
            **conversation_signals.purchase_intent(burst),
            "unsent_approved_inventory_exists": bool(next_offer),
            "offer_already_presented": bool(pending_offer),
        },
        purchase_claim=conversation_signals.purchase_claim(
            burst, confirmed_purchases=purchases
        ),
        voice_rhythm=conversation_signals.recent_creator_emoji(recent),
    )


def loaded(**kwargs):
    snap = snapshot(**kwargs)
    return live_orchestration.LoadedEvidence(
        snapshot=snap,
        packet=ContextPacket(),
        history=[],
        fan=Fan(
            id="fan-1",
            display_name="Fan",
            platform_fan_id="test_fan_1",
            auto_mode=True,
        ),
        persona=Persona(),
        commercial_state=FanCommercialState(status=FanStatus.IDLE),
        policy=CreatorPolicy(),
        next_offer=kwargs.get("next_offer"),
        active_session=None,
        pending_payment=None,
        sent_ppv=[],
        within_daily_caps=True,
        stack=SimpleNamespace(profile_id="cleo_v3"),
    )


def reasons(text: str, evidence, execution=None) -> list[str]:
    return live_orchestration.writer_contract_reasons(
        [text],
        evidence,
        execution or ApprovedExecution(validation=ValidationResult(True, "none")),
        mode=MODE_AUTO,
    )


# --- 1. The feed that does not exist ---------------------------------------


def test_no_feed_integration_is_stated_rather_than_assumed():
    evidence = conversation_signals.publication_evidence(())

    assert evidence["authoritative_posts_available"] is False
    assert evidence["integration"] == "none"
    assert "not evidence" in evidence["rule"].lower()


def test_a_fan_mentioned_post_is_recorded_as_his_context():
    burst = (message("fan", "that bikini post had me weak", "m-1"),)
    references = conversation_signals.fan_publication_references(burst, burst)

    assert references == (
        {"source_ref": "m-1", "fan_words": "that bikini post had me weak"},
    )


def test_the_creator_may_talk_about_the_post_he_raised():
    evidence = loaded(
        latest="that bikini post had me weak",
        burst=(message("fan", "that bikini post had me weak", "m-1"),),
    )

    assert reasons("that bikini post was a good day honestly", evidence) == []
    assert reasons("glad the bikini one got to you", evidence) == []


def test_inventing_a_post_is_refused():
    evidence = loaded(
        latest="that bikini post had me weak",
        burst=(message("fan", "that bikini post had me weak", "m-1"),),
    )

    assert "unsupported_publication_claim" in reasons(
        "wait until you see the bedroom set I just posted", evidence
    )


def test_sending_him_to_a_feed_is_refused():
    evidence = loaded()

    for copy in (
        "check my feed babe",
        "go look at my page",
        "refresh my feed",
        "my latest post is up",
        "it's on my feed now",
    ):
        assert "unsupported_publication_claim" in reasons(copy, evidence), copy


def test_ordinary_conversation_is_not_touched_by_the_publication_rule():
    evidence = loaded()

    for copy in (
        "i posted up on the couch all evening",
        "you always pick the dangerous details",
        "tell me what you were thinking about",
        "i'll post bail if you get arrested for that 😂",
    ):
        assert "unsupported_publication_claim" not in reasons(copy, evidence), copy


def test_authoritative_post_evidence_makes_the_claim_legal():
    evidence = loaded(
        creator_facts=(
            EvidenceFact(
                value="posted a bedroom set this morning",
                source_ref="feed_post:12345",
                certainty="platform_confirmed",
            ),
        )
    )

    assert evidence.snapshot.publication_evidence["authoritative_posts_available"]
    assert reasons("check my feed, I just posted it", evidence) == []


# --- 2. Money, in both directions ------------------------------------------


def test_intimacy_alone_creates_no_commercial_opportunity():
    burst = (message("fan", "i can't stop thinking about your mouth", "m-1"),)

    signal = conversation_signals.purchase_intent(burst)

    assert signal["fan_stated_buying_signal"] is False
    assert signal["signal_kinds"] == []


@pytest.mark.parametrize(
    "text,kind",
    [
        ("how much?", "asked_price"),
        ("just say the price, I can do it", "asked_price"),
        ("what's the price babe", "asked_price"),
        ("I'll pay extra baby", "offered_to_pay"),
        ("name your price", "offered_to_pay"),
        ("take my money", "offered_to_pay"),
        ("what can I buy", "asked_to_buy"),
        ("send me more", "asked_to_buy"),
        ("can i unlock it", "asked_to_buy"),
    ],
)
def test_an_explicit_fan_created_buying_signal_is_evidence(text, kind):
    signal = conversation_signals.purchase_intent((message("fan", text, "m-1"),))

    assert signal["fan_stated_buying_signal"] is True
    assert kind in signal["signal_kinds"]
    assert signal["source_ids"] == ["m-1"]


def test_a_buying_signal_survives_the_conversation_being_intimate():
    burst = (
        message("fan", "god you're driving me insane", "m-1"),
        message("fan", "just tell me the price, I'll pay extra", "m-2"),
    )

    signal = conversation_signals.purchase_intent(burst)

    assert signal["fan_stated_buying_signal"] is True
    assert signal["source_ids"] == ["m-2"]


def test_the_signal_is_about_what_he_said_not_what_he_can_afford():
    signal = conversation_signals.purchase_intent(
        (message("fan", "I just got a huge bonus at work", "m-1"),)
    )

    assert signal["fan_stated_buying_signal"] is False
    assert "never what he can afford" in signal["rule"]


def test_only_the_current_burst_counts_as_a_live_opportunity():
    """A price question three days ago is not a reason to sell today."""
    older = (message("fan", "how much for the set?", "m-old"),)
    now = (message("fan", "morning you", "m-new"),)

    assert conversation_signals.purchase_intent(now)["fan_stated_buying_signal"] is False
    assert conversation_signals.purchase_intent(older)["fan_stated_buying_signal"]


def test_the_evidence_snapshot_carries_the_opportunity_and_the_inventory():
    evidence = loaded(
        latest="just say the price, I can do it",
        burst=(message("fan", "just say the price, I can do it", "m-1"),),
        next_offer=Offer(
            offer_id="offer-1", set_id="set-1", label="approved set", price_cents=2500
        ),
    )

    opportunity = evidence.snapshot.commercial_opportunity
    assert opportunity["fan_stated_buying_signal"] is True
    assert opportunity["unsent_approved_inventory_exists"] is True


def test_the_exact_price_may_be_expressed_when_he_asked_for_it():
    offer = Offer(
        offer_id="offer-1", set_id="set-1", label="approved set", price_cents=2500
    )
    evidence = loaded(
        latest="just say the price, I can do it",
        burst=(message("fan", "just say the price, I can do it", "m-1"),),
        next_offer=offer,
    )
    execution = ApprovedExecution(
        operation="present_offer",
        offer=live_orchestration._offer_view(offer),
        validation=ValidationResult(True, "present_offer"),
    )

    assert reasons("it's $25 and worth every second of it", evidence, execution) == []


def test_a_price_the_application_did_not_approve_is_still_refused():
    offer = Offer(
        offer_id="offer-1", set_id="set-1", label="approved set", price_cents=2500
    )
    evidence = loaded(
        latest="just say the price, I can do it",
        burst=(message("fan", "just say the price, I can do it", "m-1"),),
        next_offer=offer,
    )
    execution = ApprovedExecution(
        operation="present_offer",
        offer=live_orchestration._offer_view(offer),
        validation=ValidationResult(True, "present_offer"),
    )

    assert "unapproved_price_claim" in reasons("call it $18 for you", evidence, execution)


# --- 3. A claim is not a receipt -------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I bought it",
        "i just paid",
        "I've already purchased it",
        "payment sent",
        "i sent the money",
    ],
)
def test_the_fan_saying_he_paid_is_recorded_as_a_claim(text):
    claim = conversation_signals.purchase_claim((message("fan", text, "m-1"),))

    assert claim["fan_claimed_purchase"] is True
    assert claim["authoritative_confirmation"] is False
    assert claim["source_ids"] == ["m-1"]


def test_an_authoritative_purchase_confirms_it():
    claim = conversation_signals.purchase_claim(
        (message("fan", "I bought it", "m-1"),),
        confirmed_purchases=({"purchased": True, "reference": "ref-1"},),
    )

    assert claim["authoritative_confirmation"] is True


@pytest.mark.parametrize(
    "copy",
    [
        "told you it'd be worth it 😏",
        "thanks for buying babe",
        "now that you've unlocked it, tell me what you think",
        "enjoy it baby",
        "so glad you bought it",
    ],
)
def test_acting_on_an_unconfirmed_claim_is_refused(copy):
    evidence = loaded(
        latest="I bought it",
        burst=(message("fan", "I bought it", "m-1"),),
    )

    assert "unverified_purchase_acknowledgement" in reasons(copy, evidence)


def test_a_confirmed_purchase_may_be_acknowledged():
    evidence = loaded(
        latest="I bought it",
        burst=(message("fan", "I bought it", "m-1"),),
        purchases=({"purchased": True, "reference": "ref-1"},),
    )

    assert reasons("told you it'd be worth it 😏", evidence) == []


def test_warmth_is_untouched_when_no_purchase_was_claimed():
    evidence = loaded(latest="hey you", burst=(message("fan", "hey you", "m-1"),))

    assert reasons("enjoy it baby, I mean the rest of your evening", evidence) == []


def test_the_writer_is_told_which_claims_are_unconfirmed():
    evidence = loaded(
        latest="I bought it", burst=(message("fan", "I bought it", "m-1"),)
    )
    prompt = live_orchestration.build_conversational_writer_prompt(
        evidence,
        live_orchestration.ConversationDecision(),
        ApprovedExecution(validation=ValidationResult(True, "none")),
        live_orchestration.ConversationalWorkingState(),
        mode=MODE_AUTO,
    )
    payload = prompt[1]["content"]

    assert "purchase_claim" in payload
    assert "publication_evidence" in payload
    assert "commercial_opportunity" in payload


# --- 4. Emoji rhythm, softly ------------------------------------------------


def test_repeated_emoji_are_noticed():
    recent = (
        message("creator", "mm 😏", "c-1"),
        message("creator", "keep going 😏", "c-2"),
        message("creator", "you're trouble 😏", "c-3"),
    )

    rhythm = conversation_signals.recent_creator_emoji(recent)

    assert "😏" in rhythm["repeated_in_recent_turns"]
    assert "Soft guidance only" in rhythm["note"]


def test_emoji_repetition_never_blocks_a_send():
    evidence = loaded(
        recent=(
            message("creator", "mm 😏", "c-1"),
            message("creator", "keep going 😏", "c-2"),
            message("fan", "hey", "m-1"),
        ),
        burst=(message("fan", "hey", "m-1"),),
    )

    assert reasons("you're trouble 😏", evidence) == []


# --- 5. What the balance actually permits ----------------------------------


def commercial_decision(kind, *, offer_id="", set_id="", handle=""):
    from models.conversation_decision import (
        ConversationDecision,
        OperationKind,
        ProposedOperation,
    )

    return ConversationDecision(
        proposed_operation=ProposedOperation(
            kind=OperationKind(kind),
            subject="the set he asked about",
            candidate_handle=handle,
            offer_id=offer_id,
            set_id=set_id,
        )
    )


def test_a_direct_buying_signal_leaves_the_operation_available():
    """Not forced, available. GLM still judges the moment; the state permits it."""
    record = Offer(
        offer_id="offer-1", set_id="set-1", label="approved set", price_cents=2500
    )
    evidence = loaded(
        latest="I'll pay extra baby",
        burst=(message("fan", "I'll pay extra baby", "m-1"),),
        next_offer=record,
    )
    evidence.candidate_handles = {"offer_candidate_1": record}

    assert "present_offer" in live_orchestration.legal_operations(evidence)
    validation = live_orchestration.validate_decision(
        commercial_decision(
            "present_offer", offer_id="offer-1", set_id="set-1", handle="offer_candidate_1"
        ),
        evidence,
    )
    assert validation.approved is True


def test_an_intimate_turn_with_no_buying_signal_may_simply_continue():
    """#68's rule, unchanged: nothing forces a sale."""
    evidence = loaded(
        latest="i can't stop thinking about your mouth",
        burst=(message("fan", "i can't stop thinking about your mouth", "m-1"),),
        next_offer=Offer(
            offer_id="offer-1", set_id="set-1", label="approved set", price_cents=2500
        ),
    )

    validation = live_orchestration.validate_decision(
        commercial_decision("none"), evidence
    )

    assert validation.approved is True
    assert evidence.snapshot.commercial_opportunity["fan_stated_buying_signal"] is False
    assert reasons("tell me exactly what you'd do first", evidence) == []


def test_no_inventory_means_no_commercial_operation_however_he_asks():
    evidence = loaded(
        latest="just tell me the price",
        burst=(message("fan", "just tell me the price", "m-1"),),
        next_offer=None,
    )

    assert "present_offer" not in live_orchestration.legal_operations(evidence)
    validation = live_orchestration.validate_decision(
        commercial_decision("present_offer", offer_id="offer-1", set_id="set-1"),
        evidence,
    )
    assert validation.approved is False


def test_a_rejection_leaves_the_conversation_running():
    evidence = loaded(
        latest="nah I'm good, too rich for me tonight",
        burst=(message("fan", "nah I'm good, too rich for me tonight", "m-1"),),
    )

    assert reasons("all good, I'd rather just talk anyway", evidence) == []


def test_the_decision_model_is_told_the_opportunity_and_the_rule():
    """GLM sees what he said, and the sentence that keeps it from being a funnel."""
    assert "commercial_opportunity" in live_orchestration.CONVERSATIONAL_V1_SYSTEM
    assert "never state a price" in live_orchestration.CONVERSATIONAL_V1_SYSTEM
    assert (
        "NEVER create a commercial opportunity by themselves"
        in live_orchestration.CONVERSATIONAL_V1_SYSTEM
    )
