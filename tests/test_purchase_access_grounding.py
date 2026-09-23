from __future__ import annotations

import asyncio
from types import SimpleNamespace

from models.commercial import CreatorPolicy, FanCommercialState
from models.conversation_decision import ConversationDecision, OperationKind, ProposedOperation
from models.live_orchestration import EvidenceSnapshot, TurnTrigger
from models.schemas import Fan, Persona
from services import live_orchestration
from services.conversation_signals import (
    content_access_issue,
    unsupported_platform_state_claim,
)


def run(coro):
    return asyncio.run(coro)


def test_confirmed_purchase_access_issue_is_detected():
    issue = content_access_issue(
        [{"speaker": "fan", "message_id": "m1", "text": "wait these are blurred baby, i cant open them"}],
        confirmed_purchases=[{"reference": "purchase-1", "purchased": True}],
    )
    assert issue["fan_reported_access_problem"] is True
    assert issue["confirmed_purchase_exists"] is True
    assert issue["purchase_reference"] == "purchase-1"
    assert issue["signal_kinds"]


def test_platform_ui_claims_are_rejected_for_access_issue():
    issue = {
        "fan_reported_access_problem": True,
        "confirmed_purchase_exists": True,
    }
    assert unsupported_platform_state_claim("check your unlocks, it should be there", access_issue=issue)
    assert unsupported_platform_state_claim("you gotta unlock the full version", access_issue=issue)
    assert not unsupported_platform_state_claim("that sounds like an access issue", access_issue=issue)


def test_confirmed_purchase_access_issue_forces_repair_path():
    snapshot = EvidenceSnapshot(
        creator_id="creator-1",
        fan_id="fan-1",
        trigger=TurnTrigger(kind="fan_message", identity="m1", latest_message="these are blurred"),
        state_revision="r1",
        confirmed_purchases=({"reference": "purchase-1", "purchased": True},),
        content_access_issue={
            "fan_reported_access_problem": True,
            "confirmed_purchase_exists": True,
            "purchase_reference": "purchase-1",
        },
    )
    loaded = live_orchestration.LoadedEvidence(
        snapshot=snapshot,
        packet=SimpleNamespace(),
        history=[],
        fan=Fan(id="fan-1", display_name="Fan", platform_fan_id="test_1"),
        persona=Persona(),
        commercial_state=FanCommercialState(),
        policy=CreatorPolicy(),
        next_offer=None,
        active_session=None,
        pending_payment=None,
        sent_ppv=[],
        within_daily_caps=True,
        stack=SimpleNamespace(profile_id="cleo_v3"),
    )
    decision = ConversationDecision(
        proposed_operation=ProposedOperation(kind=OperationKind.NONE),
    )

    settled = run(
        live_orchestration._authorize_conversational_v1_operation(
            loaded,
            decision=decision,
            execute_operations=True,
        )
    )

    assert settled.decision.proposed_operation.kind is OperationKind.REPAIR_CONTENT_ACCESS
    assert settled.decision.proposed_operation.purchase_id == "purchase-1"
    assert settled.execution.operation == OperationKind.REPAIR_CONTENT_ACCESS.value
