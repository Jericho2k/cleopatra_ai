"""Conversational Core v2 — session-aware.

An ALTERNATIVE to ``conversational_v1``, not a layer on it. It keeps v1's
division of authority exactly:

    evidence + persisted state -> GLM decision + proposed state delta
        -> deterministic validation / persistence -> Kimi

and adds one thing v1 lacks: a durable, validated representation of the
longer interaction the conversation is carrying out (see
``models.conversational_session``). There is still ONE semantic owner (GLM),
ONE writer (Kimi), and deterministic code remains the only authority for
inventory, prices, purchases, deliveries, permissions and persistence.

What v2 changes about a turn:

* The owner answers "what should happen next in the interaction?"
  (``next_experience_move``) before deciding whether content is needed.
* The owner may plan a provisional trajectory that binds future beats to a
  small set of eligible approved candidates; planning never offers, reserves or
  sends anything. An actual operation is still an ``operation_proposal`` that
  the same validator and executor as v1 must approve.
* Content lifecycle facts are rebuilt from the ledger before the owner decides,
  so a model can neither mark content used nor treat used content as unused.
* After a confirmed purchase or delivery, the next turn cannot silently turn
  into another sale: a new paid operation in that turn needs the fan's own ask.
* A fan's stated spending limit, once recorded from his own words, caps every
  candidate and every paid operation until he changes it.
"""

from __future__ import annotations

import json
from dataclasses import replace as dataclasses_replace
from typing import Any

from ai.generation_trace import GenerationTrace
from ai.stack_profiles import STAGE_CONVERSATIONAL_OWNER
from ai.writer_style import MODE_ASSISTED
from models.conversation_decision import (
    ConversationDecision,
    HoldReason,
    OperationKind,
    ProposedOperation,
    ResponseDisposition,
    ResponseIntent,
)
from models.conversational_core import StateDeltaValidation
from models.conversational_session import (
    CONSUMED_LIFECYCLES,
    ConversationalSessionState,
    ExperienceMove,
    SessionDeltaValidation,
)
from models.model_runtime import FAILURE_TIMEOUT
from services import live_orchestration as lo
from services.conversation_core import CORE_CONVERSATIONAL_V2
from services.conversational_core import (
    evidence_catalog_view,
    state_fingerprint,
    validate_and_apply_delta,
)
from services.conversational_session import (
    CANDIDATE_HANDLE_PREFIX,
    load_session_state,
    owner_session_view,
    reconcile_with_authority,
    remaining_spending_cents,
    save_session_state,
    session_fingerprint,
    validate_session_delta,
    writer_session_view,
)
from services.conversational_session_contract import (
    SOURCE_V2,
    SessionDecisionResult,
    parse_session_decision,
)
from services.hermes_retrieval import retrieval_enabled, retrieve_examples
from services.platform_operating_model import (
    SCHEDULED_INTENT_EFFECT,
    operation_affordances,
    platform_context,
)
from services.reply_provenance import PIPELINE_ASSISTED, PIPELINE_AUTO

PAID_OPERATIONS = frozenset(
    {OperationKind.PRESENT_OFFER, OperationKind.SEND_LOCKED_PAID_MESSAGE}
)

SESSION_OWNER_EXTENSION = """SESSION-AWARE EXTENSION (Conversational Core v2).

The payload also carries interaction_session: the application-validated, durable state of the longer interaction this conversation may be carrying out. A media or paid event is never the session; it is at most one beat inside it.

Decide in this order, every turn:
1. What should happen next in the INTERACTION? Return next_experience_move (always): {"kind":"converse|react|linger|invite_participation|pull_back|callback|develop_premise|alter_premise|transition|discover|use_content|pause|close", "intent":"the semantic goal, never wording"}.
2. Only then: does this moment need content? Media is optional. Staying in the moment, reacting to what just happened, lingering, letting him participate, pulling back, a callback, continuing or altering the shared premise, or transitioning are all complete turns. Natural pacing owns the distance between content events: sometimes many turns should pass, sometimes one is enough because he redirects or asks for something relevant. Never add turns just to create distance, and never advance to another content event merely because another candidate exists or the plan lists it next. Advance when the interaction makes that content beat appropriate.

A longer interaction is recognised, never forced. It can open in two ways:
- He opens it: he co-creates a premise, asks for an extended experience, or keeps building a scene. Move status to planning or active as fits.
- You propose it: like a skilled chatter, you may recognise a moment where a longer interaction would genuinely suit THIS conversation (shared momentum, a premise worth developing, a natural pause that invites it) and set status "proposed", with next_experience_move a natural invitation or a light check of his interest or availability ("invite_participation" or "discover"). Proposed means offered, not agreed: nothing is planned as settled and no content is implied. His answer decides: planning/active if he takes it up; abandoned, or simply carrying on the ordinary conversation, if he declines, deflects or ignores it — never press a declined proposal.
A proposal is a judgement about this moment, never a routine step. Most conversations need none. Intimacy, explicitness, elapsed turns or a past purchase are not by themselves a session opportunity, and there is no required path from conversation to a session.
Pause, close or abandon a session when he cools, pauses, ends it or leaves the thread.

The trajectory is provisional, never a script or a promise. When he changes direction, states a concrete preference, asks for something specific, cools off, pauses or rejects a direction, you may replace the FUTURE beats with session_delta.trajectory and a reason. completed beats are history: you cannot change them. A content beat binds one candidate_handle from interaction_session.content_candidates and states media_role — why that content would serve THIS interaction. Planning a beat does not offer, reserve, send or sell anything; to act, use operation_proposal exactly as usual, with the same handle, and the application decides.

After a content event (last_content_event.happened_this_turn=true), the default is to stay inside the moment it created. Do not propose another paid operation in that turn unless he explicitly asks for more; a purchase never authorises the next item.

Constraints are evidence, never inference. Record one only with the fan's own message ids as source_refs. A spending_limit needs the amount he stated (amount_cents) and the message that states it. Never infer wealth, available money or a budget. If spending_limit_known is true, never ask again.

Do not interview. There is no required interest -> availability -> budget -> session funnel. Open an information need only when the next decision materially depends on it (for example, planning a longer paid interaction genuinely requires knowing his current spending limit); then discovering it becomes the conversational goal. A direct, concrete request that legal operations can handle now is handled now, without asking about a broader budget. Ordinary conversation never becomes a commercial interview.

experience_premise has world_scope "conversation" or "imagined_scene". imagined_scene is a SHARED imagined situation he takes part in (cite his messages); develop it as imagined. It never licenses claims about the creator's real current activity, clothing, location, surroundings, filming, posting or schedule.

content_ledger and last_content_event are application facts (presented/accepted/sent/purchased). You cannot write them. A planned candidate is not presented, accepted, sent or purchased.

Add these two top-level fields to your JSON object (omit session_delta when nothing changes; omit any unchanged key inside it):
"next_experience_move": {"kind":"...", "intent":"..."},
"session_delta": {
  "status": "inactive|proposed|planning|active|paused|completed|abandoned",
  "experience_premise": {"summary":"", "world_scope":"conversation|imagined_scene", "source_refs":["message id"]},
  "interaction_goal": "what this stretch is trying to achieve conversationally",
  "fan_participation": {"mode":"unknown|leading|co_creating|following|brief|redirecting|cooling|pausing", "gist":""},
  "add_constraints": [{"constraint_id":"", "kind":"spending_limit|availability|preference|boundary|other", "statement":"", "amount_cents":0, "source_type":"explicit_fan_statement", "source_refs":["message id"]}],
  "constraint_corrections": [{"replaces_constraint_id":"", "replacement":{"constraint_id":"", "kind":"", "statement":"", "amount_cents":0, "source_type":"explicit_fan_statement", "source_refs":[]}}],
  "open_information_need": {"topic":"spending_limit|content_preference|availability|direction|other", "why_material":"why the next decision depends on it"},
  "resolve_information_needs": ["need id"],
  "complete_current_beat": {"outcome":"what the beat achieved"},
  "trajectory": {"reason":"required when replacing existing future beats", "beats":[{"beat_id":"", "kind":"conversation|callback|participation|discovery|content|transition|close", "intent":"", "candidate_handle":"", "media_role":""}]},
  "start_beat": {"beat_id":"a planned beat id, or a new one", "kind":"", "intent":""},
  "tempo": "build|linger|pull_back|continue|transition|redirect|pause|close",
  "content_direction": "a direction for future content the fan himself expressed, if any"
}
"""

CONVERSATIONAL_V2_SYSTEM = lo.CONVERSATIONAL_V1_SYSTEM + "\n" + SESSION_OWNER_EXTENSION

#: Same minimum-object repair as v1. Session state never advances under a
#: format failure; the next readable turn resumes from the persisted state.
CONVERSATIONAL_V2_REPAIR_SYSTEM = lo.CONVERSATIONAL_V1_REPAIR_SYSTEM + (
    '- Omit session_delta entirely. You may include "next_experience_move": '
    '{"kind":"converse"}.\n'
)

SESSION_WRITER_EXTENSION = """SESSION CONTEXT (Conversational Core v2). interaction_session is the application-validated state of the longer interaction this reply belongs to. Write the next moment of THAT interaction: honour its premise, current beat, next_experience_move and tempo so the conversation reads as one continuing experience rather than filler between sales. A reply may carry the interaction with no content at all.

If premise_scope is imagined, keep imagined material explicitly imagined; it never becomes a claim about the creator's real current activity, clothing, location, surroundings, filming, posts or schedule. what_just_happened comes from the ledger: when content was just purchased or sent, stay in the moment it created instead of moving to another offer. Never announce, count or tease future content, never mention sessions, beats, plans, trajectories or constraints, and do not re-plan — the semantic decision already chose what happens next."""


# ---------------------------------------------------------------------------
# Owner
# ---------------------------------------------------------------------------


def exposed_candidates(
    loaded: lo.LoadedEvidence, state: ConversationalSessionState
) -> dict[str, Any]:
    """Eligible approved candidates, minus anything the ledger shows consumed."""
    consumed = state.session.consumed_refs()
    return {
        handle: offer
        for handle, offer in (getattr(loaded, "candidate_handles", {}) or {}).items()
        if handle.startswith(CANDIDATE_HANDLE_PREFIX)
        and str(offer.set_id) not in consumed
    }


async def decide_conversational_v2(
    loaded: lo.LoadedEvidence,
    state: ConversationalSessionState,
    candidates: dict[str, Any],
    *,
    owner_complete: Any = None,
) -> tuple[
    ConversationDecision, GenerationTrace, dict[str, Any], dict[str, Any], ExperienceMove
]:
    """One GLM semantic decision plus a proposed session delta; one repair max."""
    spec = loaded.stack.profile.stage(STAGE_CONVERSATIONAL_OWNER)
    target = spec.primary_target()
    max_tokens = lo._owner_max_tokens(spec)
    legal = lo.legal_operations(loaded)
    payload = {
        "turn_id": loaded.snapshot.trigger.identity,
        "conversation_revision": loaded.snapshot.state_revision,
        "evidence_snapshot": loaded.snapshot.as_dict(),
        "evidence_catalog": evidence_catalog_view(loaded.snapshot),
        "working_state": state.working.as_dict(),
        "working_state_fingerprint": state_fingerprint(state.working),
        "interaction_session": owner_session_view(
            state, snapshot=loaded.snapshot, candidates=candidates
        ),
        "session_state_fingerprint": session_fingerprint(state),
        "platform_context": loaded.snapshot.platform_context or platform_context(),
        "legal_operations": operation_affordances(legal),
        "scheduled_intent_affordance": SCHEDULED_INTENT_EFFECT,
    }
    user_content = json.dumps(payload, ensure_ascii=False, default=str)
    trace = GenerationTrace()
    if target is not None:
        trace.record_request(
            primary_target=target,
            fallback_target=spec.fallback_target(),
            profile=loaded.stack.profile_id,
            policy="glm_semantic_decision_only",
            deadline_seconds=0.0,
        )
    attempt = await lo._call_conversational_owner(
        loaded,
        target=target,
        max_tokens=max_tokens,
        user_content=user_content,
        system=CONVERSATIONAL_V2_SYSTEM,
        label="initial",
        owner_complete=owner_complete,
        parser=parse_session_decision,
    )
    lo._record_owner_attempt(trace, attempt, log_tag="CONVERSATIONAL V2 OWNER CALL")
    total_latency_ms = attempt.latency_ms
    attempts = 1
    repaired = False
    if not attempt.result.usable:
        first_failure = f"{attempt.failure_category}: {attempt.failure_detail}"
        repair = await lo._call_conversational_owner(
            loaded,
            target=lo._repair_target(target),
            max_tokens=max_tokens,
            user_content=user_content
            + "\nFORMAT REPAIR CONTEXT:\n"
            + json.dumps(
                {
                    "previous_attempt_failed_because": attempt.failure_category,
                    "required_minimum_object": {
                        "disposition": "reply",
                        "response_goal": "semantic goal",
                        "operation_proposal": {"kind": "none"},
                    },
                }
            ),
            system=CONVERSATIONAL_V2_REPAIR_SYSTEM,
            label="repair",
            owner_complete=owner_complete,
            parser=parse_session_decision,
        )
        lo._record_owner_attempt(trace, repair, log_tag="CONVERSATIONAL V2 OWNER CALL")
        total_latency_ms += repair.latency_ms
        attempts = 2
        if repair.result.usable:
            trace.repaired = True
            repaired = True
            attempt = repair
        else:
            trace.record_failure(
                outcome="conversational_v2_owner_invalid",
                reason=(
                    "the conversational owner did not answer usably after one "
                    f"repair: first={first_failure}; "
                    f"repair={repair.failure_category}: {repair.failure_detail}"
                ),
                attempts=attempts,
                pinned_attempts=attempts,
                alternate_attempts=0,
                elapsed_ms=total_latency_ms,
                deadline_exceeded=repair.failure_category == FAILURE_TIMEOUT,
            )
            return (
                ConversationDecision(
                    disposition=ResponseDisposition.SILENCE,
                    hold=HoldReason.INSUFFICIENT_EVIDENCE,
                    hold_detail="conversational_v2_owner_invalid",
                    source=SOURCE_V2,
                    confidence=0.0,
                ),
                trace,
                {},
                {},
                ExperienceMove(),
            )
    trace.record_success(
        target=attempt.target if attempt.target is not None else target,
        role="conversational_decision",
        attempt_index=attempts - 1,
        upstream_provider=attempt.upstream_provider,
        outcome=(
            "conversational_v2_repair_success"
            if repaired
            else "conversational_v2_first_try_success"
        ),
        attempts=attempts,
        pinned_attempts=attempts,
        alternate_attempts=0,
        elapsed_ms=total_latency_ms,
        usage=attempt.usage,
        reported_cost_usd=attempt.reported_cost_usd,
        served_model=attempt.served_model,
    )
    result = attempt.result
    session_delta = (
        {}
        if repaired or not isinstance(result, SessionDecisionResult)
        else dict(result.session_delta)
    )
    move = (
        result.next_experience_move
        if isinstance(result, SessionDecisionResult)
        else ExperienceMove()
    )
    working_delta = {} if repaired else dict(result.state_delta or {})
    return result.decision, trace, working_delta, session_delta, move


# ---------------------------------------------------------------------------
# Deterministic authority over the proposed operation
# ---------------------------------------------------------------------------


def _conversational_only(decision: ConversationDecision) -> ConversationDecision:
    return dataclasses_replace(
        decision,
        proposed_operation=ProposedOperation(),
        response_intent=ResponseIntent.ANSWER_AND_CONTINUE,
        disposition=(
            ResponseDisposition.REPLY
            if decision.disposition is not ResponseDisposition.SILENCE
            else decision.disposition
        ),
        hold=HoldReason.NONE
        if decision.disposition is not ResponseDisposition.SILENCE
        else decision.hold,
    )


def session_operation_refusals(
    loaded: lo.LoadedEvidence,
    decision: ConversationDecision,
    state: ConversationalSessionState,
) -> tuple[str, ...]:
    """v2-only guards, applied BEFORE the shared v1 validator.

    They can only remove an operation, never authorise one.
    """
    op = decision.proposed_operation
    if op.kind not in PAID_OPERATIONS:
        return ()
    reasons: list[str] = []
    offer = (
        loaded.commercial_state.pending_offer
        if op.kind is OperationKind.SEND_LOCKED_PAID_MESSAGE
        and loaded.commercial_state.pending_offer is not None
        else loaded.next_offer
    )
    session = state.session
    event = session.last_event
    snapshot = loaded.snapshot
    if (
        event.kind in {lifecycle.value for lifecycle in CONSUMED_LIFECYCLES}
        and event.observed_turn_ref == snapshot.trigger.identity
        and not (snapshot.commercial_opportunity or {}).get("fan_stated_buying_signal")
        and not (snapshot.media_request or {}).get("present")
    ):
        reasons.append(
            "content was just purchased or sent; the next beat stays in the "
            "interaction unless the fan asks for more"
        )
    if offer is not None:
        if str(offer.set_id) in session.consumed_refs():
            reasons.append("that content was already sent or purchased")
        remaining = remaining_spending_cents(state)
        if remaining is not None and offer.price_cents > remaining:
            reasons.append("the fan's own stated spending limit does not cover it")
    return tuple(reasons)


def _bind_candidate(loaded: lo.LoadedEvidence, decision: ConversationDecision) -> bool:
    """Point the shared validator at the approved candidate the owner chose.

    Only an exposed, application-priced candidate handle can be bound, and
    never over an offer that is already pending: a presented offer stays the
    one exact thing an acceptance can bind to.
    """
    op = decision.proposed_operation
    if op.kind not in PAID_OPERATIONS or not op.candidate_handle:
        return False
    if loaded.commercial_state.pending_offer is not None or loaded.pending_payment:
        return False
    chosen = (getattr(loaded, "candidate_handles", {}) or {}).get(op.candidate_handle)
    if chosen is None or not op.candidate_handle.startswith(CANDIDATE_HANDLE_PREFIX):
        return False
    if chosen is not loaded.next_offer:
        loaded.next_offer = chosen
        return True
    return False


async def authorize_v2_operation(
    loaded: lo.LoadedEvidence,
    *,
    decision: ConversationDecision,
    state: ConversationalSessionState,
    execute_operations: bool,
) -> lo.ConversationalV1Settlement:
    _bind_candidate(loaded, decision)
    refusals = session_operation_refusals(loaded, decision, state)
    proposed = decision.proposed_operation.kind.value
    if refusals:
        print(
            "[CONVERSATIONAL V2 SESSION GUARD] refused="
            + proposed
            + " reasons="
            + "; ".join(refusals)
        )
        decision = _conversational_only(decision)
    settled = await lo._authorize_conversational_v1_operation(
        loaded, decision=decision, execute_operations=execute_operations
    )
    settled_decision = settled.decision
    op = settled_decision.proposed_operation
    if (
        settled.execution.validation.approved
        and op.kind in PAID_OPERATIONS
        and loaded.next_offer is not None
    ):
        offer = (
            loaded.commercial_state.pending_offer
            if op.kind is OperationKind.SEND_LOCKED_PAID_MESSAGE
            and loaded.commercial_state.pending_offer is not None
            else loaded.next_offer
        )
        # Record the exact references authority approved, so provenance and
        # an Assisted approval rebind to THIS content, not to "the next one".
        settled_decision = dataclasses_replace(
            settled_decision,
            proposed_operation=dataclasses_replace(
                op, offer_id=offer.offer_id, set_id=offer.set_id
            ),
        )
    return lo.ConversationalV1Settlement(
        decision=settled_decision,
        execution=settled.execution,
        replies=[],
        locally_repaired=settled.locally_repaired or bool(refusals),
        proposed_operation=proposed,
        operation_rejection_reasons=(
            tuple(refusals) + tuple(settled.operation_rejection_reasons)
        ),
    )


# ---------------------------------------------------------------------------
# Turn preparation
# ---------------------------------------------------------------------------


def _session_packet(
    before: ConversationalSessionState, validation: SessionDeltaValidation
) -> dict[str, Any]:
    return {
        "session_state_before": before.as_dict(),
        "session_state_before_fingerprint": session_fingerprint(before),
        "proposed_session_delta": validation.proposed,
        "accepted_session_fields": list(validation.accepted_fields),
        "rejected_session_fields": dict(validation.rejected_fields),
        "session_reconciled": list(validation.reconciled),
        "session_state_after": validation.state_after.as_dict(),
        "session_state_after_fingerprint": session_fingerprint(validation.state_after),
    }


async def _load(
    *,
    creator_id: str,
    fan_id: str,
    trigger_kind: str,
    trigger_identity: str,
    latest_message: str,
    scheduled_goal: str,
    state: ConversationalSessionState,
) -> lo.LoadedEvidence:
    remaining = remaining_spending_cents(state)
    return await lo.load_evidence(
        creator_id=creator_id,
        fan_id=fan_id,
        trigger_kind=trigger_kind,
        trigger_identity=trigger_identity,
        latest_message=latest_message,
        scheduled_goal=scheduled_goal,
        content_candidates=True,
        desired_experience_hint=state.session.content_direction or None,
        extra_ceiling_cents=remaining,
    )


async def prepare_conversational_v2_turn(
    *,
    creator_id: str,
    fan_id: str,
    trigger_kind: str,
    trigger_identity: str,
    latest_message: str,
    scheduled_goal: str = "",
    mode: str,
    execute_operations: bool,
    hermes_retrieval_override: bool | None = None,
    owner_complete: Any = None,
) -> lo.PreparedTurn:
    stored = await load_session_state(creator_id, fan_id)
    loaded = await _load(
        creator_id=creator_id,
        fan_id=fan_id,
        trigger_kind=trigger_kind,
        trigger_identity=trigger_identity,
        latest_message=latest_message,
        scheduled_goal=scheduled_goal,
        state=stored,
    )

    def _reconcile(evidence: lo.LoadedEvidence):
        return reconcile_with_authority(
            stored,
            snapshot=evidence.snapshot,
            pending_offer=evidence.commercial_state.pending_offer,
            accepted_set_id=getattr(evidence.commercial_state, "accepted_offer_set_id", None),
        )

    reconciled, changes = _reconcile(loaded)
    candidates = exposed_candidates(loaded, reconciled)
    decision, decision_trace, working_delta, session_delta, move = (
        await decide_conversational_v2(
            loaded, reconciled, candidates, owner_complete=owner_complete
        )
    )
    unavailable = lo._unavailable_evidence_requests(decision, loaded)
    if unavailable:
        loaded = await _load(
            creator_id=creator_id,
            fan_id=fan_id,
            trigger_kind=trigger_kind,
            trigger_identity=trigger_identity,
            latest_message=latest_message,
            scheduled_goal=scheduled_goal,
            state=stored,
        )
        reconciled, changes = _reconcile(loaded)
        candidates = exposed_candidates(loaded, reconciled)
        (
            decision,
            second_trace,
            working_delta,
            session_delta,
            move,
        ) = await decide_conversational_v2(
            loaded, reconciled, candidates, owner_complete=owner_complete
        )
        decision_trace = lo._merge_decision_traces(decision_trace, second_trace)
        still_missing = lo._unavailable_evidence_requests(decision, loaded)
        if still_missing:
            decision = ConversationDecision(
                disposition=ResponseDisposition.HANDOFF,
                hold=HoldReason.INSUFFICIENT_EVIDENCE,
                hold_detail="same_turn_evidence_unavailable: " + ",".join(still_missing),
                proposed_operation=ProposedOperation(
                    kind=OperationKind.HAND_OFF_TO_HUMAN,
                    subject="missing required evidence",
                ),
                source=SOURCE_V2,
                confidence=0.0,
            )

    known_thread_ids = {
        str(thread.get("id"))
        for thread in loaded.snapshot.unresolved_obligations
        if thread.get("id")
    }
    try:
        working_validation = validate_and_apply_delta(
            reconciled.working,
            working_delta,
            snapshot=loaded.snapshot,
            known_thread_ids=known_thread_ids,
        )
    except Exception as exc:  # noqa: BLE001 - a delta may never kill a reply
        print(f"[CONVERSATIONAL V2 WORKING DELTA UNREADABLE] {type(exc).__name__}: {exc}")
        working_validation = StateDeltaValidation(
            proposed=working_delta if isinstance(working_delta, dict) else {},
            rejected_fields={"state_delta": f"unreadable: {type(exc).__name__}"},
            state_after=reconciled.working.model_copy(deep=True),
        )
    try:
        validation = validate_session_delta(
            reconciled,
            session_delta,
            snapshot=loaded.snapshot,
            candidates=candidates,
            working_validation=working_validation,
            next_experience_move=move,
            reconciled=changes,
            base_revision=stored.revision,
        )
    except Exception as exc:  # noqa: BLE001 - interpretation never kills a reply
        print(f"[CONVERSATIONAL V2 SESSION DELTA UNREADABLE] {type(exc).__name__}: {exc}")
        fallback = reconciled.model_copy(deep=True)
        fallback.revision = stored.revision + (1 if changes else 0)
        validation = SessionDeltaValidation(
            proposed=session_delta if isinstance(session_delta, dict) else {},
            rejected_fields={"session_delta": f"unreadable: {type(exc).__name__}"},
            reconciled=list(changes),
            state_after=fallback,
        )
    state_after = validation.state_after

    authorized = await authorize_v2_operation(
        loaded,
        decision=decision,
        state=state_after,
        execute_operations=execute_operations,
    )
    decision = authorized.decision
    execution = authorized.execution
    locally_repaired = authorized.locally_repaired

    behavior_tags, situation_tags = lo._hermes_tags(
        decision, state_after.working, latest_message
    )
    loaded.hermes_retrieval_active = retrieval_enabled(hermes_retrieval_override)
    loaded.hermes_examples = retrieve_examples(
        current_text=latest_message,
        behavior_tags=behavior_tags,
        situation_tags=situation_tags,
        enabled_override=hermes_retrieval_override,
    )
    replies: list[str] = []
    trace = GenerationTrace()
    if decision.disposition is ResponseDisposition.REPLY:
        replies, trace = await lo._write_conversational_v1_turn(
            loaded,
            decision,
            execution,
            state_after.working,
            mode=mode,
            extra_system=SESSION_WRITER_EXTENSION,
            extra_payload={
                "interaction_session": writer_session_view(
                    state_after,
                    snapshot=loaded.snapshot,
                    operation=execution.operation,
                )
            },
            feature="conversational_v2_writer",
            conversation_core=CORE_CONVERSATIONAL_V2,
        )
        decision, execution, replies, copy_repaired = (
            lo._validate_conversational_v1_reply(
                decision, replies, execution, loaded, mode=mode
            )
        )
        locally_repaired = locally_repaired or copy_repaired

    provenance = lo._provenance(
        loaded,
        decision,
        execution,
        trace,
        mode=(PIPELINE_ASSISTED if mode == MODE_ASSISTED else PIPELINE_AUTO),
        conversation_core=CORE_CONVERSATIONAL_V2,
        working_state_before=reconciled.working,
        state_delta_validation=working_validation,
        decision_trace=decision_trace,
    )
    packet = provenance.context.get("packet")
    if isinstance(packet, dict):
        packet.update(_session_packet(stored, validation))
    provenance.decision.update(
        {
            "next_experience_move": state_after.session.next_experience_move.kind.value,
            "session_status": state_after.session.status.value,
            "session_operation_rejections": "; ".join(
                authorized.operation_rejection_reasons
            ),
        }
    )
    if locally_repaired:
        provenance.record_transform("conversational_v2_local_repair")
    return lo.PreparedTurn(
        loaded=loaded,
        decision=decision,
        execution=execution,
        replies=replies,
        provenance=provenance,
        writer_trace=trace,
        decision_trace=decision_trace,
        conversation_core=CORE_CONVERSATIONAL_V2,
        session_state_before=stored,
        session_validation=validation,
    )


async def persist_session_state(prepared: lo.PreparedTurn) -> None:
    """Persist the validated session document with a revision compare-and-swap."""
    before = prepared.session_state_before
    validation = prepared.session_validation
    if prepared.state_persisted or before is None or validation is None:
        return
    after = validation.state_after
    if after.revision != before.revision:
        await save_session_state(
            prepared.loaded.snapshot.creator_id,
            prepared.loaded.fan.id,
            expected_revision=before.revision,
            state=after,
        )
    prepared.state_persisted = True
    packet = prepared.provenance.context.get("packet")
    if isinstance(packet, dict):
        packet["session_state_persisted"] = True


async def prepare_v2_assisted_approval(
    provenance: Any, *, creator_id: str, fan_id: str
) -> lo.PreparedTurn:
    """Rebind a v2 Assisted token to current authoritative state, or refuse."""
    if provenance.creator_id != str(creator_id) or provenance.fan_id != str(fan_id):
        raise lo.LiveOrchestrationError("the Assisted approval token is out of scope")
    recorded = provenance.decision or {}
    packet = (provenance.context or {}).get("packet") or {}
    try:
        recorded_before = ConversationalSessionState.model_validate(
            packet.get("session_state_before")
        )
        recorded_after = ConversationalSessionState.model_validate(
            packet.get("session_state_after")
        )
    except Exception as exc:
        raise lo.LiveOrchestrationError(
            "the Assisted Core v2 draft has invalid session-state provenance"
        ) from exc
    current_state = await load_session_state(creator_id, fan_id)
    if session_fingerprint(current_state) != session_fingerprint(recorded_before):
        raise lo.LiveOrchestrationError(
            "the Core v2 session state changed after this Assisted draft was generated"
        )
    history = await lo.get_conversation_history(fan_id)
    latest = next((row.content for row in reversed(history) if row.role == "fan"), "")
    loaded = await _load(
        creator_id=creator_id,
        fan_id=fan_id,
        trigger_kind="assisted_approval",
        trigger_identity=provenance.turn_id,
        latest_message=latest,
        scheduled_goal="",
        state=current_state,
    )
    expected_revision = str(recorded.get("semantic_state_revision") or "")
    if not expected_revision or loaded.snapshot.state_revision != expected_revision:
        raise lo.LiveOrchestrationError(
            "the conversation changed after this Assisted draft was generated"
        )
    try:
        operation = OperationKind(str(recorded.get("semantic_operation") or "none"))
    except ValueError as exc:
        raise lo.LiveOrchestrationError(
            "the Assisted draft carries an unknown semantic operation"
        ) from exc
    set_id = str(recorded.get("semantic_set_id") or "")
    if (
        operation in PAID_OPERATIONS
        and set_id
        and loaded.commercial_state.pending_offer is None
    ):
        for handle, offer in (loaded.candidate_handles or {}).items():
            if handle.startswith(CANDIDATE_HANDLE_PREFIX) and offer.set_id == set_id:
                loaded.next_offer = offer
                break
    decision = ConversationDecision(
        proposed_operation=ProposedOperation(
            kind=operation,
            offer_id=str(recorded.get("semantic_offer_id") or ""),
            set_id=set_id,
            payment_reference=str(recorded.get("semantic_payment_reference") or ""),
            purchase_id=str(recorded.get("semantic_purchase_id") or ""),
            subject="approved Assisted operation"
            if operation is not OperationKind.NONE
            else "",
        ),
        disposition=ResponseDisposition.REPLY,
        source=SOURCE_V2,
    )
    refusals = session_operation_refusals(loaded, decision, recorded_after)
    if refusals:
        raise lo.LiveOrchestrationError(
            "Assisted operation refused by session state: " + "; ".join(refusals)
        )
    execution = await lo._prepare_execution(decision, loaded, execute_operations=True)
    if not execution.validation.approved:
        raise lo.LiveOrchestrationError(
            "Assisted operation validation refused: "
            + "; ".join(execution.validation.reasons)
        )
    return lo.PreparedTurn(
        loaded=loaded,
        decision=decision,
        execution=execution,
        replies=[],
        provenance=provenance,
        writer_trace=GenerationTrace(),
        conversation_core=CORE_CONVERSATIONAL_V2,
        session_state_before=current_state,
        session_validation=SessionDeltaValidation(state_after=recorded_after),
    )


__all__ = [
    "CONVERSATIONAL_V2_SYSTEM",
    "SESSION_WRITER_EXTENSION",
    "authorize_v2_operation",
    "decide_conversational_v2",
    "persist_session_state",
    "prepare_conversational_v2_turn",
    "prepare_v2_assisted_approval",
    "session_operation_refusals",
]
