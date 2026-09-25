"""Deterministic authority over ``conversational_v2`` session state.

GLM proposes a ``session_delta``; nothing in it is trusted. This module:

* rebuilds the content lifecycle (presented / accepted / sent / purchased) from
  authoritative records BEFORE the owner decides, so a model can never mark
  content used or unused (:func:`reconcile_with_authority`);
* validates every proposed field independently, keeping the safe ones and
  reporting the refused ones (:func:`validate_session_delta`);
* renders bounded, role-specific views — the owner sees opaque content handles
  and never a catalogue; the writer sees the interaction, never future content
  (:func:`owner_session_view`, :func:`writer_session_view`);
* persists one versioned row per creator/fan with a revision compare-and-swap,
  in its own table, so v1 and v2 never share state.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError

from core.supabase import get_supabase
from models.commercial import Offer
from models.conversational_core import (
    ElementStatus,
    EpistemicType,
    StateDeltaValidation,
    WorldScope,
)
from models.conversational_session import (
    CONSUMED_LIFECYCLES,
    ENDED_STATUSES,
    LIFECYCLE_RANK,
    LIVE_STATUSES,
    MAX_COMPLETED_BEATS,
    MAX_CONSTRAINTS,
    MAX_INFORMATION_NEEDS,
    MAX_PREVIOUS_SESSIONS,
    MAX_REPLANS,
    MAX_TRAJECTORY_BEATS,
    MAX_USED_CONTENT,
    SESSION_STATE_SCHEMA_VERSION,
    STATUS_TRANSITIONS,
    BeatKind,
    BeatSource,
    CompletedBeat,
    ConstraintKind,
    ContentLifecycle,
    ContentUse,
    ConversationalSessionState,
    ExperienceMove,
    ExperiencePremise,
    InformationNeed,
    InformationTopic,
    InteractionSession,
    KnownConstraint,
    LastEvent,
    NeedStatus,
    PlannedBeat,
    ProposedBeat,
    ProposedConstraint,
    ReplanRecord,
    SessionDelta,
    SessionDeltaValidation,
    SessionSummary,
)
from models.live_orchestration import EvidenceSnapshot
from services.conversational_core import (
    CoreStateConflictError,
    CoreStateCorruptionError,
    CoreStateError,
    EvidenceAuthority,
    _TRANSACTION_CLAIM,
    _unsafe_interpretation,
    evidence_catalog,
)

SESSION_TABLE = "conversational_session_states"

#: Handles for approved content the owner may bind a beat to. ``offer_candidate_1``
#: is the exact same next offer v1 would see; the others are bounded, eligible
#: alternatives the application priced under every explicit ceiling.
CANDIDATE_HANDLE_PREFIX = "offer_candidate_"

_NUMBER = re.compile(r"(?<![\w.])\$?\s*(\d{1,6}(?:[.,]\d{1,2})?)(?![\w])")


def _plain(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


_PRICE_TEXT = re.compile(
    r"[$€£]\s*\d|\b\d+(?:[.,]\d+)?\s*(?:dollars?|bucks?|usd|eur|euros?|cents?)\b",
    re.IGNORECASE,
)
_COMPLETED_TRANSACTION = re.compile(
    r"\b(?:has|have|had|was|were|is|are|already|just|got)\s+(?:been\s+)?"
    r"(?:paid|purchased|bought|sent|delivered|attached|unlocked|charged|refunded)\b|"
    r"\b(?:payment|purchase)\s+(?:is\s+|was\s+)?(?:confirmed|received|completed)\b",
    re.IGNORECASE,
)


def _unsafe_intent(value: Any) -> str:
    """Goals and intents may TALK ABOUT money; they may not assert it happened.

    v1's interpretation check refuses any transaction vocabulary, which is right
    for statements of fact but would refuse "learn his limit before planning a
    longer paid evening". An intent still may not state a price (the writer
    would repeat it), assert a completed transaction (only the ledger can), or
    assert unscoped present-world creator activity.
    """
    text = _plain(value, 1_000)
    if _PRICE_TEXT.search(text):
        return "an intent cannot state a price; the application owns prices"
    if _COMPLETED_TRANSACTION.search(text):
        return "an intent cannot assert payment or delivery; the ledger records those"
    reason = _unsafe_interpretation(text)
    if reason and "present-world" in reason:
        return reason
    return ""


def empty_session_state() -> ConversationalSessionState:
    return ConversationalSessionState()


def session_fingerprint(state: ConversationalSessionState) -> str:
    encoded = json.dumps(state.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def fan_message_texts(snapshot: EvidenceSnapshot) -> dict[str, str]:
    """Raw fan-authored text by message id: the only proof of a fan statement."""
    texts: dict[str, str] = {}
    for row in (*snapshot.recent_messages, *snapshot.latest_fan_burst):
        if str(row.get("speaker") or "") != "fan":
            continue
        message_id = str(row.get("message_id") or "").strip()
        if message_id:
            texts[message_id] = str(row.get("text") or "")
    trigger = str(snapshot.trigger.identity or "").strip()
    if trigger and snapshot.trigger.latest_message and trigger not in texts:
        texts[trigger] = snapshot.trigger.latest_message
    return texts


def session_evidence_catalog(
    snapshot: EvidenceSnapshot,
) -> dict[str, EvidenceAuthority]:
    """v1's catalog plus every visible raw fan message as fan evidence."""
    catalog = dict(evidence_catalog(snapshot))
    for message_id in fan_message_texts(snapshot):
        catalog.setdefault(
            message_id, EvidenceAuthority(EpistemicType.EXPLICIT_FAN_STATEMENT)
        )
    return catalog


def application_spending_limits(snapshot: EvidenceSnapshot) -> dict[str, int]:
    """Spending limits the application already holds from its own records."""
    limits: dict[str, int] = {}
    for key in ("explicit_current_limit_cents", "explicit_current_available_cents"):
        try:
            cents = int((snapshot.spending_limits or {}).get(key) or 0)
        except (TypeError, ValueError):
            cents = 0
        if cents > 0:
            limits[key] = cents
    return limits


def _text_states_amount(text: str, amount_cents: int) -> bool:
    for match in _NUMBER.finditer(str(text or "")):
        try:
            value = Decimal(match.group(1).replace(",", "."))
        except InvalidOperation:
            continue
        if int(value * 100) == int(amount_cents):
            return True
    return False


def _cents(row: dict[str, Any]) -> int:
    try:
        if row.get("price_cents") not in (None, ""):
            return int(row["price_cents"])
        return int(round(float(row.get("price") or 0) * 100))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Deterministic reconciliation with authoritative records
# ---------------------------------------------------------------------------


def authoritative_content_facts(
    snapshot: EvidenceSnapshot,
    *,
    pending_offer: Offer | None = None,
    accepted_set_id: str | None = None,
) -> dict[str, tuple[ContentLifecycle, str, int]]:
    """set_id -> (strongest lifecycle, evidence ref, price cents).

    Built only from the delivery ledger, the confirmed-purchase ledger and the
    commercial state row. No model output reaches this function.
    """
    facts: dict[str, tuple[ContentLifecycle, str, int]] = {}

    def _note(set_id: Any, lifecycle: ContentLifecycle, ref: str, cents: int) -> None:
        key = str(set_id or "").strip()
        if not key:
            return
        current = facts.get(key)
        if current is None or LIFECYCLE_RANK[lifecycle] > LIFECYCLE_RANK[current[0]]:
            facts[key] = (lifecycle, ref, cents)

    if pending_offer is not None:
        _note(pending_offer.set_id, ContentLifecycle.PRESENTED, "pending_offer", 0)
    if accepted_set_id:
        _note(accepted_set_id, ContentLifecycle.ACCEPTED, "accepted_offer", 0)
    for row in snapshot.confirmed_deliveries:
        _note(
            row.get("set_id"),
            ContentLifecycle.SENT,
            str(row.get("platform_message_id") or row.get("reference") or "delivery"),
            _cents(row),
        )
    for row in snapshot.confirmed_purchases:
        _note(
            row.get("set_id"),
            ContentLifecycle.PURCHASED,
            str(row.get("reference") or row.get("payment_reference") or "purchase"),
            _cents(row),
        )
    return facts


def reconcile_with_authority(
    state: ConversationalSessionState,
    *,
    snapshot: EvidenceSnapshot,
    pending_offer: Offer | None = None,
    accepted_set_id: str | None = None,
) -> tuple[ConversationalSessionState, list[str]]:
    """Advance content facts from authoritative records. Never regress them.

    A lifecycle only ever moves forward: once content was sent or purchased it
    cannot become unused again because a ledger page was truncated or a model
    omitted it. Beats bound to consumed content are completed by the
    APPLICATION, with the ledger as their source.
    """
    after = state.model_copy(deep=True)
    session = after.session
    turn_ref = str(snapshot.trigger.identity or "")
    changes: list[str] = []
    facts = authoritative_content_facts(
        snapshot, pending_offer=pending_offer, accepted_set_id=accepted_set_id
    )
    known: dict[str, ContentLifecycle] = {}
    for use in session.used_content:
        prior = known.get(use.content_ref)
        if prior is None or LIFECYCLE_RANK[use.lifecycle] > LIFECYCLE_RANK[prior]:
            known[use.content_ref] = use.lifecycle

    newest: tuple[int, ContentLifecycle, str] | None = None
    for set_id, (lifecycle, ref, _cents_value) in facts.items():
        prior = known.get(set_id)
        if prior is not None and LIFECYCLE_RANK[lifecycle] <= LIFECYCLE_RANK[prior]:
            continue
        session.content_fact_seq += 1
        session.used_content.append(
            ContentUse(
                content_ref=set_id,
                lifecycle=lifecycle,
                source_ref=_plain(ref, 200),
                observed_turn_ref=turn_ref,
                seq=session.content_fact_seq,
                price_cents=max(0, int(_cents_value or 0)),
            )
        )
        known[set_id] = lifecycle
        changes.append(f"content:{lifecycle.value}")
        rank = LIFECYCLE_RANK[lifecycle]
        if newest is None or rank > newest[0]:
            newest = (rank, lifecycle, set_id)
    if len(session.used_content) > MAX_USED_CONTENT:
        # Keep the strongest fact per content; drop superseded weaker rows first.
        strongest: dict[str, ContentUse] = {}
        for use in session.used_content:
            held = strongest.get(use.content_ref)
            if held is None or LIFECYCLE_RANK[use.lifecycle] >= LIFECYCLE_RANK[
                held.lifecycle
            ]:
                strongest[use.content_ref] = use
        session.used_content = list(strongest.values())[-MAX_USED_CONTENT:]

    if newest is not None:
        session.last_event = LastEvent(
            kind=newest[1].value, content_ref=newest[2], observed_turn_ref=turn_ref
        )
        if newest[1] in CONSUMED_LIFECYCLES:
            session.turns_since_content_event = 0

    consumed = session.consumed_refs()
    remaining: list[PlannedBeat] = []
    for beat in session.tentative_trajectory:
        if beat.content_ref and beat.content_ref in consumed:
            _complete(
                session,
                beat,
                outcome="content " + known[beat.content_ref].value + " (ledger)",
                turn_ref=turn_ref,
                source=BeatSource.APPLICATION,
            )
            changes.append(f"beat_consumed:{beat.beat_id}")
        else:
            remaining.append(beat)
    session.tentative_trajectory = remaining
    current = session.current_beat
    if current is not None and current.content_ref and current.content_ref in consumed:
        _complete(
            session,
            current,
            outcome="content " + known[current.content_ref].value + " (ledger)",
            turn_ref=turn_ref,
            source=BeatSource.APPLICATION,
        )
        session.current_beat = None
        changes.append(f"beat_consumed:{current.beat_id}")
    after = ConversationalSessionState.model_validate(after.model_dump(mode="json"))
    return after, changes


def session_spent_cents(state: ConversationalSessionState) -> int:
    """Confirmed spend recorded since the current spending limit was stated."""
    constraint = state.session.spending_constraint()
    if constraint is None:
        return 0
    return sum(
        use.price_cents
        for use in state.session.used_content
        if use.lifecycle is ContentLifecycle.PURCHASED
        and use.seq > constraint.after_content_seq
    )


def remaining_spending_cents(state: ConversationalSessionState) -> int | None:
    """What the fan's own stated limit still covers, or None when he stated none."""
    constraint = state.session.spending_constraint()
    if constraint is None or not constraint.amount_cents:
        return None
    return int(constraint.amount_cents) - session_spent_cents(state)


# ---------------------------------------------------------------------------
# Delta validation
# ---------------------------------------------------------------------------


def _parse_session_delta(raw: Any) -> tuple[SessionDelta | None, dict[str, str]]:
    if raw in (None, "", {}):
        return SessionDelta(), {}
    if not isinstance(raw, dict):
        return None, {"session_delta": "session_delta must be an object"}
    try:
        return SessionDelta.model_validate(raw), {}
    except ValidationError as exc:
        rejected: dict[str, str] = {}
        bad_roots = {str(error["loc"][0]) for error in exc.errors() if error.get("loc")}
        for root in sorted(bad_roots):
            if root in {"used_content", "last_event"}:
                rejected[root] = (
                    "content lifecycle is application-owned; the owner cannot write it"
                )
            else:
                rejected[root] = "malformed or unknown session field"
        salvage = {key: value for key, value in raw.items() if key not in bad_roots}
        try:
            return SessionDelta.model_validate(salvage), rejected
        except ValidationError:
            return None, {
                "session_delta": "session_delta could not be safely parsed",
                **rejected,
            }


def _new_session_id(snapshot: EvidenceSnapshot, previous: int) -> str:
    seed = f"{snapshot.creator_id}:{snapshot.fan_id}:{snapshot.trigger.identity}:{previous}"
    return "session-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def _complete(
    session: InteractionSession,
    beat: PlannedBeat,
    *,
    outcome: str,
    turn_ref: str,
    source: BeatSource,
) -> None:
    session.completed_beats.append(
        CompletedBeat(
            beat_id=beat.beat_id,
            kind=beat.kind,
            intent=beat.intent,
            outcome=_plain(outcome, 300),
            content_ref=beat.content_ref,
            completed_turn_ref=turn_ref,
            source=source,
        )
    )
    overflow = len(session.completed_beats) - MAX_COMPLETED_BEATS
    if overflow > 0:
        session.completed_beats = session.completed_beats[overflow:]
        session.completed_beats_trimmed += overflow


def _archive(state: ConversationalSessionState, turn_ref: str) -> None:
    session = state.session
    if not session.session_id:
        return
    state.previous_sessions.append(
        SessionSummary(
            session_id=session.session_id,
            final_status=session.status,
            premise=session.experience_premise.summary,
            completed_beats=len(session.completed_beats)
            + session.completed_beats_trimmed,
            content_events=sum(
                1 for use in session.used_content if use.lifecycle in CONSUMED_LIFECYCLES
            ),
            ended_turn_ref=turn_ref,
        )
    )
    state.previous_sessions = state.previous_sessions[-MAX_PREVIOUS_SESSIONS:]


def _refs_ok(
    refs: list[str],
    catalog: dict[str, EvidenceAuthority],
) -> str:
    if not refs:
        return "needs an evidence reference"
    unknown = [ref for ref in refs if ref not in catalog]
    if unknown:
        return "unknown evidence reference: " + ", ".join(unknown[:3])
    return ""


def _validate_constraint(
    proposal: ProposedConstraint,
    *,
    catalog: dict[str, EvidenceAuthority],
    fan_texts: dict[str, str],
) -> str:
    reason = _refs_ok(proposal.source_refs, catalog)
    if reason:
        return reason
    authorities = [catalog[ref] for ref in proposal.source_refs]
    if proposal.source_type is EpistemicType.EXPLICIT_FAN_STATEMENT:
        if not all(
            item.source_type is EpistemicType.EXPLICIT_FAN_STATEMENT
            for item in authorities
        ):
            return "a fan constraint must cite the fan's own messages"
    elif proposal.source_type is EpistemicType.TRANSACTION_FACT:
        if not all(item.authoritative_transaction for item in authorities):
            return "a transaction constraint must cite authoritative ledger evidence"
    elif proposal.source_type is EpistemicType.CREATOR_CONFIG:
        if not all(
            item.source_type is EpistemicType.CREATOR_CONFIG for item in authorities
        ):
            return "a creator constraint must cite creator configuration"
    else:
        return "constraints are evidence-only; inference cannot establish one"
    if proposal.kind is ConstraintKind.SPENDING_LIMIT:
        if proposal.source_type is not EpistemicType.EXPLICIT_FAN_STATEMENT:
            return "a spending limit exists only when the fan states it"
        if not proposal.amount_cents:
            return "a spending limit needs the amount the fan stated"
        if not any(
            _text_states_amount(fan_texts.get(ref, ""), proposal.amount_cents)
            for ref in proposal.source_refs
        ):
            return "the cited fan message does not state that amount"
    elif proposal.amount_cents is not None:
        return "only a spending limit carries an amount"
    if proposal.kind is not ConstraintKind.SPENDING_LIMIT:
        unsafe = _unsafe_interpretation(proposal.statement)
        if unsafe:
            return unsafe
    return ""


def _validate_beat(
    proposal: ProposedBeat,
    *,
    candidates: dict[str, Offer],
    consumed: set[str],
) -> tuple[str, str]:
    """Return (refusal reason, resolved content_ref)."""
    if proposal.kind is None:
        return "a new beat needs a kind", ""
    if proposal.intent:
        unsafe = _unsafe_intent(proposal.intent)
        if unsafe:
            return unsafe, ""
    if proposal.media_role:
        if _TRANSACTION_CLAIM.search(proposal.media_role):
            return "a media role describes interaction value, not money or delivery", ""
    if proposal.kind is not BeatKind.CONTENT:
        if proposal.candidate_handle:
            return "only a content beat may bind approved content", ""
        return "", ""
    handle = proposal.candidate_handle
    if not handle:
        return "a content beat must bind one exposed approved candidate", ""
    offer = candidates.get(handle)
    if offer is None:
        return "no eligible approved content exists for that handle", ""
    if str(offer.set_id) in consumed:
        return "that content was already sent or purchased", ""
    if not _plain(proposal.media_role, 300):
        return (
            "a content beat must say what role the content plays in the interaction",
            "",
        )
    return "", str(offer.set_id)


def _beat_id(proposal: ProposedBeat, *, taken: set[str], index: int, turn: str) -> str:
    wanted = re.sub(r"[^a-zA-Z0-9_.-]+", "-", proposal.beat_id or "").strip("-")[:80]
    if wanted and wanted not in taken:
        return wanted
    seed = hashlib.sha256(f"{turn}:{index}:{proposal.intent}".encode()).hexdigest()[:8]
    base = f"beat-{seed}"
    candidate = base
    suffix = 1
    while candidate in taken:
        suffix += 1
        candidate = f"{base}-{suffix}"
    return candidate


def validate_session_delta(
    state: ConversationalSessionState,
    raw_delta: Any,
    *,
    snapshot: EvidenceSnapshot,
    candidates: dict[str, Offer] | None = None,
    working_validation: StateDeltaValidation | None = None,
    next_experience_move: ExperienceMove | None = None,
    reconciled: list[str] | None = None,
    base_revision: int | None = None,
) -> SessionDeltaValidation:
    """Apply safe session fields and report every field that was refused.

    ``state`` is the RECONCILED state. ``working_validation`` is the result of
    v1's own validator applied to the embedded working state; it is folded in
    unchanged so v2 inherits every v1 protection verbatim.
    """
    proposed = copy.deepcopy(raw_delta) if isinstance(raw_delta, dict) else {}
    after = state.model_copy(deep=True)
    accepted: list[str] = []
    delta, rejected = _parse_session_delta(raw_delta)
    turn_ref = str(snapshot.trigger.identity or "")
    catalog = session_evidence_catalog(snapshot)
    fan_texts = fan_message_texts(snapshot)
    candidates = dict(candidates or {})

    if working_validation is not None:
        after.working = working_validation.state_after.model_copy(deep=True)
        accepted.extend(f"working.{name}" for name in working_validation.accepted_fields)
        rejected.update(
            {
                f"working.{name}": reason
                for name, reason in working_validation.rejected_fields.items()
            }
        )

    session = after.session
    if delta is not None:
        # 1. Status. A move into a live status from an ended one starts a NEW
        #    session; content facts and still-active constraints carry over.
        if delta.status is not None and delta.status is not session.status:
            allowed = STATUS_TRANSITIONS.get(session.status, frozenset())
            if delta.status not in allowed:
                rejected["status"] = (
                    f"cannot move a {session.status.value} session to "
                    f"{delta.status.value}"
                )
            else:
                if session.status in ENDED_STATUSES and delta.status in LIVE_STATUSES:
                    _archive(after, turn_ref)
                    carried_constraints = session.active_constraints()
                    carried_content = list(session.used_content)
                    carried_event = session.last_event
                    session = InteractionSession(
                        session_id=_new_session_id(
                            snapshot, len(after.previous_sessions)
                        ),
                        status=delta.status,
                        known_constraints=carried_constraints,
                        used_content=carried_content,
                        content_fact_seq=state.session.content_fact_seq,
                        last_event=carried_event,
                        started_turn_ref=turn_ref,
                        experience_premise=state.session.experience_premise,
                        fan_participation=state.session.fan_participation,
                        turns_since_content_event=state.session.turns_since_content_event,
                    )
                    after.session = session
                else:
                    session.status = delta.status
                    if delta.status in ENDED_STATUSES:
                        if session.current_beat is not None:
                            _complete(
                                session,
                                session.current_beat,
                                outcome=f"session {delta.status.value}",
                                turn_ref=turn_ref,
                                source=BeatSource.MODEL,
                            )
                            session.current_beat = None
                        dropped = [beat.beat_id for beat in session.tentative_trajectory]
                        if dropped:
                            session.replans.append(
                                ReplanRecord(
                                    turn_ref=turn_ref,
                                    reason=f"session {delta.status.value}",
                                    replaced_beat_ids=dropped[:12],
                                )
                            )
                            session.replans = session.replans[-MAX_REPLANS:]
                        session.tentative_trajectory = []
                accepted.append("status")

        # 2. Premise: conversational or imagined, never real-world.
        if delta.experience_premise is not None:
            premise = delta.experience_premise
            reason = _refs_ok(premise.source_refs, catalog)
            if not reason and premise.world_scope not in {
                WorldScope.CONVERSATION,
                WorldScope.IMAGINED_SCENE,
            }:
                reason = "a premise is conversational or imagined, never a real-world fact"
            if not reason and premise.world_scope is WorldScope.IMAGINED_SCENE:
                if not any(
                    catalog[ref].source_type is EpistemicType.EXPLICIT_FAN_STATEMENT
                    for ref in premise.source_refs
                ):
                    reason = (
                        "a SHARED imagined premise needs the fan's own participation "
                        "as evidence"
                    )
                elif _TRANSACTION_CLAIM.search(premise.summary):
                    reason = (
                        "a premise cannot assert payment, purchase, price, or delivery"
                    )
            elif not reason:
                reason = _unsafe_interpretation(premise.summary)
            if reason:
                rejected["experience_premise"] = reason
            else:
                session.experience_premise = ExperiencePremise(
                    summary=premise.summary,
                    world_scope=premise.world_scope,
                    source_refs=premise.source_refs,
                    established_turn_ref=turn_ref,
                )
                accepted.append("experience_premise")

        for name in ("interaction_goal", "content_direction"):
            value = getattr(delta, name)
            if value is None:
                continue
            reason = _unsafe_intent(value)
            if reason:
                rejected[name] = reason
            else:
                setattr(session, name, _plain(value, 300 if name == "interaction_goal" else 200))
                accepted.append(name)

        if delta.fan_participation is not None:
            reason = _unsafe_intent(delta.fan_participation.gist)
            if reason:
                rejected["fan_participation"] = reason
            else:
                session.fan_participation = delta.fan_participation
                accepted.append("fan_participation")

        # 3. Constraints: evidence only.
        existing_ids = {c.constraint_id for c in session.known_constraints}
        for index, proposal in enumerate(delta.add_constraints):
            key = f"add_constraints[{index}]"
            reason = _validate_constraint(proposal, catalog=catalog, fan_texts=fan_texts)
            if not reason and proposal.constraint_id in existing_ids:
                reason = "constraint_id already exists; use a correction"
            if not reason and len(session.known_constraints) >= MAX_CONSTRAINTS:
                reason = "constraint capacity reached"
            if (
                not reason
                and proposal.kind is ConstraintKind.SPENDING_LIMIT
                and session.spending_constraint() is not None
            ):
                reason = "a spending limit is already known; correct it instead"
            if reason:
                rejected[key] = reason
                continue
            session.known_constraints.append(
                KnownConstraint(
                    **proposal.model_dump(mode="json"),
                    introduced_turn_ref=turn_ref,
                    after_content_seq=session.content_fact_seq,
                )
            )
            existing_ids.add(proposal.constraint_id)
            accepted.append(key)
        by_id = {c.constraint_id: c for c in session.known_constraints}
        for index, correction in enumerate(delta.constraint_corrections):
            key = f"constraint_corrections[{index}]"
            old = by_id.get(correction.replaces_constraint_id)
            replacement = correction.replacement
            reason = ""
            if old is None or old.status is not ElementStatus.ACTIVE:
                reason = "correction target is not an active constraint"
            elif replacement.source_type is not EpistemicType.EXPLICIT_FAN_STATEMENT:
                reason = "only the fan's own statement may change a known constraint"
            else:
                reason = _validate_constraint(
                    replacement, catalog=catalog, fan_texts=fan_texts
                )
            if not reason and replacement.constraint_id in existing_ids:
                reason = "replacement constraint_id already exists"
            if not reason and len(session.known_constraints) >= MAX_CONSTRAINTS:
                reason = "constraint capacity reached"
            if reason:
                rejected[key] = reason
                continue
            old.status = ElementStatus.SUPERSEDED
            old.superseded_by = replacement.constraint_id
            new = KnownConstraint(
                **replacement.model_dump(mode="json"),
                introduced_turn_ref=turn_ref,
                after_content_seq=session.content_fact_seq,
            )
            session.known_constraints.append(new)
            by_id[new.constraint_id] = new
            existing_ids.add(new.constraint_id)
            accepted.append(key)

        # 4. Information needs: only what is materially missing.
        for need_id in delta.resolve_information_needs:
            for need in session.information_needs:
                if need.need_id == need_id and need.status is NeedStatus.OPEN:
                    need.status = NeedStatus.RESOLVED
                    need.closed_turn_ref = turn_ref
                    if "resolve_information_needs" not in accepted:
                        accepted.append("resolve_information_needs")
        if session.spending_constraint() is not None:
            for need in session.information_needs:
                if (
                    need.topic is InformationTopic.SPENDING_LIMIT
                    and need.status is NeedStatus.OPEN
                ):
                    need.status = NeedStatus.RESOLVED
                    need.closed_turn_ref = turn_ref
        if delta.open_information_need is not None:
            need = delta.open_information_need
            reason = _unsafe_intent(need.why_material)
            if not reason and need.topic is InformationTopic.SPENDING_LIMIT:
                if session.spending_constraint() is not None:
                    reason = "the fan's spending limit is already known; do not ask again"
                elif application_spending_limits(snapshot):
                    reason = (
                        "the application already holds the fan's explicit spending "
                        "limit; do not ask again"
                    )
            if not reason and any(
                open_need.topic is need.topic and open_need.status is NeedStatus.OPEN
                for open_need in session.information_needs
            ):
                reason = "that information need is already open"
            if reason:
                rejected["open_information_need"] = reason
            else:
                session.information_needs.append(
                    InformationNeed(
                        need_id=f"need-{need.topic.value}-{len(session.information_needs) + 1}",
                        topic=need.topic,
                        why_material=_plain(need.why_material, 300),
                        opened_turn_ref=turn_ref,
                    )
                )
                session.information_needs = session.information_needs[
                    -MAX_INFORMATION_NEEDS:
                ]
                accepted.append("open_information_need")

        live = session.status in LIVE_STATUSES
        consumed = session.consumed_refs()
        taken = {beat.beat_id for beat in session.completed_beats}
        if session.current_beat is not None:
            taken.add(session.current_beat.beat_id)

        # 5. Complete the current beat (append-only history).
        if delta.complete_current_beat is not None:
            outcome = delta.complete_current_beat.outcome
            reason = ""
            if session.current_beat is None:
                reason = "there is no current beat to complete"
            elif _TRANSACTION_CLAIM.search(outcome):
                reason = (
                    "a beat outcome cannot claim payment or delivery; the ledger "
                    "records those"
                )
            else:
                reason = _unsafe_interpretation(outcome)
            if reason:
                rejected["complete_current_beat"] = reason
            else:
                _complete(
                    session,
                    session.current_beat,
                    outcome=outcome or "completed",
                    turn_ref=turn_ref,
                    source=BeatSource.MODEL,
                )
                session.current_beat = None
                accepted.append("complete_current_beat")

        # 6. Replace FUTURE beats. Completed beats are untouchable.
        if delta.trajectory is not None:
            trajectory = delta.trajectory
            reason = ""
            if not live:
                reason = "a trajectory needs a proposed, planning, active or paused session"
            elif session.tentative_trajectory and not _plain(trajectory.reason, 300):
                reason = "replacing an existing trajectory needs a replan reason"
            elif trajectory.reason:
                reason = _unsafe_intent(trajectory.reason)
            if reason:
                rejected["trajectory"] = reason
            else:
                new_beats: list[PlannedBeat] = []
                for index, proposal in enumerate(trajectory.beats):
                    key = f"trajectory.beats[{index}]"
                    beat_reason, content_ref = _validate_beat(
                        proposal, candidates=candidates, consumed=consumed
                    )
                    if not beat_reason and len(new_beats) >= MAX_TRAJECTORY_BEATS:
                        beat_reason = "trajectory capacity reached"
                    if beat_reason:
                        rejected[key] = beat_reason
                        continue
                    beat_id = _beat_id(proposal, taken=taken, index=index, turn=turn_ref)
                    taken.add(beat_id)
                    new_beats.append(
                        PlannedBeat(
                            beat_id=beat_id,
                            kind=proposal.kind,
                            intent=_plain(proposal.intent, 300),
                            content_ref=content_ref,
                            media_role=_plain(proposal.media_role, 300),
                            planned_turn_ref=turn_ref,
                        )
                    )
                replaced = [beat.beat_id for beat in session.tentative_trajectory]
                if replaced:
                    reason_text = _plain(trajectory.reason, 300)
                    session.replans.append(
                        ReplanRecord(
                            turn_ref=turn_ref,
                            reason=reason_text,
                            replaced_beat_ids=replaced[:12],
                        )
                    )
                    session.replans = session.replans[-MAX_REPLANS:]
                    session.last_replan_reason = reason_text
                elif trajectory.reason:
                    session.last_replan_reason = _plain(trajectory.reason, 300)
                session.tentative_trajectory = new_beats
                accepted.append("trajectory")

        # 7. Start a beat: a planned one by id, or an ad hoc one.
        if delta.start_beat is not None:
            proposal = delta.start_beat
            reason = ""
            if not live:
                reason = "a beat needs a live session"
            elif (
                session.current_beat is not None
                and session.current_beat.beat_id != proposal.beat_id
            ):
                reason = "complete the current beat before starting another"
            if reason:
                rejected["start_beat"] = reason
            else:
                planned = next(
                    (
                        beat
                        for beat in session.tentative_trajectory
                        if proposal.beat_id and beat.beat_id == proposal.beat_id
                    ),
                    None,
                )
                if planned is not None:
                    session.tentative_trajectory = [
                        beat
                        for beat in session.tentative_trajectory
                        if beat.beat_id != planned.beat_id
                    ]
                    planned.started_turn_ref = turn_ref
                    session.current_beat = planned
                    accepted.append("start_beat")
                elif (
                    session.current_beat is not None
                    and session.current_beat.beat_id == proposal.beat_id
                ):
                    accepted.append("start_beat")
                else:
                    beat_reason, content_ref = _validate_beat(
                        proposal, candidates=candidates, consumed=consumed
                    )
                    if beat_reason:
                        rejected["start_beat"] = beat_reason
                    else:
                        beat_id = _beat_id(proposal, taken=taken, index=99, turn=turn_ref)
                        session.current_beat = PlannedBeat(
                            beat_id=beat_id,
                            kind=proposal.kind,
                            intent=_plain(proposal.intent, 300),
                            content_ref=content_ref,
                            media_role=_plain(proposal.media_role, 300),
                            planned_turn_ref=turn_ref,
                            started_turn_ref=turn_ref,
                        )
                        accepted.append("start_beat")

        if delta.tempo is not None:
            session.tempo = delta.tempo
            accepted.append("tempo")

    if next_experience_move is not None:
        reason = _unsafe_intent(next_experience_move.intent)
        if reason:
            rejected["next_experience_move"] = reason
        elif next_experience_move != session.next_experience_move:
            session.next_experience_move = next_experience_move
            accepted.append("next_experience_move")

    # Per-turn bookkeeping. Counting a turn is itself a durable fact, so a
    # session that talks for five turns records five turns without any delta.
    if session.status in LIVE_STATUSES:
        session.turns_in_session += 1
    consumed_now = (
        session.last_event.observed_turn_ref == turn_ref
        and session.last_event.kind in {lc.value for lc in CONSUMED_LIFECYCLES}
    )
    if not consumed_now:
        session.turns_since_content_event += 1

    changed = bool(accepted) or bool(reconciled)
    prior = state.revision if base_revision is None else base_revision
    after.revision = prior + 1
    if not changed and after.session == state.session and after.working == state.working:
        after.revision = prior
    after = ConversationalSessionState.model_validate(after.model_dump(mode="json"))
    return SessionDeltaValidation(
        proposed=proposed,
        accepted_fields=accepted,
        rejected_fields=rejected,
        reconciled=list(reconciled or []),
        state_after=after,
    )


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def _handle_for(content_ref: str, candidates: dict[str, Offer]) -> str:
    for handle, offer in candidates.items():
        if str(offer.set_id) == content_ref:
            return handle
    return ""


def owner_session_view(
    state: ConversationalSessionState,
    *,
    snapshot: EvidenceSnapshot,
    candidates: dict[str, Offer],
) -> dict[str, Any]:
    """What GLM may know about the longer interaction. Bounded, opaque, no prices."""
    session = state.session
    consumed = session.consumed_refs()

    def _beat(beat: PlannedBeat) -> dict[str, Any]:
        row: dict[str, Any] = {
            "beat_id": beat.beat_id,
            "kind": beat.kind.value,
            "intent": beat.intent,
            "status": "planned_not_promised",
        }
        if beat.content_ref:
            handle = _handle_for(beat.content_ref, candidates)
            row["content"] = handle or (
                "already_used" if beat.content_ref in consumed else "not_currently_eligible"
            )
            row["media_role"] = beat.media_role
        return row

    used_labels: dict[str, str] = {}
    ledger: list[dict[str, Any]] = []
    for use in session.used_content:
        label = _handle_for(use.content_ref, candidates) or used_labels.setdefault(
            use.content_ref, f"earlier_item_{len(used_labels) + 1}"
        )
        ledger.append({"content": label, "fact": use.lifecycle.value})
    planned_handles = {
        _handle_for(beat.content_ref, candidates): beat.beat_id
        for beat in [*session.tentative_trajectory, *(filter(None, [session.current_beat]))]
        if beat.content_ref
    }
    remaining = remaining_spending_cents(state)
    return {
        "schema_version": state.schema_version,
        "revision": state.revision,
        "status": session.status.value,
        "session_id": session.session_id,
        "turns_in_session": session.turns_in_session,
        "turns_since_content_event": session.turns_since_content_event,
        "experience_premise": {
            "summary": session.experience_premise.summary,
            "world_scope": session.experience_premise.world_scope.value,
        },
        "interaction_goal": session.interaction_goal,
        "fan_participation": session.fan_participation.model_dump(mode="json"),
        "known_constraints": [
            {
                "constraint_id": c.constraint_id,
                "kind": c.kind.value,
                "statement": c.statement,
                "source_type": c.source_type.value,
            }
            for c in session.active_constraints()
        ],
        "spending_limit_known": bool(
            session.spending_constraint() or application_spending_limits(snapshot)
        ),
        "spending_limit_reached": remaining is not None and remaining <= 0,
        "open_information_needs": [
            {"need_id": n.need_id, "topic": n.topic.value, "why_material": n.why_material}
            for n in session.information_needs
            if n.status is NeedStatus.OPEN
        ],
        "current_beat": _beat(session.current_beat) if session.current_beat else None,
        "tentative_trajectory": [_beat(beat) for beat in session.tentative_trajectory],
        "recent_completed_beats": [
            {
                "beat_id": beat.beat_id,
                "kind": beat.kind.value,
                "outcome": beat.outcome,
                "source": beat.source.value,
            }
            for beat in session.completed_beats[-8:]
        ],
        "next_experience_move_last_turn": session.next_experience_move.model_dump(
            mode="json"
        ),
        "tempo": session.tempo.value,
        "content_direction": session.content_direction,
        "last_content_event": {
            "fact": session.last_event.kind,
            "happened_this_turn": bool(
                session.last_event.kind != "none"
                and session.last_event.observed_turn_ref == snapshot.trigger.identity
            ),
        },
        "last_replan_reason": session.last_replan_reason,
        "content_candidates": [
            {
                "candidate_handle": handle,
                "asset_type": offer.asset_type,
                "description": _plain(offer.legal_description or offer.label, 300),
                "planned_in_beat": planned_handles.get(handle, ""),
                # Ledger facts only: a pending offer reads "presented", never
                # "not offered", and planning never changes this value.
                "fact": (
                    lifecycle.value
                    if (lifecycle := session.lifecycle_of(str(offer.set_id))) is not None
                    else "eligible_candidate_not_offered"
                ),
            }
            for handle, offer in candidates.items()
        ],
        "content_ledger": ledger[-12:],
        "previous_sessions": len(state.previous_sessions),
    }


def writer_session_view(
    state: ConversationalSessionState, *, snapshot: EvidenceSnapshot, operation: str
) -> dict[str, Any]:
    """What Kimi needs to stay inside one continuing interaction.

    Future content is deliberately absent: the writer must never pre-announce,
    count, or tease inventory the owner merely planned.
    """
    session = state.session
    premise = session.experience_premise
    event = session.last_event
    return {
        "status": session.status.value,
        "experience_premise": premise.summary,
        "premise_scope": (
            "imagined — develop it as a shared imagined scene; it is never a "
            "real-world claim about the creator"
            if premise.world_scope is WorldScope.IMAGINED_SCENE
            else "conversation"
        ),
        "interaction_goal": session.interaction_goal,
        "current_beat": (
            {"kind": session.current_beat.kind.value, "intent": session.current_beat.intent}
            if session.current_beat
            else None
        ),
        "recently_completed": [
            {"kind": beat.kind.value, "outcome": beat.outcome}
            for beat in session.completed_beats[-3:]
        ],
        "next_experience_move": session.next_experience_move.model_dump(mode="json"),
        "tempo": session.tempo.value,
        "fan_participation": session.fan_participation.model_dump(mode="json"),
        "what_just_happened": (
            f"the ledger confirms content was {event.kind} since the last turn"
            if event.kind in {"sent", "purchased"}
            and event.observed_turn_ref == snapshot.trigger.identity
            else "nothing new in the ledger"
        ),
        "turns_since_content_event": session.turns_since_content_event,
        "operation_this_turn": operation,
        "trajectory_note": (
            "The interaction has a provisional direction that is NOT a promise. "
            "Never announce, count, or tease future content."
        ),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


async def load_session_state(
    creator_id: str, fan_id: str, *, db: Any = None
) -> ConversationalSessionState:
    def _read() -> dict[str, Any] | None:
        result = (
            (db or get_supabase())
            .table(SESSION_TABLE)
            .select("schema_version, revision, state")
            .eq("creator_id", str(creator_id))
            .eq("fan_id", str(fan_id))
            .limit(1)
            .execute()
        )
        rows = (
            result.data
            if isinstance(result.data, list)
            else ([result.data] if result.data else [])
        )
        return rows[0] if rows else None

    try:
        row = await asyncio.to_thread(_read)
    except Exception as exc:
        raise CoreStateError(
            "Core v2 session state could not be loaded; apply "
            "db/conversational_session_state_v2.sql"
        ) from exc
    if not row:
        return empty_session_state()
    try:
        payload = dict(row.get("state") or {})
        payload["schema_version"] = row.get("schema_version") or payload.get(
            "schema_version"
        )
        payload["revision"] = int(
            row.get("revision")
            if row.get("revision") is not None
            else payload.get("revision", 0)
        )
        return ConversationalSessionState.model_validate(payload)
    except (TypeError, ValueError, ValidationError) as exc:
        raise CoreStateCorruptionError(
            "Core v2 session state is invalid and will not be routed through "
            "another runtime"
        ) from exc


async def save_session_state(
    creator_id: str,
    fan_id: str,
    *,
    expected_revision: int,
    state: ConversationalSessionState,
    db: Any = None,
) -> ConversationalSessionState:
    if state.revision == expected_revision:
        return state
    payload = {
        "creator_id": str(creator_id),
        "fan_id": str(fan_id),
        "schema_version": SESSION_STATE_SCHEMA_VERSION,
        "revision": state.revision,
        "state": state.as_dict(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    def _write() -> bool:
        client = db or get_supabase()
        if expected_revision == 0:
            existing = (
                client.table(SESSION_TABLE)
                .select("revision")
                .eq("creator_id", str(creator_id))
                .eq("fan_id", str(fan_id))
                .limit(1)
                .execute()
            )
            if existing.data:
                return False
            result = client.table(SESSION_TABLE).insert(payload).execute()
            return bool(result.data)
        result = (
            client.table(SESSION_TABLE)
            .update(payload)
            .eq("creator_id", str(creator_id))
            .eq("fan_id", str(fan_id))
            .eq("revision", expected_revision)
            .execute()
        )
        return bool(result.data)

    try:
        saved = await asyncio.to_thread(_write)
    except Exception as exc:
        if "duplicate" in str(exc).lower() or "23505" in str(exc):
            raise CoreStateConflictError(
                "Core v2 session state changed during this turn"
            ) from exc
        raise CoreStateError("Core v2 session state could not be persisted") from exc
    if not saved:
        raise CoreStateConflictError("Core v2 session state changed during this turn")
    return state
