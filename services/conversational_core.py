"""Persistence and deterministic delta validation for Conversational Core v1."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from core.supabase import get_supabase
from models.conversational_core import (
    CORE_STATE_SCHEMA_VERSION,
    ConversationalWorkingState,
    ElementStatus,
    EpistemicType,
    EstablishedElement,
    StateDeltaValidation,
    WorkingStateDelta,
    WorldScope,
)
from models.live_orchestration import EvidenceSnapshot

STATE_TABLE = "conversational_core_states"


class CoreStateError(RuntimeError):
    """A selected Core v1 turn could not safely load or persist its own state."""


class CoreStateCorruptionError(CoreStateError):
    """The durable row exists but does not satisfy the versioned contract."""


class CoreStateConflictError(CoreStateError):
    """Another turn changed the state after this owner read it."""


@dataclass(frozen=True)
class EvidenceAuthority:
    source_type: EpistemicType
    authoritative_transaction: bool = False
    completed_transaction: bool = False


_TRANSACTION_CLAIM = re.compile(
    r"\b(?:paid|payment|purchased?|bought|delivered|sent|attached|unlocked|"
    r"price|cost|charged|refunded)\b|[$€£]\s*\d",
    re.IGNORECASE,
)
_TRANSACTION_COMPLETION_CLAIM = re.compile(
    r"\b(?:paid|payment (?:completed|confirmed|received)|purchased?|bought|"
    r"delivered|sent|attached|unlocked|charged|refunded)\b",
    re.IGNORECASE,
)
_PRESENT_WORLD_CLAIM = re.compile(
    r"\b(?:the creator|she|i(?:['’]m| am))\s+(?:is\s+|currently\s+|right now\s+|"
    r"just\s+)?(?:wearing|sitting|lying|cooking|driving|working|shopping|"
    r"showering|heading|at home|at work|in bed)\b",
    re.IGNORECASE,
)


def _plain(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def empty_working_state() -> ConversationalWorkingState:
    return ConversationalWorkingState()


def state_fingerprint(state: ConversationalWorkingState) -> str:
    encoded = json.dumps(state.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def evidence_catalog(snapshot: EvidenceSnapshot) -> dict[str, EvidenceAuthority]:
    """Exact references the owner is allowed to cite in a delta.

    The working state is deliberately absent. A prior model-authored state can
    help interpretation but can never become proof that an event occurred.
    """
    catalog: dict[str, EvidenceAuthority] = {}
    trigger_ref = str(snapshot.trigger.identity or "").strip()
    if trigger_ref:
        catalog[trigger_ref] = EvidenceAuthority(EpistemicType.EXPLICIT_FAN_STATEMENT)
    for fact in snapshot.creator_facts:
        if fact.source_ref:
            catalog[str(fact.source_ref)] = EvidenceAuthority(
                EpistemicType.CREATOR_CONFIG
            )
    for fact in (*snapshot.historical_facts, *snapshot.conversation_episodes):
        if fact.source_ref:
            catalog[str(fact.source_ref)] = EvidenceAuthority(
                EpistemicType.MODEL_INFERENCE
            )
    for row in snapshot.confirmed_purchases:
        ref = str(row.get("reference") or row.get("payment_reference") or "").strip()
        if ref:
            catalog[ref] = EvidenceAuthority(EpistemicType.TRANSACTION_FACT, True, True)
    for row in snapshot.confirmed_deliveries:
        ref = str(
            row.get("platform_message_id")
            or row.get("reference")
            or row.get("payment_reference")
            or ""
        ).strip()
        if ref:
            catalog[ref] = EvidenceAuthority(EpistemicType.TRANSACTION_FACT, True, True)
    pending = snapshot.pending_payment or {}
    pending_ref = str(pending.get("reference") or "").strip()
    if pending_ref:
        # Pending is authoritative as a pending record, never as proof of payment.
        catalog[pending_ref] = EvidenceAuthority(
            EpistemicType.TRANSACTION_FACT, True, False
        )
    return catalog


def evidence_catalog_view(snapshot: EvidenceSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "source_ref": ref,
            "allowed_source_type": authority.source_type.value,
            "authoritative_transaction": authority.authoritative_transaction,
            "completed_transaction": authority.completed_transaction,
        }
        for ref, authority in evidence_catalog(snapshot).items()
    ]


def _parse_delta(raw: Any) -> tuple[WorkingStateDelta | None, dict[str, str]]:
    if raw in (None, ""):
        return WorkingStateDelta(), {}
    if not isinstance(raw, dict):
        return None, {"state_delta": "state_delta must be an object"}
    try:
        return WorkingStateDelta.model_validate(raw), {}
    except ValidationError as exc:
        rejected: dict[str, str] = {}
        # Reject malformed optional fields locally, then salvage the rest.
        bad_roots = {str(error["loc"][0]) for error in exc.errors() if error.get("loc")}
        for root in sorted(bad_roots):
            rejected[root] = "malformed optional state field"
        salvage = {key: value for key, value in raw.items() if key not in bad_roots}
        try:
            return WorkingStateDelta.model_validate(salvage), rejected
        except ValidationError:
            return None, {
                "state_delta": "state_delta could not be safely parsed",
                **rejected,
            }


def _validate_element(
    element: Any,
    *,
    catalog: dict[str, EvidenceAuthority],
) -> str:
    if not _plain(element.claim, 500):
        return "an established element needs a non-empty claim"
    refs = list(element.source_refs or [])
    if not refs:
        return "an established element needs an evidence reference"
    unknown = [ref for ref in refs if ref not in catalog]
    if unknown:
        return "unknown evidence reference: " + ", ".join(unknown[:3])

    authorities = [catalog[ref] for ref in refs]
    source_type = element.source_type
    if source_type is EpistemicType.TRANSACTION_FACT:
        if not all(item.authoritative_transaction for item in authorities):
            return "transaction facts require authoritative transaction evidence"
        if _TRANSACTION_COMPLETION_CLAIM.search(element.claim) and not all(
            item.completed_transaction for item in authorities
        ):
            return "pending transaction evidence cannot establish completion"
    elif source_type is EpistemicType.CREATOR_CONFIG:
        if not all(
            item.source_type is EpistemicType.CREATOR_CONFIG for item in authorities
        ):
            return "creator_config may cite only creator configuration"
    elif source_type is EpistemicType.EXPLICIT_FAN_STATEMENT:
        if not all(
            item.source_type is EpistemicType.EXPLICIT_FAN_STATEMENT
            for item in authorities
        ):
            return "explicit fan statements require raw fan-message evidence"
    elif source_type in {EpistemicType.SHARED_IMAGINED, EpistemicType.SCENE_ASSUMPTION}:
        if not all(
            item.source_type
            in {EpistemicType.EXPLICIT_FAN_STATEMENT, EpistemicType.MODEL_INFERENCE}
            for item in authorities
        ):
            return "imagined or assumed scene elements need conversational evidence"

    if element.world_scope is WorldScope.PRESENT_WORLD:
        return "model-authored state cannot establish a present-world creator fact"
    if (
        element.world_scope is WorldScope.TRANSACTION
        and source_type is not EpistemicType.TRANSACTION_FACT
    ):
        return "transaction scope requires transaction_fact"
    if (
        source_type is EpistemicType.SHARED_IMAGINED
        and element.world_scope is not WorldScope.IMAGINED_SCENE
    ):
        return "shared_imagined must remain in imagined_scene scope"
    if (
        _TRANSACTION_CLAIM.search(element.claim)
        and source_type is not EpistemicType.TRANSACTION_FACT
    ):
        return "payment, purchase, price, and delivery claims require transaction_fact"
    return ""


def _unsafe_interpretation(value: Any) -> str:
    text = _plain(value, 1_000)
    if _TRANSACTION_CLAIM.search(text):
        return "interpretive state cannot assert payment, purchase, price, or delivery"
    if _PRESENT_WORLD_CLAIM.search(text) and not re.search(
        r"\b(?:imagin(?:e|ed|ary)|pretend|roleplay|daydream|shared scene)\b",
        text,
        re.IGNORECASE,
    ):
        return "interpretive state cannot assert unsupported present-world activity"
    return ""


def validate_and_apply_delta(
    state: ConversationalWorkingState,
    raw_delta: Any,
    *,
    snapshot: EvidenceSnapshot,
    known_thread_ids: set[str] | None = None,
) -> StateDeltaValidation:
    """Apply safe fields and report every field that was refused.

    Invalid optional fields never turn into a handoff. The accepted state is a
    fresh object; callers can persist it only after their normal stale-turn check.
    """
    proposed = copy.deepcopy(raw_delta) if isinstance(raw_delta, dict) else {}
    delta, rejected = _parse_delta(raw_delta)
    if delta is None:
        return StateDeltaValidation(
            proposed=proposed,
            rejected_fields=rejected,
            state_after=state.model_copy(deep=True),
        )

    after = state.model_copy(deep=True)
    accepted: list[str] = []
    catalog = evidence_catalog(snapshot)

    scalar_scene = {
        "scene_summary": "summary",
        "current_action_focus": "current_action_focus",
        "current_direction": "current_direction",
        "pacing": "pacing",
    }
    for delta_name, state_name in scalar_scene.items():
        value = getattr(delta, delta_name)
        if value is not None:
            if delta_name != "pacing":
                reason = _unsafe_interpretation(value)
                if reason:
                    rejected[delta_name] = reason
                    continue
            setattr(after.active_scene, state_name, value)
            accepted.append(delta_name)
    scalar_flow = {
        "initiative_holder": "initiative_holder",
        "current_focus": "current_focus",
        "participation_gist": "participation_gist",
    }
    for delta_name, state_name in scalar_flow.items():
        value = getattr(delta, delta_name)
        if value is not None:
            if delta_name != "initiative_holder":
                reason = _unsafe_interpretation(value)
                if reason:
                    rejected[delta_name] = reason
                    continue
            setattr(after.flow, state_name, value)
            accepted.append(delta_name)

    existing_ids = {
        element.element_id for element in after.active_scene.established_elements
    }
    for index, element in enumerate(delta.add_established_elements):
        key = f"add_established_elements[{index}]"
        reason = _validate_element(element, catalog=catalog)
        if element.element_id in existing_ids:
            reason = reason or "element_id already exists; state history is immutable"
        if len(after.active_scene.established_elements) >= 40:
            reason = reason or "established element capacity reached"
        if reason:
            rejected[key] = reason
            continue
        after.active_scene.established_elements.append(
            EstablishedElement(
                **element.model_dump(mode="json"),
                introduced_turn_ref=str(snapshot.trigger.identity or ""),
            )
        )
        existing_ids.add(element.element_id)
        accepted.append(key)

    by_id = {
        element.element_id: element
        for element in after.active_scene.established_elements
    }
    for index, correction in enumerate(delta.corrections):
        key = f"corrections[{index}]"
        old = by_id.get(correction.replaces_element_id)
        reason = ""
        if old is None or old.status is not ElementStatus.ACTIVE:
            reason = "correction target is not an active established element"
        elif (
            correction.replacement.source_type
            is not EpistemicType.EXPLICIT_FAN_STATEMENT
        ):
            reason = "only an explicit fan correction may supersede an element"
        else:
            reason = _validate_element(correction.replacement, catalog=catalog)
        if correction.replacement.element_id in existing_ids:
            reason = reason or "replacement element_id already exists"
        if len(after.active_scene.established_elements) >= 40:
            reason = reason or "established element capacity reached"
        if reason:
            rejected[key] = reason
            continue
        replacement = EstablishedElement(
            **correction.replacement.model_dump(mode="json"),
            introduced_turn_ref=str(snapshot.trigger.identity or ""),
        )
        old.status = ElementStatus.SUPERSEDED
        old.superseded_by = replacement.element_id
        after.active_scene.established_elements.append(replacement)
        by_id[replacement.element_id] = replacement
        existing_ids.add(replacement.element_id)
        accepted.append(key)

    unresolved = list(
        dict.fromkeys(
            [
                *after.active_scene.unresolved_possibilities,
                *after.flow.unresolved_possibilities,
            ]
        )
    )
    for possibility in delta.resolve_unresolved_possibilities:
        if possibility in unresolved:
            unresolved.remove(possibility)
    accepted_possibilities = False
    for index, possibility in enumerate(delta.add_unresolved_possibilities):
        cleaned = _plain(possibility, 300)
        reason = _unsafe_interpretation(cleaned)
        if reason:
            rejected[f"add_unresolved_possibilities[{index}]"] = reason
            continue
        if cleaned and cleaned not in unresolved:
            unresolved.append(cleaned)
            accepted_possibilities = True
    unresolved = unresolved[:12]
    after.active_scene.unresolved_possibilities = unresolved
    after.flow.unresolved_possibilities = list(unresolved)
    if accepted_possibilities:
        accepted.append("add_unresolved_possibilities")
    if delta.resolve_unresolved_possibilities:
        accepted.append("resolve_unresolved_possibilities")

    if delta.active_thread_ids is not None:
        known = known_thread_ids or set()
        valid = [
            thread_id for thread_id in delta.active_thread_ids if thread_id in known
        ]
        invalid = [
            thread_id for thread_id in delta.active_thread_ids if thread_id not in known
        ]
        after.flow.active_thread_ids = list(dict.fromkeys(valid))[:12]
        if valid or not delta.active_thread_ids:
            accepted.append("active_thread_ids")
        if invalid:
            rejected["active_thread_ids"] = (
                "unknown continuity thread ids: " + ", ".join(invalid[:3])
            )

    if delta.has_shared_imagined_scene is not None:
        has_supported_imagined = any(
            element.status is ElementStatus.ACTIVE
            and element.source_type is EpistemicType.SHARED_IMAGINED
            and element.world_scope is WorldScope.IMAGINED_SCENE
            for element in after.active_scene.established_elements
        )
        if delta.has_shared_imagined_scene and not has_supported_imagined:
            rejected["has_shared_imagined_scene"] = (
                "shared scene requires a supported shared_imagined element"
            )
        else:
            after.active_scene.has_shared_imagined_scene = (
                delta.has_shared_imagined_scene
            )
            accepted.append("has_shared_imagined_scene")

    after.revision = state.revision + (1 if accepted else 0)
    # Defensive reconstruction makes it impossible for assignment above to
    # leave a value outside the persisted schema.
    after = ConversationalWorkingState.model_validate(after.model_dump(mode="json"))
    return StateDeltaValidation(
        proposed=proposed,
        accepted_fields=accepted,
        rejected_fields=rejected,
        state_after=after,
    )


async def load_working_state(
    creator_id: str,
    fan_id: str,
    *,
    db: Any = None,
) -> ConversationalWorkingState:
    def _read() -> dict[str, Any] | None:
        result = (
            (db or get_supabase())
            .table(STATE_TABLE)
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
            "Core v1 working state could not be loaded; apply db/conversational_working_state_v1.sql"
        ) from exc
    if not row:
        return empty_working_state()
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
        return ConversationalWorkingState.model_validate(payload)
    except (TypeError, ValueError, ValidationError) as exc:
        raise CoreStateCorruptionError(
            "Core v1 working state is invalid and will not be routed through another runtime"
        ) from exc


async def save_working_state(
    creator_id: str,
    fan_id: str,
    *,
    expected_revision: int,
    state: ConversationalWorkingState,
    db: Any = None,
) -> ConversationalWorkingState:
    if state.revision == expected_revision:
        return state
    payload = {
        "creator_id": str(creator_id),
        "fan_id": str(fan_id),
        "schema_version": CORE_STATE_SCHEMA_VERSION,
        "revision": state.revision,
        "state": state.as_dict(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    def _write() -> bool:
        client = db or get_supabase()
        if expected_revision == 0:
            existing = (
                client.table(STATE_TABLE)
                .select("revision")
                .eq("creator_id", str(creator_id))
                .eq("fan_id", str(fan_id))
                .limit(1)
                .execute()
            )
            if existing.data:
                return False
            result = client.table(STATE_TABLE).insert(payload).execute()
            return bool(result.data)
        result = (
            client.table(STATE_TABLE)
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
                "Core v1 working state changed during this turn"
            ) from exc
        raise CoreStateError("Core v1 working state could not be persisted") from exc
    if not saved:
        raise CoreStateConflictError("Core v1 working state changed during this turn")
    return state
