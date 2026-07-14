"""Passive, evidence-backed fan intelligence.

Failures are best-effort and must never block a reply. The model only proposes
observations; validation and merge decisions are deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from ai.model_providers import complete, get_runtime_target
from db.fan_intelligence_queries import (
    get_facts_for_key,
    insert_fact,
    insert_observation,
    mark_facts_contradicted,
    update_fact,
)
from models.fan_intelligence import (
    ExtractionEnvelope,
    FactCategory,
    FactCertainty,
    FactStatus,
    MergeAction,
    MergePlan,
    ProposedObservation,
    ValidatedObservation,
)
from models.model_runtime import ModelTelemetryContext
from models.schemas import Message
from services.model_telemetry import record_model_failure, record_model_result


_ALLOWED_KEYS: dict[str, FactCategory] = {
    "preferred_name": FactCategory.IDENTITY,
    "age": FactCategory.IDENTITY,
    "location": FactCategory.IDENTITY,
    "timezone": FactCategory.IDENTITY,
    "occupation": FactCategory.IDENTITY,
    "relationship_status": FactCategory.IDENTITY,
    "usual_availability": FactCategory.AVAILABILITY,
    "weekday_availability": FactCategory.AVAILABILITY,
    "weekend_availability": FactCategory.AVAILABILITY,
    "payday": FactCategory.COMMERCIAL,
    "content_interest": FactCategory.PREFERENCE,
    "disliked_content": FactCategory.PREFERENCE,
    "kink_interest": FactCategory.PREFERENCE,
    "preferred_tone": FactCategory.PREFERENCE,
    "preferred_dynamic": FactCategory.PREFERENCE,
    "preferred_format": FactCategory.PREFERENCE,
    "hard_limit": FactCategory.BOUNDARY,
    "stated_budget_cents": FactCategory.COMMERCIAL,
    "accepted_price_cents": FactCategory.COMMERCIAL,
    "rejected_price_cents": FactCategory.COMMERCIAL,
    "counteroffer_cents": FactCategory.COMMERCIAL,
    "price_sensitivity": FactCategory.COMMERCIAL,
    "purchase_intent": FactCategory.COMMERCIAL,
    "objection_pattern": FactCategory.BEHAVIOR,
}

_MULTI_VALUE_KEYS = {
    "content_interest",
    "disliked_content",
    "kink_interest",
    "preferred_format",
    "hard_limit",
    "objection_pattern",
}

_EXPLICIT_ONLY_KEYS = {
    "age",
    "payday",
    "hard_limit",
    "stated_budget_cents",
    "accepted_price_cents",
    "rejected_price_cents",
    "counteroffer_cents",
}

_MONEY_KEYS = {
    "stated_budget_cents",
    "accepted_price_cents",
    "rejected_price_cents",
    "counteroffer_cents",
}

_SYSTEM_PROMPT = """You extract durable fan CRM facts from messages on a paid adult creator platform.
Return only valid JSON in this exact shape: {"observations": [...]}.

Only extract a fact newly supported by the LATEST FAN MESSAGE. Use recent context only to resolve pronouns or what a price refers to. Never extract facts about the creator. Never treat roleplay, fantasy, jokes, or one-off dirty talk as real-world personal facts.

Allowed fact_key values:
preferred_name, age, location, timezone, occupation, relationship_status,
usual_availability, weekday_availability, weekend_availability, payday,
content_interest, disliked_content, kink_interest, preferred_tone,
preferred_dynamic, preferred_format, hard_limit, stated_budget_cents,
accepted_price_cents, rejected_price_cents, counteroffer_cents,
price_sensitivity, purchase_intent, objection_pattern.

Rules:
- One atomic fact per observation. Do not return lists inside value.
- certainty is only "explicit" or "strong_inference".
- Use strong_inference rarely. Weak implications must be omitted.
- age, payday, hard limits, and all money facts require explicit wording.
- Money values must be integer cents: $40 -> 4000.
- Do not claim a purchase happened from chat text. Purchases come from payment events.
- evidence must be an exact short quote from the latest fan message.
- confidence must reflect evidence strength, not model confidence.
- Empty is correct when nothing durable was learned.

Observation shape:
{"category":"commercial","fact_key":"payday","value":"Friday","certainty":"explicit","confidence":0.98,"evidence":"I get paid Friday"}
"""


def fan_intelligence_enabled() -> bool:
    return os.getenv("FAN_INTELLIGENCE_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_spaces(value: str) -> str:
    return " ".join(value.split()).strip()


def _evidence_is_exact(evidence: str, fan_message: str) -> bool:
    return _normalize_spaces(evidence).casefold() in _normalize_spaces(fan_message).casefold()


def _money_from_evidence(evidence: str) -> int | None:
    match = re.search(r"(?:\$|usd\s*)(\d+(?:\.\d{1,2})?)", evidence, re.IGNORECASE)
    if not match:
        return None
    return int(round(float(match.group(1)) * 100))


def _normalize_value(observation: ProposedObservation) -> Any | None:
    key = observation.fact_key
    value = observation.value

    if key in _MONEY_KEYS:
        evidence_amount = _money_from_evidence(observation.evidence)
        if evidence_amount is not None:
            return evidence_amount
        if isinstance(value, dict):
            value = value.get("amount_cents")
        try:
            cents = int(value)
        except (TypeError, ValueError):
            return None
        return cents if 0 <= cents <= 10_000_000 else None

    if key == "age":
        try:
            age = int(value)
        except (TypeError, ValueError):
            return None
        return age if 18 <= age <= 120 else None

    if isinstance(value, bool) or value is None:
        return None

    if isinstance(value, (int, float)):
        return value

    if not isinstance(value, str):
        return None

    text = _normalize_spaces(value)
    if not text or len(text) > 300:
        return None
    return text


def parse_extraction_payload(raw_text: str) -> ExtractionEnvelope:
    cleaned = "\n".join(
        line for line in (raw_text or "").splitlines() if not line.lstrip().startswith("```")
    ).strip()
    if not cleaned:
        raise ValueError("empty extraction response")
    payload = json.loads(cleaned)
    return ExtractionEnvelope.model_validate(payload)


def validate_observation(
    proposed: ProposedObservation,
    *,
    fan_message: str,
) -> ValidatedObservation | None:
    expected_category = _ALLOWED_KEYS.get(proposed.fact_key)
    if expected_category is None or proposed.category != expected_category:
        return None
    if proposed.certainty == FactCertainty.STRONG_INFERENCE and proposed.fact_key in _EXPLICIT_ONLY_KEYS:
        return None
    minimum = 0.65 if proposed.certainty == FactCertainty.EXPLICIT else 0.85
    if proposed.confidence < minimum:
        return None
    if not proposed.evidence or not _evidence_is_exact(proposed.evidence, fan_message):
        return None

    value = _normalize_value(proposed)
    if value is None:
        return None

    return ValidatedObservation(
        category=proposed.category,
        fact_key=proposed.fact_key,
        value_json=value,
        normalized_value=_canonical_json(value).casefold(),
        certainty=proposed.certainty,
        confidence=round(float(proposed.confidence), 4),
        evidence_text=_normalize_spaces(proposed.evidence)[:500],
    )


def plan_fact_merge(
    existing_facts: list[dict[str, Any]],
    observation: ValidatedObservation,
) -> MergePlan:
    same_value = next(
        (
            fact
            for fact in existing_facts
            if str(fact.get("normalized_value") or "") == observation.normalized_value
        ),
        None,
    )
    if same_value:
        return MergePlan(
            action=MergeAction.REINFORCE,
            matched_fact_id=str(same_value["id"]),
            reason="same value observed again",
        )

    if observation.fact_key in _MULTI_VALUE_KEYS:
        return MergePlan(
            action=MergeAction.ADD_MULTI_VALUE,
            reason="multi-value fact",
        )

    active = [fact for fact in existing_facts if fact.get("is_active", True)]
    if not active:
        return MergePlan(action=MergeAction.CREATE, reason="no active fact")

    if observation.certainty == FactCertainty.EXPLICIT and all(
        fact.get("status") == FactStatus.INFERRED.value for fact in active
    ):
        return MergePlan(
            action=MergeAction.REPLACE_INFERRED,
            conflicting_fact_ids=[str(fact["id"]) for fact in active],
            reason="explicit evidence supersedes inference",
        )

    return MergePlan(
        action=MergeAction.CONFLICT,
        conflicting_fact_ids=[str(fact["id"]) for fact in active],
        reason="conflicting durable value",
    )


def _initial_status(observation: ValidatedObservation) -> FactStatus:
    if observation.certainty == FactCertainty.EXPLICIT:
        return FactStatus.EXPLICIT
    return FactStatus.INFERRED


def observation_dedupe_key(
    *,
    creator_id: str,
    fan_id: str,
    source_message_id: str | None,
    observation: ValidatedObservation,
) -> str:
    raw = "|".join(
        [
            creator_id,
            fan_id,
            source_message_id or "",
            observation.fact_key,
            observation.normalized_value,
            observation.evidence_text.casefold(),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _merge_one(
    *,
    creator_id: str,
    fan_id: str,
    source_message_id: str | None,
    observation: ValidatedObservation,
    extraction_provider: str,
    extraction_model: str,
) -> None:
    inserted = await insert_observation(
        {
            "creator_id": creator_id,
            "fan_id": fan_id,
            "source_message_id": source_message_id,
            "category": observation.category.value,
            "fact_key": observation.fact_key,
            "observed_value_json": observation.value_json,
            "normalized_value": observation.normalized_value,
            "confidence": observation.confidence,
            "certainty": observation.certainty.value,
            "source_type": observation.source_type,
            "evidence_text": observation.evidence_text,
            "extraction_provider": extraction_provider,
            "extraction_model": extraction_model,
            "dedupe_key": observation_dedupe_key(
                creator_id=creator_id,
                fan_id=fan_id,
                source_message_id=source_message_id,
                observation=observation,
            ),
        }
    )
    if not inserted:
        return

    existing = await get_facts_for_key(fan_id, observation.fact_key)
    plan = plan_fact_merge(existing, observation)

    if plan.action == MergeAction.REINFORCE and plan.matched_fact_id:
        fact = next(f for f in existing if str(f["id"]) == plan.matched_fact_id)
        confirmations = int(fact.get("confirmation_count") or 1) + 1
        prior_status = str(fact.get("status") or FactStatus.INFERRED.value)
        status = prior_status
        if observation.certainty == FactCertainty.EXPLICIT:
            status = FactStatus.CONFIRMED.value if confirmations >= 2 else FactStatus.EXPLICIT.value
        await update_fact(
            plan.matched_fact_id,
            {
                "confidence": max(float(fact.get("confidence") or 0), observation.confidence),
                "status": status,
                "source_type": observation.source_type,
                "last_evidence_message_id": source_message_id,
                "last_evidence_text": observation.evidence_text,
                "confirmation_count": confirmations,
                "is_active": True,
            },
        )
        return

    if plan.action == MergeAction.REPLACE_INFERRED:
        await mark_facts_contradicted(plan.conflicting_fact_ids, deactivate=True)
        await insert_fact(
            creator_id=creator_id,
            fan_id=fan_id,
            observation=observation,
            source_message_id=source_message_id,
            status=FactStatus.EXPLICIT,
            is_active=True,
        )
        return

    if plan.action == MergeAction.CONFLICT:
        await mark_facts_contradicted(plan.conflicting_fact_ids, deactivate=True)
        await insert_fact(
            creator_id=creator_id,
            fan_id=fan_id,
            observation=observation,
            source_message_id=source_message_id,
            status=FactStatus.CONTRADICTED,
            is_active=False,
        )
        return

    if plan.action in {MergeAction.CREATE, MergeAction.ADD_MULTI_VALUE}:
        await insert_fact(
            creator_id=creator_id,
            fan_id=fan_id,
            observation=observation,
            source_message_id=source_message_id,
            status=_initial_status(observation),
            is_active=True,
        )


async def learn_from_fan_message(
    *,
    creator_id: str,
    fan_id: str,
    fan_message: str,
    source_message_id: str | None = None,
    conversation_history: list[Message] | None = None,
) -> None:
    """Extract and merge durable facts without ever blocking reply generation."""

    if not fan_intelligence_enabled() or not (fan_message or "").strip():
        return

    target = get_runtime_target("EXTRACTOR")
    context_lines: list[str] = []
    for message in (conversation_history or [])[-6:]:
        speaker = "Fan" if message.role == "fan" else "Creator"
        content = (message.content or "").strip()
        if content:
            context_lines.append(f"{speaker}: {content}")

    user_prompt = (
        "RECENT CONTEXT (reference only):\n"
        + ("\n".join(context_lines) if context_lines else "[none]")
        + "\n\nLATEST FAN MESSAGE (the only source of new facts):\n"
        + fan_message.strip()
    )
    telemetry = ModelTelemetryContext(
        feature="fan_intelligence_extraction",
        creator_id=creator_id,
        fan_id=fan_id,
        metadata={"source_message_id": source_message_id},
    )

    try:
        result = await complete(
            target,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=int(os.getenv("EXTRACTOR_MAX_TOKENS", "700") or 700),
            temperature=0.0,
        )
        try:
            envelope = parse_extraction_payload(result.text)
        except Exception as exc:
            await record_model_result(
                result,
                telemetry,
                success=False,
                parse_valid=False,
                error=f"invalid extraction JSON: {exc}",
            )
            print(f"[FAN INTELLIGENCE] invalid extraction fan={fan_id}: {exc}")
            return

        validated = [
            observation
            for proposed in envelope.observations
            if (
                observation := validate_observation(
                    proposed,
                    fan_message=fan_message,
                )
            )
            is not None
        ]
        telemetry = ModelTelemetryContext(
            feature="fan_intelligence_extraction",
            creator_id=creator_id,
            fan_id=fan_id,
            metadata={
                "source_message_id": source_message_id,
                "proposed_observations": len(envelope.observations),
                "validated_observations": len(validated),
            },
        )
        await record_model_result(
            result,
            telemetry,
            success=True,
            parse_valid=True,
            error=None,
        )

        for observation in validated:
            try:
                await _merge_one(
                    creator_id=creator_id,
                    fan_id=fan_id,
                    source_message_id=source_message_id,
                    observation=observation,
                    extraction_provider=target.provider,
                    extraction_model=target.model,
                )
            except Exception as exc:
                print(
                    f"[FAN INTELLIGENCE] merge failed fan={fan_id} "
                    f"key={observation.fact_key}: {exc}"
                )
        if validated:
            print(f"[FAN INTELLIGENCE] fan={fan_id} merged={len(validated)}")
    except Exception as exc:
        await record_model_failure(target, telemetry, error=str(exc))
        print(f"[FAN INTELLIGENCE] extraction failed fan={fan_id}: {exc}")
