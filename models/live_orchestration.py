"""Typed, versioned evidence and execution records for the selected live core."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

EVIDENCE_VERSION = "live_evidence_v1"


@dataclass(frozen=True)
class EvidenceFact:
    value: str
    source_ref: str
    certainty: str = "stated"


@dataclass(frozen=True)
class TurnTrigger:
    kind: str
    identity: str
    latest_message: str = ""
    scheduled_goal: str = ""


@dataclass(frozen=True)
class EvidenceSnapshot:
    """Everything either model may know about one turn.

    Customer text is data inside this object, never concatenated into the
    system policy.  ``truncation`` names every bounded section that lost data.
    Transaction evidence and the newest customer message are assembled before
    optional memory and therefore are never displaced by historical text.
    """

    creator_id: str
    fan_id: str
    trigger: TurnTrigger
    state_revision: str
    creator_facts: tuple[EvidenceFact, ...] = ()
    creator_voice: dict[str, Any] = field(default_factory=dict)
    recent_turns: tuple[dict[str, Any], ...] = ()
    historical_facts: tuple[EvidenceFact, ...] = ()
    unresolved_obligations: tuple[dict[str, Any], ...] = ()
    corrections: tuple[dict[str, Any], ...] = ()
    approved_inventory: tuple[dict[str, Any], ...] = ()
    pending_offer: dict[str, Any] | None = None
    confirmed_purchases: tuple[dict[str, Any], ...] = ()
    confirmed_deliveries: tuple[dict[str, Any], ...] = ()
    pending_payment: dict[str, Any] | None = None
    spending_limits: dict[str, Any] = field(default_factory=dict)
    operator_constraints: dict[str, Any] = field(default_factory=dict)
    memory_status: dict[str, Any] = field(default_factory=dict)
    truncation: dict[str, int] = field(default_factory=dict)
    version: str = EVIDENCE_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class ValidationResult:
    approved: bool
    operation: str
    record_refs: dict[str, str] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    state_revision: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ApprovedExecution:
    """Deterministic facts the writer and executor may rely on."""

    operation: str = "none"
    offer: dict[str, Any] | None = None
    delivery: dict[str, Any] | None = None
    payment_reference: str = ""
    approval_required: bool = False
    validation: ValidationResult = field(
        default_factory=lambda: ValidationResult(True, "none")
    )

    def writer_view(self) -> dict[str, Any]:
        delivery = dict(self.delivery or {})
        # Media identifiers are executor inputs, not language-model evidence.
        delivery.pop("media_ids", None)
        return {
            "operation": self.operation,
            "offer": self.offer,
            "delivery": delivery or None,
            "payment_reference": self.payment_reference or None,
            "approval_required": self.approval_required,
        }
