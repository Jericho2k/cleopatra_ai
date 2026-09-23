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
    latest_fan_burst: tuple[dict[str, Any], ...] = ()
    recent_messages: tuple[dict[str, Any], ...] = ()
    recent_turns: tuple[dict[str, Any], ...] = ()
    historical_facts: tuple[EvidenceFact, ...] = ()
    conversation_episodes: tuple[EvidenceFact, ...] = ()
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
    #: What is known — and, today, what is NOT known — about anything the
    #: creator published. Approved vault inventory is permission to offer
    #: privately; it has never been evidence that a feed post exists. Stating
    #: that in evidence is what stops "check my feed" being invented.
    publication_evidence: dict[str, Any] = field(default_factory=dict)
    #: Feed/post references the FAN supplied. These may be discussed as his
    #: context; they are not promoted to creator publication facts.
    fan_publication_references: tuple[dict[str, Any], ...] = ()
    #: An explicit, fan-created buying opportunity and the messages that show
    #: it. A record of what he SAID — never an inference about what he can
    #: afford, which is the funnel this architecture removed.
    commercial_opportunity: dict[str, Any] = field(default_factory=dict)
    #: Whether the fan claimed to have paid, and whether anything authoritative
    #: agrees. A claim is not a receipt.
    purchase_claim: dict[str, Any] = field(default_factory=dict)
    #: Current fan-reported inability to access a confirmed purchase.
    #: This is evidence for support/repair, never authority to charge again.
    content_access_issue: dict[str, Any] = field(default_factory=dict)
    #: Soft rhythm context for the writer: what the recent creator bubbles have
    #: been leaning on. Never a send-blocking rule.
    voice_rhythm: dict[str, Any] = field(default_factory=dict)
    #: Application-owned statement of which product surface this conversation
    #: inhabits. Models consume it as evidence rather than inferring location or
    #: capabilities from conversational prose.
    platform_context: dict[str, Any] = field(default_factory=dict)
    #: A current fan-authored request to see creator-controlled media. This is
    #: independent of explicitness and willingness to pay, and never forces an
    #: operation; it makes the real media affordances a conscious decision.
    media_request: dict[str, Any] = field(default_factory=dict)
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
        offer = dict(self.offer or {})
        for key in ("offer_id", "set_id", "label", "media_count", "record_kind"):
            offer.pop(key, None)
        delivery = dict(self.delivery or {})
        # Media identifiers are executor inputs, not language-model evidence.
        for key in ("media_ids", "set_id", "offer_id", "step_index"):
            delivery.pop(key, None)
        return {
            "operation": self.operation,
            "offer": offer or None,
            "delivery": delivery or None,
            "payment_reference": self.payment_reference or None,
            "approval_required": self.approval_required,
        }
