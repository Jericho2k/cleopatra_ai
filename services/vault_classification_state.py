"""When an already-classified vault item must be classified AGAIN — and when not.

The rule
--------
A successful classification at the current classifier version is FINISHED WORK.
An ordinary daily vault sync discovers and updates metadata; it does not spend
vision or media credits re-deriving an answer it already has.

Re-classification happens for exactly five reasons, and this module is the only
place that decides:

1. **New media.** Never classified, or classified into nothing.
2. **Classifier version moved.** ``classification_version`` is older than the
   running one. Deliberately an explicit, confirmed operator upgrade rather
   than something a sync does on its own — a version bump across a 10,000-item
   vault is a paid re-analysis, not a background task.
3. **The stored classification is incomplete, stale or errored.** A partial
   result (thumbnail only, because the media-cost guard refused a deep video
   scan) or a failed one is not finished work, so it is eligible to be
   retried when conditions may have changed.
4. **The operator asked.** The manual re-analyze control, which is explicit,
   rate-limited and allowed to override everything here.
5. **The media identity changed.** The row now points at different bytes — a
   different platform media id, a different MIME type — so the stored answer
   describes an asset that is no longer there.

Anything else is a no-op, and a no-op is the point.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

# Stored in ``creator_vault_media.classification_status``.
STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_PENDING = "pending"
STATUS_ERROR = "error"

# A stored status that does NOT represent finished work.
UNFINISHED_STATUSES = frozenset({STATUS_PARTIAL, STATUS_PENDING, STATUS_ERROR})

# Stable reason codes, so telemetry and tests agree on why work happened.
REASON_NEW = "new_media"
REASON_VERSION = "classifier_version"
REASON_UNFINISHED = "incomplete_or_error"
REASON_IDENTITY = "media_identity_changed"
REASON_OPERATOR = "operator_requested"
REASON_CURRENT = "already_classified"


def media_identity_key(row: dict[str, Any]) -> str:
    """A short fingerprint of WHICH BYTES this row points at.

    Built from the platform media identity and the declared type rather than
    from the signed URL: a signed URL is rotated constantly and rotating it is
    not a change of content, so keying on it would re-classify the whole vault
    every time links were refreshed — exactly the cost this module exists to
    avoid.

    The filename is deliberately excluded too. It is cosmetic, an operator may
    rename an asset, and nothing in the classification is derived from it.
    """
    parts = [
        str(row.get("fansly_media_id") or row.get("media_id") or "").strip(),
        str(row.get("mimetype") or "").strip().lower(),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ClassificationDecision:
    needed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"needed": self.needed, "reason": self.reason}


def classification_decision(
    row: dict[str, Any],
    *,
    classifier_version: int,
    operator_requested: bool = False,
    allow_version_upgrade: bool = False,
) -> ClassificationDecision:
    """Whether this row needs classification work, and why.

    ``allow_version_upgrade`` is False for ordinary sync and new-media runs: a
    version bump is a confirmed, paid operator action, so a daily sync that
    happened to run after a deploy must not quietly start re-analysing the
    whole vault.
    """
    if operator_requested:
        return ClassificationDecision(True, REASON_OPERATOR)

    category = str(row.get("content_category") or "").strip()
    classified_at = str(row.get("classified_at") or "").strip()
    if not category or not classified_at:
        return ClassificationDecision(True, REASON_NEW)

    status = str(row.get("classification_status") or "").strip().lower()
    if status in UNFINISHED_STATUSES:
        return ClassificationDecision(True, REASON_UNFINISHED)

    stored_key = str(row.get("classification_media_key") or "").strip()
    if stored_key and stored_key != media_identity_key(row):
        return ClassificationDecision(True, REASON_IDENTITY)

    try:
        version = int(row.get("classification_version") or 0)
    except (TypeError, ValueError):
        version = 0
    if version < int(classifier_version):
        if allow_version_upgrade:
            return ClassificationDecision(True, REASON_VERSION)
        # Eligible, but only through an explicit confirmed upgrade run. A
        # normal sync leaves it alone.
        return ClassificationDecision(False, REASON_CURRENT)

    return ClassificationDecision(False, REASON_CURRENT)


def needs_classification(
    row: dict[str, Any],
    *,
    classifier_version: int,
    operator_requested: bool = False,
    allow_version_upgrade: bool = False,
) -> bool:
    return classification_decision(
        row,
        classifier_version=classifier_version,
        operator_requested=operator_requested,
        allow_version_upgrade=allow_version_upgrade,
    ).needed


def select_items_for_classification(
    rows: Iterable[dict[str, Any]],
    *,
    classifier_version: int,
    allow_version_upgrade: bool = False,
    operator_requested: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter a candidate batch down to the rows that actually need work.

    Returns the rows plus a count per reason, which is what lets a run report
    "312 new, 4 retried after a partial result, 9,684 already classified"
    instead of a bare total that hides whether anything was re-spent.
    """
    selected: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for row in rows:
        decision = classification_decision(
            row,
            classifier_version=classifier_version,
            operator_requested=operator_requested,
            allow_version_upgrade=allow_version_upgrade,
        )
        reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
        if decision.needed:
            selected.append(row)
    return selected, reasons
