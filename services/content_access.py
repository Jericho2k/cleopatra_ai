"""Repairing access to something a customer already paid for.

``docs/autonomy_architecture_review.md`` §3B. The baseline treated "I can't
open it" as a sales opportunity: it called the platform directly with the
existing price, outside the authoritative delivery service, after commercial
state had already been updated. PR #48 removed that and put a
``content_access_issue`` review hold in its place — containment, and explicitly
labelled as such:

    This is containment, not autonomous repair. [...] It does not resolve media
    entitlements, refresh expired URLs, inspect the dashboard renderer, or
    notify an operator outside the existing review workflow. It sends no
    customer-facing repair claim. A safe verified access-recovery workflow
    remains necessary.

This module is that workflow. It holds one rule, from review §4: *delivery
claims must be tied to the operation result*. Nothing here tells a customer
anything has been fixed until the platform has accepted the repair and returned
a receipt for it.

**The evidence comes first.** ``inspect_content_access`` is read-only. It joins
what this backend believes — the delivery ledger, which is the only authority on
whether money arrived — to what the platform currently shows for that message:
is it still there, does it still carry its media, and does the account media
still resolve. An operator opening a frozen conversation sees those side by
side, which is the question they were previously left to answer from the
customer's word alone.

**The repair is bounded by what was paid for.** ``resend_paid_content`` re-sends
exactly the media of a delivery the ledger records as ``purchased``, at no
charge, through the ordinary message path. Three properties make that safe, and
each is enforced here rather than assumed:

* it refuses unless the ledger says ``purchased`` — a customer who has not paid
  is not owed a free copy, and a complaint is not proof of payment;
* it sends the media unpriced, so the platform cannot present a second charge
  for an item already bought;
* it re-sends the exact ``media_ids`` from the ledger row, never a re-planned
  selection, so a repair cannot quietly become a different offer.

**Nothing here decides on its own.** Every function is called by an operator
resolving a review hold. Full Auto still stops at the hold; the review is
explicit that a request to fix access outranks a new commercial suggestion, and
an automated repair that talks to the customer is a bigger step than the
evidence for it currently supports.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.apifansly_gate import apifansly_enabled
from core.supabase import get_supabase
from db.queries import clear_fan_review, save_message
from services import content_access_repairs as repairs
from services.content_access_repairs import RepairInProgress
from services.db_reliability import retry_transient_db_operation
from models.conversation_continuity import ResolvedBy, ThreadKind, ThreadStatus
from services.conversation_continuity import open_threads_for, resolve_thread
from services.ppv_delivery_ledger import PAID_STATUS, list_fan_deliveries

#: The review reason Full Auto writes when the analyzer reports that a customer
#: cannot reach something. Defined here so the producer in
#: ``services/suggestions.py`` and the resolver share one string.
REVIEW_REASON = "content_access_issue"

#: What an operator may do about it.
RESOLUTION_RESEND = "resend_paid_content"
RESOLUTION_ACCESS_RESTORED = "access_restored"
RESOLUTION_NOT_AN_ACCESS_ISSUE = "not_an_access_issue"
RESOLUTIONS = (
    RESOLUTION_RESEND,
    RESOLUTION_ACCESS_RESTORED,
    RESOLUTION_NOT_AN_ACCESS_ISSUE,
)

#: What the platform says about the paid message right now.
PLATFORM_VISIBLE = "visible"
PLATFORM_MESSAGE_GONE = "message_not_visible"
PLATFORM_MEDIA_GONE = "media_not_visible"
PLATFORM_UNKNOWN = "unknown"

#: The text that accompanies a repair. Deliberately a constant and deliberately
#: plain: this is the one customer-facing sentence in an access repair, it is
#: sent only after the platform has accepted the media, and it claims nothing
#: beyond "here it is again". No apology script, no upsell, no writer call —
#: a model asked to phrase this could promise a fix that has not happened.
RESEND_MESSAGE = "here it is again 💕"


class ContentAccessError(RuntimeError):
    """A content-access repair could not be completed safely."""


@dataclass
class PaidItem:
    """One delivery the ledger records as paid, and its platform state."""

    reference: str
    media_ids: list[str]
    price_cents: int
    platform_message_id: str
    purchased_at: str
    platform_state: str = PLATFORM_UNKNOWN
    platform_detail: str = ""
    #: The repair already attempted for this exact purchase under the current
    #: review case, if there is one. Carried per item rather than per customer
    #: so the panel can disable the one button that would duplicate, instead of
    #: disabling all of them or none.
    repair: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "media_ids": list(self.media_ids),
            "price_cents": int(self.price_cents),
            "platform_message_id": self.platform_message_id,
            "purchased_at": self.purchased_at,
            "platform_state": self.platform_state,
            "platform_detail": self.platform_detail,
            "repair": self.repair,
        }


@dataclass
class AccessEvidence:
    """Everything an operator needs to decide, and nothing they have to infer."""

    fan_id: str
    creator_id: str
    frozen: bool
    review_reason: str
    paid_items: list[PaidItem] = field(default_factory=list)
    #: Why the platform side could not be checked, when it could not be.
    platform_error: str = ""
    #: The identity of the hold being looked at. The panel sends it back with
    #: the resolution, so the repair is bound to the hold the operator was
    #: actually reading rather than to whatever is current when it lands.
    review_case_id: str = ""
    #: Every repair attempted for this customer, newest first.
    repairs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def repairable(self) -> list[PaidItem]:
        """Paid items a free resend would be a valid answer to.

        A paid item whose message the platform can no longer see is the clearest
        case. One that is still visible is included too: the customer may be
        looking at a broken render rather than a missing message, and resending
        a copy they already own costs nothing and cannot double-charge.
        """
        return [item for item in self.paid_items if item.media_ids]

    @property
    def selection_required(self) -> bool:
        """Whether the operator must say WHICH purchase they mean.

        More than one repairable item and the backend cannot know which one the
        complaint was about. It used to pick the most recent, so a complaint
        about an older item resent a newer one — the customer gets a second
        copy of something that was working and still cannot open the thing they
        asked about.
        """
        return len(self.repairable) > 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "fan_id": self.fan_id,
            "creator_id": self.creator_id,
            "frozen": self.frozen,
            "review_reason": self.review_reason,
            "review_case_id": self.review_case_id,
            "paid_items": [item.as_dict() for item in self.paid_items],
            "repairable_count": len(self.repairable),
            "platform_error": self.platform_error,
            # Said explicitly rather than left to be inferred from an empty
            # list: "this customer has paid for nothing" and "we could not
            # check" are different answers and lead to different actions.
            "has_paid_content": bool(self.paid_items),
            # The panel must not offer a default it cannot justify.
            "selection_required": self.selection_required,
            "repairs": list(self.repairs),
        }


async def _load_fan(fan_id: str) -> dict[str, Any]:
    def _load() -> dict[str, Any]:
        return (
            get_supabase()
            .table("fans")
            .select(
                "creator_id, platform_fan_id, fansly_group_id, "
                "needs_human_review, review_reason, review_case_id"
            )
            .eq("id", fan_id)
            .single()
            .execute()
        ).data or {}

    fan = await retry_transient_db_operation(
        lambda: asyncio.to_thread(_load),
        label=f"content access fan={fan_id}",
        log_prefix="ACCESS RETRY",
    )
    if not fan.get("creator_id"):
        raise ContentAccessError("fan was not found")
    return fan


async def _creator_account_id(creator_id: str) -> str:
    def _load() -> str:
        row = (
            get_supabase()
            .table("creators")
            .select("apifansly_account_id")
            .eq("id", creator_id)
            .single()
            .execute()
        ).data or {}
        return str(row.get("apifansly_account_id") or "")

    return await asyncio.to_thread(_load)


def _platform_state_for(
    item: PaidItem,
    messages: list[dict[str, Any]],
    account_media: list[dict[str, Any]],
) -> tuple[str, str]:
    """Classify what the platform currently shows for one paid message.

    Read-only and deliberately coarse. It answers "is the thing he paid for
    still there?", which is what an operator needs, and it does not attempt to
    diagnose a renderer or a CDN — the review lists inspecting the dashboard
    renderer as out of scope, and guessing at one from here would be inventing
    a cause.
    """
    sent = next(
        (
            row
            for row in messages
            if str(row.get("id") or "") == item.platform_message_id
        ),
        None,
    )
    if sent is None:
        return (
            PLATFORM_MESSAGE_GONE,
            "the platform no longer lists the message this content was sent in",
        )

    attachment_ids = {
        str(attachment.get("contentId") or "")
        for attachment in (sent.get("attachments") or [])
        if isinstance(attachment, dict) and attachment.get("contentId")
    }
    if not attachment_ids:
        return (
            PLATFORM_MEDIA_GONE,
            "the message is still there but no longer carries any media",
        )

    resolved = [
        row
        for row in account_media
        if isinstance(row, dict)
        and (
            str(row.get("id") or "") in attachment_ids
            or str(row.get("mediaId") or "") in attachment_ids
        )
    ]
    if not resolved:
        return (
            PLATFORM_MEDIA_GONE,
            "the message still references media the platform will not return",
        )
    return (
        PLATFORM_VISIBLE,
        "the platform still shows the message and its media",
    )


async def inspect_content_access(fan_id: str) -> AccessEvidence:
    """What this customer paid for, and what the platform shows for it now.

    Read-only: it sends nothing, changes nothing, and clears no hold. It is the
    first half of the workflow on purpose — the review's objection to the old
    behaviour was that it acted on a complaint without establishing anything,
    and an operator repeating that mistake faster is not an improvement.
    """
    fan = await _load_fan(fan_id)
    creator_id = str(fan["creator_id"])

    deliveries = await list_fan_deliveries(creator_id, fan_id)
    paid_items = [
        PaidItem(
            reference=str(row.get("reference") or ""),
            media_ids=[str(value) for value in (row.get("media_ids") or []) if value],
            price_cents=int(row.get("price_cents") or 0),
            platform_message_id=str(row.get("platform_message_id") or ""),
            purchased_at=str(row.get("purchased_at") or ""),
        )
        for row in deliveries
        if str(row.get("status") or "") == PAID_STATUS
    ]

    case_id = str(fan.get("review_case_id") or "")
    evidence = AccessEvidence(
        fan_id=str(fan_id),
        creator_id=creator_id,
        frozen=bool(fan.get("needs_human_review")),
        review_reason=str(fan.get("review_reason") or ""),
        review_case_id=case_id,
        paid_items=paid_items,
    )

    # What has already been attempted. Read before the platform call, so a
    # reload always shows the operation's real state even when the provider is
    # unreachable — an in-flight or unknown repair invisible to the operator is
    # exactly what makes them press the button again.
    attempted = await repairs.history(creator_id=creator_id, fan_id=fan_id)
    evidence.repairs = [repair.as_dict() for repair in attempted]
    current = {
        repair.reference: repair
        for repair in reversed(attempted)
        if repair.review_case_id == case_id
    }
    for item in paid_items:
        repair = current.get(item.reference)
        if repair is not None:
            item.repair = repair.as_dict()

    group_id = str(fan.get("fansly_group_id") or "")
    account_id = await _creator_account_id(creator_id) if paid_items else ""
    if not paid_items:
        return evidence
    if not (group_id and account_id and apifansly_enabled()):
        # Reported rather than guessed. A blank platform_state next to a
        # confident verdict would be the same kind of false certainty this
        # module exists to remove.
        evidence.platform_error = (
            "no live platform route for this conversation; showing ledger only"
        )
        return evidence

    from services.apifansly import CHAT_MESSAGE_PAGE_MAX, list_chat_messages

    try:
        # One page, at the provider's documented maximum. A wider sweep would
        # cost credits on every operator glance at a frozen conversation, and a
        # complaint is almost always about something bought recently; a paid
        # item older than this page reports as unknown rather than as missing.
        messages, account_media, _ = await list_chat_messages(
            account_id, group_id, limit=CHAT_MESSAGE_PAGE_MAX
        )
    except Exception as exc:
        evidence.platform_error = f"could not read the conversation: {exc}"
        return evidence

    # A full page means the read may simply not go back far enough, so "not on
    # this page" is inconclusive. A short page means the whole conversation was
    # returned, and an absent message really is absent. Without the distinction
    # a paid item older than ten messages is indistinguishable from one the
    # platform has removed, and an operator is sent after a problem that may not
    # exist.
    whole_conversation_seen = len(messages) < CHAT_MESSAGE_PAGE_MAX

    for item in paid_items:
        if not item.platform_message_id:
            item.platform_state = PLATFORM_UNKNOWN
            item.platform_detail = (
                "this delivery has no platform message id, so it cannot be located"
            )
            continue
        state, detail = _platform_state_for(item, list(messages), list(account_media))
        if state == PLATFORM_MESSAGE_GONE and not whole_conversation_seen:
            state = PLATFORM_UNKNOWN
            detail = (
                "older than the newest page of this conversation, so its "
                "current state was not checked"
            )
        item.platform_state, item.platform_detail = state, detail
    return evidence


async def resend_paid_content(
    fan_id: str,
    *,
    reference: str = "",
    review_case_id: str = "",
    actor: str = "",
) -> dict[str, Any]:
    """Send a paid item again, free, at most once, and only once accepted.

    ``reference`` picks one of the customer's paid deliveries; the most recent
    is used when it is omitted. The ledger is the authority throughout: the
    media that goes out is the media that row records, and a row that is not
    ``purchased`` is refused rather than repaired.

    THE CLAIM
    ---------
    Everything before the claim is a read. The claim is written and committed
    BEFORE the platform call, and it is what makes a retry reconcile instead of
    send. Three failures reproduced at backend 4a1683a without it: a DB failure
    after a successful send followed by an operator retry sent twice; two
    concurrent operators sent twice; and neither left any record that a send
    had happened at all. See db/content_access_repair_v1.sql.

    No lock is held across the platform call — the claim is a committed row,
    not a held transaction, so a provider's latency never pins a connection and
    a process that dies mid-send leaves its claim behind to be found.

    Returns the receipt. Raises ``ContentAccessError`` when there is nothing
    that can be safely resent, which leaves the review hold exactly where it
    was — an operator must be able to tell "repaired" from "could not repair",
    and a cleared hold would say the first.
    """
    fan = await _load_fan(fan_id)
    creator_id = str(fan["creator_id"])
    group_id = str(fan.get("fansly_group_id") or "")
    account_id = await _creator_account_id(creator_id)
    case_id = str(review_case_id or fan.get("review_case_id") or "")

    deliveries = await list_fan_deliveries(creator_id, fan_id)
    paid = [row for row in deliveries if str(row.get("status") or "") == PAID_STATUS]
    if reference:
        matched = [row for row in paid if str(row.get("reference") or "") == reference]
        if not matched and paid:
            # A reference that names nothing is a mistake, not an instruction to
            # fall back to the newest item. Falling back is how an operator
            # asking for one purchase resends a different one.
            raise ContentAccessError(
                "that purchase is not in this customer's confirmed deliveries. "
                "Reload the access panel and choose again — the reference may "
                "belong to another conversation."
            )
        paid = matched
    if not paid:
        raise ContentAccessError(
            "this customer has no confirmed purchase to resend. A complaint is "
            "not proof of payment, and sending paid media to someone the ledger "
            "does not show as having bought it would be giving it away."
        )

    repairable = [row for row in paid if (row.get("media_ids") or [])]
    if not reference and len(repairable) > 1:
        # The backend cannot know which purchase the complaint was about. It
        # used to assume the most recent one, so a complaint about an older
        # item resent a newer one: the customer receives a second copy of
        # something that was working and still cannot open what they asked
        # about, and the hold is cleared as though it were repaired.
        options = ", ".join(str(row.get("reference") or "") for row in repairable)
        raise ContentAccessError(
            "this customer has more than one confirmed purchase, so which one "
            "to resend has to be chosen rather than assumed. Select the item "
            f"the complaint is about and try again (available: {options})."
        )

    # list_fan_deliveries orders newest first.
    row = paid[0]
    item_reference = str(row.get("reference") or "")
    media_ids = [str(value) for value in (row.get("media_ids") or []) if value]
    if not media_ids:
        raise ContentAccessError(
            "the confirmed purchase does not record which media it delivered, "
            "so there is nothing safe to resend"
        )
    if not (group_id and account_id and apifansly_enabled()):
        raise ContentAccessError(
            "there is no live delivery route for this conversation"
        )

    # Everything above is a read and refuses without a claim, so a repair that
    # was never possible leaves no row suggesting one was attempted.
    try:
        repair = await repairs.claim(
            creator_id=creator_id,
            fan_id=fan_id,
            reference=item_reference,
            review_case_id=case_id,
            media_ids=media_ids,
            claimed_by=actor,
        )
    except RepairInProgress as exc:
        raise ContentAccessError(exc.repair.operator_message()) from exc

    from services.apifansly import send_message as send_apifansly_message
    from services.apifansly import sent_message_id

    # Unpriced on purpose. message_payload only adds access_type/price when a
    # price is passed, so omitting it is what makes this a free copy of
    # something already bought rather than a second charge.
    try:
        response_body = await send_apifansly_message(
            account_id,
            group_id,
            content=RESEND_MESSAGE,
            media_ids=media_ids,
        )
    except Exception as exc:
        # The request left this process. Whether the platform acted on it is
        # not knowable from an exception — a timeout on the response side looks
        # identical to one on the request side. So this is `unknown`, never
        # `failed`: a person decides whether to send again, because that is the
        # decision that can cost the customer a duplicate.
        await repairs.mark_unknown(
            repair,
            detail=f"the platform call raised {type(exc).__name__}",
        )
        raise ContentAccessError(
            "the resend could not be completed and it is not known whether the "
            "platform received it. The conversation stays on hold and the "
            "attempt is recorded, so a second send is a decision rather than a "
            "reflex."
        ) from exc

    platform_message_id = sent_message_id(response_body)
    if not platform_message_id:
        # No receipt means no proof it arrived — and equally no proof it did
        # not. Recorded as unknown rather than failed, and the hold stays: an
        # unverified repair must never be reported as one.
        await repairs.mark_unknown(
            repair,
            detail="the platform accepted the resend but returned no message id",
        )
        raise ContentAccessError(
            "the platform accepted the resend but returned no message id, so "
            "it cannot be confirmed as delivered"
        )

    # Settled before the local receipt is written, and deliberately in that
    # order: the send is now proven, so no retry may send again, whatever
    # happens next. A failure below costs a local record, not a duplicate.
    repair = await repairs.confirm(repair, platform_message_id=platform_message_id)

    repaired_at = datetime.now(timezone.utc).isoformat()
    await save_message(
        fan_id=fan_id,
        creator_id=creator_id,
        role="creator",
        content=RESEND_MESSAGE,
        was_ai_suggested=False,
        fansly_message_id=platform_message_id,
        media_context={
            # Recorded as a repair, not as a sale. The price is zero and the
            # reference names the purchase it restores, so this message can
            # never be read back as a second PPV against the same media.
            "content_access_repair": {
                "restored_reference": item_reference,
                "media_ids": media_ids,
                "price_cents": 0,
                "original_price_cents": int(row.get("price_cents") or 0),
                "platform_message_id": platform_message_id,
                "repaired_at": repaired_at,
                # The claim this message belongs to, so the row and the
                # operation can be joined after the fact.
                "repair_id": repair.id,
            }
        },
    )
    print(
        f"[CONTENT ACCESS] fan={fan_id} resend=accepted "
        f"reference={item_reference} media={media_ids} price_cents=0 "
        f"message={platform_message_id} repair={repair.id} case={case_id}"
    )
    return {
        "status": "resent",
        "reference": item_reference,
        "media_ids": media_ids,
        "platform_message_id": platform_message_id,
        "repaired_at": repaired_at,
        "repair": repair.as_dict(),
    }


async def resolve_content_access(
    fan_id: str,
    *,
    resolution: str,
    reference: str = "",
    review_case_id: str = "",
    actor: str = "",
) -> dict[str, Any]:
    """Apply an operator's decision about a content-access hold.

    The hold is cleared only on a path that actually concluded something:
    a verified resend, an operator stating they restored access themselves, or
    an operator stating the analyzer misread the message. A failed repair
    raises and leaves the conversation frozen.

    The hold that is cleared is THE hold this resolution was about. It used to
    be an unconditional update by fan id, so a repair that finished after a
    crisis hold had been raised cleared the crisis hold on its way out —
    reproduced at backend 4a1683a, where a conversation frozen for crisis
    language silently resumed. The case id is read before the work starts and
    compared at the end, and a hold that moved underneath is left alone and
    said so in the result rather than reported as cleared.
    """
    if resolution not in RESOLUTIONS:
        raise ContentAccessError(f"unsupported content-access resolution: {resolution}")

    fan = await _load_fan(fan_id)
    if str(fan.get("review_reason") or "") != REVIEW_REASON:
        raise ContentAccessError(
            "this conversation is not on hold for a content-access problem"
        )
    # Read before anything slow happens, so the comparison at the end is
    # against the hold the operator was actually looking at.
    case_id = str(fan.get("review_case_id") or "")

    # The panel sends back the case it rendered. A mismatch means the hold was
    # cleared and re-raised, or replaced by a different one, between the
    # operator reading the evidence and acting on it — so the decision was made
    # about a situation that no longer exists. Refused rather than applied to
    # whatever is current now, which is the same class of mistake as resending
    # the newest purchase because none was named.
    if review_case_id and review_case_id != case_id:
        raise ContentAccessError(
            "this hold changed while you were looking at it, so the decision "
            "would be applied to a different situation than the one on screen. "
            "Reload the access panel and decide again."
        )

    result: dict[str, Any]
    if resolution == RESOLUTION_RESEND:
        result = await resend_paid_content(
            fan_id, reference=reference, review_case_id=case_id, actor=actor
        )
    elif resolution == RESOLUTION_ACCESS_RESTORED:
        result = {"status": "access_restored_by_operator", "fan_id": str(fan_id)}
        print(f"[CONTENT ACCESS] fan={fan_id} resolution=restored_by_operator")
    else:
        result = {"status": "not_an_access_issue", "fan_id": str(fan_id)}
        print(f"[CONTENT ACCESS] fan={fan_id} resolution=not_an_access_issue")

    await _close_access_complaints(
        creator_id=str(fan["creator_id"]),
        fan_id=str(fan_id),
        resolution=resolution,
    )
    cleared = await retry_transient_db_operation(
        lambda: clear_fan_review(fan_id, expected_case_id=case_id),
        label=f"clear content-access review fan={fan_id}",
        log_prefix="ACCESS RETRY",
    )
    if not cleared:
        print(
            f"[CONTENT ACCESS] fan={fan_id} hold_not_cleared: a different hold "
            f"is in place now (resolved case={case_id})"
        )
    return {
        **result,
        "review_cleared": bool(cleared),
        # Stated rather than inferred from review_cleared, because the two
        # answers lead an operator to different next actions: the repair itself
        # succeeded, and the conversation is still frozen for something else.
        "review_case_id": case_id,
        "hold_superseded": not cleared,
    }


#: How each resolution closes the obligation the complaint created.
#:
#: A resend and an operator-restored fix both DEALT with it. "Not an access
#: problem" did not: it says the analyzer misread the message, so the
#: obligation was never real and is cancelled rather than reported as met.
_THREAD_OUTCOME: dict[str, ThreadStatus] = {
    RESOLUTION_RESEND: ThreadStatus.FULFILLED,
    RESOLUTION_ACCESS_RESTORED: ThreadStatus.FULFILLED,
    RESOLUTION_NOT_AN_ACCESS_ISSUE: ThreadStatus.CANCELLED,
}


async def _close_access_complaints(
    *, creator_id: str, fan_id: str, resolution: str
) -> None:
    """Close the obligation the complaint created, now that it is answered.

    Clearing the review flag alone would leave the conversation still carrying
    "he cannot access what he paid for" forever, and every later turn would be
    written as though the problem were live. Best effort: a thread that outlives
    its resolution is a stale line in a prompt, never a wrong action, so this
    must not be able to fail the operator's resolution.
    """
    status = _THREAD_OUTCOME.get(resolution)
    if status is None:
        return
    try:
        carried = await open_threads_for(creator_id, fan_id, expire_first=False)
        for thread in carried:
            if thread.kind != ThreadKind.COMPLAINT:
                continue
            await resolve_thread(
                thread.id,
                status=status,
                resolved_by=ResolvedBy.OPERATOR,
                note=f"content-access resolution: {resolution}",
            )
    except Exception as exc:  # pragma: no cover - continuity is never fatal
        print(f"[CONTENT ACCESS] could not close the complaint thread fan={fan_id}: {exc}")
