"""Authoritative payment-claim reconciliation shared by runtime paths."""

from db.commercial_queries import schedule_action
from services.followup_lifecycle import pending_reference
from services.ppv_reconciliation import PPVReconcileDisposition, reconcile_pending_ppv


async def verify_ppv_purchase(
    fan_id: str,
    creator_id: str,
    pending: dict,
) -> None:
    """Verify against platform state; a fan-authored claim is never proof."""
    try:
        reference = pending_reference(pending)
        result = await reconcile_pending_ppv(
            creator_id=creator_id,
            fan_id=fan_id,
            expected_reference=reference,
        )
        if result.disposition == PPVReconcileDisposition.PENDING and result.retry_at:
            await schedule_action(
                creator_id=creator_id,
                fan_id=fan_id,
                action_type="PPV_RECONCILE",
                execute_at=result.retry_at,
                payload={"payment_reference": reference},
                dedupe_key=f"ppv-reconcile:{fan_id}:{reference}",
            )
        print(
            f"[PPV VERIFY] fan={fan_id} disposition={result.disposition.value} "
            f"reference={reference} reason={result.reason}"
        )
    except Exception as exc:
        # The durable action remains pending and will retry with backoff.
        print(f"[PPV VERIFY ERROR] fan={fan_id}: {exc}")
