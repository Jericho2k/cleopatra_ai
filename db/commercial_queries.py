"""Persistence for commercial policy, fan state and scheduled actions."""
import asyncio
from datetime import datetime, timedelta, timezone

from core.pagination import fetch_all_rows
from core.supabase import get_supabase
from models.commercial import CreatorPolicy, FanCommercialState, PackageOption
from services.media_packages import build_offer_packages, usable_sets


async def get_creator_policy(creator_id: str) -> CreatorPolicy:
    def _get():
        response = (
            get_supabase().table("creator_commercial_policies")
            .select("*").eq("creator_id", creator_id).execute()
        )
        return (response.data or [None])[0]

    row = await asyncio.to_thread(_get)
    if not row:
        return CreatorPolicy()
    row.pop("creator_id", None)
    row.pop("updated_at", None)
    try:
        return CreatorPolicy(**row)
    except Exception:
        return CreatorPolicy()


async def save_creator_policy(creator_id: str, policy: CreatorPolicy) -> CreatorPolicy:
    payload = {"creator_id": creator_id, **policy.model_dump(mode="json")}
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()

    def _upsert():
        response = get_supabase().table("creator_commercial_policies").upsert(
            payload, on_conflict="creator_id"
        ).execute()
        return (response.data or [payload])[0]

    await asyncio.to_thread(_upsert)
    return policy


async def get_fan_state(fan_id: str) -> FanCommercialState:
    def _get():
        response = (
            get_supabase().table("fan_commercial_states")
            .select("*").eq("fan_id", fan_id).execute()
        )
        return (response.data or [None])[0]

    row = await asyncio.to_thread(_get)
    if not row:
        return FanCommercialState()
    for key in ("fan_id", "creator_id", "updated_at"):
        row.pop(key, None)
    try:
        return FanCommercialState(**row)
    except Exception:
        return FanCommercialState()


async def save_fan_state(
    fan_id: str,
    creator_id: str,
    state: FanCommercialState,
) -> None:
    payload = {
        "fan_id": fan_id,
        "creator_id": creator_id,
        "status": state.status.value,
        "desired_experience": state.desired_experience,
        "preferences_snapshot": state.preferences_snapshot,
        "confirmed_budget_cents": state.confirmed_budget_cents,
        "budget_source": state.budget_source,
        "offered_packages": [p.model_dump(mode="json") for p in state.offered_packages],
        "selected_package_id": state.selected_package_id,
        "selected_package_set_id": state.selected_package_set_id,
        "selected_package_set_ids": state.selected_package_set_ids,
        "selected_package_label": state.selected_package_label,
        "selected_package_price_cents": state.selected_package_price_cents,
        "last_offer_at": state.last_offer_at.isoformat() if state.last_offer_at else None,
        "payday_raw": state.payday_raw,
        "payday_at": state.payday_at.isoformat() if state.payday_at else None,
        "payday_confidence": state.payday_confidence,
        "last_declined_price_cents": state.last_declined_price_cents,
        "teaser_messages_used": state.teaser_messages_used,
        "free_session_started_at": (
            state.free_session_started_at.isoformat()
            if state.free_session_started_at else None
        ),
        "free_session_ended_at": (
            state.free_session_ended_at.isoformat()
            if state.free_session_ended_at else None
        ),
        "last_session_completed_at": (
            state.last_session_completed_at.isoformat()
            if state.last_session_completed_at else None
        ),
        "last_session_revenue_cents": state.last_session_revenue_cents,
        "last_session_package_id": state.last_session_package_id,
        "last_session_set_ids": state.last_session_set_ids,
        "last_session_experience": state.last_session_experience,
        "last_abandoned_ppv_at": (
            state.last_abandoned_ppv_at.isoformat()
            if state.last_abandoned_ppv_at else None
        ),
        "last_abandoned_media_id": state.last_abandoned_media_id,
        "next_followup_at": (
            state.next_followup_at.isoformat() if state.next_followup_at else None
        ),
        "next_followup_type": state.next_followup_type,
        "next_followup_payload": state.next_followup_payload,
        "next_followup_dedupe_key": state.next_followup_dedupe_key,
        "last_followup_at": (
            state.last_followup_at.isoformat() if state.last_followup_at else None
        ),
        "last_inactivity_reengagement_at": (
            state.last_inactivity_reengagement_at.isoformat()
            if state.last_inactivity_reengagement_at else None
        ),
        "inactivity_reengagement_window_started_at": (
            state.inactivity_reengagement_window_started_at.isoformat()
            if state.inactivity_reengagement_window_started_at else None
        ),
        "inactivity_reengagement_count": state.inactivity_reengagement_count,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    def _upsert():
        get_supabase().table("fan_commercial_states").upsert(
            payload,
            on_conflict="fan_id",
        ).execute()

    await asyncio.to_thread(_upsert)


async def merge_fan_ai_summary(fan_id: str, patch: dict) -> None:
    """Immediately persist high-value facts such as a stated payday.

    The periodic memory summarizer is deliberately not relied on for commercial
    promises because it may not run on this message.
    """
    def _merge():
        db = get_supabase()
        response = db.table("fans").select("ai_summary").eq("id", fan_id).single().execute()
        summary = ((response.data or {}).get("ai_summary") or {}).copy()
        summary.update({key: value for key, value in patch.items() if value is not None})
        db.table("fans").update({"ai_summary": summary}).eq("id", fan_id).execute()

    await asyncio.to_thread(_merge)


async def get_offerable_packages(
    creator_id: str,
    fan_id: str,
    policy: CreatorPolicy,
    price_learning: dict | None = None,
    desired_experience: str | None = None,
    hard_ceiling_cents: int | None = None,
) -> list[PackageOption]:
    """Build up to two coherent, multi-step packages from approved vault sets."""
    def _get():
        db = get_supabase()
        rows = (
            db.table("vault_sets")
            .select(
                "id, title, description, location, outfit, suggested_price, tags, "
                "explicit_min, explicit_max, media_ids, base_price_cents, "
                "min_price_cents, max_price_cents, dynamic_pricing_enabled"
            )
            .eq("creator_id", creator_id)
            .eq("status", "approved")
            .execute()
        ).data or []

        # Paginated: this set is the "never offer this again" list. Truncated at
        # 1,000 creator messages it silently forgets older sends, and Cleopatra
        # re-offers content the fan already received. Ordered by id, which is
        # unique, so a page boundary cannot drop or repeat a row.
        sent_rows = fetch_all_rows(
            lambda start, end: db.table("messages")
            .select("media_context")
            .eq("fan_id", fan_id)
            .eq("role", "creator")
            .not_.is_("media_context", "null")
            .order("id")
            .range(start, end)
            .execute()
        )
        sent_set_ids: set[str] = set()
        sent_media_ids: set[str] = set()
        for row in sent_rows:
            ppv = (row.get("media_context") or {}).get("ppv") or {}
            if ppv.get("set_id"):
                sent_set_ids.add(str(ppv["set_id"]))
            for media_id in (ppv.get("media_ids") or [ppv.get("media_id")]):
                if media_id:
                    sent_media_ids.add(str(media_id))

        fan_row = (
            db.table("fans").select("ai_summary, preferences")
            .eq("id", fan_id).single().execute()
        ).data or {}
        summary = fan_row.get("ai_summary") or {}
        preferences = fan_row.get("preferences") or {}
        preferred_tags = list(summary.get("kinks") or [])
        if isinstance(preferences, dict):
            preferred_tags.extend(str(value) for value in preferences.values() if isinstance(value, str))
        elif isinstance(preferences, list):
            preferred_tags.extend(str(value) for value in preferences)

        available = usable_sets(rows, sent_set_ids)
        for row in available:
            row["media_ids"] = [
                str(media_id)
                for media_id in (row.get("media_ids") or [])
                if str(media_id) not in sent_media_ids
            ]
        available = [row for row in available if row.get("media_ids")]
        return available, preferred_tags

    rows, preferred_tags = await asyncio.to_thread(_get)
    from db.pricing_policy_queries import get_effective_price_learning_policy

    pricing_policy = await get_effective_price_learning_policy(creator_id)
    return build_offer_packages(
        rows,
        policy,
        preferred_tags=preferred_tags,
        price_learning=price_learning,
        desired_experience=desired_experience,
        hard_ceiling_cents=hard_ceiling_cents,
        pricing_policy=pricing_policy,
    )


async def schedule_action(
    creator_id: str,
    fan_id: str,
    action_type: str,
    execute_at: datetime,
    payload: dict,
    dedupe_key: str,
    *,
    replace_existing: bool = True,
) -> None:
    row = {
        "creator_id": creator_id,
        "fan_id": fan_id,
        "action_type": action_type,
        "execute_at": execute_at.isoformat(),
        "payload": payload,
        "dedupe_key": dedupe_key,
        "status": "PENDING",
        "attempts": 0,
        "locked_at": None,
        "last_error": None,
    }

    def _upsert():
        get_supabase().table("scheduled_actions").upsert(
            row,
            on_conflict="dedupe_key",
            ignore_duplicates=not replace_existing,
        ).execute()

    await asyncio.to_thread(_upsert)


async def ensure_action_pending(
    creator_id: str,
    fan_id: str,
    action_type: str,
    execute_at: datetime,
    payload: dict,
    dedupe_key: str,
) -> None:
    """Repair a missing/terminal durable action without disturbing a live claim."""
    row = {
        "creator_id": creator_id,
        "fan_id": fan_id,
        "action_type": action_type,
        "execute_at": execute_at.isoformat(),
        "payload": payload,
        "dedupe_key": dedupe_key,
        "status": "PENDING",
        "attempts": 0,
        "locked_at": None,
        "last_error": None,
    }

    def _ensure():
        db = get_supabase()
        existing = (
            db.table("scheduled_actions")
            .select("id, status, last_error")
            .eq("dedupe_key", dedupe_key)
            .limit(1)
            .execute()
        ).data or []
        current = existing[0] if existing else None
        # Shared with the batched repair pass so the two can never disagree
        # about what "needs repair" means.
        if not action_needs_repair(current):
            return
        if current is None:
            db.table("scheduled_actions").insert(row).execute()
        else:
            db.table("scheduled_actions").update(row).eq(
                "id", current["id"]
            ).execute()

    await asyncio.to_thread(_ensure)


async def cancel_actions_for_fan(
    fan_id: str,
    action_type: str | None = None,
) -> None:
    def _cancel():
        cancellable_statuses = (
            ["PENDING", "FAILED", "PROCESSING"]
            if action_type == "AUTO_REPLY"
            else ["PENDING", "FAILED"]
        )
        query = (
            get_supabase().table("scheduled_actions")
            .update({"status": "CANCELLED"})
            .eq("fan_id", fan_id)
            .in_("status", cancellable_statuses)
        )
        if action_type:
            query = query.eq("action_type", action_type)
        query.execute()

    await asyncio.to_thread(_cancel)


async def cancel_action_by_dedupe_key(dedupe_key: str) -> None:
    def _cancel():
        (
            get_supabase().table("scheduled_actions")
            .update({"status": "CANCELLED"})
            .eq("dedupe_key", dedupe_key)
            .in_("status", ["PENDING", "FAILED"])
            .execute()
        )

    await asyncio.to_thread(_cancel)


# Set once per process when the atomic claim function turns out to be missing,
# so a rolling deploy that reaches the new code before the migration falls back
# once rather than paying a failed RPC on every poll.
_ATOMIC_CLAIM_AVAILABLE = True


def _looks_like_missing_function(error: Exception) -> bool:
    """Whether this error means the RPC is not deployed yet.

    Deliberately narrow. A genuine failure inside the function — a constraint
    violation, a deadlock — must surface, not silently downgrade the claim path.
    """

    text = str(error).lower()
    return (
        "claim_due_actions" in text
        and (
            "could not find" in text
            or "does not exist" in text
            or "undefined function" in text
            or "pgrst202" in text
        )
    )


async def claim_due_actions(limit: int = 20, stale_minutes: int = 10) -> list[dict]:
    """Take ownership of up to ``limit`` due and ``limit`` stale actions.

    One atomic statement (see db/scheduled_action_claim_v1.sql) rather than two
    selects plus a compare-and-swap UPDATE per row — 22 round trips for a batch
    of 20. The database side is also strictly safer than the CAS it replaces:
    FOR UPDATE SKIP LOCKED means two workers never select the same row, so the
    race the CAS existed to lose is never entered.

    The per-row CAS remains as the fallback for a deployment whose migration has
    not been applied yet. It is exactly the previous implementation, and it is
    correct on its own — this is a rollout affordance, not a weaker path.
    """

    global _ATOMIC_CLAIM_AVAILABLE

    if _ATOMIC_CLAIM_AVAILABLE:
        try:
            response = await asyncio.to_thread(
                lambda: get_supabase()
                .rpc(
                    "claim_due_actions",
                    {"p_limit": int(limit), "p_stale_minutes": int(stale_minutes)},
                )
                .execute()
            )
            return list(response.data or [])
        except Exception as error:
            if not _looks_like_missing_function(error):
                raise
            _ATOMIC_CLAIM_AVAILABLE = False
            print(
                "[SCHEDULED ACTIONS] claim_due_actions() is not deployed; "
                "falling back to per-row compare-and-swap. Apply "
                "db/scheduled_action_claim_v1.sql."
            )

    return await _claim_due_actions_by_cas(limit, stale_minutes)


async def _claim_due_actions_by_cas(limit: int, stale_minutes: int) -> list[dict]:
    now = datetime.now(timezone.utc)
    stale_before = (now - timedelta(minutes=stale_minutes)).isoformat()

    def _claim():
        db = get_supabase()
        due = (
            db.table("scheduled_actions")
            .select("*")
            .eq("status", "PENDING")
            .lte("execute_at", now.isoformat())
            .order("execute_at")
            .limit(limit)
            .execute()
        ).data or []
        stale = (
            db.table("scheduled_actions")
            .select("*")
            .eq("status", "PROCESSING")
            .lt("locked_at", stale_before)
            .limit(limit)
            .execute()
        ).data or []

        claimed = []
        for row in due + stale:
            query = (
                db.table("scheduled_actions")
                .update({"status": "PROCESSING", "locked_at": now.isoformat()})
                .eq("id", row["id"])
                .eq("status", row["status"])
            )
            if row["status"] == "PROCESSING" and row.get("locked_at"):
                query = query.eq("locked_at", row["locked_at"])
            response = query.execute()
            if response.data:
                claimed.append(row)
        return claimed

    return await asyncio.to_thread(_claim)


async def complete_action(action_id: str) -> None:
    def _done():
        get_supabase().table("scheduled_actions").update(
            {"status": "COMPLETED", "locked_at": None}
        ).eq("id", action_id).eq("status", "PROCESSING").execute()

    await asyncio.to_thread(_done)


async def fail_action(
    action_id: str,
    error: str,
    attempts: int,
    max_attempts: int = 3,
) -> None:
    status = "FAILED" if attempts + 1 >= max_attempts else "PENDING"
    retry_at = datetime.now(timezone.utc) + timedelta(
        minutes=min(60, 5 * (2 ** max(0, int(attempts))))
    )

    def _fail():
        payload = {
            "status": status,
            "attempts": attempts + 1,
            "last_error": error[:500],
            "locked_at": None,
        }
        if status == "PENDING":
            payload["execute_at"] = retry_at.isoformat()
        get_supabase().table("scheduled_actions").update(payload).eq(
            "id", action_id
        ).eq("status", "PROCESSING").execute()

    await asyncio.to_thread(_fail)


async def fail_action_terminal(
    action_id: str,
    code: str,
    detail: str,
    attempts: int,
) -> None:
    """Fail an action once, with no further retries (REL-005).

    For failures where retrying runs the same expensive pipeline to reach the
    same answer — the creator is disconnected, there is no delivery route at
    all. The eight-attempt budget exists to outlast a bad ten minutes at a
    provider; spending it on a configuration problem costs eight analyzer and
    writer runs and still cannot send.

    last_error carries a machine-readable marker so the operator health surface
    can tell a broken binding from a flaky provider without parsing prose.
    """
    from core.action_failures import TERMINAL_PREFIX

    marker = f"{TERMINAL_PREFIX}:{code}: {detail}"[:500]

    def _fail():
        get_supabase().table("scheduled_actions").update({
            "status": "FAILED",
            "attempts": attempts + 1,
            "last_error": marker,
            "locked_at": None,
        }).eq("id", action_id).eq("status", "PROCESSING").execute()

    await asyncio.to_thread(_fail)


async def reschedule_action(
    action_id: str,
    execute_at: datetime,
    *,
    payload: dict | None = None,
) -> None:
    update = {
        "status": "PENDING",
        "execute_at": execute_at.isoformat(),
        "locked_at": None,
        "last_error": None,
    }
    if payload is not None:
        update["payload"] = payload

    def _reschedule():
        get_supabase().table("scheduled_actions").update(update).eq(
            "id", action_id
        ).eq("status", "PROCESSING").execute()

    await asyncio.to_thread(_reschedule)


async def update_action_payload(action_id: str, payload: dict) -> None:
    """Persist delivery evidence while an action is still being processed.

    Scheduled text delivery crosses two systems (API Fansly and Supabase), so the
    payload doubles as a small durable send journal.  A stale worker can use it to
    reconcile an accepted platform message instead of blindly sending it again.
    """

    def _update():
        (
            get_supabase().table("scheduled_actions")
            .update({"payload": payload})
            .eq("id", action_id)
            .execute()
        )

    await asyncio.to_thread(_update)


async def retry_failed_action_for_fan(action_id: str, fan_id: str) -> bool:
    """Explicitly requeue one failed action after an operator reviews its error."""
    now = datetime.now(timezone.utc).isoformat()

    def _retry() -> bool:
        response = (
            get_supabase().table("scheduled_actions")
            .update({
                "status": "PENDING",
                "execute_at": now,
                "attempts": 0,
                "locked_at": None,
                "last_error": None,
            })
            .eq("id", action_id)
            .eq("fan_id", fan_id)
            .eq("status", "FAILED")
            .execute()
        )
        return bool(response.data)

    return await asyncio.to_thread(_retry)


async def get_scheduled_actions_for_fan(
    fan_id: str,
    *,
    statuses: tuple[str, ...] = ("PENDING", "PROCESSING", "FAILED"),
) -> list[dict]:
    def _get():
        query = (
            get_supabase().table("scheduled_actions")
            .select("*")
            .eq("fan_id", fan_id)
            .order("execute_at")
        )
        if statuses:
            query = query.in_("status", list(statuses))
        return query.execute().data or []

    return await asyncio.to_thread(_get)


async def get_followup_obligations(
    page_size: int = 500,
    *,
    due_before: datetime | None = None,
) -> list[dict]:
    """Return durable follow-up obligations using stable pagination.

    ``due_before`` bounds the scan to obligations that could fire soon. Without
    it every row with a future ``next_followup_at`` — including one scheduled for
    next Friday — was fetched on every worker cycle for a week. The repair pass
    only has to guarantee that a durable action exists *before* its execute time,
    so a short upcoming horizon is sufficient and the invariant is unchanged.

    The former hard 100-row limit could permanently starve repairs once an
    agency accumulated more than 100 active obligations.
    """
    def _get():
        db = get_supabase()
        size = max(1, min(int(page_size), 1_000))
        offset = 0
        rows: list[dict] = []
        while True:
            query = (
                db.table("fan_commercial_states")
                .select(
                    "fan_id, creator_id, next_followup_at, next_followup_type, "
                    "next_followup_payload, next_followup_dedupe_key"
                )
                .not_.is_("next_followup_at", "null")
                .not_.is_("next_followup_type", "null")
            )
            if due_before is not None:
                query = query.lte("next_followup_at", due_before.isoformat())
            page = (
                query
                .order("fan_id")
                .range(offset, offset + size - 1)
                .execute()
            ).data or []
            rows.extend(page)
            if len(page) < size:
                return rows
            offset += size

    return await asyncio.to_thread(_get)


# PostgREST puts ``in.(...)`` filters in the query string, so the key list has to
# stay well inside a sane URL length. 200 keys per request keeps the repair pass
# at a handful of round trips even with a large horizon.
_DEDUPE_KEY_CHUNK = 200


async def get_action_states_by_dedupe_key(dedupe_keys: list[str]) -> dict[str, dict]:
    """Return ``{dedupe_key: {id, status, last_error}}`` for the given keys.

    One bounded query per chunk replaces the per-obligation SELECT that made the
    repair pass cost two round trips for every outstanding obligation.
    """
    keys = [key for key in dict.fromkeys(str(k) for k in dedupe_keys) if key]
    if not keys:
        return {}

    def _get() -> dict[str, dict]:
        db = get_supabase()
        found: dict[str, dict] = {}
        for start in range(0, len(keys), _DEDUPE_KEY_CHUNK):
            chunk = keys[start:start + _DEDUPE_KEY_CHUNK]
            rows = (
                db.table("scheduled_actions")
                .select("id, dedupe_key, status, last_error")
                .in_("dedupe_key", chunk)
                .execute()
            ).data or []
            for row in rows:
                key = str(row.get("dedupe_key") or "")
                if key:
                    found[key] = row
        return found

    return await asyncio.to_thread(_get)


async def bulk_upsert_pending_actions(rows: list[dict]) -> int:
    """Insert or reset a batch of durable actions, keyed on ``dedupe_key``.

    Used only by the repair pass, which has already established that each row
    either has no action at all or has one in a terminal state that must be
    recreated. Rows with a live PENDING/PROCESSING action are never passed here,
    so this cannot disturb an in-flight claim.
    """
    if not rows:
        return 0

    def _upsert() -> int:
        db = get_supabase()
        written = 0
        for start in range(0, len(rows), _DEDUPE_KEY_CHUNK):
            chunk = rows[start:start + _DEDUPE_KEY_CHUNK]
            db.table("scheduled_actions").upsert(
                chunk,
                on_conflict="dedupe_key",
            ).execute()
            written += len(chunk)
        return written

    return await asyncio.to_thread(_upsert)


def action_needs_repair(existing: dict | None) -> bool:
    """Decide whether a durable action must be recreated for a live obligation.

    Deliberately identical to the single-row logic in ``ensure_action_pending``:
    a missing action, an action already COMPLETED for an obligation that is still
    current, or one FAILED by the rolling-deployment compatibility error. Every
    other state (PENDING, PROCESSING, CANCELLED, other FAILED) is left alone.
    """
    if not existing:
        return True
    status = existing.get("status")
    if status == "COMPLETED":
        return True
    return (
        status == "FAILED"
        and "get_creator_auto_mode_default" in str(existing.get("last_error") or "")
    )
