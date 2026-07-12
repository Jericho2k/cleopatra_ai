"""Persistence for commercial policy, fan state and scheduled actions."""
import asyncio
from datetime import datetime, timedelta, timezone

from core.supabase import get_supabase
from models.commercial import CreatorPolicy, FanCommercialState, PackageOption


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
) -> list[PackageOption]:
    """Return up to two real, approved set-backed options near policy targets."""
    def _get():
        db = get_supabase()
        rows = (
            db.table("vault_sets")
            .select("id, title, suggested_price, tags, explicit_min, explicit_max")
            .eq("creator_id", creator_id)
            .eq("status", "approved")
            .execute()
        ).data or []

        # Exclude sets already sent where the message recorded a set_id.
        sent_rows = (
            db.table("messages")
            .select("media_context")
            .eq("fan_id", fan_id)
            .eq("role", "creator")
            .not_.is_("media_context", "null")
            .execute()
        ).data or []
        sent_set_ids = {
            str((row.get("media_context") or {}).get("ppv", {}).get("set_id"))
            for row in sent_rows
            if (row.get("media_context") or {}).get("ppv", {}).get("set_id")
        }

        candidates: list[PackageOption] = []
        for row in rows:
            set_id = str(row.get("id") or "")
            price = row.get("suggested_price")
            if not set_id or set_id in sent_set_ids or price is None:
                continue
            try:
                price_cents = int(round(float(price) * 100))
            except (TypeError, ValueError):
                continue
            if price_cents <= 0:
                continue
            label = str(row.get("title") or "private set").strip()
            tags = row.get("tags") or []
            experience = ", ".join(str(tag) for tag in tags[:3]) if tags else None
            candidates.append(PackageOption(
                package_id=f"set:{set_id}",
                label=label,
                price_cents=price_cents,
                set_id=set_id,
                experience=experience,
            ))
        return candidates

    candidates = await asyncio.to_thread(_get)
    if not candidates:
        return []

    def nearest(target: int, excluded: set[str]) -> PackageOption | None:
        remaining = [c for c in candidates if c.package_id not in excluded]
        if not remaining:
            return None
        return min(remaining, key=lambda c: abs(c.price_cents - target))

    chosen: list[PackageOption] = []
    quick = nearest(policy.quick_package_target_cents, set())
    if quick:
        chosen.append(quick)
    if policy.offer_two_packages:
        full = nearest(
            policy.full_package_target_cents,
            {package.package_id for package in chosen},
        )
        if full:
            chosen.append(full)
    return sorted(chosen, key=lambda package: package.price_cents)


async def schedule_action(
    creator_id: str,
    fan_id: str,
    action_type: str,
    execute_at: datetime,
    payload: dict,
    dedupe_key: str,
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
        ).execute()

    await asyncio.to_thread(_upsert)


async def cancel_actions_for_fan(
    fan_id: str,
    action_type: str | None = None,
) -> None:
    def _cancel():
        query = (
            get_supabase().table("scheduled_actions")
            .update({"status": "CANCELLED"})
            .eq("fan_id", fan_id)
            .eq("status", "PENDING")
        )
        if action_type:
            query = query.eq("action_type", action_type)
        query.execute()

    await asyncio.to_thread(_cancel)


async def claim_due_actions(limit: int = 20, stale_minutes: int = 10) -> list[dict]:
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
            response = (
                db.table("scheduled_actions")
                .update({"status": "PROCESSING", "locked_at": now.isoformat()})
                .eq("id", row["id"])
                .eq("status", row["status"])
                .execute()
            )
            if response.data:
                claimed.append(row)
        return claimed

    return await asyncio.to_thread(_claim)


async def complete_action(action_id: str) -> None:
    def _done():
        get_supabase().table("scheduled_actions").update(
            {"status": "COMPLETED"}
        ).eq("id", action_id).execute()

    await asyncio.to_thread(_done)


async def fail_action(
    action_id: str,
    error: str,
    attempts: int,
    max_attempts: int = 3,
) -> None:
    status = "FAILED" if attempts + 1 >= max_attempts else "PENDING"

    def _fail():
        get_supabase().table("scheduled_actions").update({
            "status": status,
            "attempts": attempts + 1,
            "last_error": error[:500],
            "locked_at": None,
        }).eq("id", action_id).execute()

    await asyncio.to_thread(_fail)
