"""The owner's persistent simulation workspace: state, test fans, delayed actions.

The simulator is a long-lived testing environment, not a scratchpad. Everything
it shows is the REAL persisted state the production pipeline reads and writes;
nothing here computes a parallel simulator-only version of lifecycle, price
learning, affordability, sessions or commercial status. When a number looks
wrong in this panel, the number is wrong in production too.

Three capabilities live here:

``simulation_state``
    One read of everything the right-hand panel shows for a test fan.

``create_test_fan``
    A real, persistent fan row with a ``test_`` platform id and nothing else:
    no conversation, no purchases, no learned budget, no stale commercial or
    session state. It is structurally impossible for this to create a Fansly
    fan — the platform id is generated here and always carries the prefix, and
    the simulator's own boundary refuses anything that does not.

``run_scheduled_action_now``
    Fires a pending durable action immediately, through the SAME handler the
    worker uses, inside ``simulation_scope()``. That is how a payday follow-up
    scheduled for next Friday is tested today without a fake clock: the real
    revalidation, the real planner, the real writer and the real state
    transitions all run; only the wait is skipped, and the transport is refused.

Every function here is owner-only and test-fan-only. The route layer proves the
caller is an allowlisted simulator user and that the fan belongs to the creator
and carries the ``test_`` prefix; each function re-checks the prefix itself
rather than trusting its caller, because "runs the real pipeline with delivery
disabled" is exactly the capability that must never point at a real fan.
"""

from __future__ import annotations

import asyncio
import secrets
from datetime import datetime, timezone
from typing import Any

from core.apifansly_gate import simulation_scope
from core.simulation import TEST_FAN_PREFIX, is_simulatable_fan
from core.supabase import get_supabase


class SimulationWorkspaceError(RuntimeError):
    """A workspace operation could not be completed."""


class NotASimulationFan(SimulationWorkspaceError):
    """Refused: the target fan is not an owner test fan."""


# Actions the owner may fire early. Deliberately a list rather than "anything
# pending": these are the proactive, time-delayed promises the commercial layer
# makes, which is the whole reason a fake clock would otherwise be needed.
# AUTO_REPLY is excluded — an inbound reply is triggered by typing in the
# simulator, not by waiting — and so is anything whose handler reconciles
# against the platform.
RUNNABLE_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "PAYDAY_REENGAGEMENT",
        "POST_SESSION_FOLLOWUP",
        "ABANDONED_PPV_FOLLOWUP",
        "ABANDONED_OFFER_FOLLOWUP",
        "INACTIVITY_REENGAGEMENT",
        "POST_PURCHASE_REACTION",
        "OFFER_EXPIRY",
    }
)

ACTION_LABELS: dict[str, str] = {
    "PAYDAY_REENGAGEMENT": "Payday follow-up",
    "POST_SESSION_FOLLOWUP": "Post-session follow-up",
    "ABANDONED_PPV_FOLLOWUP": "Abandoned PPV follow-up",
    "ABANDONED_OFFER_FOLLOWUP": "Abandoned offer follow-up",
    "INACTIVITY_REENGAGEMENT": "Inactivity re-engagement",
    "POST_PURCHASE_REACTION": "Post-purchase reaction",
    "OFFER_EXPIRY": "Offer expiry",
    "AUTO_REPLY": "Auto reply",
    "PPV_RECONCILE": "PPV reconciliation",
    "PROCESS_INBOUND_MESSAGE": "Process inbound message",
}


async def _fan_row(fan_id: str, creator_id: str | None = None) -> dict:
    def _get() -> dict | None:
        query = (
            get_supabase().table("fans")
            .select(
                "id, creator_id, display_name, platform_fan_id, total_spent, "
                "spend_tier, active_session, pending_ppv_check, sales_log, "
                "needs_human_review, sale_paused_at, auto_mode, ai_summary"
            )
            .eq("id", str(fan_id))
        )
        if creator_id:
            query = query.eq("creator_id", str(creator_id))
        rows = query.limit(1).execute().data or []
        return rows[0] if rows else None

    row = await asyncio.to_thread(_get)
    if not row:
        raise NotASimulationFan(f"fan {fan_id} not found")
    if not is_simulatable_fan(row.get("platform_fan_id")):
        # The boundary, re-checked here rather than trusted from the caller.
        raise NotASimulationFan("not a simulation fan")
    return row


async def require_simulation_fan(fan_id: str, creator_id: str | None = None) -> dict:
    """The fan row, or NotASimulationFan. Never returns a real fan."""
    return await _fan_row(fan_id, creator_id)


# ---------------------------------------------------------------------------
# Right panel: the persisted state of one test fan
# ---------------------------------------------------------------------------


async def simulation_state(*, creator_id: str, fan_id: str) -> dict[str, Any]:
    """Everything the simulator's state panel shows, from authoritative sources.

    Each block is read through the same helper production reads it through, so
    this panel cannot drift from what the pipeline actually acts on. A block
    that fails to load is reported as ``None`` rather than failing the whole
    request: a diagnostic panel that shows nothing because one table was slow
    is worse than one that shows most of the truth.
    """
    from db.affordability_queries import get_affordability_state
    from db.commercial_queries import get_fan_state
    from db.fan_intelligence_queries import get_fan_intelligence_context
    from db.fan_lifecycle_queries import (
        get_lifecycle_state,
        get_purchase_aggregates,
        lifecycle_row_to_context,
    )
    from db.price_learning_queries import get_price_learning_profile
    from services.ai_stack import resolve_ai_stack

    fan = await require_simulation_fan(fan_id, creator_id)

    async def _safe(label: str, coro):
        try:
            return await coro
        except Exception as exc:
            print(f"[SIMULATION STATE] {label} read failed fan={fan_id}: {exc}")
            return None

    (
        commercial,
        lifecycle_row,
        purchases,
        affordability,
        price_learning,
        intelligence,
        actions,
        stack,
    ) = await asyncio.gather(
        _safe("commercial_state", get_fan_state(fan_id)),
        _safe("lifecycle", get_lifecycle_state(fan_id)),
        _safe("purchases", get_purchase_aggregates(fan_id)),
        _safe("affordability", get_affordability_state(fan_id)),
        _safe("price_learning", get_price_learning_profile(fan_id)),
        _safe("fan_intelligence", get_fan_intelligence_context(fan_id)),
        _safe("scheduled_actions", pending_scheduled_actions(fan_id)),
        _safe("ai_stack", resolve_ai_stack(creator_id=creator_id, fan_id=fan_id)),
    )

    commercial_json = (
        commercial.model_dump(mode="json") if commercial is not None else {}
    )
    session = fan.get("active_session") or None

    return {
        "fan": {
            "id": str(fan.get("id")),
            "display_name": fan.get("display_name") or str(fan.get("id")),
            "platform_fan_id": fan.get("platform_fan_id"),
            "simulation": True,
            "auto_mode": fan.get("auto_mode"),
            "needs_human_review": bool(fan.get("needs_human_review")),
            "sale_paused_at": fan.get("sale_paused_at"),
        },
        "ai_stack": (stack.to_dict() if stack is not None else None),
        # Confirmed money only. This is simulated spend on a simulated fan and
        # is excluded from every production revenue view (see
        # core.simulation.exclude_simulation_fans).
        "spend": {
            "total_spent": float(fan.get("total_spent") or 0),
            "purchase_count": int((purchases or {}).get("purchase_count") or 0),
            "total_spent_cents": int((purchases or {}).get("total_spent_cents") or 0),
            "highest_purchase_cents": int(
                (purchases or {}).get("highest_purchase_cents") or 0
            ),
            "spend_tier": fan.get("spend_tier"),
        },
        "lifecycle": lifecycle_row_to_context(lifecycle_row) if lifecycle_row else None,
        "affordability": (
            affordability.model_dump(mode="json") if affordability is not None else None
        ),
        "price_learning": (
            price_learning.to_context() if price_learning is not None else None
        ),
        "commercial": {
            "status": commercial_json.get("status"),
            "desired_experience": commercial_json.get("desired_experience"),
            "confirmed_budget_cents": commercial_json.get("confirmed_budget_cents"),
            "budget_source": commercial_json.get("budget_source"),
            "offered_packages": commercial_json.get("offered_packages") or [],
            "selected_package_id": commercial_json.get("selected_package_id"),
            "selected_package_label": commercial_json.get("selected_package_label"),
            "selected_package_price_cents": commercial_json.get(
                "selected_package_price_cents"
            ),
            "last_offer_at": commercial_json.get("last_offer_at"),
            "last_declined_price_cents": commercial_json.get(
                "last_declined_price_cents"
            ),
            "payday_raw": commercial_json.get("payday_raw"),
            "payday_at": commercial_json.get("payday_at"),
            "next_followup_at": commercial_json.get("next_followup_at"),
            "next_followup_type": commercial_json.get("next_followup_type"),
        },
        "active_session": session,
        "pending_ppv": fan.get("pending_ppv_check") or None,
        "fan_intelligence": intelligence or {},
        "scheduled_actions": actions or [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Delayed behaviour, without a fake clock
# ---------------------------------------------------------------------------


async def pending_scheduled_actions(fan_id: str) -> list[dict[str, Any]]:
    """Durable actions still owed to this fan, soonest first."""

    def _get() -> list[dict]:
        return (
            get_supabase().table("scheduled_actions")
            .select("id, action_type, execute_at, status, attempts, last_error, payload")
            .eq("fan_id", str(fan_id))
            .in_("status", ["PENDING", "PROCESSING"])
            .order("execute_at")
            .limit(25)
            .execute()
        ).data or []

    rows = await asyncio.to_thread(_get)
    return [
        {
            "id": str(row.get("id")),
            "action_type": str(row.get("action_type") or ""),
            "label": ACTION_LABELS.get(
                str(row.get("action_type") or ""), str(row.get("action_type") or "")
            ),
            "execute_at": row.get("execute_at"),
            "status": row.get("status"),
            "attempts": int(row.get("attempts") or 0),
            "last_error": row.get("last_error"),
            "can_run_now": str(row.get("action_type") or "") in RUNNABLE_ACTION_TYPES,
        }
        for row in rows
    ]


async def run_scheduled_action_now(
    *,
    creator_id: str,
    fan_id: str,
    action_id: str,
) -> dict[str, Any]:
    """Fire one pending action immediately, through the production handler.

    This is deliberately not a reimplementation. It claims the row the way the
    worker claims it, then calls ``workers.scheduled_actions._resolve_action``,
    so the revalidation gate, the session planner, the writer, the persistence
    and every resulting state transition are the real ones. The only two
    differences are that the wait is skipped and that the whole thing runs
    inside ``simulation_scope()``, which refuses every API Fansly request made
    by this task or anything it spawns.
    """
    from workers.scheduled_actions import _resolve_action

    await require_simulation_fan(fan_id, creator_id)

    def _load() -> dict | None:
        rows = (
            get_supabase().table("scheduled_actions")
            .select("*")
            .eq("id", str(action_id))
            .eq("fan_id", str(fan_id))
            .eq("creator_id", str(creator_id))
            .limit(1)
            .execute()
        ).data or []
        return rows[0] if rows else None

    action = await asyncio.to_thread(_load)
    if not action:
        raise SimulationWorkspaceError("No such scheduled action for this test fan.")

    action_type = str(action.get("action_type") or "")
    if action_type not in RUNNABLE_ACTION_TYPES:
        raise SimulationWorkspaceError(
            f"{action_type or 'This action'} cannot be run early from the simulator."
        )
    if str(action.get("status") or "") not in {"PENDING", "PROCESSING"}:
        raise SimulationWorkspaceError(
            f"That action is already {str(action.get('status') or 'resolved').lower()}."
        )

    now = datetime.now(timezone.utc)

    def _claim() -> bool:
        response = (
            get_supabase().table("scheduled_actions")
            .update({"status": "PROCESSING", "locked_at": now.isoformat()})
            .eq("id", str(action_id))
            .eq("status", action.get("status"))
            .execute()
        )
        return bool(response.data)

    if not await asyncio.to_thread(_claim):
        # The worker got there first. Reporting that honestly is better than
        # running the handler twice.
        raise SimulationWorkspaceError(
            "The scheduled-actions worker claimed that action first."
        )

    claimed = {**action, "status": "PROCESSING", "locked_at": now.isoformat()}
    sent_counter = [0]
    print(
        f"[SIMULATION] run-now creator={creator_id} fan={fan_id} "
        f"action={action_id} type={action_type}"
    )
    with simulation_scope():
        outcome = await _resolve_action(claimed, sent_counter=sent_counter)

    print(
        f"[SIMULATION] run-now complete fan={fan_id} action={action_id} "
        f"outcome={outcome} sent={sent_counter[0]}"
    )
    return {
        "status": "ok",
        "simulation": True,
        "action_id": str(action_id),
        "action_type": action_type,
        "outcome": outcome,
        "messages_sent": int(sent_counter[0]),
    }


# ---------------------------------------------------------------------------
# Creating a test fan
# ---------------------------------------------------------------------------


def generate_test_platform_fan_id() -> str:
    """A platform id that is structurally a test id and cannot be a Fansly one.

    The prefix is not decoration: ``core.simulation.is_simulatable_fan`` is the
    only thing that makes a fan eligible for the simulator at all, and
    ``services.suggestions`` uses the same prefix to route delivery locally
    instead of to the platform. Generating the id here — never accepting one
    from a client — is what makes it impossible for this control to create a
    fan that later turns out to be real.
    """
    return f"{TEST_FAN_PREFIX}{secrets.token_hex(6)}"


async def create_test_fan(
    *,
    creator_id: str,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Create one clean, persistent simulation fan for this creator.

    "Clean" is explicit rather than inherited from column defaults: no
    conversation, no purchase history, no learned budget, no stale commercial or
    session state. Every other per-fan system (commercial state, affordability,
    price learning, lifecycle, fan intelligence) is keyed by fan id and simply
    has no rows yet, which is the same thing a brand-new real fan looks like.
    """
    platform_fan_id = generate_test_platform_fan_id()
    name = (display_name or "").strip() or f"Test fan {platform_fan_id[-4:]}"

    row = {
        "creator_id": str(creator_id),
        "platform_fan_id": platform_fan_id,
        "display_name": name,
        "total_spent": 0,
        "spend_tier": "new",
        "sales_log": [],
        "active_session": None,
        "pending_ppv_check": None,
        "needs_human_review": False,
        "sale_paused_at": None,
        "ai_summary": None,
        # Full Auto on by default: a test fan exists to exercise the Auto path.
        "auto_mode": True,
    }

    def _insert() -> dict:
        response = get_supabase().table("fans").insert(row).execute()
        created = (response.data or [None])[0]
        if not created:
            raise SimulationWorkspaceError("The database did not return the new fan.")
        return created

    try:
        created = await asyncio.to_thread(_insert)
    except SimulationWorkspaceError:
        raise
    except Exception as exc:
        print(f"[SIMULATION] test fan creation failed creator={creator_id}: {exc}")
        raise SimulationWorkspaceError("Could not create the test fan.") from exc

    print(
        f"[SIMULATION] created test fan creator={creator_id} "
        f"fan={created.get('id')} platform_fan={platform_fan_id}"
    )
    return {
        "id": str(created.get("id")),
        "display_name": created.get("display_name") or name,
        "platform_fan_id": platform_fan_id,
        "simulation": True,
    }
