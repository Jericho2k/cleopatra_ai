"""FastAPI entrypoint for Cleopatra.

Routes are thin and delegate all logic to services.
"""

import asyncio
import hashlib
from core.tasks import spawn
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ai.generator import generate_replies
from ai.prompt_builder import build_prompt
from ai.situation_analyzer import analyze_situation
from ai.rag import find_similar_exchanges
from ai.stage_classifier import classify_stage
from core.action_telemetry import stage as action_stage
from core.bounded_state import BoundedIdSet, prune_expired
from core.pagination import fetch_all_rows_async
from core.supabase import (
    close_supabase_client,
    describe as supabase_transport_description,
    get_supabase,
)
from core.vault_gate import VAULT_GATE
from core.webhooks import valid_hmac_sha256_signature
from core.tenancy import (
    require_account_path_access,
    require_creator_access,
    require_creator_fan_access,
    require_creator_path_access,
    require_fan_path_access,
    require_ppv_approval_path_access,
    require_vault_item_path_access,
)
from db.fan_intelligence_queries import get_fan_intelligence_context
from db.queries import (
    create_fan,
    get_conversation_history,
    get_creator_persona,
    get_fan,
    get_fan_by_id,
    get_ppv_offers,
    MessageWriteResult,
    save_message,
    save_message_result,
    update_message_media_context,
)
from db.commercial_queries import schedule_action
from models.commercial import CreatorPolicy
from models.schemas import (
    ConversationContext,
    Fan,
    Persona,
    SuggestionRequest,
    SuggestionResponse,
)
from services.fan_intelligence import learn_from_fan_message
from services.db_reliability import retry_db_read, retry_transient_db_operation
from core.apifansly_gate import (
    REASON_DISABLED,
    apifansly_enabled,
    describe_apifansly,
)
from services.apifansly import (
    ApiFanslyAccountAccessError,
    ApiFanslyConfigurationError,
    account_media_prices,
    client_scope as apifansly_client_scope,
    close_shared_client as close_apifansly_client,
    current_account as apifansly_current_account,
    download_media as apifansly_download_media,
    headers as apifansly_headers,
    is_fansly_cdn_url,
    list_chat_messages as apifansly_list_chat_messages,
    list_chats as apifansly_list_chats,
    list_vault_album_media as apifansly_list_vault_album_media,
    list_vault_albums as apifansly_list_vault_albums,
    raise_for_response as raise_for_apifansly_response,
    send_message as send_apifansly_message,
    response_message as apifansly_response_message,
    url as apifansly_url,
    sent_message_id,
    usage_snapshot as apifansly_usage_snapshot,
)
from services.auto_audience import AutoAudiencePolicy
from services.fansly_poller import FanslyPoller
from services.fansly_session_store import SessionStore
from services.model_availability import (
    current_model_availability,
    model_availability_scheduler,
)
from services.shoot_fingerprint import (
    add_semantic_shoot_evidence,
    build_shoot_clusters,
    build_shoot_fingerprint,
    cluster_debug_summary,
)
from services.vault_classifier import (
    VaultClassifierError,
    classify_vault_image,
)
from services.voice_calibration import (
    list_voice_calibration_candidates,
    save_voice_calibration,
)
from services.suggestions import (
    _should_update_memory,
    _update_fan_ai_summary,
    _update_fan_memory,
    get_suggestions,
    schedule_auto_reply,
)
from services.vault_operations import (
    MANUAL_RECATEGORIZATION_DAILY_LIMIT,
    categorize_new_batch_enabled,
    manual_recategorization_usage,
    normalize_media_ids,
)
from services.vault_sync import (
    ordered_vault_albums,
    should_stop_album_scan,
    vault_sync_cooldown,
)
from services.video_frames import (
    FrameSettings,
    build_contact_sheet,
    extract_frames,
    ffmpeg_available,
)
from services.vault_metadata import (
    VAULT_CLASSIFIER_VERSION,
    build_set_description,
    classification_confidence,
    explicitness_from_evidence,
    media_description,
    normalize_media_category,
    semantic_tags,
    useful_text,
)


# REL-004 — a bounded FIFO, not a set that empties itself.
#
# This used to be a plain set cleared wholesale at 1,000 entries, which meant
# item 1,001 erased the previous 1,000 identities in one go and left the dedupe
# window empty. A redelivery arriving in that window was processed twice.
#
# Only a cheap short-circuit in front of the database: message identity is
# enforced by the unique index from REL-002, so losing this on restart costs a
# little duplicate work and never correctness.
_processed_messages = BoundedIdSet(
    maxsize=max(100, int(os.getenv("PROCESSED_MESSAGE_CACHE_SIZE", "5000")))
)
_vault_sync_state: dict = {}
_vault_sync_retry_after: dict[str, float] = {}
# "queued" is an active status: a creator waiting for a vault slot must not
# be started a second time by the scheduler or an operator (VAULT-001).
_VAULT_SYNC_ACTIVE_STATUSES = {"queued", "running", "categorizing_new"}

# VAULT-003 — a random identity for THIS process, minted once at import.
#
# Deliberately not a PID or a hostname: two containers can share a host and a
# PID can be reused, and either collision would make a dead process's
# interrupted run look like a live one. It is only ever compared for equality.
_PROCESS_ID = uuid.uuid4().hex
_protected_video_download_gate = asyncio.Semaphore(1)
_active_chat_binding_retry_after: dict[str, float] = {}
_active_chat_binding_tasks: dict[str, asyncio.Task] = {}
_ACTIVE_CHAT_BINDING_RETRY_SECONDS = 15 * 60
_VAULT_SYNC_INTERVAL_HOURS = max(
    1.0, float(os.getenv("VAULT_SYNC_INTERVAL_HOURS", "24"))
)
_VAULT_AUTOSYNC_CHECK_SECONDS = max(
    300, int(os.getenv("VAULT_AUTOSYNC_CHECK_SECONDS", "3600"))
)
_VAULT_ACCESS_DENIED_RETRY_SECONDS = max(
    3600, int(os.getenv("VAULT_ACCESS_DENIED_RETRY_SECONDS", str(24 * 60 * 60)))
)


async def get_or_fetch_group_id(apifansly_id: str, platform_fan_id: str, fan_id: str) -> str | None:
    """Find the group_id for a fan by scanning recent chats."""

    try:
        async with apifansly_client_scope() as client:
            cursor = None
            for _ in range(5):  # check up to 5 pages
                chats, accounts, cursor = await apifansly_list_chats(
                    apifansly_id,
                    cursor=cursor,
                    client=client,
                )
                account_lookup = {str(a.get("id", "")): a for a in accounts}

                for chat in chats:
                    if str(chat.get("partnerAccountId", "")) == str(platform_fan_id):
                        group_id = str(chat.get("groupId", ""))
                        if group_id:
                            db = get_supabase()
                            update = {"fansly_group_id": group_id}
                            # Also grab display name and avatar
                            account = account_lookup.get(str(platform_fan_id), {})
                            display_name = account.get("displayName") or account.get("username")
                            if display_name:
                                update["display_name"] = display_name
                            avatar = account.get("avatar", {})
                            if avatar and avatar.get("locations"):
                                update["avatar_url"] = avatar["locations"][0].get("location")
                            await asyncio.to_thread(
                                lambda u=update: db.table("fans")
                                .update(u)
                                .eq("id", fan_id)
                                .execute()
                            )
                            print(f"[GROUP_ID] Found group_id={group_id} name={display_name} for fan={fan_id}")
                            return group_id
                if not cursor or not chats:
                    break
    except Exception as e:
        print(f"[GROUP_ID ERROR] {e}")
    return None


async def _resolve_active_chat_group_id(
    *,
    account_id: str,
    platform_fan_id: str,
    fan_id: str,
) -> tuple[str | None, int]:
    """Coalesce and throttle expensive chat-list scans for an unbound fan."""
    loop = asyncio.get_running_loop()
    now = loop.time()
    existing = _active_chat_binding_tasks.get(fan_id)
    retry_after = _active_chat_binding_retry_after.get(fan_id, 0.0)
    if existing is None and now < retry_after:
        return None, max(1, int(retry_after - now))

    owner = existing is None
    task = existing
    if task is None:
        _active_chat_binding_retry_after[fan_id] = (
            now + _ACTIVE_CHAT_BINDING_RETRY_SECONDS
        )
        task = asyncio.create_task(
            get_or_fetch_group_id(account_id, platform_fan_id, fan_id)
        )
        _active_chat_binding_tasks[fan_id] = task

    try:
        group_id = await task
    finally:
        if owner and _active_chat_binding_tasks.get(fan_id) is task:
            _active_chat_binding_tasks.pop(fan_id, None)

    if group_id:
        _active_chat_binding_retry_after.pop(fan_id, None)
        return str(group_id), 0
    remaining = _active_chat_binding_retry_after.get(fan_id, loop.time()) - loop.time()
    return None, max(1, int(remaining))


async def send_fansly_message(account_id: str, group_id: str, text: str) -> str | None:
    """Deliver plain text and return its durable platform identity.

    A 2xx response without a message ID is not sufficient evidence for an
    automated send.  Callers can reconcile an ambiguous acceptance instead of
    persisting an unbound local row or immediately sending a duplicate.
    """
    try:
        response_body = await send_apifansly_message(
            account_id,
            group_id,
            content=text,
        )
        platform_message_id = sent_message_id(response_body)
        if not platform_message_id:
            raise RuntimeError("platform accepted text but did not return a message ID")
        print(
            f"[SEND] account={account_id} group={group_id} accepted=true "
            f"message={platform_message_id}"
        )
        return platform_message_id
    except Exception as e:
        print(f"[SEND ERROR] {e}")
        return None


def inbound_message_dedupe_key(
    fan_id: str,
    platform_message_id: str,
    content: str,
    group_id: str,
) -> str:
    """One stable key per inbound platform event.

    The platform message id is the natural key and is present on every real
    delivery. The content-hash fallback keeps a rare id-less delivery idempotent
    across redeliveries of the same payload rather than silently creating a
    second obligation each time.
    """
    if platform_message_id:
        return f"inbound-message:{fan_id}:{platform_message_id}"
    digest = hashlib.sha256(
        f"{fan_id}|{group_id}|{content}".encode("utf-8", "replace")
    ).hexdigest()[:32]
    return f"inbound-message:{fan_id}:h{digest}"


async def accept_inbound_message(
    *,
    fan_id: str,
    creator_id: str,
    content: str,
    platform_message_id: str,
    group_id: str,
    api_account_id: str,
    creator_platform_id: str,
    auto_mode: bool,
    attachments: list[dict],
) -> "MessageWriteResult":
    """Persist a fan message and ensure exactly one processing obligation.

    This is the durable acceptance boundary shared by the webhook and the poller
    fallback. Both can deliver the same platform message, and two database
    constraints collapse that to one unit of work:

    * ``save_message_result`` upserts on ``(creator_id, fansly_message_id)``
      (REL-002), so one platform message is one row however a race resolves, and
      reports which caller actually inserted it.
    * the scheduled-action dedupe key, written with ``replace_existing=False``,
      makes one processing obligation per event. A redelivery must not reset an
      action that is already PENDING (it would run twice), PROCESSING (it would
      race an in-flight run), or COMPLETED (it would reprocess an answered
      message).

    The obligation is ensured even when ``inserted`` is False, and that is
    deliberate. Skipping it on a duplicate would reopen the exact window this
    boundary exists to close: a process killed between the message insert and
    this call would leave a persisted message that nothing is obliged to
    process, and every later redelivery would see ``inserted=False`` and decline
    to repair it. The insert-if-absent action write is what makes the crash
    recoverable; it is a no-op in every other case.

    Returns the message write result so the caller can report a redelivery
    without changing what was durably accepted.
    """
    media_context = (
        {
            "attachments": [
                {
                    "contentId": item.get("contentId"),
                    "type": item.get("contentType", 1),
                }
                for item in attachments
                if isinstance(item, dict) and item.get("contentId")
            ]
        }
        if attachments
        else None
    )
    write = await save_message_result(
        fan_id,
        creator_id,
        "fan",
        content,
        fansly_message_id=platform_message_id or None,
        media_context=media_context,
    )
    dedupe_key = inbound_message_dedupe_key(
        fan_id, platform_message_id, content, group_id
    )
    await schedule_action(
        creator_id=creator_id,
        fan_id=fan_id,
        action_type="PROCESS_INBOUND_MESSAGE",
        execute_at=datetime.now(timezone.utc),
        payload={
            "platform_message_id": platform_message_id or None,
            "message_row_id": write.message_id,
            "message_content": content,
            "group_id": group_id,
            "api_account_id": api_account_id,
            "creator_platform_id": creator_platform_id,
            "auto_mode": bool(auto_mode),
            "attachments": (media_context or {}).get("attachments", []),
            "received_at": datetime.now(timezone.utc).isoformat(),
        },
        dedupe_key=dedupe_key,
        replace_existing=False,
    )
    return write


def notify_scheduled_worker() -> None:
    """Ask the scheduled-actions loop to start its next cycle immediately."""
    try:
        from workers.scheduled_actions import notify_work_available

        notify_work_available()
    except Exception:  # pragma: no cover - never fail a durable ACK on a nudge
        pass


async def run_durable_inbound_message(action: dict) -> "object":
    """Process one accepted inbound message outside the webhook request.

    Called by the scheduled-actions worker for PROCESS_INBOUND_MESSAGE. Every
    slow dependency the webhook used to carry lives here: the API Fansly media
    lookup, the analyzer, the writer, and Auto scheduling. A failure retries with
    the queue's normal backoff instead of becoming a webhook timeout.
    """
    from workers.scheduled_actions import HandlerResult

    payload = action.get("payload") or {}
    fan_id = str(action["fan_id"])
    creator_id = str(action["creator_id"])
    message_content = str(payload.get("message_content") or "")
    mid = str(payload.get("platform_message_id") or "")
    group_id = str(payload.get("group_id") or "")
    api_account_id = str(payload.get("api_account_id") or "")
    creator_platform_id = str(payload.get("creator_platform_id") or "")
    message_row_id = payload.get("message_row_id")
    attachments = payload.get("attachments") or []

    # Media enrichment: the webhook knows attachment IDs but not signed
    # locations. Best effort — active-chat reconciliation is still the fallback,
    # exactly as before, and a failure here must not block the reply pipeline.
    if attachments and api_account_id and group_id and message_row_id:
        try:
            with action_stage("media_enrich_ms"):
                recent_messages, account_media, _ = await apifansly_list_chat_messages(
                    api_account_id,
                    group_id,
                    limit=10,
                )
                source_message = next(
                    (
                        item for item in recent_messages
                        if str(item.get("id") or "") == mid
                    ),
                    None,
                )
                resolved = (
                    _apifansly_message_row(
                        source_message,
                        fan_id=fan_id,
                        creator_id=creator_id,
                        creator_platform_id=creator_platform_id,
                        media_lookup=_apifansly_account_media_lookup(account_media),
                    )
                    if source_message
                    else None
                )
                if resolved and resolved.get("media_context"):
                    await update_message_media_context(
                        str(message_row_id), resolved["media_context"]
                    )
        except Exception as exc:
            print(
                f"[WEBHOOK MEDIA ENRICH] deferred fan={fan_id} "
                f"message={mid} error={type(exc).__name__}"
            )

    with action_stage("inbound_pipeline_ms"):
        await process_incoming_fan_message(
            fan_id,
            creator_id,
            message_content,
            bool(payload.get("auto_mode")),
            mid or None,
        )
    return HandlerResult(sent_message=False, reason="inbound message processed")


async def process_incoming_fan_message(
    fan_id: str,
    creator_id: str,
    message_content: str,
    auto_mode: bool,
    message_id: str | None,
) -> None:
    """Shared pipeline: history already includes the new fan message."""
    try:
        from services.ppv_delivery import cancel_pending_ppv_approvals

        cancelled = await cancel_pending_ppv_approvals(
            fan_id,
            reason="fan_returned_before_operator_approval",
        )
        if cancelled:
            print(f"[PPV APPROVAL] fan={fan_id} cancelled={cancelled} reason=fan_returned")
    except Exception as exc:
        # Approval-table availability must not block inbound conversation during
        # a rolling migration. The prepared PPV still cannot be sent by this path.
        print(f"[PPV APPROVAL CANCEL ERROR] fan={fan_id}: {exc}")
    try:
        from services.commercial_orchestrator import acknowledge_fan_return

        await acknowledge_fan_return(creator_id, fan_id)
    except Exception as exc:
        # Conversation can continue. The worker also revalidates recent fan
        # activity before any proactive message is sent.
        print(f"[OFFER FOLLOWUP CANCEL ERROR] fan={fan_id}: {exc}")
    conversation_history = await get_conversation_history(fan_id)
    fan_profile = await get_fan_by_id(fan_id)
    if fan_profile is None:
        fan_profile = Fan(id=fan_id, display_name=fan_id)

    spawn(
        learn_from_fan_message(
            creator_id=creator_id,
            fan_id=fan_id,
            fan_message=message_content,
            source_message_id=message_id,
            conversation_history=conversation_history,
        ),
        name=f"fan_intelligence:{fan_id}",
    )
    audience_row, memberships, auto_availability = await asyncio.gather(
        asyncio.to_thread(
            lambda: get_supabase()
            .from_("creators")
            .select("auto_audience_policy")
            .eq("id", creator_id)
            .single()
            .execute()
        ),
        asyncio.to_thread(
        lambda: get_supabase()
        .from_("fan_list_members")
        .select("list_id, fan_lists(exclude_from_auto)")
        .eq("fan_id", fan_id)
        .execute()
        ),
        _creator_auto_availability(creator_id),
    )
    from services.auto_audience import AutoAudiencePolicy, evaluate_auto_eligibility

    try:
        audience_policy = AutoAudiencePolicy(
            **((audience_row.data or {}).get("auto_audience_policy") or {})
        )
    except Exception:
        audience_policy = AutoAudiencePolicy()
    fan_list_ids = {
        str(row.get("list_id"))
        for row in (memberships.data or [])
        if row.get("list_id")
    }
    legacy_excluded_ids = {
        str(row.get("list_id"))
        for row in (memberships.data or [])
        if row.get("list_id") and row.get("fan_lists", {}).get("exclude_from_auto", False)
    }
    if legacy_excluded_ids:
        audience_policy.exclude_list_ids = list(
            dict.fromkeys([*audience_policy.exclude_list_ids, *legacy_excluded_ids])
        )
    is_new_fan = not any(message.role == "creator" for message in conversation_history)
    eligibility = evaluate_auto_eligibility(
        creator_auto=bool(auto_mode),
        fan_auto_override=fan_profile.auto_mode,
        needs_human_review=bool(getattr(fan_profile, "needs_human_review", False)),
        policy=audience_policy,
        fan_list_ids=fan_list_ids,
        total_spent=int(getattr(fan_profile, "total_spent", 0) or 0),
        spend_tier=str(getattr(fan_profile, "spend_tier", "cold") or "cold"),
        is_new_fan=is_new_fan,
    )
    effective_auto = (
        eligibility.eligible
        and bool(auto_availability.get("auto_available"))
    )
    effective_reason = (
        eligibility.reason
        if auto_availability.get("auto_available")
        else "no_approved_sets"
    )

    print(
        f"[AUTO MODE] creator={creator_id} creator_auto={auto_mode} "
        f"fan_auto={fan_profile.auto_mode} effective_auto={effective_auto} "
        f"reason={effective_reason} fan={fan_id}"
    )

    if effective_auto:
        await schedule_auto_reply(
            fan_id,
            creator_id,
            conversation_history=conversation_history,
            source_message_id=message_id,
        )
        fan_msg_count = len([m for m in conversation_history if m.role == "fan"])
        print(
            f"[MEMORY CHECK] fan={fan_id} fan_messages={fan_msg_count} "
            f"should_update={_should_update_memory(conversation_history)}"
        )
        if _should_update_memory(conversation_history):
            spawn(_update_fan_memory(fan_id, creator_id, conversation_history, fan_profile.total_spent), name="update_fan_memory")
            spawn(_update_fan_ai_summary(fan_id, conversation_history), name="update_fan_ai_summary")
        return

    fan_intelligence = await get_fan_intelligence_context(fan_id)
    creator_persona = await get_creator_persona(creator_id)
    if creator_persona is None:
        creator_persona = Persona()
    ppv_offers = await get_ppv_offers(creator_id)

    conversation_stage = classify_stage(conversation_history, fan_profile)
    similar_exchanges = await find_similar_exchanges(
        message_content, creator_id, enabled=False
    )

    ctx_without_situation = ConversationContext(
        fan_message=message_content,
        conversation_history=conversation_history,
        fan_profile=fan_profile,
        creator_persona=creator_persona,
        similar_exchanges=similar_exchanges,
        conversation_stage=conversation_stage,
        creator_name="a creator",
        ppv_offers=ppv_offers,
        fan_intelligence=fan_intelligence,
    )

    situation = await analyze_situation(ctx_without_situation)
    if fan_intelligence:
        situation["learned_fan_intelligence"] = fan_intelligence

    ctx = ConversationContext(
        fan_message=message_content,
        conversation_history=conversation_history,
        fan_profile=fan_profile,
        creator_persona=creator_persona,
        similar_exchanges=similar_exchanges,
        conversation_stage=conversation_stage,
        creator_name="a creator",
        situation=situation,
        ppv_offers=ppv_offers,
        fan_intelligence=fan_intelligence,
    )

    prompt = build_prompt(ctx)
    replies = await generate_replies(prompt, creator_persona)

    db = get_supabase()
    if message_id:
        await asyncio.to_thread(
            lambda: db.table("suggestions").insert({
                "fan_id": fan_id,
                "creator_id": creator_id,
                "fansly_message_id": message_id,
                "suggestions": replies,
                "stage": conversation_stage.value,
            }).execute()
        )

    fan_msg_count = len([m for m in conversation_history if m.role == "fan"])
    print(
        f"[MEMORY CHECK] fan={fan_id} fan_messages={fan_msg_count} "
        f"should_update={_should_update_memory(conversation_history)}"
    )
    if _should_update_memory(conversation_history):
        spawn(_update_fan_memory(fan_id, creator_id, conversation_history, fan_profile.total_spent), name="update_fan_memory")
        spawn(_update_fan_ai_summary(fan_id, conversation_history), name="update_fan_ai_summary")


class ReplyRequest(BaseModel):
    fan_id: str
    creator_id: str
    content: str
    was_ai_suggested: bool = False


class VaultMediaUrlsRequest(BaseModel):
    media_ids: list[str] = Field(default_factory=list, max_length=1000)


class VoiceCalibrationUpdateRequest(BaseModel):
    enabled: bool = False
    approved_message_ids: list[str] = Field(default_factory=list, max_length=30)


class WebhookPayload(BaseModel):
    type: str
    record: dict


class ConnectCreatorRequest(BaseModel):
    name: str
    email: str
    password: str
    user_id: str
    countryCode: str = "US"
    creator_id: str | None = None


class Connect2FARequest(BaseModel):
    twofa_token: str
    code: str
    name: str
    email: str
    password: str
    countryCode: str = "US"
    user_id: str = ""
    creator_id: str | None = None


async def handle_new_fan_message(account_id: str, group_id: str, message: dict):
    """
    Fires when the poller detects a new fan message.
    Acts as a fallback to the ApiFansly webhook — same pipeline,
    but skips any message already processed by the webhook.

    account_id  = Fansly account ID of the model
    group_id    = Fansly conversation group ID
    message     = raw Fansly message dict (id, senderId, content, createdAt, attachments, etc.)
    """
    message_id = str(message.get("id", ""))
    platform_fan_id = str(message.get("senderId", ""))
    content = (message.get("content") or "").strip()

    attachments = message.get("attachments") or []
    has_attachments = len(attachments) > 0

    if not platform_fan_id:
        return
    if not content and not has_attachments:
        return

    # Skip if already handled by the ApiFansly webhook
    if message_id and message_id in _processed_messages:
        print(f"[POLLER] Skipping already-processed message_id={message_id}")
        return

    # Register in dedup set to prevent webhook double-processing this same message
    if message_id:
        _processed_messages.add(message_id)

    print(
        f"[POLLER] New message model={account_id} fan={platform_fan_id} "
        f"message_id={message_id} content={content[:80]}"
    )

    db = get_supabase()

    # Look up creator by their Fansly account ID
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("id, auto_mode")
        .eq("fansly_account_id", account_id)
        .limit(1)
        .execute()
    )
    if not creator_row.data:
        print(f"[POLLER] Creator not found for fansly_account_id={account_id}")
        return

    creator_id = creator_row.data[0]["id"]
    auto_mode = creator_row.data[0].get("auto_mode", False)

    # Get or create fan
    fan = await get_fan(creator_id, platform_fan_id)
    if not fan:
        fan = await create_fan(creator_id, platform_fan_id, f"Fan_{platform_fan_id[-6:]}")
        spawn(_enrich_fan_profile(fan.id, creator_id, platform_fan_id), name="enrich_fan_profile")

    # Update group_id if not already stored
    if not fan.fansly_group_id and group_id:
        await asyncio.to_thread(
            lambda: db.table("fans")
            .update({"fansly_group_id": group_id})
            .eq("id", fan.id)
            .execute()
        )

    # Same durable acceptance as the webhook. The shared dedupe key is what makes
    # webhook-and-poller delivery of one message safe: whichever arrives second
    # finds the obligation already present and does nothing, instead of running a
    # second analyzer and writer pass over the same fan message.
    await accept_inbound_message(
        fan_id=str(fan.id),
        creator_id=str(creator_id),
        content=content,
        platform_message_id=message_id or "",
        group_id=str(group_id or ""),
        api_account_id="",
        creator_platform_id=str(account_id or ""),
        auto_mode=bool(auto_mode),
        attachments=[a for a in attachments if isinstance(a, dict)],
    )
    notify_scheduled_worker()


session_store: SessionStore = None
fansly_poller: FanslyPoller = None
ppv_sweep_task: asyncio.Task | None = None
vault_autosync_task: asyncio.Task | None = None
scheduled_actions_task: asyncio.Task | None = None
chat_reconcile_task: asyncio.Task | None = None
model_availability_task: asyncio.Task | None = None


async def ppv_sweep_scheduler():
    """Runs stale PPV verification sweep every 15 minutes."""
    while True:
        await asyncio.sleep(15 * 60)
        try:
            # The sweep itself only repairs durable rows; the provider call it
            # can lead to is refused by the transport and handled as an
            # unavailable verification, so it is safe to keep running.
            print("[CRON] Running PPV sweep...")
            from services.suggestions import sweep_stale_ppv_checks

            await sweep_stale_ppv_checks()
        except Exception as e:
            print(f"[CRON PPV SWEEP ERROR] {e}")


async def _scheduled_actions_scheduler():
    """Runs the commercial scheduled-actions queue (payday re-engagement, etc).

    The loop itself lives in the worker module so its polling, backlog and
    repair-cadence behaviour is testable without starting FastAPI. It replaces
    the old flat 60-second sleep, which drained at most one batch per minute no
    matter how deep the queue was.
    """
    from workers.scheduled_actions import scheduled_actions_loop

    try:
        await scheduled_actions_loop()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"[CRON SCHEDULED ACTIONS ERROR] {e}")


async def vault_autosync_scheduler():
    """Check hourly and incrementally sync each connected creator once per day.

    ``_run_vault_sync`` owns the exact IDs imported by that run and, when the
    creator opted in, categorizes only those IDs. The scheduler must never scan
    every uncategorized record after a sync because that can accidentally rerun
    the initial-vault job.
    """
    while True:
        try:
            if not apifansly_enabled():
                # Intentionally offline. The startup line already said so, and a
                # per-cycle message every hour would be noise, not information.
                await asyncio.sleep(_VAULT_AUTOSYNC_CHECK_SECONDS)
                continue
            print("[CRON] Vault auto-sync pass...")
            db = get_supabase()
            # The read that opens the cycle. Losing it to a connection recycle
            # used to cost the whole hourly pass.
            creators = await retry_db_read(
                lambda: db.table("creators")
                .select("id, last_vault_sync_at")
                .not_.is_("apifansly_account_id", "null")
                .execute(),
                label="vault_autosync.creators",
            )
            for c in (creators.data or []):
                cid = str(c["id"])
                if _vault_sync_retry_after.get(cid, 0) > time.time():
                    continue
                cd = vault_sync_cooldown(
                    c.get("last_vault_sync_at"),
                    interval_hours=_VAULT_SYNC_INTERVAL_HOURS,
                )
                if not cd["allowed"]:
                    continue
                try:
                    res = await sync_vault_start(cid)
                    if res.get("status") == "started":
                        print(f"[CRON] auto-sync started creator={cid}")
                except Exception as exc:
                    print(f"[CRON VAULT CREATOR ERROR] creator={cid}: {exc}")
        except Exception as e:
            print(f"[CRON VAULT AUTOSYNC ERROR] {e}")
        await asyncio.sleep(_VAULT_AUTOSYNC_CHECK_SECONDS)


_chat_reconcile_denied_bindings: set[tuple[str, str]] = set()
_chat_reconcile_due_at: dict[str, float] = {}


def _chat_reconcile_interval_seconds(
    *,
    creator_auto_mode: bool,
    has_auto_fan: bool,
) -> int:
    env_name = (
        "CHAT_RECONCILE_ACTIVE_MINUTES"
        if creator_auto_mode or has_auto_fan
        else "CHAT_RECONCILE_IDLE_MINUTES"
    )
    default = 10 if env_name.endswith("ACTIVE_MINUTES") else 30
    try:
        minutes = int(os.environ.get(env_name, str(default)))
    except ValueError:
        minutes = default
    return min(max(minutes, 5), 120) * 60


def _chat_message_sync_needed(
    platform_last_message_id: str,
    stored_checkpoint: str,
    *,
    is_new_chat: bool,
    group_binding_changed: bool,
) -> bool:
    """Fetch messages only when the chat's remote last-message marker moved.

    API-001. The checkpoint is `fans.chat_last_message_id`, written at the end
    of the last successful reconciliation of this chat and read back from the
    fans page sync_chats already loads. Because it is durable, a process restart
    no longer turns every known conversation into a cold sync — which at 20
    creators x ~2,000 chats was ~40,000 provider calls per deploy.

    Every ambiguous case resolves toward reconciling. A call this returns False
    for is one where the remote marker is byte-identical to the marker we stored
    after successfully syncing that same chat, so suppressing it cannot lose a
    message:

      * new chat                  -> sync (nothing has ever been imported)
      * rebound group             -> sync (the checkpoint belongs to the old
                                     conversation and means nothing here)
      * no checkpoint stored      -> sync (never synced, or pre-migration row)
      * platform reports no marker-> sync (cannot prove nothing changed)
      * marker differs            -> sync (including a deleted newest message,
                                     which moves lastMessageId)
      * marker identical          -> skip
    """
    if is_new_chat or group_binding_changed:
        return True
    if not stored_checkpoint:
        return True
    if not platform_last_message_id:
        return True
    return platform_last_message_id != stored_checkpoint


async def _reconcile_chat_creators_once(
    creators: list[dict],
) -> dict[str, int]:
    """Reconcile every usable creator without one stale binding stopping all.

    A 409 means the API key cannot access that exact API Fansly connection ID.
    Retrying it every ten minutes wastes credits and hides healthy creators'
    work in error noise. The binding is suppressed only for this process and
    exact ID; reconnecting the creator writes a new ID which is picked up on the
    next pass without requiring a restart.
    """
    processed = 0
    skipped = 0
    failed = 0
    for creator in creators:
        creator_id = str(creator.get("id") or "")
        account_id = str(creator.get("apifansly_account_id") or "")
        if not creator_id or not account_id:
            continue
        binding = (creator_id, account_id)
        for stale in list(_chat_reconcile_denied_bindings):
            if stale[0] == creator_id and stale != binding:
                _chat_reconcile_denied_bindings.discard(stale)
        if binding in _chat_reconcile_denied_bindings:
            skipped += 1
            continue
        try:
            result = await sync_chats(creator_id, incremental=True)
            processed += 1
            if result.get("status") == "ok" and result.get("new_chats"):
                print(
                    f"[CHAT RECONCILE] creator={creator_id} "
                    f"new={result['new_chats']} "
                    f"synced={result.get('synced', 0)}"
                )
        except HTTPException as exc:
            failed += 1
            if exc.status_code == 409:
                _chat_reconcile_denied_bindings.add(binding)
                print(
                    f"[CHAT RECONCILE PAUSED] creator={creator_id} "
                    "reason=stored_apifansly_binding_inaccessible "
                    "action=reconnect_creator"
                )
            else:
                print(
                    f"[CHAT RECONCILE CREATOR ERROR] creator={creator_id} "
                    f"status={exc.status_code} detail={exc.detail}"
                )
        except Exception as exc:
            failed += 1
            print(
                f"[CHAT RECONCILE CREATOR ERROR] creator={creator_id}: {exc}"
            )
    return {
        "processed": processed,
        "skipped_inaccessible": skipped,
        "failed": failed,
    }


async def chat_reconciliation_scheduler():
    """Reconcile active creators more often than idle creators."""
    while True:
        try:
            tick_minutes = int(
                os.environ.get("CHAT_RECONCILE_TICK_MINUTES", "5")
            )
        except ValueError:
            tick_minutes = 5
        await asyncio.sleep(min(max(tick_minutes, 1), 10) * 60)
        if not apifansly_enabled():
            # Every branch below exists to make provider calls. Skipping the
            # tick entirely is what keeps the logs clean: the alternative is one
            # 402 per creator per cycle, which is exactly the noise this switch
            # was added to remove.
            continue
        try:
            import time

            db = get_supabase()
            creators_result = await retry_db_read(
                lambda: db.table("creators")
                .select("id, apifansly_account_id, auto_mode")
                .not_.is_("apifansly_account_id", "null")
                .execute(),
                label="chat_reconcile.creators",
            )

            # This used to read every auto-mode fan in the DEPLOYMENT — no
            # creator filter — and build a set of creator ids from it. PostgREST
            # capped that globally at 1,000 rows, so once enough fans had auto
            # mode on anywhere, creators whose fans fell past the cap were
            # misread as having none and were dropped to the 30-minute idle
            # reconcile interval instead of 10.
            #
            # The question is only ever "does this creator have at least one",
            # so ask it that way. One bounded existence check, and only for a
            # creator that is due and whose own auto_mode has not already
            # answered it.
            async def _has_auto_fan(creator_id: str) -> bool:
                probe = await asyncio.to_thread(
                    lambda cid=creator_id: db.table("fans")
                    .select("id")
                    .eq("creator_id", cid)
                    .eq("auto_mode", True)
                    .limit(1)
                    .execute()
                )
                return bool(probe.data)

            now = time.monotonic()
            due: list[dict] = []
            active_creator_ids: set[str] = set()
            for creator in creators_result.data or []:
                creator_id = str(creator.get("id") or "")
                if not creator_id:
                    continue
                active_creator_ids.add(creator_id)
                if now < _chat_reconcile_due_at.get(creator_id, 0):
                    continue
                due.append(creator)
                creator_auto_mode = bool(creator.get("auto_mode"))
                _chat_reconcile_due_at[creator_id] = now + (
                    _chat_reconcile_interval_seconds(
                        creator_auto_mode=creator_auto_mode,
                        # Creator-level auto mode already selects the active
                        # interval, so the probe is skipped entirely there.
                        has_auto_fan=(
                            False
                            if creator_auto_mode
                            else await _has_auto_fan(creator_id)
                        ),
                    )
                )
            for creator_id in list(_chat_reconcile_due_at):
                if creator_id not in active_creator_ids:
                    _chat_reconcile_due_at.pop(creator_id, None)

            # REL-004 — retry maps are cleaned on work the process is already
            # doing, rather than by a background task whose only job is a few
            # dicts.
            #
            # _active_chat_binding_retry_after only ever had entries REMOVED on
            # success, so a fan whose binding never resolved — deleted, or
            # permanently unresolvable — stayed in it for the life of the
            # process. Dropping entries whose backoff has already elapsed is
            # both the cleanup and a no-op semantically: an expired deadline is
            # exactly the state "no backoff applies", which is what an absent
            # entry means. The map is therefore bounded by the number of fans in
            # backoff AT ONCE rather than by the number ever seen.
            loop_now = asyncio.get_running_loop().time()
            expired_bindings = prune_expired(
                _active_chat_binding_retry_after,
                loop_now,
                # A fan with an in-flight resolution task keeps its entry: there
                # the entry is also what coalesces concurrent callers onto one
                # scan, which outlives the backoff deadline.
                protect=set(_active_chat_binding_tasks),
            )
            expired_vault = prune_expired(
                _vault_sync_retry_after,
                time.time(),
                keep=active_creator_ids,
            )
            if expired_bindings or expired_vault:
                print(
                    f"[STATE PRUNE] chat_bindings={expired_bindings} "
                    f"vault_backoff={expired_vault} "
                    f"processed_messages={len(_processed_messages)}"
                )

            if due:
                await _reconcile_chat_creators_once(due)
        except Exception as exc:
            print(f"[CRON CHAT RECONCILE INFRA ERROR] {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_store, fansly_poller, ppv_sweep_task, vault_autosync_task, scheduled_actions_task, chat_reconcile_task, model_availability_task

    # SEC-004: state the resolved deployment mode once at boot. An unset or
    # unrecognised APP_ENV resolves to production, so a misconfigured deploy is
    # visible in the logs instead of silently running with relaxed auth.
    from core.environment import describe_environment

    print(f"[STARTUP] {describe_environment()}")
    # One line, once. Everything downstream suppresses its own work silently,
    # so this is the only place the disabled connector is announced.
    print(describe_apifansly())

    # Every Supabase call in this process runs on the event loop's default
    # executor via asyncio.to_thread. Left unconfigured that pool is sized
    # min(32, cpu_count + 4) — eight threads on a 4-vCPU container, shared by
    # the worker, the schedulers and every inbound webhook. Make the ceiling
    # explicit and visible rather than an accident of the container size.
    from core import db_executor

    db_executor.install(asyncio.get_running_loop())
    # The Supabase transport is a deployment-wide decision (HTTP/2 off, pool
    # size, timeouts) that used to be invisible defaults inside supabase-py.
    print(f"[BOOT] {supabase_transport_description()}")
    print(f"[STARTUP] {db_executor.describe()}")

    supabase = get_supabase()
    session_store = SessionStore(
        supabase=supabase,
        encryption_key=os.environ["FANSLY_SESSION_KEY"],
    )
    await session_store.load_all()

    fansly_poller = FanslyPoller(
        session_store=session_store,
        on_new_message=handle_new_fan_message,
    )
    await fansly_poller.start_all()
    ppv_sweep_task = asyncio.create_task(ppv_sweep_scheduler())
    vault_autosync_task = asyncio.create_task(vault_autosync_scheduler())
    scheduled_actions_task = asyncio.create_task(_scheduled_actions_scheduler())
    chat_reconcile_task = asyncio.create_task(chat_reconciliation_scheduler())
    model_availability_task = asyncio.create_task(model_availability_scheduler())

    yield

    if fansly_poller:
        await fansly_poller.stop_all()
    if ppv_sweep_task:
        ppv_sweep_task.cancel()
    if vault_autosync_task:
        vault_autosync_task.cancel()
    if scheduled_actions_task:
        scheduled_actions_task.cancel()
    if chat_reconcile_task:
        chat_reconcile_task.cancel()
    if model_availability_task:
        model_availability_task.cancel()

    # PERF-006 — the API Fansly connection pool is process-wide, so shutdown is
    # the only place that closes it. Sockets are released here rather than at
    # the end of every individual call.
    await close_apifansly_client()

    # Same reasoning for the PostgREST pool: process-wide, so shutdown is the
    # only place that releases its sockets.
    close_supabase_client()

    # The database pool's threads are not daemons, so a lingering pool would
    # keep the process alive after the event loop stops.
    db_executor.shutdown()


app = FastAPI(lifespan=lifespan)

# CORS: restrict to known dashboard origins instead of a blanket wildcard.
# - localhost / 127.0.0.1 (any port) and Vercel deploys are always allowed, so
#   local dev and *.vercel.app dashboards work with no extra config.
# - For a custom production domain, set CORS_ALLOW_ORIGINS in the environment
#   (comma-separated, e.g. "https://app.example.com,https://www.example.com").
_cors_origins = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()
]

# --- API authentication ---------------------------------------------------
# Path-based policy so we don't have to touch ~33 route decorators:
#   • OPTIONS (CORS preflight) and health/root  -> always open
#   • API Fansly webhook                        -> verify its HMAC signature in-route
#   • internal database webhook                 -> require WEBHOOK_SECRET header
#   • everything else (operator/CRUD/admin)     -> require DASHBOARD_API_SECRET
# Unconfigured secrets fail open only under an explicit APP_ENV=development,
# and closed everywhere else including an unset APP_ENV (see core/environment.py).
from starlette.responses import JSONResponse
from core.auth import (
    _is_dev,
    _consteq,
    authenticated_dashboard_user,
    dashboard_user_id,
)

_PUBLIC_PATHS = {"/health", "/health/ready", "/"}
_WEBHOOK_PATHS = {"/generate-suggestions"}
_SIGNED_WEBHOOK_PATHS = {"/webhook/fansly"}


@app.middleware("http")
async def api_auth_middleware(request, call_next):
    path = request.url.path
    if request.method == "OPTIONS" or path in _PUBLIC_PATHS:
        return await call_next(request)

    # Signature verification needs the untouched raw request body, so the API
    # Fansly route authenticates itself. Do not require a custom header that the
    # provider does not send.
    if path in _SIGNED_WEBHOOK_PATHS:
        return await call_next(request)

    if path in _WEBHOOK_PATHS:
        expected = os.environ.get("WEBHOOK_SECRET")
        supplied = request.headers.get("x-webhook-secret")
    else:
        expected = os.environ.get("DASHBOARD_API_SECRET")
        supplied = request.headers.get("x-api-key")

    if not expected:
        if _is_dev():
            return await call_next(request)
        return JSONResponse({"detail": "Server auth is not configured"}, status_code=500)

    if not supplied or not _consteq(supplied, expected):
        return JSONResponse({"detail": "Missing or invalid credentials"}, status_code=401)

    if path not in _WEBHOOK_PATHS:
        try:
            request.state.dashboard_user_id = await authenticated_dashboard_user(
                request.headers.get("authorization")
            )
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    return await call_next(request)


# Keep CORS outside the authentication middleware so browser clients can read
# authentication failures. Otherwise a rejected credential looks like a generic
# network error because the early 401 response has no Access-Control-Allow-Origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://[a-z0-9-]+\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# --- Intentional connector disablement -------------------------------------
#
# An endpoint whose real purpose is remote platform delivery or a fresh remote
# fetch cannot pretend to succeed while APIFANSLY_ENABLED=false. It answers 503
# with a stable machine-readable reason instead.
#
# The body keeps ``detail`` as a human-readable STRING because the dashboard
# already surfaces ``body.detail`` directly in error toasts; a dict there would
# render as "[object Object]". The machine-readable code travels alongside it in
# ``reason``, so both audiences are served without changing any existing
# response shape.


class ConnectorDisabled(Exception):
    """Raised by the route guard below; rendered by the handler beneath it."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@app.exception_handler(ConnectorDisabled)
async def _connector_disabled_handler(_request: Request, exc: ConnectorDisabled):
    return JSONResponse(
        status_code=503,
        content={
            "status": "connector_disabled",
            "reason": REASON_DISABLED,
            "detail": exc.detail,
        },
    )


def require_apifansly_connector() -> None:
    """Route dependency for actions that genuinely need the remote platform.

    Used only where success would otherwise be a lie — a live send, or a fetch
    whose whole purpose is fresh remote data. Read-only local endpoints keep
    working, because the product stays usable while Fansly is switched off.
    """
    if not apifansly_enabled():
        raise ConnectorDisabled(
            "The API Fansly connector is disabled for this deployment, so "
            "nothing was sent or fetched. Set APIFANSLY_ENABLED=true to "
            "restore live platform access."
        )


@app.post("/suggestions", response_model=SuggestionResponse)
async def suggestions(req: SuggestionRequest, request: Request) -> SuggestionResponse:
    await require_creator_fan_access(request, req.creator_id, req.fan_id)
    return await get_suggestions(
        fan_id=req.fan_id,
        creator_id=req.creator_id,
        fan_message=req.message,
        creator_name="a creator",
    )


@app.post("/regenerate-suggestions", response_model=SuggestionResponse)
async def regenerate_suggestions(
    req: SuggestionRequest,
    request: Request,
) -> SuggestionResponse:
    await require_creator_fan_access(request, req.creator_id, req.fan_id)
    result = await get_suggestions(
        fan_id=req.fan_id,
        creator_id=req.creator_id,
        fan_message=req.message,
        creator_name="a creator",
        save_fan_message=False,
    )

    if result.suggestions:
        db = get_supabase()
        await asyncio.to_thread(
            lambda: db.table("suggestions").insert({
                "fan_id": req.fan_id,
                "creator_id": req.creator_id,
                "suggestions": result.suggestions,
                "stage": result.stage.value,
            }).execute()
        )

    return result


@app.post("/reply", dependencies=[Depends(require_apifansly_connector)])
async def save_reply(req: ReplyRequest, request: Request) -> dict:
    await require_creator_fan_access(request, req.creator_id, req.fan_id)

    db = get_supabase()
    fan_row, creator_row = await asyncio.gather(
        asyncio.to_thread(
            lambda: db.table("fans")
            .select("fansly_group_id")
            .eq("id", req.fan_id).single().execute()
        ),
        asyncio.to_thread(
            lambda: db.table("creators")
            .select("apifansly_account_id")
            .eq("id", req.creator_id).single().execute()
        ),
    )

    group_id = (fan_row.data or {}).get("fansly_group_id")
    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")

    if not group_id or not apifansly_id:
        raise HTTPException(status_code=409, detail="No live delivery route for this fan")
    try:
        response_body = await send_apifansly_message(
            str(apifansly_id),
            str(group_id),
            content=req.content,
        )
    except Exception as exc:
        print(f"[SEND ERROR] {exc}")
        raise HTTPException(
            status_code=502,
            detail="Fansly did not accept the message",
        ) from exc
    platform_message_id = sent_message_id(response_body)

    message_id = await save_message(
        req.fan_id,
        req.creator_id,
        "creator",
        req.content,
        req.was_ai_suggested,
        fansly_message_id=platform_message_id,
    )

    return {"status": "ok", "message_id": message_id}


@app.post(
    "/connect-creator",
    dependencies=[Depends(require_apifansly_connector)],
)
async def connect_creator(req: ConnectCreatorRequest, request: Request) -> dict:
    if req.creator_id:
        await require_creator_access(request, req.creator_id)
    operator_id = dashboard_user_id(request) or req.user_id
    if not operator_id:
        raise HTTPException(status_code=401, detail="Missing dashboard user session")
    print(
        f"[CONNECT] creator_id={req.creator_id or 'new'} "
        f"name={req.name} country={req.countryCode}"
    )

    async with apifansly_client_scope() as client:
        response = await client.post(
            apifansly_url("connect"),
            headers=apifansly_headers(json_content=True),
            json={
                "username": req.email,
                "password": req.password,
                "name": req.name,
                "countryCode": req.countryCode,
            },
            timeout=30,
        )
        if not response.is_success:
            return {
                "success": False,
                "error": f"API Fansly connection failed: {apifansly_response_message(response)}",
            }
        data = response.json()
        print(
            f"[CONNECT] status={response.status_code} "
            f"requires_2fa={bool(data.get('data', {}).get('requires_2fa'))}"
        )

        if data.get("data", {}).get("requires_2fa"):
            return {
                "requires_2fa": True,
                "twofa_token": data["data"].get("twofa_token", ""),
                "masked_email": data["data"].get("masked_email", ""),
                "message": data["data"].get("message", ""),
            }

        apifansly_account_id = data.get("data", {}).get("account_id")
        fansly_account_id = data.get("data", {}).get("data", {}).get("response", {}).get("accountId")

        if not apifansly_account_id:
            return {"success": False, "error": "Failed to connect account"}

        db = get_supabase()
        creator_values = {
            "platform_username": req.name,
            "platform": "fansly",
            "fansly_account_id": str(fansly_account_id),
            "apifansly_account_id": apifansly_account_id,
        }
        if req.creator_id:
            creator_row = await asyncio.to_thread(
                lambda: db.table("creators")
                .update(creator_values)
                .eq("id", req.creator_id)
                .execute()
            )
        else:
            creator_row = await asyncio.to_thread(
                lambda: db.table("creators").insert({
                    **creator_values,
                    "auto_mode": False,
                }).execute()
            )

        creator = creator_row.data[0] if creator_row.data else None
        if not creator:
            return {"success": False, "error": "Failed to save creator connection"}

        if not req.creator_id:
            await asyncio.to_thread(
                lambda: db.table("chatter_creators").insert({
                    "chatter_id": operator_id,
                    "creator_id": creator["id"],
                }).execute()
            )

        spawn(sync_chats_background(creator["id"]), name="sync_chats_background")

        return {
            "success": True,
            "creator": creator,
            "reconnected": bool(req.creator_id),
        }


@app.post(
    "/connect-creator-2fa",
    dependencies=[Depends(require_apifansly_connector)],
)
async def connect_creator_2fa(req: Connect2FARequest, request: Request) -> dict:
    if req.creator_id:
        await require_creator_access(request, req.creator_id)
    operator_id = dashboard_user_id(request) or req.user_id
    if not operator_id:
        raise HTTPException(status_code=401, detail="Missing dashboard user session")

    async with apifansly_client_scope() as client:
        response = await client.post(
            apifansly_url("verify-2fa"),
            headers=apifansly_headers(json_content=True),
            json={
                "username": req.email,
                "password": req.password,
                "name": req.name,
                "twoFactorToken": req.twofa_token,
                "twoFactorCode": req.code,
                "countryCode": req.countryCode,
            },
            timeout=30,
        )
        if not response.is_success:
            return {
                "success": False,
                "error": f"API Fansly 2FA verification failed: {apifansly_response_message(response)}",
            }
        data = response.json()
        print(f"[2FA] status={response.status_code} creator_id={req.creator_id or 'new'}")

        apifansly_account_id = data.get("data", {}).get("account_id")
        fansly_account_id = data.get("data", {}).get("data", {}).get("response", {}).get("accountId")

        if not apifansly_account_id:
            return {"success": False, "error": "2FA verification failed"}

        db = get_supabase()
        creator_values = {
            "platform_username": req.name,
            "platform": "fansly",
            "fansly_account_id": str(fansly_account_id),
            "apifansly_account_id": apifansly_account_id,
        }
        if req.creator_id:
            creator_row = await asyncio.to_thread(
                lambda: db.table("creators")
                .update(creator_values)
                .eq("id", req.creator_id)
                .execute()
            )
        else:
            creator_row = await asyncio.to_thread(
                lambda: db.table("creators").insert({
                    **creator_values,
                    "auto_mode": False,
                }).execute()
            )

        creator = creator_row.data[0] if creator_row.data else None
        if not creator:
            return {"success": False, "error": "Failed to save creator connection"}

        if not req.creator_id:
            await asyncio.to_thread(
                lambda: db.table("chatter_creators").insert({
                    "chatter_id": operator_id,
                    "creator_id": creator["id"],
                }).execute()
            )

        spawn(sync_chats_background(creator["id"]), name="sync_chats_background")

        return {
            "success": True,
            "creator": creator,
            "reconnected": bool(req.creator_id),
        }


async def sync_chats_background(creator_id: str) -> None:
    try:
        await sync_chats(creator_id)
        print(f"[SYNC] Chats synced for creator={creator_id}")
    except Exception as e:
        print(f"[SYNC ERROR] {e}")


def _apifansly_account_media_lookup(account_media: list[dict]) -> dict[str, dict]:
    def first_location(value: object) -> str | None:
        if isinstance(value, dict):
            direct = value.get("location")
            if isinstance(direct, str) and direct.startswith("https://"):
                return direct
            for key in ("locations", "variants", "media", "preview"):
                found = first_location(value.get(key))
                if found:
                    return found
        elif isinstance(value, list):
            for nested in value:
                found = first_location(nested)
                if found:
                    return found
        return None

    lookup: dict[str, dict] = {}
    for item in account_media:
        if not isinstance(item, dict):
            continue
        media = item.get("media") or {}
        media_url = first_location(media) or first_location(item)
        prices = account_media_prices(item)
        positive_prices = [price for price in prices if price > 0]
        raw_price = positive_prices[0] if positive_prices else (prices[0] if prices else 0)
        info = {
            "url": media_url,
            "price": raw_price,
            "is_ppv": bool(positive_prices),
            "purchased": bool(
                item.get("purchased", item.get("isPurchased", False))
            ),
            "access": item.get("access"),
            "mimetype": (
                media.get("mimetype")
                or media.get("mimeType")
                or item.get("mimetype")
                or item.get("mimeType")
            ),
            "filename": (
                media.get("filename")
                or media.get("fileName")
                or item.get("filename")
                or item.get("fileName")
            ),
        }
        for key in (item.get("id"), item.get("mediaId")):
            if key:
                lookup[str(key)] = info
    return lookup


def _apifansly_message_row(
    message: dict,
    *,
    fan_id: str,
    creator_id: str,
    creator_platform_id: str,
    media_lookup: dict[str, dict],
) -> dict | None:
    message_id = str(message.get("id") or "")
    if not message_id:
        return None
    content = str(message.get("content") or "")
    attachments = message.get("attachments") or []
    if not content and not attachments:
        return None

    created_at = message.get("createdAt")
    try:
        timestamp = float(created_at or 0)
    except (TypeError, ValueError):
        timestamp = 0
    if timestamp > 0:
        if timestamp > 1e12:
            timestamp /= 1000
        sent_at = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
    else:
        sent_at = datetime.now(timezone.utc).isoformat()

    resolved_attachments = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        content_id = str(attachment.get("contentId") or "")
        info = media_lookup.get(content_id) or {}
        resolved_attachments.append({
            "contentId": content_id,
            "url": info.get("url"),
            "type": attachment.get("contentType", 1),
            "mimetype": info.get("mimetype"),
            "filename": info.get("filename"),
            "price": info.get("price"),
            "is_ppv": info.get("is_ppv"),
            "purchased": info.get("purchased"),
            "access": info.get("access"),
        })

    sender_id = str(message.get("senderId") or "")
    return {
        "fan_id": fan_id,
        "creator_id": creator_id,
        "role": "creator" if sender_id == creator_platform_id else "fan",
        "content": content,
        "fansly_message_id": message_id,
        "sent_at": sent_at,
        "media_context": (
            {"attachments": resolved_attachments}
            if resolved_attachments
            else None
        ),
    }


def _matching_unbound_creator_message(
    platform_row: dict,
    candidates: list[dict],
    *,
    used_ids: set[str] | None = None,
    tolerance_seconds: float = 15.0,
) -> dict | None:
    """Match a just-sent local row that is missing its Fansly identity."""
    if platform_row.get("role") != "creator":
        return None
    content = " ".join(str(platform_row.get("content") or "").split())
    if not content:
        return None
    try:
        platform_time = datetime.fromisoformat(
            str(platform_row.get("sent_at") or "").replace("Z", "+00:00")
        )
    except ValueError:
        return None

    excluded = used_ids or set()
    matches: list[tuple[float, dict]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        if (
            not candidate_id
            or candidate_id in excluded
            or candidate.get("role") != "creator"
            or candidate.get("fansly_message_id")
            or " ".join(str(candidate.get("content") or "").split()) != content
        ):
            continue
        try:
            candidate_time = datetime.fromisoformat(
                str(candidate.get("sent_at") or "").replace("Z", "+00:00")
            )
        except ValueError:
            continue
        try:
            delta = abs((candidate_time - platform_time).total_seconds())
        except TypeError:
            continue
        if delta <= tolerance_seconds:
            matches.append((delta, candidate))
    return min(matches, key=lambda match: match[0])[1] if matches else None


async def _sync_recent_fan_messages(
    *,
    creator_id: str,
    fan_id: str,
    account_id: str,
    creator_platform_id: str,
    group_id: str,
    creator_auto_mode: bool,
    client=None,
) -> dict:
    """Import the latest API Fansly page and process only newly seen fan text."""
    db = get_supabase()
    messages, account_media, _ = await apifansly_list_chat_messages(
        account_id,
        group_id,
        limit=10,
        client=client,
    )
    # API-001 — the reconciliation checkpoint is NOT written here. It has to be
    # the marker list_chats reports, because that is what the next pass compares
    # against; the newest id on this page is a different value from a different
    # endpoint and the two would never compare equal. sync_chats writes it after
    # this call returns. Callers that reach this function outside a chat-list
    # pass (an Auto reply reconciling its own fan) therefore leave the
    # checkpoint alone, which costs at most one extra call on the next pass and
    # can never suppress one.
    message_ids = [
        str(message.get("id"))
        for message in messages
        if message.get("id")
    ]
    if not message_ids:
        return {"imported": 0, "inbound": 0, "media_updated": 0}

    async def _load_existing_message_rows():
        return await asyncio.gather(
            asyncio.to_thread(
                lambda: db.table("messages")
                .select("id, fansly_message_id, role, media_context")
                .eq("fan_id", fan_id)
                .in_("fansly_message_id", message_ids)
                .execute()
            ),
            asyncio.to_thread(
                lambda: db.table("messages")
                .select(
                    "id, fansly_message_id, role, content, sent_at, "
                    "media_context"
                )
                .eq("fan_id", fan_id)
                .eq("creator_id", creator_id)
                .eq("role", "creator")
                .is_("fansly_message_id", "null")
                .order("sent_at", desc=True)
                .limit(20)
                .execute()
            ),
        )

    existing, unbound = await retry_transient_db_operation(
        _load_existing_message_rows,
        label=f"load_recent_messages:{fan_id}",
    )
    existing_by_platform_id = {
        str(row.get("fansly_message_id")): row
        for row in (existing.data or [])
        if row.get("fansly_message_id")
    }
    media_lookup = _apifansly_account_media_lookup(account_media)
    rows = []
    media_updates = []
    identity_updates = []
    used_unbound_ids: set[str] = set()
    for message in reversed(messages):
        row = _apifansly_message_row(
            message,
            fan_id=fan_id,
            creator_id=creator_id,
            creator_platform_id=creator_platform_id,
            media_lookup=media_lookup,
        )
        if not row:
            continue
        platform_message_id = str(message.get("id") or "")
        current = existing_by_platform_id.get(platform_message_id)
        if not current:
            candidate = _matching_unbound_creator_message(
                row,
                unbound.data or [],
                used_ids=used_unbound_ids,
            )
            if candidate:
                candidate_id = str(candidate["id"])
                used_unbound_ids.add(candidate_id)
                identity_updates.append({
                    "id": candidate_id,
                    "fansly_message_id": platform_message_id,
                    "media_context": row.get("media_context"),
                })
                existing_by_platform_id[platform_message_id] = candidate
                continue
            rows.append(row)
            continue
        attachments = (row.get("media_context") or {}).get("attachments") or []
        current_attachments = (
            (current.get("media_context") or {}).get("attachments") or []
        )
        has_resolved_media = any(item.get("url") for item in attachments)
        already_resolved = any(item.get("url") for item in current_attachments)
        discovered_attachments = bool(attachments) and not current_attachments
        discovered_metadata = any(
            item.get("mimetype") or item.get("filename")
            for item in attachments
        ) and not any(
            item.get("mimetype") or item.get("filename")
            for item in current_attachments
        )
        if (
            current.get("role") == "fan"
            and (
                discovered_attachments
                or discovered_metadata
                or (has_resolved_media and not already_resolved)
            )
        ):
            media_updates.append({
                "id": str(current["id"]),
                "media_context": row["media_context"],
            })

    if rows:
        await asyncio.to_thread(
            lambda: db.table("messages").insert(rows).execute()
        )
    for update in identity_updates:
        payload = {"fansly_message_id": update["fansly_message_id"]}
        if update.get("media_context"):
            payload["media_context"] = update["media_context"]
        await retry_transient_db_operation(
            lambda item=update, values=payload: asyncio.to_thread(
                lambda: db.table("messages")
                .update(values)
                .eq("id", item["id"])
                .eq("fan_id", fan_id)
                .is_("fansly_message_id", "null")
                .execute()
            ),
            label=f"reconcile_message_identity:{update['id']}",
        )
    for update in media_updates:
        await retry_transient_db_operation(
            lambda item=update: asyncio.to_thread(
                lambda: db.table("messages")
                .update({"media_context": item["media_context"]})
                .eq("id", item["id"])
                .eq("fan_id", fan_id)
                .execute()
            ),
            label=f"update_message_media:{update['id']}",
        )
    inbound = [row for row in rows if row["role"] == "fan"]
    newest_text = next(
        (row for row in reversed(inbound) if str(row.get("content") or "").strip()),
        None,
    )
    if newest_text:
        await process_incoming_fan_message(
            fan_id,
            creator_id,
            str(newest_text["content"]),
            creator_auto_mode,
            str(newest_text["fansly_message_id"]),
        )
    result = {
        "imported": len(rows),
        "inbound": len(inbound),
        "media_updated": len(media_updates),
        "identity_reconciled": len(identity_updates),
        "attachments_seen": sum(
            len((row.get("media_context") or {}).get("attachments") or [])
            for row in rows
        ),
        "attachments_resolved": sum(
            1
            for row in rows
            for item in ((row.get("media_context") or {}).get("attachments") or [])
            if item.get("url")
        ),
    }
    if (
        result["imported"]
        or result["media_updated"]
        or result["identity_reconciled"]
    ):
        print(
            f"[SYNC FAN MESSAGES] fan={fan_id} imported={result['imported']} "
            f"inbound={result['inbound']} media_updated={result['media_updated']} "
            f"identity_reconciled={result['identity_reconciled']} "
            f"attachments={result['attachments_resolved']}/{result['attachments_seen']}"
        )
    return result


@app.post(
    "/sync-fan-messages/{creator_id}/{fan_id}",
    dependencies=[Depends(require_creator_fan_access)],
)
async def sync_recent_fan_messages(creator_id: str, fan_id: str) -> dict:
    """Low-cost active-chat reconciliation for managed API Fansly accounts."""

    if not apifansly_enabled():
        # The dashboard polls this on every conversation open, on tab focus and
        # every 15 minutes. With the connector intentionally off it must not
        # become a recurring failure: answer with the same counter keys the
        # success path returns, so app/page.tsx computes changed == 0 and simply
        # waits out its normal safety interval.
        return {
            "status": "skipped",
            "reason": REASON_DISABLED,
            "imported": 0,
            "inbound": 0,
            "media_updated": 0,
            "identity_reconciled": 0,
        }

    db = get_supabase()
    async def _load_bindings():
        return await asyncio.gather(
            asyncio.to_thread(
                lambda: db.table("fans")
                .select("fansly_group_id, platform_fan_id")
                .eq("id", fan_id)
                .eq("creator_id", creator_id)
                .single()
                .execute()
            ),
            asyncio.to_thread(
                lambda: db.table("creators")
                .select("apifansly_account_id, fansly_account_id, auto_mode")
                .eq("id", creator_id)
                .single()
                .execute()
            ),
        )

    fan_result, creator_result = await retry_transient_db_operation(
        _load_bindings,
        label=f"active_chat_bindings:{creator_id}:{fan_id}",
    )
    fan = fan_result.data or {}
    creator = creator_result.data or {}
    account_id = str(creator.get("apifansly_account_id") or "")
    platform_id = str(creator.get("fansly_account_id") or "")
    group_id = str(fan.get("fansly_group_id") or "")
    platform_fan_id = str(fan.get("platform_fan_id") or "")
    if not account_id or not platform_id or not platform_fan_id:
        missing = [
            name
            for name, value in (
                ("API Fansly account", account_id),
                ("creator platform account", platform_id),
                ("fan platform account", platform_fan_id),
            )
            if not value
        ]
        raise HTTPException(
            status_code=409,
            detail=(
                "The active chat is missing its "
                f"{', '.join(missing)} binding. Reconnect or resync this creator."
            ),
        )

    if not group_id:
        group_id, retry_after_seconds = await _resolve_active_chat_group_id(
            account_id=account_id,
            platform_fan_id=platform_fan_id,
            fan_id=fan_id,
        )
        if not group_id:
            print(
                f"[ACTIVE CHAT BINDING] fan={fan_id} group_missing=true "
                f"retry_after={retry_after_seconds}s"
            )
            return {
                "status": "binding_pending",
                "imported": 0,
                "inbound": 0,
                "media_updated": 0,
                "identity_reconciled": 0,
                "retry_after_seconds": retry_after_seconds,
            }

    async with apifansly_client_scope() as client:
        result = await _sync_recent_fan_messages(
            creator_id=creator_id,
            fan_id=fan_id,
            account_id=account_id,
            creator_platform_id=platform_id,
            group_id=group_id,
            creator_auto_mode=bool(creator.get("auto_mode")),
            client=client,
        )
    return {"status": "ok", **result}


_FANSLY_LISTS_DEFAULT_INTERVAL_HOURS = 6


def _fansly_lists_interval_hours() -> float:
    try:
        return max(
            0.25,
            float(os.environ.get("FANSLY_LISTS_SYNC_INTERVAL_HOURS", "6")),
        )
    except (TypeError, ValueError):
        return float(_FANSLY_LISTS_DEFAULT_INTERVAL_HOURS)


async def _claim_platform_purchase(
    *,
    creator_id: str,
    platform_order_id: str,
    fan_id: str | None = None,
    event_type: str = "ppv.purchased",
    account_media_id: str | None = None,
    price_cents: int | None = None,
) -> str:
    """Win or lose the right to process one platform order (REL-003).

    Returns 'claimed' to exactly one caller, 'duplicate' to every other
    including a concurrent one, and 'no_identity' when the platform did not
    supply an order id.

    A deployment that has not yet applied db/purchase_identity_v1.sql has no
    such function. That is reported as 'unavailable' rather than raising, so a
    rolling deploy in either order keeps working: the caller falls back to the
    pre-existing sales_log scan, which is what it did before this sprint.
    """
    db = get_supabase()
    try:
        result = await asyncio.to_thread(
            lambda: db.rpc(
                "claim_platform_purchase",
                {
                    "p_creator_id": creator_id,
                    "p_platform_order_id": platform_order_id,
                    "p_fan_id": fan_id,
                    "p_event_type": event_type,
                    "p_account_media_id": account_media_id,
                    "p_price_cents": price_cents,
                },
            ).execute()
        )
    except Exception as exc:
        print(
            f"[PPV WEBHOOK] purchase ledger unavailable creator={creator_id} "
            f"order={platform_order_id}: {exc}"
        )
        return "unavailable"
    return str(result.data or "unavailable")


async def _complete_platform_purchase(
    creator_id: str,
    platform_order_id: str,
) -> None:
    db = get_supabase()
    try:
        await asyncio.to_thread(
            lambda: db.rpc(
                "complete_platform_purchase",
                {
                    "p_creator_id": creator_id,
                    "p_platform_order_id": platform_order_id,
                },
            ).execute()
        )
    except Exception as exc:
        # The purchase itself is already recorded; this only marks the ledger
        # row. Leaving it 'claimed' still deduplicates correctly.
        print(
            f"[PPV WEBHOOK] could not settle ledger creator={creator_id} "
            f"order={platform_order_id}: {exc}"
        )


async def _release_platform_purchase(
    creator_id: str,
    platform_order_id: str,
) -> None:
    """Undo a claim that did not result in a recorded purchase.

    Failing to release is the one way this design could lose a sale, so the
    failure is logged loudly rather than swallowed silently.
    """
    db = get_supabase()
    try:
        await asyncio.to_thread(
            lambda: db.rpc(
                "release_platform_purchase",
                {
                    "p_creator_id": creator_id,
                    "p_platform_order_id": platform_order_id,
                },
            ).execute()
        )
    except Exception as exc:
        print(
            f"[PPV WEBHOOK CLAIM STUCK] creator={creator_id} "
            f"order={platform_order_id} action=manual_review: {exc}"
        )


async def _fansly_lists_sync_due(creator_id: str) -> bool:
    """Whether the mirrored lists are stale enough to refresh on this pass."""
    db = get_supabase()
    rows = (
        await asyncio.to_thread(
            lambda: db.table("creators")
            .select("last_fansly_lists_sync_at")
            .eq("id", creator_id)
            .limit(1)
            .execute()
        )
    ).data or []
    if not rows:
        return False
    raw = rows[0].get("last_fansly_lists_sync_at")
    if not raw:
        return True
    try:
        last = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    age_hours = (
        datetime.now(timezone.utc) - last
    ).total_seconds() / 3600
    return age_hours >= _fansly_lists_interval_hours()


async def _sync_fansly_lists_if_due(
    creator_id: str,
    account_id: str,
    *,
    force: bool,
) -> dict | None:
    """Refresh mirrored Fansly lists, never failing the surrounding chat sync.

    A 401/403 is surfaced as an explicit reconnect state rather than swallowed;
    services.fansly_lists has already recorded it against the creator, and the
    caller's own binding backoff still owns the retry cadence.
    """
    from services.apifansly import ApiFanslyAccountAccessError
    from services.fansly_lists import (
        lists_sync_enabled,
        sync_fansly_lists_single_flight,
    )

    if not lists_sync_enabled():
        return None
    if not apifansly_enabled():
        # Every call this makes is remote. Reported the same way the feature's
        # own flag is, so the caller's status handling is unchanged.
        return {"status": "skipped", "reason": REASON_DISABLED}
    try:
        if not force and not await _fansly_lists_sync_due(creator_id):
            return None
        # Single-flight: the staleness check above answers "should this run",
        # not "am I the one running it". A manual refresh landing at the same
        # moment as this pass would otherwise reconcile the same membership
        # twice, each seeing the other's half-applied state.
        return await sync_fansly_lists_single_flight(creator_id, account_id)
    except ApiFanslyAccountAccessError as exc:
        print(f"[FANSLY LISTS ACCESS DENIED] creator={creator_id}: {exc}")
        return {"status": "access_denied", "detail": str(exc)}
    except Exception as exc:
        print(f"[FANSLY LISTS ERROR] creator={creator_id}: {exc}")
        return {"status": "error", "detail": str(exc)}


@app.post(
    "/sync-chats/{creator_id}",
    dependencies=[Depends(require_creator_path_access), Depends(require_apifansly_connector)],
)
async def sync_chats(
    creator_id: str,
    incremental: bool = False,
    force: bool = False,
) -> dict:

    db = get_supabase()
    creator_row = await asyncio.to_thread(
        lambda cid=creator_id: db.table("creators")
        .select("apifansly_account_id, fansly_account_id, auto_mode")
        .eq("id", cid)
        .single()
        .execute()
    )

    creator = creator_row.data or {}
    apifansly_id = creator.get("apifansly_account_id")
    creator_platform_id = str(creator.get("fansly_account_id") or "")
    creator_auto_mode = bool(creator.get("auto_mode"))
    if not apifansly_id:
        return {"status": "error", "message": "no apifansly account"}

    if incremental and not force:
        claim = await asyncio.to_thread(
            lambda: db.rpc(
                "claim_chat_reconciliation",
                {"p_creator_id": creator_id, "p_min_interval_minutes": 9},
            ).execute()
        )
        if not bool(claim.data):
            return {"status": "cooldown", "synced": 0, "new_chats": 0}

    # API-003 + FE-006 — one paginated read of the creator's fans, used for
    # three things that each used to cost their own round trips:
    #
    #  * the complete set of known platform ids, so the incremental early-break
    #    can actually trigger. This select had no .range(), so PostgREST capped
    #    it at 1,000 rows and page_ids.issubset(...) was almost never true for a
    #    creator past that. Those creators re-paginated the entire Fansly chat
    #    list every reconcile pass, forever.
    #
    #  * the persisted display name and group binding, so an UPDATE is issued
    #    only when a value actually changed. Every UPDATE is delivered to every
    #    subscribed dashboard as a realtime event, so 2,000 unchanged chats used
    #    to mean 2,000 writes and a 2,000-event burst every pass (FE-006).
    #
    #  * the fan's row id, replacing a per-chat get_fan() — which was a
    #    select("*") pulling every JSONB column to read one uuid.
    from core.pagination import fetch_all_rows_async

    existing_fans = await fetch_all_rows_async(
        lambda start, end: db.table("fans")
        # chat_last_message_id is the API-001 reconciliation checkpoint. It is
        # read here rather than in its own query precisely because this page
        # already exists: the durable checkpoint costs no extra round trip.
        .select(
            "id, platform_fan_id, fansly_group_id, display_name, avatar_url, "
            "chat_last_message_id"
        )
        .eq("creator_id", creator_id)
        # A unique total order: pages cannot drop or repeat a row.
        .order("id")
        .range(start, end)
        .execute()
    )
    fans_by_platform_id: dict[str, dict] = {}
    for row in existing_fans:
        platform_id_value = str(row.get("platform_fan_id") or "")
        if platform_id_value:
            fans_by_platform_id[platform_id_value] = row
    existing_platform_ids: set[str] = set(fans_by_platform_id)

    async with apifansly_client_scope() as client:
        all_chats = []
        account_lookup: dict[str, dict] = {}
        cursor = None

        while True:
            try:
                chats, accounts, cursor = await apifansly_list_chats(
                    str(apifansly_id),
                    cursor=cursor,
                    client=client,
                )
            except ApiFanslyAccountAccessError as exc:
                print(f"[SYNC AUTH ERROR] creator={creator_id}: {exc}")
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except ApiFanslyConfigurationError as exc:
                print(f"[SYNC CONFIG ERROR] creator={creator_id}: {exc}")
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            for a in accounts:
                aid = str(a.get("id", ""))
                if aid:
                    account_lookup[aid] = a

            if not all_chats:
                print(
                    f"[SYNC FIRST PAGE] chats={len(chats)} "
                    f"accounts={len(accounts)}"
                )

            print(f"[SYNC CHATS] batch={len(chats)} total={len(all_chats)+len(chats)} nextCursor={cursor}")

            if not chats:
                break

            all_chats.extend(chats)

            if incremental:
                page_ids = {
                    str(chat.get("partnerAccountId", ""))
                    for chat in chats
                    if chat.get("partnerAccountId")
                }
                # Fansly returns the most recently active chats first. Once a
                # whole page is already known, older pages cannot contain a new
                # chat-list entry for this reconciliation pass.
                if page_ids and page_ids.issubset(existing_platform_ids):
                    break

            if not cursor:
                break

        synced = 0
        updated = 0
        new_chats = 0
        new_messages = 0
        for chat in all_chats:
            platform_fan_id = str(chat.get("partnerAccountId", ""))
            account_data = account_lookup.get(platform_fan_id, {})
            fan_name = (
                account_data.get("displayName")
                or account_data.get("username")
                or chat.get("partnerUsername", f"Fan_{platform_fan_id[-6:]}")
            )
            avatar_url = None
            avatar = account_data.get("avatar", {})
            if avatar and avatar.get("locations"):
                avatar_url = avatar["locations"][0].get("location")
            group_id = str(chat.get("groupId", ""))

            if not platform_fan_id or not group_id:
                continue

            known = fans_by_platform_id.get(platform_fan_id)
            is_new_chat = known is None
            if known is None:
                created = await create_fan(creator_id, platform_fan_id, fan_name)
                new_chats += 1
                # create_fan writes the display name and nothing else, so the
                # binding below is a genuine change for a new fan.
                known = {
                    "id": str(created.id),
                    "platform_fan_id": platform_fan_id,
                    "fansly_group_id": None,
                    "display_name": fan_name,
                    "avatar_url": None,
                }
                fans_by_platform_id[platform_fan_id] = known

            fan_row_id = str(known["id"])

            # Compare against what is already stored rather than writing
            # unconditionally. The values are already in hand, so this costs no
            # extra read.
            #
            # API-001 — the write is deliberately deferred until after the
            # message sync below, so the reconciliation checkpoint rides the
            # SAME update statement as the binding/display corrections instead
            # of adding a second one. Unchanged chats still issue no write at
            # all, and a chat that did sync issues exactly one.
            update_payload: dict[str, object] = {}
            group_binding_changed = (
                str(known.get("fansly_group_id") or "") != group_id
            )
            if group_binding_changed:
                update_payload["fansly_group_id"] = group_id
            if str(known.get("display_name") or "") != fan_name:
                update_payload["display_name"] = fan_name
            if avatar_url and str(known.get("avatar_url") or "") != avatar_url:
                update_payload["avatar_url"] = avatar_url

            synced += 1
            platform_last_message_id = str(
                chat.get("lastMessageId") or ""
            )
            should_sync_messages = _chat_message_sync_needed(
                platform_last_message_id,
                str(known.get("chat_last_message_id") or ""),
                is_new_chat=is_new_chat,
                group_binding_changed=group_binding_changed,
            )
            if incremental and creator_platform_id and should_sync_messages:
                try:
                    recent = await _sync_recent_fan_messages(
                        creator_id=creator_id,
                        fan_id=fan_row_id,
                        account_id=str(apifansly_id),
                        creator_platform_id=creator_platform_id,
                        group_id=group_id,
                        creator_auto_mode=creator_auto_mode,
                        client=client,
                    )
                    new_messages += int(recent.get("imported") or 0)
                    # Only after the import actually succeeded. A checkpoint
                    # written for a failed sync would suppress the retry.
                    #
                    # The stored marker is the one list_chats reported, not the
                    # newest id on the fetched page: the next pass compares
                    # against list_chats, so the two have to come from the same
                    # source or they would never match and nothing would ever be
                    # suppressed.
                    if platform_last_message_id:
                        update_payload["chat_last_message_id"] = (
                            platform_last_message_id
                        )
                        update_payload["chat_last_synced_at"] = (
                            datetime.now(timezone.utc).isoformat()
                        )
                except ApiFanslyAccountAccessError:
                    raise
                except Exception as exc:
                    print(
                        f"[SYNC MESSAGES ERROR] creator={creator_id} "
                        f"fan={fan_row_id} group={group_id}: {exc}"
                    )
            elif group_binding_changed and known.get("chat_last_message_id"):
                # Rebound to a different conversation without importing from it
                # on this pass (a full sync does not read messages per chat).
                # The old checkpoint describes the OLD chat, so leaving it would
                # let a stale marker suppress the first sync of the new one.
                #
                # Guarded on there being a checkpoint to clear: a brand new fan
                # has none, and writing null over null would add two fields to
                # every new-fan update for no effect.
                update_payload["chat_last_message_id"] = None
                update_payload["chat_last_synced_at"] = None

            if update_payload:
                await asyncio.to_thread(
                    lambda fid=fan_row_id, p=update_payload: db.table("fans")
                    .update(p)
                    .eq("id", fid)
                    .execute()
                )
                known.update(update_payload)
                updated += 1

        await _stamp_vault_op(creator_id, "last_chat_reconcile_at")
        audience_sync = None
        if force or not incremental:
            try:
                from services.fansly_audience import sync_fansly_audience

                audience_sync = await sync_fansly_audience(
                    creator_id,
                    str(apifansly_id),
                )
            except Exception as exc:
                # Chat sync remains useful even when a newer audience endpoint is
                # temporarily unavailable. The failure is explicit in the result.
                audience_sync = {"status": "error", "detail": str(exc)}
                print(
                    f"[FANSLY AUDIENCE ERROR] creator={creator_id}: {exc}"
                )
        # Fansly list mirroring rides the same account-synchronization lifecycle
        # rather than adding a scheduler: every full sync refreshes it, and an
        # incremental pass refreshes it only once the mirror is stale.
        lists_sync = await _sync_fansly_lists_if_due(
            creator_id,
            str(apifansly_id),
            force=bool(force or not incremental),
        )
        print(
            f"[SYNC CHATS] incremental={incremental} total_chats={len(all_chats)} "
            f"synced={synced} updated={updated} new={new_chats}"
        )
        return {
            "status": "ok",
            "mode": "incremental" if incremental else "full",
            "synced": synced,
            "updated": updated,
            "new_chats": new_chats,
            "new_messages": new_messages,
            "audience": audience_sync,
            "lists": lists_sync,
        }


@app.get("/apifansly-usage")
async def get_apifansly_usage() -> dict:
    """Expose rolling, secret-free provider usage to dashboard operators."""
    return apifansly_usage_snapshot()


@app.post(
    "/load-history/{creator_id}/{fan_id}",
    dependencies=[Depends(require_creator_fan_access), Depends(require_apifansly_connector)],
)
async def load_fan_history(creator_id: str, fan_id: str) -> dict:

    db = get_supabase()

    fan_row = await asyncio.to_thread(
        lambda: db.table("fans")
        .select("fansly_group_id, platform_fan_id")
        .eq("id", fan_id)
        .single()
        .execute()
    )
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("apifansly_account_id, fansly_account_id")
        .eq("id", creator_id)
        .single()
        .execute()
    )

    group_id = (fan_row.data or {}).get("fansly_group_id")
    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
    fansly_account_id = str((creator_row.data or {}).get("fansly_account_id", ""))

    if not group_id or not apifansly_id:
        return {"status": "error", "message": "missing fan or creator info"}

    # Paginated: a truncated set makes history import re-attempt inserts the
    # unique (creator_id, fansly_message_id) index then rejects, so a long
    # conversation turned into a wave of failing writes on every load.
    existing_rows = await fetch_all_rows_async(
        lambda start, end: db.table("messages")
        .select("fansly_message_id")
        .eq("fan_id", fan_id)
        .order("id")
        .range(start, end)
        .execute()
    )
    existing_ids = {
        r["fansly_message_id"] for r in existing_rows if r.get("fansly_message_id")
    }

    all_messages = []
    all_media = {}
    cursor = None

    print(f"[LOAD HISTORY URL] apifansly_id={apifansly_id} group_id={group_id}")

    async with apifansly_client_scope() as client:
        while True:
            messages, account_media_batch, cursor = (
                await apifansly_list_chat_messages(
                    str(apifansly_id),
                    str(group_id),
                    cursor=cursor,
                    limit=50,
                    client=client,
                )
            )

            print(f"[LOAD HISTORY] batch={len(messages)} total={len(all_messages)+len(messages)} nextCursor={cursor}")

            for am in account_media_batch:
                content_id_1 = str(am.get("id", ""))
                content_id_2 = str(am.get("mediaId", ""))
                media = am.get("media", {})
                locations = media.get("locations", [])
                variants = media.get("variants", [])
                url = None
                if locations:
                    url = locations[0].get("location")
                elif variants and variants[0].get("locations"):
                    url = variants[0]["locations"][0].get("location")
                price = int(am.get("price") or 0)
                purchased = bool(am.get("purchased", am.get("isPurchased", False)))
                access = am.get("access")
                media_info = {
                    "url": url,
                    "price": price / 100 if price > 100 else price,
                    "is_ppv": price > 0,
                    "purchased": purchased,
                    "access": access,
                }
                if content_id_1:
                    all_media[content_id_1] = media_info
                if content_id_2 and content_id_2 != content_id_1:
                    all_media[content_id_2] = media_info

            all_messages.extend(messages)

            if not cursor or not messages:
                break

    print(f"[MEDIA LOOKUP] keys={list(all_media.keys())[:5]}")

    rows_to_insert: list[dict] = []
    for msg in reversed(all_messages):
        msg_id = str(msg.get("id", ""))
        if not msg_id or msg_id in existing_ids:
            continue

        content = msg.get("content", "")
        sender_id = str(msg.get("senderId", ""))
        role = "fan" if sender_id != fansly_account_id else "creator"

        created_at = msg.get("createdAt")
        if created_at and created_at > 0:
            ts = float(created_at)
            if ts > 1e12:
                ts /= 1000.0
            sent_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        else:
            sent_at = datetime.now(timezone.utc).isoformat()

        attachments = msg.get("attachments", [])
        media_context = None
        if attachments:
            resolved = []
            for att in attachments:
                content_id = str(att.get("contentId", ""))
                print(f"[ATT RESOLVE] contentId={content_id} found={content_id in all_media}")
                info = all_media.get(content_id)
                if isinstance(info, dict):
                    resolved.append({
                        "contentId": content_id,
                        "url": info.get("url"),
                        "type": att.get("contentType", 1),
                        "price": info.get("price"),
                        "is_ppv": info.get("is_ppv"),
                        "purchased": info.get("purchased"),
                        "access": info.get("access"),
                    })
                else:
                    resolved.append({
                        "contentId": content_id,
                        "url": info,
                        "type": att.get("contentType", 1),
                    })
            media_context = {"attachments": resolved}

        if not content and not attachments:
            continue

        row = {
            "fan_id": fan_id,
            "creator_id": creator_id,
            "role": role,
            "content": content,
            "fansly_message_id": msg_id,
            "sent_at": sent_at,
            "media_context": media_context,
        }

        rows_to_insert.append(row)
        existing_ids.add(msg_id)

    for start in range(0, len(rows_to_insert), 250):
        batch = rows_to_insert[start:start + 250]
        await asyncio.to_thread(
            lambda rows=batch: db.table("messages").insert(rows).execute()
        )

    imported = len(rows_to_insert)

    if imported > 0:
        conversation_history = await get_conversation_history(fan_id)
        fan_profile = await get_fan_by_id(fan_id)

        if fan_profile and len(conversation_history) >= 10:
            spawn(_update_fan_ai_summary(fan_id, conversation_history), name="update_fan_ai_summary")
            spawn(
                _update_fan_memory(fan_id, creator_id, conversation_history, fan_profile.total_spent),
                name="update_fan_memory",
            )

    return {"status": "ok", "imported": imported}


@app.post(
    "/mark-all-read/{creator_id}",
    dependencies=[Depends(require_creator_path_access), Depends(require_apifansly_connector)],
)
async def mark_all_read(creator_id: str) -> dict:

    db = get_supabase()
    creator_row = await asyncio.to_thread(
        lambda cid=creator_id: db.table("creators")
        .select("apifansly_account_id")
        .eq("id", cid)
        .single()
        .execute()
    )
    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
    if not apifansly_id:
        return {"status": "error"}

    async with apifansly_client_scope() as client:
        await client.post(
            apifansly_url(f"{apifansly_id}/chats/mark-as-read"),
            headers=apifansly_headers(),
            timeout=10,
        )
    return {"status": "ok"}


def _first_media_location(value) -> str:
    if not isinstance(value, list):
        return ""
    for entry in value:
        if isinstance(entry, dict) and entry.get("location"):
            return str(entry["location"])
    return ""


def _vault_media_visual_urls(media: dict) -> tuple[str, str]:
    """Extract the original URL and a real image thumbnail when available."""
    original_url = _first_media_location(media.get("locations"))
    thumbnail_url = ""
    for variant in media.get("variants") or []:
        if not isinstance(variant, dict):
            continue
        variant_mimetype = str(variant.get("mimetype") or "").lower()
        variant_filename = str(variant.get("filename") or "").lower()
        is_image = variant_mimetype.startswith("image/") or variant_filename.endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
        if is_image:
            thumbnail_url = _first_media_location(variant.get("locations"))
            if thumbnail_url:
                break
    if str(media.get("mimetype") or "").startswith("image/") and not thumbnail_url:
        thumbnail_url = original_url
    return original_url, thumbnail_url


@app.post(
    "/sync-vault/{creator_id}",
    dependencies=[Depends(require_creator_path_access), Depends(require_apifansly_connector)],
)
async def sync_vault(creator_id: str) -> dict:

    db = get_supabase()
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("apifansly_account_id")
        .eq("id", creator_id)
        .single()
        .execute()
    )

    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
    async with apifansly_client_scope() as client:
        # Step 1: Get all albums
        albums = await apifansly_list_vault_albums(
            str(apifansly_id),
            client=client,
        )
        print(f"[VAULT] found {len(albums)} albums")

        total_synced = 0

        # Step 2: For each album, fetch media
        for album in albums:
            album_id = album.get("id")
            album_title = album.get("title") or f"Album_{album_id}"
            item_count = album.get("itemCount", 0)
            print(f"[VAULT] album={album_title} items={item_count}")

            cursor = None
            while True:
                items, cursor = await apifansly_list_vault_album_media(
                    str(apifansly_id),
                    str(album_id),
                    cursor=cursor,
                    limit=50,
                    client=client,
                )

                print(f"[VAULT] album={album_title} batch={len(items)} cursor={cursor}")

                if not items:
                    break

                for item in items:
                    media = item.get("media", {})
                    media_id = str(media.get("id", ""))
                    mimetype = media.get("mimetype", "")
                    price = item.get("price", 0)

                    url, thumbnail_url = _vault_media_visual_urls(media)

                    if not media_id or not url:
                        continue

                    await asyncio.to_thread(
                        lambda cid=creator_id, mid=media_id, u=url, tu=thumbnail_url, mt=mimetype, fn=media.get("filename", ""), aid=album_id, at=album_title, pr=price: db.table("creator_vault_media").upsert({
                            "creator_id": cid,
                            "media_id": mid,
                            "fansly_media_id": mid,
                            "url": u,
                            "thumbnail_url": tu or None,
                            "mimetype": mt,
                            "filename": fn,
                            "album_id": aid,
                            "album_title": at,
                            "price": pr,
                        }, on_conflict="creator_id,media_id").execute()
                    )
                    total_synced += 1

                if not cursor:
                    break

        return {"status": "ok", "synced": total_synced}


@app.post(
    "/sync-vault-start/{creator_id}",
    dependencies=[Depends(require_creator_path_access), Depends(require_apifansly_connector)],
)
async def sync_vault_start(creator_id: str, force: bool = False) -> dict:
    if _vault_sync_state.get(creator_id, {}).get("status") in _VAULT_SYNC_ACTIVE_STATUSES:
        return {"status": "already_running"}
    retry_after = _vault_sync_retry_after.get(creator_id, 0)
    if not force and retry_after > time.time():
        return {
            "status": "retry_backoff",
            "retry_after_seconds": round(retry_after - time.time()),
            "requires_reconnect": True,
        }
    if force:
        _vault_sync_retry_after.pop(creator_id, None)
    # The background scheduler checks hourly, while API Fansly is called for a
    # creator at most once per 24 hours unless an operator explicitly forces it.
    if not force:
        cd = await _vault_cooldown_remaining(creator_id, "last_vault_sync_at")
        if not cd["allowed"]:
            return {"status": "cooldown", **cd}
    # Spawned immediately, but it waits for a creator-level slot before doing
    # any work. The scheduler can therefore mark every due creator without
    # starting every due creator (VAULT-001).
    _vault_sync_state[creator_id] = {
        "status": "queued",
        "synced": 0,
        "total": 0,
        "album": "Waiting for a vault slot…",
    }
    # VAULT-003 — durable from the moment the obligation exists, not from the
    # moment a slot frees up. A restart while queued is just as invisible to the
    # operator as a restart while running.
    await _mark_vault_run_started(creator_id)
    spawn(_run_vault_sync(creator_id), name="run_vault_sync")
    return {"status": "started"}


@app.get(
    "/sync-vault-status/{creator_id}",
    dependencies=[Depends(require_creator_path_access)],
)
async def sync_vault_status(creator_id: str) -> dict:
    state = _vault_sync_state.get(creator_id)
    if state is not None:
        return state
    # No in-process state. Either nothing has been asked of this creator, or a
    # run was cut off by a restart — which used to be reported identically, as
    # "idle" (VAULT-003).
    return await _durable_vault_sync_state(creator_id)


async def _vault_existing_media_ids(creator_id: str) -> set[str]:
    """Load every existing media ID; Supabase caps ordinary selects at 1,000."""
    db = get_supabase()
    page_size = 1000
    offset = 0
    result: set[str] = set()
    while True:
        response = await asyncio.to_thread(
            lambda start=offset: db.table("creator_vault_media")
            .select("media_id")
            .eq("creator_id", creator_id)
            .order("media_id")
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = response.data or []
        result.update(str(row["media_id"]) for row in rows if row.get("media_id"))
        if len(rows) < page_size:
            return result
        offset += page_size


async def _run_vault_sync(creator_id: str) -> None:

    db = get_supabase()
    try:
        async with VAULT_GATE.acquire(creator_id=creator_id, kind="vault_sync"):
            _vault_sync_state[creator_id] = {
                "status": "running",
                "synced": 0,
                "total": 0,
                "album": "",
            }
            await _run_vault_sync_locked(creator_id, db)
    finally:
        # VAULT-003 — released on every terminal path including cancellation,
        # because a run that ended is 'idle' whether it succeeded or not. Only a
        # run that never got here should look interrupted; leaving the marker
        # behind would make a completed sync report interrupted forever.
        await _mark_vault_run_finished(creator_id)


async def _run_vault_sync_locked(creator_id: str, db) -> None:
    """The sync itself. Runs only while holding one creator-level vault slot."""


    try:
        creator_row = await asyncio.to_thread(
            lambda: db.table("creators")
            .select(
                "apifansly_account_id, auto_categorize_new_media, "
                "vault_initial_categorized_at"
            )
            .eq("id", creator_id)
            .single()
            .execute()
        )
        apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
        auto_categorize_new = bool(
            (creator_row.data or {}).get("auto_categorize_new_media", True)
        )
        initial_completed_at = (creator_row.data or {}).get(
            "vault_initial_categorized_at"
        )
        existing_ids = await _vault_existing_media_ids(creator_id)

        async with apifansly_client_scope() as client:
            albums = await apifansly_list_vault_albums(
                str(apifansly_id),
                client=client,
            )

            # Albums overlap heavily (especially Fansly's standard "All"
            # album). Summing itemCount produced misleading progress such as
            # 57/114 for 57 unique assets. The largest album is the best
            # available unique-vault estimate before all pages are scanned.
            total = max(
                (int(a.get("itemCount") or 0) for a in albums),
                default=0,
            )
            already = len(existing_ids)
            new_total = max(total - already, 0)
            synced = 0
            new_item_ids: list[str] = []

            _vault_sync_state[creator_id] = {"status": "running", "synced": 0, "total": 0, "album": "Starting..."}

            for album in ordered_vault_albums(albums):
                album_id = album.get("id")
                album_title = album.get("title") or f"Album_{album_id}"
                cursor = None
                consecutive_dupe_batches = 0

                while True:
                    is_first_page = cursor is None
                    items, cursor = await apifansly_list_vault_album_media(
                        str(apifansly_id),
                        str(album_id),
                        cursor=cursor,
                        limit=50,
                        client=client,
                    )

                    if not items:
                        break

                    batch = []
                    all_dupes = True
                    for item in items:
                        media = item.get("media", {})
                        media_id = str(media.get("id", ""))
                        if not media_id or media_id in existing_ids:
                            continue
                        all_dupes = False

                        mimetype = media.get("mimetype", "")
                        price = item.get("price", 0)

                        url, thumbnail_url = _vault_media_visual_urls(media)

                        if not url:
                            continue

                        batch.append({
                            "creator_id": creator_id,
                            "media_id": media_id,
                            "fansly_media_id": media_id,
                            "url": url,
                            "thumbnail_url": thumbnail_url or None,
                            "mimetype": mimetype,
                            "filename": media.get("filename", ""),
                            "album_id": album_id,
                            "album_title": album_title,
                            "price": price,
                        })
                        existing_ids.add(media_id)
                    if batch:
                        saved_rows = await asyncio.to_thread(
                            lambda b=batch: db.table("creator_vault_media")
                            .upsert(b, on_conflict="creator_id,media_id")
                            .execute()
                        )
                        returned_ids = [
                            str(row.get("id"))
                            for row in (saved_rows.data or [])
                            if row.get("id")
                        ]
                        if not returned_ids:
                            media_ids = [str(row["media_id"]) for row in batch]
                            looked_up = await asyncio.to_thread(
                                lambda mids=media_ids: db.table("creator_vault_media")
                                .select("id")
                                .eq("creator_id", creator_id)
                                .in_("media_id", mids)
                                .execute()
                            )
                            returned_ids = [
                                str(row.get("id"))
                                for row in (looked_up.data or [])
                                if row.get("id")
                            ]
                        new_item_ids.extend(returned_ids)
                        synced += len(batch)
                    consecutive_dupe_batches = (
                        consecutive_dupe_batches + 1 if all_dupes else 0
                    )

                    _vault_sync_state[creator_id] = {"status": "running", "synced": synced, "total": new_total, "album": album_title}
                    print(f"[VAULT SYNC] album={album_title} synced={synced}/{new_total} cursor={cursor}")

                    # ``lastItemId`` proves that this is the newest page. If it is
                    # entirely local, no older page can contain a later upload.
                    # Keep the conservative three-page fallback for older API
                    # responses that do not expose a comparable last-item ID.
                    if should_stop_album_scan(
                        items=items,
                        next_cursor=cursor,
                        is_first_page=is_first_page,
                        last_item_id=album.get("lastItemId"),
                        all_items_known=all_dupes,
                        consecutive_known_batches=consecutive_dupe_batches,
                    ):
                        break

        new_item_ids = normalize_media_ids(new_item_ids)
        categorized_new = 0
        category_errors = 0
        # Initial vault analysis is an explicit paid action. Auto-categorize
        # applies only after that one-time setup has completed.
        if (
            initial_completed_at
            and categorize_new_batch_enabled(auto_categorize_new, new_item_ids)
        ):
            _vault_sync_state[creator_id] = {
                "status": "categorizing_new",
                "synced": synced,
                "total": new_total,
                "album": "Categorizing newly imported media…",
            }
            _categorize_state[creator_id] = {
                "status": "running",
                "mode": "new",
                "done": 0,
                "total": len(new_item_ids),
                "errors": 0,
            }
            await _run_vault_categorization(
                creator_id,
                item_ids=new_item_ids,
                mark_initial=False,
            )
            category_state = _categorize_state.get(creator_id, {})
            categorized_new = int(category_state.get("done") or 0)
            category_errors = int(category_state.get("errors") or 0)

        await _stamp_vault_op(creator_id, "last_vault_sync_at")
        _vault_sync_retry_after.pop(creator_id, None)
        _vault_sync_state[creator_id] = {
            "status": "done",
            "synced": synced,
            "total": new_total,
            "album": "",
            "auto_categorize_new_media": auto_categorize_new,
            "categorized_new": categorized_new,
            "categorization_errors": category_errors,
        }
        print(
            f"[VAULT SYNC] done synced={synced} categorized_new={categorized_new} "
            f"category_errors={category_errors}"
        )

    except ApiFanslyAccountAccessError as e:
        _vault_sync_retry_after[creator_id] = (
            time.time() + _VAULT_ACCESS_DENIED_RETRY_SECONDS
        )
        print(
            f"[VAULT SYNC ACCESS DENIED] creator={creator_id} "
            f"retry_in={_VAULT_ACCESS_DENIED_RETRY_SECONDS}s: {e}"
        )
        _vault_sync_state[creator_id] = {
            "status": "error",
            "synced": 0,
            "total": 0,
            "album": str(e),
            "requires_reconnect": True,
            "retry_after_seconds": _VAULT_ACCESS_DENIED_RETRY_SECONDS,
        }
    except Exception as e:
        import traceback
        print(f"[VAULT SYNC ERROR] {e}")
        print(traceback.format_exc())
        _vault_sync_state[creator_id] = {"status": "error", "synced": 0, "total": 0, "album": str(e)}


@app.post(
    "/upload-vault-media/{creator_id}",
    dependencies=[Depends(require_creator_path_access), Depends(require_apifansly_connector)],
)
async def upload_vault_media(creator_id: str, request: Request) -> dict:

    db = get_supabase()
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("apifansly_account_id, auto_categorize_new_media")
        .eq("id", creator_id)
        .single()
        .execute()
    )
    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
    auto_categorize_new = bool(
        (creator_row.data or {}).get("auto_categorize_new_media", True)
    )
    form = await request.form()
    file = form.get("file")
    album_title = str(form.get("album_title") or "Uncategorized")
    album_id = str(form.get("album_id") or "")

    if not file:
        return {"status": "error", "message": "no file"}

    file_bytes = await file.read()
    filename = file.filename
    mimetype = file.content_type

    async with apifansly_client_scope() as client:
        upload_resp = await client.post(
            apifansly_url(f"{apifansly_id}/media/upload"),
            headers=apifansly_headers(),
            files={"file": (filename, file_bytes, mimetype)},
            timeout=60,
        )
        raise_for_apifansly_response(
            upload_resp,
            operation="media upload",
            account_id=apifansly_id,
        )
        upload_data = upload_resp.json()
        print(f"[UPLOAD] initiate response: {upload_data}")

        job_id = upload_data.get("data", {}).get("jobId")
        if not job_id:
            return {"status": "error", "message": "no jobId returned", "raw": str(upload_data)}

        media_id = None
        url = None
        for attempt in range(30):
            await asyncio.sleep(2)
            status_resp = await client.get(
                apifansly_url(f"media/upload/{job_id}/status"),
                headers=apifansly_headers(),
                timeout=15,
            )
            raise_for_apifansly_response(
                status_resp,
                operation="media upload status",
                account_id=apifansly_id,
            )
            status_data = status_resp.json()
            state = status_data.get("data", {}).get("state")
            print(f"[UPLOAD] job={job_id} attempt={attempt+1} state={state}")

            if state == "completed":
                result = status_data.get("data", {}).get("result", {})
                media_id = str(result.get("mediaId", ""))
                account_media = result.get("accountMedia", [])
                if account_media:
                    media_obj = account_media[0].get("media", {})
                    locations = media_obj.get("locations", [])
                    variants = media_obj.get("variants", [])
                    if locations:
                        url = locations[0].get("location")
                    elif variants and variants[0].get("locations"):
                        url = variants[0]["locations"][0].get("location")
                break
            elif state == "failed":
                return {"status": "error", "message": "upload job failed"}

        if not media_id or not url:
            return {"status": "error", "message": "upload timed out or no media URL"}

        ai_description = str(form.get("ai_description") or "")
        row = {
            "creator_id": creator_id,
            "media_id": media_id,
            "fansly_media_id": media_id,
            "url": url,
            "mimetype": mimetype,
            "filename": filename,
            "album_id": album_id,
            "album_title": album_title,
            "price": 0,
        }
        if ai_description:
            row["ai_description"] = ai_description
        db_result = await asyncio.to_thread(
            lambda: db.table("creator_vault_media")
            .upsert(row, on_conflict="creator_id,media_id")
            .select()
            .single()
            .execute()
        )
        saved = db_result.data or row
        print(f"[UPLOAD] saved to DB media_id={media_id}")
        # User-uploaded media follows the same opt-in rule as synced media.
        if saved.get("id") and auto_categorize_new:
            spawn(_categorize_single_item_and_save(saved), name="categorize_single_item")
        return {"status": "ok", "item": saved}


def _classification_update_payload(result: dict) -> dict:
    return {
        "content_category": result["content_category"],
        "ai_description": result["ai_description"],
        "price_min": result["price_min"],
        "price_max": result["price_max"],
        "explicitness_level": result.get("explicitness", 3),
        "good_for": result.get("good_for", "standalone"),
        "tags": result.get("tags", []),
        "scene_id": result.get("scene_id", ""),
        "scene_location": result.get("scene_location", ""),
        "scene_outfit": result.get("scene_outfit", ""),
        "scene_lighting": result.get("scene_lighting", ""),
        "classification_version": result.get(
            "classification_version", VAULT_CLASSIFIER_VERSION
        ),
        "classification_model": result.get("classification_model", ""),
        "classification_source": result.get("classification_source", ""),
        "classification_confidence": result.get("classification_confidence", 0),
        "classification_metadata": result.get("classification_metadata", {}),
        "classified_at": result.get("classified_at") or datetime.now(timezone.utc).isoformat(),
    }


async def _categorize_single_item_and_save(item: dict) -> None:
    try:
        result = await _categorize_single_item(item)
        db = get_supabase()
        await asyncio.to_thread(
            lambda: db.table("creator_vault_media")
            .update(_classification_update_payload(result))
            .eq("id", item["id"])
            .execute()
        )
        print(f"[UPLOAD CATEGORIZE] item={item['id']} category={result['content_category']}")
    except Exception as e:
        print(f"[UPLOAD CATEGORIZE ERROR] {e}")


VAULT_CATEGORIES = {
    "teaser_clothed":   {"min": 0,   "max": 0,   "label": "Clothed teaser (free)"},
    "teaser_bundle":    {"min": 0,   "max": 0,   "label": "Teaser bundle no nudity (free)"},
    "legs_feet":        {"min": 15,  "max": 70,  "label": "Legs / feet / armpits"},
    "lingerie_photo":   {"min": 10,  "max": 80,  "label": "Lingerie photo"},
    "lingerie_video":   {"min": 15,  "max": 90,  "label": "Lingerie video"},
    "nude_photo":       {"min": 15,  "max": 80,  "label": "Nude photo"},
    "nude_video":       {"min": 20,  "max": 110, "label": "Nude video"},
    "striptease_video": {"min": 15,  "max": 100, "label": "Striptease video"},
    "closeup_photo":    {"min": 25,  "max": 130, "label": "Closeup photo"},
    "closeup_video":    {"min": 25,  "max": 130, "label": "Closeup video"},
    "dictate_video":    {"min": 15,  "max": 50,  "label": "Dictate / dirty talk video"},
    "solo_toy_video":   {"min": 30,  "max": 150, "label": "Solo / toy / orgasm video"},
    "solo_toy_photo":   {"min": 20,  "max": 80,  "label": "Solo / toy photo"},
    "explicit_photo":   {"min": 25,  "max": 130, "label": "Explicit solo photo"},
    "explicit_video":   {"min": 35,  "max": 170, "label": "Explicit solo video"},
    "bg_content":       {"min": 50,  "max": 300, "label": "BG (boy-girl) content"},
    "task":             {"min": 10,  "max": 50,  "label": "Task / custom request"},
    "other":            {"min": 0,   "max": 0,   "label": "Other / unclear"},
}

class VaultVisualAccessError(RuntimeError):
    """The classifier could not obtain a usable visual for a vault item."""


async def _download_visual_candidate(
    visual_url: str,
    *,
    client,
) -> tuple[bytes, str]:
    """Try the CDN directly, then the managed protected-media endpoint."""
    direct_status = "not_attempted"
    try:
        response = await client.get(visual_url, timeout=25)
        direct_status = f"http_{response.status_code}_{len(response.content)}b"
        if response.status_code == 200 and len(response.content) > 1000:
            return bytes(response.content), "direct_cdn"
    except Exception as exc:
        direct_status = f"{type(exc).__name__}"

    if is_fansly_cdn_url(visual_url):
        try:
            content = await apifansly_download_media(
                visual_url,
                client=client,
            )
            return content, "apifansly_media_download"
        except Exception as exc:
            raise VaultVisualAccessError(
                "The protected Fansly media could not be downloaded "
                f"(direct={direct_status}; proxy={type(exc).__name__})."
            ) from exc

    raise VaultVisualAccessError(
        f"The media source could not be downloaded ({direct_status})."
    )


async def _refresh_vault_item_urls(item: dict) -> dict | None:
    """Refresh an expired signed Fansly URL from the item's stored album."""
    creator_id = str(item.get("creator_id") or "")
    album_id = str(item.get("album_id") or "")
    media_id = str(
        item.get("fansly_media_id") or item.get("media_id") or ""
    )
    if not creator_id or not album_id or not media_id:
        return None

    db = get_supabase()
    creator = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("apifansly_account_id")
        .eq("id", creator_id)
        .single()
        .execute()
    )
    account_id = str((creator.data or {}).get("apifansly_account_id") or "")
    if not account_id:
        return None

    import httpx

    cursor = None
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for _ in range(100):
            entries, cursor = await apifansly_list_vault_album_media(
                account_id,
                album_id,
                cursor=cursor,
                limit=50,
                client=client,
            )
            for entry in entries:
                media = entry.get("media") if isinstance(entry, dict) else None
                if not isinstance(media, dict):
                    continue
                candidate_id = str(
                    entry.get("mediaId") or media.get("id") or ""
                )
                if candidate_id != media_id:
                    continue
                url, thumbnail_url = _vault_media_visual_urls(media)
                if not url:
                    return None
                updates = {
                    "url": url,
                    "thumbnail_url": thumbnail_url or None,
                    "mimetype": media.get("mimetype") or item.get("mimetype"),
                    "filename": media.get("filename") or item.get("filename"),
                }
                await asyncio.to_thread(
                    lambda: db.table("creator_vault_media")
                    .update(updates)
                    .eq("id", item["id"])
                    .eq("creator_id", creator_id)
                    .execute()
                )
                return {**item, **updates}
            if not cursor:
                break
    return None


def _write_temp_video(content: bytes) -> str:
    import tempfile

    with tempfile.NamedTemporaryFile(
        prefix="cleopatra-video-",
        suffix=".media",
        delete=False,
    ) as temporary:
        temporary.write(content)
        return temporary.name


async def _video_classifier_image(
    video_url: str,
    *,
    client,
) -> tuple[bytes, str]:
    """Return a chronological keyframe sheet and a retrieval audit label."""
    import os

    settings = FrameSettings.from_env()
    if not settings.enabled:
        raise VaultVisualAccessError(
            "Video-frame analysis is disabled in this deployment."
        )
    if not ffmpeg_available():
        raise VaultVisualAccessError(
            "Video-frame analysis is unavailable because FFmpeg is missing."
        )

    direct = await extract_frames(video_url, settings=settings)
    if len(direct.frames) >= 2:
        sheet, count = await asyncio.to_thread(
            build_contact_sheet,
            direct.frames,
        )
        if count >= 2 and sheet:
            return sheet, f"video_frames_{count}_direct_cdn"

    if not is_fansly_cdn_url(video_url):
        raise VaultVisualAccessError(
            "The video could not provide at least two readable keyframes."
        )

    # Protected videos occasionally reject direct ffmpeg range requests. The
    # documented API Fansly proxy is a bounded fallback, serialized to avoid
    # loading several large clips into the Railway container at once.
    async with _protected_video_download_gate:
        content = await apifansly_download_media(
            video_url,
            client=client,
            timeout=max(settings.timeout_seconds * 2, 60),
        )
        temporary_path = await asyncio.to_thread(
            _write_temp_video,
            content,
        )
        try:
            protected = await extract_frames(
                temporary_path,
                settings=settings,
            )
        finally:
            try:
                await asyncio.to_thread(os.unlink, temporary_path)
            except FileNotFoundError:
                pass
    if len(protected.frames) < 2:
        raise VaultVisualAccessError(
            "The protected video could not provide at least two readable keyframes."
        )
    sheet, count = await asyncio.to_thread(
        build_contact_sheet,
        protected.frames,
    )
    if count < 2 or not sheet:
        raise VaultVisualAccessError(
            "The extracted video frames were blank or unreadable."
        )
    return sheet, f"video_frames_{count}_apifansly_download"


async def _load_vault_visual(
    item: dict,
    *,
    is_video: bool,
    client=None,
) -> tuple[bytes, str, str]:
    """Return image bytes, evidence source, and retrieval method."""
    import httpx

    source = "video_thumbnail" if is_video else "image"
    visual_url = str(
        (item.get("thumbnail_url") if is_video else item.get("url")) or ""
    )
    first_error: Exception | None = None

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(follow_redirects=True)
    try:
        if is_video:
            video_errors: list[Exception] = []
            video_url = str(item.get("url") or "")
            if video_url:
                try:
                    content, method = await _video_classifier_image(
                        video_url,
                        client=client,
                    )
                    return content, "video_frames", method
                except Exception as exc:
                    video_errors.append(exc)

            try:
                refreshed_video = await _refresh_vault_item_urls(item)
            except Exception as exc:
                refreshed_video = None
                video_errors.append(exc)
            refreshed_url = str((refreshed_video or {}).get("url") or "")
            if refreshed_url and refreshed_url != video_url:
                try:
                    content, method = await _video_classifier_image(
                        refreshed_url,
                        client=client,
                    )
                    return content, "video_frames", method + "_after_refresh"
                except Exception as exc:
                    video_errors.append(exc)

            detail = str(video_errors[-1]) if video_errors else (
                "Fansly did not provide the original video URL."
            )
            raise VaultVisualAccessError(
                "This video was left unclassified because multiple real "
                f"keyframes could not be extracted. {detail}"
            ) from (video_errors[-1] if video_errors else None)

        if visual_url:
            try:
                content, method = await _download_visual_candidate(
                    visual_url,
                    client=client,
                )
                return content, source, method
            except Exception as exc:
                first_error = exc

        try:
            refreshed = await _refresh_vault_item_urls(item)
        except Exception as exc:
            refreshed = None
            refresh_error = exc
        else:
            refresh_error = None

        if refreshed:
            refreshed_url = str(
                (
                    refreshed.get("thumbnail_url")
                    if is_video
                    else refreshed.get("url")
                )
                or ""
            )
            if refreshed_url:
                content, method = await _download_visual_candidate(
                    refreshed_url,
                    client=client,
                )
                return content, source, method + "_after_refresh"

        if is_video and not visual_url:
            reason = (
                "Fansly did not provide an image thumbnail for this video. "
                "The item was left unclassified rather than guessed from its filename."
            )
        else:
            reason = (
                "The protected media link is unavailable or expired. "
                "Reconnect the creator's API Fansly account or sync the vault to "
                "refresh signed media links, then retry."
            )
        cause = refresh_error or first_error
        raise VaultVisualAccessError(reason) from cause
    finally:
        if owns_client:
            await client.aclose()


def _prepare_classifier_image(visual_bytes: bytes) -> bytes:
    """Normalize a source image without blocking the async classification loop."""
    import io
    from PIL import Image, ImageOps

    image = Image.open(io.BytesIO(visual_bytes))
    image.seek(0)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((896, 896), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=84, optimize=True)
    return buffer.getvalue()


async def _categorize_single_item(
    item: dict,
    *,
    allow_core_qwen_fallback: bool = True,
    force_qwen: bool = False,
    visual_client=None,
) -> dict:
    """Classify one vault item into the versioned provider-neutral contract.

    Images are resized before upload to control vision-token cost.  Videos use
    their real platform thumbnail rather than guessing from a filename.  A
    provider/fetch/parse failure is raised so the retry loop can leave the item
    stale instead of permanently saving an empty ``other`` classification.
    """
    mimetype = str(item.get("mimetype") or "").lower()
    item_id = item.get("id", "")
    is_video = mimetype.startswith("video") if mimetype else False

    try:
        if visual_client is None:
            visual_bytes, source, fetch_method = await _load_vault_visual(
                item,
                is_video=is_video,
            )
        else:
            visual_bytes, source, fetch_method = await _load_vault_visual(
                item,
                is_video=is_video,
                client=visual_client,
            )

        # Classification does not need original-resolution media.  Normalizing
        # every asset to a compact JPEG makes cost predictable and also handles
        # thumbnails whose declared MIME type is missing or inaccurate.
        classifier_image = await asyncio.to_thread(
            _prepare_classifier_image,
            visual_bytes,
        )
        shoot_fingerprint = await build_shoot_fingerprint(classifier_image)
        local_visual = shoot_fingerprint.get("local") or {}
        data = await classify_vault_image(
            classifier_image,
            is_video=is_video,
            album_title=str(item.get("album_title") or ""),
            filename=str(item.get("filename") or ""),
            local_visual=local_visual,
            allow_core_qwen_fallback=allow_core_qwen_fallback,
            force_qwen=force_qwen,
        )
        model = str(data.pop("_classification_model", "nudenet-3.4.2"))
        provider_metadata = dict(data.pop("_provider_metadata", {}) or {})
        provider = str(provider_metadata.get("provider") or "local_nudenet")
        shoot_fingerprint = add_semantic_shoot_evidence(
            shoot_fingerprint,
            data,
        )
        data.pop("_semantic_fingerprint", None)
        vision_error = provider_metadata.get("vision_error_reason")
        qwen_status = provider_metadata.get("qwen_status")
        fallback_reasons = provider_metadata.get("qwen_fallback_reasons") or []
        deferred_reasons = provider_metadata.get("qwen_deferred_reasons") or []
        print(
            f"[CATEGORIZE RAW] item={item_id} provider={provider} "
            f"model={model} category={data.get('category')} "
            f"explicitness={data.get('explicitness')} "
            f"vision={provider_metadata.get('vision_status')}"
            + (f" qwen={qwen_status}" if qwen_status else "")
            + (
                f" fallback={','.join(fallback_reasons)}"
                if fallback_reasons
                else ""
            )
            + (
                f" deferred={','.join(deferred_reasons)}"
                if deferred_reasons
                else ""
            )
            + (f" vision_error={vision_error}" if vision_error else "")
        )
        if not data.get("colors"):
            data["colors"] = local_visual.get("palette_names") or []
        if useful_text(data.get("scene_lighting")) == "":
            data["scene_lighting"] = local_visual.get("lighting") or ""
        if useful_text(data.get("visual_tone")) == "":
            data["visual_tone"] = local_visual.get("visual_tone") or ""
        data["orientation"] = local_visual.get("orientation") or ""
        data["image_dimensions"] = (
            f"{local_visual.get('width')}x{local_visual.get('height')}"
            if local_visual.get("width") and local_visual.get("height")
            else ""
        )
        explicitness = explicitness_from_evidence(data)
        # Deterministically repair category/media mismatches and conservative
        # labels before they can become sellable set metadata.
        category = normalize_media_category(
            data.get("category"),
            explicitness=explicitness,
            is_video=is_video,
        )

        price_info = VAULT_CATEGORIES[category]
        good_for = data.get("good_for", "standalone")
        if good_for not in ["opener", "mid_session", "closer", "standalone"]:
            good_for = "standalone"
        data["explicitness"] = explicitness
        data["category"] = category
        data["good_for"] = good_for
        tags = semantic_tags(data)
        scene_id = useful_text(data.get("scene_id"))
        location = useful_text(data.get("scene_location"))
        outfit = useful_text(data.get("scene_outfit"))
        lighting = useful_text(data.get("scene_lighting"))
        confidence = classification_confidence(data.get("confidence"), source=source)
        metadata = {
            key: data.get(key)
            for key in (
                "description", "description_complete", "mood",
                "sexual_activity", "body_focus", "action",
                "pose", "framing", "props", "colors", "nudity",
                "visible_anatomy", "visual_tone", "orientation",
                "image_dimensions", "rich_visual_descriptor",
            )
        }
        metadata["age_review_required"] = bool(
            provider_metadata.get("age_review_required")
        )
        metadata["shoot_fingerprint"] = shoot_fingerprint
        metadata.update({
            "category": category,
            "explicitness": explicitness,
            "good_for": good_for,
            "tags": tags,
            "scene_location": location,
            "scene_outfit": outfit,
            "scene_lighting": lighting,
            "scene_id": scene_id,
            "evidence_source": source,
            "fetch_method": fetch_method,
            "classifier_provider": provider,
            "provider_details": provider_metadata,
        })
        print(
            f"[SHOOT FINGERPRINT] item={item_id} "
            f"status={shoot_fingerprint.get('status')} "
            f"palette={local_visual.get('palette_names') or []}"
        )

        return {
            "id": item_id,
            "content_category": category,
            "ai_description": media_description(metadata, source=source),
            "price_min": price_info["min"],
            "price_max": price_info["max"],
            "explicitness": explicitness,
            "good_for": good_for,
            "tags": tags,
            "scene_id": scene_id,
            "scene_location": location,
            "scene_outfit": outfit,
            "scene_lighting": lighting,
            "classification_version": VAULT_CLASSIFIER_VERSION,
            "classification_model": model,
            "classification_source": source,
            "classification_confidence": confidence,
            "classification_metadata": metadata,
            "classified_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        print(f"[CATEGORIZE] item={item_id} error={e}")
        raise


async def _vault_cooldown_remaining(creator_id: str, column: str) -> dict:
    """Return the daily API Fansly vault-sync cooldown for one creator."""
    db = get_supabase()
    row = await asyncio.to_thread(
        lambda: db.table("creators").select(column).eq("id", creator_id).single().execute()
    )
    last_str = (row.data or {}).get(column)
    return vault_sync_cooldown(
        last_str,
        interval_hours=_VAULT_SYNC_INTERVAL_HOURS,
    )


async def _mark_vault_run_started(creator_id: str) -> None:
    """Claim the vault run for this process (VAULT-003).

    Best effort by design. A deployment that has not applied
    db/vault_sync_interruption_v1.sql yet has no such columns; the sync itself
    is unaffected and only the interrupted/idle distinction is unavailable, so
    this must never be the thing that stops a vault import.
    """
    db = get_supabase()
    try:
        await asyncio.to_thread(
            lambda: db.table("creators")
            .update({
                "vault_sync_started_at": datetime.now(timezone.utc).isoformat(),
                "vault_sync_finished_at": None,
                "vault_sync_owner": _PROCESS_ID,
            })
            .eq("id", creator_id)
            .execute()
        )
    except Exception as exc:
        print(f"[VAULT STATE] could not mark start creator={creator_id}: {exc}")


async def _mark_vault_run_finished(creator_id: str) -> None:
    """Release the run. Called for success AND for failure.

    A finished run is 'idle' whether it succeeded or errored — the error is
    reported separately. What must not survive is the in-flight marker, or the
    next process would report a completed run as interrupted forever.
    """
    db = get_supabase()
    try:
        await asyncio.to_thread(
            lambda: db.table("creators")
            .update({
                "vault_sync_finished_at": datetime.now(timezone.utc).isoformat(),
                "vault_sync_owner": None,
            })
            .eq("id", creator_id)
            .execute()
        )
    except Exception as exc:
        print(f"[VAULT STATE] could not mark finish creator={creator_id}: {exc}")


async def _durable_vault_sync_state(creator_id: str) -> dict:
    """What the database says about a run this process knows nothing about.

    Only consulted when in-process state is empty, which after a restart is the
    normal case rather than an error.
    """
    db = get_supabase()
    try:
        rows = (
            await asyncio.to_thread(
                lambda: db.table("creators")
                .select(
                    "vault_sync_started_at, vault_sync_finished_at, "
                    "vault_sync_owner"
                )
                .eq("id", creator_id)
                .limit(1)
                .execute()
            )
        ).data or []
    except Exception:
        # The columns are missing (migration not applied) or the read failed.
        # Fall back to the pre-existing answer rather than inventing a state.
        return {"status": "idle", "synced": 0, "total": 0, "album": ""}

    row = rows[0] if rows else {}
    started = str(row.get("vault_sync_started_at") or "")
    finished = str(row.get("vault_sync_finished_at") or "")
    owner = str(row.get("vault_sync_owner") or "")

    if started and not finished and owner and owner != _PROCESS_ID:
        # A run began under a process that is no longer here. Recovery is
        # already automatic — an interrupted sync never stamped
        # last_vault_sync_at, so the cooldown never started and the scheduler
        # will pick it up — so this is a description, not an alarm, and there is
        # nothing for an operator to clear.
        return {
            "status": "interrupted",
            "synced": 0,
            "total": 0,
            "album": "",
            "interrupted_at": started,
            "recoverable": True,
            "detail": (
                "A vault synchronisation was interrupted by a restart. "
                "The next automatic or manual run resumes it."
            ),
        }
    return {"status": "idle", "synced": 0, "total": 0, "album": ""}


async def _stamp_vault_op(creator_id: str, column: str) -> None:
    from datetime import datetime, timezone
    db = get_supabase()
    await asyncio.to_thread(
        lambda: db.table("creators")
        .update({column: datetime.now(timezone.utc).isoformat()})
        .eq("id", creator_id).execute()
    )


async def _count_uncategorized(creator_id: str) -> int:
    db = get_supabase()
    r = await asyncio.to_thread(
        lambda: db.table("creator_vault_media")
        .select("id", count="exact", head=True)
        .eq("creator_id", creator_id)
        .or_("content_category.is.null,content_category.eq.")
        .execute()
    )
    return r.count or 0


async def _count_stale_classifications(creator_id: str) -> int:
    db = get_supabase()
    result = await asyncio.to_thread(
        lambda: db.table("creator_vault_media")
        .select("id", count="exact", head=True)
        .eq("creator_id", creator_id)
        .lt("classification_version", VAULT_CLASSIFIER_VERSION)
        .execute()
    )
    return result.count or 0


async def _stale_approved_set_media_ids(creator_id: str) -> list[str]:
    """Return DB row IDs for stale media already used by approved sets."""
    db = get_supabase()
    set_rows = await asyncio.to_thread(
        lambda: db.table("vault_sets")
        .select("media_ids")
        .eq("creator_id", creator_id)
        .eq("status", "approved")
        .execute()
    )
    external_ids = normalize_media_ids(
        media_id
        for row in (set_rows.data or [])
        for media_id in (row.get("media_ids") or [])
    )
    result: list[str] = []
    for start in range(0, len(external_ids), 250):
        chunk = external_ids[start:start + 250]
        rows = await asyncio.to_thread(
            lambda ids=chunk: db.table("creator_vault_media")
            .select("id")
            .eq("creator_id", creator_id)
            .in_("fansly_media_id", ids)
            .lt("classification_version", VAULT_CLASSIFIER_VERSION)
            .execute()
        )
        result.extend(str(row["id"]) for row in (rows.data or []) if row.get("id"))
    return normalize_media_ids(result)


async def _video_frame_upgrade_media_ids(creator_id: str) -> list[str]:
    """Return video rows that have never been classified from real frames."""
    db = get_supabase()
    result: list[str] = []
    page_size = 1000
    offset = 0
    while True:
        rows = await asyncio.to_thread(
            lambda start=offset: db.table("creator_vault_media")
            .select("id")
            .eq("creator_id", creator_id)
            .ilike("mimetype", "video/%")
            .or_(
                f"classification_version.lt.{VAULT_CLASSIFIER_VERSION},"
                "classification_source.is.null,"
                "classification_source.neq.video_frames"
            )
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = rows.data or []
        result.extend(
            str(row["id"])
            for row in batch
            if row.get("id")
        )
        if len(batch) < page_size:
            break
        offset += page_size
    return normalize_media_ids(result)


_categorize_state: dict = {}

# VAULT-002 — how many classification results are persisted per round trip, and
# how long a partial batch may wait. Small enough that completed work is never
# held in memory for long and the operator's progress number stays live; large
# enough that a 10,000-item vault is ~100 writes rather than 10,000. Internal
# constants on purpose: an env var per batch size is configuration sprawl.
_CLASSIFICATION_WRITE_BATCH = 100
_CLASSIFICATION_FLUSH_SECONDS = 5.0


@app.post(
    "/categorize-vault/{creator_id}",
    dependencies=[Depends(require_creator_path_access)],
)
async def categorize_vault(
    creator_id: str,
    mode: str = "auto",
    force: bool = False,
    confirm_upgrade: bool = False,
    upgrade_scope: str = "all",
) -> dict:
    """Start initial, new-media-only, or explicit version-upgrade work.

    ``force`` remains accepted for compatibility with older dashboard builds,
    but it cannot unlock a completed initial-vault run or reprocess categorized
    media. A confirmed upgrade can target older metadata, approved-set media,
    or videos that have not yet been inspected through real keyframes.
    """
    del force
    if _categorize_state.get(creator_id, {}).get("status") == "running":
        return {"status": "already_running", "state": _categorize_state[creator_id]}

    db = get_supabase()
    creator = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("vault_initial_categorized_at")
        .eq("id", creator_id)
        .single()
        .execute()
    )
    initial_completed_at = (creator.data or {}).get("vault_initial_categorized_at")
    resolved_mode = mode.strip().lower()
    if resolved_mode == "auto":
        resolved_mode = "new" if initial_completed_at else "initial"
    if resolved_mode not in {"initial", "new", "upgrade"}:
        raise HTTPException(
            status_code=400,
            detail="mode must be 'initial', 'new', or 'upgrade'",
        )
    if resolved_mode == "initial" and initial_completed_at:
        return {
            "status": "initial_already_completed",
            "initial_completed_at": initial_completed_at,
        }

    upgrade_item_ids: list[str] | None = None
    if resolved_mode == "upgrade":
        upgrade_scope = upgrade_scope.strip().lower()
        if upgrade_scope not in {"approved", "all", "videos"}:
            raise HTTPException(
                status_code=400,
                detail="upgrade_scope must be 'approved', 'videos', or 'all'",
            )
        if upgrade_scope == "approved":
            upgrade_item_ids = await _stale_approved_set_media_ids(creator_id)
            pending = len(upgrade_item_ids)
        elif upgrade_scope == "videos":
            upgrade_item_ids = await _video_frame_upgrade_media_ids(creator_id)
            pending = len(upgrade_item_ids)
        else:
            pending = await _count_stale_classifications(creator_id)
    else:
        pending = await _count_uncategorized(creator_id)
    if resolved_mode == "upgrade" and pending and not confirm_upgrade:
        return {
            "status": "confirmation_required",
            "mode": "upgrade",
            "items": pending,
            "upgrade_scope": upgrade_scope,
            "classifier_version": VAULT_CLASSIFIER_VERSION,
            "message": (
                "This is a one-time paid re-analysis of legacy metadata. "
                "Retry with confirm_upgrade=true to start it."
            ),
        }
    if pending == 0:
        return {
            "status": "nothing_to_categorize",
            "mode": resolved_mode,
            "uncategorized": 0,
            "stale_classifications": 0,
        }

    _categorize_state[creator_id] = {
        "status": "queued",
        "mode": resolved_mode,
        "done": 0,
        "total": pending,
        "errors": 0,
    }
    await _stamp_vault_op(creator_id, "last_categorize_at")
    spawn(
        _run_vault_categorization_job(
            creator_id,
            item_ids=upgrade_item_ids,
            mark_initial=resolved_mode == "initial",
            upgrade_legacy=resolved_mode == "upgrade",
        ),
        name=f"run_vault_categorization:{resolved_mode}",
    )
    return {
        "status": "started",
        "mode": resolved_mode,
        "items": pending,
        "upgrade_scope": upgrade_scope if resolved_mode == "upgrade" else None,
        "classifier_version": VAULT_CLASSIFIER_VERSION,
    }


@app.get(
    "/categorize-vault-status/{creator_id}",
    dependencies=[Depends(require_creator_path_access)],
)
async def categorize_vault_status(creator_id: str) -> dict:
    return _categorize_state.get(creator_id, {"status": "idle", "done": 0, "total": 0})


class VaultCategorizationSettingsRequest(BaseModel):
    auto_categorize_new_media: bool


@app.put(
    "/creator/{creator_id}/vault-categorization-settings",
    dependencies=[Depends(require_creator_path_access)],
)
async def update_vault_categorization_settings(
    creator_id: str,
    settings: VaultCategorizationSettingsRequest,
) -> dict:
    db = get_supabase()
    result = await asyncio.to_thread(
        lambda: db.table("creators")
        .update({"auto_categorize_new_media": settings.auto_categorize_new_media})
        .eq("id", creator_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="creator not found")
    return {
        "status": "ok",
        "auto_categorize_new_media": settings.auto_categorize_new_media,
    }


async def _manual_recategorization_usage(creator_id: str) -> dict:
    db = get_supabase()
    result = await asyncio.to_thread(
        lambda: db.rpc(
            "vault_recategorization_usage",
            {
                "p_creator_id": creator_id,
                "p_daily_limit": MANUAL_RECATEGORIZATION_DAILY_LIMIT,
            },
        ).execute()
    )
    row = (result.data or [{}])[0]
    return manual_recategorization_usage(
        int(row.get("used") or 0),
        int(row.get("daily_limit") or MANUAL_RECATEGORIZATION_DAILY_LIMIT),
    )


@app.get(
    "/creator/{creator_id}/vault-categorization-overview",
    dependencies=[Depends(require_creator_path_access)],
)
async def vault_categorization_overview(creator_id: str) -> dict:
    db = get_supabase()
    creator = await asyncio.to_thread(
        lambda: db.table("creators")
        .select(
            "vault_initial_categorized_at, auto_categorize_new_media, "
            "last_vault_sync_at"
        )
        .eq("id", creator_id)
        .single()
        .execute()
    )
    if not creator.data:
        raise HTTPException(status_code=404, detail="creator not found")
    stale_approved = await _stale_approved_set_media_ids(creator_id)
    return {
        "initial_completed_at": creator.data.get("vault_initial_categorized_at"),
        "auto_categorize_new_media": bool(
            creator.data.get("auto_categorize_new_media", True)
        ),
        "last_vault_sync_at": creator.data.get("last_vault_sync_at"),
        "vault_sync_interval_hours": _VAULT_SYNC_INTERVAL_HOURS,
        "active_sync": _vault_sync_state.get(
            creator_id,
            {"status": "idle", "synced": 0, "total": 0, "album": ""},
        ),
        # Deployment-wide, so an operator can see that their creator is queued
        # behind other creators rather than stalled (VAULT-001).
        "vault_gate": VAULT_GATE.snapshot(),
        "uncategorized": await _count_uncategorized(creator_id),
        "stale_classifications": await _count_stale_classifications(creator_id),
        "stale_approved_classifications": len(stale_approved),
        "video_frame_upgrades": len(
            await _video_frame_upgrade_media_ids(creator_id)
        ),
        "classifier_version": VAULT_CLASSIFIER_VERSION,
        "manual_reanalysis": await _manual_recategorization_usage(creator_id),
        "active_run": _categorize_state.get(
            creator_id,
            {"status": "idle", "done": 0, "total": 0},
        ),
    }


@app.get(
    "/vault-media-url/{creator_id}/{media_id}",
    dependencies=[Depends(require_creator_path_access)],
)
async def get_vault_media_url(creator_id: str, media_id: str) -> dict:
    """Look up a vault media item's URL and thumbnail by fansly_media_id."""
    db = get_supabase()
    row = await asyncio.to_thread(
        lambda: db.table("creator_vault_media")
        .select("url, thumbnail_url, mimetype")
        .eq("creator_id", creator_id)
        .eq("fansly_media_id", media_id)
        .limit(1)
        .execute()
    )
    if not row.data:
        return {"url": None, "thumbnail_url": None, "mimetype": None}
    item = row.data[0]
    return {
        "url": item.get("url"),
        "thumbnail_url": item.get("thumbnail_url"),
        "mimetype": item.get("mimetype"),
    }


@app.post(
    "/vault-media-urls/{creator_id}",
    dependencies=[Depends(require_creator_path_access)],
)
async def get_vault_media_urls(
    creator_id: str,
    request: VaultMediaUrlsRequest,
) -> dict:
    """Resolve a message's PPV thumbnails in one bounded database query."""
    media_ids = normalize_media_ids(request.media_ids)
    if not media_ids:
        return {"media": {}}
    db = get_supabase()
    def _load_media() -> list[dict]:
        rows: list[dict] = []
        for start in range(0, len(media_ids), 250):
            result = (
                db.table("creator_vault_media")
                .select("fansly_media_id, url, thumbnail_url, mimetype")
                .eq("creator_id", creator_id)
                .in_("fansly_media_id", media_ids[start:start + 250])
                .execute()
            )
            rows.extend(result.data or [])
        return rows

    rows = await asyncio.to_thread(_load_media)
    resolved = {
        str(row["fansly_media_id"]): {
            "url": row.get("url"),
            "thumbnail_url": row.get("thumbnail_url"),
            "mimetype": row.get("mimetype"),
        }
        for row in rows
        if row.get("fansly_media_id")
    }
    return {
        "media": {
            media_id: resolved.get(
                media_id,
                {"url": None, "thumbnail_url": None, "mimetype": None},
            )
            for media_id in media_ids
        }
    }


async def _run_vault_categorization_job(
    creator_id: str,
    *,
    item_ids: list[str] | None = None,
    mark_initial: bool = False,
    upgrade_legacy: bool = False,
) -> None:
    """An operator-started categorisation run, holding one creator-level slot.

    The run started from inside ``_run_vault_sync`` deliberately does NOT come
    through here: it already holds that sync's slot, and taking a second one
    would deadlock the gate at a limit of 1 (VAULT-001).
    """

    async with VAULT_GATE.acquire(creator_id=creator_id, kind="vault_categorize"):
        state = _categorize_state.get(creator_id)
        if isinstance(state, dict):
            state["status"] = "running"
        await _run_vault_categorization(
            creator_id,
            item_ids=item_ids,
            mark_initial=mark_initial,
            upgrade_legacy=upgrade_legacy,
        )


async def _run_vault_categorization(
    creator_id: str,
    *,
    item_ids: list[str] | None = None,
    mark_initial: bool = False,
    upgrade_legacy: bool = False,
) -> None:
    db = get_supabase()
    try:
        all_items: list[dict] = []
        target_ids = normalize_media_ids(item_ids)
        select_fields = (
            "id, creator_id, media_id, fansly_media_id, album_id, "
            "url, thumbnail_url, mimetype, filename, album_title"
        )
        if target_ids:
            # URL-safe chunks also make the exact new-media contract explicit.
            for start in range(0, len(target_ids), 250):
                chunk = target_ids[start:start + 250]
                if upgrade_legacy:
                    rows = await asyncio.to_thread(
                        lambda ids=chunk: db.table("creator_vault_media")
                        .select(select_fields)
                        .eq("creator_id", creator_id)
                        .in_("id", ids)
                        .execute()
                    )
                else:
                    rows = await asyncio.to_thread(
                        lambda ids=chunk: db.table("creator_vault_media")
                        .select(select_fields)
                        .eq("creator_id", creator_id)
                        .in_("id", ids)
                        .or_("content_category.is.null,content_category.eq.")
                        .execute()
                    )
                all_items.extend(rows.data or [])
        else:
            # Collect the complete target list before writing any rows. This
            # prevents offset pagination from skipping rows as their version is
            # updated during a legacy upgrade.
            page_size = 1000
            from_idx = 0
            while True:
                def _load_page(offset: int = from_idx):
                    query = (
                        db.table("creator_vault_media")
                        .select(select_fields)
                        .eq("creator_id", creator_id)
                    )
                    if upgrade_legacy:
                        query = query.lt(
                            "classification_version", VAULT_CLASSIFIER_VERSION
                        )
                    else:
                        query = query.or_(
                            "content_category.is.null,content_category.eq."
                        )
                    return query.range(offset, offset + page_size - 1).execute()

                rows = await asyncio.to_thread(_load_page)
                batch = rows.data or []
                all_items.extend(batch)
                if len(batch) < page_size:
                    break
                from_idx += page_size

        total = len(all_items)
        _categorize_state[creator_id]["total"] = total
        mode = "new" if target_ids else ("upgrade" if upgrade_legacy else "initial")
        print(f"[CATEGORIZE] creator={creator_id} mode={mode} items={total}")

        import time
        import httpx

        started_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        _categorize_state[creator_id]["started_at"] = started_at
        done = 0
        errors = 0
        provider_failures = 0
        qwen_fallbacks = 0
        semantic_failures = 0
        semantic_enabled = bool(
            os.environ.get("VAULT_SEMANTIC_BASE_URL", "").strip()
        )
        try:
            configured_concurrency = int(
                os.environ.get(
                    "VAULT_CATEGORIZATION_CONCURRENCY",
                    "12" if semantic_enabled else "2",
                )
            )
        except ValueError:
            configured_concurrency = 12 if semantic_enabled else 2
        batch_size = min(max(configured_concurrency, 1), 32)
        allow_core_qwen_fallback = total <= 100
        _categorize_state[creator_id]["concurrency"] = batch_size
        _categorize_state[creator_id]["core_qwen_fallback"] = (
            allow_core_qwen_fallback
        )
        print(
            f"[CATEGORIZE] creator={creator_id} "
            f"concurrency={batch_size} semantic={semantic_enabled} "
            f"core_qwen_fallback={allow_core_qwen_fallback}"
        )
        limits = httpx.Limits(
            max_connections=max(batch_size * 2, 16),
            max_keepalive_connections=max(batch_size, 8),
        )

        # ---- VAULT-002: worker pool instead of fixed-window batches ----
        # This used to be `for i in range(0, total, batch_size)` with an
        # asyncio.gather per slice. That gather is a barrier: a slice of 12
        # finishes only when its slowest item finishes, so one video needing
        # ffmpeg frame extraction (~35 s) held eleven idle slots against eleven
        # images that each took about a second. A fixed number of workers
        # pulling from a shared cursor uses the same concurrency budget without
        # ever idling a slot behind someone else's video.
        #
        # The concurrency limit itself is unchanged and is still never exceeded:
        # exactly `batch_size` workers exist, so at most `batch_size` items are
        # in flight.
        next_index = 0
        completed = 0
        abort_reason = ""
        pending_writes: list[dict] = []
        last_flush = time.monotonic()
        write_lock = asyncio.Lock()
        cursor_lock = asyncio.Lock()

        async def flush_writes(*, force: bool = False) -> None:
            """Persist accumulated classifications in one bounded round trip.

            Each result used to be its own awaited UPDATE, so a 10,000-item
            vault was 10,000 sequential round trips — several minutes of pure
            database latency. Rows are batched instead, and the batch is small
            enough that completed work is never held in memory for long: the
            point is fewer writes, not one giant write at the end.
            """

            nonlocal pending_writes, last_flush, done
            async with write_lock:
                due = (
                    force
                    or len(pending_writes) >= _CLASSIFICATION_WRITE_BATCH
                    or (
                        pending_writes
                        and time.monotonic() - last_flush >= _CLASSIFICATION_FLUSH_SECONDS
                    )
                )
                if not due or not pending_writes:
                    return
                rows, pending_writes = pending_writes, []
                last_flush = time.monotonic()
                await retry_transient_db_operation(
                    lambda batch=rows: asyncio.to_thread(
                        lambda: db.table("creator_vault_media")
                        .upsert(batch, on_conflict="id")
                        .execute()
                    ),
                    label=f"save_vault_classifications:{len(rows)}",
                )
                # ``done`` stays "persisted", not "classified", so the operator's
                # progress number never runs ahead of the database.
                done += len(rows)

        def report_progress() -> None:
            elapsed = max(time.monotonic() - started_monotonic, 0.001)
            rate = done / elapsed
            remaining = max(total - done - errors, 0)
            eta = round(remaining / rate) if rate > 0 else None
            _categorize_state[creator_id].update({
                "done": done,
                "errors": errors,
                "qwen_fallbacks": qwen_fallbacks,
                "semantic_failures": semantic_failures,
                "elapsed_seconds": round(elapsed),
                "items_per_minute": round(rate * 60, 1),
                "estimated_seconds_remaining": eta,
            })

        async def next_item() -> dict | None:
            nonlocal next_index
            async with cursor_lock:
                if abort_reason or next_index >= total:
                    return None
                item = all_items[next_index]
                next_index += 1
                return item

        async def worker(visual_client) -> None:
            nonlocal completed, errors, provider_failures
            nonlocal qwen_fallbacks, semantic_failures, abort_reason
            while True:
                item = await next_item()
                if item is None:
                    return
                try:
                    result = await _categorize_single_item_with_retry(
                        item,
                        allow_core_qwen_fallback=allow_core_qwen_fallback,
                        visual_client=visual_client,
                    )
                except Exception as error:
                    errors += 1
                    if isinstance(error, VaultClassifierError):
                        provider_failures += 1
                        if provider_failures >= 3:
                            # Unchanged abort rule: three provider failures stop
                            # the run. Workers notice on their next pull, so
                            # nothing new starts and in-flight items finish.
                            abort_reason = (
                                "Vault categorization stopped after three provider "
                                f"failures: {error}"
                            )
                    continue

                provider_details = (
                    (result.get("classification_metadata") or {})
                    .get("provider_details") or {}
                )
                if provider_details.get("qwen_status") == "ready":
                    qwen_fallbacks += 1
                if provider_details.get("semantic_status") == "fallback":
                    semantic_failures += 1

                # on_conflict targets the primary key, so this is an update of
                # an existing row. creator_id and media_id travel with it so the
                # row is fully identified and can never be written under another
                # creator.
                pending_writes.append({
                    "id": result["id"],
                    "creator_id": item["creator_id"],
                    "media_id": item["media_id"],
                    **_classification_update_payload(result),
                })
                completed += 1
                await flush_writes()
                report_progress()

                # The self-hosted semantic service scales independently.
                # Preserve the old provider throttle only for legacy
                # configurations: one worker sleeping after each item reproduces
                # the previous rate of batch_size items per 1.5 seconds.
                if not semantic_enabled:
                    await asyncio.sleep(1.5)

        async with httpx.AsyncClient(
            follow_redirects=True,
            limits=limits,
        ) as visual_client:
            try:
                await asyncio.gather(
                    *[worker(visual_client) for _ in range(batch_size)]
                )
            finally:
                # Work that was already classified is persisted even when the
                # run aborts, exactly as the per-item writes used to be.
                await flush_writes(force=True)

        report_progress()
        print(
            f"[CATEGORIZE] done={done}/{total} errors={errors} "
            f"qwen_fallbacks={qwen_fallbacks} "
            f"semantic_failures={semantic_failures}"
        )

        if abort_reason:
            _categorize_state[creator_id].update({
                "status": "error",
                "done": done,
                "errors": errors,
                "error": abort_reason,
                "aborted_remaining": max(total - done - errors, 0),
            })
            print(
                f"[CATEGORIZE ABORTED] creator={creator_id} "
                f"done={done}/{total} errors={errors} reason={abort_reason}"
            )
            return

        if mark_initial and errors == 0:
            await _stamp_vault_op(creator_id, "vault_initial_categorized_at")
        try:
            refreshed_sets = await _refresh_vault_set_descriptions(creator_id)
        except Exception as refresh_error:
            refreshed_sets = 0
            print(
                f"[SET METADATA] creator={creator_id} refresh failed: {refresh_error}"
            )
        _categorize_state[creator_id].update({
            "status": "done",
            "initial_locked": bool(mark_initial and errors == 0),
            "sets_refreshed": refreshed_sets,
        })
        elapsed = max(time.monotonic() - started_monotonic, 0.001)
        _categorize_state[creator_id].update({
            "elapsed_seconds": round(elapsed),
            "estimated_seconds_remaining": 0,
        })
        print(
            f"[CATEGORIZE] complete done={done} errors={errors} "
            f"elapsed_s={elapsed:.1f} rate={done / elapsed * 60:.1f}/min"
        )

    except Exception as e:
        import traceback
        print(f"[CATEGORIZE ERROR] {e}")
        traceback.print_exc()
        _categorize_state[creator_id]["status"] = "error"


async def _refresh_vault_set_descriptions(creator_id: str) -> int:
    """Rebuild existing set semantics from their exact current media rows."""
    db = get_supabase()
    set_result = await asyncio.to_thread(
        lambda: db.table("vault_sets")
        .select("id, media_ids")
        .eq("creator_id", creator_id)
        .execute()
    )
    sets = set_result.data or []
    external_ids = normalize_media_ids(
        media_id
        for vault_set in sets
        for media_id in (vault_set.get("media_ids") or [])
    )
    media_by_id: dict[str, dict] = {}
    fields = (
        "fansly_media_id, content_category, ai_description, explicitness_level, "
        "scene_location, scene_outfit, scene_lighting, mimetype, tags, "
        "classification_metadata"
    )
    for start in range(0, len(external_ids), 250):
        chunk = external_ids[start:start + 250]
        result = await asyncio.to_thread(
            lambda ids=chunk: db.table("creator_vault_media")
            .select(fields)
            .eq("creator_id", creator_id)
            .in_("fansly_media_id", ids)
            .execute()
        )
        for row in result.data or []:
            media_id = str(row.get("fansly_media_id") or "")
            if media_id:
                media_by_id[media_id] = row

    refreshed = 0
    for vault_set in sets:
        items = [
            media_by_id[str(media_id)]
            for media_id in (vault_set.get("media_ids") or [])
            if str(media_id) in media_by_id
        ]
        if not items:
            continue
        description = build_set_description(items)
        await asyncio.to_thread(
            lambda sid=vault_set["id"], text=description: db.table("vault_sets")
            .update({
                "description": text,
                "metadata_version": VAULT_CLASSIFIER_VERSION,
            })
            .eq("id", sid)
            .execute()
        )
        refreshed += 1
    print(f"[SET METADATA] creator={creator_id} refreshed={refreshed}")
    return refreshed


async def _categorize_single_item_with_retry(
    item: dict,
    max_retries: int = 3,
    *,
    allow_core_qwen_fallback: bool = True,
    visual_client=None,
) -> dict:
    """Wrap _categorize_single_item with exponential backoff on 429."""
    for attempt in range(max_retries):
        try:
            return await _categorize_single_item(
                item,
                allow_core_qwen_fallback=allow_core_qwen_fallback,
                visual_client=visual_client,
            )
        except Exception as e:
            message = str(e).lower()
            if (
                "429" in message or "throttl" in message
            ) and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"[CATEGORIZE] rate limited, waiting {wait}s before retry")
                await asyncio.sleep(wait)
            else:
                raise
    return await _categorize_single_item(
        item,
        allow_core_qwen_fallback=allow_core_qwen_fallback,
        visual_client=visual_client,
    )


@app.post(
    "/recategorize-item/{item_id}",
    dependencies=[Depends(require_vault_item_path_access)],
)
async def recategorize_item(item_id: str) -> dict:
    db = get_supabase()
    row = await asyncio.to_thread(
        lambda: db.table("creator_vault_media")
        .select(
            "id, creator_id, media_id, fansly_media_id, album_id, "
            "url, thumbnail_url, mimetype, filename, album_title"
        )
        .eq("id", item_id)
        .single()
        .execute()
    )
    item = row.data
    if not item:
        return {"status": "error", "message": "item not found"}

    creator_id = str(item.get("creator_id") or "")
    try:
        claim = await asyncio.to_thread(
            lambda: db.rpc(
                "claim_vault_recategorization",
                {
                    "p_creator_id": creator_id,
                    "p_media_id": str(item_id),
                    "p_daily_limit": MANUAL_RECATEGORIZATION_DAILY_LIMIT,
                },
            ).execute()
        )
    except Exception as exc:
        if "daily vault re-categorization limit reached" in str(exc).lower():
            raise HTTPException(
                status_code=429,
                detail=(
                    "The daily AI re-analysis limit has been reached. "
                    "Manual metadata editing is still available."
                ),
            ) from exc
        raise

    claim_row = (claim.data or [{}])[0]
    usage = manual_recategorization_usage(
        int(claim_row.get("used") or 0),
        MANUAL_RECATEGORIZATION_DAILY_LIMIT,
    )

    try:
        result = await _categorize_single_item(item, force_qwen=True)
    except VaultVisualAccessError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except VaultClassifierError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"{exc} The existing media details were preserved; "
                "please retry later."
            ),
        ) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "The vision classifier did not return usable structured metadata. "
                "The existing media details were preserved; please retry later."
            ),
        ) from exc
    await asyncio.to_thread(
        lambda: db.table("creator_vault_media")
        .update(_classification_update_payload(result))
        .eq("id", item_id)
        .execute()
    )
    updated = await asyncio.to_thread(
        lambda: db.table("creator_vault_media")
        .select("*")
        .eq("id", item_id)
        .single()
        .execute()
    )
    return {"status": "ok", "item": updated.data, "manual_reanalysis": usage}


@app.get(
    "/media/{account_id}/{content_id}",
    dependencies=[Depends(require_account_path_access), Depends(require_apifansly_connector)],
)
async def get_media_url(account_id: str, content_id: str) -> dict:

    async with apifansly_client_scope() as client:
        response = await client.get(
            apifansly_url(f"{account_id}/media/{content_id}"),
            headers=apifansly_headers(),
            timeout=10,
        )
        raise_for_apifansly_response(
            response,
            operation="vault media URL lookup",
            account_id=account_id,
        )
        print(f"[MEDIA] status={response.status_code} body={response.text[:300]}")
        return response.json()


@app.post("/generate-suggestions")
async def generate_suggestions_webhook(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
) -> dict:
    if payload.type != "INSERT":
        return {"status": "skipped"}
    record = payload.record
    message_id = record.get("id")
    message_content = record.get("content")
    print(f"[WEBHOOK] message_id={message_id} role={record.get('role')} content={message_content[:30]}")
    if record.get("role") != "fan":
        return {"status": "skipped"}
    fansly_msg_id = record.get("fansly_message_id")
    if fansly_msg_id:
        return {"status": "skipped - handled by fansly webhook"}
    if message_id in _processed_messages:
        return {"status": "duplicate"}
    _processed_messages.add(message_id)

    fan_id = record.get("fan_id")
    creator_id = record.get("creator_id")
    if not all([fan_id, creator_id, message_content, message_id]):
        return {"status": "skipped"}

    db = get_supabase()
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators").select("auto_mode").eq("id", creator_id).single().execute()
    )
    auto_mode = (creator_row.data or {}).get("auto_mode", False)

    await process_incoming_fan_message(
        str(fan_id), str(creator_id), str(message_content), auto_mode, str(message_id),
    )
    return {"status": "ok"}


async def _enrich_fan_profile(fan_id: str, creator_id: str, platform_fan_id: str) -> None:
    """Fetch real username, avatar and group_id by scanning chats list."""
    try:
        db = get_supabase()
        creator_row = await asyncio.to_thread(
            lambda: db.table("creators")
            .select("apifansly_account_id")
            .eq("id", creator_id)
            .single()
            .execute()
        )
        apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
        if apifansly_id:
            await get_or_fetch_group_id(apifansly_id, platform_fan_id, fan_id)
    except Exception as e:
        print(f"[FAN ENRICH ERROR] {e}")


@app.post("/webhook/fansly")
async def fansly_webhook(request: Request) -> dict:
    raw_body = await request.body()
    signing_secret = (
        os.environ.get("APIFANSLY_WEBHOOK_SECRET")
        or os.environ.get("WEBHOOK_SECRET", "")
    )
    supplied_signature = request.headers.get("signature")
    legacy_secret = request.headers.get("x-webhook-secret")
    signature_valid = valid_hmac_sha256_signature(
        raw_body,
        supplied_signature,
        signing_secret,
    )
    # Retain the old shared-secret header for local tools while production API
    # Fansly deliveries use the cryptographic Signature header.
    legacy_valid = bool(
        signing_secret
        and legacy_secret
        and _consteq(legacy_secret, signing_secret)
    )
    if not signing_secret:
        if not _is_dev():
            raise HTTPException(500, "Server webhook signing secret is not configured")
    elif not signature_valid and not legacy_valid:
        raise HTTPException(401, "Invalid webhook signature")

    try:
        payload = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(400, "Invalid webhook payload")
    if not isinstance(payload, dict):
        raise HTTPException(400, "Invalid webhook payload")

    print(
        f"[FANSLY WEBHOOK] event={payload.get('event')} "
        f"account={payload.get('accountId')}"
    )

    event = payload.get("event")
    data = payload.get("data") or {}
    api_account_id = str(payload.get("accountId") or "")
    db = get_supabase()

    creator: dict = {}
    if api_account_id:
        creator_result = await asyncio.to_thread(
            lambda: db.table("creators")
            .select("id, auto_mode, auto_mode_new_fans, fansly_account_id")
            .eq("apifansly_account_id", api_account_id)
            .limit(1)
            .execute()
        )
        creator = (creator_result.data or [{}])[0]
    if api_account_id and not creator:
        print(
            f"[FANSLY WEBHOOK] ignored unknown API account={api_account_id} "
            f"event={event}"
        )
        return {"status": "creator_not_found"}

    if event == "ppv.purchased":
        creator_id = str(creator.get("id") or "")
        platform_fan_id = str(data.get("accountId") or "")
        account_media_id = str(data.get("accountMediaId") or "")
        platform_order_id = str(data.get("orderId") or "")
        price_cents = int(
            ((data.get("orderMetadata") or {}).get("accountMediaPrice"))
            or 0
        )
        if not creator_id or not platform_fan_id:
            return {"status": "invalid_ppv_purchase_event"}

        fan_result = await asyncio.to_thread(
            lambda: db.table("fans")
            .select("id, pending_ppv_check, sales_log")
            .eq("creator_id", creator_id)
            .eq("platform_fan_id", platform_fan_id)
            .limit(1)
            .execute()
        )
        fans = fan_result.data or []
        if not fans:
            print(
                f"[PPV WEBHOOK] fan not found creator={creator_id} "
                f"platform_fan={platform_fan_id}"
            )
            return {"status": "fan_not_found"}

        fan_row = fans[0]

        # REL-003 — platform order identity is decided by PostgreSQL, not by
        # scanning a jsonb array in Python.
        #
        # The scan below still runs, but only AFTER the claim and only as a
        # compatibility check: orders processed before this migration exist in
        # sales_log and in no ledger, and must keep deduplicating. It is not the
        # concurrency authority any more and cannot be — two concurrent
        # deliveries both read a log without the order in it.
        #
        # The claim is taken before any work, so every path that returns without
        # recording a purchase has to release it (see _release below). Otherwise
        # a redelivery of a genuinely unprocessed order would be rejected as a
        # duplicate and the sale would be lost.
        purchase_claim = "no_identity"
        if platform_order_id:
            purchase_claim = await _claim_platform_purchase(
                creator_id=creator_id,
                platform_order_id=platform_order_id,
                fan_id=str(fan_row["id"]),
                account_media_id=account_media_id or None,
                price_cents=price_cents or None,
            )
            if purchase_claim == "duplicate":
                print(
                    f"[PPV WEBHOOK] duplicate order={platform_order_id} "
                    f"creator={creator_id}"
                )
                return {"status": "duplicate"}

        async def _release() -> None:
            """Give the order back so a later redelivery can still apply it."""
            if purchase_claim == "claimed":
                await _release_platform_purchase(creator_id, platform_order_id)

        if platform_order_id and any(
            str(entry.get("platform_order_id") or "") == platform_order_id
            for entry in (fan_row.get("sales_log") or [])
        ):
            # Already in this fan's history from before the ledger existed.
            # Settle the claim as processed rather than releasing it, so the
            # ledger now carries the identity too and the scan stops being
            # needed for this order.
            if purchase_claim == "claimed":
                await _complete_platform_purchase(creator_id, platform_order_id)
            return {"status": "duplicate"}

        delivery_result = await asyncio.to_thread(
            lambda: db.table("ppv_deliveries")
            .select(
                "reference, status, media_ids, price_cents, source, set_id, "
                "step_index, platform_message_id, claimed_at, delivered_at"
            )
            .eq("creator_id", creator_id)
            .eq("fan_id", str(fan_row["id"]))
            .in_("status", ["claimed", "delivered_pending", "abandoned"])
            .order("claimed_at")
            .execute()
        )
        delivery_rows = delivery_result.data or []

        def _delivery_matches(delivery: dict) -> bool:
            media_ids = normalize_media_ids(delivery.get("media_ids") or [])
            if account_media_id and media_ids and account_media_id not in media_ids:
                return False
            expected = int(delivery.get("price_cents") or 0)
            if expected and price_cents:
                return abs(expected - price_cents) <= max(
                    100,
                    int(expected * 0.1),
                )
            return True

        matching_active = [
            delivery for delivery in delivery_rows if _delivery_matches(delivery)
            and delivery.get("status") in {"claimed", "delivered_pending"}
        ]
        matching_abandoned = [
            delivery for delivery in delivery_rows if _delivery_matches(delivery)
            and delivery.get("status") == "abandoned"
        ]
        delivery = (
            matching_active[0]
            if matching_active
            else (matching_abandoned[-1] if matching_abandoned else None)
        )
        if delivery:
            delivery_media_ids = normalize_media_ids(delivery.get("media_ids") or [])
            pending = {
                "reference": delivery.get("reference"),
                "media_id": delivery_media_ids[0] if delivery_media_ids else None,
                "media_ids": delivery_media_ids,
                "price": float(delivery.get("price_cents") or 0) / 100,
                "price_cents": int(delivery.get("price_cents") or 0),
                "source": delivery.get("source"),
                "set_id": delivery.get("set_id"),
                "step_index": delivery.get("step_index"),
                "platform_message_id": delivery.get("platform_message_id"),
                "sent_at": delivery.get("delivered_at") or delivery.get("claimed_at"),
            }
        else:
            pending = fan_row.get("pending_ppv_check") or {}
        expected_cents = int(
            pending.get("price_cents")
            or round(float(pending.get("price") or 0) * 100)
        )
        if expected_cents and price_cents:
            delta = abs(expected_cents - price_cents)
            if delta > max(100, int(expected_cents * 0.1)):
                print(
                    f"[PPV WEBHOOK] unmatched price fan={fan_row['id']} "
                    f"expected={expected_cents} actual={price_cents}"
                )
                await _release()
                return {"status": "unmatched_ppv_purchase"}

        pending_media_ids = normalize_media_ids(
            pending.get("media_ids") or [pending.get("media_id")]
        )
        if (
            account_media_id
            and pending_media_ids
            and account_media_id not in pending_media_ids
        ):
            print(
                f"[PPV WEBHOOK] unmatched media fan={fan_row['id']} "
                f"expected={pending_media_ids} actual={account_media_id}"
            )
            await _release()
            return {"status": "unmatched_ppv_purchase"}

        purchase_media_id = str(
            account_media_id
            or pending.get("media_id")
        )
        if not purchase_media_id:
            await _release()
            return {"status": "invalid_ppv_purchase_event"}

        from services.suggestions import record_ppv_purchase

        try:
            await record_ppv_purchase(
                str(fan_row["id"]),
                purchase_media_id,
                (price_cents / 100.0) if price_cents else None,
                pending_override=pending,
                platform_order_id=platform_order_id or None,
            )
        except Exception:
            # The downstream effects did not all land. Release so the platform's
            # redelivery can retry rather than being told it is a duplicate.
            await _release()
            raise
        if purchase_claim == "claimed":
            await _complete_platform_purchase(creator_id, platform_order_id)
        print(
            f"[PPV WEBHOOK] confirmed fan={fan_row['id']} "
            f"media={purchase_media_id} cents={price_cents}"
        )
        return {"status": "ok"}

    if event == "tips.received":
        # The current documented tip payload identifies the connected creator
        # but not the sending fan. Never guess by searching platform_fan_id
        # globally; a later transaction sync can attribute it safely.
        print(
            f"[TIP WEBHOOK] reconciliation required creator={creator.get('id')} "
            f"correlation={data.get('correlationId')} cents={data.get('amount')}"
        )
        return {"status": "queued_for_reconciliation"}

    if event == "subscriptions.new":
        if not creator or not api_account_id:
            return {"status": "creator_not_found"}
        if not apifansly_enabled():
            return {"status": "skipped", "reason": REASON_DISABLED}
        from services.fansly_audience import sync_fansly_audience

        spawn(
            sync_fansly_audience(str(creator["id"]), api_account_id),
            name="sync_fansly_audience",
        )
        return {"status": "audience_sync_scheduled"}

    if event != "messages.received":
        return {"status": "skipped"}

    platform_fan_id = str(data.get("senderId", ""))
    message_content = (data.get("content") or "").strip()
    group_id = data.get("groupId", "")
    message_id = data.get("id", "")

    attachments_raw = data.get("attachments")
    if attachments_raw is None:
        attachments_raw = []
    elif not isinstance(attachments_raw, list):
        attachments_raw = [attachments_raw]
    has_attachments = len(attachments_raw) > 0

    if not platform_fan_id:
        return {"status": "skipped"}
    if not message_content and not has_attachments:
        return {"status": "skipped"}

    interactions = data.get("interactions") or []
    creator_platform_id = str(creator.get("fansly_account_id") or "")
    if not creator_platform_id and interactions:
        creator_platform_id = str(interactions[0].get("userId", "") or "")

    if not creator_platform_id:
        return {"status": "skipped"}

    if platform_fan_id == creator_platform_id:
        return {"status": "skipped"}

    # Outgoing message capture disabled — fan messages create chats on first message

    print(
        f"[WEBHOOK] message_id={message_id} fan={platform_fan_id} "
        f"creator_platform={creator_platform_id} content={message_content[:50]}"
    )

    if not creator:
        # Backward-compatible fallback for older webhook deliveries that did not
        # include the top-level API account identifier.
        creator_result = await asyncio.to_thread(
            lambda: db.table("creators")
            .select("id, auto_mode, auto_mode_new_fans, fansly_account_id")
            .eq("fansly_account_id", creator_platform_id)
            .limit(1)
            .execute()
        )
        creator = (creator_result.data or [{}])[0]
    if not creator:
        print(f"[WEBHOOK] creator not found for platform_id={creator_platform_id}")
        return {"status": "creator_not_found"}

    creator_id = creator["id"]
    auto_mode = creator.get("auto_mode", False)
    mid = str(message_id) if message_id else ""

    # ---- Durable acceptance boundary -------------------------------------
    #
    # Everything from here to the 2xx is the minimum needed to make the event
    # survive a restart: identify the fan, persist the message under the
    # platform-message unique index, and ensure exactly one processing
    # obligation exists. Media resolution (a live API Fansly call), the
    # analyzer, the writer and Auto scheduling all moved into the durable
    # PROCESS_INBOUND_MESSAGE action.
    #
    # If any of it fails we must NOT acknowledge: a 5xx makes the platform
    # redeliver, and redelivery is now harmless. Acknowledging a lost event is
    # not.
    try:
        fan = await get_fan(creator_id, platform_fan_id)
        if not fan:
            fan = await create_fan(
                creator_id, platform_fan_id, f"Fan_{platform_fan_id[-6:]}"
            )
            spawn(
                _enrich_fan_profile(fan.id, creator_id, platform_fan_id),
                name="enrich_fan_profile",
            )

        if group_id:
            await asyncio.to_thread(
                lambda: db.table("fans")
                .update({"fansly_group_id": str(group_id)})
                .eq("id", fan.id)
                .execute()
            )

        write = await accept_inbound_message(
            fan_id=str(fan.id),
            creator_id=str(creator_id),
            content=message_content,
            platform_message_id=mid,
            group_id=str(group_id or ""),
            api_account_id=api_account_id,
            creator_platform_id=creator_platform_id,
            auto_mode=bool(auto_mode),
            attachments=attachments_raw,
        )
    except Exception as exc:
        # Never acknowledge an event we did not durably accept.
        print(
            f"[WEBHOOK INGEST ERROR] fan={platform_fan_id} message={mid}: "
            f"{type(exc).__name__}: {exc}"
        )
        raise HTTPException(
            status_code=503,
            detail="Could not durably accept the event; please redeliver.",
        )

    # Durable. Start the worker's next cycle now rather than at its idle poll.
    notify_scheduled_worker()
    if not write.inserted:
        print(
            f"[WEBHOOK] duplicate platform message creator={creator_id} "
            f"message={mid} — pipeline not re-run"
        )
        return {"status": "duplicate"}
    return {"status": "accepted"}


@app.delete(
    "/creators/{creator_id}",
    dependencies=[Depends(require_creator_path_access)],
)
async def delete_creator(creator_id: str) -> dict:
    db = get_supabase()

    await asyncio.to_thread(
        lambda cid=creator_id: db.table("reengagement_log").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("reengagement_settings").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("creator_vault_media").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("fan_lists").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("blocked_words").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("scripts").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("ppv_offers").delete().eq("creator_id", cid).execute()
    )

    # Paginated: deleting a creator has to reach every one of their fans. A
    # truncated read left rows behind and the creator delete then failed on a
    # foreign key, or worse, succeeded and orphaned them.
    fan_rows = await fetch_all_rows_async(
        lambda start, end: db.table("fans")
        .select("id")
        .eq("creator_id", creator_id)
        .order("id")
        .range(start, end)
        .execute()
    )
    fan_ids = [f["id"] for f in fan_rows]

    for fan_id in fan_ids:
        await asyncio.to_thread(
            lambda fid=fan_id: db.table("suggestions").delete().eq("fan_id", fid).execute()
        )
        await asyncio.to_thread(
            lambda fid=fan_id: db.table("messages").delete().eq("fan_id", fid).execute()
        )
        await asyncio.to_thread(
            lambda fid=fan_id: db.table("fan_list_members").delete().eq("fan_id", fid).execute()
        )

    await asyncio.to_thread(
        lambda cid=creator_id: db.table("fans").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("chatter_creators").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("creators").delete().eq("id", cid).execute()
    )

    return {"status": "ok"}


@app.get("/my-creators")
async def get_my_creators(request: Request, user_id: str | None = None) -> dict:
    operator_id = dashboard_user_id(request) or user_id
    if not operator_id:
        raise HTTPException(status_code=401, detail="Missing dashboard user session")
    db = get_supabase()
    # Both of these are selects, so a lost connection costs nothing but a
    # repeat. Before this they were bare to_thread calls: one PostgREST
    # connection recycle mid-request and the dashboard got a 500 and told the
    # operator the database was gone, when the next connection would have
    # answered immediately.
    links = await retry_db_read(
        lambda: db.table("chatter_creators")
        .select("creator_id")
        .eq("chatter_id", operator_id)
        .execute(),
        label="my_creators.links",
    )
    creator_ids = [r["creator_id"] for r in (links.data or [])]
    if not creator_ids:
        return {"creators": []}

    creators = await retry_db_read(
        lambda: db.table("creators")
        .select("id, platform_username, fansly_account_id, apifansly_account_id, persona, auto_mode")
        .in_("id", creator_ids)
        .execute(),
        label="my_creators.creators",
    )
    return {"creators": creators.data or []}


class CreatorAutoModeRequest(BaseModel):
    enabled: bool


class FanAutoModeRequest(BaseModel):
    auto_mode: bool | None


async def _creator_auto_availability(creator_id: str) -> dict:
    result = await asyncio.to_thread(
        lambda: get_supabase().table("vault_sets")
        .select("id", count="exact")
        .eq("creator_id", creator_id)
        .eq("status", "approved")
        .limit(1)
        .execute()
    )
    count = int(result.count or 0)
    if not apifansly_enabled():
        # Real Full Auto has nowhere to deliver, so it is not available for real
        # fans no matter how many sets are approved. Assisted generation is
        # untouched, and the owner-only simulator is an explicit exception that
        # never routes through this check.
        return {
            "auto_available": False,
            "approved_sets": count,
            "reason": "connector_disabled",
        }
    return {"auto_available": count > 0, "approved_sets": count}


def _auto_locked_detail(availability: dict) -> str:
    """Say which of the two reasons locked Auto, rather than always blaming sets."""
    if availability.get("reason") == "connector_disabled":
        return (
            "Auto mode is unavailable because the API Fansly connector is "
            "disabled for this deployment."
        )
    return "Auto mode is locked until at least one vault set is approved."


@app.get(
    "/creator/{creator_id}/auto-availability",
    dependencies=[Depends(require_creator_path_access)],
)
async def auto_availability(creator_id: str) -> dict:
    """Auto-mode is only available when at least one approved set exists.
    Single source of truth for the dashboard's auto gate + a backend guard."""
    return await _creator_auto_availability(creator_id)


@app.put(
    "/creator/{creator_id}/auto-mode",
    dependencies=[Depends(require_creator_path_access)],
)
async def update_creator_auto_mode(
    creator_id: str,
    request: CreatorAutoModeRequest,
) -> dict:
    availability = await _creator_auto_availability(creator_id)
    if request.enabled and not availability["auto_available"]:
        raise HTTPException(status_code=409, detail=_auto_locked_detail(availability))
    await asyncio.to_thread(
        lambda: get_supabase().table("creators")
        .update({"auto_mode": request.enabled})
        .eq("id", creator_id)
        .execute()
    )
    return {
        "status": "ok",
        "auto_mode": request.enabled,
        **availability,
    }


@app.put(
    "/fan/{fan_id}/auto-mode",
    dependencies=[Depends(require_fan_path_access)],
)
async def update_fan_auto_mode(
    fan_id: str,
    request: FanAutoModeRequest,
) -> dict:
    db = get_supabase()
    fan_result = await asyncio.to_thread(
        lambda: db.table("fans")
        .select("creator_id")
        .eq("id", fan_id)
        .single()
        .execute()
    )
    creator_id = str((fan_result.data or {}).get("creator_id") or "")
    if not creator_id:
        raise HTTPException(status_code=404, detail="Fan not found.")
    availability = await _creator_auto_availability(creator_id)
    if request.auto_mode is True and not availability["auto_available"]:
        raise HTTPException(status_code=409, detail=_auto_locked_detail(availability))
    await asyncio.to_thread(
        lambda: db.table("fans")
        .update({"auto_mode": request.auto_mode})
        .eq("id", fan_id)
        .execute()
    )
    return {
        "status": "ok",
        "auto_mode": request.auto_mode,
        **availability,
    }


@app.get(
    "/creator/{creator_id}/commercial-policy",
    dependencies=[Depends(require_creator_path_access)],
)
async def read_commercial_policy(creator_id: str) -> dict:
    from db.commercial_queries import get_creator_policy

    policy = await get_creator_policy(creator_id)
    return {"creator_id": creator_id, "policy": policy.model_dump(mode="json")}


@app.put(
    "/creator/{creator_id}/commercial-policy",
    dependencies=[Depends(require_creator_path_access)],
)
async def update_commercial_policy(creator_id: str, policy: CreatorPolicy) -> dict:
    from db.commercial_queries import save_creator_policy

    saved = await save_creator_policy(creator_id, policy)
    return {"status": "ok", "creator_id": creator_id, "policy": saved.model_dump(mode="json")}


@app.get(
    "/creator/{creator_id}/voice-calibration",
    dependencies=[Depends(require_creator_path_access)],
)
async def read_voice_calibration(creator_id: str) -> dict:
    persona, candidates = await asyncio.gather(
        get_creator_persona(creator_id),
        list_voice_calibration_candidates(creator_id),
    )
    persona = persona or Persona()
    approved = set(persona.voice_calibration_message_ids)
    return {
        "creator_id": creator_id,
        "beta": True,
        "enabled": persona.voice_calibration_enabled,
        "approved_message_ids": persona.voice_calibration_message_ids,
        "approved_samples": persona.voice_calibration_samples,
        "candidates": [
            {**candidate, "approved": candidate["id"] in approved}
            for candidate in candidates
        ],
    }


@app.put(
    "/creator/{creator_id}/voice-calibration",
    dependencies=[Depends(require_creator_path_access)],
)
async def update_voice_calibration(
    creator_id: str,
    request: VoiceCalibrationUpdateRequest,
) -> dict:
    persona = await save_voice_calibration(
        creator_id,
        enabled=request.enabled,
        approved_message_ids=request.approved_message_ids,
    )
    return {
        "status": "ok",
        "creator_id": creator_id,
        "beta": True,
        "enabled": persona.voice_calibration_enabled,
        "approved_message_ids": persona.voice_calibration_message_ids,
        "approved_samples": persona.voice_calibration_samples,
    }


@app.get(
    "/creator/{creator_id}/auto-audience-policy",
    dependencies=[Depends(require_creator_path_access)],
)
async def read_auto_audience_policy(creator_id: str) -> dict:
    db = get_supabase()
    result = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("auto_audience_policy")
        .eq("id", creator_id)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="creator not found")
    try:
        policy = AutoAudiencePolicy(**(result.data.get("auto_audience_policy") or {}))
    except Exception:
        policy = AutoAudiencePolicy()
    return {"creator_id": creator_id, "policy": policy.model_dump(mode="json")}


@app.put(
    "/creator/{creator_id}/auto-audience-policy",
    dependencies=[Depends(require_creator_path_access)],
)
async def update_auto_audience_policy(
    creator_id: str,
    policy: AutoAudiencePolicy,
) -> dict:
    db = get_supabase()
    result = await asyncio.to_thread(
        lambda: db.table("creators")
        .update({"auto_audience_policy": policy.model_dump(mode="json")})
        .eq("id", creator_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="creator not found")
    return {"status": "ok", "creator_id": creator_id, "policy": policy.model_dump(mode="json")}


@app.get(
    "/creator/{creator_id}/fansly-lists",
    dependencies=[Depends(require_creator_path_access)],
)
async def read_fansly_lists(creator_id: str) -> dict:
    """Return the creator's mirrored Fansly lists and their sync state.

    A read, not a refresh: the dashboard calls this on load, while the actual
    API Fansly work happens on the account-synchronization lifecycle or through
    the explicit refresh below.
    """
    from services.fansly_lists import lists_sync_enabled, read_lists_sync_state

    state = await read_lists_sync_state(creator_id)
    return {"creator_id": creator_id, "enabled": lists_sync_enabled(), **state}


@app.post(
    "/creator/{creator_id}/sync-fansly-lists",
    dependencies=[Depends(require_creator_path_access), Depends(require_apifansly_connector)],
)
async def sync_creator_fansly_lists(creator_id: str) -> dict:
    """Explicit operator-triggered refresh of the creator's Fansly lists.

    creator_id is never trusted on its own: require_creator_path_access has
    already confirmed the caller is assigned to this creator, and the API
    Fansly account ID is read from the creator row rather than the request.
    """
    from services.apifansly import ApiFanslyAccountAccessError
    from services.fansly_lists import (
        lists_sync_enabled,
        read_lists_sync_state,
        sync_fansly_lists_single_flight,
    )

    if not lists_sync_enabled():
        raise HTTPException(
            status_code=409,
            detail="Fansly list synchronization is disabled",
        )

    db = get_supabase()
    rows = (
        await asyncio.to_thread(
            lambda: db.table("creators")
            .select("apifansly_account_id")
            .eq("id", creator_id)
            .limit(1)
            .execute()
        )
    ).data or []
    account_id = str((rows[0] if rows else {}).get("apifansly_account_id") or "")
    if not account_id:
        raise HTTPException(
            status_code=409,
            detail="Creator is not connected to an API Fansly account",
        )

    try:
        # Runs to completion before responding, so "synced" in the response
        # means the reconciliation actually finished — not that a background
        # task was created that a restart could discard. A second operator
        # pressing Refresh gets status=already_syncing instead of a second
        # concurrent reconciliation.
        result = await sync_fansly_lists_single_flight(creator_id, account_id)
    except ApiFanslyAccountAccessError as exc:
        # Same reconnect semantics every other API Fansly access failure uses.
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    state = await read_lists_sync_state(creator_id)
    return {"creator_id": creator_id, **result, **state}


@app.get(
    "/creator/{creator_id}/auto-audience-preview",
    dependencies=[Depends(require_creator_path_access)],
)
async def preview_auto_audience(creator_id: str) -> dict:
    from collections import Counter
    from services.auto_audience import AutoAudiencePolicy, evaluate_auto_eligibility

    from core.pagination import fetch_all_rows_async

    db = get_supabase()

    # SEC-002. The membership read used to have no creator filter at all and was
    # executed with service-role credentials, so every agency's rows were pulled
    # into this request and filtered in Python. Worse, PostgREST truncates at
    # 1,000 rows *globally*, so past that point the requesting creator's own rows
    # were usually absent and the exclusion policy silently stopped applying.
    #
    # The creator's own list ids are resolved first, and the membership query is
    # constrained to them inside Postgres. Every read here is paginated with a
    # deterministic order; each is scoped to this creator.
    creator_result, fan_rows, list_rows, message_rows = await asyncio.gather(
        asyncio.to_thread(
            lambda: db.table("creators")
            .select("auto_mode, auto_audience_policy")
            .eq("id", creator_id)
            .single()
            .execute()
        ),
        fetch_all_rows_async(
            lambda start, end: db.table("fans")
            .select("id, auto_mode, total_spent, spend_tier, needs_human_review")
            .eq("creator_id", creator_id)
            .order("id")
            .range(start, end)
            .execute()
        ),
        fetch_all_rows_async(
            lambda start, end: db.table("fan_lists")
            .select("id, exclude_from_auto")
            .eq("creator_id", creator_id)
            .order("id")
            .range(start, end)
            .execute()
        ),
        # Only "has this creator ever messaged this fan" is needed. Ordering by
        # the primary key keeps paging total; ordering by fan_id alone would let
        # Postgres break ties differently between pages.
        fetch_all_rows_async(
            lambda start, end: db.table("messages")
            .select("id, fan_id")
            .eq("creator_id", creator_id)
            .eq("role", "creator")
            .order("id")
            .range(start, end)
            .execute()
        ),
    )
    creator = creator_result.data or {}
    if not creator:
        raise HTTPException(status_code=404, detail="creator not found")
    try:
        policy = AutoAudiencePolicy(**(creator.get("auto_audience_policy") or {}))
    except Exception:
        policy = AutoAudiencePolicy()
    creator_message_fans = {
        str(row.get("fan_id")) for row in message_rows if row.get("fan_id")
    }

    creator_list_ids = [str(row["id"]) for row in list_rows if row.get("id")]
    legacy_exclusions: set[str] = {
        str(row["id"]) for row in list_rows
        if row.get("id") and row.get("exclude_from_auto")
    }

    membership_rows: list[dict] = []
    if creator_list_ids:
        membership_rows = await fetch_all_rows_async(
            lambda start, end: db.table("fan_list_members")
            .select("fan_id, list_id")
            .in_("list_id", creator_list_ids)
            .order("list_id")
            .order("fan_id")
            .range(start, end)
            .execute()
        )

    memberships: dict[str, set[str]] = {}
    for row in membership_rows:
        fan_key = str(row.get("fan_id") or "")
        list_key = str(row.get("list_id") or "")
        if fan_key and list_key:
            memberships.setdefault(fan_key, set()).add(list_key)
    if legacy_exclusions:
        # sorted() so the merged list is stable across requests; set iteration
        # order is not. Only membership in this list is tested, so order is not
        # behavioural — it just makes the response reproducible.
        policy.exclude_list_ids = list(
            dict.fromkeys([*policy.exclude_list_ids, *sorted(legacy_exclusions)])
        )

    reasons: Counter[str] = Counter()
    reasons_if_creator_on: Counter[str] = Counter()
    eligible = 0
    eligible_if_creator_on = 0
    for fan in fan_rows:
        fan_id = str(fan["id"])
        eligibility_inputs = {
            "fan_auto_override": fan.get("auto_mode"),
            "needs_human_review": bool(fan.get("needs_human_review", False)),
            "policy": policy,
            "fan_list_ids": memberships.get(fan_id, set()),
            "total_spent": int(fan.get("total_spent") or 0),
            "spend_tier": str(fan.get("spend_tier") or "cold"),
            "is_new_fan": fan_id not in creator_message_fans,
        }
        result = evaluate_auto_eligibility(
            creator_auto=bool(creator.get("auto_mode", False)),
            **eligibility_inputs,
        )
        enabled_result = evaluate_auto_eligibility(
            creator_auto=True,
            **eligibility_inputs,
        )
        reasons[result.reason] += 1
        reasons_if_creator_on[enabled_result.reason] += 1
        eligible += int(result.eligible)
        eligible_if_creator_on += int(enabled_result.eligible)
    total = len(fan_rows)
    return {
        "creator_id": creator_id,
        "creator_auto_mode": bool(creator.get("auto_mode", False)),
        "eligible": eligible,
        "ineligible": total - eligible,
        "total": total,
        "reasons": dict(reasons),
        "eligible_if_creator_on": eligible_if_creator_on,
        "ineligible_if_creator_on": total - eligible_if_creator_on,
        "reasons_if_creator_on": dict(reasons_if_creator_on),
    }


@app.get(
    "/fan/{fan_id}/commercial-state",
    dependencies=[Depends(require_fan_path_access)],
)
async def read_commercial_state(fan_id: str) -> dict:
    from db.commercial_queries import get_fan_state

    state = await get_fan_state(fan_id)
    return {"fan_id": fan_id, "state": state.model_dump(mode="json")}


class CancelFollowupRequest(BaseModel):
    action_type: str | None = None


class OperatorPPVRequest(BaseModel):
    media_ids: list[str]
    price_cents: int
    message_content: str = ""
    set_id: str | None = None


class ResolvePPVApprovalRequest(BaseModel):
    resolved_by: str | None = None


class ResolveFanReviewRequest(BaseModel):
    resolution: str
    amount: float | None = None


@app.get(
    "/fan/{fan_id}/full-auto-status",
    dependencies=[Depends(require_fan_path_access)],
)
async def read_full_auto_status(fan_id: str) -> dict:
    from services.full_auto_operations import (
        FullAutoStatusUnavailable,
        get_fan_full_auto_snapshot,
    )

    try:
        return await get_fan_full_auto_snapshot(fan_id)
    except FullAutoStatusUnavailable as exc:
        print(f"[FULL AUTO STATUS] temporarily unavailable fan={fan_id}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Full Auto status is temporarily unavailable. Please retry.",
        ) from exc


@app.post(
    "/fan/{fan_id}/resolve-review",
    dependencies=[Depends(require_fan_path_access)],
)
async def resolve_review(fan_id: str, request: ResolveFanReviewRequest) -> dict:
    """Resolve a frozen conversation through a deterministic backend action."""
    from services.ppv_recovery import PPVRecoveryError, resolve_fan_review

    try:
        return await resolve_fan_review(
            fan_id,
            resolution=request.resolution,
            amount=request.amount,
        )
    except PPVRecoveryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get(
    "/creator/{creator_id}/full-auto-health",
    dependencies=[Depends(require_creator_path_access)],
)
async def read_full_auto_health(creator_id: str) -> dict:
    from services.full_auto_operations import (
        FullAutoStatusUnavailable,
        get_creator_full_auto_health,
    )

    try:
        return await get_creator_full_auto_health(creator_id)
    except FullAutoStatusUnavailable as exc:
        print(f"[FULL AUTO HEALTH] temporarily unavailable creator={creator_id}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Full Auto health is temporarily unavailable. Please retry.",
        ) from exc


@app.get(
    "/creator/{creator_id}/fansly-integration-health",
    dependencies=[Depends(require_creator_path_access)],
)
async def read_fansly_integration_health(creator_id: str) -> dict:
    """Verify that the current API key can access this creator connection."""
    import httpx

    db = get_supabase()
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("platform, fansly_account_id, apifansly_account_id")
        .eq("id", creator_id)
        .single()
        .execute()
    )
    creator = creator_row.data or {}
    account_id = str(creator.get("apifansly_account_id") or "").strip()
    common = {
        "creator_id": creator_id,
        "platform": creator.get("platform"),
        "connected": bool(account_id),
        "stored_fansly_account_id": bool(creator.get("fansly_account_id")),
    }
    if not apifansly_enabled():
        # Deliberately offline. Reported as its own state, and without spending
        # a provider call to rediscover what configuration already tells us.
        # "disabled" is not "access_denied": nothing needs reconnecting.
        return {
            **common,
            "configured": True,
            "accessible": False,
            "status": "connector_disabled",
            "requires_reconnect": False,
            "detail": (
                "The API Fansly connector is intentionally disabled "
                "(APIFANSLY_ENABLED=false)."
            ),
        }
    if not account_id:
        return {
            **common,
            "configured": True,
            "accessible": False,
            "status": "not_connected",
            "requires_reconnect": True,
            "detail": "Connect this creator to API Fansly.",
        }

    try:
        await apifansly_current_account(str(account_id))
    except ApiFanslyConfigurationError as exc:
        return {
            **common,
            "configured": False,
            "accessible": False,
            "status": "misconfigured",
            "requires_reconnect": False,
            "detail": str(exc),
        }

    except ApiFanslyAccountAccessError as exc:
        return {
            **common,
            "configured": True,
            "accessible": False,
            "status": "access_denied",
            "requires_reconnect": True,
            "detail": str(exc),
        }
    except (httpx.HTTPError, RuntimeError) as exc:
        return {
            **common,
            "configured": True,
            "accessible": False,
            "status": "upstream_error",
            "requires_reconnect": False,
            "detail": str(exc),
        }

    return {
        **common,
        "configured": True,
        "accessible": True,
        "status": "healthy",
        "requires_reconnect": False,
        "detail": "API Fansly account access is healthy.",
    }


@app.get(
    "/creator/{creator_id}/ppv-approvals",
    dependencies=[Depends(require_creator_path_access)],
)
async def read_ppv_approvals(creator_id: str, status: str = "pending") -> dict:
    from services.ppv_delivery import list_ppv_approval_requests

    if status not in {"pending", "sending", "sent", "rejected", "cancelled", "failed"}:
        raise HTTPException(status_code=400, detail="invalid approval status")
    return {
        "creator_id": creator_id,
        "status": status,
        "requests": await list_ppv_approval_requests(creator_id, status=status),
    }


@app.post(
    "/ppv-approvals/{request_id}/approve",
    dependencies=[Depends(require_ppv_approval_path_access)],
)
async def approve_ppv_approval(
    request_id: str,
    request: ResolvePPVApprovalRequest,
) -> dict:
    from services.ppv_delivery import PPVDeliveryError, approve_ppv_request

    try:
        return await approve_ppv_request(request_id, resolved_by=request.resolved_by)
    except PPVDeliveryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post(
    "/ppv-approvals/{request_id}/reject",
    dependencies=[Depends(require_ppv_approval_path_access)],
)
async def reject_ppv_approval(
    request_id: str,
    request: ResolvePPVApprovalRequest,
) -> dict:
    from services.ppv_delivery import PPVDeliveryError, reject_ppv_request

    try:
        return await reject_ppv_request(request_id, resolved_by=request.resolved_by)
    except PPVDeliveryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get(
    "/fan/{fan_id}/operator-ppv-options",
    dependencies=[Depends(require_creator_fan_access)],
)
async def read_operator_ppv_options(fan_id: str, creator_id: str) -> dict:
    """Return approved sets and vault media with authoritative sale/send state."""
    db = get_supabase()
    fan_row = await asyncio.to_thread(
        lambda: db.table("fans")
        .select("creator_id, sales_log, not_sold_log, pending_ppv_check")
        .eq("id", fan_id)
        .single()
        .execute()
    )
    fan = fan_row.data or {}
    if str(fan.get("creator_id") or "") != str(creator_id):
        raise HTTPException(status_code=404, detail="fan not found for creator")

    try:
        from services.ppv_delivery_ledger import list_fan_deliveries

        vault_rows, set_rows, message_rows, deliveries = await asyncio.gather(
            asyncio.to_thread(
                lambda: db.table("creator_vault_media")
                .select(
                    "id, fansly_media_id, media_id, url, thumbnail_url, mimetype, filename, "
                    "album_title, content_category, ai_description, price_min, price_max, is_active"
                )
                .eq("creator_id", creator_id)
                .eq("is_active", True)
                .execute()
            ),
            asyncio.to_thread(
                lambda: db.table("vault_sets")
                .select(
                    "id, title, description, media_ids, suggested_price, base_price_cents, min_price_cents, "
                    "max_price_cents, dynamic_pricing_enabled, tags, status"
                )
                .eq("creator_id", creator_id)
                .eq("status", "approved")
                .order("created_at", desc=True)
                .execute()
            ),
            asyncio.to_thread(
                lambda: db.table("messages")
                .select("media_context")
                .eq("creator_id", creator_id)
                .eq("fan_id", fan_id)
                .eq("role", "creator")
                .not_.is_("media_context->ppv", "null")
                .order("sent_at", desc=True)
                .execute()
            ),
            list_fan_deliveries(creator_id, fan_id),
        )
    except Exception as exc:
        print(f"[OPERATOR PPV OPTIONS] fan={fan_id} creator={creator_id} error={exc}", flush=True)
        raise HTTPException(
            status_code=502,
            detail="Could not load the creator vault for this PPV.",
        ) from exc

    from services.ppv_status import build_media_status_by_id

    status_by_id = build_media_status_by_id(
        deliveries=deliveries,
        message_rows=message_rows.data or [],
        sales_log=fan.get("sales_log") or [],
        not_sold_log=fan.get("not_sold_log") or [],
        pending_ppv=fan.get("pending_ppv_check") or None,
    )

    media = []
    for row in (vault_rows.data or []):
        external_id = str(row.get("fansly_media_id") or row.get("media_id") or "")
        status = status_by_id.get(external_id, "unused")
        media.append({**row, "external_media_id": external_id, "fan_sale_status": status})

    return {
        "fan_id": fan_id,
        "creator_id": creator_id,
        "has_payment_pending": bool(fan.get("pending_ppv_check")),
        "media": media,
        "approved_sets": set_rows.data or [],
    }


@app.post(
    "/fan/{fan_id}/operator-ppv",
    dependencies=[Depends(require_creator_fan_access), Depends(require_apifansly_connector)],
)
async def send_operator_ppv(
    fan_id: str,
    creator_id: str,
    request: OperatorPPVRequest,
) -> dict:
    from services.ppv_delivery import (
        PPVDeliveryError,
        cancel_pending_ppv_approvals,
        send_locked_ppv,
    )

    exact_ids = normalize_media_ids(request.media_ids)
    if not exact_ids:
        raise HTTPException(status_code=400, detail="select at least one media item")
    if request.price_cents <= 0 or request.price_cents > 1_000_000:
        raise HTTPException(status_code=400, detail="enter a valid PPV price")

    db = get_supabase()
    rows = await asyncio.to_thread(
        lambda: db.table("creator_vault_media")
        .select("fansly_media_id, media_id, price_min, price_max, is_active")
        .eq("creator_id", creator_id)
        .in_("fansly_media_id", exact_ids)
        .execute()
    )
    found = {
        str(row.get("fansly_media_id") or row.get("media_id") or ""): row
        for row in (rows.data or [])
    }
    if set(found) != set(exact_ids):
        raise HTTPException(status_code=400, detail="one or more media items are not in this creator's vault")
    if any(row.get("is_active") is False for row in found.values()):
        raise HTTPException(status_code=400, detail="inactive media cannot be sent")

    if request.set_id:
        set_row = await asyncio.to_thread(
            lambda: db.table("vault_sets")
            .select("media_ids, status, min_price_cents, max_price_cents")
            .eq("id", request.set_id)
            .eq("creator_id", creator_id)
            .single()
            .execute()
        )
        approved_set = set_row.data or {}
        if approved_set.get("status") != "approved":
            raise HTTPException(status_code=400, detail="the selected set is not approved")
        if set(normalize_media_ids(approved_set.get("media_ids") or [])) != set(exact_ids):
            raise HTTPException(status_code=400, detail="selected media no longer matches the approved set")
        minimum = int(approved_set.get("min_price_cents") or 0)
        maximum = int(approved_set.get("max_price_cents") or 0)
        if minimum and request.price_cents < minimum:
            raise HTTPException(status_code=400, detail=f"price is below the set minimum (${minimum / 100:g})")
        if maximum and request.price_cents > maximum:
            raise HTTPException(status_code=400, detail=f"price is above the set maximum (${maximum / 100:g})")

    await cancel_pending_ppv_approvals(fan_id, reason="operator_sent_manual_ppv")
    try:
        return await send_locked_ppv(
            creator_id=creator_id,
            fan_id=fan_id,
            media_ids=exact_ids,
            price_cents=request.price_cents,
            message_content=request.message_content,
            source="operator",
            was_ai_suggested=False,
            set_id=request.set_id,
            step_index=None,
        )
    except PPVDeliveryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post(
    "/fan/{fan_id}/cancel-followup",
    dependencies=[Depends(require_fan_path_access)],
)
async def cancel_followup(
    fan_id: str,
    request: CancelFollowupRequest,
) -> dict:
    from services.full_auto_operations import cancel_fan_followup

    return await cancel_fan_followup(fan_id, request.action_type)


@app.post(
    "/fan/{fan_id}/retry-followup/{action_id}",
    dependencies=[Depends(require_fan_path_access)],
)
async def retry_followup(fan_id: str, action_id: str) -> dict:
    from services.full_auto_operations import retry_fan_followup

    return await retry_fan_followup(fan_id, action_id)


@app.post(
    "/plan-session/{creator_id}/{fan_id}",
    dependencies=[Depends(require_creator_fan_access)],
)
async def plan_session(
    creator_id: str,
    fan_id: str,
    request: Request = None,
) -> dict:
    """Create a coherent, confirmed-budget multi-step paid session."""
    body = {}
    if request is not None:
        try:
            body = await request.json()
        except Exception:
            body = {}

    selected_set_ids = body.get("selected_set_ids") or []
    if not selected_set_ids and body.get("selected_set_id"):
        selected_set_ids = [body["selected_set_id"]]

    from services.session_planner import plan_session_for_fan
    return await plan_session_for_fan(
        creator_id,
        fan_id,
        selected_set_ids=selected_set_ids,
        selected_price_cents=body.get("selected_price_cents"),
        confirmed_kinks=body.get("confirmed_kinks") or [],
    )


@app.get(
    "/session/{fan_id}",
    dependencies=[Depends(require_fan_path_access)],
)
async def get_session(fan_id: str) -> dict:
    from db.queries import get_fan_session

    session = await get_fan_session(fan_id)
    return {"session": session}


@app.post(
    "/session/{fan_id}/advance",
    dependencies=[Depends(require_fan_path_access)],
)
async def advance_session(fan_id: str) -> dict:
    """Mark current step sent. Purchase confirmation advances the index."""
    from db.queries import get_fan_session, save_fan_session
    from services.session_lifecycle import mark_step_sent

    session = await get_fan_session(fan_id)
    if not session:
        return {"status": "no_session"}
    try:
        session = mark_step_sent(session)
    except ValueError as exc:
        return {"status": "blocked", "message": str(exc)}
    await save_fan_session(fan_id, session)
    return {
        "status": "ok",
        "current_index": session.get("current_index", 0),
        "awaiting_purchase_index": session.get("awaiting_purchase_index"),
        "remaining": len(session.get("plan") or []) - int(session.get("current_index", 0) or 0),
    }


@app.post(
    "/session/{fan_id}/purchased/{media_id}",
    dependencies=[Depends(require_fan_path_access)],
)
async def mark_session_purchased(
    fan_id: str,
    media_id: str,
    amount: float | None = None,
) -> dict:
    from services.suggestions import record_ppv_purchase

    await record_ppv_purchase(fan_id, media_id, amount)
    return {"status": "ok", "fan_id": fan_id, "media_id": media_id}


@app.post(
    "/fan/{fan_id}/record-purchase/{media_id}",
    dependencies=[Depends(require_fan_path_access)],
)
async def record_purchase_endpoint(fan_id: str, media_id: str, amount: float | None = None):
    from services.suggestions import record_ppv_purchase
    await record_ppv_purchase(fan_id, media_id, amount)
    return {"status": "ok", "fan_id": fan_id, "media_id": media_id}


@app.delete(
    "/session/{fan_id}",
    dependencies=[Depends(require_fan_path_access)],
)
async def clear_session(fan_id: str) -> dict:
    """End and clear the active session."""
    from db.queries import save_fan_session

    await save_fan_session(fan_id, None)
    return {"status": "ok"}


@app.post(
    "/enrich-fan/{fan_id}",
    dependencies=[Depends(require_fan_path_access)],
)
async def enrich_fan_endpoint(fan_id: str) -> dict:
    db = get_supabase()
    fan_row = await asyncio.to_thread(
        lambda: db.table("fans")
        .select("platform_fan_id, creator_id")
        .eq("id", fan_id)
        .single()
        .execute()
    )
    data = fan_row.data or {}
    platform_fan_id = data.get("platform_fan_id")
    creator_id = data.get("creator_id")
    if not platform_fan_id or not creator_id:
        return {"status": "error", "message": "fan not found"}
    await _enrich_fan_profile(fan_id, creator_id, platform_fan_id)
    return {"status": "ok"}


# --- Owner-only Full Auto simulator ----------------------------------------
#
# Private to allowlisted Supabase accounts. Every route below is invisible and
# inaccessible to ordinary agency tenants, and every rejection is the same 404
# the tenancy layer uses, so a caller cannot probe which condition it failed or
# learn that another tenant's creator or fan exists.
#
# The frontend hides the simulator using GET /simulation-capabilities, but that
# is convenience only: nothing below trusts the client.


class SimulateInboundRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    fast: bool = True


class SimulateDeclineRequest(BaseModel):
    reason: str = "simulated_decline"


async def _creator_ids_for_user_cached(request: Request) -> set[str]:
    """The caller's assigned creator ids, reusing the tenancy layer's cache.

    Goes through core.tenancy rather than re-querying chatter_creators here, so
    the simulator can never see a different set of creators than every other
    route in the application.
    """
    from core.tenancy import _creator_ids_for_user

    cached = getattr(request.state, "allowed_creator_ids", None)
    if cached is None:
        cached = await _creator_ids_for_user(str(dashboard_user_id(request) or ""))
        request.state.allowed_creator_ids = cached
    return set(cached)


async def _require_simulatable_fan(
    request: Request,
    creator_id: str,
    fan_id: str,
) -> dict:
    """All six simulation preconditions, in order, with one shared rejection.

    1-3 (flag, authenticated user, allowlist) come from core.simulation;
    4 and 5 reuse the ordinary tenancy helper, so the simulator is subject to
    exactly the same tenancy model as every other creator route rather than a
    parallel one; 6 is the ``test_`` platform-fan boundary, enforced here on
    every simulation mutation so knowing a real fan's UUID is never enough.
    """
    from core.simulation import is_simulatable_fan, not_found, require_simulation_user

    await require_simulation_user(request)
    await require_creator_fan_access(request, creator_id, fan_id)

    row = await asyncio.to_thread(
        lambda: get_supabase()
        .table("fans")
        .select("id, creator_id, platform_fan_id, display_name")
        .eq("id", fan_id)
        .eq("creator_id", creator_id)
        .limit(1)
        .execute()
    )
    rows = row.data or []
    if not rows:
        raise not_found()
    fan = rows[0]
    if not is_simulatable_fan(fan.get("platform_fan_id")):
        # A real Fansly fan must never become eligible. Same 404 as everything
        # else: the caller learns nothing about why.
        raise not_found()
    return fan


@app.get("/simulation-capabilities")
async def simulation_capabilities(request: Request) -> dict:
    """What privileged local tooling this authenticated account may use.

    Returns a boolean and nothing else. It never exposes the allowlist, the
    environment variables, any user id, or why another user is not allowed.
    """
    from core.simulation import request_may_simulate

    return {"auto_simulation": bool(request_may_simulate(request))}


@app.get("/simulation/creators")
async def simulation_creators(request: Request) -> dict:
    """Creators the caller may simulate against, each with its test fans only.

    Scoped by the caller's ordinary creator assignments, then filtered to
    ``test_`` fans, so the simulator's pickers cannot enumerate real fans.
    """
    from core.simulation import TEST_FAN_PREFIX, require_simulation_user

    await require_simulation_user(request)
    allowed = await _creator_ids_for_user_cached(request)
    if not allowed:
        return {"creators": []}

    db = get_supabase()
    ids = sorted(allowed)
    try:
        # Both are selects, so a lost PostgREST connection costs nothing but a
        # repeat. Bare to_thread calls here meant one connection recycle
        # mid-request turned into an error the dashboard renders as "you have
        # no creators", which is indistinguishable from the real empty case.
        creators_result, fans_result = await asyncio.gather(
            retry_db_read(
                # The creator display field is platform_username. `name` exists
                # in db/ci_baseline_schema.sql but NOT in production, so
                # selecting it returned PostgREST 42703 and this route 404'd
                # while /simulation-capabilities answered 200. The CI fixture is
                # a test fixture, not the authoritative production schema.
                lambda: db.table("creators")
                .select("id, platform_username")
                .in_("id", ids)
                .execute(),
                label="simulation.creators",
            ),
            retry_db_read(
                lambda: db.table("fans")
                .select("id, display_name, creator_id, platform_fan_id")
                .in_("creator_id", ids)
                .like("platform_fan_id", f"{TEST_FAN_PREFIX}%")
                .order("display_name")
                .limit(500)
                .execute(),
                label="simulation.test_fans",
            ),
        )
    except Exception as exc:
        print(f"[SIMULATION] creator listing failed: {exc}")
        # 503, not 404. Authorization already passed above, so this caller is a
        # verified allowlisted owner and a distinguishable error tells them
        # nothing they may not know. A 404 here would be read as an empty
        # creator list, which is a lie about the data rather than a report of a
        # failed read.
        raise HTTPException(
            status_code=503,
            detail="Could not read creators for simulation. Please retry.",
        ) from exc

    creators = creators_result.data or []
    fans = fans_result.data or []

    by_creator: dict[str, list[dict]] = {}
    for fan in fans:
        # Defence in depth: the LIKE above is a database filter, and this is the
        # same boundary applied in Python so a driver quirk cannot widen it.
        if not str(fan.get("platform_fan_id") or "").startswith(TEST_FAN_PREFIX):
            continue
        by_creator.setdefault(str(fan.get("creator_id")), []).append(
            {
                "id": str(fan.get("id")),
                "display_name": fan.get("display_name") or str(fan.get("id")),
            }
        )
    return {
        "creators": [
            {
                "id": str(creator.get("id")),
                # The API contract stays "name" — the dashboard type is
                # unchanged. Only the column it is read from is corrected.
                "name": creator.get("platform_username") or str(creator.get("id")),
                "test_fans": by_creator.get(str(creator.get("id")), []),
            }
            for creator in creators
        ]
    }


@app.post("/creator/{creator_id}/fan/{fan_id}/simulate-inbound")
async def simulate_inbound(
    creator_id: str,
    fan_id: str,
    body: SimulateInboundRequest,
    request: Request,
) -> dict:
    """Persist one fan message and run the REAL Full Auto turn it triggers.

    Deliberately unaffected by APIFANSLY_ENABLED: the whole point is that this
    works while the connector is off, because it never touches it.
    """
    from services.suggestions import run_simulated_inbound

    fan = await _require_simulatable_fan(request, creator_id, fan_id)
    print(
        f"[SIMULATION] inbound creator={creator_id} fan={fan_id} "
        f"platform_fan={fan.get('platform_fan_id')} fast={body.fast}"
    )
    return await run_simulated_inbound(
        fan_id=fan_id,
        creator_id=creator_id,
        message=body.message,
        fast=body.fast,
    )


@app.post("/creator/{creator_id}/fan/{fan_id}/simulate-purchase")
async def simulate_purchase(creator_id: str, fan_id: str, request: Request) -> dict:
    """Confirm the pending simulated PPV through the real purchase transition.

    Reuses record_ppv_purchase rather than reimplementing the state machine, so
    the lifecycle, affordability, price-learning and session effects a real
    purchase produces are the ones observed here.
    """
    from services.suggestions import record_ppv_purchase

    await _require_simulatable_fan(request, creator_id, fan_id)
    pending = (
        await asyncio.to_thread(
            lambda: get_supabase()
            .table("fans")
            .select("pending_ppv_check")
            .eq("id", fan_id)
            .single()
            .execute()
        )
    ).data or {}
    check = pending.get("pending_ppv_check") or {}
    media_id = str(check.get("media_id") or "")
    if not media_id:
        return {"status": "no_pending_ppv", "simulation": True}
    await record_ppv_purchase(
        fan_id,
        media_id,
        float(check.get("price") or 0),
        pending_override=check,
    )
    return {
        "status": "ok",
        "simulation": True,
        "media_id": media_id,
        "amount": float(check.get("price") or 0),
    }


@app.post("/creator/{creator_id}/fan/{fan_id}/simulate-decline")
async def simulate_decline(
    creator_id: str,
    fan_id: str,
    body: SimulateDeclineRequest,
    request: Request,
) -> dict:
    """Decline the pending simulated PPV through the real decline transition."""
    from db.queries import get_fan_session, save_fan_session, set_fan_decline_lock
    from services.session_lifecycle import mark_step_declined

    await _require_simulatable_fan(request, creator_id, fan_id)
    pending = (
        await asyncio.to_thread(
            lambda: get_supabase()
            .table("fans")
            .select("pending_ppv_check")
            .eq("id", fan_id)
            .single()
            .execute()
        )
    ).data or {}
    check = pending.get("pending_ppv_check") or {}
    await set_fan_decline_lock(fan_id, check.get("price"))
    session = await get_fan_session(fan_id)
    if session and session.get("awaiting_purchase_index") is not None:
        session = mark_step_declined(session, reason=body.reason, pause=True)
        await save_fan_session(fan_id, session)
    return {"status": "ok", "simulation": True, "session": session}


async def require_local_test_endpoints() -> None:
    """SEC-005 — the /test/* helpers must not exist outside development/test.

    404 rather than 403: production should not advertise that these routes are
    implemented at all.
    """
    from core.environment import local_test_endpoints_enabled

    if not local_test_endpoints_enabled():
        raise HTTPException(status_code=404, detail="Not Found")


@app.post(
    "/test/simulate-ppv-purchase",
    dependencies=[
        Depends(require_local_test_endpoints),
        Depends(require_fan_path_access),
    ],
)
async def simulate_ppv_purchase(fan_id: str, request: Request) -> dict:
    """Dev only — simulate a fan purchasing a pending PPV."""
    from db.queries import get_fan_session, save_fan_session
    from datetime import datetime

    db = get_supabase()

    # Get pending PPV check
    fan_row = await asyncio.to_thread(
        lambda: db.table("fans")
        # sales_log must be selected: it used to be read from a row that never
        # contained it, so the append below silently replaced the fan's entire
        # sales history with one fabricated entry (SEC-005).
        .select(
            "pending_ppv_check, total_spent, active_session, ai_summary, sales_log"
        )
        .eq("id", fan_id)
        .single()
        .execute()
    )
    fan_data = fan_row.data or {}
    pending = fan_data.get("pending_ppv_check")

    if not pending:
        return {"status": "error", "message": "No pending PPV check found for this fan"}

    media_id = pending.get("media_id")
    price = pending.get("price", 0)
    current_spent = fan_data.get("total_spent") or 0
    new_spent = current_spent + int(price)

    summary = fan_data.get("ai_summary") or {}


    # Append to the existing history rather than replacing it. A non-list value
    # would otherwise be silently discarded, so refuse instead of destroying it.
    existing_sales_log = fan_data.get("sales_log")
    if existing_sales_log is None:
        existing_sales_log = []
    if not isinstance(existing_sales_log, list):
        return {
            "status": "error",
            "message": "fan sales_log is not a list; refusing to overwrite it",
        }
    sales_log = [*existing_sales_log]
    sales_log.append({
        "date": datetime.utcnow().strftime("%d.%m.%Y"),
        "item": f"PPV media {media_id}",
        "amount": int(price),
        "chatter": "AI",
    })

    def _calc_tier(spent: int) -> str:
        if spent >= 500: return "whale"
        if spent >= 100: return "active"
        if spent >= 20: return "casual"
        return "cold"

    new_tier = _calc_tier(new_spent)

    # Mark as purchased
    await asyncio.to_thread(
        lambda: db.table("fans").update({
            "total_spent": new_spent,
            "pending_ppv_check": None,
            "ai_summary": summary,
            "sales_log": sales_log,
            "spend_tier": new_tier,
        }).eq("id", fan_id).execute()
    )

    # Update session plan item as purchased
    session = await get_fan_session(fan_id)
    if session:
        for item in session.get("plan", []):
            if item.get("media_id") == media_id:
                item["purchased"] = True
        await save_fan_session(fan_id, session)

    print(f"[TEST] Simulated PPV purchase fan={fan_id} media={media_id} price=${price} new_total=${new_spent}")
    return {
        "status": "ok",
        "media_id": media_id,
        "price": price,
        "new_total_spent": new_spent,
    }


@app.post(
    "/test/inject-message",
    dependencies=[
        Depends(require_local_test_endpoints),
        Depends(require_creator_fan_access),
    ],
)
async def test_inject_message(
    fan_id: str,
    creator_id: str,
    content: str,
    auto_mode: bool | None = None,
) -> dict:
    """Dev testing only — simulate a fan message without a Fansly webhook.

    auto_mode used to be hardcoded True, so a helper call could trigger a real
    Full Auto send to a real fan for a creator whose Auto is off (SEC-005). It
    now resolves the creator's actual setting. An explicit auto_mode=false can
    force a non-delivering injection; auto_mode=true is honoured only when the
    creator really has Auto on, so the helper can never be the reason a message
    is sent.
    """
    from db.queries import save_message

    creator_auto = False
    try:
        creator_row = await asyncio.to_thread(
            lambda: get_supabase()
            .table("creators")
            .select("auto_mode")
            .eq("id", creator_id)
            .single()
            .execute()
        )
        creator_auto = bool((creator_row.data or {}).get("auto_mode", False))
    except Exception as exc:
        # Fail closed: an unreadable creator row must not authorise a send.
        print(f"[TEST INJECT] creator auto_mode unreadable creator={creator_id}: {exc}")
        creator_auto = False

    effective_auto = creator_auto if auto_mode is None else (creator_auto and auto_mode)

    await save_message(fan_id, creator_id, "fan", content, was_ai_suggested=False)
    await process_incoming_fan_message(
        fan_id,
        creator_id,
        content,
        auto_mode=effective_auto,
        message_id=None,
    )
    return {
        "status": "ok",
        "fan_id": fan_id,
        "content": content,
        "auto_mode": effective_auto,
        "creator_auto_mode": creator_auto,
    }


@app.post(
    "/generate-sets/{creator_id}",
    dependencies=[Depends(require_creator_path_access)],
)
async def generate_sets(creator_id: str) -> dict:
    from db.queries import propose_sets, propose_video_ppvs
    db = get_supabase()

    items, page = [], 0
    while True:
        rows = await asyncio.to_thread(
            lambda p=page: db.table("creator_vault_media")
            .select(
                "fansly_media_id, content_category, ai_description, explicitness_level, "
                "scene_id, scene_location, scene_outfit, scene_lighting, album_title, "
                "mimetype, price_min, price_max, tags, good_for, classification_metadata"
                ", classification_source, is_active"
            )
            .eq("creator_id", creator_id)
            .eq("is_active", True)
            .range(p * 1000, p * 1000 + 999)
            .execute()
        )
        batch = rows.data or []
        items.extend(batch)
        if len(batch) < 1000:
            break
        page += 1

    photo_proposals = propose_sets(items)
    video_proposals = propose_video_ppvs(items)

    approved_rows = await asyncio.to_thread(
        lambda: db.table("vault_sets")
        .select("media_ids")
        .eq("creator_id", creator_id)
        .eq("status", "approved")
        .execute()
    )
    approved_exact_sets = {
        tuple(sorted(normalize_media_ids(row.get("media_ids") or [])))
        for row in (approved_rows.data or [])
    }
    proposed = [
        proposal
        for proposal in [*photo_proposals, *video_proposals]
        if tuple(sorted(proposal["media_ids"])) not in approved_exact_sets
    ]

    # Wipe prior AI drafts; never touch approved or manual sets
    await asyncio.to_thread(
        lambda: db.table("vault_sets").delete()
        .eq("creator_id", creator_id).eq("status", "draft").eq("source", "ai").execute()
    )

    to_insert = []
    for proposal in proposed:
        payload = {
            "creator_id": creator_id, "description": proposal["description"],
            "title": proposal["title"], "location": proposal["location"],
            "outfit": proposal["outfit"],
            "explicit_min": proposal["explicit_min"],
            "explicit_max": proposal["explicit_max"],
            "media_ids": proposal["media_ids"],
            "preview_media_id": proposal["preview_media_id"],
            "suggested_price": proposal["suggested_price"],
            "tags": proposal["tags"],
            "metadata_version": proposal["metadata_version"],
            "status": "draft", "source": "ai",
        }
        for field in ("base_price_cents", "min_price_cents", "max_price_cents"):
            if field in proposal:
                payload[field] = proposal[field]
        to_insert.append(payload)

    inserted = 0
    for i in range(0, len(to_insert), 100):
        chunk = to_insert[i:i + 100]
        await asyncio.to_thread(lambda c=chunk: db.table("vault_sets").insert(c).execute())
        inserted += len(chunk)

    return {
        "status": "ok",
        "drafts_created": inserted,
        "photo_sets_created": sum(len(row["media_ids"]) > 1 for row in proposed),
        "video_ppvs_created": sum(
            "individual_video" in (row.get("tags") or []) for row in proposed
        ),
        "from_items": len(items),
    }


@app.post(
    "/creator/{creator_id}/vault-sets/{set_id}/generate-description",
    dependencies=[Depends(require_creator_path_access)],
)
async def generate_vault_set_description(
    creator_id: str,
    set_id: str,
) -> dict:
    """Build and save a manual set description from its classified media."""
    db = get_supabase()
    set_result = await retry_transient_db_operation(
        lambda: asyncio.to_thread(
            lambda: db.table("vault_sets")
            .select("id, media_ids")
            .eq("id", set_id)
            .eq("creator_id", creator_id)
            .limit(1)
            .execute()
        ),
        label=f"load_vault_set:{set_id}",
    )
    set_rows = set_result.data or []
    if not set_rows:
        raise HTTPException(status_code=404, detail="Vault set not found.")
    vault_set = set_rows[0]
    media_ids = normalize_media_ids(vault_set.get("media_ids") or [])
    if not media_ids:
        raise HTTPException(
            status_code=409,
            detail="Add media to this set before generating its description.",
        )

    items: list[dict] = []
    fields = (
        "fansly_media_id, content_category, ai_description, explicitness_level, "
        "scene_location, scene_outfit, scene_lighting, mimetype, tags, "
        "classification_metadata"
    )
    for start in range(0, len(media_ids), 250):
        chunk = media_ids[start:start + 250]
        result = await retry_transient_db_operation(
            lambda ids=chunk: asyncio.to_thread(
                lambda: db.table("creator_vault_media")
                .select(fields)
                .eq("creator_id", creator_id)
                .in_("fansly_media_id", ids)
                .execute()
            ),
            label=f"load_vault_set_media:{set_id}:{start}",
        )
        items.extend(result.data or [])
    if not items:
        raise HTTPException(
            status_code=409,
            detail="The selected media could not be found in this creator's vault.",
        )

    description = build_set_description(items)
    if not description:
        raise HTTPException(
            status_code=409,
            detail="The selected media needs AI categorization first.",
        )
    await retry_transient_db_operation(
        lambda: asyncio.to_thread(
            lambda: db.table("vault_sets")
            .update({
                "description": description,
                "metadata_version": VAULT_CLASSIFIER_VERSION,
            })
            .eq("id", set_id)
            .eq("creator_id", creator_id)
            .execute()
        ),
        label=f"save_vault_set_description:{set_id}",
    )
    return {
        "status": "ok",
        "set_id": set_id,
        "description": description,
        "metadata_version": VAULT_CLASSIFIER_VERSION,
        "media_count": len(items),
    }


@app.get(
    "/debug-shoot-clusters/{creator_id}",
    dependencies=[Depends(require_creator_path_access)],
)
async def debug_shoot_clusters(creator_id: str) -> dict:
    """Explain photoshoot grouping without returning raw visual vectors."""
    db = get_supabase()
    items: list[dict] = []
    page = 0
    while True:
        rows = await asyncio.to_thread(
            lambda p=page: db.table("creator_vault_media")
            .select(
                "fansly_media_id, content_category, explicitness_level, "
                "scene_location, scene_outfit, album_title, "
                "classification_metadata"
            )
            .eq("creator_id", creator_id)
            .range(p * 1000, p * 1000 + 999)
            .execute()
        )
        batch = rows.data or []
        items.extend(batch)
        if len(batch) < 1000:
            break
        page += 1
    clusters = build_shoot_clusters(items)
    summaries = [
        cluster_debug_summary(cluster)
        for cluster in clusters
    ]
    return {
        "status": "ok",
        "from_items": len(items),
        "visual_clusters": sum(
            row["method"].startswith("local_visual")
            and len(row["media_ids"]) >= 2
            for row in summaries
        ),
        "unresolved_items": sum(
            row["method"] == "unresolved"
            for row in summaries
        ),
        "clusters": summaries,
    }


@app.get(
    "/debug-scenes/{creator_id}",
    dependencies=[Depends(require_creator_path_access)],
)
async def debug_scenes(creator_id: str) -> dict:
    from db.queries import get_vault_for_session, build_scenes
    vault = await get_vault_for_session(creator_id, min_explicitness=2)
    scenes = build_scenes(vault)
    return {
        "total_items": len(vault),
        "scene_count": len(scenes),
        "scenes": [
            {k: s[k] for k in ("scene_key", "location", "outfit",
                               "explicit_min", "explicit_max", "count", "categories")}
            for s in scenes
        ],
    }


# /health and /health/ready are in _PUBLIC_PATHS so the platform healthcheck can
# reach them without a credential. Queue depths, cycle timings and gate counters
# are operational detail rather than a public fact about the deployment, so the
# full document is returned only to a caller that already holds the dashboard
# key. An anonymous prober gets the status and the reason categories, which is
# everything a healthcheck needs.
_HEALTH_PUBLIC_KEYS = (
    "status",
    "liveness",
    "checked_at",
    "degraded_reasons",
    "fatal_reasons",
)


def _health_detail_allowed(request: Request | None) -> bool:
    if request is None:
        return True
    expected = os.environ.get("DASHBOARD_API_SECRET")
    if not expected:
        return bool(_is_dev())
    supplied = request.headers.get("x-api-key")
    return bool(supplied and _consteq(supplied, expected))


def _health_payload(document: dict, request: Request | None) -> dict:
    if _health_detail_allowed(request):
        return dict(document)
    return {key: document[key] for key in _HEALTH_PUBLIC_KEYS if key in document}


@app.get("/health")
async def health(request: Request = None) -> dict:
    """Operator-facing health. Always HTTP 200 while the process is alive.

    Railway's healthcheck hits this path, so it deliberately does not fail the
    request for degraded external state. A throttled model provider must never
    restart a container that is holding a durable queue; that turns a provider
    incident into an outage. Infrastructure-fatal conditions are reported in
    ``fatal_reasons`` and are what ``/health/ready`` refuses on.
    """
    from services.operational_health import collect

    try:
        document = await collect()
    except Exception as exc:  # pragma: no cover - health must not 500
        return {
            "status": "unknown",
            "liveness": "ok",
            "error": type(exc).__name__,
            "vault_classifier_version": VAULT_CLASSIFIER_VERSION,
        }
    return {
        **_health_payload(document, request),
        "vault_classifier_version": VAULT_CLASSIFIER_VERSION,
        "vault_semantics_configured": bool(
            os.environ.get("VAULT_SEMANTIC_BASE_URL", "").strip()
        ),
    }


@app.get("/health/ready")
async def health_ready(response: Response, request: Request = None) -> dict:
    """Readiness. 503 only when the process genuinely cannot do its job.

    That is the database being unreachable, and nothing else. Queue depth and
    provider availability are reported here too, but they never change the
    status code — they are backlog signals, not reasons to take the process out
    of service.
    """
    from services.operational_health import collect

    try:
        document = await collect()
    except Exception as exc:  # pragma: no cover
        response.status_code = 503
        return {"status": "unhealthy", "error": type(exc).__name__}
    if document.get("fatal_reasons"):
        response.status_code = 503
    return _health_payload(document, request)


@app.get("/model-runtime-health")
async def model_runtime_health() -> dict:
    """Expose cached provider-model availability without spending AI tokens."""
    from services.analyzer_telemetry import analyzer_health

    return {
        **current_model_availability(),
        # REL-001 — degraded-analysis counts, so an analyzer incident is
        # countable without reading logs.
        "analyzer": analyzer_health(hours=1),
        "analyzer_24h": analyzer_health(hours=24),
    }


from routes.fansly import fansly_router

app.include_router(fansly_router)
