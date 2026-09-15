"""FastAPI entrypoint for Cleopatra.

Routes are thin and delegate all logic to services.
"""

import asyncio
import hashlib
from core.tasks import spawn
import json
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

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
from models.content_pricing import VAULT_CATEGORIES as CONTENT_CATEGORY_RANGES
from models.vault_pricing import price_bounds
from models.schemas import (
    ConversationContext,
    Fan,
    Persona,
    SuggestionRequest,
    SuggestionResponse,
)
from services.ai_stack import resolve_ai_stack
from services.fan_intelligence import learn_from_fan_message
from services.db_reliability import retry_db_read, retry_transient_db_operation
from core.apifansly_gate import (
    REASON_DISABLED,
    apifansly_enabled,
    describe_apifansly,
)
from core.simulation import exclude_simulation_fans, is_simulation_message
from core.simulation_catalog import (
    exclude_simulation_only,
    run_live_catalog_query,
)
from services.apifansly import (
    ApiFanslyAccountAccessError,
    ApiFanslyConfigurationError,
    CATEGORY_ACCOUNT,
    CATEGORY_LIVE_CHAT,
    CATEGORY_RECONCILIATION,
    CATEGORY_VAULT,
    account_media_lookup as apifansly_account_media_lookup,
    chat_message_row as apifansly_chat_message_row,
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
    record_raw_call as record_apifansly_raw_call,
    record_webhook_event as record_apifansly_webhook_event,
    send_message as send_apifansly_message,
    response_message as apifansly_response_message,
    url as apifansly_url,
    sent_message_id,
    usage_category as apifansly_usage_category,
    usage_snapshot as apifansly_usage_snapshot,
    VAULT_MEDIA_DOWNLOAD_OPERATION,
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
    ExtractedFrames,
    FrameSettings,
    build_contact_sheet,
    extract_frames,
    ffmpeg_available,
)
from services.video_semantics import (
    DEFAULT_FRAMES_PER_SHEET,
    DEFAULT_MAX_SHEETS,
    chronological_batches,
    combine_batch_observations,
    commercial_role,
    describe_video_record,
    observation_from_classification,
)
from services.media_cost_guard import (
    REASON_UNKNOWN_SIZE as MEDIA_GUARD_REASON_UNKNOWN,
    DownloadDecision,
    auto_download_limits,
    estimated_credits_for_bytes,
    evaluate_download,
    parse_content_length,
)
from services.vault_classification_state import (
    STATUS_COMPLETE as CLASSIFICATION_STATUS_COMPLETE,
    STATUS_PARTIAL as CLASSIFICATION_STATUS_PARTIAL,
    STATUS_PENDING as CLASSIFICATION_STATUS_PENDING,
    media_identity_key,
    select_items_for_classification,
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

# How the classified pixels were obtained. Stable strings: they are persisted on
# every row, reported in telemetry, and read by the operator surface, so they
# are a contract rather than log prose. The order is the cost order.
RETRIEVAL_PLATFORM_THUMBNAIL = "platform_thumbnail"
RETRIEVAL_DIRECT_CDN = "direct_cdn"
RETRIEVAL_DIRECT_FRAMES = "direct_video_frames"
RETRIEVAL_REFRESHED_URL = "refreshed_signed_url"
RETRIEVAL_APIFANSLY_DOWNLOAD = "apifansly_media_download"

# How a sampled video is broken into contact sheets. Four frames per sheet is a
# 2x2 grid at ~440px a cell, which is the largest number of moments that
# survives being one image; more than that and the model describes a mosaic.
# The sheet ceiling bounds VLM calls per video regardless of frame count.
_VIDEO_FRAMES_PER_SHEET = DEFAULT_FRAMES_PER_SHEET
_VIDEO_MAX_SHEETS = DEFAULT_MAX_SHEETS
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

    # A returning fan with a long past and almost no local context is the case
    # this exists for: Cleopatra must be able to pick the conversation up rather
    # than answer as if she had never met him.
    #
    # Bounded on purpose. This fetches the newest few pages — about thirty
    # messages — and nothing else. The archive behind them may be five hundred
    # provider pages; none of it is fetched here, because a fan waiting for a
    # reply must never wait through it. The deep pass is resumable and runs
    # later, behind live work.
    #
    # Costs nothing in the common case: a fan with enough recent local context
    # returns immediately without a provider call. Best-effort, because history
    # must never be the reason a conversation cannot be answered.
    try:
        from services.fan_history import warm_resume

        warm = await warm_resume(creator_id=creator_id, fan_id=fan_id)
        if warm.get("imported"):
            print(
                f"[HISTORY WARM RESUME] fan={fan_id} "
                f"imported={warm['imported']} pages={warm.get('pages')}"
            )
    except Exception as exc:
        print(f"[HISTORY WARM RESUME ERROR] fan={fan_id}: {exc}")

    conversation_history = await get_conversation_history(fan_id)
    fan_profile = await get_fan_by_id(fan_id)
    if fan_profile is None:
        fan_profile = Fan(id=fan_id, display_name=fan_id)

    # The extractor stage of the same AI stack that will answer this message.
    inbound_stack = await resolve_ai_stack(creator_id=creator_id, fan_id=fan_id)
    spawn(
        learn_from_fan_message(
            creator_id=creator_id,
            fan_id=fan_id,
            fan_message=message_content,
            source_message_id=message_id,
            conversation_history=conversation_history,
            profile_id=inbound_stack.profile_id,
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
history_backfill_task: asyncio.Task | None = None


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
                # Reconciliation is background work, not a live conversation.
                # Categorising it keeps the credit breakdown honest about which
                # part of the product is actually spending.
                with apifansly_usage_category(CATEGORY_RECONCILIATION):
                    await _reconcile_chat_creators_once(due)
        except Exception as exc:
            print(f"[CRON CHAT RECONCILE INFRA ERROR] {exc}")


_HISTORY_BACKFILL_TICK_SECONDS = 120


async def history_backfill_scheduler():
    """Advance deep historical backfill, always behind live conversation.

    Deliberately the lowest-priority loop in the process:

      * It does nothing at all unless HISTORY_BACKFILL_ENABLED is on. A
        deployment that has not asked for archive imports does not get them.
      * Every tick and every page re-asks whether live work is happening, and
        yields entirely when it is. A returning fan must never wait behind a
        backfill.
      * It stops when the optional history credit budget is spent. That budget
        governs THIS loop and nothing else — live conversation, delivery and
        purchase reconciliation are never gated on it.
      * Paging and compaction are separate bounded steps, so a model outage
        cannot stall the import and a provider outage cannot stall compaction.
    """
    while True:
        await asyncio.sleep(_HISTORY_BACKFILL_TICK_SECONDS)
        try:
            from services.fan_history import (
                backfill_scheduler_pass,
                history_backfill_enabled,
            )

            if not history_backfill_enabled() or not apifansly_enabled():
                continue
            result = await backfill_scheduler_pass()
            if result.get("fans"):
                print(
                    f"[CRON HISTORY] fans={result['fans']} "
                    f"credits={result.get('estimated_credits', 0)}"
                )
                await _compact_recent_backfills()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[CRON HISTORY ERROR] {exc}")


async def _compact_recent_backfills(limit: int = 3) -> None:
    """Turn freshly imported history into durable facts, a few fans at a time."""
    from db.fan_history_queries import list_unfinished_backfills
    from services.apifansly import live_work_in_progress
    from services.fan_history_memory import (
        compact_fan_history,
        history_extraction_enabled,
    )

    if not history_extraction_enabled():
        return
    for row in await list_unfinished_backfills(limit=limit):
        if live_work_in_progress():
            return
        try:
            await compact_fan_history(
                creator_id=str(row.get("creator_id") or ""),
                fan_id=str(row.get("fan_id") or ""),
            )
        except Exception as exc:
            print(f"[CRON HISTORY COMPACT ERROR] fan={row.get('fan_id')}: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_store, fansly_poller, ppv_sweep_task, vault_autosync_task, scheduled_actions_task, chat_reconcile_task, model_availability_task, history_backfill_task

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
    history_backfill_task = asyncio.create_task(history_backfill_scheduler())

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
    if history_backfill_task:
        history_backfill_task.cancel()

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


@app.middleware("http")
async def _unhandled_error_middleware(request: Request, call_next):
    """Turn an unhandled route exception into a CORS-visible JSON 500.

    Without this, an exception that escapes a route is answered by Starlette's
    outermost ServerErrorMiddleware, whose response never passes back through
    the CORS middleware below. The browser therefore sees a response with no
    Access-Control-Allow-Origin, blocks it, and rejects the fetch with the
    opaque TypeError "Failed to fetch" — which is exactly what the Simulator
    surfaced while the backend had already logged a full traceback nobody could
    correlate with it.

    This middleware is registered BEFORE the CORS middleware, so CORS wraps it
    and the JSON body actually reaches the client. The body stays deliberately
    thin — a stable error id and the exception type — because an ordinary agency
    account must not learn anything about internals from a crash. The id is what
    ties it to the traceback in the logs.
    """
    try:
        return await call_next(request)
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all boundary
        error_id = uuid.uuid4().hex[:12]
        print(
            f"[UNHANDLED ERROR] id={error_id} path={request.url.path} "
            f"type={type(exc).__name__} error={exc}"
        )
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "detail": (
                    "The server failed to handle this request. Quote error "
                    f"{error_id} when reporting it."
                ),
                "error_id": error_id,
                "error_type": type(exc).__name__,
            },
        )


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


async def _record_sent_creator_canon(
    *,
    creator_id: str,
    fan_id: str,
    sent_reply: str,
) -> None:
    """Persist ordinary self-facts from a reply an operator just sent.

    Runs entirely in the background: the message is already delivered, so this
    must add nothing to the operator's response time and must never fail the
    request. Only writer versions that allow improvised personal facts record
    them (ai/writer_style.py), which keeps the frozen profiles frozen.
    """
    try:
        from ai.writer_style import persists_improvised_facts
        from services.ai_stack import resolve_ai_stack
        from services.creator_canon import persist_sent_creator_facts

        stack = await resolve_ai_stack(creator_id=creator_id, fan_id=fan_id)
        if not persists_improvised_facts(stack.profile.writer_prompt_version()):
            return

        history = await get_conversation_history(fan_id)
        fan_messages = [message for message in history if message.role == "fan"]
        await persist_sent_creator_facts(
            creator_id=creator_id,
            sent_reply=sent_reply,
            fan_message=fan_messages[-1].content if fan_messages else "",
            conversation_history=history,
            fan_id=fan_id,
            profile_id=stack.profile_id,
        )
    except Exception as exc:
        print(f"[CREATOR CANON] assisted capture skipped fan={fan_id}: {exc}")


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

    # An assisted reply becomes creator canon here and nowhere earlier: this is
    # the point at which a suggestion stops being a candidate and becomes
    # something the fan actually received. The two suggestions the operator did
    # not pick never reach this code, so they cannot establish anything.
    spawn(
        _record_sent_creator_canon(
            creator_id=req.creator_id,
            fan_id=req.fan_id,
            sent_reply=req.content,
        ),
        name=f"creator_canon:{req.fan_id}",
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
        record_apifansly_raw_call(
            response,
            operation="account connect",
            category=CATEGORY_ACCOUNT,
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
        record_apifansly_raw_call(
            response,
            operation="account 2fa verification",
            category=CATEGORY_ACCOUNT,
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


# API-004 — the platform-message parser lives in services/apifansly.py so live
# reconciliation, the active-chat endpoint and historical backfill all build the
# same row from the same payload. These two names are kept as thin aliases
# because several call sites and tests already read this way.
_apifansly_account_media_lookup = apifansly_account_media_lookup
_apifansly_message_row = apifansly_chat_message_row


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

    # The dashboard calls this when an operator opens a conversation and on tab
    # focus, so it IS a live conversation happening right now. Marking the scope
    # both attributes its credits correctly and tells deep-history work to stand
    # aside while an operator is looking at this chat.
    with apifansly_usage_category(CATEGORY_LIVE_CHAT):
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

    # A previously known fan whose local context is too thin to answer from
    # gets the newest few pages of history now, cheaply and boundedly, before
    # the reply path runs. Best-effort: history is never allowed to be the
    # reason a conversation cannot be answered.
    warm: dict = {}
    try:
        from services.fan_history import warm_resume

        warm = await warm_resume(creator_id=creator_id, fan_id=fan_id)
    except Exception as exc:
        print(f"[HISTORY WARM RESUME ERROR] fan={fan_id}: {exc}")
        warm = {"status": "error", "detail": str(exc)[:200]}
    if warm.get("imported"):
        result = {
            **result,
            "imported": int(result.get("imported") or 0) + int(warm["imported"]),
        }
    return {"status": "ok", **result, "warm_resume": warm}


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
async def load_fan_history(
    creator_id: str,
    fan_id: str,
    pages: int | None = None,
    deep: bool = True,
) -> dict:
    """Import this fan's conversation history, cursor-based and resumable.

    What changed and why
    --------------------
    This route used to page an entire conversation inside one request, in
    memory, asking for 50 messages a page. Both halves of that were wrong.

    API Fansly documents `limit min=1 max=10` on the chat-messages endpoint, so
    a request for 50 returned 10 and the import silently paid five times the
    pages it believed it was buying. A 5,000-message fan is therefore 500
    provider round trips — a fact about the upstream API, not a tunable — and
    holding an HTTP request open for 500 sequential calls meant any
    interruption threw away every page already paid for.

    So the work is now durable and bounded. One call advances the cursor by a
    bounded number of pages and returns; the cursor lives in
    public.fan_history_backfill, so the next call resumes where this one
    stopped instead of restarting. Pressing this button twice imports nothing
    twice: persistence is idempotent on (creator_id, fansly_message_id).

    ``imported`` is still the number of messages this call newly persisted, so
    the dashboard's existing toast keeps working unchanged.
    """
    from services.fan_history import (
        advance_backfill,
        deep_backfill_pages_per_run,
        fan_history_status,
        warm_resume,
    )

    # Warm resume first, so an operator who opens a stale conversation and
    # presses Load history gets answerable context from the first few pages
    # even if the deep pass is later throttled.
    warm = await warm_resume(creator_id=creator_id, fan_id=fan_id)

    result: dict = {}
    if deep:
        result = await advance_backfill(
            creator_id=creator_id,
            fan_id=fan_id,
            max_pages=int(pages) if pages else deep_backfill_pages_per_run(),
            # An operator pressing this button is themselves the live activity.
            # Yielding to "a live call happened seconds ago" would make the
            # button do nothing exactly when it is pressed.
            respect_live_priority=False,
        )

    imported = int(warm.get("imported") or 0) + int(result.get("imported") or 0)
    credits = float(warm.get("estimated_credits") or 0.0) + float(
        result.get("estimated_credits") or 0.0
    )

    # Unchanged from the previous implementation of this route, and deliberately
    # so: the operator presses Load history expecting the profile panel to fill
    # in, and those two documents are what it renders. Historical compaction
    # (services/fan_history_memory.py) adds evidence-backed FACTS alongside
    # them; it does not replace the summary the dashboard already shows.
    if imported > 0:
        conversation_history = await get_conversation_history(fan_id)
        fan_profile = await get_fan_by_id(fan_id)
        if fan_profile and len(conversation_history) >= 10:
            spawn(
                _update_fan_ai_summary(fan_id, conversation_history),
                name="update_fan_ai_summary",
            )
            spawn(
                _update_fan_memory(
                    fan_id, creator_id, conversation_history, fan_profile.total_spent
                ),
                name="update_fan_memory",
            )

    status = await fan_history_status(fan_id)
    return {
        "status": "ok",
        "imported": imported,
        "warm_resume": warm,
        "deep": result or {"status": "skipped"},
        "estimated_credits": round(credits, 3),
        **status,
    }


@app.get(
    "/fan-history/{creator_id}/{fan_id}",
    dependencies=[Depends(require_creator_fan_access)],
)
async def fan_history_progress(creator_id: str, fan_id: str) -> dict:
    """How much of this fan's history is imported, and what it has cost.

    Needs no provider call and no connector: it reads the durable checkpoint.
    """
    from services.fan_history import fan_history_status

    return {"status": "ok", "fan_id": fan_id, **await fan_history_status(fan_id)}


@app.post(
    "/fan-history/{creator_id}/{fan_id}/compact",
    dependencies=[Depends(require_creator_fan_access)],
)
async def compact_fan_history_endpoint(
    creator_id: str,
    fan_id: str,
    chunks: int | None = None,
) -> dict:
    """Compact already-imported history into durable evidence-backed facts.

    Deliberately separate from importing. Paging costs provider credits and
    compaction costs model tokens; they resume independently, and a fan whose
    archive is fully imported can be re-compacted without paying for the
    archive again.
    """
    from services.fan_history_memory import compact_fan_history

    return await compact_fan_history(
        creator_id=creator_id,
        fan_id=fan_id,
        max_chunks=int(chunks) if chunks else None,
    )


@app.get(
    "/creator-history-usage/{creator_id}",
    dependencies=[Depends(require_creator_path_access)],
)
async def creator_history_usage(creator_id: str) -> dict:
    """Rolled-up historical-backfill cost and progress for one creator."""
    from db.fan_history_queries import creator_history_totals
    from services.apifansly import (
        CHAT_MESSAGE_PAGE_MAX,
        background_history_budget_state,
    )

    totals = await creator_history_totals(creator_id)
    return {
        "status": "ok",
        "creator_id": creator_id,
        "history": totals,
        "budget": background_history_budget_state(),
        "messages_per_page_max": CHAT_MESSAGE_PAGE_MAX,
        "note": (
            "Credit figures are estimates from observed response sizes. API "
            "Fansly's Usage dashboard is authoritative."
        ),
    }


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
        # A raw post that never reaches services.apifansly.request(), so its
        # credit is recorded explicitly. It is not free just because nobody
        # reads the response.
        response = await client.post(
            apifansly_url(f"{apifansly_id}/chats/mark-as-read"),
            headers=apifansly_headers(),
            timeout=10,
        )
        record_apifansly_raw_call(
            response,
            operation="chat mark as read",
            account_id=str(apifansly_id),
            category=CATEGORY_LIVE_CHAT,
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
            # Upload is metered on what was sent, not on the small JSON that
            # comes back. Counting the response would understate this call by
            # orders of magnitude for a video.
            media_bytes=len(file_bytes or b""),
            category=CATEGORY_VAULT,
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
                category=CATEGORY_VAULT,
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
    """The columns one classification writes.

    The persistence columns travel with every write, because they are what the
    next sync reads to decide this row is finished work and must be skipped. A
    result that omitted them would be re-classified forever.

    A PENDING result — deeper analysis refused on cost — writes only those
    persistence columns. It must not touch content_category, price or
    explicitness: there is no evidence for them, and overwriting a previous
    classification with blanks would lose real metadata to a cost decision.
    """
    if result.get("pending"):
        return {
            "classification_status": result["classification_status"],
            "classification_skip_reason": result.get(
                "classification_skip_reason", ""
            ),
            "classification_media_key": result.get("classification_media_key", ""),
            "classification_retrieval_method": "",
            "classification_media_bytes": 0,
            "classification_media_credits": 0.0,
            "classification_frames_sampled": 0,
            "classification_metadata": result.get("classification_metadata", {}),
        }
    return {
        "classification_status": result.get(
            "classification_status", CLASSIFICATION_STATUS_COMPLETE
        ),
        "classification_skip_reason": result.get("classification_skip_reason", ""),
        "classification_media_key": result.get("classification_media_key", ""),
        "classification_retrieval_method": result.get(
            "classification_retrieval_method", ""
        ),
        "classification_media_bytes": int(
            result.get("classification_media_bytes") or 0
        ),
        "classification_media_credits": float(
            result.get("classification_media_credits") or 0.0
        ),
        "classification_frames_sampled": int(
            result.get("classification_frames_sampled") or 0
        ),
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


# The approved commercial range per category lives in models/content_pricing.py,
# so package pricing and the classifier cannot drift apart.
VAULT_CATEGORIES = CONTENT_CATEGORY_RANGES

class VaultVisualAccessError(RuntimeError):
    """The classifier could not obtain a usable visual for a vault item."""


class VaultMediaCostRefusal(VaultVisualAccessError):
    """Deeper analysis was possible but too expensive to perform automatically.

    Distinct from its parent because it is not a failure. Nothing is broken,
    nothing needs fixing, and retrying will deterministically refuse again: the
    asset is simply larger than an automatic run may transfer. The item is
    recorded as ``pending`` with the reason, rather than being repeatedly
    re-attempted as if it had errored.
    """

    def __init__(self, decision: DownloadDecision) -> None:
        super().__init__(decision.operator_message())
        self.decision = decision


# ---------------------------------------------------------------------------
# Obtaining the pixels — and what that is allowed to cost
# ---------------------------------------------------------------------------
#
# Retrieval is ordered by COST, cheapest first, and the only expensive option is
# last and guarded:
#
#   1. direct CDN access to the signed asset          free
#      - for a video this is an ffmpeg range sample: the decoder seeks to each
#        offset and reads only the bytes that frame needs, so a 12-frame sample
#        of a 250 MB clip moves a few megabytes, not 250
#   2. a refreshed signed URL, then (1) again          one metadata call
#   3. the existing platform thumbnail                 free
#   4. the API Fansly protected-media proxy            2 CREDITS PER MEGABYTE
#
# Step 4 is the one that used to happen silently. It now requires a policy
# decision from services.media_cost_guard, taken BEFORE any byte moves, from a
# size probed for free at the CDN.
#
# For a video, step 3 comes before step 4 deliberately. A thumbnail yields a
# real, honest, partial classification for nothing; paying ~500 credits to
# upgrade it is not a decision a background job gets to make on its own. The
# result is marked ``partial`` and says so, and an operator who wants the deep
# scan can ask for it explicitly — the manual tier has its own, larger ceiling.
#
# For a still image the order is 1, 2, 4: there is no thumbnail to fall back to
# that is not the image itself, and a photo's worst case is single-digit
# megabytes rather than a quarter of a gigabyte.


@dataclass
class VaultVisual:
    """The pixels to classify, plus how they were obtained and what it cost."""

    source: str
    retrieval_method: str
    image: bytes = b""
    frames: list[bytes] = dataclass_field(default_factory=list)
    offsets_seconds: list[float] = dataclass_field(default_factory=list)
    duration_seconds: float = 0.0
    media_bytes: int = 0
    status: str = CLASSIFICATION_STATUS_COMPLETE
    skip_reason: str = ""
    skip_message: str = ""

    @property
    def estimated_credits(self) -> float:
        return round(estimated_credits_for_bytes(self.media_bytes), 3)


async def _probe_media_size(visual_url: str, *, client) -> int | None:
    """What the CDN says this asset weighs, or None.

    Costs nothing: both attempts go straight to the signed CDN URL, not through
    the billed proxy. HEAD first; a signed CDN that refuses HEAD will usually
    still answer a one-byte range GET with a Content-Range naming the total,
    which is the cheapest honest way to learn a size.
    """
    for attempt in ("head", "range"):
        try:
            if attempt == "head":
                response = await client.head(visual_url, timeout=15)
            else:
                response = await client.get(
                    visual_url,
                    timeout=15,
                    headers={"Range": "bytes=0-0"},
                )
        except Exception:
            continue
        if response.status_code >= 400:
            continue
        size = parse_content_length(response.headers)
        if size and size > 0:
            return size
    return None


async def _download_direct_cdn(visual_url: str, *, client) -> tuple[bytes, str]:
    """Fetch the signed asset straight from the CDN. Returns (bytes, status)."""
    try:
        response = await client.get(visual_url, timeout=25)
    except Exception as exc:
        return b"", type(exc).__name__
    status = f"http_{response.status_code}_{len(response.content)}b"
    if response.status_code == 200 and len(response.content) > 1000:
        return bytes(response.content), status
    return b"", status


async def _guarded_proxy_download(
    visual_url: str,
    *,
    client,
    manual: bool,
    is_video: bool,
    account_id: str = "",
) -> tuple[bytes, DownloadDecision]:
    """The billed path, taken only when the guard allows it.

    Returns ``(b"", decision)`` when refused, so the caller can record exactly
    why deeper analysis did not happen rather than reporting a generic failure.
    A still image may proceed on an unreadable size; a video may not.
    """
    size = await _probe_media_size(visual_url, client=client)
    decision = evaluate_download(
        content_length_bytes=size,
        manual=manual,
        allow_unknown_size=not is_video,
    )
    if not decision.allowed:
        print(
            f"[VAULT MEDIA GUARD] refused reason={decision.reason} "
            f"manual={manual} video={is_video} "
            f"size_mb={decision.estimated_megabytes} "
            f"credits={decision.estimated_credits:.1f}"
        )
        return b"", decision

    # Attributed to the vault so the credits land in the right bucket, and to
    # the creator's account so "who caused this" is answerable.
    with apifansly_usage_category(CATEGORY_VAULT):
        content = await apifansly_download_media(
            visual_url,
            client=client,
            account_id=account_id or None,
            operation=VAULT_MEDIA_DOWNLOAD_OPERATION,
        )
    print(
        f"[VAULT MEDIA GUARD] downloaded bytes={len(content)} "
        f"credits={estimated_credits_for_bytes(len(content)):.1f} "
        f"manual={manual} video={is_video}"
    )
    return content, decision


async def _download_visual_candidate(
    visual_url: str,
    *,
    client,
    manual: bool = False,
    is_video: bool = False,
    account_id: str = "",
) -> tuple[bytes, str, int]:
    """One still image, by the cheapest route that works.

    Returns ``(bytes, retrieval_method, billed_bytes)``. ``billed_bytes`` is
    zero for every free path and is what the per-item cost telemetry records.
    """
    content, direct_status = await _download_direct_cdn(visual_url, client=client)
    if content:
        return content, RETRIEVAL_DIRECT_CDN, 0

    if not is_fansly_cdn_url(visual_url):
        raise VaultVisualAccessError(
            f"The media source could not be downloaded ({direct_status})."
        )

    try:
        downloaded, decision = await _guarded_proxy_download(
            visual_url,
            client=client,
            manual=manual,
            is_video=is_video,
            account_id=account_id,
        )
    except Exception as exc:
        raise VaultVisualAccessError(
            "The protected Fansly media could not be downloaded "
            f"(direct={direct_status}; proxy={type(exc).__name__})."
        ) from exc
    if not downloaded:
        raise VaultVisualAccessError(decision.operator_message())
    return downloaded, RETRIEVAL_APIFANSLY_DOWNLOAD, len(downloaded)


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
    # A metadata listing, not a media transfer: attributed to the vault so the
    # cost of refreshing links is visible next to the cost of classifying.
    with apifansly_usage_category(CATEGORY_VAULT):
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


async def _sample_video_frames(video_url: str) -> ExtractedFrames:
    """Duration-aware keyframes straight from the signed URL. Costs nothing.

    This is the path that should serve essentially every video: ffmpeg range
    requests move a few megabytes to sample a clip of any length, and the
    result is real chronological coverage rather than a poster frame.
    """
    settings = FrameSettings.from_env()
    if not settings.enabled:
        raise VaultVisualAccessError(
            "Video-frame analysis is disabled in this deployment."
        )
    if not ffmpeg_available():
        raise VaultVisualAccessError(
            "Video-frame analysis is unavailable because FFmpeg is missing."
        )
    return await extract_frames(video_url, settings=settings)


async def _sample_downloaded_video(content: bytes) -> ExtractedFrames:
    """Keyframes from an already-transferred file. Serialised on purpose."""
    import os as _os

    settings = FrameSettings.from_env()
    async with _protected_video_download_gate:
        temporary_path = await asyncio.to_thread(_write_temp_video, content)
        try:
            return await extract_frames(temporary_path, settings=settings)
        finally:
            try:
                await asyncio.to_thread(_os.unlink, temporary_path)
            except FileNotFoundError:
                pass


async def _load_vault_visual(
    item: dict,
    *,
    is_video: bool,
    client=None,
    manual: bool = False,
    account_id: str = "",
) -> VaultVisual:
    """Obtain something classifiable, by the cheapest route that works."""
    import httpx

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(follow_redirects=True)
    try:
        if is_video:
            return await _load_video_visual(
                item,
                client=client,
                manual=manual,
                account_id=account_id,
            )

        visual_url = str(item.get("url") or "")
        first_error: Exception | None = None
        if visual_url:
            try:
                content, method, billed = await _download_visual_candidate(
                    visual_url,
                    client=client,
                    manual=manual,
                    is_video=False,
                    account_id=account_id,
                )
                return VaultVisual(
                    source="image",
                    retrieval_method=method,
                    image=content,
                    media_bytes=billed,
                )
            except Exception as exc:
                first_error = exc

        try:
            refreshed = await _refresh_vault_item_urls(item)
        except Exception as exc:
            refreshed = None
            refresh_error: Exception | None = exc
        else:
            refresh_error = None

        refreshed_url = str((refreshed or {}).get("url") or "")
        if refreshed_url:
            content, method, billed = await _download_visual_candidate(
                refreshed_url,
                client=client,
                manual=manual,
                is_video=False,
                account_id=account_id,
            )
            return VaultVisual(
                source="image",
                retrieval_method=(
                    RETRIEVAL_REFRESHED_URL
                    if method == RETRIEVAL_DIRECT_CDN
                    else method
                ),
                image=content,
                media_bytes=billed,
            )

        raise VaultVisualAccessError(
            "The media could not be read. Its signed link is unavailable or "
            "expired; the next vault sync refreshes those links automatically."
        ) from (refresh_error or first_error)
    finally:
        if owns_client:
            await client.aclose()


async def _load_video_visual(
    item: dict,
    *,
    client,
    manual: bool,
    account_id: str,
) -> VaultVisual:
    """A video's pixels, in strict cost order. See the note above this section."""
    errors: list[str] = []
    # The guard's verdict, kept so a refusal can be reported with its real cost
    # rather than as a generic "could not read this video".
    last_decision: DownloadDecision | None = None
    refusals: list[DownloadDecision] = []

    # 1. Direct range sampling of the stored signed URL. Free, and the path
    #    that should serve essentially every video.
    video_url = str(item.get("url") or "")
    if video_url:
        try:
            sampled = await _sample_video_frames(video_url)
            if len(sampled.frames) >= 2:
                return VaultVisual(
                    source="video_frames",
                    retrieval_method=RETRIEVAL_DIRECT_FRAMES,
                    frames=sampled.frames,
                    offsets_seconds=sampled.offsets_seconds,
                    duration_seconds=sampled.duration_seconds,
                )
            errors.append(f"direct sampling produced {len(sampled.frames)} frames")
        except Exception as exc:
            errors.append(f"direct sampling {type(exc).__name__}")

    # 2. A refreshed signed URL, then range sampling again. One metadata call.
    refreshed_url = ""
    try:
        refreshed = await _refresh_vault_item_urls(item)
        refreshed_url = str((refreshed or {}).get("url") or "")
        if refreshed_url and refreshed_url != video_url:
            sampled = await _sample_video_frames(refreshed_url)
            if len(sampled.frames) >= 2:
                return VaultVisual(
                    source="video_frames",
                    retrieval_method=RETRIEVAL_REFRESHED_URL,
                    frames=sampled.frames,
                    offsets_seconds=sampled.offsets_seconds,
                    duration_seconds=sampled.duration_seconds,
                )
            errors.append("refreshed sampling produced too few frames")
        if refreshed:
            item = {**item, **refreshed}
    except Exception as exc:
        errors.append(f"refresh {type(exc).__name__}")

    thumbnail_url = str(item.get("thumbnail_url") or "")

    # 3/4. The order here is the whole point of the media-cost guard.
    #
    # Automatic: the free thumbnail first. A partial classification for nothing
    # beats a complete one for 500 credits that nobody asked for.
    #
    # Manual: the operator explicitly asked for deep analysis and the manual
    # ceiling is theirs to spend, so the guarded download is tried first and
    # the thumbnail is the fallback.
    if manual:
        deep = await _video_deep_scan(
            item,
            client=client,
            manual=True,
            account_id=account_id,
            errors=errors,
            refusals=refusals,
        )
        if deep is not None:
            return deep
        thumbnail = await _video_thumbnail_visual(
            thumbnail_url,
            client=client,
            manual=manual,
            account_id=account_id,
            skip_reason=MEDIA_GUARD_REASON_UNKNOWN,
            skip_message=(
                "Deep video scan could not be completed, so the thumbnail was "
                "classified instead."
            ),
            errors=errors,
        )
        if thumbnail is not None:
            return thumbnail
    else:
        decision = None
        if thumbnail_url:
            # Establish what a deep scan WOULD have cost, so the partial result
            # can say why it stopped. The probe is a free CDN request.
            probe_url = str(item.get("url") or "")
            if probe_url and is_fansly_cdn_url(probe_url):
                size = await _probe_media_size(probe_url, client=client)
                decision = evaluate_download(
                    content_length_bytes=size,
                    manual=False,
                    allow_unknown_size=False,
                )
                last_decision = decision
            thumbnail = await _video_thumbnail_visual(
                thumbnail_url,
                client=client,
                manual=manual,
                account_id=account_id,
                skip_reason=(
                    decision.reason if decision else MEDIA_GUARD_REASON_UNKNOWN
                ),
                skip_message=(
                    decision.operator_message()
                    if decision
                    else (
                        "Deep video scan skipped to avoid a high "
                        "media-transfer cost."
                    )
                ),
                errors=errors,
            )
            if thumbnail is not None:
                return thumbnail

        # No usable thumbnail. The guarded download is the last option, and it
        # proceeds only for an asset whose size is known and small.
        deep = await _video_deep_scan(
            item,
            client=client,
            manual=False,
            account_id=account_id,
            errors=errors,
            refusals=refusals,
        )
        if deep is not None:
            return deep

    # Nothing free worked and the billed path was refused on cost. That is a
    # policy outcome with a number attached, not a fault, so it is reported as
    # one: the item is left pending with the reason, and no credits are spent.
    refused = refusals[-1] if refusals else last_decision
    if refused is not None and not refused.allowed:
        raise VaultMediaCostRefusal(refused)

    detail = "; ".join(errors[-3:]) or "no readable source"
    raise VaultVisualAccessError(
        "This video could not be classified: no keyframes and no thumbnail "
        f"could be read ({detail}). It was left unclassified rather than "
        "guessed from its filename."
    )


async def _video_thumbnail_visual(
    thumbnail_url: str,
    *,
    client,
    manual: bool,
    account_id: str,
    skip_reason: str,
    skip_message: str,
    errors: list[str],
) -> VaultVisual | None:
    """The platform thumbnail, as a deliberately PARTIAL classification."""
    if not thumbnail_url:
        return None
    try:
        content, method, billed = await _download_visual_candidate(
            thumbnail_url,
            client=client,
            manual=manual,
            # A thumbnail is a still image whatever it depicts, so it is the
            # image tier of the guard that applies to it, not the video tier.
            is_video=False,
            account_id=account_id,
        )
    except Exception as exc:
        errors.append(f"thumbnail {type(exc).__name__}")
        return None
    return VaultVisual(
        source="video_thumbnail",
        retrieval_method=(
            RETRIEVAL_PLATFORM_THUMBNAIL if billed == 0 else method
        ),
        image=content,
        media_bytes=billed,
        status=CLASSIFICATION_STATUS_PARTIAL,
        skip_reason=skip_reason,
        skip_message=skip_message,
    )


async def _video_deep_scan(
    item: dict,
    *,
    client,
    manual: bool,
    account_id: str,
    errors: list[str],
    refusals: list[DownloadDecision],
) -> VaultVisual | None:
    """Transfer the original through the billed proxy, if the guard allows.

    Returns None when the guard refuses or the transfer fails, so the caller
    falls through to the next option rather than failing the whole item.
    """
    video_url = str(item.get("url") or "")
    if not video_url or not is_fansly_cdn_url(video_url):
        return None
    try:
        content, decision = await _guarded_proxy_download(
            video_url,
            client=client,
            manual=manual,
            is_video=True,
            account_id=account_id,
        )
    except Exception as exc:
        errors.append(f"proxy {type(exc).__name__}")
        return None
    if not content:
        errors.append(f"guard {decision.reason}")
        refusals.append(decision)
        return None

    billed = len(content)
    try:
        sampled = await _sample_downloaded_video(content)
    except Exception as exc:
        errors.append(f"downloaded sampling {type(exc).__name__}")
        return None
    if len(sampled.frames) < 2:
        errors.append("downloaded sampling produced too few frames")
        return None
    return VaultVisual(
        source="video_frames",
        retrieval_method=RETRIEVAL_APIFANSLY_DOWNLOAD,
        frames=sampled.frames,
        offsets_seconds=sampled.offsets_seconds,
        duration_seconds=sampled.duration_seconds,
        media_bytes=billed,
    )


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


def _pending_classification_payload(item: dict, decision: DownloadDecision) -> dict:
    """What is written when deeper analysis was refused on cost.

    Deliberately carries NO content category, explicitness or price. Those
    would be guesses, and a guessed category becomes sellable set metadata. The
    row records that it was looked at, what it would have cost, and why it
    stopped — which is an honest answer and a reversible one: a later manual
    re-analysis, or a sync where direct sampling succeeds, replaces it.
    """
    return {
        "id": item.get("id", ""),
        "pending": True,
        "classification_status": CLASSIFICATION_STATUS_PENDING,
        "classification_skip_reason": decision.reason,
        "classification_media_key": media_identity_key(item),
        "classification_retrieval_method": "",
        "classification_media_bytes": 0,
        "classification_media_credits": 0.0,
        "classification_frames_sampled": 0,
        "classification_metadata": {
            "analysis_skipped": {
                "reason": decision.reason,
                "message": decision.operator_message(),
                "estimated_megabytes": decision.estimated_megabytes,
                "estimated_credits": round(decision.estimated_credits, 2),
            }
        },
    }


async def _classify_video_batches(
    visual: VaultVisual,
    *,
    item: dict,
    allow_core_qwen_fallback: bool,
    force_qwen: bool,
) -> tuple[dict, dict, dict, dict]:
    """Classify a video as several chronological batches, then combine them.

    Returns ``(primary_classification, shoot_fingerprint, local_visual,
    video_record)``.

    The primary classification is the batch with the highest explicitness,
    because that is what decides the CATEGORY and therefore the approved price
    range: a clip that ends explicit is an explicit clip, and pricing it from
    its clothed opening would be wrong in the direction that loses money and
    mis-sells content. Everything about how the clip MOVES lives in the video
    record, which is merged over the primary result below.

    The shoot fingerprint comes from the first batch's sheet, so same-shoot
    grouping keys off the opening frames — the part most likely to share
    setting and styling with the stills from the same session.
    """
    batches = chronological_batches(
        visual.frames,
        visual.offsets_seconds,
        frames_per_sheet=_VIDEO_FRAMES_PER_SHEET,
        max_sheets=_VIDEO_MAX_SHEETS,
    )
    album_title = str(item.get("album_title") or "")
    filename = str(item.get("filename") or "")

    sheets: list[tuple[Any, bytes, int]] = []
    for batch in batches:
        sheet, used = await asyncio.to_thread(build_contact_sheet, batch.frames)
        if sheet and used >= 1:
            sheets.append((batch, sheet, used))
    if not sheets:
        raise VaultVisualAccessError(
            "The extracted video frames were blank or unreadable."
        )

    first_fingerprint = await build_shoot_fingerprint(sheets[0][1])
    local_visual = first_fingerprint.get("local") or {}

    results: list[tuple[Any, dict, int]] = []
    for batch, sheet, used in sheets:
        classified = await classify_vault_image(
            sheet,
            is_video=True,
            album_title=album_title,
            filename=filename,
            local_visual=local_visual,
            allow_core_qwen_fallback=allow_core_qwen_fallback,
            force_qwen=force_qwen,
        )
        results.append((batch, classified, used))

    observations = [
        observation_from_classification(
            classified,
            batch=batch,
            frames_used=used,
        )
        for batch, classified, used in results
    ]
    record = combine_batch_observations(
        observations,
        duration_seconds=visual.duration_seconds,
    )

    # The batch that decides category and price: the most explicit one.
    primary_index = max(
        range(len(results)),
        key=lambda index: (
            int(results[index][1].get("explicitness") or 0),
            observations[index].nudity_rank,
        ),
    )
    primary = dict(results[primary_index][1])

    # Facts that belong to the WHOLE clip replace the single batch's view of
    # them. Anything the record could not determine leaves the primary batch's
    # value alone rather than blanking it.
    prose = describe_video_record(record)
    if prose:
        primary["description"] = prose
        primary["description_complete"] = True
    for key, value in (
        ("scene_location", record.get("setting")),
        ("good_for", commercial_role(record)),
    ):
        if value:
            primary[key] = value
    if record.get("progression"):
        # The outfit field becomes the progression when there is one, because
        # "clothed → lingerie → nude" is the truthful answer to "what is the
        # wardrobe in this video" and a single state is not.
        primary["scene_outfit"] = " → ".join(record["progression"][:4])
    for key, source_key in (
        ("props", "props"),
        ("sexual_activity", "activities"),
        ("visible_anatomy", "visible_anatomy"),
    ):
        values = record.get(source_key) or []
        if values:
            merged = list(primary.get(key) or [])
            for value in values:
                if not any(str(value).lower() == str(k).lower() for k in merged):
                    merged.append(value)
            primary[key] = merged
    if record.get("tags"):
        merged_tags = list(primary.get("tags") or [])
        for tag in record["tags"]:
            if not any(str(tag).lower() == str(k).lower() for k in merged_tags):
                merged_tags.append(tag)
        primary["tags"] = merged_tags

    provider_metadata = dict(primary.get("_provider_metadata") or {})
    provider_metadata["video_batches"] = len(results)
    provider_metadata["video_frames_sampled"] = sum(used for _, _, used in results)
    provider_metadata["video_duration_seconds"] = round(
        float(visual.duration_seconds or 0.0), 2
    )
    primary["_provider_metadata"] = provider_metadata

    shoot_fingerprint = first_fingerprint
    print(
        f"[VIDEO SEMANTICS] item={item.get('id', '')} "
        f"duration={record.get('duration_label')} "
        f"frames={record.get('sampled_frames')} batches={len(results)} "
        f"progression={'→'.join(record.get('progression') or []) or 'none'} "
        f"changes={len(record.get('scene_changes') or [])}"
    )
    return primary, shoot_fingerprint, local_visual, record


async def _categorize_single_item(
    item: dict,
    *,
    allow_core_qwen_fallback: bool = True,
    force_qwen: bool = False,
    visual_client=None,
    manual: bool = False,
    account_id: str = "",
) -> dict:
    """Classify one vault item into the versioned provider-neutral contract.

    Images are resized before upload to control vision-token cost. A video is
    sampled across its whole duration and classified in small chronological
    batches, which are then folded into one video-level semantic record — see
    ``_classify_video_batches``. A provider/fetch/parse failure is raised so the
    retry loop can leave the item stale instead of permanently saving an empty
    ``other`` classification.

    ``manual`` marks an operator-initiated re-analysis, which the media-cost
    guard allows a larger transfer budget and permits to attempt a deep video
    scan ahead of the thumbnail.
    """
    mimetype = str(item.get("mimetype") or "").lower()
    item_id = item.get("id", "")
    is_video = mimetype.startswith("video") if mimetype else False

    try:
        try:
            visual = await _load_vault_visual(
                item,
                is_video=is_video,
                client=visual_client,
                manual=manual,
                account_id=account_id,
            )
        except VaultMediaCostRefusal as refusal:
            # A policy outcome, not a fault: record why and spend nothing.
            print(
                f"[CATEGORIZE PENDING] item={item_id} "
                f"reason={refusal.decision.reason} "
                f"would_cost_credits={refusal.decision.estimated_credits:.0f}"
            )
            return _pending_classification_payload(item, refusal.decision)
        source = visual.source
        fetch_method = visual.retrieval_method

        video_record: dict[str, Any] = {}
        if visual.frames:
            data, shoot_fingerprint, local_visual, video_record = (
                await _classify_video_batches(
                    visual,
                    item=item,
                    allow_core_qwen_fallback=allow_core_qwen_fallback,
                    force_qwen=force_qwen,
                )
            )
        else:
            # Classification does not need original-resolution media.
            # Normalizing every asset to a compact JPEG makes cost predictable
            # and also handles thumbnails whose declared MIME type is missing
            # or inaccurate.
            classifier_image = await asyncio.to_thread(
                _prepare_classifier_image,
                visual.image,
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
            # What the retrieval cost, per item, so vault media spend is
            # attributable to an asset rather than inferred from a total.
            "retrieval": {
                "method": visual.retrieval_method,
                "media_bytes": visual.media_bytes,
                "estimated_credits": visual.estimated_credits,
                "frames_sampled": len(visual.frames),
                "duration_seconds": round(float(visual.duration_seconds or 0), 2),
            },
        })
        if video_record:
            # The whole-clip semantic record: progression, setting, beginning /
            # middle / ending, meaningful changes and their timestamps.
            metadata["video"] = video_record
        if visual.skip_reason:
            metadata["analysis_skipped"] = {
                "reason": visual.skip_reason,
                "message": visual.skip_message,
            }
        print(
            f"[SHOOT FINGERPRINT] item={item_id} "
            f"status={shoot_fingerprint.get('status')} "
            f"palette={local_visual.get('palette_names') or []}"
        )
        # A video-level record is a better description than the per-image
        # prose builder can produce, because only it knows the clip moved.
        description = (
            describe_video_record(video_record)
            if video_record
            else media_description(metadata, source=source)
        ) or media_description(metadata, source=source)

        return {
            "id": item_id,
            "content_category": category,
            "ai_description": description,
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
            "classification_status": visual.status,
            "classification_skip_reason": visual.skip_reason,
            "classification_media_key": media_identity_key(item),
            "classification_retrieval_method": visual.retrieval_method,
            "classification_media_bytes": visual.media_bytes,
            "classification_media_credits": visual.estimated_credits,
            "classification_frames_sampled": len(visual.frames),
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


async def _count_classification_status(creator_id: str, status: str) -> int:
    """How many of this creator's rows are in one classification state.

    Used for the two states an operator can act on: ``partial`` (classified
    from a thumbnail because a deep video scan would have cost too much) and
    ``pending`` (not classified at all, for the same reason).
    """
    db = get_supabase()
    try:
        result = await asyncio.to_thread(
            lambda: db.table("creator_vault_media")
            .select("id", count="exact", head=True)
            .eq("creator_id", creator_id)
            .eq("classification_status", status)
            .execute()
        )
        return result.count or 0
    except Exception as exc:
        # The column arrives with db/vault_classification_persistence_v1.sql. A
        # backend deployed ahead of its migration reports zero rather than
        # failing the whole overview, which is true by construction: without
        # the column no row can be in that state.
        print(f"[VAULT] classification_status unavailable ({status}): {exc}")
        return 0


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
        # Items where the media-cost guard stopped short of a deep video scan.
        # Surfaced so an operator can see the trade being made rather than
        # wondering why some videos read thinner than others.
        "partial_classifications": await _count_classification_status(
            creator_id, CLASSIFICATION_STATUS_PARTIAL
        ),
        "pending_classifications": await _count_classification_status(
            creator_id, CLASSIFICATION_STATUS_PENDING
        ),
        "auto_media_download_limits": auto_download_limits().to_dict(),
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
        # Every provider call this run makes — link refreshes and any guarded
        # media download alike — is attributed to the vault, so the existing
        # credit telemetry can answer "what did classification cost?" without a
        # second accounting system.
        with apifansly_usage_category(CATEGORY_VAULT):
            await _run_vault_categorization(
                creator_id,
                item_ids=item_ids,
                mark_initial=mark_initial,
                upgrade_legacy=upgrade_legacy,
            )


async def _creator_apifansly_account_id(creator_id: str) -> str:
    """The creator's platform account id, for credit attribution.

    Best effort: a failed read costs attribution on the usage snapshot, never
    the classification run itself.
    """
    try:
        row = await asyncio.to_thread(
            lambda: get_supabase()
            .table("creators")
            .select("apifansly_account_id")
            .eq("id", creator_id)
            .single()
            .execute()
        )
        return str((row.data or {}).get("apifansly_account_id") or "")
    except Exception as exc:
        print(f"[CATEGORIZE] account attribution unavailable {creator_id}: {exc}")
        return ""


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
            "url, thumbnail_url, mimetype, filename, album_title, "
            # Read so services.vault_classification_state can decide, per row,
            # whether this is finished work. Without these the run would either
            # re-spend on everything or never retry a partial result.
            "content_category, classified_at, classification_version, "
            "classification_status, classification_media_key"
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

        mode = "new" if target_ids else ("upgrade" if upgrade_legacy else "initial")

        # The persistence gate. A row with a successful classification at the
        # current version is finished work and is dropped here, so an ordinary
        # daily sync spends nothing re-deriving answers it already has. An
        # explicit confirmed upgrade is the one path allowed to reprocess on a
        # version bump alone.
        candidates = len(all_items)
        all_items, selection_reasons = select_items_for_classification(
            all_items,
            classifier_version=VAULT_CLASSIFIER_VERSION,
            allow_version_upgrade=upgrade_legacy,
            operator_requested=upgrade_legacy,
        )
        total = len(all_items)
        _categorize_state[creator_id]["total"] = total
        _categorize_state[creator_id]["skipped_already_classified"] = max(
            candidates - total, 0
        )
        _categorize_state[creator_id]["selection_reasons"] = selection_reasons
        print(
            f"[CATEGORIZE] creator={creator_id} mode={mode} items={total} "
            f"candidates={candidates} reasons={selection_reasons}"
        )
        if not total:
            _categorize_state[creator_id].update({
                "status": "done",
                "done": 0,
                "errors": 0,
                "nothing_to_do": True,
            })
            print(
                f"[CATEGORIZE] creator={creator_id} nothing to do — "
                f"{candidates} candidate rows are already classified"
            )
            return

        account_id = await _creator_apifansly_account_id(creator_id)

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
        cost_deferred = 0
        media_bytes_spent = 0
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
                # What this run actually spent at the billed media proxy, and
                # how many items stopped short rather than spend it.
                "cost_deferred": cost_deferred,
                "media_bytes": media_bytes_spent,
                "media_credits": round(
                    estimated_credits_for_bytes(media_bytes_spent)
                    if media_bytes_spent
                    else 0.0,
                    2,
                ),
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
            nonlocal cost_deferred, media_bytes_spent
            while True:
                item = await next_item()
                if item is None:
                    return
                try:
                    result = await _categorize_single_item_with_retry(
                        item,
                        allow_core_qwen_fallback=allow_core_qwen_fallback,
                        visual_client=visual_client,
                        account_id=account_id,
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

                if result.get("pending"):
                    # Refused on media cost. Persisted so the operator can see
                    # it, counted separately so a run does not report a cost
                    # decision as a classification.
                    cost_deferred += 1
                provider_details = (
                    (result.get("classification_metadata") or {})
                    .get("provider_details") or {}
                )
                if provider_details.get("qwen_status") == "ready":
                    qwen_fallbacks += 1
                if provider_details.get("semantic_status") == "fallback":
                    semantic_failures += 1
                media_bytes_spent += int(
                    result.get("classification_media_bytes") or 0
                )

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
            f"semantic_failures={semantic_failures} "
            f"cost_deferred={cost_deferred} "
            f"media_mb={media_bytes_spent / (1024 * 1024):.1f} "
            f"media_credits={estimated_credits_for_bytes(media_bytes_spent):.1f}"
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
    account_id: str = "",
) -> dict:
    """Wrap _categorize_single_item with exponential backoff on 429."""
    for attempt in range(max_retries):
        try:
            return await _categorize_single_item(
                item,
                allow_core_qwen_fallback=allow_core_qwen_fallback,
                visual_client=visual_client,
                account_id=account_id,
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
        account_id=account_id,
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
        # Manual, operator-initiated, and already capped at a few per day: the
        # media-cost guard grants this path its larger ceiling and lets it try
        # a deep video scan ahead of the thumbnail.
        result = await _categorize_single_item(
            item,
            force_qwen=True,
            manual=True,
            account_id=await _creator_apifansly_account_id(creator_id),
        )
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


async def _simulation_owns_message(message_id: object, record: dict) -> bool:
    """Whether the owner-only simulator already owns this message's turn.

    Split out of the route so the decision is testable against a real Supabase
    database-webhook payload rather than only against the route's happy path.
    """
    from core.simulation import (
        is_simulation_owned_message_id,
        message_row_is_simulation_owned,
        payload_media_context,
    )

    if is_simulation_owned_message_id(message_id):
        return True

    media_context, present = payload_media_context(record)
    if present:
        return is_simulation_message(media_context)

    # The payload said nothing about media_context. That is not evidence that
    # the row has none — it is the absence of evidence, and the database is the
    # only authority. One indexed primary-key read, on a path that otherwise
    # runs the whole analyzer.
    #
    # Wrapped again here even though the helper swallows its own failures: an
    # ownership check must never be able to turn a fan's message into a 500 and
    # a Supabase redelivery loop. Not positively identified means processed.
    try:
        return await message_row_is_simulation_owned(message_id)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WEBHOOK] simulation ownership check failed id={message_id}: {exc}")
        return False


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

    # The owner-only simulator persists its fan message with an ordinary INSERT,
    # which fires this same database webhook. It then drives the real Full Auto
    # turn itself, so processing the row here as well ran situation analysis,
    # commercial state, price learning and the conversation director twice for
    # one simulated turn — and made every simulator reading invalid.
    #
    # Nothing is skipped by shape: only a row the simulator explicitly marked is
    # ignored, so an ordinary production fan message — including one typed into
    # a test fan's chat by hand — is processed exactly as before. The response is
    # a 2xx so Supabase treats the delivery as handled rather than retrying it.
    # Ownership is settled in three ways, cheapest first, and only the LAST one
    # depends on the payload being shaped the way a unit test shapes it:
    #
    #   1. this process wrote the row as a simulator event;
    #   2. the record carries the marker (decoded, or as jsonb text);
    #   3. the record did not carry a media_context key at all, so the database
    #      row is read and asked directly.
    #
    # (3) is the one that matters in production. Nothing here can skip an
    # ordinary fan message: every branch requires a positive identification.
    if await _simulation_owns_message(message_id, record):
        print(f"[WEBHOOK] simulation-owned message skipped id={message_id}")
        return {"status": "skipped - owner simulation"}

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

    # API Fansly bills 80 received webhook events as one credit. Counted after
    # authentication, so a rejected forgery cannot inflate the estimate, and
    # before any routing, so an event for an unknown account still counts — the
    # provider billed for delivering it either way.
    record_apifansly_webhook_event(
        payload.get("event"),
        account_id=payload.get("accountId"),
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
    def _count():
        def _build(apply_filter: bool):
            query = (
                get_supabase().table("vault_sets")
                .select("id", count="exact")
                .eq("creator_id", creator_id)
                .eq("status", "approved")
            )
            # Mirrored test content never makes a creator "ready for Auto".
            if apply_filter:
                query = exclude_simulation_only(query)
            return query.limit(1).execute()

        return run_live_catalog_query(_build, label="auto_availability.approved_sets")

    result = await asyncio.to_thread(_count)
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
            # Owner test fans are excluded: this preview is a statement about
            # how many real customers Full Auto would answer, and a simulation
            # fan is not one. Filtered in the database rather than in Python so
            # a page boundary cannot let one through.
            lambda start, end: exclude_simulation_fans(
                db.table("fans")
                .select("id, auto_mode, total_spent, spend_tier, needs_human_review")
                .eq("creator_id", creator_id)
            )
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

    # An operator sends these by hand, so the simulation filter is applied
    # unconditionally (include_simulation=False): the manual path must never be
    # able to put mirrored test media in front of a real fan, not even when it
    # is somehow invoked from inside a simulated turn.
    def _load_operator_vault_media():
        def _build(apply_filter: bool):
            query = (
                db.table("creator_vault_media")
                .select(
                    "id, fansly_media_id, media_id, url, thumbnail_url, mimetype, filename, "
                    "album_title, content_category, ai_description, price_min, price_max, is_active"
                )
                .eq("creator_id", creator_id)
                .eq("is_active", True)
            )
            if apply_filter:
                query = exclude_simulation_only(query)
            return query.execute()

        return run_live_catalog_query(
            _build, label="operator_ppv.vault_media", include_simulation=False
        )

    def _load_operator_approved_sets():
        def _build(apply_filter: bool):
            query = (
                db.table("vault_sets")
                .select(
                    "id, title, description, media_ids, suggested_price, base_price_cents, min_price_cents, "
                    "max_price_cents, dynamic_pricing_enabled, tags, status"
                )
                .eq("creator_id", creator_id)
                .eq("status", "approved")
            )
            if apply_filter:
                query = exclude_simulation_only(query)
            return query.order("created_at", desc=True).execute()

        return run_live_catalog_query(
            _build, label="operator_ppv.vault_sets", include_simulation=False
        )

    try:
        from services.ppv_delivery_ledger import list_fan_deliveries

        vault_rows, set_rows, message_rows, deliveries = await asyncio.gather(
            asyncio.to_thread(_load_operator_vault_media),
            asyncio.to_thread(_load_operator_approved_sets),
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

    # Report the bounds actually enforced below, not the raw columns. A legacy
    # row backfilled to min = max = base would otherwise show the operator a
    # one-price band while the commercial layer prices it across its approved
    # category range.
    approved_sets = []
    for row in (set_rows.data or []):
        base, minimum, maximum, dynamic = price_bounds(row)
        approved_sets.append({
            **row,
            "base_price_cents": base,
            "min_price_cents": minimum,
            "max_price_cents": maximum,
            "dynamic_pricing_enabled": dynamic,
        })

    return {
        "fan_id": fan_id,
        "creator_id": creator_id,
        "has_payment_pending": bool(fan.get("pending_ppv_check")),
        "media": media,
        "approved_sets": approved_sets,
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
            .select(
                "media_ids, status, suggested_price, tags, base_price_cents, "
                "min_price_cents, max_price_cents, dynamic_pricing_enabled"
            )
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
        # One authority for what this content may cost, so an operator and the
        # commercial layer cannot disagree about the same set.
        _, minimum, maximum, _ = price_bounds(approved_set)
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


# --- The Full Auto simulator ------------------------------------------------
#
# Two tiers, defined in core.simulation and enforced here.
#
# AGENCY — any authenticated operator, when the deployment has the simulator on.
# Scoped entirely by ordinary creator tenancy: an agency simulates the creators
# it already holds, against ``test_`` fans of those creators, planning against
# those creators' own approved vault. No route below widens tenancy for it.
#
# OWNER — the allowlisted platform owner, additionally permitted the CROSS-TENANT
# catalog mirror (source discovery, mirroring, un-mirroring, and mirrored-media
# preview whose provenance points at another tenant). Those routes are guarded
# by ``require_simulation_owner``; an agency account gets the same 404 as
# everything else and cannot learn the capability exists.
#
# Every rejection is that one 404, so a caller cannot probe which condition it
# failed or learn that another tenant's creator or fan exists.
#
# The frontend hides what an account may not use via GET /simulation-capabilities,
# but that is convenience only: nothing below trusts the client.


class SimulateInboundRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    fast: bool = True


class SimulateDeclineRequest(BaseModel):
    reason: str = "simulated_decline"


class SimulationCatalogMirrorRequest(BaseModel):
    """Which creator's vault to mirror into which creator's TEST catalog."""

    source_creator_id: str = Field(min_length=1, max_length=64)
    target_creator_id: str = Field(min_length=1, max_length=64)


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

    1-3 (master switch, agency access, authenticated user) come from
    core.simulation; 4 and 5 reuse the ordinary tenancy helper, so the simulator
    is subject to exactly the same tenancy model as every other creator route
    rather than a parallel one; 6 is the ``test_`` platform-fan boundary,
    enforced here on every simulation mutation so knowing a real fan's UUID is
    never enough.

    Identical for an agency operator and for the owner. Owner authority adds
    cross-tenant MIRROR capability, never cross-tenant simulation: nobody
    simulates a creator they are not assigned.
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
    """What local tooling this authenticated account may use.

    Three booleans about the CALLER and nothing else. It never exposes the
    allowlist, the environment variables, any user id, or why another user is
    not allowed.

    ``auto_simulation``     may open the simulator at all (agency tier).
    ``simulation_mirror``   may use the cross-tenant catalog mirror (owner).
    ``operator_diagnostics`` may see low-level retrieval/cost detail in
                            otherwise ordinary operator surfaces. Owner tier,
                            because it is diagnostic noise for an agency rather
                            than a second security boundary — the data it
                            reveals is the caller's own creators' either way.
    """
    from core.simulation import (
        request_is_platform_operator,
        request_is_simulation_owner,
        request_may_simulate,
    )

    return {
        "auto_simulation": bool(request_may_simulate(request)),
        "simulation_mirror": bool(request_is_simulation_owner(request)),
        # Identity, not a feature switch: turning the simulator off must not
        # also strip the owner's diagnostics from unrelated surfaces.
        "operator_diagnostics": bool(request_is_platform_operator(request)),
    }


def _read_simulation_test_fans(db, creator_ids: list[str]):
    """Read this creator's test fans, tolerating a missing ai_stack_profile column.

    ``ai_stack_profile`` is the per-test-fan AI Stack override, which is what
    lets two test fans under one creator be compared turn for turn. It arrives
    with db/ai_stack_profile_v1.sql.

    Until that migration is applied, PostgREST answers an unknown column with
    42703 and fails the WHOLE read — which is exactly how this endpoint went
    down before (#31), on a different column. So the select is retried without
    it. A backend that ships ahead of its migration keeps listing test fans; it
    simply reports no per-fan override yet, which is true by construction.
    """
    from core.simulation import TEST_FAN_PREFIX

    def _query(columns: str):
        return (
            db.table("fans")
            .select(columns)
            .in_("creator_id", creator_ids)
            .like("platform_fan_id", f"{TEST_FAN_PREFIX}%")
            .order("display_name")
            .limit(500)
            .execute()
        )

    try:
        return _query("id, display_name, creator_id, platform_fan_id, ai_stack_profile")
    except Exception as exc:
        text = str(exc).lower()
        if "ai_stack_profile" not in text:
            raise
        print(
            "[SIMULATION] ai_stack_profile column missing — apply "
            "db/ai_stack_profile_v1.sql. Listing test fans without per-fan "
            "AI stack overrides."
        )
        return _query("id, display_name, creator_id, platform_fan_id")


@app.get("/simulation/creators")
async def simulation_creators(request: Request) -> dict:
    """Creators the caller may simulate against, each with its test fans only.

    Scoped by the caller's ordinary creator assignments, then filtered to
    ``test_`` fans, so the simulator's pickers cannot enumerate real fans.

    This is the only creator listing the simulator has, and it is the same for
    an agency and for the owner: both see exactly the creators they are
    assigned. Being the platform owner is not a global read of the creators
    table — the cross-tenant listing is ``/simulation/catalog/sources``, which
    answers a different question and grants nothing here.
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
                lambda: _read_simulation_test_fans(db, ids),
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
                "platform_fan_id": fan.get("platform_fan_id"),
                "ai_stack_profile": fan.get("ai_stack_profile"),
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
    from core.simulation import (
        request_is_platform_operator,
        request_is_simulation_owner,
    )
    from services.ai_stack_visibility import public_message_rows
    from services.suggestions import run_simulated_inbound

    fan = await _require_simulatable_fan(request, creator_id, fan_id)
    # Only an owner's turn may plan against mirrored cross-tenant test rows.
    # An agency's turn plans against this creator's own approved vault and
    # sets — the same inventory live planning would use.
    mirrored = request_is_simulation_owner(request)
    # Each creator message carries the durable "which brain wrote this" marker
    # in its media_context, and that marker names the provider, the model, the
    # writer route and the prompt version. Diagnostics, so the platform owner
    # keeps all of it and an agency is told the profile alone.
    diagnostics = request_is_platform_operator(request)
    print(
        f"[SIMULATION] inbound creator={creator_id} fan={fan_id} "
        f"platform_fan={fan.get('platform_fan_id')} fast={body.fast} "
        f"mirrored_catalog={mirrored}"
    )
    try:
        turn = await run_simulated_inbound(
            fan_id=fan_id,
            creator_id=creator_id,
            message=body.message,
            fast=body.fast,
            include_mirrored_catalog=mirrored,
        )
        if not diagnostics:
            turn = {
                **turn,
                "creator_messages": public_message_rows(
                    turn.get("creator_messages")
                ),
            }
        return turn
    except HTTPException:
        raise
    except Exception as exc:
        # The caller here is an allowlisted owner running a diagnostic tool, and
        # the whole point of the tool is to find out what broke. A generic 500
        # told them nothing, and — because an unhandled exception's response
        # never passed through CORS — the browser reduced it further to "Failed
        # to fetch". Full traceback to the logs, the exception type and a
        # correlating id to the owner, no internals in the body.
        error_id = uuid.uuid4().hex[:12]
        print(
            f"[SIMULATION ERROR] id={error_id} creator={creator_id} fan={fan_id} "
            f"type={type(exc).__name__} error={exc}"
        )
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=(
                f"The simulated turn failed inside the backend "
                f"({type(exc).__name__}). Quote error {error_id} to find the "
                "full traceback in the server logs."
            ),
        ) from exc


async def _require_simulation_catalog_access(
    request: Request,
    source_creator_id: str,
    target_creator_id: str,
) -> None:
    """OWNER tier, with deliberately different rules for source and target.

    The two sides of a mirror are not the same kind of thing, and requiring the
    same authorization for both was the bug this asymmetry fixes.

    TARGET — written to. Ordinary creator tenancy applies, unchanged: mirrored
    rows are inserted into this creator's catalog, so the caller must be
    someone who ordinarily holds it. This is the simulation creator (Sophia).

    SOURCE — read from, metadata only, never modified. Owner-gated but NOT
    tenancy-gated, because the whole point is to build a realistic test catalog
    from a real, AGENCY-OWNED creator's vault (Eliz). Requiring ordinary tenancy
    on the source would mean assigning the platform owner to that agency's
    creator, which hands over its chats, fans and revenue in order to copy some
    scene metadata — far more access than the job needs, and permanent.

    What the looser source rule does NOT grant, and why that is safe:

    * no write of any kind reaches the source (``mirror_creator_catalog`` and
      ``delete_creator_catalog_mirror`` scope every statement to the target,
      and deletions additionally require ``simulation_only = true`` plus a
      matching ``source_creator_id``);
    * the source does not become simulatable — ``/simulation/creators`` is
      untouched and still tenancy-scoped, so the owner cannot chat as it;
    * no other route widens. This is the only place a cross-tenant creator id
      is accepted, and only in the source position;
    * mirrored rows stay ``simulation_only`` with rewritten ``sim:`` ids, so
      the source's platform media ids never become deliverable under the
      target.

    The owner allowlist decides WHO, and every rejection is the same 404 the
    rest of the simulator uses, so an agency account cannot discover that this
    capability exists or probe for creator ids with it.

    This is the ONE capability the owner tier exists for. Since the simulator
    itself became available to ordinary agency operators, passing the agency
    check is emphatically not enough here: an agency operator legitimately
    holds its own creators, so tenancy on the TARGET succeeds for it, and the
    owner allowlist is the only thing standing between it and another tenant's
    vault. ``require_simulation_owner`` — not ``require_simulation_user`` — is
    therefore the whole boundary.
    """
    from core.simulation import not_found, require_simulation_owner
    from services.simulation_catalog import mirror_source_exists

    await require_simulation_owner(request)

    allowed = await _creator_ids_for_user_cached(request)
    if str(target_creator_id) not in allowed:
        # The written-to side. Unchanged, and the reason an owner still cannot
        # mirror INTO a creator they do not hold.
        raise not_found()

    if not await mirror_source_exists(str(source_creator_id)):
        # Only reachable by an authenticated allowlisted owner, who may already
        # enumerate every creator through the source listing, so naming a
        # mistyped id costs nothing and saves a confusing empty mirror.
        raise not_found()


@app.get("/simulation/catalog/sources")
async def list_simulation_catalog_sources(request: Request) -> dict:
    """Creators whose vault may be COPIED FROM. Owner only, cross-tenant.

    Deliberately separate from ``/simulation/creators``, which answers a
    different question — who the caller may simulate AS — and stays
    tenancy-scoped for everybody. A creator appearing here gains nothing: it
    does not enter the simulator selector, it creates no assignment, and the
    only thing it enables is being named as the SOURCE of a mirror whose target
    the caller must ordinarily hold.

    This route enumerates EVERY creator in the deployment, across tenants, so
    it is the sharpest edge in the simulator and is owner-gated accordingly.
    An agency operator — who may now use the simulator perfectly legitimately —
    gets the same 404 an unauthenticated caller does, and therefore cannot
    learn that other tenants exist, let alone name one as a mirror source.

    Returns the minimum the picker needs: id, display name, and whether there is
    real approved content worth mirroring. Nothing about the account itself.
    """
    from core.simulation import require_simulation_owner
    from services.simulation_catalog import list_mirror_source_creators

    await require_simulation_owner(request)
    try:
        sources = await list_mirror_source_creators()
    except Exception as exc:
        print(f"[SIMULATION CATALOG] source listing failed: {exc}")
        # 503, not 404: authorization already passed, so this caller is a
        # verified owner and an empty list would be a lie about the data rather
        # than a report of a failed read.
        raise HTTPException(
            status_code=503,
            detail="Could not read mirror sources. Please retry.",
        ) from exc
    return {"sources": [source.to_dict() for source in sources]}


@app.post("/simulation/catalog/mirror")
async def mirror_simulation_catalog(
    body: SimulationCatalogMirrorRequest,
    request: Request,
) -> dict:
    """Refresh the target creator's TEST catalog from a source vault. Owner only.

    Mirrored rows are marked ``simulation_only`` and carry rewritten ``sim:``
    media ids, so they are visible to the simulator, excluded from live package
    planning, and refused by every delivery path. The source creator's vault is
    read only — a mirror can be rebuilt or deleted without touching it.
    """
    from services.simulation_catalog import (
        SimulationCatalogError,
        mirror_creator_catalog,
    )

    await _require_simulation_catalog_access(
        request, body.source_creator_id, body.target_creator_id
    )
    try:
        result = await mirror_creator_catalog(
            source_creator_id=body.source_creator_id,
            target_creator_id=body.target_creator_id,
        )
    except SimulationCatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"[SIMULATION CATALOG] mirror failed: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Could not mirror the simulation catalog. Please retry.",
        ) from exc
    return {"status": "ok", **result.to_dict()}


@app.post("/simulation/catalog/mirror/delete")
async def delete_simulation_catalog_mirror(
    body: SimulationCatalogMirrorRequest,
    request: Request,
) -> dict:
    """Remove a mirror. Deletes only mirrored rows on the target creator.

    POST rather than DELETE because the operation is identified by a body, not
    by a path: the pair (source, target) is what names a mirror.
    """
    from services.simulation_catalog import (
        SimulationCatalogError,
        delete_creator_catalog_mirror,
    )

    await _require_simulation_catalog_access(
        request, body.source_creator_id, body.target_creator_id
    )
    try:
        result = await delete_creator_catalog_mirror(
            source_creator_id=body.source_creator_id,
            target_creator_id=body.target_creator_id,
        )
    except SimulationCatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"[SIMULATION CATALOG] mirror delete failed: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Could not delete the simulation catalog mirror. Please retry.",
        ) from exc
    return {"status": "ok", **result.to_dict()}


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


class SimulationTestFanRequest(BaseModel):
    """Create one owner test fan. The platform id is NEVER client-supplied."""

    creator_id: str = Field(min_length=1, max_length=64)
    display_name: str | None = Field(default=None, max_length=80)


class SimulationMediaPreviewRequest(BaseModel):
    media_ids: list[str] = Field(default_factory=list, max_length=250)


class AIStackOverrideRequest(BaseModel):
    """Select an AI Stack Profile by identifier, or clear the override.

    A stable identifier from the backend registry, never a provider or model
    string. The frontend cannot name a model here, by construction.
    """

    ai_stack_profile: str | None = None


@app.post("/simulation/test-fans")
async def create_simulation_test_fan(
    body: SimulationTestFanRequest,
    request: Request,
) -> dict:
    """Create a clean, persistent simulation fan under a creator you hold.

    The platform id is generated server-side and always carries the ``test_``
    prefix, so this control cannot produce a fan the rest of the system would
    treat as real. Nothing about a Fansly account is touched.

    Agency operators may do this for their own creators: a ``test_`` fan is
    exactly the isolation boundary that makes simulating safe, so being able to
    create one is part of the agency tier rather than an owner privilege. The
    ordinary tenancy check below is what keeps it to creators the caller holds.
    """
    from core.simulation import require_simulation_user
    from services.simulation_workspace import (
        SimulationWorkspaceError,
        create_test_fan,
    )

    await require_simulation_user(request)
    # The ordinary creator tenancy check, so a simulator user still cannot
    # create a fan under a creator they are not assigned to.
    await require_creator_path_access(request, body.creator_id)
    try:
        fan = await create_test_fan(
            creator_id=body.creator_id,
            display_name=body.display_name,
        )
    except SimulationWorkspaceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "fan": fan}


@app.get("/creator/{creator_id}/fan/{fan_id}/simulation-state")
async def read_simulation_state(
    creator_id: str,
    fan_id: str,
    request: Request,
) -> dict:
    """The persisted state of one test fan, from authoritative sources."""
    from core.simulation import not_found
    from services.simulation_workspace import NotASimulationFan, simulation_state

    await _require_simulatable_fan(request, creator_id, fan_id)
    try:
        return await simulation_state(creator_id=creator_id, fan_id=fan_id)
    except NotASimulationFan as exc:
        raise not_found() from exc


@app.post("/creator/{creator_id}/fan/{fan_id}/simulation-actions/{action_id}/run-now")
async def run_simulation_action_now(
    creator_id: str,
    fan_id: str,
    action_id: str,
    request: Request,
) -> dict:
    """Fire one pending scheduled action immediately, through the real handler.

    Simulator users, own creators, test fans only. This is how delayed behaviour — payday
    re-engagement, post-session follow-up, re-engagement after silence — is
    tested without waiting days and without a fake clock: the production
    revalidation, planner, writer and state transitions all run, inside
    ``simulation_scope()``, so no platform call is possible.
    """
    from core.simulation import not_found, request_is_simulation_owner
    from services.simulation_workspace import (
        NotASimulationFan,
        SimulationWorkspaceError,
        run_scheduled_action_now,
    )

    await _require_simulatable_fan(request, creator_id, fan_id)
    try:
        return await run_scheduled_action_now(
            creator_id=creator_id,
            fan_id=fan_id,
            action_id=action_id,
            include_mirrored_catalog=request_is_simulation_owner(request),
        )
    except NotASimulationFan as exc:
        raise not_found() from exc
    except SimulationWorkspaceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/creator/{creator_id}/simulation-media-previews")
async def read_simulation_media_previews(
    creator_id: str,
    body: SimulationMediaPreviewRequest,
    request: Request,
) -> dict:
    """Display URLs for mirrored ``sim:`` test media on a creator you hold.

    Resolved through the mirror's provenance from the SOURCE creator's vault.
    Nothing is written, and the source's platform media id is never copied onto
    this creator's catalog, so a previewable row does not become a deliverable
    one: simulation preview access and live delivery authority stay separate.

    Open to any simulator user for their own creators, but the SOURCE side is
    tiered. The owner resolves any provenance, because resolving a mirror they
    created is what the mirror is for. Everyone else resolves only provenance
    pointing at a vault they already hold, so a cross-tenant mirror renders as
    nothing rather than becoming a way to read another tenant's media through a
    creator of one's own.
    """
    from core.simulation import request_is_simulation_owner, require_simulation_user
    from services.simulation_catalog import resolve_simulation_media_previews

    await require_simulation_user(request)
    await require_creator_path_access(request, creator_id)
    allowed_sources = (
        None
        if request_is_simulation_owner(request)
        else await _creator_ids_for_user_cached(request)
    )
    media = await resolve_simulation_media_previews(
        creator_id=creator_id,
        media_ids=list(body.media_ids or []),
        allowed_source_creator_ids=allowed_sources,
    )
    return {"media": media}


@app.put("/creator/{creator_id}/fan/{fan_id}/ai-stack")
async def update_simulation_fan_ai_stack(
    creator_id: str,
    fan_id: str,
    body: AIStackOverrideRequest,
    request: Request,
) -> dict:
    """Pin one TEST fan to an AI Stack Profile. Simulator users, own creators.

    This is what lets "Test Fan A -> cleo_legacy_v1" and "Test Fan B -> cleo_v2"
    run under the same creator and be compared turn for turn. The route refuses
    any fan that is not a ``test_`` fan, and the read path in services/ai_stack
    re-checks the prefix independently, so a value that reached a real fan row
    by any other means still has no effect.
    """
    from core.simulation import not_found
    from services.ai_stack import set_simulation_fan_profile_override

    await _require_simulatable_fan(request, creator_id, fan_id)
    try:
        stored = await set_simulation_fan_profile_override(
            fan_id, body.ai_stack_profile
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"[AI STACK] simulation override write failed fan={fan_id}: {exc}")
        raise not_found() from exc
    return {"status": "ok", "fan_id": fan_id, "ai_stack_profile": stored}


@app.get("/content-price-ranges")
async def read_content_price_ranges(request: Request) -> dict:
    """The agency's approved commercial price range per content category.

    Read-only, and the same table models/content_pricing.py prices every offer
    from. The Sets UI needs it for two honest things: to name where a set's
    allowed range came from ("nude_photo default"), and to restore that default
    after an operator has customised one.

    Authenticated by the application's own middleware (deployment key plus the
    signed-in Supabase session) but deliberately not creator-scoped: this is
    deployment configuration rather than anyone's data — the same numbers for
    every tenant, containing nothing about any creator, fan or sale.
    """
    from models.content_pricing import VAULT_CATEGORIES

    # Referenced so the signature stays honest about needing an authenticated
    # request; the middleware has already rejected an unauthenticated one.
    _ = dashboard_user_id(request)
    return {
        "categories": [
            {
                "category": name,
                "label": value["label"],
                "min_dollars": int(value["min"]),
                "max_dollars": int(value["max"]),
                # A free/teaser/unclear category is NOT an approved commercial
                # range and must never be presented as one.
                "priced": int(value["max"]) > 0,
            }
            for name, value in sorted(VAULT_CATEGORIES.items())
        ]
    }


# --- Agency pricing strategy ------------------------------------------------
#
# Railway's PRICE_LEARNING_* variables are DEPLOYMENT DEFAULTS. An agency
# configures strategy here, and the precedence the backend already implements is
# creator override, then agency policy, then those defaults
# (db/pricing_policy_queries.get_effective_price_learning_policy).
#
# These routes are ordinary creator-scoped operator endpoints, not owner-only:
# an agency configuring its own pricing is the point. Authorization is the
# standard tenancy check, so an agency can only read or write a policy for a
# creator it is actually assigned, and the agency scope it may write is the one
# its own creator belongs to.
#
# Nothing here changes a pricing formula. A preset is a named set of values for
# fields that already exist, and every write is validated against
# PriceLearningPolicy before it is stored.


class PricingPolicyRequest(BaseModel):
    """Set a strategy preset, explicit advanced settings, or both."""

    scope: Literal["creator", "agency"] = "creator"
    preset: str | None = None
    # Advanced tuning for operators who want the exact numbers. Applied on top
    # of the preset when both are sent, so "Aggressive, but cap the ceiling" is
    # expressible in one request.
    settings: dict[str, Any] | None = None


async def _pricing_scope_target(
    request: Request,
    creator_id: str,
    scope: str,
) -> tuple[str, str]:
    """Resolve a request's scope to (scope_type, scope_id), or 400/403.

    An agency-scope write is allowed only for the agency scope this creator
    actually belongs to. That keeps the write inside the same tenancy boundary
    as everything else: an operator cannot address another agency's policy by
    naming its scope id, because no scope id is ever accepted from the client.
    """
    from db.pricing_policy_queries import get_agency_scope_id

    await require_creator_path_access(request, creator_id)
    if scope == "creator":
        return "CREATOR", str(creator_id)

    agency_scope_id = await get_agency_scope_id(creator_id)
    if not agency_scope_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "This creator is not a member of an agency pricing scope, so "
                "there is no agency policy to configure. Set a creator policy "
                "instead."
            ),
        )
    return "AGENCY", agency_scope_id


@app.get(
    "/creator/{creator_id}/pricing-policy",
    dependencies=[Depends(require_creator_path_access)],
)
async def read_pricing_policy(creator_id: str) -> dict:
    """The effective pricing policy and where each layer's values came from."""
    from db.pricing_policy_queries import (
        environment_price_learning_policy,
        get_agency_scope_id,
        get_effective_price_learning_policy,
        get_policy_scope_settings,
    )
    from services.price_learning import price_learning_enabled
    from services.pricing_presets import describe_presets, preset_for_policy

    agency_scope_id = await get_agency_scope_id(creator_id)
    creator_settings, agency_settings, effective = await asyncio.gather(
        get_policy_scope_settings("CREATOR", creator_id),
        (
            get_policy_scope_settings("AGENCY", agency_scope_id)
            if agency_scope_id
            else asyncio.sleep(0, result={})
        ),
        get_effective_price_learning_policy(creator_id),
    )
    return {
        "creator_id": creator_id,
        "agency_scope_id": agency_scope_id,
        # Honest about the deployment gate: adaptive pricing does nothing at all
        # unless PRICE_LEARNING_ENABLED is true, and the UI must not present a
        # strategy as active while the backend has the feature switched off.
        "price_learning_enabled": price_learning_enabled(),
        "price_learning_env_var": "PRICE_LEARNING_ENABLED",
        "environment_defaults": environment_price_learning_policy().model_dump(),
        "agency": {
            "settings": agency_settings,
            "preset": preset_for_policy(agency_settings) if agency_scope_id else None,
        },
        "creator": {
            "settings": creator_settings,
            "preset": preset_for_policy(creator_settings) if creator_settings else None,
        },
        "effective": effective.model_dump(),
        "effective_preset": preset_for_policy(effective.model_dump()),
        "presets": describe_presets(),
    }


@app.put("/creator/{creator_id}/pricing-policy")
async def update_pricing_policy(
    creator_id: str,
    body: PricingPolicyRequest,
    request: Request,
) -> dict:
    """Write one pricing-policy scope. Agencies configure their own only."""
    from db.pricing_policy_queries import (
        get_effective_price_learning_policy,
        get_policy_scope_settings,
        save_policy_scope_settings,
    )
    from services.pricing_presets import apply_preset, normalize_preset

    scope_type, scope_id = await _pricing_scope_target(request, creator_id, body.scope)

    settings = await get_policy_scope_settings(scope_type, scope_id)
    if body.preset is not None:
        if normalize_preset(body.preset) is None:
            raise HTTPException(
                status_code=400,
                detail="Unknown pricing preset.",
            )
        settings = apply_preset(settings, body.preset)
    if body.settings:
        # Advanced values win over the preset, so both can be sent together.
        settings = {**settings, **body.settings}

    try:
        stored = await save_policy_scope_settings(scope_type, scope_id, settings)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"[PRICING POLICY] write failed {scope_type}/{scope_id}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Could not save the pricing policy. Please retry.",
        ) from exc

    effective = await get_effective_price_learning_policy(creator_id)
    print(
        f"[PRICING POLICY] {scope_type}/{scope_id} updated by creator={creator_id} "
        f"fields={sorted(stored)}"
    )
    return {
        "status": "ok",
        "scope": scope_type.lower(),
        "scope_id": scope_id,
        "settings": stored,
        "effective": effective.model_dump(),
    }


# --- AI Stack administration ------------------------------------------------
#
# The whole conversational AI configuration is a named profile
# (ai/stack_profiles.py). These routes let an operator read what each profile
# actually means and pin one creator to one.
#
# They share the simulator's gate, which is what "AI stack selection is part of
# the simulator" has always meant here — so as the simulator opened to agency
# operators, so did these, for the creators those operators already hold.
# That is deliberate: comparing profiles turn for turn is the reason to
# simulate at all, and a profile choice is scoped to one creator by the
# ordinary tenancy check below. ``/ai-stack/profiles`` itself is the registry
# of what may be chosen — deployment configuration, the same for every tenant
# and containing nothing about any creator, fan or sale.
#
# There is deliberately no way to submit a provider or model string. The only
# thing a client may send is a stable profile identifier, which is validated
# against the backend registry.


@app.get("/ai-stack/profiles")
async def read_ai_stack_profiles(request: Request) -> dict:
    """The registry a caller may choose from, at the detail it may see.

    Two answers, one route. An agency operator gets the product-level identity
    of every profile — ``{"id": "cleo_v3", "name": "Cleo V3"}`` — which is
    exactly what the Simulator's dropdown needs and the whole of what an agency
    is told. The platform owner additionally gets the routing: every stage's
    provider, model, fallback, prompt version and generation settings.

    The reduction happens HERE, on the response, not in the dashboard. A field
    a client is not supposed to have must not be in the body it receives; see
    services/ai_stack_visibility.py for why and for what counts as which.
    """
    from ai.stack_profiles import PROFILE_ENV_VAR, environment_profile_id
    from core.simulation import request_is_platform_operator, require_simulation_user
    from services.ai_stack_visibility import registry_view

    await require_simulation_user(request)
    diagnostics = request_is_platform_operator(request)
    body: dict = {
        "profiles": registry_view(diagnostics=diagnostics),
        # A profile id, which is product-level in exactly the way the ids in
        # ``profiles`` are: it says which stack answers by default, not what
        # that stack is made of.
        "environment_profile": environment_profile_id(),
        # Told plainly, so a client never has to infer the shape it got from
        # which keys happen to be present.
        "diagnostics": diagnostics,
    }
    if diagnostics:
        # The name of the deployment variable is operator configuration; an
        # agency has nothing to do with it and is not shown it.
        body["environment_variable"] = PROFILE_ENV_VAR
    return body


@app.get("/creator/{creator_id}/ai-stack")
async def read_creator_ai_stack(creator_id: str, request: Request) -> dict:
    """This creator's override and the profile that would actually answer."""
    from ai.stack_profiles import environment_profile_id
    from core.simulation import require_simulation_user
    from services.ai_stack import creator_profile_override, resolve_ai_stack

    await require_simulation_user(request)
    await require_creator_path_access(request, creator_id)
    override = await creator_profile_override(creator_id)
    effective = await resolve_ai_stack(creator_id=creator_id)
    return {
        "creator_id": creator_id,
        "override": override,
        "environment_profile": environment_profile_id(),
        "effective": effective.to_dict(),
    }


@app.put("/creator/{creator_id}/ai-stack")
async def update_creator_ai_stack(
    creator_id: str,
    body: AIStackOverrideRequest,
    request: Request,
) -> dict:
    """Persist (or clear) this creator's AI Stack Profile override.

    Persistent and creator-scoped rather than session-scoped, because Full Auto
    answers asynchronously from a worker where no browser session exists.
    """
    from core.simulation import require_simulation_user
    from services.ai_stack import resolve_ai_stack, set_creator_profile_override

    await require_simulation_user(request)
    await require_creator_path_access(request, creator_id)
    try:
        stored = await set_creator_profile_override(creator_id, body.ai_stack_profile)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"[AI STACK] creator override write failed creator={creator_id}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Could not save the AI stack override. Please retry.",
        ) from exc
    effective = await resolve_ai_stack(creator_id=creator_id)
    print(f"[AI STACK] creator={creator_id} override set to {stored or 'none'}")
    return {
        "status": "ok",
        "creator_id": creator_id,
        "override": stored,
        "effective": effective.to_dict(),
    }


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
        for field in (
            "base_price_cents",
            "min_price_cents",
            "max_price_cents",
            # Experience metadata (db/experience_director_v1.sql). Generated
            # once here, during classification, rather than asked of the live
            # writer — see services/scene_metadata.py.
            "paid_sellable",
            "scene_key",
            "scene_premise",
            "intensity_level",
            "reveals",
            "setup_line",
            "continuation",
        ):
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
async def model_runtime_health(request: Request) -> dict:
    """Cached writer availability, without spending AI tokens.

    Every authenticated operator needs the verdict — an agency whose replies
    are degraded should see that in the health banner. Only the platform owner
    needs the supply chain behind it: the cached document names each configured
    provider and model, and ``detail`` names them again in prose, so an agency
    gets the status, the timestamp and a generic sentence instead.

    The analyzer counters are deliberately unredacted: they are numbers of
    degraded analyses by reason code, and name no provider or model.
    """
    from core.simulation import request_is_platform_operator
    from services.ai_stack_visibility import public_model_health
    from services.analyzer_telemetry import analyzer_health

    health = current_model_availability()
    if not request_is_platform_operator(request):
        health = public_model_health(health)
    return {
        **health,
        # REL-001 — degraded-analysis counts, so an analyzer incident is
        # countable without reading logs.
        "analyzer": analyzer_health(hours=1),
        "analyzer_24h": analyzer_health(hours=24),
    }


from routes.fansly import fansly_router

app.include_router(fansly_router)
