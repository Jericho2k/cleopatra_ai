"""FastAPI entrypoint for Cleopatra.

Routes are thin and delegate all logic to services.
"""

import asyncio
from core.tasks import spawn
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ai.generator import generate_replies
from ai.prompt_builder import build_prompt
from ai.situation_analyzer import analyze_situation
from ai.rag import find_similar_exchanges
from ai.stage_classifier import classify_stage
from core.supabase import get_supabase
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
    save_message,
)
from models.commercial import CreatorPolicy
from models.schemas import (
    ConversationContext,
    Fan,
    Persona,
    SuggestionRequest,
    SuggestionResponse,
)
from services.fan_intelligence import learn_from_fan_message
from services.apifansly import (
    ApiFanslyAccountAccessError,
    ApiFanslyConfigurationError,
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
)
from services.auto_audience import AutoAudiencePolicy
from services.fansly_poller import FanslyPoller
from services.fansly_session_store import SessionStore
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


_processed_messages: set = set()
_vault_sync_state: dict = {}


async def get_or_fetch_group_id(apifansly_id: str, platform_fan_id: str, fan_id: str) -> str | None:
    """Find the group_id for a fan by scanning recent chats."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
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


async def send_fansly_message(account_id: str, group_id: str, text: str) -> bool:
    try:
        await send_apifansly_message(
            account_id,
            group_id,
            content=text,
        )
        print(f"[SEND] account={account_id} group={group_id} accepted=true")
        return True
    except Exception as e:
        print(f"[SEND ERROR] {e}")
        return False


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
    audience_row, memberships = await asyncio.gather(
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
    effective_auto = eligibility.eligible

    print(
        f"[AUTO MODE] creator={creator_id} creator_auto={auto_mode} "
        f"fan_auto={fan_profile.auto_mode} effective_auto={effective_auto} "
        f"reason={eligibility.reason} fan={fan_id}"
    )

    if effective_auto:
        schedule_auto_reply(fan_id, creator_id)
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
        if len(_processed_messages) > 1000:
            _processed_messages.clear()

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

    await process_incoming_fan_message(
        str(fan.id), creator_id, content, auto_mode, message_id or None,
    )


session_store: SessionStore = None
fansly_poller: FanslyPoller = None
ppv_sweep_task: asyncio.Task | None = None
vault_autosync_task: asyncio.Task | None = None
scheduled_actions_task: asyncio.Task | None = None
chat_reconcile_task: asyncio.Task | None = None


async def ppv_sweep_scheduler():
    """Runs stale PPV verification sweep every 15 minutes."""
    while True:
        await asyncio.sleep(15 * 60)
        try:
            print("[CRON] Running PPV sweep...")
            from services.suggestions import sweep_stale_ppv_checks

            await sweep_stale_ppv_checks()
        except Exception as e:
            print(f"[CRON PPV SWEEP ERROR] {e}")


async def _scheduled_actions_scheduler():
    """Runs the commercial scheduled-actions queue (payday re-engagement, etc)."""
    from workers.scheduled_actions import process_once
    while True:
        await asyncio.sleep(60)
        try:
            sent = await process_once()
            if sent:
                print(f"[CRON] scheduled actions: sent {sent}")
        except Exception as e:
            print(f"[CRON SCHEDULED ACTIONS ERROR] {e}")


async def vault_autosync_scheduler():
    """Once a day, sync creators whose vault is stale.

    ``_run_vault_sync`` owns the exact IDs imported by that run and, when the
    creator opted in, categorizes only those IDs. The scheduler must never scan
    every uncategorized record after a sync because that can accidentally rerun
    the initial-vault job.
    """
    while True:
        await asyncio.sleep(24 * 60 * 60)
        try:
            print("[CRON] Vault auto-sync pass...")
            db = get_supabase()
            creators = await asyncio.to_thread(
                lambda: db.table("creators").select("id").execute()
            )
            for c in (creators.data or []):
                cid = c["id"]
                cd = await _vault_cooldown_remaining(cid, "last_vault_sync_at")
                if not cd["allowed"]:
                    continue
                # Reuse the guarded endpoint logic; it stamps + skips if already running.
                res = await sync_vault_start(cid)
                if res.get("status") != "started":
                    continue
                print(f"[CRON] auto-sync started creator={cid}")
        except Exception as e:
            print(f"[CRON VAULT AUTOSYNC ERROR] {e}")


async def chat_reconciliation_scheduler():
    """Incrementally reconcile Fansly chat lists with a durable DB lease."""
    while True:
        await asyncio.sleep(10 * 60)
        try:
            db = get_supabase()
            creators = await asyncio.to_thread(
                lambda: db.table("creators")
                .select("id")
                .not_.is_("apifansly_account_id", "null")
                .execute()
            )
            for creator in (creators.data or []):
                result = await sync_chats(str(creator["id"]), incremental=True)
                if result.get("status") == "ok" and result.get("new_chats"):
                    print(
                        f"[CHAT RECONCILE] creator={creator['id']} "
                        f"new={result['new_chats']} synced={result.get('synced', 0)}"
                    )
        except Exception as exc:
            print(f"[CRON CHAT RECONCILE ERROR] {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_store, fansly_poller, ppv_sweep_task, vault_autosync_task, scheduled_actions_task, chat_reconcile_task

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
#   • external webhooks                         -> require WEBHOOK_SECRET
#   • everything else (operator/CRUD/admin)     -> require DASHBOARD_API_SECRET
# Unconfigured secrets fail open in dev, closed in prod (see core/auth.py).
from starlette.responses import JSONResponse
from core.auth import (
    _is_dev,
    _consteq,
    authenticated_dashboard_user,
    dashboard_user_id,
)

_PUBLIC_PATHS = {"/health", "/"}
_WEBHOOK_PATHS = {"/webhook/fansly", "/generate-suggestions"}


@app.middleware("http")
async def api_auth_middleware(request, call_next):
    path = request.url.path
    if request.method == "OPTIONS" or path in _PUBLIC_PATHS:
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


@app.post("/reply")
async def save_reply(req: ReplyRequest, request: Request) -> dict:
    await require_creator_fan_access(request, req.creator_id, req.fan_id)
    await save_message(
        req.fan_id,
        req.creator_id,
        "creator",
        req.content,
        req.was_ai_suggested
    )

    db = get_supabase()
    fan_row = await asyncio.to_thread(
        lambda: db.table("fans")
        .select("fansly_group_id")
        .eq("id", req.fan_id).single().execute()
    )
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("apifansly_account_id")
        .eq("id", req.creator_id).single().execute()
    )

    group_id = (fan_row.data or {}).get("fansly_group_id")
    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")

    if group_id and apifansly_id:
        await send_fansly_message(apifansly_id, group_id, req.content)

    return {"status": "ok"}


@app.post("/connect-creator")
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
    import httpx

    async with httpx.AsyncClient() as client:
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


@app.post("/connect-creator-2fa")
async def connect_creator_2fa(req: Connect2FARequest, request: Request) -> dict:
    if req.creator_id:
        await require_creator_access(request, req.creator_id)
    operator_id = dashboard_user_id(request) or req.user_id
    if not operator_id:
        raise HTTPException(status_code=401, detail="Missing dashboard user session")
    import httpx

    async with httpx.AsyncClient() as client:
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


@app.post(
    "/sync-chats/{creator_id}",
    dependencies=[Depends(require_creator_path_access)],
)
async def sync_chats(
    creator_id: str,
    incremental: bool = False,
    force: bool = False,
) -> dict:
    import httpx

    db = get_supabase()
    creator_row = await asyncio.to_thread(
        lambda cid=creator_id: db.table("creators")
        .select("apifansly_account_id, fansly_account_id")
        .eq("id", cid)
        .single()
        .execute()
    )

    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
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

    existing_platform_ids: set[str] = set()
    if incremental:
        existing = await asyncio.to_thread(
            lambda: db.table("fans")
            .select("platform_fan_id")
            .eq("creator_id", creator_id)
            .execute()
        )
        existing_platform_ids = {
            str(row.get("platform_fan_id"))
            for row in (existing.data or [])
            if row.get("platform_fan_id")
        }

    async with httpx.AsyncClient() as client:
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
        new_chats = 0
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

            fan = await get_fan(creator_id, platform_fan_id)
            if not fan:
                fan = await create_fan(creator_id, platform_fan_id, fan_name)
                new_chats += 1

            update_payload = {
                "fansly_group_id": group_id,
                "display_name": fan_name,
            }
            if avatar_url:
                update_payload["avatar_url"] = avatar_url

            await asyncio.to_thread(
                lambda fid=fan.id, p=update_payload: db.table("fans").update(p).eq("id", fid).execute()
            )
            synced += 1

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
        print(
            f"[SYNC CHATS] incremental={incremental} total_chats={len(all_chats)} "
            f"synced={synced} new={new_chats}"
        )
        return {
            "status": "ok",
            "mode": "incremental" if incremental else "full",
            "synced": synced,
            "new_chats": new_chats,
            "audience": audience_sync,
        }


@app.post(
    "/load-history/{creator_id}/{fan_id}",
    dependencies=[Depends(require_creator_fan_access)],
)
async def load_fan_history(creator_id: str, fan_id: str) -> dict:
    import httpx

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

    existing = await asyncio.to_thread(
        lambda: db.table("messages")
        .select("fansly_message_id")
        .eq("fan_id", fan_id)
        .execute()
    )
    existing_ids = {r["fansly_message_id"] for r in (existing.data or []) if r.get("fansly_message_id")}

    all_messages = []
    all_media = {}
    cursor = None

    print(f"[LOAD HISTORY URL] apifansly_id={apifansly_id} group_id={group_id}")

    async with httpx.AsyncClient() as client:
        while True:
            messages, account_media_batch, cursor = (
                await apifansly_list_chat_messages(
                    str(apifansly_id),
                    str(group_id),
                    cursor=cursor,
                    limit=10,
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

    imported = 0
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

        await asyncio.to_thread(
            lambda r=row: db.table("messages").insert(r).execute()
        )
        existing_ids.add(msg_id)
        imported += 1

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
    dependencies=[Depends(require_creator_path_access)],
)
async def mark_all_read(creator_id: str) -> dict:
    import httpx

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

    async with httpx.AsyncClient() as client:
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
    dependencies=[Depends(require_creator_path_access)],
)
async def sync_vault(creator_id: str) -> dict:
    import httpx

    db = get_supabase()
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("apifansly_account_id")
        .eq("id", creator_id)
        .single()
        .execute()
    )

    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
    async with httpx.AsyncClient() as client:
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
    dependencies=[Depends(require_creator_path_access)],
)
async def sync_vault_start(creator_id: str, force: bool = False) -> dict:
    if creator_id in _vault_sync_state and _vault_sync_state[creator_id].get("status") == "running":
        return {"status": "already_running"}
    # Cost guard: vault sync is expensive — once per 7 days unless forced.
    if not force:
        cd = await _vault_cooldown_remaining(creator_id, "last_vault_sync_at")
        if not cd["allowed"]:
            return {"status": "cooldown", **cd}
    _vault_sync_state[creator_id] = {"status": "running", "synced": 0, "total": 0, "album": ""}
    await _stamp_vault_op(creator_id, "last_vault_sync_at")
    spawn(_run_vault_sync(creator_id), name="run_vault_sync")
    return {"status": "started"}


@app.get(
    "/sync-vault-status/{creator_id}",
    dependencies=[Depends(require_creator_path_access)],
)
async def sync_vault_status(creator_id: str) -> dict:
    state = _vault_sync_state.get(creator_id, {"status": "idle", "synced": 0, "total": 0, "album": ""})
    return state


async def _run_vault_sync(creator_id: str) -> None:
    import httpx

    db = get_supabase()
    try:
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
        existing_rows = await asyncio.to_thread(
            lambda: db.table("creator_vault_media")
            .select("media_id")
            .eq("creator_id", creator_id)
            .execute()
        )
        existing_ids = {r["media_id"] for r in (existing_rows.data or [])}

        async with httpx.AsyncClient() as client:
            albums = await apifansly_list_vault_albums(
                str(apifansly_id),
                client=client,
            )

            total = sum(a.get("itemCount", 0) for a in albums)
            already = len(existing_ids)
            new_total = max(total - already, 0)
            synced = 0
            new_item_ids: list[str] = []

            _vault_sync_state[creator_id] = {"status": "running", "synced": 0, "total": 0, "album": "Starting..."}

            for album in albums:
                album_id = album.get("id")
                album_title = album.get("title") or f"Album_{album_id}"
                cursor = None
                consecutive_dupe_batches = 0

                while True:
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
                    consecutive_dupe_batches = consecutive_dupe_batches + 1 if all_dupes else 0

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

                    _vault_sync_state[creator_id] = {"status": "running", "synced": synced, "total": new_total, "album": album_title}
                    print(f"[VAULT SYNC] album={album_title} synced={synced}/{new_total} cursor={cursor}")

                    if not cursor or consecutive_dupe_batches >= 3:
                        break

        new_item_ids = normalize_media_ids(new_item_ids)
        categorized_new = 0
        category_errors = 0
        if categorize_new_batch_enabled(auto_categorize_new, new_item_ids):
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

    except Exception as e:
        import traceback
        print(f"[VAULT SYNC ERROR] {e}")
        print(traceback.format_exc())
        _vault_sync_state[creator_id] = {"status": "error", "synced": 0, "total": 0, "album": str(e)}


@app.post(
    "/upload-vault-media/{creator_id}",
    dependencies=[Depends(require_creator_path_access)],
)
async def upload_vault_media(creator_id: str, request: Request) -> dict:
    import httpx

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

    async with httpx.AsyncClient() as client:
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

CATEGORY_LIST = "\n".join([
    f"- {k}: {v['label']} (price range ${v['min']}-${v['max']})"
    for k, v in VAULT_CATEGORIES.items()
])


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


async def _load_vault_visual(item: dict, *, is_video: bool) -> tuple[bytes, str, str]:
    """Return image bytes, evidence source, and retrieval method."""
    import httpx

    source = "video_thumbnail" if is_video else "image"
    visual_url = str(
        (item.get("thumbnail_url") if is_video else item.get("url")) or ""
    )
    first_error: Exception | None = None

    if visual_url:
        async with httpx.AsyncClient(follow_redirects=True) as client:
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
            async with httpx.AsyncClient(follow_redirects=True) as client:
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


async def _categorize_single_item(item: dict) -> dict:
    """Classify one vault item into the versioned provider-neutral contract.

    Images are resized before upload to control vision-token cost.  Videos use
    their real platform thumbnail rather than guessing from a filename.  A
    provider/fetch/parse failure is raised so the retry loop can leave the item
    stale instead of permanently saving an empty ``other`` classification.
    """
    from anthropic import AsyncAnthropic
    import base64
    import io
    from PIL import Image

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.environ.get("VAULT_CLASSIFIER_MODEL", "claude-sonnet-4-6")
    mimetype = str(item.get("mimetype") or "")
    item_id = item.get("id", "")
    is_video = mimetype.startswith("video") if mimetype else False

    try:
        visual_bytes, source, fetch_method = await _load_vault_visual(
            item,
            is_video=is_video,
        )

        # Classification does not need original-resolution media.  Normalizing
        # every asset to a compact JPEG makes cost predictable and also handles
        # thumbnails whose declared MIME type is missing or inaccurate.
        image = Image.open(io.BytesIO(visual_bytes))
        image.seek(0)
        image = image.convert("RGB")
        image.thumbnail((896, 896), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=84, optimize=True)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        asset_kind = "video thumbnail" if is_video else "image"
        prompt = f"""Classify this adult creator {asset_kind} for private vault search and package matching.
The creator and all depicted participants must be adults. Describe only what is visibly supported; never invent an act, location, outfit, or prop. A video thumbnail is partial evidence, so describe the visible frame and lower confidence when the complete video cannot be inferred.

Filename: {item.get('filename') or 'unknown'}
Album/folder: {item.get('album_title') or 'unknown'}
Allowed category keys:
{CATEGORY_LIST}

EXPLICITNESS is strictly what is visible:
- 0 = ordinary SFW clothing/selfie
- 1 = censored, blurred, implied, or fully covered teaser
- 2 = suggestive/flirty but clothed
- 3 = lingerie, bikini, or see-through with no explicit nudity
- 4 = exposed breasts, butt, or genitals without an explicit sex act
- 5 = explicit sexual activity, toy use, spread pose, oral sex, or penetration
Lingerie alone is never above 3. Do not infer explicitness from the album name.
This is neutral inventory metadata, not erotic writing. Do not omit or euphemize
visible nudity, exposed anatomy, or sexual activity. Record them factually using
the controlled fields below, while never guessing details that are not visible.

Return ONLY one JSON object with exactly these keys:
{{"category":"category_key","description":"2-4 factual, non-erotic sentences covering the visible subject, clothing/nudity, pose/action, environment and distinguishing details","mood":"playful|intimate|teasing|explicit|casual","explicitness":0,"nudity":"none|implied|partial|full","visible_anatomy":["only visibly exposed: breasts|buttocks|vulva|penis|anus"],"good_for":"opener|mid_session|closer|standalone","tags":["specific searchable themes"],"sexual_activity":["only visibly supported activities"],"body_focus":["visible focal areas"],"action":"specific visible action or none","pose":"specific pose","framing":"selfie|portrait|full body|close-up|wide|other","props":["visible props"],"colors":["dominant outfit/scene colors"],"scene_location":"specific location or unknown","scene_outfit":"specific outfit/nudity state or unknown","scene_lighting":"natural|bright|dim|flash|colored|unknown","scene_id":"short stable shoot slug derived from album/location/outfit","confidence":0.0}}
"""
        response = await client.messages.create(
            model=model,
            max_tokens=850,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )

        content = "\n".join(
            str(block.text)
            for block in response.content
            if getattr(block, "type", "") == "text"
            and getattr(block, "text", None)
        ).strip()
        if not content:
            raise ValueError(
                f"classifier returned no text (stop_reason={response.stop_reason})"
            )
        content = content.replace("```json", "").replace("```", "").strip()
        print(f"[CATEGORIZE RAW] item={item_id} response={content[:300]}")
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("classifier did not return a JSON object")
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
                "description", "mood", "sexual_activity", "body_focus", "action",
                "pose", "framing", "props", "colors", "nudity",
                "visible_anatomy",
            )
        }
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
        })

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


_VAULT_COOLDOWN_DAYS = 7


async def _vault_cooldown_remaining(creator_id: str, column: str) -> dict:
    """Returns cooldown info for an expensive vault op.
    {allowed: bool, last_at: str|None, days_remaining: float, next_allowed_at: str|None}.
    column is 'last_vault_sync_at' or 'last_categorize_at'."""
    from datetime import datetime, timezone, timedelta
    db = get_supabase()
    row = await asyncio.to_thread(
        lambda: db.table("creators").select(column).eq("id", creator_id).single().execute()
    )
    last_str = (row.data or {}).get(column)
    if not last_str:
        return {"allowed": True, "last_at": None, "days_remaining": 0, "next_allowed_at": None}
    try:
        last = datetime.fromisoformat(last_str.replace("Z", "+00:00"))
    except Exception:
        return {"allowed": True, "last_at": last_str, "days_remaining": 0, "next_allowed_at": None}
    now = datetime.now(timezone.utc)
    next_allowed = last + timedelta(days=_VAULT_COOLDOWN_DAYS)
    if now >= next_allowed:
        return {"allowed": True, "last_at": last_str, "days_remaining": 0, "next_allowed_at": None}
    remaining = (next_allowed - now).total_seconds() / 86400
    return {
        "allowed": False, "last_at": last_str,
        "days_remaining": round(remaining, 1),
        "next_allowed_at": next_allowed.isoformat(),
    }


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


_categorize_state: dict = {}


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
    media.  A classifier-version upgrade is a separate, confirmed operation and
    can only select rows older than the current metadata contract.
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
        if upgrade_scope not in {"approved", "all"}:
            raise HTTPException(
                status_code=400,
                detail="upgrade_scope must be 'approved' or 'all'",
            )
        if upgrade_scope == "approved":
            upgrade_item_ids = await _stale_approved_set_media_ids(creator_id)
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
        "status": "running",
        "mode": resolved_mode,
        "done": 0,
        "total": pending,
        "errors": 0,
    }
    await _stamp_vault_op(creator_id, "last_categorize_at")
    spawn(
        _run_vault_categorization(
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
        .select("vault_initial_categorized_at, auto_categorize_new_media")
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
        "uncategorized": await _count_uncategorized(creator_id),
        "stale_classifications": await _count_stale_classifications(creator_id),
        "stale_approved_classifications": len(stale_approved),
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
                        .lt("classification_version", VAULT_CLASSIFIER_VERSION)
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

        done = 0
        errors = 0
        # 2 concurrent max to respect 30k token/min rate limit
        batch_size = 2
        for i in range(0, total, batch_size):
            batch = all_items[i:i + batch_size]
            results = await asyncio.gather(
                *[_categorize_single_item_with_retry(item) for item in batch],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                    continue
                await asyncio.to_thread(
                    lambda r=result: db.table("creator_vault_media")
                    .update(_classification_update_payload(r))
                    .eq("id", r["id"])
                    .execute()
                )
                done += 1
            _categorize_state[creator_id].update({"done": done, "errors": errors})
            print(f"[CATEGORIZE] done={done}/{total} errors={errors}")
            await asyncio.sleep(1.5)  # respect rate limit between batches

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
        print(f"[CATEGORIZE] complete done={done} errors={errors}")

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
        "scene_location, scene_outfit, scene_lighting, mimetype, tags"
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


async def _categorize_single_item_with_retry(item: dict, max_retries: int = 3) -> dict:
    """Wrap _categorize_single_item with exponential backoff on 429."""
    for attempt in range(max_retries):
        try:
            return await _categorize_single_item(item)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"[CATEGORIZE] rate limited, waiting {wait}s before retry")
                await asyncio.sleep(wait)
            else:
                raise
    return await _categorize_single_item(item)


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
        result = await _categorize_single_item(item)
    except VaultVisualAccessError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
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
    dependencies=[Depends(require_account_path_access)],
)
async def get_media_url(account_id: str, content_id: str) -> dict:
    import httpx

    async with httpx.AsyncClient() as client:
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
    if len(_processed_messages) > 1000:
        _processed_messages.clear()

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
async def fansly_webhook(payload: dict) -> dict:
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
        price_cents = int(
            ((data.get("orderMetadata") or {}).get("accountMediaPrice"))
            or 0
        )
        if not creator_id or not platform_fan_id:
            return {"status": "invalid_ppv_purchase_event"}

        fan_result = await asyncio.to_thread(
            lambda: db.table("fans")
            .select("id, pending_ppv_check")
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
        pending = fan_row.get("pending_ppv_check") or {}
        expected_cents = int(
            pending.get("price_cents")
            or round(float(pending.get("price") or 0) * 100)
        )
        if expected_cents and price_cents:
            delta = abs(expected_cents - price_cents)
            if delta > max(100, int(expected_cents * 0.1)):
                from db.queries import freeze_fan_for_review

                await freeze_fan_for_review(
                    str(fan_row["id"]),
                    "ppv_webhook_price_mismatch",
                )
                print(
                    f"[PPV WEBHOOK] price mismatch fan={fan_row['id']} "
                    f"expected={expected_cents} actual={price_cents}"
                )
                return {"status": "review_required"}

        purchase_media_id = str(
            pending.get("media_id")
            or account_media_id
        )
        if not purchase_media_id:
            return {"status": "invalid_ppv_purchase_event"}

        from services.suggestions import record_ppv_purchase

        await record_ppv_purchase(
            str(fan_row["id"]),
            purchase_media_id,
            (price_cents / 100.0) if price_cents else None,
        )
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
    creator_platform_id = None
    if interactions:
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

    fan = await get_fan(creator_id, platform_fan_id)
    if not fan:
        fan = await create_fan(creator_id, platform_fan_id, f"Fan_{platform_fan_id[-6:]}")
        spawn(_enrich_fan_profile(fan.id, creator_id, platform_fan_id), name="enrich_fan_profile")

    if message_id:
        mid = str(message_id)
        existing = await asyncio.to_thread(
            lambda: db.table("messages")
            .select("id")
            .eq("fansly_message_id", mid)
            .limit(1)
            .execute()
        )
        if existing.data:
            return {"status": "duplicate"}

    if group_id:
        await asyncio.to_thread(
            lambda: db.table("fans")
            .update({"fansly_group_id": str(group_id)})
            .eq("id", fan.id)
            .execute()
        )

    mid = str(message_id) if message_id else ""

    media_context = (
        {"attachments": [{"contentId": a.get("contentId")} for a in attachments_raw]}
        if attachments_raw
        else None
    )

    if mid:
        await save_message(
            fan.id,
            creator_id,
            "fan",
            message_content,
            fansly_message_id=mid,
            media_context=media_context,
        )
    else:
        await save_message(
            fan.id,
            creator_id,
            "fan",
            message_content,
            media_context=media_context,
        )

    await process_incoming_fan_message(
        fan.id,
        creator_id,
        message_content,
        auto_mode,
        mid if mid else None,
    )

    return {"status": "ok"}


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

    fans = await asyncio.to_thread(
        lambda cid=creator_id: db.table("fans").select("id").eq("creator_id", cid).execute()
    )
    fan_ids = [f["id"] for f in (fans.data or [])]

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
    links = await asyncio.to_thread(
        lambda: db.table("chatter_creators")
        .select("creator_id")
        .eq("chatter_id", operator_id)
        .execute()
    )
    creator_ids = [r["creator_id"] for r in (links.data or [])]
    if not creator_ids:
        return {"creators": []}

    creators = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("id, platform_username, fansly_account_id, apifansly_account_id, persona, auto_mode")
        .in_("id", creator_ids)
        .execute()
    )
    return {"creators": creators.data or []}


@app.get(
    "/creator/{creator_id}/auto-availability",
    dependencies=[Depends(require_creator_path_access)],
)
async def auto_availability(creator_id: str) -> dict:
    """Auto-mode is only available when at least one approved set exists.
    Single source of truth for the dashboard's auto gate + a backend guard."""
    db = get_supabase()
    r = await asyncio.to_thread(
        lambda: db.table("vault_sets")
        .select("id", count="exact")
        .eq("creator_id", creator_id)
        .eq("status", "approved")
        .limit(1)
        .execute()
    )
    count = r.count or 0
    return {"auto_available": count > 0, "approved_sets": count}


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
    "/creator/{creator_id}/auto-audience-preview",
    dependencies=[Depends(require_creator_path_access)],
)
async def preview_auto_audience(creator_id: str) -> dict:
    from collections import Counter
    from services.auto_audience import AutoAudiencePolicy, evaluate_auto_eligibility

    db = get_supabase()
    creator_result, fan_result, lists_result, messages_result = await asyncio.gather(
        asyncio.to_thread(
            lambda: db.table("creators")
            .select("auto_mode, auto_audience_policy")
            .eq("id", creator_id)
            .single()
            .execute()
        ),
        asyncio.to_thread(
            lambda: db.table("fans")
            .select("id, auto_mode, total_spent, spend_tier, needs_human_review")
            .eq("creator_id", creator_id)
            .execute()
        ),
        asyncio.to_thread(
            lambda: db.table("fan_list_members")
            .select("fan_id, list_id, fan_lists(exclude_from_auto, creator_id)")
            .execute()
        ),
        asyncio.to_thread(
            lambda: db.table("messages")
            .select("fan_id")
            .eq("creator_id", creator_id)
            .eq("role", "creator")
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
        str(row.get("fan_id")) for row in (messages_result.data or []) if row.get("fan_id")
    }
    memberships: dict[str, set[str]] = {}
    legacy_exclusions: set[str] = set()
    for row in (lists_result.data or []):
        joined = row.get("fan_lists") or {}
        if str(joined.get("creator_id") or "") != str(creator_id):
            continue
        fan_key = str(row.get("fan_id") or "")
        list_key = str(row.get("list_id") or "")
        if fan_key and list_key:
            memberships.setdefault(fan_key, set()).add(list_key)
            if joined.get("exclude_from_auto"):
                legacy_exclusions.add(list_key)
    if legacy_exclusions:
        policy.exclude_list_ids = list(
            dict.fromkeys([*policy.exclude_list_ids, *legacy_exclusions])
        )

    reasons: Counter[str] = Counter()
    reasons_if_creator_on: Counter[str] = Counter()
    eligible = 0
    eligible_if_creator_on = 0
    for fan in (fan_result.data or []):
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
    total = len(fan_result.data or [])
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
        .select("creator_id, sales_log, pending_ppv_check")
        .eq("id", fan_id)
        .single()
        .execute()
    )
    fan = fan_row.data or {}
    if str(fan.get("creator_id") or "") != str(creator_id):
        raise HTTPException(status_code=404, detail="fan not found for creator")

    try:
        vault_rows, set_rows, message_rows = await asyncio.gather(
            asyncio.to_thread(
                lambda: db.table("creator_vault_media")
                .select(
                    "id, fansly_media_id, media_id, url, thumbnail_url, mimetype, filename, "
                    "album_title, content_category, ai_description, price_min, price_max, is_active"
                )
                .eq("creator_id", creator_id)
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
                .order("sent_at", desc=True)
                .limit(1000)
                .execute()
            ),
        )
    except Exception as exc:
        print(f"[OPERATOR PPV OPTIONS] fan={fan_id} creator={creator_id} error={exc}", flush=True)
        raise HTTPException(
            status_code=502,
            detail="Could not load the creator vault for this PPV.",
        ) from exc

    sent_ids: set[str] = set()
    for message in (message_rows.data or []):
        ppv = (message.get("media_context") or {}).get("ppv") or {}
        sent_ids.update(normalize_media_ids(ppv.get("media_ids") or [ppv.get("media_id")]))
    purchased_ids: set[str] = set()
    for sale in (fan.get("sales_log") or []):
        purchased_ids.update(
            normalize_media_ids(sale.get("media_ids") or [sale.get("media_id")])
        )
    pending_ids = set(
        normalize_media_ids(
            (fan.get("pending_ppv_check") or {}).get("media_ids")
            or [(fan.get("pending_ppv_check") or {}).get("media_id")]
        )
    )

    media = []
    for row in (vault_rows.data or []):
        external_id = str(row.get("fansly_media_id") or row.get("media_id") or "")
        status = (
            "sold" if external_id in purchased_ids
            else "payment_pending" if external_id in pending_ids
            else "sent" if external_id in sent_ids
            else "unused"
        )
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
    dependencies=[Depends(require_creator_fan_access)],
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


@app.post(
    "/test/simulate-ppv-purchase",
    dependencies=[Depends(require_fan_path_access)],
)
async def simulate_ppv_purchase(fan_id: str, request: Request) -> dict:
    """Dev only — simulate a fan purchasing a pending PPV."""
    from db.queries import get_fan_session, save_fan_session
    from datetime import datetime

    db = get_supabase()

    # Get pending PPV check
    fan_row = await asyncio.to_thread(
        lambda: db.table("fans")
        .select("pending_ppv_check, total_spent, active_session, ai_summary")
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

    from datetime import datetime
    # Get existing sales_log
    sales_log = fan_data.get("sales_log") or []
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
    dependencies=[Depends(require_creator_fan_access)],
)
async def test_inject_message(fan_id: str, creator_id: str, content: str) -> dict:
    """Dev testing only — simulate a fan message without Fansly webhook."""
    from db.queries import save_message
    await save_message(fan_id, creator_id, "fan", content, was_ai_suggested=False)
    await process_incoming_fan_message(fan_id, creator_id, content, auto_mode=True, message_id=None)
    return {"status": "ok", "fan_id": fan_id, "content": content}


@app.post(
    "/generate-sets/{creator_id}",
    dependencies=[Depends(require_creator_path_access)],
)
async def generate_sets(creator_id: str) -> dict:
    from db.queries import propose_sets
    db = get_supabase()

    items, page = [], 0
    while True:
        rows = await asyncio.to_thread(
            lambda p=page: db.table("creator_vault_media")
            .select(
                "fansly_media_id, content_category, ai_description, explicitness_level, "
                "scene_id, scene_location, scene_outfit, scene_lighting, album_title, "
                "mimetype, price_min, price_max, tags, good_for, classification_metadata"
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

    proposed = propose_sets(items)

    # Wipe prior AI drafts; never touch approved or manual sets
    await asyncio.to_thread(
        lambda: db.table("vault_sets").delete()
        .eq("creator_id", creator_id).eq("status", "draft").eq("source", "ai").execute()
    )

    to_insert = [{
        "creator_id": creator_id, "description": s["description"],
        "title": s["title"], "location": s["location"], "outfit": s["outfit"],
        "explicit_min": s["explicit_min"], "explicit_max": s["explicit_max"],
        "media_ids": s["media_ids"], "preview_media_id": s["preview_media_id"],
        "suggested_price": s["suggested_price"], "tags": s["tags"],
        "metadata_version": s["metadata_version"],
        "status": "draft", "source": "ai",
    } for s in proposed]

    inserted = 0
    for i in range(0, len(to_insert), 100):
        chunk = to_insert[i:i + 100]
        await asyncio.to_thread(lambda c=chunk: db.table("vault_sets").insert(c).execute())
        inserted += len(chunk)

    return {"status": "ok", "drafts_created": inserted, "from_items": len(items)}


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


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


from routes.fansly import fansly_router

app.include_router(fansly_router)
