import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from core.supabase import get_supabase
from core.tenancy import require_account_path_access, require_creator_access
from main import fansly_poller, session_store

fansly_router = APIRouter(prefix="/fansly", tags=["fansly"])


class ConnectAccountRequest(BaseModel):
    creator_id: str
    account_id: str
    username: str
    auth_token: str
    client_id: str
    client_check: str
    session_cookie: str
    proxy_url: str


@fansly_router.post("/connect")
async def connect_fansly_account(req: ConnectAccountRequest, request: Request):
    """
    Onboard a model's Fansly account into Cleopatra.

    Agency provides the 4 values from browser DevTools once.
    We store everything encrypted and start polling immediately.

    How the agency gets these values:
    1. Log into Fansly in Chrome
    2. Open DevTools → Network tab
    3. Reload the page
    4. Click any request to apiv3.fansly.com
    5. In Headers tab, copy:
       - Authorization header value
       - fansly-client-id header value
       - fansly-client-check header value
       - From the Cookie header, find f-s-c=XXXXX and copy just XXXXX
    """
    await require_creator_access(request, req.creator_id)

    from services.fansly_client import FanslyClient, SessionExpiredError

    client = FanslyClient(
        account_id=req.account_id,
        auth_token=req.auth_token,
        client_id=req.client_id,
        client_check=req.client_check,
        session_cookie=req.session_cookie,
        proxy_url=req.proxy_url,
    )
    try:
        async with client as c:
            me = await c.get_me()
    except SessionExpiredError:
        raise HTTPException(status_code=401, detail="Session already expired. Try again.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not connect: {str(e)}")

    await session_store.register(
        account_id=req.account_id,
        username=req.username,
        creator_id=req.creator_id,
        auth_token=req.auth_token,
        client_id=req.client_id,
        client_check=req.client_check,
        session_cookie=req.session_cookie,
        proxy_url=req.proxy_url,
    )
    await fansly_poller.start(req.account_id)

    return {
        "success": True,
        "account_id": req.account_id,
        "username": me["account"]["username"],
        "message": "Account connected and polling started.",
    }


class SendMessageRequest(BaseModel):
    content: str


@fansly_router.post(
    "/accounts/{account_id}/groups/{group_id}/messages",
    dependencies=[Depends(require_account_path_access)],
)
async def send_message(account_id: str, group_id: str, req: SendMessageRequest):
    """Send a message from a model account to a fan conversation."""
    try:
        async with session_store.get_client(account_id) as client:
            result = await client.send_message(group_id, req.content)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@fansly_router.get(
    "/accounts/{account_id}/groups/{group_id}/messages",
    dependencies=[Depends(require_account_path_access)],
)
async def get_messages(
    account_id: str,
    group_id: str,
    limit: int = 20,
    before_id: str = None,
):
    """Fetch message history for a conversation."""
    async with session_store.get_client(account_id) as client:
        data = await client.get_messages(group_id, limit=limit, before_id=before_id)
    return data


@fansly_router.get(
    "/accounts/{account_id}/groups",
    dependencies=[Depends(require_account_path_access)],
)
async def get_chat_groups(account_id: str):
    """Get all conversations for a model account."""
    async with session_store.get_client(account_id) as client:
        groups = await client.get_chat_groups(limit=50)
    return {"groups": groups}


@fansly_router.get("/health")
async def polling_health(creator_id: str, request: Request):
    """Return legacy poller health for one authorized creator only."""
    await require_creator_access(request, creator_id)
    result = await asyncio.to_thread(
        lambda: get_supabase()
        .table("creators")
        .select("fansly_account_id")
        .eq("id", creator_id)
        .single()
        .execute()
    )
    account_id = str((result.data or {}).get("fansly_account_id") or "")
    health = session_store.get_health() if session_store else {}
    return {
        "creator_id": creator_id,
        "connected": bool(account_id),
        "polling": bool(
            account_id
            and fansly_poller
            and account_id in getattr(fansly_poller, "_tasks", {})
        ),
        "session": health.get(account_id) if account_id else None,
    }
