import asyncio
import hashlib
import json
import random
import time
import httpx

# ── Fill these in ──────────────────────────────────────────────────────────
TEST_EMAIL = "your_test_email@example.com"
TEST_PASSWORD = "your_test_password"
TEST_PROXY = None  # "http://user:pass@host:port" or None to test without proxy first
# ───────────────────────────────────────────────────────────────────────────

BASE_URL = "https://apiv3.fansly.com/api/v1"


def generate_client_id() -> str:
    """
    Fansly client-id is a snowflake-style ID.
    We generate a fresh one — this becomes the 'device fingerprint'.
    """
    ts = int(time.time() * 1000)
    rand = random.randint(0, 0xFFFFF)
    return str((ts << 20) | rand)


def fake_client_check(client_id: str) -> str:
    """
    We don't know the real algorithm yet.
    Testing with an MD5 truncation first — if server rejects, we'll know.
    """
    return hashlib.md5(client_id.encode()).hexdigest()[:13]


async def attempt_login(
    email: str,
    password: str,
    client_id: str,
    client_check: str,
    proxy: str = None,
) -> dict:
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/json",
        "Origin": "https://fansly.com",
        "Referer": "https://fansly.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "fansly-client-id": client_id,
        "fansly-client-ts": str(int(time.time() * 1000)),
        "fansly-client-check": client_check,
        "Cookie": f"fansly-d={client_id}; f-d={client_id}",
    }

    payload = {
        "email": email,
        "password": password,
        "deviceId": client_id,
    }

    async with httpx.AsyncClient(proxy=proxy, timeout=30.0) as client:
        resp = await client.post(
            f"{BASE_URL}/account/login?ngsw-bypass=true",
            headers=headers,
            json=payload,
        )
        return resp.status_code, resp.headers, resp.text


async def verify_session(
    auth_token: str,
    client_id: str,
    client_check: str,
    session_cookie: str,
    proxy: str = None,
) -> dict:
    """After login, verify the session works on /account/me"""
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://fansly.com",
        "Referer": "https://fansly.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Authorization": auth_token,
        "fansly-client-id": client_id,
        "fansly-client-ts": str(int(time.time() * 1000)),
        "fansly-client-check": client_check,
        "Cookie": f"fansly-d={client_id}; f-d={client_id}; f-s-c={session_cookie}",
    }

    async with httpx.AsyncClient(proxy=proxy, timeout=30.0) as client:
        resp = await client.get(
            f"{BASE_URL}/account/me?ngsw-bypass=true",
            headers=headers,
        )
        return resp.status_code, resp.text


async def main():
    client_id = generate_client_id()
    print(f"Generated client_id: {client_id}")

    # ── Test 1: Fake client-check ──────────────────────────────────────────
    print("\n── Test 1: Login with FAKE client-check ──────────────────────")
    fake_check = fake_client_check(client_id)
    print(f"Using fake client-check: {fake_check}")

    status, resp_headers, body = await attempt_login(
        TEST_EMAIL, TEST_PASSWORD, client_id, fake_check, TEST_PROXY
    )
    print(f"Status: {status}")

    if status == 200:
        data = json.loads(body)
        if data.get("success"):
            print("✅ LOGIN SUCCEEDED with fake client-check!")
            print("   → client-check is NOT validated server-side")
            print("   → We do NOT need Playwright")

            # Extract session values
            resp_data = data["response"]
            token = resp_data.get("token")
            session_id = resp_data.get("sessionId") or resp_data.get("id")

            # Build Authorization header value (base64 of "sessionId:1:2:token")
            import base64
            auth_raw = f"{session_id}:1:2:{token}"
            auth_token = base64.b64encode(auth_raw.encode()).decode()

            # Extract session cookie from Set-Cookie
            session_cookie = ""
            set_cookie = resp_headers.get("set-cookie", "")
            if "f-s-c=" in set_cookie:
                start = set_cookie.index("f-s-c=") + 6
                end = set_cookie.index(";", start) if ";" in set_cookie[start:] else len(set_cookie)
                session_cookie = set_cookie[start:end]

            print(f"\n── Session values (store these encrypted) ──")
            print(f"account_id:     {resp_data.get('accountId', 'see response')}")
            print(f"auth_token:     {auth_token}")
            print(f"client_id:      {client_id}")
            print(f"client_check:   {fake_check}  ← fake, works fine")
            print(f"session_cookie: {session_cookie}")
            print(f"\nFull response:\n{json.dumps(resp_data, indent=2)[:500]}")

            # ── Verify session works ───────────────────────────────────────
            print("\n── Verifying session with /account/me ────────────────")
            await asyncio.sleep(1)
            me_status, me_body = await verify_session(
                auth_token, client_id, fake_check, session_cookie, TEST_PROXY
            )
            print(f"Status: {me_status}")
            if me_status == 200:
                print("✅ Session verified! /account/me works.")
            else:
                print(f"⚠️  /account/me failed: {me_body[:300]}")

        else:
            print(f"❌ Login failed (success=false): {body[:300]}")

    elif status == 422 or status == 400:
        print(f"❌ Validation error — client-check MAY be validated: {body[:300]}")
        print("   → Try Test 2 below or consider Playwright")

    elif status == 401:
        print(f"❌ Wrong credentials or client-check rejected: {body[:300]}")
        print("   → Check email/password first")

    elif status == 403:
        print(f"❌ 403 Forbidden — likely client-check IS validated server-side")
        print(f"   Body: {body[:300]}")
        print("   → Need to reverse engineer the JS or use Playwright")

    else:
        print(f"❌ Unexpected status {status}: {body[:300]}")

    # ── Test 2: No client-check header at all ─────────────────────────────
    print("\n── Test 2: Login with NO client-check header ─────────────────")
    # Reuse same client_id, just omit the check header entirely
    status2, _, body2 = await attempt_login(
        TEST_EMAIL, TEST_PASSWORD, client_id, "", TEST_PROXY
    )
    print(f"Status: {status2}")
    if status2 == 200:
        print("✅ Works without client-check at all — definitely not validated!")
    else:
        print(f"Response: {body2[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
