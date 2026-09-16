#!/usr/bin/env python3
"""Prove the direct media transport works for one account before trusting it.

Why this exists
---------------
Two things in ``services/fansly_direct.py`` are observed from browser traffic
rather than published by Fansly, and neither can be confirmed from a test
suite:

1. whether a protected CDN asset will be served to an authenticated session
   read at all;
2. what the account-media endpoint is called and what shape its answer takes,
   which is what a signed-URL refresh depends on.

Guessing wrong is cheap in production — the transport fails and the provider
fallback pays for the transfer, exactly as designed — but it also means the
migration quietly saves nothing. This script answers both questions against one
real account, in a few seconds, before ``FANSLY_TRANSPORT_MEDIA_DOWNLOAD`` is
ever set to ``direct``.

It is read-only. It sends no messages and changes nothing on the platform.

Usage
-----
    python scripts/verify_fansly_direct.py --account <fansly_account_id> \\
        --url "https://cdn3.fansly.com/..." [--media-id <id>]

The URL is any protected asset location already stored in the vault; take one
from ``vault_media.url``. Credentials come from the encrypted session store, so
``SUPABASE_URL``, ``SUPABASE_SERVICE_KEY`` and ``FANSLY_SESSION_KEY`` must be
set exactly as they are in the deployment.

Read the output as a go/no-go:

* both routes OK      -> set the transport to ``direct`` for this account
* authenticated only  -> set it, and leave ``FANSLY_DIRECT_FALLBACK`` on; the
                         expired-signature case will still reach the provider
* neither             -> do NOT switch; the endpoint path in
                         ``FANSLY_DIRECT_MEDIA_PATH`` is the first thing to
                         re-check against browser traffic
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys


def _fail(message: str) -> None:
    print(f"  FAIL  {message}")


def _ok(message: str) -> None:
    print(f"  OK    {message}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", required=True, help="Fansly account id")
    parser.add_argument("--url", required=True, help="a protected CDN asset URL")
    parser.add_argument(
        "--media-id",
        default="",
        help="account-media id for the signed-URL refresh check",
    )
    args = parser.parse_args()

    from core.supabase import get_supabase
    from services import fansly_direct as direct
    from services.fansly_session_store import SessionStore

    if not direct.is_fansly_host(args.url):
        _fail(f"{args.url} is not an HTTPS Fansly host; nothing to verify")
        return 2

    store = SessionStore(
        supabase=get_supabase(),
        encryption_key=os.environ["FANSLY_SESSION_KEY"],
    )
    await store.load_all()
    direct.set_session_provider(store)

    print(f"\nAccount {args.account}")

    print("\n[1/3] session is alive")
    try:
        async with store.get_client(args.account) as client:
            me = await client.get_me()
        _ok(f"signed in as {me.get('account', {}).get('username', '?')}")
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {exc}")
        print("\nThe session is dead or absent. Re-connect the account first.")
        return 1

    print("\n[2/3] authenticated CDN read (the common case)")
    authenticated_ok = False
    try:
        async with store.get_client(args.account) as client:
            response = await client.download_asset(args.url, authenticated=True)
        size = len(response.content or b"")
        if response.status_code == 200 and size > direct.MIN_PLAUSIBLE_ASSET_BYTES:
            authenticated_ok = True
            _ok(
                f"{size} bytes, "
                f"~{direct.provider_credits_for_bytes(size):.1f} provider "
                "credits avoided per read of this asset"
            )
        else:
            _fail(f"HTTP {response.status_code}, {size} bytes")
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {exc}")

    print("\n[3/3] signed-URL refresh (the expired-signature case)")
    refresh_ok = False
    if not args.media_id:
        print("  SKIP  pass --media-id to check this route")
    else:
        path = os.environ.get("FANSLY_DIRECT_MEDIA_PATH") or "/account/media"
        try:
            async with store.get_client(args.account) as client:
                payload = await client.get_account_media([args.media_id])
            locations = direct._extract_media_locations(payload)
            if locations:
                refresh_ok = True
                _ok(f"{path} returned {len(locations)} usable location(s)")
            else:
                _fail(
                    f"{path} answered, but no Fansly location was found in it. "
                    "The response shape has changed; check the keys "
                    "fansly_direct._extract_media_locations looks for."
                )
        except Exception as exc:
            _fail(
                f"{path}: {type(exc).__name__}: {exc} — if this is a 404, set "
                "FANSLY_DIRECT_MEDIA_PATH to the path your browser actually "
                "calls."
            )

    print("\nVerdict")
    if authenticated_ok and refresh_ok:
        print(
            "  Both routes work. Safe to set\n"
            f"    FANSLY_TRANSPORT_MEDIA_DOWNLOAD=direct\n"
            f"    FANSLY_DIRECT_ACCOUNTS={args.account}\n"
            "  Leave FANSLY_DIRECT_FALLBACK unset while you watch it."
        )
        return 0
    if authenticated_ok:
        print(
            "  The common case works; expired signatures will still fall back\n"
            "  to the paid provider. Worth switching on, with fallback left on."
        )
        return 0
    print(
        "  Do not switch this account to direct yet. Re-capture the header\n"
        "  profile from browser traffic and re-run."
    )
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
