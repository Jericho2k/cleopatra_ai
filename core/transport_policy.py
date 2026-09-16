"""Which transport serves each platform operation: the paid provider, or us.

The problem this exists to solve
--------------------------------
Every platform operation in this codebase currently goes through API Fansly,
which bills per call, per response kilobyte, and — expensively — per megabyte
of media moved. Some of those operations we can serve ourselves, directly
against Fansly, using the creator's own session (``services.fansly_client``).
Some we cannot, or cannot yet.

That is not one decision. It is one decision *per operation*, and the right
answer changes as each direct implementation is written and proven. A migration
that flipped everything at once would be a migration that could only be rolled
back by a deploy.

So the choice is data, not code shape:

* **per operation**, because "we serve our own media downloads" and "we send
  our own PPV messages" carry wildly different risk and are ready at wildly
  different times;
* **per account**, because the first creator to run on a new transport should
  be one chosen deliberately, not all of them;
* **runtime**, because turning a transport off when it misbehaves must not wait
  for a release.

Reading the policy
------------------
Precedence, most specific first:

1. ``FANSLY_TRANSPORT_<OPERATION>`` — e.g. ``FANSLY_TRANSPORT_MEDIA_DOWNLOAD``
2. ``FANSLY_TRANSPORT_DEFAULT``
3. ``provider`` — the paid path, which is what every deployment does today

and then, when the resolved answer is ``direct``, the account allowlist
``FANSLY_DIRECT_ACCOUNTS`` narrows it. An empty or unset allowlist means "every
account", so a deployment that wants a canary sets it, and a deployment that
has finished migrating clears it.

Fallback
--------
``FANSLY_DIRECT_FALLBACK`` (default on) decides what happens when the direct
transport fails: fall back to the paid provider and keep working, or surface
the failure. Leave it on while a direct implementation is young — a failed
direct call that silently costs a credit is a much better outcome than a fan
waiting on a reply that never comes. Turn it off once you want a direct
regression to be loud, and to be certain you are not quietly still paying.

This module holds no network code and knows nothing about either transport. It
answers one question — *who should serve this?* — so that the answer is
auditable in one place instead of inferred from twenty call sites.
"""
from __future__ import annotations

import os
from typing import Iterable


TRANSPORT_PROVIDER = "provider"
TRANSPORT_DIRECT = "direct"

VALID_TRANSPORTS = (TRANSPORT_PROVIDER, TRANSPORT_DIRECT)

# Stable operation names. These are the migration's unit of work: each one is
# a thing that can be moved off the provider on its own, and each one is a row
# in the transport report. They are strings rather than an enum because they
# also appear in environment variable names and in telemetry.
OP_MEDIA_DOWNLOAD = "media_download"
OP_CHAT_LISTING = "chat_listing"
OP_CHAT_MESSAGES = "chat_messages"
OP_MESSAGE_SEND = "message_send"
OP_MESSAGE_DELETE = "message_delete"
OP_VAULT_ALBUMS = "vault_albums"
OP_VAULT_ALBUM_MEDIA = "vault_album_media"
OP_ACCOUNT_LISTS = "account_lists"
OP_FOLLOWERS = "followers"
OP_SUBSCRIBERS = "subscribers"
OP_TOP_SUPPORTERS = "top_supporters"
OP_ACCOUNT_PROFILE = "account_profile"

OPERATIONS: tuple[str, ...] = (
    OP_MEDIA_DOWNLOAD,
    OP_CHAT_LISTING,
    OP_CHAT_MESSAGES,
    OP_MESSAGE_SEND,
    OP_MESSAGE_DELETE,
    OP_VAULT_ALBUMS,
    OP_VAULT_ALBUM_MEDIA,
    OP_ACCOUNT_LISTS,
    OP_FOLLOWERS,
    OP_SUBSCRIBERS,
    OP_TOP_SUPPORTERS,
    OP_ACCOUNT_PROFILE,
)

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}


def _env(name: str) -> str:
    return str(os.environ.get(name, "") or "").strip().lower()


def _normalize(value: str) -> str | None:
    """Map an environment value onto a transport, or None if it says nothing.

    Unrecognised text resolves to None rather than to ``direct``, so a typo in
    a deployment variable leaves the paid-but-working transport in place
    instead of silently switching a creator onto an unproven one.
    """
    if not value:
        return None
    if value in VALID_TRANSPORTS:
        return value
    # A boolean spelling is accepted because "turn direct on for sends" is how
    # an operator thinks about it, and guessing wrong here is cheap to avoid.
    if value in _TRUE_VALUES:
        return TRANSPORT_DIRECT
    if value in _FALSE_VALUES:
        return TRANSPORT_PROVIDER
    return None


def _env_var_name(operation: str) -> str:
    return f"FANSLY_TRANSPORT_{str(operation or '').strip().upper()}"


def direct_account_allowlist() -> tuple[str, ...]:
    """Accounts permitted on a direct transport, or empty meaning "all"."""
    raw = str(os.environ.get("FANSLY_DIRECT_ACCOUNTS", "") or "")
    return tuple(
        part.strip() for part in raw.replace(";", ",").split(",") if part.strip()
    )


def account_allowed(account_id: str | None) -> bool:
    """Whether this account may use a direct transport at all.

    An unknown account is refused whenever an allowlist is set: a canary that
    silently applied to calls with no account attached would not be a canary.
    """
    allowlist = direct_account_allowlist()
    if not allowlist:
        return True
    return str(account_id or "").strip() in allowlist


def configured_transport(operation: str) -> str:
    """The transport configured for this operation, before account narrowing."""
    specific = _normalize(_env(_env_var_name(operation)))
    if specific is not None:
        return specific
    default = _normalize(_env("FANSLY_TRANSPORT_DEFAULT"))
    if default is not None:
        return default
    return TRANSPORT_PROVIDER


def transport_for(operation: str, *, account_id: str | None = None) -> str:
    """Who should serve this call right now."""
    if configured_transport(operation) != TRANSPORT_DIRECT:
        return TRANSPORT_PROVIDER
    return TRANSPORT_DIRECT if account_allowed(account_id) else TRANSPORT_PROVIDER


def direct_enabled(operation: str, *, account_id: str | None = None) -> bool:
    return transport_for(operation, account_id=account_id) == TRANSPORT_DIRECT


def fallback_enabled() -> bool:
    """Whether a failed direct call may retry on the paid provider."""
    raw = _env("FANSLY_DIRECT_FALLBACK")
    if not raw:
        return True
    return raw not in _FALSE_VALUES


def direct_operations() -> tuple[str, ...]:
    """Every operation currently configured to run direct."""
    return tuple(
        operation
        for operation in OPERATIONS
        if configured_transport(operation) == TRANSPORT_DIRECT
    )


def snapshot(operations: Iterable[str] = OPERATIONS) -> dict[str, object]:
    """The resolved policy, for the operations dashboard and the boot log."""
    return {
        "transports": {
            operation: configured_transport(operation) for operation in operations
        },
        "direct_operations": list(direct_operations()),
        "account_allowlist": list(direct_account_allowlist()),
        "fallback_enabled": fallback_enabled(),
    }


def describe() -> str:
    """One line for the startup log, so the resolved policy is never a guess."""
    direct = direct_operations()
    if not direct:
        return "[TRANSPORT] every platform operation served by API Fansly"
    allowlist = direct_account_allowlist()
    scope = (
        f"{len(allowlist)} allowlisted account(s)" if allowlist else "all accounts"
    )
    fallback = "with provider fallback" if fallback_enabled() else "no fallback"
    return (
        f"[TRANSPORT] direct: {', '.join(direct)} for {scope} ({fallback}); "
        "everything else served by API Fansly"
    )
