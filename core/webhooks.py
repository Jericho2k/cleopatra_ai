"""Helpers for authenticating external webhook deliveries."""

from __future__ import annotations

import hashlib
import hmac


def valid_hmac_sha256_signature(
    body: bytes,
    supplied_signature: str | None,
    signing_secret: str,
) -> bool:
    """Validate a provider HMAC-SHA256 signature over the raw request body.

    API providers commonly send either the bare hexadecimal digest or prefix it
    with ``sha256=``. Supporting both forms keeps the comparison strict without
    weakening authentication.
    """

    supplied = str(supplied_signature or "").strip()
    if not supplied or not signing_secret:
        return False
    if supplied.lower().startswith("sha256="):
        supplied = supplied.split("=", 1)[1].strip()

    expected = hmac.new(
        signing_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(supplied.lower(), expected.lower())
