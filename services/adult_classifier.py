"""Adult-content classifier client.

A purpose-built moderation endpoint rates explicitness on this catalogue
without refusing, which a general-purpose vision model will not reliably do.
Only scores come from here — nothing in this module picks a category or a
price. Fusion with the vision model lives in ``services.vault_classification``.

Sightengine's ``nudity-2.1`` model is implemented. The client is selected by
``NSFW_CLASSIFIER_PROVIDER`` so a second vendor is a new ``_classify_*``
function plus a dispatch entry, not a change to any caller.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from services.vault_classification import ClassifierVerdict, verdict_from_nudity


SIGHTENGINE_BASE_URL = "https://api.sightengine.com"
SIGHTENGINE_NUDITY_MODEL = "nudity-2.1"


class AdultClassifierError(RuntimeError):
    """The classifier was configured but could not produce a verdict."""


@dataclass(frozen=True)
class ClassifierSettings:
    provider: str = "disabled"
    api_user: str = ""
    api_secret: str = ""
    base_url: str = SIGHTENGINE_BASE_URL
    timeout_seconds: float = 20.0

    @property
    def enabled(self) -> bool:
        return self.provider not in {"", "disabled", "off", "none"}

    @classmethod
    def from_env(cls) -> "ClassifierSettings":
        provider = os.getenv("NSFW_CLASSIFIER_PROVIDER", "disabled").strip().lower()
        return cls(
            provider=provider,
            api_user=os.getenv("SIGHTENGINE_API_USER", "").strip(),
            api_secret=os.getenv("SIGHTENGINE_API_SECRET", "").strip(),
            base_url=(
                os.getenv("NSFW_CLASSIFIER_BASE_URL", "").strip()
                or SIGHTENGINE_BASE_URL
            ).rstrip("/"),
            timeout_seconds=float(os.getenv("NSFW_CLASSIFIER_TIMEOUT", "20") or 20),
        )


async def classify_image(
    data: bytes,
    *,
    filename: str = "frame.jpg",
    media_type: str = "image/jpeg",
    settings: ClassifierSettings | None = None,
) -> ClassifierVerdict:
    """Score one image. Returns an unavailable verdict when not configured.

    A missing classifier degrades to vision-only classification rather than
    failing the item — a vault run must not stall because one vendor is down.
    """
    resolved = settings or ClassifierSettings.from_env()
    if not resolved.enabled or not data:
        return ClassifierVerdict()

    if resolved.provider == "sightengine":
        return await _classify_sightengine(
            data,
            filename=filename,
            media_type=media_type,
            settings=resolved,
        )

    raise AdultClassifierError(
        f"Unsupported NSFW_CLASSIFIER_PROVIDER {resolved.provider!r}. "
        "Supported: 'sightengine', 'disabled'."
    )


async def _classify_sightengine(
    data: bytes,
    *,
    filename: str,
    media_type: str,
    settings: ClassifierSettings,
) -> ClassifierVerdict:
    import httpx

    if not settings.api_user or not settings.api_secret:
        raise AdultClassifierError(
            "NSFW_CLASSIFIER_PROVIDER=sightengine requires SIGHTENGINE_API_USER "
            "and SIGHTENGINE_API_SECRET."
        )

    async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
        response = await client.post(
            f"{settings.base_url}/1.0/check.json",
            data={
                "models": SIGHTENGINE_NUDITY_MODEL,
                "api_user": settings.api_user,
                "api_secret": settings.api_secret,
            },
            files={"media": (filename, data, media_type)},
        )

    if response.status_code != 200:
        raise AdultClassifierError(
            f"Sightengine HTTP {response.status_code}: {response.text[:200]}"
        )

    payload: dict[str, Any] = response.json()
    if str(payload.get("status") or "").lower() != "success":
        error = payload.get("error") or {}
        raise AdultClassifierError(
            f"Sightengine error: {error.get('message') or payload}"
        )

    verdict = verdict_from_nudity(payload.get("nudity"))
    if not verdict.available:
        raise AdultClassifierError(
            "Sightengine returned no recognizable nudity classes; "
            "check that the nudity-2.1 model is enabled on the account."
        )
    return verdict
