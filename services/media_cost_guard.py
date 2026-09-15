"""The hard ceiling on billed media transfer during vault classification.

The failure this exists to prevent
----------------------------------
API Fansly meters media transfer at 2 credits per megabyte. Classification used
to fall back, silently and automatically, to pulling an entire original video
through that billed proxy whenever direct frame extraction failed. A 250 MB
clip is ~500 credits. A background sync over a vault of them is a bill nobody
authorised and nobody saw coming, produced by a code path whose only log line
said "protected media download".

So the fallback is no longer silent and no longer automatic. It is a POLICY
decision, taken here, before a byte moves.

What "safe" means
-----------------
Refusing costs a partial classification. Allowing costs real money. Those are
not symmetric, so the guard is deliberately biased:

* an unknown size is a REFUSAL, not an optimistic attempt. "We could not
  determine the cost" and "the cost is acceptable" are different answers, and
  only one of them may spend credits automatically;
* the limits apply to the ESTIMATE, computed from the same 2 credits/MB rule
  the provider bills on, so an operator reading the limit in credits and the
  invoice in credits sees the same number;
* both a size limit and a credit limit apply, and the tighter one wins. They
  are two views of one quantity on purpose: an operator who thinks in
  megabytes and an operator who thinks in credits can each set the one they
  understand without having to convert.

Automatic versus manual
-----------------------
A background sync classifying a thousand items and an operator clicking
"re-analyze" on one are different risks, so they get different ceilings.
Manual is still bounded — an operator cannot authorise an unlimited download by
clicking — but bounded far more generously, because it is one asset, chosen
deliberately, already rate-limited to a few a day.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from services.apifansly import (
    CREDIT_BYTES_PER_MB,
    CREDIT_MEDIA_CREDITS_PER_MB,
)

# Automatic (background sync, new-media categorisation) ceilings.
#
# 25 MB is a deliberate "a few minutes of compressed video, or any photo"
# figure: it comfortably covers the assets where a download genuinely rescues a
# classification, and excludes the long originals where the credits are.
# At 2 credits/MB it is 50 credits, which is the matching default below.
DEFAULT_MAX_AUTO_MEDIA_DOWNLOAD_MB = 25.0
DEFAULT_MAX_AUTO_MEDIA_DOWNLOAD_CREDITS = 50.0

# Manual, operator-initiated deep analysis. One asset, chosen on purpose, and
# already capped at a few per day by the recategorisation limit.
DEFAULT_MAX_MANUAL_MEDIA_DOWNLOAD_MB = 250.0
DEFAULT_MAX_MANUAL_MEDIA_DOWNLOAD_CREDITS = 500.0

# Stable reason codes. Contract with the dashboard, the telemetry and the
# tests, so they are strings rather than prose.
REASON_ALLOWED = "within_limits"
REASON_UNKNOWN_SIZE = "unknown_size"
REASON_OVER_SIZE = "exceeds_size_limit"
REASON_OVER_CREDITS = "exceeds_credit_limit"
REASON_DISABLED = "download_disabled"


def _float_env(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    # Zero or negative means "never download automatically", which is a
    # legitimate and useful configuration, so it is preserved rather than
    # clamped back up to the default.
    return max(value, 0.0)


def estimated_credits_for_bytes(media_bytes: int | float | None) -> float:
    """What the provider would bill for moving this many bytes.

    The same rule ``services.apifansly.estimate_call_credits`` applies, kept in
    one place so a limit expressed in credits and the credits later recorded
    for the transfer cannot disagree.
    """
    try:
        size = float(media_bytes or 0)
    except (TypeError, ValueError):
        return 0.0
    if size <= 0:
        return 0.0
    return max(
        1.0,
        CREDIT_MEDIA_CREDITS_PER_MB * size / float(CREDIT_BYTES_PER_MB),
    )


@dataclass(frozen=True)
class MediaDownloadLimits:
    """One tier's ceiling, in both units."""

    max_megabytes: float
    max_credits: float
    manual: bool = False

    @property
    def max_bytes(self) -> float:
        return self.max_megabytes * float(CREDIT_BYTES_PER_MB)

    @property
    def enabled(self) -> bool:
        return self.max_megabytes > 0 and self.max_credits > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_megabytes": round(self.max_megabytes, 2),
            "max_credits": round(self.max_credits, 2),
            "manual": self.manual,
            "enabled": self.enabled,
        }


def auto_download_limits() -> MediaDownloadLimits:
    """The ceiling for background/automatic classification."""
    return MediaDownloadLimits(
        max_megabytes=_float_env(
            "VAULT_MAX_AUTO_MEDIA_DOWNLOAD_MB",
            DEFAULT_MAX_AUTO_MEDIA_DOWNLOAD_MB,
        ),
        max_credits=_float_env(
            "VAULT_MAX_AUTO_MEDIA_DOWNLOAD_CREDITS",
            DEFAULT_MAX_AUTO_MEDIA_DOWNLOAD_CREDITS,
        ),
        manual=False,
    )


def manual_download_limits() -> MediaDownloadLimits:
    """The ceiling for an operator-initiated single-asset re-analysis."""
    return MediaDownloadLimits(
        max_megabytes=_float_env(
            "VAULT_MAX_MANUAL_MEDIA_DOWNLOAD_MB",
            DEFAULT_MAX_MANUAL_MEDIA_DOWNLOAD_MB,
        ),
        max_credits=_float_env(
            "VAULT_MAX_MANUAL_MEDIA_DOWNLOAD_CREDITS",
            DEFAULT_MAX_MANUAL_MEDIA_DOWNLOAD_CREDITS,
        ),
        manual=True,
    )


def limits_for(*, manual: bool) -> MediaDownloadLimits:
    return manual_download_limits() if manual else auto_download_limits()


@dataclass(frozen=True)
class DownloadDecision:
    """Whether a billed media download may happen, and what it would cost."""

    allowed: bool
    reason: str
    estimated_bytes: int
    estimated_credits: float
    limits: MediaDownloadLimits

    @property
    def estimated_megabytes(self) -> float:
        return round(self.estimated_bytes / float(CREDIT_BYTES_PER_MB), 2)

    def operator_message(self) -> str:
        """One sentence an operator can act on. No CDN or auth jargon.

        A protected asset is infrastructure behaviour, not something the agency
        configured or can fix, so nothing here suggests they change a setting
        on the platform.
        """
        if self.allowed:
            return ""
        if self.reason == REASON_DISABLED:
            return (
                "Deep video analysis is switched off for automatic runs in this "
                "deployment, so only the thumbnail was classified."
            )
        if self.reason == REASON_UNKNOWN_SIZE:
            return (
                "Deep video scan skipped: the transfer size could not be "
                "determined in advance, so it was not attempted automatically."
            )
        return (
            "Deep video scan skipped to avoid a high media-transfer cost "
            f"(about {self.estimated_megabytes:.0f} MB, "
            f"~{self.estimated_credits:.0f} credits)."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "estimated_bytes": self.estimated_bytes,
            "estimated_megabytes": self.estimated_megabytes,
            "estimated_credits": round(self.estimated_credits, 2),
            "limits": self.limits.to_dict(),
        }


def evaluate_download(
    *,
    content_length_bytes: int | None,
    manual: bool = False,
    limits: MediaDownloadLimits | None = None,
    allow_unknown_size: bool = False,
) -> DownloadDecision:
    """Decide whether one billed media download may proceed.

    ``content_length_bytes`` is what the CDN said the asset weighs, obtained
    for free by a HEAD or a one-byte range request. ``None`` means it would not
    say — and for a VIDEO, a size we cannot read is a size we cannot afford to
    assume, because the difference between the guesses is two orders of
    magnitude of credits.

    ``allow_unknown_size`` exists for STILL IMAGES, and only for them. A photo
    that fails direct CDN access is the case the proxy was built for, its worst
    case is single-digit megabytes, and refusing it on an unreadable
    Content-Length would leave ordinary photos unclassified to guard against a
    cost that a photo cannot incur. A known size is still checked against the
    same ceilings, so a "photo" that turns out to weigh 300 MB is still
    refused.
    """
    resolved = limits if limits is not None else limits_for(manual=manual)

    if not resolved.enabled:
        return DownloadDecision(
            allowed=False,
            reason=REASON_DISABLED,
            estimated_bytes=int(content_length_bytes or 0),
            estimated_credits=estimated_credits_for_bytes(content_length_bytes),
            limits=resolved,
        )

    try:
        size = int(content_length_bytes) if content_length_bytes is not None else -1
    except (TypeError, ValueError):
        size = -1
    if size < 0:
        return DownloadDecision(
            allowed=bool(allow_unknown_size),
            reason=REASON_ALLOWED if allow_unknown_size else REASON_UNKNOWN_SIZE,
            estimated_bytes=0,
            estimated_credits=0.0,
            limits=resolved,
        )

    credits = estimated_credits_for_bytes(size)
    if size > resolved.max_bytes:
        return DownloadDecision(
            allowed=False,
            reason=REASON_OVER_SIZE,
            estimated_bytes=size,
            estimated_credits=credits,
            limits=resolved,
        )
    if credits > resolved.max_credits:
        return DownloadDecision(
            allowed=False,
            reason=REASON_OVER_CREDITS,
            estimated_bytes=size,
            estimated_credits=credits,
            limits=resolved,
        )
    return DownloadDecision(
        allowed=True,
        reason=REASON_ALLOWED,
        estimated_bytes=size,
        estimated_credits=credits,
        limits=resolved,
    )


def parse_content_length(headers: Any) -> int | None:
    """Read a transfer size from response headers, or None if unusable.

    Understands both ``Content-Length`` and the ``Content-Range`` form a
    one-byte range request answers with (``bytes 0-0/26214400``), because a
    signed CDN that rejects HEAD will usually still answer a range GET — and
    that is the cheapest honest way to learn what an asset weighs.
    """
    if headers is None:
        return None

    def _get(name: str) -> str:
        try:
            return str(headers.get(name) or "").strip()
        except Exception:
            return ""

    content_range = _get("content-range")
    if "/" in content_range:
        total = content_range.rsplit("/", 1)[-1].strip()
        if total.isdigit():
            return int(total)

    content_length = _get("content-length")
    if content_length.isdigit():
        value = int(content_length)
        # A one-byte range response reports a Content-Length of 1, which
        # describes the slice rather than the asset. Only trust it when no
        # range was involved.
        if not content_range:
            return value
    return None
