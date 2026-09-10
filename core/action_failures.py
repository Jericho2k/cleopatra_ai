"""Why a scheduled action failed, and therefore whether retrying it is sane.

Audit reference: REL-005.

Sprint 1 removed the writer's 3x generation retry amplification. The second
layer remained: the durable action itself retried EVERYTHING eight times with
backoff, regardless of why it failed. An AUTO_REPLY that could never send —
the creator's API Fansly account is disconnected, so there is no delivery route
at all — still ran the analyzer and the writer eight times before giving up.
That is eight full model pipelines, at cost, to reach a conclusion that was
knowable before the first one.

Four classes, and the retry budget follows from which one applies:

TRANSIENT
    Model 429/5xx, API Fansly 503, a database blip, a binding that has not
    resolved yet. The world is expected to change on its own. Bounded retry
    with backoff — the existing behaviour, and still the default for anything
    unrecognised, because guessing "permanent" wrongly drops a real message.

PERMANENT
    The creator is not connected, there is no account binding after
    authoritative resolution, configuration is missing. Retrying runs the same
    expensive pipeline to reach the same answer. Fail once, immediately, with a
    reason an operator can act on. Raised as ``PermanentActionFailure``.

OBSOLETE
    The fan replied, a human sent, the creator turned Auto off, the triggering
    state was replaced. There is nothing to retry because there is nothing left
    to do. Already handled upstream by the revalidation gate in
    ``_should_still_send``, which completes the action rather than failing it.

WRITER QUALITY
    The bounded generation policy produced no usable candidate. Sometimes worth
    one more attempt, never worth eight: the input has not changed, so the
    ninth attempt is as likely to fail as the second. Raised as
    ``WriterQualityFailure`` and given a small budget of its own.

Nothing here weakens fail-closed behaviour. A permanent failure still sends no
message; it just stops paying for the discovery repeatedly.
"""

from __future__ import annotations

# Written into scheduled_actions.last_error ahead of the human-readable text, so
# the operator health surface can separate "the provider is having a bad ten
# minutes" from "twenty fans cannot send because a binding is broken" without
# parsing prose. Kept short and stable — it is effectively a wire format.
TERMINAL_PREFIX = "TERMINAL"


class ActionFailure(Exception):
    """Base for failures that carry a classification.

    ``code`` is a stable machine-readable slug (``creator_not_connected``), not
    a sentence. It is what the health surface groups by and what an operator
    ends up searching for.
    """

    code: str = "unknown"

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        super().__init__(message or code)

    @property
    def detail(self) -> str:
        return str(self)


class PermanentActionFailure(ActionFailure):
    """This action cannot succeed until a human changes something.

    Raise it only where the conclusion is authoritative. A missing value that
    a later resolution step might still supply is TRANSIENT, not this: the cost
    of guessing wrong here is a silently dropped message, which is worse than
    the retries this avoids.
    """

    def marker(self) -> str:
        return f"{TERMINAL_PREFIX}:{self.code}: {self.detail}"


class WriterQualityFailure(ActionFailure):
    """Generation produced nothing usable under the current bounded policy."""


def is_terminal_error(last_error: str | None) -> bool:
    return bool(last_error) and str(last_error).startswith(f"{TERMINAL_PREFIX}:")


def terminal_code(last_error: str | None) -> str:
    """Pull the slug back out of a stored last_error. '' when not terminal."""
    if not is_terminal_error(last_error):
        return ""
    remainder = str(last_error)[len(TERMINAL_PREFIX) + 1 :]
    return remainder.split(":", 1)[0].strip()
