"""What was actually running when a reply was produced.

``docs/autonomy_architecture_review.md`` §6 step 1 ends with an instruction the
rest of this repository could not previously satisfy: *confirm deployed SHA and
active flags*. Section 2 explains why it matters. The supplied failure excerpts
do not identify the deployed commit or the enabled flags, so no excerpt can be
used as evidence against a particular piece of code. A transcript read without
that pair is an argument about source that may never have been running.

Two values, recorded next to every reply this backend sends:

``sha``
    The commit the running container was built from. Read from the environment
    first, because a container built from a git archive has no ``.git`` to
    interrogate — Railway, Render and GitHub Actions each export it under a
    different name. The working tree is the fallback, for local runs and tests.

``flags``
    The behaviour-affecting environment variables, resolved as the process
    actually sees them. This is an explicit allowlist, not a dump of
    ``os.environ``: a provenance record is written into a message row and read
    back by operators, so a credential must not be able to arrive there by
    someone adding a variable. ``_assert_no_secrets`` enforces that at import,
    so widening the list wrongly fails the test suite rather than leaking.

The snapshot is deliberately reported two ways. ``flags_digest`` is eight hex
characters, small enough to store on every message; the full mapping is logged
once at startup and served by the build endpoint. Two replies with the same
digest ran under the same configuration, which is the question an investigation
actually asks.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.environment import app_env

#: Environment variables that name the deployed commit, most specific first.
#: Each platform exports its own; the explicit override comes first so a
#: deployment can always state the truth even where none of the others exist.
_SHA_ENV_VARS = (
    "CLEOPATRA_BUILD_SHA",
    "RAILWAY_GIT_COMMIT_SHA",
    "GIT_COMMIT_SHA",
    "SOURCE_VERSION",
    "RENDER_GIT_COMMIT",
    "GITHUB_SHA",
)

#: The flags that can change what a conversation does. Grouped the way an
#: investigation reads them: which brain answered, which controllers were on,
#: which memory was being written, and which platform boundary was live.
#:
#: Secrets are structurally excluded — see ``_assert_no_secrets``. A base URL is
#: not a secret and is included, because pointing at a different upstream is
#: exactly the kind of difference a transcript comparison must not miss.
OBSERVED_FLAGS: tuple[str, ...] = (
    # Deployment mode.
    "APP_ENV",
    # Which model answers, and how routing is allowed to move.
    "CHAT_PROVIDER",
    "ANALYZER_PROVIDER",
    "CLEOPATRA_MODEL_CATALOG",
    "OPENROUTER_BASE_URL",
    "SELF_HOSTED_BASE_URL",
    "SHOW_V3_PROMPT",
    # Conversational and commercial controllers (review §3F: several systems
    # direct the same reply, and they have different defaults).
    "COMMERCIAL_LAYER_ENABLED",
    "CONVERSATION_DIRECTOR_ENABLED",
    "EXPERIENCE_DIRECTOR_ENABLED",
    "ADAPTIVE_SESSION_PLANNER_ENABLED",
    "AFFORDABILITY_ENABLED",
    "PRICE_LEARNING_ENABLED",
    "FAN_LIFECYCLE_ENABLED",
    # Durable memory (review §3E: live extraction is flag-gated and defaults
    # off in code, which says nothing about the deployed value).
    "FAN_INTELLIGENCE_ENABLED",
    "HISTORY_EXTRACTION_ENABLED",
    "HISTORY_BACKFILL_ENABLED",
    # The platform boundary and the simulator.
    "APIFANSLY_ENABLED",
    "AUTO_SIMULATION_ENABLED",
    "AUTO_SIMULATION_AGENCY_ACCESS",
    "FANSLY_LISTS_SYNC_ENABLED",
    # Telemetry, because an absent trace is otherwise indistinguishable from a
    # turn that never happened.
    "MODEL_TELEMETRY_ENABLED",
)

#: A flag whose name contains one of these never enters a provenance record.
_SECRET_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")

#: What an unset variable reports as. Deliberately not "" — "unset" and "set to
#: empty" resolve differently in this codebase (see ``core/environment.py``),
#: and a snapshot that conflated them would hide the exact class of bug that
#: SEC-004 was.
UNSET = "<unset>"


def _assert_no_secrets(names: tuple[str, ...]) -> None:
    """Refuse at import to publish anything named like a credential."""
    leaking = [name for name in names if any(m in name for m in _SECRET_MARKERS)]
    if leaking:
        raise RuntimeError(
            "core.build_info.OBSERVED_FLAGS must not contain credentials; "
            f"remove {', '.join(sorted(leaking))}"
        )


_assert_no_secrets(OBSERVED_FLAGS)


def _sha_from_working_tree() -> str:
    """Resolve HEAD by reading ``.git`` directly, with no git binary.

    Local runs and the test suite have a working tree; containers usually do
    not. Anything unreadable resolves to empty rather than raising: build
    information must never be the reason a reply is not sent.
    """
    try:
        git_dir = Path(__file__).resolve().parent.parent / ".git"
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            ref_path = git_dir / ref
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
            # A packed ref — the loose file is gone once git has packed it.
            packed = git_dir / "packed-refs"
            if packed.exists():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.startswith("#") or " " not in line:
                        continue
                    sha, name = line.split(" ", 1)
                    if name.strip() == ref:
                        return sha.strip()
            return ""
        return head
    except Exception:
        return ""


@lru_cache(maxsize=1)
def build_sha() -> str:
    """The commit this process is running, or ``"unknown"``.

    Cached: a running container cannot change the code it was built from, and
    this is read once per reply.
    """
    for name in _SHA_ENV_VARS:
        value = str(os.environ.get(name, "")).strip()
        if value:
            return value
    return _sha_from_working_tree() or "unknown"


def short_sha(sha: str | None = None) -> str:
    """The first twelve characters, which is what a log line should carry."""
    resolved = sha if sha is not None else build_sha()
    return resolved[:12] if resolved and resolved != "unknown" else "unknown"


def active_flags() -> dict[str, str]:
    """Every observed flag, as this process actually sees it right now.

    Not cached. Tests and the simulator change flags within one process, and a
    snapshot that reported the value at import time would be describing a
    configuration that is no longer running.
    """
    flags: dict[str, str] = {}
    for name in OBSERVED_FLAGS:
        if name in os.environ:
            flags[name] = str(os.environ[name]).strip()
        else:
            flags[name] = UNSET
    return flags


def flags_digest(flags: dict[str, str] | None = None) -> str:
    """A short stable fingerprint of the whole flag snapshot.

    Small enough to store on every message. Equal digests mean two replies ran
    under the same configuration; different digests mean the comparison between
    them is not a like-for-like one, whatever else the transcripts show.
    """
    resolved = active_flags() if flags is None else flags
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def build_snapshot(*, include_flags: bool = True) -> dict[str, Any]:
    """The whole ground-truth answer to "what was running?".

    ``include_flags=False`` returns just the identifying triple, which is what
    goes onto a message row: the full mapping belongs in the startup log and the
    build endpoint, not duplicated onto every reply.
    """
    flags = active_flags()
    snapshot: dict[str, Any] = {
        "sha": build_sha(),
        "env": app_env(),
        "flags_digest": flags_digest(flags),
    }
    if include_flags:
        snapshot["flags"] = flags
    return snapshot


def describe_build() -> str:
    """One line for startup logs, so the running commit is never a guess."""
    flags = active_flags()
    enabled = [
        name
        for name, value in sorted(flags.items())
        if value.lower() in {"1", "true", "yes", "on"}
    ]
    return (
        f"[BUILD] sha={short_sha()} env={app_env()} "
        f"flags_digest={flags_digest(flags)} "
        f"enabled={','.join(enabled) if enabled else 'none'}"
    )
