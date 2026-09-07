"""Deployment-mode resolution — the one place that decides "is this production?".

SEC-004. The historical rule was ``os.environ.get("APP_ENV", "development") ==
"development"``, so the *absence* of a variable selected the permissive mode. One
missing Railway variable therefore disabled dashboard auth, tenancy checks, and
webhook signature verification simultaneously.

The polarity is now inverted: relaxed behaviour requires an explicit, recognised
development value. Anything else — unset, misspelled, empty, ``staging``,
``Production`` with a stray space — resolves to production and fails closed.

    APP_ENV=development / dev / local   -> development
    APP_ENV=test                        -> test
    APP_ENV=production / prod           -> production
    APP_ENV unset or unrecognised       -> production (fail closed, warns once)

``test`` is deliberately *not* development: the backend test-suite exercises the
production enforcement paths, so it must keep failing closed. What ``test``
unlocks is the local-only helper routes (SEC-005), nothing security-relevant.
"""
from __future__ import annotations

import os

DEVELOPMENT = "development"
TEST = "test"
PRODUCTION = "production"

# Only these spellings select a non-production mode. The mapping is exhaustive
# on purpose: an unrecognised value must not be able to widen access.
_KNOWN_ENVIRONMENTS = {
    "development": DEVELOPMENT,
    "dev": DEVELOPMENT,
    "local": DEVELOPMENT,
    "test": TEST,
    "testing": TEST,
    "production": PRODUCTION,
    "prod": PRODUCTION,
}

_warned_values: set[str] = set()


def _warn_once(raw: str) -> None:
    if raw in _warned_values:
        return
    _warned_values.add(raw)
    shown = raw or "<unset>"
    print(
        f"[APP_ENV] {shown!r} is not a recognised environment — running with "
        "production semantics (fail closed). Set APP_ENV explicitly to one of: "
        "development, test, production."
    )


def app_env() -> str:
    """Resolve APP_ENV to one of development/test/production, defaulting closed."""
    raw = str(os.environ.get("APP_ENV", "")).strip()
    resolved = _KNOWN_ENVIRONMENTS.get(raw.lower())
    if resolved is None:
        _warn_once(raw)
        return PRODUCTION
    return resolved


def is_development() -> bool:
    """True only for an explicit development value.

    This is the gate for every relaxed-auth branch. It must never be true by
    default, and ``test`` must not satisfy it.
    """
    return app_env() == DEVELOPMENT


def is_test() -> bool:
    return app_env() == TEST


def is_production() -> bool:
    return app_env() == PRODUCTION


def local_test_endpoints_enabled() -> bool:
    """Whether the ``/test/*`` helper routes exist at all (SEC-005).

    Development and automated tests may use them. Production must behave as
    though they were never registered.
    """
    return app_env() in {DEVELOPMENT, TEST}


def describe_environment() -> str:
    """One line for startup logs, so the resolved mode is never a guess."""
    raw = str(os.environ.get("APP_ENV", "")).strip() or "<unset>"
    return f"APP_ENV={raw} resolved={app_env()}"
