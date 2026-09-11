#!/usr/bin/env python3
"""Live writer smoke check: does this route return copy the writer can USE?

Not part of the automated suite: it spends real credits and needs a real
provider key, so CI must never run it. Invoke it by hand when validating a
deployment, a provider pin, or a model change.

    OPENROUTER_API_KEY=... python scripts/openrouter_smoke.py
    TOGETHER_API_KEY=...   python scripts/openrouter_smoke.py --route complex

Why it changed
--------------

The previous version reported transport and caching only. It printed

    attempt 1 output=32 text=''
    attempt 2 output=32 text=''

and exited 0, so a writer that produced NOTHING passed the smoke twice. That is
exactly the production failure it was supposed to catch: Kimi K2.6 reasons by
default, spent its whole completion budget on hidden reasoning, and returned
``message.content = null``. Routing was perfect. Caching worked. The writer was
dead.

So this now tests writer usefulness, not routing:

* the request carries the writer's real contract — return a JSON array of short
  chat replies and nothing else;
* the response must have non-empty message content;
* that content must parse the way ``ai.generator.parse_reply_outcome`` parses
  it, into at least one usable reply;
* the pinned upstream must be the one that was configured, so an unnoticed
  provider change is a failure rather than a surprise in production.

Any of those failing exits non-zero.

The cache test is kept: two requests share a long identical prefix and one
conversation affinity key, and the second should report cached tokens if
provider-side prefix caching is live. A cold cache is reported, not failed —
it is a cost regression, not a broken writer.

No secret is printed and no conversation content is sent: the prompt is
synthetic filler and the requested output is two fixed nonsense strings.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai import openrouter_routing  # noqa: E402
from ai.generator import parse_reply_outcome  # noqa: E402
from ai.model_providers import complete, find_catalog_target  # noqa: E402
from ai.model_providers import provider_transport_defaults  # noqa: E402
from ai.session_affinity import writer_end_user_id, writer_session_id  # noqa: E402
from ai.writer_router import (  # noqa: E402
    COMPLEX_WRITER_MODEL,
    COMPLEX_WRITER_PROVIDER,
    DEFAULT_WRITER_MODEL,
    DEFAULT_WRITER_PROVIDER,
)
from models.model_runtime import ModelTarget, resolve_cost_usd  # noqa: E402
from models.schemas import Persona  # noqa: E402

# Long enough that a provider has something worth caching, and identical across
# both calls so the second request can hit the first request's prefix.
_STABLE_PREFIX = (
    "You are a smoke-test assistant standing in for a chat writer.\n"
    + "\n".join(f"Rule {index}: answer briefly and literally." for index in range(1, 400))
    + "\n\nOutput contract: reply with a JSON array of 2 short strings and "
    "NOTHING else. No prose before or after it, no code fence, no explanation. "
    'Example of the exact shape: ["first line", "second line"]'
)

# The two strings the model is asked to return. Fixed nonsense on purpose: the
# smoke must never send anything resembling real conversation content, and a
# deterministic answer means a failure is the transport or the model, never the
# writer's own bot-phrase validator disliking a turn of phrase it invented.
_EXPECTED = ["alpha one", "beta two"]

_USER_TURN = (
    'Return exactly this JSON array and nothing else: ["alpha one", "beta two"]'
)

# Enough headroom that a NON-reasoning model comfortably finishes the array, and
# little enough that a model still spending its budget on hidden reasoning runs
# out and is caught. That asymmetry is the point of the number.
_MAX_TOKENS = 200


def _target(route: str) -> ModelTarget:
    """Resolve the same target production would use for this route."""
    if route == "complex":
        provider = os.getenv("WRITER_COMPLEX_PROVIDER", COMPLEX_WRITER_PROVIDER)
        model = os.getenv("WRITER_COMPLEX_MODEL", COMPLEX_WRITER_MODEL)
    else:
        provider = os.getenv("WRITER_DEFAULT_PROVIDER", DEFAULT_WRITER_PROVIDER)
        model = os.getenv("WRITER_DEFAULT_MODEL", DEFAULT_WRITER_MODEL)

    provider = provider.strip().lower()
    catalog = find_catalog_target(provider, model)
    if catalog is not None:
        return catalog

    base_url, api_key_env = provider_transport_defaults(provider)
    return ModelTarget(
        name=f"{provider}:{model}",
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
    )


def _check_usable(text: str) -> tuple[bool, str, list[str]]:
    """Apply the writer's own parser. Transport success is not success."""
    if not text.strip():
        return False, "message content is empty", []
    outcome = parse_reply_outcome(text, Persona())
    if not outcome.replies:
        return False, f"content did not parse into usable replies ({outcome.reason})", []
    if [reply.strip().lower() for reply in outcome.replies] != _EXPECTED:
        return (
            False,
            f"parsed {outcome.replies!r}, expected {_EXPECTED!r} — the route no "
            "longer follows the writer's output contract",
            outcome.replies,
        )
    return True, "", outcome.replies


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        choices=("default", "complex"),
        default="default",
        help="Which production writer target to smoke (default: the ordinary route).",
    )
    args = parser.parse_args()

    target = _target(args.route)
    key_env = target.api_key_env or ""
    if not os.getenv(key_env, "").strip():
        print(f"{key_env} is not set; nothing to smoke test.")
        return 2

    expected_upstreams = (
        {name.strip().lower() for name in openrouter_routing.pinned_providers(target.metadata) or []}
        if target.provider == "openrouter"
        else set()
    )

    session_id = writer_session_id("smoke-creator", "smoke-fan")
    end_user_id = writer_end_user_id("smoke-creator", "smoke-fan")

    print(f"route            : {args.route}")
    print(f"provider         : {target.provider}")
    print(f"model            : {target.model}")
    print(f"base_url         : {target.base_url}")
    print(f"reasoning_enabled: {target.metadata.get('reasoning_enabled')}")
    if target.provider == "openrouter":
        print(f"provider pin     : {openrouter_routing.provider_preferences(target.metadata)}")
    print(f"session affinity : {session_id}")
    print()

    failures: list[str] = []
    cached_on_second = 0

    for attempt in (1, 2):
        try:
            result = await complete(
                target,
                system=_STABLE_PREFIX,
                messages=[{"role": "user", "content": _USER_TURN}],
                max_tokens=_MAX_TOKENS,
                temperature=0.0,
                session_id=session_id,
                end_user_id=end_user_id,
            )
        except Exception as error:
            # A pinned-provider, data_collection or non-serverless rejection must
            # be loud, not quietly retried against some other upstream.
            print(f"attempt {attempt}: FAILED {type(error).__name__}: {error}")
            return 1

        usage = result.usage
        usable, why, replies = _check_usable(result.text)
        if attempt == 2:
            cached_on_second = usage.cache_read_tokens

        print(
            f"attempt {attempt}: upstream={result.upstream_provider or 'not reported'} "
            f"input={usage.input_tokens} cached={usage.cache_read_tokens} "
            f"cache_write={usage.cache_write_tokens} output={usage.output_tokens} "
            f"latency={result.latency_ms}ms "
            f"cost=${resolve_cost_usd(target, usage, reported_cost_usd=result.reported_cost_usd):.6f} "
            f"({'reported' if result.reported_cost_usd is not None else 'estimated'})"
        )
        print(
            f"           content_empty={str(not result.text.strip()).lower()} "
            f"usable={str(usable).lower()} parsed_replies={len(replies)}"
        )

        if not usable:
            # The exact failure the old smoke passed through.
            failures.append(f"attempt {attempt}: {why}")

        # An unannounced upstream change silently changes writing style, price
        # and cache locality, so it is a failure here rather than a surprise in
        # production.
        served_by = (result.upstream_provider or "").strip().lower()
        if expected_upstreams and served_by and served_by not in expected_upstreams:
            failures.append(
                f"attempt {attempt}: served by upstream {result.upstream_provider!r}, "
                f"pinned to {sorted(expected_upstreams)}"
            )

    print()
    if cached_on_second == 0:
        print(
            "NOTE: cached=0 on attempt 2 — the upstream did not serve this prefix "
            "from cache. Check that both calls reached the same provider before "
            "changing the prompt. Reported, not failed: this is a cost "
            "regression, not a broken writer."
        )
    else:
        print(f"prefix cache: OK ({cached_on_second} cached tokens on attempt 2)")

    if failures:
        print("\nSMOKE FAILED — this route cannot be used as a writer:")
        for failure in failures:
            print(f"  - {failure}")
        print(
            "\nIf content_empty=true with output_tokens at or near the limit, the "
            "model spent its budget on hidden reasoning. Set "
            '"reasoning_enabled": false in the catalog entry for this target.'
        )
        return 1

    print("\nSMOKE PASSED — the route returned usable writer output on both attempts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
