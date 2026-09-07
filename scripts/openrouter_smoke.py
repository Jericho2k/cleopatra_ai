#!/usr/bin/env python3
"""Live OpenRouter smoke check for the ordinary Kimi writer.

Not part of the automated suite: it spends real credits and needs a real
OPENROUTER_API_KEY, so CI must never run it. Invoke it by hand when validating a
deployment or a provider-pin change.

    OPENROUTER_API_KEY=... python scripts/openrouter_smoke.py

It sends two identical-prefix requests using the same conversation affinity key
and prints, for each: the upstream provider OpenRouter used, prompt/cached/
completion tokens, reported cost, and latency. The second call is the one that
should show cached_tokens > 0 if provider-side prefix caching is working.

No secret is printed and no conversation content is sent — the prompt is
synthetic filler.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai import openrouter_routing  # noqa: E402
from ai.model_providers import complete, find_catalog_target  # noqa: E402
from ai.session_affinity import writer_end_user_id, writer_session_id  # noqa: E402
from ai.writer_router import DEFAULT_WRITER_MODEL  # noqa: E402
from models.model_runtime import ModelTarget, resolve_cost_usd  # noqa: E402

# Long enough that a provider has something worth caching, and identical across
# both calls so the second request can hit the first request's prefix.
_STABLE_PREFIX = (
    "You are a smoke-test assistant. Follow these rules exactly.\n"
    + "\n".join(f"Rule {index}: answer briefly and literally." for index in range(1, 400))
)


def _target() -> ModelTarget:
    model = os.getenv("WRITER_DEFAULT_MODEL", DEFAULT_WRITER_MODEL)
    catalog = find_catalog_target("openrouter", model)
    if catalog is not None:
        return catalog
    return ModelTarget(
        name=f"openrouter:{model}",
        provider="openrouter",
        model=model,
        base_url=openrouter_routing.base_url(),
        api_key_env=openrouter_routing.DEFAULT_API_KEY_ENV,
    )


async def main() -> int:
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        print("OPENROUTER_API_KEY is not set; nothing to smoke test.")
        return 2

    target = _target()
    session_id = writer_session_id("smoke-creator", "smoke-fan")
    end_user_id = writer_end_user_id("smoke-creator", "smoke-fan")

    print(f"model            : {target.model}")
    print(f"base_url         : {target.base_url}")
    print(f"provider pin     : {openrouter_routing.provider_preferences(target.metadata)}")
    print(f"session affinity : {session_id}")
    print()

    for attempt in (1, 2):
        try:
            result = await complete(
                target,
                system=_STABLE_PREFIX,
                messages=[{"role": "user", "content": f"Say OK. Attempt {attempt}."}],
                max_tokens=32,
                temperature=0.0,
                session_id=session_id,
                end_user_id=end_user_id,
            )
        except Exception as error:
            # A pinned-provider or data_collection rejection must be loud, not
            # quietly retried against some other upstream.
            print(f"attempt {attempt}: FAILED {type(error).__name__}: {error}")
            return 1

        usage = result.usage
        print(
            f"attempt {attempt}: upstream={result.upstream_provider or 'not reported'} "
            f"input={usage.input_tokens} cached={usage.cache_read_tokens} "
            f"cache_write={usage.cache_write_tokens} output={usage.output_tokens} "
            f"latency={result.latency_ms}ms "
            f"cost=${resolve_cost_usd(target, usage, reported_cost_usd=result.reported_cost_usd):.6f} "
            f"({'reported' if result.reported_cost_usd is not None else 'estimated'}) "
            f"text={result.text.strip()[:40]!r}"
        )

    print(
        "\nA cached count of 0 on attempt 2 means the pinned upstream did not "
        "serve this prefix from cache. Check that both calls reached the same "
        "provider before changing the prompt."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
