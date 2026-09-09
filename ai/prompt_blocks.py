"""Structured prompt content and how each transport consumes it.

COST-002a — ``build_prompt`` has always emitted the system message as ordered
content blocks carrying ``cache_control``, but ``generate_replies`` flattened
them to a plain string before calling ``complete``, so the cache directive was
built and then thrown away on every single call. Anthropic prompt caching was
inert for the whole deployment.

The split now happens at the transport rather than at the caller:

* Anthropic consumes the blocks natively and honours ``cache_control``.
* Every OpenAI-compatible provider — OpenRouter, Together, self-hosted — wants a
  single string and relies on *implicit* prefix caching, so the blocks are
  joined there. Joining preserves block order, which is what keeps the cacheable
  prefix byte-identical between turns.

Callers therefore pass whichever shape is natural and never have to know which
provider they are about to reach.
"""

from __future__ import annotations

from typing import Any

# Anthropic will not cache a block shorter than its per-model minimum: 1,024
# tokens for Sonnet and Opus, 2,048 for Haiku. Below the lower of those a
# ``cache_control`` marker cannot do anything on any Anthropic model, so it is
# left off rather than sent as noise. Above it, whether the marker engages is
# the model's own business and the request succeeds either way.
#
# Four characters per token is the same rough conversion used by
# scripts/measure_prompt_cache.py. It only has to be right enough to keep a
# clearly-too-short block from being marked.
MIN_CACHEABLE_CHARS = 4 * 1024


def cacheable_system_blocks(
    stable: str,
    volatile: str = "",
) -> list[dict[str, Any]]:
    """Build a system message whose stable prefix is marked for caching.

    ``stable`` must be byte-identical between turns of one conversation;
    ``volatile`` is appended behind it so a per-message addition cannot evict the
    cached prefix.
    """

    stable_block: dict[str, Any] = {"type": "text", "text": stable}
    if len(stable) >= MIN_CACHEABLE_CHARS:
        stable_block["cache_control"] = {"type": "ephemeral"}

    blocks = [stable_block]
    if volatile:
        blocks.append({"type": "text", "text": volatile})
    return blocks


def flatten_message_content(content: Any) -> str:
    """Collapse structured prompt content into the plain string a transport wants.

    Block order is preserved and no marker is ever serialized into the text, so
    an OpenAI-compatible provider sees exactly the prose an Anthropic one does.
    """

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def has_cache_control(content: Any) -> bool:
    """Whether any block in this content carries a provider cache directive."""

    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("cache_control")
        for block in content
    )
