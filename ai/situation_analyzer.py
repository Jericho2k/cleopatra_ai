"""Step 1 of the prompt chain: understand the conversation.

This module extracts observations only. Commercial decisions are made by the
policy layer. A deterministic normalizer handles common offer-selection phrases
so one mixed sentence cannot be collapsed into a generic decline.

REL-001 — a failed analysis is now distinguishable from a real one. The fallback
used to be a complete, well-formed, entirely neutral analysis with no marker, so
downstream code could not tell "the fan said nothing commercial" from "we never
got an answer and guessed". During a provider incident Full Auto kept selling on
guessed state. Every result now carries ``analysis_degraded``; callers that make
autonomous commercial decisions must check it with ``analysis_is_degraded``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ai.model_providers import complete, get_runtime_target
from models.model_runtime import ModelTelemetryContext
from models.schemas import ConversationContext
from services.analyzer_telemetry import record_analysis_outcome
from services.commercial_events import normalize_commercial_facts
from services.model_telemetry import record_model_failure, record_model_result

# Why an analysis could not be trusted. Deliberately coarse: these travel into
# telemetry and the dashboard, and must never carry provider payloads or keys.
DEGRADED_TRANSPORT = "transport_error"
DEGRADED_PARSE = "parse_error"
DEGRADED_SHAPE = "unexpected_response"


def analysis_is_degraded(situation: dict | None) -> bool:
    """True when this analysis was fabricated rather than returned by the model."""
    return bool((situation or {}).get("analysis_degraded"))


def degraded_reason(situation: dict | None) -> str:
    return str((situation or {}).get("degraded_reason") or "")


async def analyze_situation(
    ctx: ConversationContext,
    *,
    telemetry_context: dict[str, Any] | None = None,
) -> dict:
    recent = ctx.conversation_history[-12:]
    convo = "\n".join(
        f"{'Fan' if message.role == 'fan' else 'Creator'}: {message.content}"
        for message in recent
    )

    user_content = f"""You are analyzing an adult creator chat so another system can decide the correct business action.

Conversation so far:
{convo}

Latest fan message: "{ctx.fan_message}"

Return ONLY valid JSON with exactly these fields:
{{
  "fan_mood": "excited/bored/horny/lonely/curious/frustrated/romantic/testing/shy",
  "fan_intent": "brief description of what the latest message means",
  "conversation_energy": "rising/flat/dropping",
  "strategic_move": "mirror_warmth/tease_and_deflect/get_curious/hint_at_content/build_tension/re_engage/push_for_ppv/acknowledge_compliment_and_redirect",
  "tone": "playful/warm/flirty/mysterious/direct/casual",
  "personal_details_mentioned": ["facts the fan just stated"],
  "avoid_repeating": "what the creator should avoid repeating",

  "purchase_signal": "none/ready_to_buy/bought/declined/money_available/uncertain",
  "offer_response": "none/accepted/declined/counteroffer/deferred",
  "selected_offer_price_usd": "number only, or empty string",
  "selected_offer_position": "first/second/empty",
  "current_budget_limit_usd": "number only, or empty string",
  "cannot_afford_any_offer_now": "true/false",
  "deferred_purchase_intent": "true/false",

  "resend_requested": "true/false",
  "crisis_signal": "none/self_harm/harm_to_others",
  "wants_explicit": "true/false",
  "wants_media": "true/false",
  "payday_raw": "exact timing words only, e.g. Friday/next week/the 1st, or empty",
  "payday_confidence": 0.0,
  "budget_stated_usd": "number only if the fan explicitly says what he has available now, otherwise empty",
  "desired_experience": "the fan's concrete requested theme/action/location/outfit/body focus/format in a short natural phrase, or empty"
}}

COMMERCIAL INTERPRETATION RULES:
- Treat facts independently. A fan can select a cheaper offer now AND mention a future payday.
- Example: "can we do the $28 one, I don't have more right now, I get paid Friday" means:
  offer_response=accepted, selected_offer_price_usd=28, current_budget_limit_usd=28,
  cannot_afford_any_offer_now=false, payday_raw=Friday, deferred_purchase_intent=false,
  purchase_signal=ready_to_buy.
- cannot_afford_any_offer_now=true only when he cannot buy ANY offered option now.
- "I can't spend more than $28" is a limit, not a refusal, if he accepts the $28 option.
- declined means he refused the available offer(s), not merely that he chose the cheaper one.
- deferred means he wants a specific offer later rather than now.
- money_available means previously unavailable money is available now.
- A compliment alone is NOT a content request. "you look sexy", "cute", "hot", or
  "that last post was sexy" means wants_explicit=false, wants_media=false,
  purchase_signal=none unless the same message directly asks to see, receive,
  unlock, buy, or continue content.
- wants_media=true only for a direct request for photos, video, content, a set,
  a custom, a session, an unlock, or a clear affirmative response to an exact
  offer.
- wants_explicit=true only when the fan actively asks for explicit text/action,
  describes what he wants to do or see, or directly escalates the interaction.
  A sexual adjective by itself is warm interest, not offer readiness.
- "show me more", "send the video", "how much", "can I see the full set", and
  "I want you so bad" are active intent. "and sexy 🥵" by itself is not.
- desired_experience must preserve the fan's specific searchable words. Examples:
  "show me the shower set" -> "shower set"; "any feet videos?" -> "feet video";
  "I want the red lingerie one" -> "red lingerie". Never collapse a concrete
  request into "other". Leave it empty when no specific experience was requested.

SAFETY:
- Crisis is not sexual roughness or consensual roleplay. Flag self_harm for plausible self-directed harm language and harm_to_others only for real intent toward a real person.
"""

    target = get_runtime_target("ANALYZER")
    metadata = telemetry_context or {}
    model_context = ModelTelemetryContext(
        feature=str(metadata.get("feature") or "situation_analyzer"),
        creator_id=metadata.get("creator_id"),
        fan_id=metadata.get("fan_id") or ctx.fan_profile.id,
        metadata={
            key: value
            for key, value in metadata.items()
            if key not in {"feature", "creator_id", "fan_id"}
        },
    )

    try:
        response = await complete(
            target,
            system="Return only the requested JSON object. Do not add commentary.",
            messages=[{"role": "user", "content": user_content}],
            max_tokens=650,
            temperature=0.0,
        )
        content = response.text.replace("```json", "").replace("```", "").strip()
        try:
            result = json.loads(content)
            parse_valid = isinstance(result, dict)
        except Exception as error:
            result = _fallback_result(DEGRADED_PARSE)
            parse_valid = False
            await record_model_result(
                response,
                model_context,
                success=False,
                parse_valid=False,
                error=f"parse_error: {error}",
            )
        else:
            if parse_valid:
                # A genuine answer. State it positively rather than relying on
                # the key's absence, so a caller cannot mistake an older or
                # partially built dict for a trusted analysis.
                result["analysis_degraded"] = False
                result["degraded_reason"] = ""
            else:
                # Valid JSON of the wrong shape — a list, a string, a number.
                # Just as unusable as a parse error, and previously it sailed
                # through as a truthy non-dict.
                result = _fallback_result(DEGRADED_SHAPE)
            await record_model_result(
                response,
                model_context,
                success=parse_valid,
                parse_valid=parse_valid,
            )
    except Exception as error:
        await record_model_failure(target, model_context, error=str(error))
        # The message is logged but never put in the result: provider errors can
        # echo request payloads, and this dict reaches the dashboard.
        print(f"[SITUATION ANALYZER ERROR] provider={target.provider} model={target.model} error={error}")
        result = _fallback_result(DEGRADED_TRANSPORT)

    # Deterministic safety backstop. It runs on degraded results too — a failed
    # analyzer must never weaken crisis detection.
    if _looks_like_self_harm(ctx.fan_message):
        result["crisis_signal"] = "self_harm"

    creator_lines = [message.content for message in recent if message.role == "creator"]
    normalized = normalize_commercial_facts(result, ctx.fan_message, creator_lines)
    # normalize_commercial_facts merges over its own defaults, so re-assert the
    # markers rather than trusting them to survive a future change there.
    normalized["analysis_degraded"] = bool(result.get("analysis_degraded"))
    normalized["degraded_reason"] = str(result.get("degraded_reason") or "")

    record_analysis_outcome(
        degraded=normalized["analysis_degraded"],
        reason=normalized["degraded_reason"],
        feature=model_context.feature,
    )
    return normalized


def _fallback_result(reason: str = DEGRADED_TRANSPORT) -> dict:
    """A syntactically complete analysis that is explicitly marked as guessed.

    The shape has to stay complete so downstream key access does not crash, but
    every consumer that acts autonomously must gate on analysis_degraded.
    """
    return {
        "analysis_degraded": True,
        "degraded_reason": reason,
        "fan_mood": "curious",
        "fan_intent": "engaging with creator",
        "conversation_energy": "flat",
        "strategic_move": "mirror_warmth",
        "tone": "playful",
        "personal_details_mentioned": [],
        "avoid_repeating": "",
        "purchase_signal": "none",
        "offer_response": "none",
        "selected_offer_price_usd": "",
        "selected_offer_position": "",
        "current_budget_limit_usd": "",
        "cannot_afford_any_offer_now": "false",
        "deferred_purchase_intent": "false",
        "resend_requested": "false",
        "crisis_signal": "none",
        "wants_explicit": "false",
        "wants_media": "false",
        "payday_raw": "",
        "payday_confidence": 0.0,
        "budget_stated_usd": "",
        "desired_experience": "",
        "commercial_interest_signal": "none",
    }


_SELF_HARM_PATTERNS = (
    "cut my vein",
    "cut my wrist",
    "slit my wrist",
    "slit my vein",
    "kill myself",
    "end my life",
    "end it all",
    "want to die",
    "don't want to live",
    "dont want to live",
    "don't want to be alive",
    "dont want to be alive",
    "not worth living",
    "harm myself",
    "hurt myself",
    "bleed out",
    "overdose",
    "take my own life",
    "suicidal",
    "suicide",
)


def _looks_like_self_harm(message: str) -> bool:
    text = (message or "").lower()
    return any(pattern in text for pattern in _SELF_HARM_PATTERNS)
