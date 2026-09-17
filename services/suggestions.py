"""Suggestion orchestration service.

Coordinates DB, stage classification, RAG, prompt building, and generation
"""

import asyncio
from core.tasks import spawn
from datetime import datetime, timezone
import json
import os
import random
import re
import time
import uuid


from ai.generator import (
    CONTRACT_AUTO_MESSAGES,
    CONTRACT_CANDIDATES,
    LEGACY_WRITER_RETRY_POLICY,
    PERSISTENT_PRIMARY_RETRY_POLICY,
    generate_replies,
)
from ai.generation_trace import GenerationTrace
from ai.prompt_builder import WRITER_TRANSCRIPT_MESSAGES
from ai.situation_analyzer import ANALYZER_TRANSCRIPT_MESSAGES
from services.context_packet import build_context_packet
from ai.writer_router import select_writer_route
from services.ppv_turn import plan_ppv_step_delivery, strip_ppv_tags
from services.content_access import REVIEW_REASON as CONTENT_ACCESS_REVIEW_REASON
from services.episode_recording import close_finished_episode
from services.conversation_continuity import (
    open_threads_for,
    recent_episodes_for,
    record_open_thread,
    summarize_threads,
)
from models.conversation_continuity import (
    EvidenceType,
    OpenThread,
    ThreadKind,
    ThreadParty,
)
from services.reply_provenance import (
    DELIVERY_PPV,
    DELIVERY_TEXT,
    PIPELINE_ASSISTED,
    PIPELINE_AUTO,
    TRANSFORM_DELIVERY_LANGUAGE,
    TRANSFORM_INVENTORY_REPAIR,
    TRANSFORM_PPV_MERGED,
    TRANSFORM_PPV_TAG_STRIPPED,
    TRANSFORM_SHAPE_APPLIED,
    ReplyProvenance,
    fingerprint,
    merge_provenance,
)
from ai.stack_profiles import STAGE_FAN_SUMMARY, get_profile
from openai import AsyncOpenAI
from core.config import get_settings
from core.supabase import get_supabase
from ai.prompt_builder import build_prompt
from ai.writer_style import (
    MODE_ASSISTED,
    MODE_AUTO,
    candidate_count as writer_candidate_count,
    enforces_message_shape as writer_enforces_message_shape,
    persists_improvised_facts as writer_persists_improvised_facts,
    persistent_primary_retries as writer_persistent_primary_retries,
    uses_auto_messages_contract as writer_uses_auto_messages_contract,
)


def writer_retry_policy(prompt_version: str):
    """The retry plan this writer version's primary model gets.

    V3 makes Kimi the real primary: four attempts spaced by the configured
    schedule before Qwen is reached at all. V1 and V2 keep the frozen two-then-
    fallback plan, because they are the comparison baseline.
    """
    return (
        PERSISTENT_PRIMARY_RETRY_POLICY
        if writer_persistent_primary_retries(prompt_version)
        else LEGACY_WRITER_RETRY_POLICY
    )
from services.commercial_orchestrator import (
    consume_free_text_allowance,
    orchestrate,
)
from models.commercial import ActionType, FanStatus
from db.fan_intelligence_queries import get_fan_intelligence_context
from db.commercial_queries import (
    cancel_action_by_dedupe_key,
    get_approved_asset_types,
    cancel_actions_for_fan,
    get_creator_policy,
    get_fan_state,
    save_fan_state,
    schedule_action,
)
from services.session_planner import plan_session_for_fan
from core.action_telemetry import record_stage, stage as action_stage
from services.human_delivery import (
    build_availability_delay,
    build_delivery_schedule,
    visible_text,
)
from services.ppv_delivery import create_ppv_approval_request
from services.db_reliability import retry_transient_db_operation
from core.apifansly_gate import apifansly_enabled, simulation_scope
from core.simulation import mark_simulation_owned_message, simulation_message_marker
from core.simulation_catalog import contains_simulation_media
from services.apifansly import (
    CATEGORY_LIVE_CHAT,
    headers as apifansly_headers,
    record_raw_call as record_apifansly_raw_call,
    shared_client as apifansly_shared_client,
    send_message as send_apifansly_message,
    sent_message_id,
    url as apifansly_url,
)
from services.ppv_persistence import (
    persist_ppv_reconciliation,
    save_ppv_message_receipt,
)
from services.followup_lifecycle import (
    complete_session_state,
)
from services.ai_stack import log_effective_stack, resolve_ai_stack
from services.fan_intelligence import learn_from_fan_message
from services.affordability import (
    get_affordability_context,
    record_confirmed_purchase,
    refresh_affordability_from_situation,
)
from services.fan_lifecycle import (
    get_fan_lifecycle_context,
    refresh_fan_lifecycle,
)
from services.price_learning import (
    get_price_learning_context,
    refresh_price_learning,
)
from services.adaptive_session_planner import plan_next_action
from services.conversation_director import direct_conversation
from services.experience_director import (
    direct_experience,
    load_scene,
    record_unlock,
    scene_metadata_for,
)
from services.commercial_policy import free_mode_on_cooldown
from services.text_intimacy import decide_text_intimacy
from services.session_lifecycle import (
    mark_step_declined,
    mark_step_purchased,
)
from services.creator_canon import persist_sent_creator_facts
from services.message_shape import (
    apply_message_shape,
    choose_message_shape,
    recent_bubble_counts,
)
from services.ppv_language import sanitize_candidates, sanitize_delivery_language
from services.inventory_authority import (
    MediaInventory,
    asset_types_from_session,
    choose_inventory_safe_reply,
    next_step_asset_type,
    sanitize_media_promises,
)
from ai.situation_analyzer import (
    analyze_situation,
    analysis_is_degraded,
    degraded_reason,
)
from ai.rag import find_similar_exchanges
from ai.stage_classifier import classify_stage
from db.queries import (
    PurchaseAggregateConflict,
    apply_purchase_to_fan,
    get_conversation_history,
    get_creator_persona,
    get_fan_by_id,
    get_fan_session,
    get_ppv_offers,
    get_sent_ppv,
    mark_ppv_purchased,
    save_fan_session,
    save_message,
    update_fan_memory,
    update_fan_ai_summary,
    update_creator_legend,
    get_creator_legend,
    get_creator_caps,
    set_fan_decline_lock,
    clear_fan_decline_lock,
    freeze_fan_for_review,
)
from models.schemas import (
    ConversationContext,
    Fan,
    Message,
    Persona,
    SuggestionResponse,
)

together_client = AsyncOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=get_settings().TOGETHER_API_KEY,
)

# What one Full Auto turn actually did. A turn that sends nothing is not one
# event but three, and reporting them as one is how a total writer failure came
# to be displayed as "Full Auto decided to send nothing this turn".
#
# These are the machine-readable names; they are produced by the real Auto path
# and are not a parallel engine. ``replied`` and ``no_send`` are observable from
# outside (a creator message exists, or it does not); ``analyzer_degraded`` and
# ``writer_failed`` are not, so the pipeline reports them explicitly.
AUTO_OUTCOME_REPLIED = "replied"
AUTO_OUTCOME_NO_SEND = "no_send"
AUTO_OUTCOME_ANALYZER_DEGRADED = "analyzer_degraded"
AUTO_OUTCOME_WRITER_FAILED = "writer_failed"
# A commercial plan could not be produced and recovery could not repair it.
# Distinct from an intentional no-send: it names a broken sale, not a choice.
AUTO_OUTCOME_PLAN_UNRECOVERABLE = "plan_unrecoverable"
# Every writer candidate promised media that does not exist, and repairing
# them left nothing sendable. Never send the promise instead.
AUTO_OUTCOME_INVENTORY_UNSAFE = "inventory_unsafe"
AUTO_OUTCOME_HUMAN_REVIEW = "human_review"


class HumanReviewHandoffError(RuntimeError):
    """A required review hold could not be persisted; retry without sending."""


class AnalyzerDegradedError(RuntimeError):
    """Full Auto refused to act on a fabricated analysis (REL-001).

    Raised rather than returned so the durable scheduled action records it and
    retries through the existing bounded backoff (max_attempts=8), instead of
    completing as though a reply had been delivered.
    """


_pending_auto_replies: dict[str, asyncio.Task] = {}


def _release_auto_reply_slot(fan_id: str) -> bool:
    """Clear a pending slot only when the current task still owns it."""
    if _pending_auto_replies.get(fan_id) is not asyncio.current_task():
        return False
    _pending_auto_replies.pop(fan_id, None)
    return True


def message_ai_stack_metadata(
    route, *, profile_id: str, trace: GenerationTrace | None = None
) -> dict:
    """The durable "which brain wrote this" marker for one creator message.

    Persisted inside ``messages.media_context``, which is existing jsonb
    metadata, so recording it needs no migration and nothing customer-visible
    changes. It is deliberately small: the profile that answered, the writer
    route it took, and the model that was actually asked. That is enough to
    answer "which AI stack produced this message?" months later, from the row
    alone, without a telemetry join.

    ``model`` and ``provider`` remained the model the router ASKED for even when
    a retry, another upstream host or the configured fallback is what actually
    answered — finding H of docs/autonomy_architecture_review.md, and the reason
    a message-level model comparison could not be trusted. When a
    ``GenerationTrace`` is supplied those two keys now name the model that
    served the text, and ``requested_model``/``requested_provider`` keep what
    was asked for, so both halves of "we asked for Kimi and got Qwen" survive on
    the row. Without a trace the marker is exactly what it was, so the callers
    that do not run the recovery ladder are unchanged.
    """
    marker: dict = {"profile": str(profile_id)}
    if route is not None:
        marker.update(
            {
                "route": route.route.value,
                "prompt_version": route.prompt_version,
                "provider": route.primary_target.provider,
                "model": route.primary_target.model,
            }
        )
    if trace is not None and trace.succeeded:
        marker["requested_provider"] = trace.requested_provider
        marker["requested_model"] = trace.requested_model
        marker["provider"] = trace.provider
        marker["model"] = trace.model
        marker["served_by_requested_model"] = trace.served_by_requested_model
        if trace.upstream_provider:
            marker["upstream_provider"] = trace.upstream_provider
    return {"ai_stack": marker}


def _with_ai_stack(media_context: dict | None, marker: dict) -> dict:
    """Merge the stack marker into whatever metadata this message already has."""
    merged = dict(media_context or {})
    merged.update(marker)
    return merged


def _is_local_test_fan(platform_fan_id: object) -> bool:
    """Only explicitly namespaced test fans may bypass live platform delivery."""
    return str(platform_fan_id or "").startswith("test_")


async def _sleep_while_current(fan_id: str, seconds: float, *, phase: str) -> bool:
    """Sleep in short slices so a newer fan message can cancel simulated typing.

    This is deliberate human realism, not queue latency, so it is attributed to
    its own telemetry stage. Reading a reply's total time without that split
    makes an intentional 8-second typing pause look like a capacity problem.
    """
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + max(0.0, float(seconds))
    bucket = (
        "availability_delay_ms" if phase == "availability" else "composition_delay_ms"
    )
    try:
        while True:
            current_task = _pending_auto_replies.get(fan_id)
            if current_task and current_task is not asyncio.current_task():
                print(f"[AUTO TIMING] fan={fan_id} cancelled phase={phase} newer_task=true")
                return False
            remaining = deadline - loop.time()
            if remaining <= 0:
                return True
            await asyncio.sleep(min(0.5, remaining))
    finally:
        record_stage(bucket, (loop.time() - started) * 1000.0)


_CONTENT_REQUEST_RE = re.compile(
    r"\b(want(?:ed|s)?\s+(?:to\s+)?see|wanna\s+see|can\s+i\s+see|see\s+more|more\s+of\s+(?:you|u)|"
    r"show\s+me|send\s+(?:me\s+)?(?:some|a|your|more|pics?|pictures?|vids?|videos?|content|nudes?)|"
    r"what\s+(?:do\s+)?(?:you|u)\s+(?:have|got|sell)|let'?s\s+play|wanna\s+play|"
    r"turn(?:s|ed)?\s+me\s+on|so\s+(?:hard|horny)|jerk(?:ing)?\s+off|touch(?:ing)?\s+myself)\b",
    re.IGNORECASE,
)


async def _crisis_freezes_chat(creator_id: str, fan_id: str, situation: dict) -> bool:
    """If a crisis is flagged and the creator's policy is 'freeze', mark the fan for
    human review and signal the auto path to stop. Returns True if the chat should be
    frozen (auto-reply must abort). 'continue' policy (default) returns False so the
    existing care-first crisis prompt handles it inline."""
    signal = (situation.get("crisis_signal") or "none")
    if signal == "none":
        return False
    try:
        caps = await get_creator_caps(creator_id)
    except Exception:
        caps = {}
    policy = (caps.get("crisis_policy") or "continue")
    if policy == "freeze":
        await freeze_fan_for_review(fan_id, f"crisis:{signal}")
        print(f"[CRISIS] fan={fan_id} FROZEN for human review (signal={signal})")
        return True
    return False


async def _within_daily_caps(creator_id: str, sent_ppv: list[dict], fan_profile) -> tuple[bool, str]:
    """Enforce the agency's per-fan daily autonomy caps before auto-selling.
    Returns (allowed, reason). No caps configured => always allowed.
    Counts today's sends/spend from sent_ppv (already loaded in the suggestion path)."""
    try:
        caps = await get_creator_caps(creator_id)
    except Exception:
        return True, ""  # never block selling on a config-read failure

    if not caps.get("caps_enabled"):
        return True, ""

    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date()

    def _is_today(sent_at: str) -> bool:
        if not sent_at:
            return False
        try:
            return datetime.fromisoformat(sent_at.replace("Z", "+00:00")).date() == today
        except Exception:
            return False

    todays = [s for s in (sent_ppv or []) if _is_today(s.get("sent_at", ""))]

    max_sends = caps.get("max_ppv_per_fan_per_day")
    if max_sends is not None and len(todays) >= int(max_sends):
        return False, f"daily send cap reached ({len(todays)}/{max_sends})"

    max_spend = caps.get("max_spend_per_fan_per_day")
    if max_spend is not None:
        spent_today = sum(int(s.get("price", 0) or 0) for s in todays if s.get("purchased"))
        if spent_today >= int(max_spend):
            return False, f"daily spend cap reached (${spent_today}/${max_spend})"

    return True, ""


def _selling_locked(fan_profile) -> bool:
    """True while the fan is under a decline lock (said he can't afford it and hasn't
    since signaled money). No planning, no PPV, no cheaper-item retry while locked."""
    return bool(getattr(fan_profile, "sale_paused_at", None))


def _fan_wants_content(message, situation):
    # Hard stop: never treat a crisis message as a buying signal. If the fan is
    # expressing genuine self-harm or intent to harm a real person, we do not sell.
    if situation and (situation.get("crisis_signal") or "none") != "none":
        return False
    # He just declined / said he's broke: "send me something for free" is not a
    # buying signal. Don't plan or push a sale on a decline turn.
    if situation and (situation.get("purchase_signal") or "none") == "declined":
        return False
    if _CONTENT_REQUEST_RE.search((message or "").lower()):
        return True
    if situation:
        if (situation.get("strategic_move") or "").lower() in {
            "push_for_ppv",
            "hint_at_content",
            "build_tension",
        }:
            return True
    return False


async def _scene_and_register(
    *,
    creator_id: str,
    fan_id: str,
    situation: dict,
    decision_context: dict | None,
    latest_message: str,
    fan_profile,
    active_session: dict | None = None,
) -> tuple[dict, dict]:
    """Advance the scene and decide the register, or degrade to neither.

    Wrapped because both are choreography: losing them costs continuity and a
    permissive register, and neither is worth failing a turn over. Degrading to
    ``({}, {})`` renders no SCENE and no TEXT INTIMACY block, which leaves the
    writer on its default voice — the safe direction, since the absent block is
    the one that would have GRANTED the explicit register.
    """
    try:
        return await _scene_and_register_inner(
            creator_id=creator_id,
            fan_id=fan_id,
            situation=situation,
            decision_context=decision_context,
            latest_message=latest_message,
            fan_profile=fan_profile,
            active_session=active_session,
        )
    except Exception as exc:
        print(f"[EXPERIENCE] scene/register unavailable fan={fan_id}: {exc}")
        return {}, {}


async def _scene_and_register_inner(
    *,
    creator_id: str,
    fan_id: str,
    situation: dict,
    decision_context: dict | None,
    latest_message: str,
    fan_profile,
    active_session: dict | None = None,
) -> tuple[dict, dict]:
    """Advance the scene, then decide how sexual this reply may be.

    The order matters and is the whole point of the split:

    * the OFFER GATE upstream reads the scene as it stood when policy decided,
      so a purchase cannot both create a scene and immediately be gated by it;
    * the REGISTER reads the scene after it advanced, because "he just unlocked
      this and is reacting to it" is exactly the state that should let the
      creator talk about it in its own register.

    Returns ``(scene_writer_context, text_intimacy_context)`` — two dicts that
    go straight onto ConversationContext. Neither can authorize a send.
    """
    decision_context = decision_context or {}
    set_id = (
        decision_context.get("accepted_offer_set_id")
        or ((decision_context.get("next_offer") or {}) or {}).get("set_id")
        or (active_session or {}).get("set_id")
    )
    scene_metadata = await scene_metadata_for(creator_id, set_id)
    scene = await direct_experience(
        creator_id=creator_id,
        fan_id=fan_id,
        situation=situation,
        commercial_decision=decision_context,
        latest_fan_message=latest_message,
        scene_metadata=scene_metadata,
    )

    policy = await get_creator_policy(creator_id)
    state = await get_fan_state(fan_id)
    intimacy = decide_text_intimacy(
        policy=policy,
        situation=situation,
        commercial_decision=decision_context,
        scene=scene.to_context(),
        fan_status=state.status,
        teaser_messages_used=state.teaser_messages_used,
        frozen_for_review=bool(getattr(fan_profile, "needs_human_review", False)),
        free_mode_on_cooldown=free_mode_on_cooldown(
            policy, state, datetime.now(timezone.utc)
        ),
    )
    if intimacy.consumes_free_allowance:
        try:
            await consume_free_text_allowance(creator_id, fan_id)
        except Exception as exc:
            # Failing to bill the allowance must not send an unbilled explicit
            # reply, so the register is dropped to flirty instead.
            print(f"[TEXT INTIMACY] allowance write failed fan={fan_id}: {exc}")
            intimacy = decide_text_intimacy(
                policy=policy,
                situation=situation,
                commercial_decision=decision_context,
                scene=scene.to_context(),
                fan_status=state.status,
                teaser_messages_used=max(
                    state.teaser_messages_used, policy.free_text_max_messages
                ),
                frozen_for_review=True,
            )
    print(
        f"[TEXT INTIMACY] fan={fan_id} level={intimacy.level.value} "
        f"reason={intimacy.reason}"
    )
    return scene.writer_context(), intimacy.to_context()


async def get_suggestions(
    fan_id: str,
    creator_id: str,
    fan_message: str,
    creator_name: str = "a creator",
    save_fan_message: bool = True,
) -> SuggestionResponse:
    (
        conversation_history,
        fan_profile,
        fan_intelligence,
        buyer_lifecycle,
        affordability,
        price_learning,
        creator_persona,
        creator_legend,
        ppv_offers,
        sent_ppv,
        active_session,
        carried_threads,
        past_episodes,
    ) = await asyncio.gather(
        get_conversation_history(fan_id),
        get_fan_by_id(fan_id),
        get_fan_intelligence_context(fan_id),
        get_fan_lifecycle_context(fan_id),
        get_affordability_context(fan_id),
        get_price_learning_context(fan_id),
        get_creator_persona(creator_id),
        get_creator_legend(creator_id),
        get_ppv_offers(creator_id),
        get_sent_ppv(fan_id),
        get_fan_session(fan_id),
        # Assisted reads the same continuity Full Auto does. Finding A was
        # exactly this kind of divergence — one mode loading evidence the other
        # did not — and the fix is the two modes sharing the load, not a second
        # copy of the logic here.
        open_threads_for(creator_id, fan_id),
        recent_episodes_for(creator_id, fan_id),
    )
    open_thread_lines = summarize_threads(carried_threads)
    episode_lines = [episode.render() for episode in past_episodes]
    if fan_profile is None:
        fan_profile = Fan(id=fan_id, display_name=fan_id)
    if creator_persona is None:
        creator_persona = Persona()

    conversation_stage = classify_stage(conversation_history, fan_profile)

    similar_exchanges = await find_similar_exchanges(
        fan_message, creator_id, enabled=False
    )

    # Which AI brain answers this turn. Resolved once, before the first model
    # call, so every stage of this turn — analyzer, writer, extractor — and the
    # metadata persisted with the reply all name the same profile.
    stack = await resolve_ai_stack(
        creator_id=creator_id,
        fan_id=fan_id,
        platform_fan_id=getattr(fan_profile, "platform_fan_id", None),
    )
    log_effective_stack(
        stack, creator_id=creator_id, fan_id=fan_id, feature="assisted_reply"
    )
    stack_profile = stack.profile

    ctx_without_situation = ConversationContext(
        fan_message=fan_message,
        open_threads=open_thread_lines,
        conversation_episodes=episode_lines,
        conversation_history=conversation_history,
        fan_profile=fan_profile,
        creator_persona=creator_persona,
        similar_exchanges=similar_exchanges,
        conversation_stage=conversation_stage,
        creator_name=creator_name,
        ppv_offers=ppv_offers,
        sent_ppv=sent_ppv,
        active_session=active_session,
        creator_legend=creator_legend,
        fan_intelligence=fan_intelligence,
        buyer_lifecycle=buyer_lifecycle,
        affordability=affordability,
        price_learning=price_learning,
        ai_stack_profile=stack.profile_id,
        writer_prompt_version=stack_profile.writer_prompt_version(),
    )

    situation = await analyze_situation(
        ctx_without_situation,
        telemetry_context={"creator_id": creator_id, "fan_id": fan_id},
        profile_id=stack.profile_id,
    )
    # REL-001 — Assisted keeps producing copy, because a human approves it before
    # anything reaches the fan, but the operator must be told the analysis behind
    # it was guessed rather than returned.
    assisted_degraded = analysis_is_degraded(situation)
    assisted_degraded_reason = degraded_reason(situation)
    if assisted_degraded:
        print(
            f"[ASSISTED ANALYZER DEGRADED] fan={fan_id} creator={creator_id} "
            f"reason={assisted_degraded_reason or 'unknown'}"
        )
    if fan_intelligence:
        situation["learned_fan_intelligence"] = fan_intelligence

    affordability = await refresh_affordability_from_situation(
        creator_id=creator_id,
        fan_id=fan_id,
        situation=situation,
        source_ref=f"assisted:{len(conversation_history)}:{fan_message}",
    )

    buyer_lifecycle = await refresh_fan_lifecycle(
        creator_id=creator_id,
        fan_id=fan_id,
        situation=situation,
        active_session=active_session,
        fan_profile=fan_profile,
        trigger_type="assisted_message",
    )

    price_learning = await refresh_price_learning(
        creator_id=creator_id,
        fan_id=fan_id,
        affordability=affordability,
        lifecycle=buyer_lifecycle,
        trigger_type="assisted_message",
    )
    situation["price_learning"] = price_learning

    # Auto-plan a session if the fan is asking for content and none is active,
    # and the creator's per-fan daily caps (if configured) aren't exceeded.
    if not active_session and not _selling_locked(fan_profile) and _fan_wants_content(fan_message, situation):
        cap_ok, cap_reason = await _within_daily_caps(creator_id, sent_ppv, fan_profile)
        if not cap_ok:
            print(f"[CAP] fan={fan_id} plan suppressed: {cap_reason}")
        else:
            try:
                plan_data = await plan_session_for_fan(creator_id, fan_id)
                if plan_data.get("status") == "ok":
                    active_session = plan_data.get("session") or await get_fan_session(fan_id)
                    print(f"[SESSION] Auto-planned session for fan={fan_id} items={len((active_session or {}).get('plan', []))}")
                else:
                    print(f"[SESSION] plan-session returned status={plan_data.get('status')} fan={fan_id}")
            except Exception as e:
                print(f"[SESSION PLAN ERROR] {e}")

    scene_context, text_intimacy_context = await _scene_and_register(
        creator_id=creator_id,
        fan_id=fan_id,
        situation=situation,
        decision_context=None,
        latest_message=fan_message,
        fan_profile=fan_profile,
        active_session=active_session,
    )

    conversation_director = await direct_conversation(
        creator_id=creator_id,
        fan_id=fan_id,
        conversation_history=conversation_history,
        latest_fan_message=fan_message,
        situation=situation,
        lifecycle=buyer_lifecycle,
        active_session=active_session,
        conversation_stage=conversation_stage.value,
        trigger_type="assisted_message",
    )
    situation["conversation_director"] = conversation_director

    session_strategy = await plan_next_action(
        creator_id=creator_id,
        fan_id=fan_id,
        situation=situation,
        lifecycle=buyer_lifecycle,
        affordability=affordability,
        price_learning=price_learning,
        active_session=active_session,
        conversation_stage=conversation_stage.value,
        conversation_director=conversation_director,
        trigger_type="assisted_message",
    )
    situation["session_strategy"] = session_strategy

    # The assisted path has no commercial decision to carry the inventory, so
    # it reads the asset types directly. A human approves these candidates, but
    # a candidate that promises a clip the creator does not have is still a
    # candidate somebody can send by accident.
    try:
        approved_asset_types = await get_approved_asset_types(creator_id)
    except Exception as exc:
        print(f"[INVENTORY] asset type read failed creator={creator_id}: {exc}")
        approved_asset_types = ()
    media_inventory = _build_turn_inventory(
        decision=None,
        active_session=active_session,
        situation=situation,
        fan_message=fan_message,
        vault_asset_types=approved_asset_types,
    )

    ctx = ConversationContext(
        fan_message=fan_message,
        conversation_history=conversation_history,
        fan_profile=fan_profile,
        creator_persona=creator_persona,
        similar_exchanges=similar_exchanges,
        conversation_stage=conversation_stage,
        creator_name=creator_name,
        situation=situation,
        ppv_offers=ppv_offers,
        sent_ppv=sent_ppv,
        active_session=active_session,
        media_inventory=media_inventory.to_context(),
        creator_legend=creator_legend,
        fan_intelligence=fan_intelligence,
        buyer_lifecycle=buyer_lifecycle,
        affordability=affordability,
        price_learning=price_learning,
        session_strategy=session_strategy,
        conversation_director=conversation_director,
        scene=scene_context,
        text_intimacy=text_intimacy_context,
        open_threads=open_thread_lines,
        conversation_episodes=episode_lines,
        ai_stack_profile=stack.profile_id,
        writer_prompt_version=stack_profile.writer_prompt_version(),
    )

    route = select_writer_route(ctx, profile_id=stack.profile_id)
    print(
        f"[WRITER ROUTE] fan={fan_id} mode=assisted profile={stack.profile_id} "
        f"route={route.route.value} "
        f"reason={route.reason} primary={route.primary_target.model} "
        f"fallback={(route.fallback_target.model if route.fallback_target else 'none')}"
    )
    # Assisted asks for a list because a human picks from one. That is a
    # property of this path, not of the writer version: the same version asked
    # for one reply on the Auto path below.
    prompt = build_prompt(
        ctx, prompt_version=route.prompt_version, reply_mode=MODE_ASSISTED
    )
    assisted_trace = GenerationTrace()
    replies = await generate_replies(
        prompt,
        creator_persona,
        trace=assisted_trace,
        max_candidates=writer_candidate_count(route.prompt_version, MODE_ASSISTED),
        output_contract=CONTRACT_CANDIDATES,
        retry_policy=writer_retry_policy(route.prompt_version),
        profile_id=stack.profile_id,
        telemetry_context={
            "creator_id": creator_id,
            "fan_id": fan_id,
            "feature": "assisted_reply",
            **route.telemetry_metadata(),
            "buyer_lifecycle_stage": buyer_lifecycle.get("stage"),
            "price_learning_mode": price_learning.get("mode"),
            "price_learning_confidence": price_learning.get("confidence"),
            "session_strategy_goal": session_strategy.get("goal"),
            "session_strategy_action": session_strategy.get("next_action"),
            "conversation_director_phase": conversation_director.get("phase"),
            "conversation_director_action": conversation_director.get("action"),
            "conversation_director_reason": conversation_director.get("transition_reason"),
        },
        target_override=route.primary_target,
        fallback_target_override=route.fallback_target,
    )

    # Assisted candidates keep their own natural shapes — a human picks one —
    # but the platform's delivery semantics still hold for all of them.
    raw_candidates = list(replies)
    replies = sanitize_candidates(replies, active_session=active_session)
    # A human approves an assisted candidate, so a promise of media that does
    # not exist is repaired rather than dropped: the operator still sees a
    # usable option, and it is one the creator can actually keep.
    replies = [
        sanitize_media_promises(
            candidate,
            media_inventory,
            active_session=active_session,
        )[0]
        or candidate
        for candidate in replies
    ]

    # Assisted's ground truth is recorded here but finished later: an operator
    # is between generation and delivery, so the record waits for the send that
    # redeems its token (services/reply_provenance.py). Candidates the operator
    # does not pick never become a message and are never attributed.
    assisted_provenance = ReplyProvenance(
        creator_id=str(creator_id), fan_id=str(fan_id), mode=PIPELINE_ASSISTED
    )
    assisted_provenance.record_trigger(
        kind="fan_message",
        text=fan_message,
        history_position=len(conversation_history) - 1,
    )
    assisted_provenance.record_context(
        history_messages=len(conversation_history),
        analyzer_window=min(len(conversation_history), ANALYZER_TRANSCRIPT_MESSAGES),
        writer_window=min(len(conversation_history), WRITER_TRANSCRIPT_MESSAGES),
        packet=build_context_packet(
            conversation_history,
            open_threads=open_thread_lines,
            episodes=episode_lines,
        ).fingerprint(),
        stack_profile=stack.profile_id,
        writer_prompt_version=stack_profile.writer_prompt_version(),
        live_state={
            "creator_legend": bool(creator_legend),
            "fan_intelligence": bool(fan_intelligence),
            "buyer_lifecycle": bool(buyer_lifecycle),
            "affordability": bool(affordability),
            "price_learning": bool(price_learning),
            "session_strategy": bool(session_strategy),
            "conversation_director": bool(conversation_director),
            "experience_scene": bool(scene_context),
            "text_intimacy": bool(text_intimacy_context),
            # Assisted runs no commercial orchestrator: a human decides whether
            # to sell. Recording the key as absent rather than false would let a
            # comparison read it as "the orchestrator declined".
            "active_session": bool(active_session),
            "media_inventory": bool(media_inventory.authorized_asset_types),
        },
    )
    assisted_provenance.record_decision(
        source="assisted_operator",
        purchase_signal=situation.get("purchase_signal"),
        crisis_signal=situation.get("crisis_signal"),
        resend_requested=situation.get("resend_requested"),
        extra={"strategic_move": situation.get("strategic_move")},
    )
    assisted_provenance.record_writer(assisted_trace)
    assisted_provenance.record_transform(
        TRANSFORM_DELIVERY_LANGUAGE, raw_candidates != replies
    )
    # Stored durably as well as in process. The in-process store is one per
    # replica, so a deploy or an autoscale event between generating and sending
    # used to lose the record and leave the reply unattributable with nothing
    # saying so (services/assisted_provenance.py).
    from services.assisted_provenance import remember as _remember_provenance

    suggestion_token = await _remember_provenance(assisted_provenance)
    print(assisted_trace.describe())

    if save_fan_message:
        evidence_message_id = await save_message(
            fan_id, creator_id, "fan", fan_message
        )
        extraction_history = [
            *conversation_history,
            Message(role="fan", content=fan_message),
        ]
        spawn(
            learn_from_fan_message(
                creator_id=creator_id,
                fan_id=fan_id,
                fan_message=fan_message,
                source_message_id=evidence_message_id,
                conversation_history=extraction_history,
                profile_id=stack.profile_id,
            ),
            name=f"fan_intelligence:{fan_id}",
        )

    if _should_update_memory(conversation_history):
        spawn(_update_fan_memory(fan_id, creator_id, conversation_history, fan_profile.total_spent), name="update_fan_memory")
        spawn(
            _update_fan_ai_summary(
                fan_id, conversation_history, profile_id=stack.profile_id
            ),
            name="update_fan_ai_summary",
        )

    return SuggestionResponse(
        suggestions=replies,
        stage=conversation_stage,
        analysis_degraded=assisted_degraded,
        analysis_degraded_reason=assisted_degraded_reason,
        suggestion_token=suggestion_token,
    )


def _render_legend(legend: dict) -> str:
    """Human-readable rendering of the canonical creator legend for the UI note."""
    if not legend:
        return ""
    labels = [
        ("name", "Name"),
        ("origin", "From"),
        ("age", "Age"),
        ("job", "Job"),
        ("background", "Background"),
    ]
    lines = [f"{label}: {legend[key]}" for key, label in labels if (legend.get(key) or "").strip()]
    other = legend.get("other") or []
    if isinstance(other, list) and other:
        lines.append("Other: " + "; ".join(other))
    return "\n".join(lines)


async def _update_fan_memory(
    fan_id: str,
    creator_id: str,
    conversation_history: list[Message],
    fan_total_spent: int,
) -> None:
    try:
        recent_messages = conversation_history[-30:]
        convo_lines: list[str] = []
        for msg in recent_messages:
            speaker = "Fan" if msg.role == "fan" else "Creator"
            convo_lines.append(f"{speaker}: {msg.content}")
        convo_text = "\n".join(convo_lines)

        system_prompt = (
            "You are a fan CRM analyst for an OnlyFans agency. "
            "Extract structured notes from conversations exactly as an experienced chatter would write them. "
            "Return only valid JSON, no markdown, no explanation."
        )
        user_prompt = (
            "Analyze this conversation and return a JSON object with exactly these fields:\n"
            "{\n"
            '  "notes": "2-3 sentence internal summary of key facts about this fan",\n'
            '  "preferences": ["ONLY stable content preferences the FAN clearly wants from the creator, stated or repeated. NOT: things he says about his own body, one-off dirty talk, or anything merely mentioned in passing. Empty list is better than guessing."],\n'
            '  "member_note": "Fill in the Member template below with what you know. Leave fields blank if unknown.\\n'
            'Age: \nLocation: \nInterests/hobbies: \nKinks: \nAdditional info: ",\n'
            '  "model_facts": {\n'
            '      "name": "the name the creator goes by, if stated (else empty)",\n'
            '      "origin": "where the creator said she is from, if stated (else empty)",\n'
            '      "age": "the creator\'s age if she stated it (else empty)",\n'
            '      "job": "the creator\'s job/what she does, if stated (else empty)",\n'
            '      "background": "any backstory the creator told about herself (else empty)",\n'
            '      "other": ["any other concrete personal facts the CREATOR stated about herself"]\n'
            "  }\n"
            "}\n\n"
            "For model_facts, ONLY include facts the creator (not the fan) actually stated about "
            "HERSELF in this conversation. Leave a field empty if she did not state it. Do not guess.\n\n"
            "SPEAKER ATTRIBUTION IS CRITICAL — read the line labels. Lines starting 'Creator:' are "
            "the creator speaking; lines starting 'Fan:' are the fan. Facts from 'Fan:' lines NEVER "
            "go into model_facts, no matter what they are. Example of the mistake to avoid: if the "
            "FAN says 'im living in miami', that is the FAN's location — it belongs in member_note, "
            "and model_facts.origin stays empty. model_facts.origin is filled ONLY by a 'Creator:' "
            "line like 'Creator: I'm a California girl'. When in doubt, leave the field empty.\n\n"
            "Conversation:\n"
            f"{convo_text}"
        )

        response = await together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )

        content = response.choices[0].message.content or ""
        lines = content.splitlines()
        cleaned = "\n".join(
            line for line in lines if not line.lstrip().startswith("```")
        ).strip()

        data = json.loads(cleaned)
        notes = data.get("notes", "")
        preferences = data.get("preferences") or []
        member_note = data.get("member_note", "")
        model_facts = data.get("model_facts") or {}
        if not isinstance(model_facts, dict):
            model_facts = {}

        # Merge creator self-facts into the canonical per-creator legend (first-wins).
        legend = {}
        if model_facts:
            try:
                legend = await update_creator_legend(creator_id, model_facts)
            except Exception as e:
                print(f"[LEGEND ERROR] creator={creator_id} error={e}")

        # Render the canonical legend to readable text for the operator's MODEL LEGEND box.
        model_note = _render_legend(legend)

        if not isinstance(preferences, list):
            preferences = []

        actual_tier = "cold"
        if fan_total_spent >= 500:
            actual_tier = "whale"
        elif fan_total_spent >= 100:
            actual_tier = "active"
        elif fan_total_spent > 0:
            actual_tier = "casual"

        await update_fan_memory(
            fan_id=fan_id,
            notes=notes,
            preferences=preferences,
            spend_tier=actual_tier,
            member_note=member_note,
            model_note=model_note,
        )
    except Exception as e:
        print(f"[MEMORY ERROR] fan={fan_id} error={e}")
        return


async def _update_fan_ai_summary(
    fan_id: str,
    conversation_history: list[Message],
    *,
    profile_id: str | None = None,
) -> None:
    try:
        summary_spec = get_profile(profile_id).stages.get(STAGE_FAN_SUMMARY)
        convo_lines = []
        for msg in conversation_history[-20:]:
            speaker = "Fan" if msg.role == "fan" else "Creator"
            convo_lines.append(f"{speaker}: {msg.content}")
        convo_text = "\n".join(convo_lines)

        system_prompt = (
            "You are an expert fan relationship analyst for an OnlyFans agency. "
            "Analyze this conversation and extract a detailed psychological and behavioral profile of the fan. "
            "Return only valid JSON, no markdown, no explanation."
        )
        user_prompt = (
            "Analyze this conversation and return a JSON object with these fields:\n"
            "{\n"
            '  "real_name": "their real name if mentioned, otherwise null",\n'
            '  "age": "their age if mentioned or clearly stated, otherwise null",\n'
            '  "location": "city/country if mentioned, otherwise null",\n'
            '  "occupation": "job or income signals if mentioned, otherwise null",\n'
            '  "hobbies": "their hobbies or interests if mentioned, otherwise null",\n'
            '  "relationship_status": "single/relationship/married/unknown",\n'
            '  "payday": "when they get paid, if mentioned in ANY form (e.g. paycheck next week, payday is the 1st, broke till Friday), otherwise null",\n'
            '  "kinks": ["ONLY kinks/preferences the FAN clearly and repeatedly expresses wanting from the creator. NOT descriptions of himself or his anatomy, NOT one-off dirty-talk phrases, NOT topics merely touched on once. Fewer, higher-confidence entries beat a keyword dump."],\n'
            '  "emotional_type": "one of: romantic | submissive | dominant | transactional | playful | mixed",\n'
            '  "spending_behavior": "description of how they spend — e.g. tips spontaneously, haggles on price, pays without hesitation",\n'
            '  "best_time_to_message": "time of day or days they seem most active, or null",\n'
            '  "reengagement_triggers": "what topics or messages get them most responsive",\n'
            '  "risk_signals": "any red flags like money problems, about to cancel, frustration — or null",\n'
            '  "summary": "3-4 sentence psychological profile of this fan — who they are, what they want, how to handle them"\n'
            "}\n\n"
            "Conversation:\n"
            f"{convo_text}"
        )

        # The model, temperature and budget come from the turn's AI Stack
        # Profile rather than being hardcoded here, so the profile detail view
        # the owner reads is the truth about this stage too. Both shipped
        # profiles name the same Together model; nothing about this pass gives a
        # reason to re-point a background summariser.
        response = await together_client.chat.completions.create(
            model=(
                summary_spec.resolved_primary()[1]
                if summary_spec
                else "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            ),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=(
                summary_spec.temperature
                if summary_spec and summary_spec.temperature is not None
                else 0.3
            ),
            max_tokens=summary_spec.resolved_max_tokens() if summary_spec else 1000,
        )

        content = response.choices[0].message.content or ""
        lines = content.splitlines()
        cleaned = "\n".join(
            line for line in lines if not line.lstrip().startswith("```")
        ).strip()

        data = json.loads(cleaned)
        await update_fan_ai_summary(fan_id=fan_id, summary=data)

    except Exception as e:
        print(f"[AI SUMMARY ERROR] fan={fan_id} error={e}")
        import traceback
        traceback.print_exc()


def _build_turn_inventory(
    *,
    decision,
    active_session: dict | None,
    situation: dict,
    fan_message: str,
    vault_asset_types: tuple[str, ...] = (),
) -> MediaInventory:
    """Assemble the turn's authoritative media capabilities.

    The commercial decision already carries the asset types of the rows the
    orchestrator loaded, so this reads no database and cannot disagree with the
    planner. Without a decision — Commercial v2 off, or an assisted turn — the
    live session plan is the authority and anything it does not contain is not
    promisable.
    """
    from services.media_packages import wants_video

    session_types = asset_types_from_session(active_session)
    desired = str(situation.get("desired_experience") or "")
    video_requested = bool(wants_video(desired) or wants_video(fan_message))

    if decision is None:
        # An active plan is the narrowest authority; otherwise the approved
        # vault is. Only with neither is the inventory genuinely unknown, and an
        # unknown inventory states nothing rather than guessing.
        authorized = session_types or tuple(vault_asset_types)
        return MediaInventory(
            authorized_asset_types=authorized,
            vault_asset_types=tuple(vault_asset_types) or session_types,
            next_step_asset_type=next_step_asset_type(active_session),
            video_requested=video_requested,
            known=bool(authorized),
            reason_codes=(
                ("authorized_from_active_session",)
                if session_types
                else ("authorized_from_approved_vault",)
            )
            if authorized
            else ("inventory_unknown",),
        )

    authorized = tuple(getattr(decision, "authorized_asset_types", None) or ())
    vault = tuple(getattr(decision, "vault_asset_types", None) or ())
    # An active plan is narrower than the offer snapshot: it names the exact
    # step that will be delivered, so it wins.
    if session_types:
        authorized = session_types
    return MediaInventory(
        authorized_asset_types=authorized,
        vault_asset_types=vault,
        next_step_asset_type=next_step_asset_type(active_session),
        video_requested=video_requested,
        known=True,
        reason_codes=("authorized_from_commercial_decision",),
    )


async def _recover_or_downgrade_plan(
    *,
    creator_id: str,
    fan_id: str,
    status: str,
    decision,
    situation: dict,
    price_learning: dict,
) -> tuple[dict | None, object, str | None]:
    """Turn a failed plan into a real turn, or into an honest failure.

    Returns ``(active_session, decision, abort_outcome)``. An ``abort_outcome``
    that is not ``None`` means the turn genuinely cannot proceed and the caller
    should stop — with that outcome recorded, never as a bare "sent nothing".

    Three things can come back:

    * a replanned session, so the decided sale proceeds against current
      inventory;
    * a downgraded decision that PRESENTS the valid replacement, used when the
      fan had already accepted an exact package that can no longer be delivered
      — he is told and asked, never silently charged for a substitute;
    * a downgraded decision to continue the conversation without an offer, used
      when there is genuinely nothing left to sell him.
    """
    from services.session_plan_recovery import (
        classify_plan_status,
        recover_session_plan,
    )

    failure_class = classify_plan_status(status)
    had_contract = bool(
        decision is not None and getattr(decision, "accepted_offer_set_id", None)
    )
    print(
        f"[SESSION] plan-session status={status} fan={fan_id} "
        f"class={failure_class.value} accepted_contract={had_contract}"
    )

    try:
        recovery = await recover_session_plan(
            creator_id=creator_id,
            fan_id=fan_id,
            status=status,
            had_accepted_contract=had_contract,
            desired_experience=str(situation.get("desired_experience") or "") or None,
            price_learning=price_learning,
        )
    except Exception as exc:
        print(f"[SESSION RECOVERY ERROR] fan={fan_id} status={status} error={exc}")
        return None, decision, AUTO_OUTCOME_PLAN_UNRECOVERABLE

    print(
        f"[SESSION RECOVERY] fan={fan_id} from={status} "
        f"outcome={recovery.reason} recovered={recovery.recovered} "
        f"present_replacement={recovery.present_replacement} "
        f"continue_without_offer={recovery.continue_without_offer}"
    )

    if recovery.recovered:
        return (
            recovery.session or await get_fan_session(fan_id),
            decision,
            None,
        )

    if decision is None:
        return None, decision, AUTO_OUTCOME_PLAN_UNRECOVERABLE

    if recovery.present_replacement:
        replacement = recovery.replacement_offer
        decision.action = ActionType.OFFER_NEXT_UNLOCK
        decision.next_offer = replacement
        decision.must_not_send_media = True
        decision.mention_price = (
            replacement.price_cents // 100 if replacement is not None else None
        )
        decision.session_budget_cents = None
        decision.accepted_offer_set_id = None
        decision.replacement_for_unavailable = True
        decision.new_status = FanStatus.OFFER_PENDING
        decision.goal = (
            "The exact thing he picked is no longer available. Acknowledge that "
            "naturally in your own voice — no systems, inventory or approval "
            "talk — and offer what is actually there now."
        )
        decision.reason = f"plan_recovery:{recovery.reason}"
        from services.inventory_authority import asset_types_from_offer

        decision.authorized_asset_types = list(asset_types_from_offer(replacement))
        return None, decision, None

    if recovery.continue_without_offer:
        decision.action = ActionType.CONTINUE_NORMAL_CHAT
        decision.next_offer = None
        decision.must_not_send_media = True
        decision.mention_price = None
        decision.session_budget_cents = None
        decision.accepted_offer_set_id = None
        decision.authorized_asset_types = []
        decision.new_status = FanStatus.IDLE
        decision.goal = (
            "Keep the conversation warm and going. Do not offer, promise or "
            "tease any paid content this turn."
        )
        decision.reason = f"plan_recovery:{recovery.reason}"
        return None, decision, None

    return None, decision, AUTO_OUTCOME_PLAN_UNRECOVERABLE


async def _debounced_auto_reply(
    fan_id: str,
    creator_id: str,
    *,
    skip_debounce: bool = False,
    skip_availability: bool = False,
    skip_human_delays: bool = False,
    expected_trigger_at: str | None = None,
    outcome_sink: dict[str, str] | None = None,
) -> None:
    """Wait for fan to finish typing, then generate and send one reply.

    ``skip_human_delays`` removes the composition and inter-part pauses. It is
    used only by the owner-only simulator, where the wait is the one thing that
    is NOT interesting: everything else — analyzer, commercial orchestrator,
    session planning, conversation director, writer routing, multipart selection
    — runs exactly as it does in production. The staleness checks those pauses
    also perform are kept; only their duration is removed.

    ``outcome_sink`` is an optional mapping the caller owns, into which this
    function records WHY the turn produced nothing when the reason is not
    visible from the outside. Only the owner-only simulator passes one; the
    durable AUTO_REPLY worker and every other caller pass nothing and behave
    exactly as before.

    It exists because "no creator message" is three different events — the turn
    deliberately said nothing, the analyzer failed closed, or the writer stack
    failed — and the simulator was reporting all three as a decision. This is
    deliberately a sink rather than a return value: the many early returns in
    this function all mean "no send", and only the writer branch needs to say
    something more specific.
    """
    try:
        # Short jittered debounce catches rapid multi-message bursts. A separate
        # length-aware delay is applied after generation to simulate reading/typing.
        if not skip_debounce:
            delay = random.uniform(7.0, 12.0)
            await asyncio.sleep(delay)

        situation: dict | None = None  # initialized early — assigned properly later

        # Fetch history now — after the wait — to get the most complete picture
        # including any messages the fan sent while we were waiting
        conversation_history = await get_conversation_history(fan_id)

        if expected_trigger_at:
            from datetime import datetime, timezone

            trigger_at = datetime.fromisoformat(
                expected_trigger_at.replace("Z", "+00:00")
            )
            if trigger_at.tzinfo is None:
                trigger_at = trigger_at.replace(tzinfo=timezone.utc)
            newer = [
                message
                for message in conversation_history
                if message.sent_at and message.sent_at > trigger_at
            ]
            if newer:
                print(
                    f"[AUTO REPLY] durable trigger obsolete fan={fan_id} "
                    f"newer_messages={len(newer)}"
                )
                return

        # Stale generation check: if another task is already pending for this fan
        # (meaning a newer message came in during our wait and reset the timer),
        # abort silently — the newer task will generate the reply
        current_task = _pending_auto_replies.get(fan_id)
        if current_task and current_task is not asyncio.current_task():
            print(f"[AUTO REPLY] Newer task exists — aborting stale generation for fan={fan_id}")
            return
        (
            fan_profile,
            fan_intelligence,
            buyer_lifecycle,
            affordability,
            price_learning,
        ) = await asyncio.gather(
            get_fan_by_id(fan_id),
            get_fan_intelligence_context(fan_id),
            get_fan_lifecycle_context(fan_id),
            get_affordability_context(fan_id),
            get_price_learning_context(fan_id),
        )
        if not fan_profile:
            return

        # The local-test boundary is established HERE, not at delivery time.
        #
        # It used to be computed just before the send, which left several paths
        # above it able to reach the provider for a test fan: the chat-list
        # lookup that resolves a missing group id, the typing indicator, and the
        # eager purchase verification spawned on a "bought" signal. Each is
        # gated on this flag. Access complaints now stop before delivery on
        # both simulated and live routes.
        local_test_delivery = _is_local_test_fan(
            getattr(fan_profile, "platform_fan_id", None)
        )

        # Frozen for human review (e.g. prior crisis under 'freeze' policy): auto-mode
        # stays out until a human clears the flag in the dashboard.
        if getattr(fan_profile, "needs_human_review", False):
            if outcome_sink is not None:
                outcome_sink["outcome"] = AUTO_OUTCOME_HUMAN_REVIEW
            print(f"[AUTO REPLY] fan={fan_id} is frozen for human review — skipping auto-reply")
            return
        # Check for a pending tip and clear it atomically before building context
        pending_tip: dict | None = None
        try:
            tip_row = await asyncio.to_thread(
                lambda: get_supabase()
                .table("fans")
                .select("pending_tip")
                .eq("id", fan_id)
                .single()
                .execute()
            )
            pending_tip = (tip_row.data or {}).get("pending_tip")
            if pending_tip:
                await asyncio.to_thread(
                    lambda: get_supabase()
                    .table("fans")
                    .update({"pending_tip": None})
                    .eq("id", fan_id)
                    .execute()
                )
                print(f"[TIP ACK] Cleared pending_tip for fan={fan_id} amount=${pending_tip.get('amount')}")
        except Exception as e:
            print(f"[TIP ACK ERROR] {e}")

        fan_messages = [m for m in conversation_history if m.role == "fan"]
        if not fan_messages:
            return
        latest_message = fan_messages[-1].content

        # Ground truth for this turn starts here, at the event that caused it
        # (docs/autonomy_architecture_review.md §6 step 1). The recorder is
        # filled in as the turn makes its decisions and emitted onto each
        # delivered message. It is a record only: nothing below reads it, and a
        # turn that returns early simply never emits one.
        provenance = ReplyProvenance(
            creator_id=str(creator_id), fan_id=str(fan_id), mode=PIPELINE_AUTO
        )
        provenance.record_trigger(
            kind="fan_message",
            text=latest_message,
            sent_at=fan_messages[-1].sent_at,
            history_position=len(conversation_history) - 1,
        )

        (
            creator_persona,
            creator_legend,
            ppv_offers,
            sent_ppv,
            active_session,
            similar_exchanges,
            carried_threads,
            past_episodes,
        ) = await asyncio.gather(
            get_creator_persona(creator_id),
            get_creator_legend(creator_id),
            get_ppv_offers(creator_id),
            get_sent_ppv(fan_id),
            get_fan_session(fan_id),
            find_similar_exchanges(latest_message, creator_id, enabled=False),
            # What this conversation is still carrying, and what earlier
            # stretches of it were about. Both get their own allowance in the
            # context packet, so an unanswered question cannot be evicted by
            # recent chatter (docs/autonomy_architecture_review.md §4).
            open_threads_for(creator_id, fan_id),
            recent_episodes_for(creator_id, fan_id),
        )
        open_thread_lines = summarize_threads(carried_threads)
        episode_lines = [episode.render() for episode in past_episodes]

        # If this turn is a return after a silence, the stretch before it is
        # now knowably over and gets closed. Nothing wrote to
        # conversation_episodes before this, so "we talked about this before"
        # was a table shape rather than something the system could say.
        #
        # Its subjects come from the obligations that conversation raised —
        # rows with source turn ids behind them — rather than from a model
        # asked to summarise, because this record is read back to a model
        # later and an invented one would launder a belief into a fact.
        closed = await close_finished_episode(
            creator_id=creator_id,
            fan_id=fan_id,
            history=conversation_history,
            subjects=[thread.summary for thread in carried_threads],
        )
        if closed is not None:
            print(
                f"[CONTINUITY] fan={fan_id} episode_closed "
                f"messages={closed.message_count} ending={closed.ended_with.value}"
            )
            episode_lines = [closed.render(), *episode_lines]

        if creator_persona is None:
            creator_persona = Persona()

        conversation_stage = classify_stage(conversation_history, fan_profile)

        # Which AI brain answers this turn. Resolved once, before the first
        # model call: the analyzer, the writer and the extractor below must all
        # be the same profile, and so must the metadata persisted with the
        # reply. A simulation fan's own override wins here, which is what lets
        # two test fans under one creator be compared turn for turn.
        stack = await resolve_ai_stack(
            creator_id=creator_id,
            fan_id=fan_id,
            platform_fan_id=getattr(fan_profile, "platform_fan_id", None),
        )
        log_effective_stack(
            stack, creator_id=creator_id, fan_id=fan_id, feature="full_auto"
        )
        stack_profile = stack.profile
        writer_prompt_version = stack_profile.writer_prompt_version()

        ctx_without_situation = ConversationContext(
            fan_message=latest_message,
            conversation_history=conversation_history,
            fan_profile=fan_profile,
            creator_persona=creator_persona,
            similar_exchanges=similar_exchanges,
            conversation_stage=conversation_stage,
            creator_name="a creator",
            creator_legend=creator_legend,
            ppv_offers=ppv_offers,
            sent_ppv=sent_ppv,
            active_session=active_session,
            fan_intelligence=fan_intelligence,
            buyer_lifecycle=buyer_lifecycle,
            affordability=affordability,
            price_learning=price_learning,
            open_threads=open_thread_lines,
            conversation_episodes=episode_lines,
            ai_stack_profile=stack.profile_id,
            writer_prompt_version=writer_prompt_version,
        )

        with action_stage("analyzer_ms"):
            situation = await analyze_situation(
                ctx_without_situation,
                telemetry_context={"creator_id": creator_id, "fan_id": fan_id},
                profile_id=stack.profile_id,
            )

        # REL-001 — Full Auto fails closed on a degraded analysis.
        #
        # A fabricated analysis is uniformly neutral: purchase_signal "none",
        # crisis_signal "none", resend_requested "false". Acting on it means a
        # fan saying "I'll take the $50 one" is read as small talk, decline locks
        # are set and cleared from guessed state, and a PPV may be resent. None
        # of that is recoverable after the message leaves.
        #
        # The stop is placed before refresh_affordability_from_situation and
        # refresh_price_learning: both WRITE commercial state derived from the
        # situation, so running them would persist the guess even though nothing
        # is sent.
        if analysis_is_degraded(situation):
            reason = degraded_reason(situation) or "unknown"
            # The deterministic self-harm backstop still applies — a failed
            # analyzer must not weaken crisis handling. _crisis_freezes_chat only
            # fires on a positive signal, which in a degraded result can only
            # have come from the regex in situation_analyzer.
            if await _crisis_freezes_chat(creator_id, fan_id, situation):
                return
            print(
                f"[AUTO ANALYZER DEGRADED] fan={fan_id} creator={creator_id} "
                f"reason={reason} — sending nothing"
            )
            raise AnalyzerDegradedError(
                f"situation analysis degraded ({reason}); Full Auto sent nothing"
            )

        # Crisis handling retains priority over a content-access complaint.
        if await _crisis_freezes_chat(creator_id, fan_id, situation):
            return

        # A request to fix an existing delivery is not authorization for a new
        # paid message. This must precede commercial state writes and planning,
        # including when the complaint also contains a purchase signal. No
        # pending PPV is required: a paid item can still be inaccessible.
        # Use the existing operator review workflow until entitlement/access
        # repair can be verified. Never promise a repair that has not happened.
        if str(situation.get("resend_requested", "false")).strip().lower() == "true":
            try:
                await freeze_fan_for_review(fan_id, CONTENT_ACCESS_REVIEW_REASON)
            except Exception as exc:
                raise HumanReviewHandoffError(
                    "could not persist content-access review hold; no reply sent"
                ) from exc
            # The complaint becomes an obligation the conversation carries,
            # not just a flag on a row. Without it, an operator resolving the
            # hold clears the freeze and the next turn has no idea anything was
            # ever wrong — which is how a customer gets sold to immediately
            # after reporting they cannot open what they bought. Recorded after
            # the hold, because a hold that failed to persist already raised.
            await record_open_thread(
                OpenThread(
                    creator_id=str(creator_id),
                    fan_id=str(fan_id),
                    kind=ThreadKind.COMPLAINT,
                    raised_by=ThreadParty.FAN,
                    summary="he says he cannot access content he paid for",
                    resolution_condition="he confirms he can open it",
                    evidence_type=EvidenceType.STATED,
                    source_message_fingerprint=fingerprint(latest_message),
                    source_turn_id=provenance.turn_id,
                )
            )
            if outcome_sink is not None:
                outcome_sink["outcome"] = AUTO_OUTCOME_HUMAN_REVIEW
            print(f"[AUTO SUPPORT] fan={fan_id} reason=content_access_issue review_required=true")
            return

        if fan_intelligence:
            situation["learned_fan_intelligence"] = fan_intelligence
        affordability = await refresh_affordability_from_situation(
            creator_id=creator_id,
            fan_id=fan_id,
            situation=situation,
            source_ref=f"auto:{len(conversation_history)}:{latest_message}",
        )
        price_learning = await refresh_price_learning(
            creator_id=creator_id,
            fan_id=fan_id,
            affordability=affordability,
            lifecycle=buyer_lifecycle,
            trigger_type="auto_message_pre_policy",
        )
        situation["price_learning"] = price_learning
        print(f"[SITUATION] fan={fan_id} signal={situation.get('purchase_signal')} move={situation.get('strategic_move')} resend={situation.get('resend_requested')} crisis={situation.get('crisis_signal', 'none')}")

        # What this exchange leaves the conversation carrying.
        #
        # Until now the only thing that ever became an open thread was a
        # content-access complaint, so an ordinary unanswered question, a
        # promise, a deferred topic and a correction had no durable lifecycle
        # at all — the tables existed and nothing wrote to them.
        #
        # Deliberately after the situation is complete and before anything
        # decides or sends: this records what happened, and nothing reads it
        # back to choose this turn's reply. A recorder that could change the
        # reply is a recorder that can break one.
        await _record_conversation_threads(
            situation,
            creator_id=creator_id,
            fan_id=fan_id,
            latest_message=latest_message,
            turn_id=provenance.turn_id,
        )

        # Commercial layer: deterministic policy decides what happens next
        # (sell / pause / tease / schedule). Flag-gated so it can be turned off
        # instantly in prod without a deploy.
        commercial_enabled = os.environ.get("COMMERCIAL_LAYER_ENABLED", "").lower() in ("1", "true", "yes")
        decision = None
        if commercial_enabled:
            try:
                cap_ok, _ = await _within_daily_caps(creator_id, sent_ppv, fan_profile)
                # The scene AS IT STANDS, before this turn advances it. Policy
                # asks it whether the conversation has earned a NEW offer yet;
                # advancing first would let the same turn both create the
                # post-unlock state and be gated by it.
                scene_before = await load_scene(fan_id)
                decision = await orchestrate(
                    creator_id=creator_id,
                    fan_id=fan_id,
                    situation=situation,
                    fan_has_bought_before=bool(getattr(fan_profile, "total_spent", 0)),
                    within_daily_caps=cap_ok,
                    frozen_for_review=bool(getattr(fan_profile, "needs_human_review", False)),
                    active_session=active_session,
                    scene=scene_before.to_context(),
                )
            except Exception as e:
                # Full Auto must fail closed. Silently reverting to the legacy
                # planner can send content after a pause or at the wrong price.
                print(f"[COMMERCIAL ERROR] fan={fan_id}: {e} — auto reply aborted")
                return

        # With commercial v2 enabled, only an acceptance may start a plan.
        # OFFER_NEXT_UNLOCK, PAUSE_* and ordinary chat must never invoke the
        # content planner. When the flag is off, preserve old behavior.
        should_plan = (
            decision is not None
            and decision.action == ActionType.SEND_NEXT_PPV_STEP
            and bool(decision.accepted_offer_set_id)
        ) if commercial_enabled else (
            not _selling_locked(fan_profile)
            and _fan_wants_content(latest_message, situation)
        )

        session_is_executable = bool(
            active_session
            and active_session.get("status") in {"active", "paused"}
        )
        if not session_is_executable and should_plan:
            cap_ok, cap_reason = await _within_daily_caps(creator_id, sent_ppv, fan_profile)
            if not cap_ok:
                print(f"[CAP] fan={fan_id} plan suppressed: {cap_reason}")
            else:
                try:
                    plan_data = await plan_session_for_fan(
                        creator_id,
                        fan_id,
                        accepted_set_id=(decision.accepted_offer_set_id if decision else None),
                        accepted_price_cents=(decision.session_budget_cents if decision else None),
                    )
                    plan_status = str(plan_data.get("status") or "")
                    if plan_status == "ok":
                        active_session = plan_data.get("session") or await get_fan_session(fan_id)
                        print(f"[SESSION] Planned for fan={fan_id} items={len((active_session or {}).get('plan', []))}")
                    elif commercial_enabled:
                        # A plan that cannot be built is not a decision to say
                        # nothing. Classify it, repair what is repairable, and
                        # only then decide what this turn actually is.
                        active_session, decision, recovery_outcome = (
                            await _recover_or_downgrade_plan(
                                creator_id=creator_id,
                                fan_id=fan_id,
                                status=plan_status,
                                decision=decision,
                                situation=situation,
                                price_learning=price_learning,
                            )
                        )
                        if recovery_outcome is not None:
                            if outcome_sink is not None:
                                outcome_sink["outcome"] = recovery_outcome
                            return
                    else:
                        print(f"[SESSION] plan-session status={plan_status} fan={fan_id}")
                except Exception as e:
                    # An exception is an infrastructure failure, not a commercial
                    # outcome. Report it as one instead of as a no-send.
                    print(f"[SESSION PLAN ERROR] fan={fan_id} error={e}")
                    if commercial_enabled:
                        if outcome_sink is not None:
                            outcome_sink["outcome"] = AUTO_OUTCOME_PLAN_UNRECOVERABLE
                        return

        # Inject tip context into situation so prompt builder can use it
        if pending_tip:
            situation["pending_tip"] = pending_tip

        # Purchase/decline reactions. With Commercial v2 enabled, the final
        # policy action — not the analyzer's raw single label — controls locks.
        purchase_signal = situation.get("purchase_signal", "none")
        pending = None
        if purchase_signal in ("bought", "declined"):
            db = get_supabase()
            fan_data = await asyncio.to_thread(
                lambda: db.table("fans").select("pending_ppv_check")
                .eq("id", fan_id).single().execute()
            )
            pending = (fan_data.data or {}).get("pending_ppv_check")
            if pending and purchase_signal == "declined":
                try:
                    pending_price_cents = int(round(float(pending.get("price") or 0) * 100))
                    affordability = await refresh_affordability_from_situation(
                        creator_id=creator_id,
                        fan_id=fan_id,
                        situation=situation,
                        source_ref=f"auto-decline:{len(conversation_history)}:{latest_message}",
                        offered_price_cents=pending_price_cents,
                    )
                except Exception as exc:
                    print(f"[AFFORDABILITY] decline price record failed fan={fan_id}: {exc}")
            if pending and purchase_signal == "bought":
                print(f"[PPV SIGNAL] fan={fan_id} bought pending={pending}")
                if local_test_delivery:
                    # A simulated PPV has no platform transaction to verify.
                    # Reconciliation for a test fan is resolved locally by the
                    # durable PPV_RECONCILE action (and by the owner's explicit
                    # simulate-purchase control), so eagerly spawning a remote
                    # verification here would be the one remote call a simulated
                    # turn could still make.
                    print(f"[AUTO TEST DELIVERY] fan={fan_id} remote_verification=skipped")
                else:
                    spawn(
                        _verify_ppv_purchase(fan_id, creator_id, pending),
                        name="verify_ppv_purchase",
                    )

        if commercial_enabled and decision is not None:
            if decision.action in {ActionType.PAUSE_NO_BUDGET, ActionType.PAUSE_UNTIL_PAYDAY}:
                try:
                    declined_price = (pending or {}).get("price")
                    await set_fan_decline_lock(fan_id, declined_price)
                    if active_session and active_session.get("awaiting_purchase_index") is not None:
                        active_session = mark_step_declined(
                            active_session,
                            reason=decision.action.value,
                            pause=True,
                        )
                        await save_fan_session(fan_id, active_session)
                    print(f"[SESSION] fan={fan_id} affordability pause ({decision.action.value})")
                except Exception as exc:
                    print(f"[DECLINE LOCK ERROR] fan={fan_id} error={exc}")
            elif decision.action in {ActionType.SEND_NEXT_PPV_STEP, ActionType.RESUME_PREVIOUS_OFFER}:
                try:
                    await clear_fan_decline_lock(fan_id)
                except Exception as exc:
                    print(f"[DECLINE UNLOCK ERROR] fan={fan_id} error={exc}")
            elif decision.action == ActionType.CONTINUE_NORMAL_CHAT and purchase_signal == "declined":
                # A normal 'no' is not proof of poverty. End only the pending offer.
                if active_session and active_session.get("awaiting_purchase_index") is not None:
                    active_session = mark_step_declined(active_session, reason="offer_declined", pause=False)
                    await save_fan_session(fan_id, None)
                    state = await get_fan_state(fan_id)
                    state.status = FanStatus.IDLE
                    state.confirmed_budget_cents = None
                    state.pending_offer = None
                    state.accepted_offer_id = None
                    state.accepted_offer_set_id = None
                    state.accepted_offer_price_cents = None
                    await save_fan_state(fan_id, creator_id, state)
        else:
            # Legacy behavior is retained only when Commercial v2 is disabled.
            if purchase_signal == "declined":
                try:
                    await set_fan_decline_lock(fan_id, (pending or {}).get("price"))
                except Exception as exc:
                    print(f"[DECLINE LOCK ERROR] fan={fan_id} error={exc}")
            elif purchase_signal == "money_available":
                try:
                    await clear_fan_decline_lock(fan_id)
                except Exception as exc:
                    print(f"[DECLINE UNLOCK ERROR] fan={fan_id} error={exc}")

        buyer_lifecycle = await refresh_fan_lifecycle(
            creator_id=creator_id,
            fan_id=fan_id,
            situation=situation,
            commercial_decision=decision,
            active_session=active_session,
            fan_profile=fan_profile,
            trigger_type="auto_message",
        )

        price_learning = await refresh_price_learning(
            creator_id=creator_id,
            fan_id=fan_id,
            affordability=affordability,
            lifecycle=buyer_lifecycle,
            trigger_type="auto_message_post_lifecycle",
        )
        situation["price_learning"] = price_learning

        decision_context = decision.model_dump(mode="json") if decision else None
        scene_context, text_intimacy_context = await _scene_and_register(
            creator_id=creator_id,
            fan_id=fan_id,
            situation=situation,
            decision_context=decision_context,
            latest_message=latest_message,
            fan_profile=fan_profile,
            active_session=active_session,
        )
        conversation_director = await direct_conversation(
            creator_id=creator_id,
            fan_id=fan_id,
            conversation_history=conversation_history,
            latest_fan_message=latest_message,
            situation=situation,
            commercial_decision=decision_context,
            lifecycle=buyer_lifecycle,
            active_session=active_session,
            conversation_stage=conversation_stage.value,
            trigger_type="auto_message",
        )
        situation["conversation_director"] = conversation_director

        session_strategy = await plan_next_action(
            creator_id=creator_id,
            fan_id=fan_id,
            situation=situation,
            commercial_decision=decision_context,
            lifecycle=buyer_lifecycle,
            affordability=affordability,
            price_learning=price_learning,
            active_session=active_session,
            conversation_stage=conversation_stage.value,
            conversation_director=conversation_director,
            trigger_type="auto_message",
        )
        situation["session_strategy"] = session_strategy

        # Message shape is decided here, not by the writer — under the writer
        # versions that ask for it. V1 and V2 send option 1 verbatim, so
        # whatever rhythm the model settles into becomes the creator's whole
        # texting personality unless something outside the model varies it.
        #
        # writer_v3 opts out (ai/writer_style.py): it asks for one reply and
        # lets the model decide whether that reply is one bubble or a few. The
        # policy module is untouched and still runs for the older profiles, so
        # a V2-vs-V3 comparison is a comparison of exactly this difference.
        shape_enforced = writer_enforces_message_shape(writer_prompt_version)
        commercial_max_messages = getattr(decision, "max_messages", None)
        creator_turn_shapes = recent_bubble_counts(conversation_history)

        def _shape(**extra):
            return choose_message_shape(
                fan_id=fan_id,
                recent_counts=creator_turn_shapes,
                turn_key=latest_message,
                max_messages=commercial_max_messages,
                is_ppv_delivery=bool(
                    getattr(decision, "action", None) is ActionType.SEND_NEXT_PPV_STEP
                ),
                **extra,
            )

        message_shape = _shape() if shape_enforced else None
        if message_shape is not None:
            print(
                f"[MESSAGE SHAPE] fan={fan_id} target={message_shape.target_bubbles} "
                f"reason={message_shape.reason} recent={creator_turn_shapes}"
            )
        else:
            print(
                f"[MESSAGE SHAPE] fan={fan_id} target=model_decides "
                f"reason=writer_version:{writer_prompt_version} "
                f"commercial_max_messages={commercial_max_messages or 'none'}"
            )

        # What media actually exists for this turn, stated rather than inferred.
        # Built from the decision (which carries the approved rows' asset types)
        # and the live session plan, so the writer's statement and the planner's
        # rows cannot disagree.
        media_inventory = _build_turn_inventory(
            decision=decision,
            active_session=active_session,
            situation=situation,
            fan_message=latest_message,
        )
        # What this turn will ATTACH, decided here from the persisted session
        # plan rather than parsed back out of the writer's prose. The writer is
        # told only that it is attached; it cannot cause, prevent, reprice or
        # mis-address a delivery (services/ppv_turn.py).
        legacy_send_now = False
        if not commercial_enabled and active_session:
            # The exact trigger the legacy prompt used, evaluated deterministically:
            # not paused for affordability, and either he just said yes or the
            # session has been open for three of his messages.
            selling_paused = bool(
                getattr(fan_profile, "sale_paused_at", None)
            ) and purchase_signal != "money_available"
            fan_msg_count = len([m for m in conversation_history if m.role == "fan"])
            msgs_since_session = fan_msg_count - int(
                active_session.get("started_at_fan_msg_count", 0) or 0
            )
            legacy_send_now = (not selling_paused) and (
                msgs_since_session >= 3 or purchase_signal == "ready_to_buy"
            )

        ppv_delivery = plan_ppv_step_delivery(
            decision=decision if commercial_enabled else None,
            active_session=active_session,
            legacy_send_now=legacy_send_now,
        )
        if ppv_delivery is not None:
            print(
                f"[PPV DELIVERY] fan={fan_id} attaching media={ppv_delivery.media_ids} "
                f"price_cents={ppv_delivery.price_cents} "
                f"set={ppv_delivery.set_id} step={ppv_delivery.step_index}"
            )

        print(
            f"[INVENTORY] fan={fan_id} "
            f"authorized={','.join(media_inventory.authorized_asset_types) or 'none'} "
            f"vault={','.join(media_inventory.vault_asset_types) or 'none'} "
            f"video_requested={media_inventory.video_requested} "
            f"may_promise_video={media_inventory.may_promise_video}"
        )

        ctx = ConversationContext(
            fan_message=latest_message,
            conversation_history=conversation_history,
            fan_profile=fan_profile,
            creator_persona=creator_persona,
            similar_exchanges=similar_exchanges,
            conversation_stage=conversation_stage,
            creator_name="a creator",
            creator_legend=creator_legend,
            situation=situation,
            ppv_offers=ppv_offers,
            sent_ppv=sent_ppv,
            active_session=active_session,
            media_inventory=media_inventory.to_context(),
            ppv_delivery=(
                ppv_delivery.writer_context() if ppv_delivery is not None else {}
            ),
            commercial_decision=decision.model_dump(mode="json") if decision else None,
            fan_intelligence=fan_intelligence,
            buyer_lifecycle=buyer_lifecycle,
            affordability=affordability,
            price_learning=price_learning,
            session_strategy=session_strategy,
            conversation_director=conversation_director,
            scene=scene_context,
            text_intimacy=text_intimacy_context,
            message_shape=message_shape.to_context() if message_shape else {},
            open_threads=open_thread_lines,
            conversation_episodes=episode_lines,
            ai_stack_profile=stack.profile_id,
            writer_prompt_version=writer_prompt_version,
        )

        # What evidence this turn was allowed to see, and who decided what it
        # does. Both are recorded before generation, so they describe the inputs
        # to the reply rather than being reconstructed from its output.
        provenance.record_context(
            history_messages=len(conversation_history),
            analyzer_window=min(
                len(conversation_history), ANALYZER_TRANSCRIPT_MESSAGES
            ),
            writer_window=min(len(conversation_history), WRITER_TRANSCRIPT_MESSAGES),
            packet=build_context_packet(
                conversation_history,
                open_threads=open_thread_lines,
                episodes=episode_lines,
            ).fingerprint(),
            stack_profile=stack.profile_id,
            writer_prompt_version=writer_prompt_version,
            live_state={
                "creator_legend": bool(creator_legend),
                "fan_intelligence": bool(fan_intelligence),
                "buyer_lifecycle": bool(buyer_lifecycle),
                "affordability": bool(affordability),
                "price_learning": bool(price_learning),
                "session_strategy": bool(session_strategy),
                "conversation_director": bool(conversation_director),
                "experience_scene": bool(scene_context),
                "text_intimacy": bool(text_intimacy_context),
                "message_shape": bool(message_shape),
                "commercial_decision": decision is not None,
                "ppv_delivery": ppv_delivery is not None,
                "active_session": bool(active_session),
                "media_inventory": bool(media_inventory.authorized_asset_types),
            },
        )
        provenance.record_decision(
            source="commercial_orchestrator" if commercial_enabled else "legacy_session",
            action=getattr(decision, "action", None),
            reason=getattr(decision, "reason", None),
            purchase_signal=situation.get("purchase_signal"),
            crisis_signal=situation.get("crisis_signal"),
            resend_requested=situation.get("resend_requested"),
            extra={
                "director_phase": conversation_director.get("phase"),
                "director_action": conversation_director.get("action"),
                "session_goal": session_strategy.get("goal"),
                "strategic_move": situation.get("strategic_move"),
            },
        )

        route = select_writer_route(ctx, profile_id=stack.profile_id)
        print(
            f"[WRITER ROUTE] fan={fan_id} mode=auto profile={stack.profile_id} "
            f"route={route.route.value} "
            f"reason={route.reason} primary={route.primary_target.model} "
            f"fallback={(route.fallback_target.model if route.fallback_target else 'none')}"
        )
        # Full Auto sends exactly one message. Under a writer version that
        # understands the distinction it asks for exactly one REPLY, in the
        # bubbles that reply naturally has — not for candidates, of which there
        # are none here because there is no operator to choose between them.
        auto_candidates = writer_candidate_count(route.prompt_version, MODE_AUTO)
        auto_contract = (
            CONTRACT_AUTO_MESSAGES
            if writer_uses_auto_messages_contract(route.prompt_version, MODE_AUTO)
            else CONTRACT_CANDIDATES
        )
        prompt = build_prompt(
            ctx, prompt_version=route.prompt_version, reply_mode=MODE_AUTO
        )
        # Which model ACTUALLY answers is a different fact from which one the
        # router asked for, and only the second used to survive to the message
        # (finding H). The trace is the return channel for the first.
        writer_trace = GenerationTrace()
        with action_stage("writer_ms"):
            replies = await generate_replies(
                prompt,
                creator_persona,
                trace=writer_trace,
                max_candidates=auto_candidates,
                output_contract=auto_contract,
                retry_policy=writer_retry_policy(route.prompt_version),
                profile_id=stack.profile_id,
                telemetry_context={
                    "creator_id": creator_id,
                    "fan_id": fan_id,
                    "feature": "auto_reply",
                    **route.telemetry_metadata(),
                "buyer_lifecycle_stage": buyer_lifecycle.get("stage"),
                "price_learning_mode": price_learning.get("mode"),
                "price_learning_confidence": price_learning.get("confidence"),
                "session_strategy_goal": session_strategy.get("goal"),
                "session_strategy_action": session_strategy.get("next_action"),
                "conversation_director_phase": conversation_director.get("phase"),
                "conversation_director_action": conversation_director.get("action"),
                "conversation_director_reason": conversation_director.get("transition_reason"),
                },
                target_override=route.primary_target,
                fallback_target_override=route.fallback_target,
            )

        provenance.record_writer(writer_trace)

        if not replies:
            # generate_replies fails closed: an empty list is never "the writer
            # chose silence", it is every configured attempt having failed or
            # produced unusable output. A no-send decision is taken well before
            # this point and never reaches the writer at all.
            print(
                f"[AUTO REPLY] writer produced no usable reply fan={fan_id} "
                f"primary={route.primary_target.model} "
                f"fallback={(route.fallback_target.model if route.fallback_target else 'none')}"
            )
            if outcome_sink is not None:
                outcome_sink["outcome"] = AUTO_OUTCOME_WRITER_FAILED
            return

        decision_action = getattr(getattr(decision, "action", None), "value", None)

        # Inventory is an invariant, not a prompt preference. A promise of media
        # that does not exist is repaired into what does, and a candidate that
        # cannot be repaired into something honest is not sent at all.
        safe_reply, inventory_repaired = choose_inventory_safe_reply(
            replies,
            media_inventory,
            decision_action=decision_action,
            active_session=active_session,
        )
        if safe_reply is None:
            print(
                f"[INVENTORY GUARD] fan={fan_id} every candidate promised "
                f"unavailable media and none survived repair — sending nothing"
            )
            if outcome_sink is not None:
                outcome_sink["outcome"] = AUTO_OUTCOME_INVENTORY_UNSAFE
            return
        provenance.record_transform(TRANSFORM_INVENTORY_REPAIR, inventory_repaired)
        if inventory_repaired:
            print(
                f"[INVENTORY GUARD] repaired unavailable-media promise fan={fan_id}"
            )
        reply = safe_reply

        # Platform semantics are an invariant, not a prompt preference: paid
        # media is attached in chat and there is no link to offer. Scoped to
        # commercial turns so ordinary talk about links is untouched.
        reply, link_repaired = sanitize_delivery_language(
            reply,
            decision_action=decision_action,
            active_session=active_session,
        )
        provenance.record_transform(TRANSFORM_DELIVERY_LANGUAGE, link_repaired)
        if link_repaired:
            print(f"[PPV LANGUAGE] repaired delivery-link phrasing fan={fan_id}")

        # The tag is no longer a control surface. A writer that still emits one
        # is not obeyed, and the string never reaches the fan.
        reply, tag_stripped = strip_ppv_tags(reply)
        provenance.record_transform(TRANSFORM_PPV_TAG_STRIPPED, tag_stripped)
        if tag_stripped:
            print(
                f"[PPV DELIVERY] fan={fan_id} stripped a writer-emitted delivery "
                "tag; attachment is decided by commercial state, not by copy"
            )
        if not reply:
            print(f"[AUTO REPLY] nothing left to send after sanitising fan={fan_id}")
            if outcome_sink is not None:
                outcome_sink["outcome"] = AUTO_OUTCOME_WRITER_FAILED
            return

        if ppv_delivery is not None:
            # Paid media is one message: its text and its attachment travel
            # together. Merging here is what makes "the text cannot claim a
            # delivery that failed" true by construction — there is no separate
            # text message that could already have left.
            reply = apply_message_shape(reply, 1)
            provenance.record_transform(TRANSFORM_PPV_MERGED)

        if shape_enforced:
            # Re-resolve the shape now that the copy exists: a reply that turned
            # out to be one short thought is never worth two bubbles.
            final_shape = _shape(
                writer_word_count=len(visible_text(reply).replace("|", " ").split())
            )
            reply = apply_message_shape(reply, final_shape.target_bubbles)
            provenance.record_transform(TRANSFORM_SHAPE_APPLIED)
        elif commercial_max_messages:
            # No shape policy for this writer version, but a commercial
            # decision that caps message parts is commercial authority, not
            # style. apply_message_shape only ever merges, so the cap is honoured
            # without imposing a bubble count the decision did not ask for.
            reply = apply_message_shape(reply, max(1, int(commercial_max_messages)))
            provenance.record_transform(TRANSFORM_SHAPE_APPLIED)

        # Final check — abort if a new message arrived while we were generating
        current_task = _pending_auto_replies.get(fan_id)
        if current_task and current_task is not asyncio.current_task():
            print(f"[AUTO REPLY] New message detected post-generation — aborting for fan={fan_id}")
            return

        # Also check DB directly — catches messages that came in during generation
        # even if the task replacement hasn't happened yet
        fresh_check = await get_conversation_history(fan_id)
        fresh_fan_msgs = [m for m in fresh_check if m.role == "fan"]
        current_fan_msgs = [m for m in conversation_history if m.role == "fan"]
        if len(fresh_fan_msgs) > len(current_fan_msgs):
            print(f"[AUTO REPLY] New fan message in DB post-generation — aborting for fan={fan_id}")
            return

        db = get_supabase()
        fan_row = await asyncio.to_thread(
            lambda: db.table("fans")
            .select("fansly_group_id, platform_fan_id")
            .eq("id", fan_id)
            .single()
            .execute()
        )
        creator_row = await asyncio.to_thread(
            lambda: db.table("creators")
            .select("fansly_account_id, apifansly_account_id")
            .eq("id", creator_id)
            .single()
            .execute()
        )

        group_id = (fan_row.data or {}).get("fansly_group_id")
        platform_fan_id = (fan_row.data or {}).get("platform_fan_id")
        apifansly_account_id = (creator_row.data or {}).get("apifansly_account_id")
        # Re-read from the row for the delivery decision, but the fan cannot
        # have changed identity mid-turn; both must agree.
        local_test_delivery = local_test_delivery or _is_local_test_fan(platform_fan_id)

        # If no group_id yet, try to find it from chats list. A test fan has no
        # remote chat to find, and this call lists chats against the provider.
        if (
            not group_id
            and apifansly_account_id
            and platform_fan_id
            and not local_test_delivery
        ):
            from main import get_or_fetch_group_id

            group_id = await get_or_fetch_group_id(apifansly_account_id, str(platform_fan_id), fan_id)

        # One name for "this turn does not wait". Availability and composition
        # were skipped by two separate flags, which is why the timing line could
        # read mode=live away=5.39s next to fast=True and leave an operator
        # unable to tell a computed delay from an awaited one.
        simulation_fast = bool(skip_human_delays)
        delays_skipped = bool(skip_human_delays or skip_availability)

        async def _pause(seconds: float, *, phase: str) -> bool:
            """Apply one human-like pause, or skip its duration in simulation.

            Still consults the pending-task slot, so a simulated turn is
            abandoned for the same reason a live one is.
            """
            return await _sleep_while_current(
                fan_id,
                0.0 if simulation_fast else seconds,
                phase=phase,
            )

        parts = [p.strip() for p in reply.split("|") if p.strip()]
        timing = build_delivery_schedule(
            latest_message,
            parts,
            conversation_history=conversation_history,
            conversation_phase=conversation_director.get("phase"),
            active_session=active_session,
        )
        # The delays are reported as what will actually happen, not as what the
        # schedule computed. In a fast simulated turn every one of them is zero,
        # and the mode says so rather than naming the live availability mode the
        # scheduler would have used.
        awaited_away = 0.0 if (simulation_fast or skip_availability) else timing.availability_delay_seconds
        awaited_compose = 0.0 if simulation_fast else timing.composition_delay_seconds
        awaited_between = (
            [0.0 for _ in timing.inter_part_delays_seconds]
            if simulation_fast
            else list(timing.inter_part_delays_seconds)
        )
        print(
            f"[AUTO TIMING] fan={fan_id} parts={len(parts)} "
            f"mode={'simulation_fast' if simulation_fast else timing.availability_mode.value} "
            f"delays_skipped={'true' if delays_skipped else 'false'} "
            f"away={awaited_away:.2f}s "
            f"compose={awaited_compose:.2f}s "
            f"between={awaited_between} "
            f"planned_mode={timing.availability_mode.value} "
            f"planned_away={timing.availability_delay_seconds:.2f}s "
            f"planned_compose={timing.composition_delay_seconds:.2f}s"
        )

        # Do not advertise typing while the simulated creator is unavailable.
        if not skip_availability and not await _pause(
            timing.availability_delay_seconds,
            phase="availability",
        ):
            return

        # A test fan may still carry a stale fansly_group_id, so the local-test
        # flag — not the presence of a binding — decides whether we advertise
        # typing to the platform.
        if group_id and apifansly_account_id and not local_test_delivery:
            try:
                # PERF-006 — one pooled connection. This sits inside the
                # human-like composition delay on the live reply path, so a
                # per-call TLS handshake here was pure added latency before the
                # fan sees anything.
                typing_response = await apifansly_shared_client().post(
                    apifansly_url(
                        f"{apifansly_account_id}/chats/{str(group_id)}/typing"
                    ),
                    headers=apifansly_headers(),
                    timeout=5,
                )
                # A raw post outside services.apifansly.request(). It still
                # costs a credit per typing indicator, and on a chatty auto-mode
                # deployment there is one of these per reply.
                record_apifansly_raw_call(
                    typing_response,
                    operation="typing indicator",
                    account_id=str(apifansly_account_id),
                    category=CATEGORY_LIVE_CHAT,
                )
            except Exception:
                pass

        if not await _pause(
            timing.composition_delay_seconds, phase="before_part_1"
        ):
            return

        last_plain_message_id: str | None = None
        for i, part in enumerate(parts):
            if i > 0:
                inter_delay = timing.inter_part_delays_seconds[i - 1]
                if not await _pause(inter_delay, phase=f"before_part_{i + 1}"):
                    return

            # Deterministic delivery. ``ppv_delivery`` was decided from the
            # persisted session plan BEFORE the writer ran, so this branch is
            # entered because commercial state says so — never because a model
            # managed to serialise a magic string into free text.
            is_ppv_part = ppv_delivery is not None
            if is_ppv_part:
                delivery_reference = uuid.uuid4().hex
                text_out = part
                media_ids = list(ppv_delivery.media_ids)
                media_id = ppv_delivery.media_id
                price = ppv_delivery.price
                current_step = (
                    ((active_session or {}).get("plan") or [None] * (ppv_delivery.step_index + 1))[
                        ppv_delivery.step_index
                    ]
                    if active_session
                    else None
                )
                ppv_media_context = ppv_delivery.media_context(
                    payment_reference=delivery_reference
                )

                approval_policy = await get_creator_policy(creator_id)
                if approval_policy.require_operator_ppv_approval:
                    state_for_approval = await get_fan_state(fan_id)
                    approval = await create_ppv_approval_request(
                        creator_id=creator_id,
                        fan_id=fan_id,
                        message_content=text_out,
                        media_ids=media_ids,
                        price_cents=ppv_delivery.price_cents,
                        set_id=ppv_delivery.set_id,
                        step_index=ppv_delivery.step_index,
                        approved_experience=(
                            state_for_approval.desired_experience
                            or state_for_approval.accepted_offer_label
                        ),
                    )
                    print(
                        f"[PPV APPROVAL] fan={fan_id} request={approval.get('id')} "
                        f"media={media_ids} price_cents={ppv_delivery.price_cents}"
                    )
                    return
            else:
                text_out = part
                ppv_media_context = None

            # Platform acceptance is authoritative. Never create a local sent
            # message or PAYMENT_PENDING state for a delivery that failed.
            #
            # A disabled connector is the same fact as a missing binding here:
            # there is no route. The worker already postpones these actions, so
            # this is the backstop for any other caller.
            if (
                not group_id
                or not apifansly_account_id
                or not apifansly_enabled()
            ) and not local_test_delivery:
                print(f"[AUTO DELIVERY ERROR] fan={fan_id}: no live delivery route")
                if is_ppv_part:
                    await freeze_fan_for_review(fan_id, "ppv_delivery_route_missing")
                return

            platform_message_id = None
            send_started = time.perf_counter()
            try:
                if local_test_delivery:
                    platform_message_id = (
                        f"local-test:{delivery_reference}" if is_ppv_part else None
                    )
                    print(
                        f"[AUTO TEST DELIVERY] fan={fan_id} "
                        f"kind={'ppv' if is_ppv_part else 'text'} accepted=true"
                    )
                elif is_ppv_part and contains_simulation_media(media_ids):
                    # Belt and braces alongside services/ppv_delivery.py: this
                    # branch builds its own platform call, so it enforces the
                    # same invariant rather than trusting that planning already
                    # did. A simulated turn never reaches here (local_test_delivery
                    # is handled above), so arriving with a sim: id means a real
                    # fan was about to receive mirrored test media.
                    print(
                        f"[AUTO DELIVERY ERROR] fan={fan_id}: refused "
                        f"simulation-only media on a live route media={media_ids}"
                    )
                    await freeze_fan_for_review(fan_id, "simulation_media_on_live_route")
                    return
                elif is_ppv_part:
                    ppv_content = text_out if text_out else random.choice([
                        "here it is 😏",
                        "just for you...",
                        "this is what I've been saving 😈",
                        "don't say I never spoil you 💋",
                    ])
                    response_body = await send_apifansly_message(
                        str(apifansly_account_id),
                        str(group_id),
                        content=ppv_content,
                        media_ids=media_ids,
                        price_dollars=float(price),
                    )
                    platform_message_id = sent_message_id(response_body)
                    if not platform_message_id:
                        raise RuntimeError(
                            "platform accepted PPV but did not return a message ID"
                        )
                    print(
                        f"[PPV SEND] accepted=true media={media_id} "
                        f"price={price} reference={delivery_reference}"
                    )
                else:
                    from main import send_fansly_message

                    platform_message_id = await send_fansly_message(
                        apifansly_account_id, str(group_id), text_out
                    )
                    if not platform_message_id:
                        raise RuntimeError("platform rejected text delivery")
            except Exception as exc:
                record_stage("fansly_send_ms", (time.perf_counter() - send_started) * 1000)
                print(f"[AUTO DELIVERY ERROR] fan={fan_id}: {exc}")
                if is_ppv_part:
                    await freeze_fan_for_review(fan_id, "ppv_send_failed")
                return
            record_stage("fansly_send_ms", (time.perf_counter() - send_started) * 1000)

            try:
                # Every creator message this pipeline writes carries the AI
                # stack that produced it, PPV and plain alike.
                stack_marker = message_ai_stack_metadata(
                    route, profile_id=stack.profile_id, trace=writer_trace
                )
                # The whole turn's ground truth, closed with this part's own
                # delivery receipt: the record cannot claim a delivery the
                # platform did not acknowledge, because the receipt is read
                # from the send result rather than from the copy.
                provenance_marker = provenance.as_metadata(
                    part=i,
                    parts=len(parts),
                    delivery_kind=DELIVERY_PPV if is_ppv_part else DELIVERY_TEXT,
                    platform_message_id=platform_message_id,
                    delivery_reference=delivery_reference if is_ppv_part else None,
                    price_cents=(
                        ppv_delivery.price_cents
                        if (is_ppv_part and ppv_delivery is not None)
                        else None
                    ),
                )
                message_metadata = merge_provenance(
                    _with_ai_stack(ppv_media_context, stack_marker),
                    provenance_marker,
                )
                if is_ppv_part:
                    await save_ppv_message_receipt(
                        fan_id=fan_id,
                        creator_id=creator_id,
                        content=text_out,
                        was_ai_suggested=True,
                        platform_message_id=platform_message_id,
                        media_context=message_metadata,
                    )
                else:
                    last_plain_message_id = await save_message(
                        fan_id=fan_id,
                        creator_id=creator_id,
                        role="creator",
                        content=text_out,
                        was_ai_suggested=True,
                        fansly_message_id=platform_message_id,
                        media_context=message_metadata,
                    )
                if i == 0:
                    print(provenance.describe())
                if local_test_delivery:
                    print(
                        f"[AUTO TEST DELIVERY] fan={fan_id} "
                        f"kind={'ppv' if is_ppv_part else 'text'} message_persisted=true"
                    )
            except Exception as exc:
                # Delivery already happened. Freeze rather than retrying and
                # risking a duplicate message on the live account.
                await freeze_fan_for_review(fan_id, "delivery_sent_but_not_persisted")
                print(f"[AUTO PERSIST ERROR] fan={fan_id}: {exc}")
                return

            if is_ppv_part:
                try:
                    from datetime import datetime, timedelta, timezone

                    policy = await retry_transient_db_operation(
                        lambda: get_creator_policy(creator_id),
                        label=f"PPV delivery policy fan={fan_id}",
                        log_prefix="PPV PERSIST RETRY",
                    )
                    sent_at = datetime.now(timezone.utc)
                    expires_at = sent_at + timedelta(
                        hours=policy.ppv_payment_window_hours
                    )
                    pending_check = {
                        "reference": delivery_reference,
                        "media_id": media_id,
                        "media_ids": media_ids,
                        "set_id": (current_step or {}).get("set_id"),
                        "step_index": (active_session or {}).get("current_index"),
                        "price": price,
                        "price_cents": int(round(price * 100)),
                        "source": "auto",
                        "sent_at": sent_at.isoformat(),
                        "expires_at": expires_at.isoformat(),
                        "verification_attempts": 0,
                        "platform_message_id": platform_message_id,
                    }
                    active_session, reconcile_at, attached = await persist_ppv_reconciliation(
                        creator_id=creator_id,
                        fan_id=fan_id,
                        pending=pending_check,
                        session=active_session,
                        platform_message_id=platform_message_id,
                    )
                    if not attached:
                        print(
                            f"[TEST DELIVERY] fan={fan_id} "
                            "reconciliation already resolved"
                        )
                        return
                    print(
                        f"[PAYMENT] fan={fan_id} state=PAYMENT_PENDING "
                        f"reference={delivery_reference} reconcile_at={reconcile_at.isoformat()} "
                        f"expires_at={expires_at.isoformat()}"
                    )
                except Exception as exc:
                    await freeze_fan_for_review(fan_id, "ppv_sent_but_reconciliation_not_persisted")
                    print(f"[PPV PERSIST ERROR] fan={fan_id}: {exc}")
                    return

            print(f"[AUTO REPLY] Sent part {i+1}: {text_out[:50]}")

        # The reply is delivered and persisted. Only now may anything the
        # creator improvised about herself become canon: a turn that aborted at
        # any of the returns above sent nothing, so it established nothing.
        if writer_persists_improvised_facts(writer_prompt_version):
            spawn(
                persist_sent_creator_facts(
                    creator_id=creator_id,
                    sent_reply=visible_text(reply),
                    fan_message=latest_message,
                    conversation_history=conversation_history,
                    fan_id=fan_id,
                    profile_id=stack.profile_id,
                ),
                name=f"creator_canon:{fan_id}",
            )

        if last_plain_message_id:
            try:
                from services.inactivity_reengagement import (
                    schedule_inactivity_reengagement,
                )

                await schedule_inactivity_reengagement(
                    creator_id=creator_id,
                    fan_id=fan_id,
                    source_message_id=last_plain_message_id,
                )
            except Exception as exc:
                # The reply is already safely delivered and persisted. Failure to
                # schedule an optional future nudge must never duplicate it.
                print(
                    f"[INACTIVITY REENGAGEMENT ERROR] fan={fan_id} "
                    f"schedule_failed={exc}"
                )

    except asyncio.CancelledError:
        raise
    except (AnalyzerDegradedError, HumanReviewHandoffError):
        # Preserve the actual failure so the durable action can retry it and
        # the simulator cannot report a failed analysis or hold as a decision
        # to send nothing.
        raise
    except Exception as e:
        print(f"[DEBOUNCED AUTO REPLY ERROR] fan={fan_id} error={e}")
        import traceback
        traceback.print_exc()
    finally:
        # A cancelled older task may finish after schedule_auto_reply has already
        # installed a newer one. Only the task that still owns this fan's slot may
        # clear it; otherwise rapid fan messages can leave multiple replies alive.
        _release_auto_reply_slot(fan_id)


# Removed: _REACTION_FISHING_LINES.
#
# Seven sentences, one picked at random at schedule time and frozen into the
# queued action, sent to every customer after every purchase. It bypassed the
# writer entirely — services/proactive.py generates from a goal and the
# conversation unless _delivery.text is already set — so the message that
# follows money changing hands was the only proactive message that never read
# the conversation it was about. services/post_purchase.py decides at execute
# time whether to say anything at all, and the writer says it.


async def record_ppv_purchase(
    fan_id: str,
    media_id: str,
    amount: float | None = None,
    *,
    pending_override: dict | None = None,
    platform_order_id: str | None = None,
) -> None:
    """Record one confirmed PPV purchase idempotently and advance its session.

    The paid-session plan moves forward only here, after confirmation. The final
    step closes and clears the active session and resets session-specific state.
    """
    def _tier(total: int) -> str:
        return "whale" if total >= 500 else "active" if total >= 100 else "casual" if total >= 20 else "cold"

    db = get_supabase()
    fan_response = await asyncio.to_thread(
        lambda: db.table("fans")
        .select(
            "total_spent, sales_log, not_sold_log, creator_id, "
            "needs_human_review, pending_ppv_check"
        )
        .eq("id", fan_id).single().execute()
    )
    row = fan_response.data or {}
    creator_id = row.get("creator_id")
    sales_log = list(row.get("sales_log") or [])
    not_sold = list(row.get("not_sold_log") or [])
    current_pending = row.get("pending_ppv_check") or {}
    pending = pending_override or current_pending
    reference = str(pending.get("reference") or "")
    current_reference = str(current_pending.get("reference") or "")
    clears_current_pending = bool(
        current_pending
        and (
            (reference and current_reference == reference)
            or (not reference and not pending_override)
        )
    )
    completed_session: dict | None = None
    lifecycle_context: dict = {}

    session = await get_fan_session(fan_id)
    if amount is None:
        amount = pending.get("price")
    if amount is None and session:
        idx = session.get("awaiting_purchase_index")
        plan = session.get("plan") or []
        if idx is not None and 0 <= int(idx) < len(plan):
            amount = plan[int(idx)].get("price")
    if amount is None:
        amount = next(
            (entry.get("amount", 0) for entry in not_sold if str(media_id) in str(entry.get("item", ""))),
            0,
        )
    amount_dollars = int(round(float(amount or 0)))

    duplicate_platform_order = bool(
        platform_order_id
        and any(
            str(entry.get("platform_order_id") or "") == platform_order_id
            for entry in sales_log
        )
    )
    if duplicate_platform_order:
        return

    already_recorded = any(
        (
            str(entry.get("payment_reference") or "") == reference
            if reference
            else str(entry.get("media_id")) == str(media_id)
        )
        for entry in sales_log
    )
    old_spent = int(row.get("total_spent", 0) or 0)
    new_spent = old_spent
    if not already_recorded:
        from datetime import datetime
        sales_log.append({
            "date": datetime.utcnow().strftime("%d.%m.%Y"),
            "item": f"PPV media {media_id}",
            "media_id": str(media_id),
            "media_ids": pending.get("media_ids") or [str(media_id)],
            "payment_reference": reference or None,
            "platform_order_id": platform_order_id or None,
            "amount": amount_dollars,
            "chatter": "Operator" if pending.get("source") == "operator" else "AI",
        })
        not_sold = [
            entry
            for entry in not_sold
            if (
                str(entry.get("payment_reference") or "") != reference
                if reference
                else str(media_id) not in str(entry.get("item", ""))
            )
        ]
        new_spent = old_spent + amount_dollars

    if reference:
        from services.ppv_delivery_ledger import transition_delivery

        await transition_delivery(
            reference,
            "purchased",
            amount_paid_cents=int(round(float(amount or 0) * 100)),
            metadata=(
                {"platform_order_id": platform_order_id}
                if platform_order_id
                else None
            ),
        )
    def _merge_purchase(current: dict | None) -> dict | None:
        """Build the fan update from the row as it actually is.

        Called once with ``None`` (meaning: use the snapshot this function
        already read) and again, with a freshly read row, whenever another
        purchase event wrote first. Everything it needs is recomputed from
        ``current`` on those later calls; nothing from the original snapshot
        survives into them, because merging into a stale total is the exact
        loss this guards against.
        """
        if current is None:
            merged_spent = new_spent
            merged_sales = sales_log
            merged_not_sold = not_sold
            merged_pending = current_pending
        else:
            fresh_sales = list(current.get("sales_log") or [])
            recorded_now = any(
                (
                    str(entry.get("payment_reference") or "") == reference
                    if reference
                    else str(entry.get("media_id")) == str(media_id)
                )
                for entry in fresh_sales
            ) or (
                bool(platform_order_id)
                and any(
                    str(entry.get("platform_order_id") or "") == platform_order_id
                    for entry in fresh_sales
                )
            )
            if recorded_now:
                # The event that beat us recorded this same purchase. Writing
                # it again would double the customer's spend.
                print(
                    f"[PURCHASE CAS] fan={fan_id} reference={reference or 'none'} "
                    "already_recorded_by_concurrent_event=true"
                )
                return None
            fresh_spent = int(current.get("total_spent") or 0)
            merged_spent = fresh_spent + (amount_dollars if not already_recorded else 0)
            merged_sales = fresh_sales + (
                [sales_log[-1]] if (not already_recorded and sales_log) else []
            )
            merged_not_sold = [
                entry
                for entry in (current.get("not_sold_log") or [])
                if (
                    str(entry.get("payment_reference") or "") != reference
                    if reference
                    else str(media_id) not in str(entry.get("item", ""))
                )
            ]
            merged_pending = current.get("pending_ppv_check") or {}

        update = {
            "total_spent": merged_spent,
            "spend_tier": _tier(merged_spent),
            "sales_log": merged_sales,
            "not_sold_log": merged_not_sold,
        }
        # Recomputed against whichever row this attempt is merging into: the
        # pending check the caller saw may have been replaced by a newer offer,
        # and clearing that one would cancel a delivery nobody has resolved.
        still_clears = bool(
            merged_pending
            and (
                (reference and str(merged_pending.get("reference") or "") == reference)
                or (not reference and not pending_override)
            )
        )
        if still_clears:
            update["pending_ppv_check"] = None
        return update

    try:
        await apply_purchase_to_fan(
            fan_id,
            merge=_merge_purchase,
            expected_total_spent=old_spent,
        )
    except PurchaseAggregateConflict as exc:
        # The money is real and could not be written down. An operator has to
        # see that, and automation must not carry on as though the customer's
        # spend were correct.
        print(f"[PURCHASE CONFLICT] fan={fan_id} {exc}")
        await freeze_fan_for_review(fan_id, "purchase_aggregate_conflict")
        raise
    await mark_ppv_purchased(fan_id, str(media_id))
    if creator_id and reference:
        await cancel_action_by_dedupe_key(
            f"ppv-reconcile:{fan_id}:{reference}"
        )
    if creator_id and clears_current_pending:
        await cancel_actions_for_fan(fan_id, "ABANDONED_PPV_FOLLOWUP")
    if creator_id and not already_recorded:
        try:
            await record_confirmed_purchase(
                creator_id=creator_id,
                fan_id=fan_id,
                amount_cents=amount_dollars * 100,
                source_ref=f"ppv:{media_id}",
                metadata={"media_id": str(media_id)},
            )
        except Exception as exc:
            print(f"[AFFORDABILITY] purchase record failed fan={fan_id}: {exc}")

        # The scene moves to AWAIT_REACTION here, at the moment the unlock
        # actually becomes true — not on his next message. The one-step
        # commercial session is cleared a few lines below, so by the time he
        # replies there is nothing left to infer a purchase from.
        try:
            unlocked_set_id = (pending or {}).get("set_id") or (session or {}).get("set_id")
            await record_unlock(
                creator_id=creator_id,
                fan_id=fan_id,
                set_id=unlocked_set_id,
                scene_metadata=await scene_metadata_for(creator_id, unlocked_set_id),
            )
        except Exception as exc:
            # Losing the scene costs continuity on the next reply. It must
            # never cost a recorded purchase.
            print(f"[EXPERIENCE] unlock record failed fan={fan_id}: {exc}")

    if session and creator_id and pending.get("step_index") is not None:
        try:
            policy = await get_creator_policy(creator_id)
            updated, completed = mark_step_purchased(
                session,
                media_id=str(media_id),
                set_id=(pending or {}).get("set_id"),
                amount_cents=amount_dollars * 100,
            )
            state = await get_fan_state(fan_id)
            if state.next_followup_type == "ABANDONED_PPV_FOLLOWUP":
                state.next_followup_at = None
                state.next_followup_type = None
                state.next_followup_payload = {}
                state.next_followup_dedupe_key = None
            if completed:
                from datetime import datetime, timezone
                completed_session = updated
                await save_fan_session(fan_id, updated)
                lifecycle_context = await refresh_fan_lifecycle(
                    creator_id=creator_id,
                    fan_id=fan_id,
                    active_session=None,
                    trigger_type="session_completed",
                )
                state, followup_obligation = complete_session_state(
                    state,
                    updated,
                    policy=policy,
                    fan_id=fan_id,
                    buyer_stage=str(lifecycle_context.get("stage") or "UNKNOWN"),
                    now=datetime.now(timezone.utc),
                )

                # Persist the obligation before clearing the completed executable
                # session. The worker can recreate a missing action after a restart.
                await save_fan_state(fan_id, creator_id, state)
                await save_fan_session(fan_id, None)
                if followup_obligation:
                    try:
                        await schedule_action(
                            creator_id=creator_id,
                            fan_id=fan_id,
                            action_type=followup_obligation.action_type,
                            execute_at=followup_obligation.execute_at,
                            payload=followup_obligation.payload,
                            dedupe_key=followup_obligation.dedupe_key,
                        )
                    except Exception as exc:
                        print(
                            f"[FOLLOWUP REPAIR NEEDED] fan={fan_id} "
                            f"type=POST_SESSION_FOLLOWUP error={exc}"
                        )
                print(
                    f"[SESSION] completed fan={fan_id} "
                    f"revenue_cents={state.last_session_revenue_cents} "
                    f"followup_at={state.next_followup_at}"
                )
            else:
                state.status = FanStatus.PAID_SESSION_ACTIVE
                await save_fan_session(fan_id, updated)
                print(
                    f"[SESSION] purchase confirmed fan={fan_id}; "
                    f"next={updated.get('current_index')}/{len(updated.get('plan') or [])}"
                )
                await save_fan_state(fan_id, creator_id, state)
        except Exception as exc:
            # Purchase accounting remains recorded; lifecycle mismatch is loudly
            # logged for human review rather than double-charging on a retry.
            await freeze_fan_for_review(fan_id, "session_purchase_reconcile_failed")
            print(f"[SESSION PURCHASE RECONCILE ERROR] fan={fan_id}: {exc}")
            raise
    elif creator_id and clears_current_pending:
        # A one-off operator PPV has no executable paid-session plan. Once the
        # purchase is confirmed, clear the payment hold without inventing a
        # session lifecycle.
        state = await get_fan_state(fan_id)
        state.status = FanStatus.IDLE
        state.pending_offer = None
        state.accepted_offer_id = None
        state.accepted_offer_set_id = None
        state.accepted_offer_label = None
        state.accepted_offer_price_cents = None
        await save_fan_state(fan_id, creator_id, state)

    if creator_id and not already_recorded:
        from datetime import datetime, timedelta, timezone

        purchased_at = datetime.now(timezone.utc)
        reaction_key = platform_order_id or reference or f"{media_id}:{purchased_at.isoformat()}"
        await schedule_action(
            creator_id=creator_id,
            fan_id=fan_id,
            action_type="POST_PURCHASE_REACTION",
            execute_at=purchased_at + timedelta(seconds=random.uniform(20.0, 55.0)),
            # No _delivery.text. It used to carry one of seven sentences picked
            # by random.choice at THIS moment — before he had a chance to
            # react, before an operator could take over — and setting it
            # short-circuits generation in services/proactive.py, so the one
            # proactive message that follows money changing hands was the only
            # one that never saw the conversation. The goal is decided at
            # execute time instead (services/post_purchase.py).
            payload={
                "purchase_at": purchased_at.isoformat(),
                "media_id": str(media_id),
            },
            dedupe_key=f"post-purchase-reaction:{fan_id}:{reaction_key}",
        )

    # Whale handoff only on the threshold crossing, not duplicate webhooks.
    if creator_id and not row.get("needs_human_review") and not already_recorded:
        try:
            caps = await get_creator_caps(creator_id)
            threshold = int(caps.get("whale_handoff_threshold") or 0)
            if threshold and old_spent < threshold <= new_spent:
                await freeze_fan_for_review(fan_id, f"whale:${new_spent}")
                print(f"[WHALE HANDOFF] fan={fan_id} crossed ${threshold} (now ${new_spent})")
        except Exception as exc:
            print(f"[WHALE HANDOFF ERROR] fan={fan_id} error={exc}")

    if creator_id:
        if completed_session is None:
            await refresh_fan_lifecycle(
                creator_id=creator_id,
                fan_id=fan_id,
                active_session=await get_fan_session(fan_id),
                trigger_type="purchase_confirmed",
            )
        await refresh_price_learning(
            creator_id=creator_id,
            fan_id=fan_id,
            trigger_type="purchase_confirmed",
        )


async def _verify_ppv_purchase(
    fan_id: str,
    creator_id: str,
    pending: dict,
) -> None:
    """Immediate verification hook; durable retries stay in scheduled_actions."""
    try:
        from services.followup_lifecycle import pending_reference
        from services.ppv_reconciliation import (
            PPVReconcileDisposition,
            reconcile_pending_ppv,
        )

        reference = pending_reference(pending)
        result = await reconcile_pending_ppv(
            creator_id=creator_id,
            fan_id=fan_id,
            expected_reference=reference,
        )
        if result.disposition == PPVReconcileDisposition.PENDING and result.retry_at:
            await schedule_action(
                creator_id=creator_id,
                fan_id=fan_id,
                action_type="PPV_RECONCILE",
                execute_at=result.retry_at,
                payload={"payment_reference": reference},
                dedupe_key=f"ppv-reconcile:{fan_id}:{reference}",
            )
        print(
            f"[PPV VERIFY] fan={fan_id} disposition={result.disposition.value} "
            f"reference={reference} reason={result.reason}"
        )
    except Exception as exc:
        # The durable action remains pending and will retry with backoff.
        print(f"[PPV VERIFY ERROR] fan={fan_id}: {exc}")


async def sweep_stale_ppv_checks() -> None:
    """
    Repair durable reconciliation actions for pending PPVs after restarts or
    partial persistence failures. Verification itself happens in the worker.
    """
    from datetime import datetime, timezone, timedelta

    try:
        db = get_supabase()

        from db.commercial_queries import ensure_action_pending
        from services.followup_lifecycle import pending_reference

        # Paginated: this sweep is the repair path for pending PPV
        # reconciliation after a restart or a partial write. Rows past the
        # 1,000-row cap were silently skipped every 15 minutes, so those fans'
        # reconciliation actions were never rebuilt and their purchases went
        # unverified. This read spans every creator, so the cap was global and
        # easy to reach.
        from core.pagination import fetch_all_rows_async

        # Paged select, so a repeat is free. A PostgREST connection recycle
        # part-way through the paging used to abort the whole sweep with
        # [PPV SWEEP FATAL] and leave every stale reconciliation unrepaired
        # until the next 15-minute pass.
        pending_rows = await retry_transient_db_operation(
            lambda: fetch_all_rows_async(
                lambda start, end: db.table("fans")
                .select(
                    "id, creator_id, pending_ppv_check, needs_human_review, "
                    "review_reason"
                )
                .not_.is_("pending_ppv_check", "null")
                .order("id")
                .range(start, end)
                .execute()
            ),
            label="ppv_sweep.pending_rows",
        )

        if not pending_rows:
            return

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=5)
        stale = []

        for row in pending_rows:
            if row.get("needs_human_review"):
                continue
            pending = row.get("pending_ppv_check") or {}
            sent_at_str = pending.get("sent_at", "")
            if not sent_at_str:
                continue
            try:
                sent_dt = datetime.fromisoformat(sent_at_str.replace("Z", "+00:00"))
                # Make naive datetimes timezone-aware
                if sent_dt.tzinfo is None:
                    sent_dt = sent_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            if sent_dt < cutoff:
                stale.append((str(row["id"]), str(row["creator_id"]), pending))

        if not stale:
            return

        print(f"[PPV SWEEP] repairing {len(stale)} pending reconciliation action(s)")

        for fan_id, creator_id, pending in stale:
            try:
                reference = pending_reference(pending)
                await ensure_action_pending(
                    creator_id=creator_id,
                    fan_id=fan_id,
                    action_type="PPV_RECONCILE",
                    execute_at=now,
                    payload={"payment_reference": reference},
                    dedupe_key=f"ppv-reconcile:{fan_id}:{reference}",
                )
            except Exception as e:
                print(f"[PPV SWEEP ERROR] fan={fan_id} error={e}")

    except Exception as e:
        print(f"[PPV SWEEP FATAL] {e}")


async def schedule_auto_reply(
    fan_id: str,
    creator_id: str,
    *,
    conversation_history: list[Message] | None = None,
    source_message_id: str | None = None,
) -> None:
    """Persist a replaceable Auto reply obligation that survives redeploys."""
    existing = _pending_auto_replies.get(fan_id)
    if existing and not existing.done():
        existing.cancel()
        print(f"[AUTO REPLY] Reset timer for fan={fan_id}")
    from datetime import datetime, timedelta, timezone

    from db.queries import get_creator_sleep_hours
    from services.followup_lifecycle import next_awake_time

    history = conversation_history or await get_conversation_history(fan_id)
    latest_fan = next((message for message in reversed(history) if message.role == "fan"), None)
    trigger_at = (
        latest_fan.sent_at.astimezone(timezone.utc)
        if latest_fan and latest_fan.sent_at
        else datetime.now(timezone.utc)
    )
    active_session = await get_fan_session(fan_id)
    mode, availability_seconds = build_availability_delay(
        history,
        active_session=active_session,
    )
    sleep_start, sleep_end = await get_creator_sleep_hours(creator_id)
    now = datetime.now(timezone.utc)
    awake_at = next_awake_time(
        now,
        sleep_start_hour=sleep_start,
        sleep_end_hour=sleep_end,
        timezone_name="UTC",
    )
    wake_jitter = random.uniform(60.0, 600.0) if awake_at > now else 0.0
    execute_at = max(now, awake_at) + timedelta(
        seconds=wake_jitter + random.uniform(7.0, 12.0) + availability_seconds
    )
    await cancel_actions_for_fan(fan_id, "AUTO_REPLY")
    dedupe_source = source_message_id or trigger_at.isoformat()
    await schedule_action(
        creator_id=creator_id,
        fan_id=fan_id,
        action_type="AUTO_REPLY",
        execute_at=execute_at,
        payload={
            "trigger_message_id": source_message_id,
            "trigger_content": latest_fan.content if latest_fan else "",
            "trigger_sent_at": trigger_at.isoformat(),
            "availability_mode": mode.value,
        },
        dedupe_key=f"auto-reply:{fan_id}:{dedupe_source}",
    )
    print(
        f"[AUTO REPLY QUEUED] fan={fan_id} durable=true mode={mode.value} "
        f"execute_at={execute_at.isoformat()}"
    )


async def deliver_scheduled_auto_reply(action: dict) -> bool:
    """Execute a claimed Auto obligation and report whether a reply now exists."""
    fan_id = str(action["fan_id"])
    creator_id = str(action["creator_id"])
    payload = action.get("payload") or {}
    trigger_at_raw = str(payload.get("trigger_sent_at") or "")

    if action.get("status") == "PROCESSING":
        from main import sync_recent_fan_messages

        result = await sync_recent_fan_messages(creator_id, fan_id)
        if result.get("status") != "ok":
            raise RuntimeError("could not reconcile stale Auto reply before retry")

    if trigger_at_raw:
        from datetime import datetime, timezone

        trigger_at = datetime.fromisoformat(trigger_at_raw.replace("Z", "+00:00"))
        if trigger_at.tzinfo is None:
            trigger_at = trigger_at.replace(tzinfo=timezone.utc)
        current_history = await get_conversation_history(fan_id, limit=10)
        newer = [
            message
            for message in current_history
            if message.sent_at and message.sent_at > trigger_at
        ]
        if newer:
            return any(message.role == "creator" for message in newer)

    task = asyncio.create_task(
        _debounced_auto_reply(
            fan_id,
            creator_id,
            skip_debounce=True,
            skip_availability=True,
            expected_trigger_at=trigger_at_raw or None,
        )
    )
    _pending_auto_replies[fan_id] = task
    try:
        await task
    except asyncio.CancelledError:
        return False

    if not trigger_at_raw:
        return False
    trigger_at = datetime.fromisoformat(trigger_at_raw.replace("Z", "+00:00"))
    if trigger_at.tzinfo is None:
        trigger_at = trigger_at.replace(tzinfo=timezone.utc)
    history = await get_conversation_history(fan_id, limit=10)
    return any(
        message.role == "creator"
        and message.sent_at
        and message.sent_at > trigger_at
        for message in history
    )


def _should_update_memory(conversation_history: list[Message]) -> bool:
    count = len([m for m in conversation_history if m.role == "fan"])
    return count > 0 and count % 10 == 0


# ---------------------------------------------------------------------------
# Owner-only local Full Auto simulation
# ---------------------------------------------------------------------------
#
# The point of this function is what it does NOT contain. There is no second
# prompt, no simplified engine, no alternate model routing and no commercial
# bypass. It persists the fan's message the same way an inbound platform message
# is persisted, then awaits the same ``_debounced_auto_reply`` the durable
# AUTO_REPLY worker awaits, with the same arguments the worker passes plus a
# flag that removes the deliberate human-like pauses.
#
# Everything the simulated turn reads — full conversation history including rows
# inserted by hand in SQL, fan profile, fan intelligence, buyer lifecycle,
# affordability, price learning, creator persona, creator legend, PPV offers and
# history, the active session, the situation analyzer, conversation stage, the
# commercial orchestrator, session planning, the conversation director, writer
# routing, the OpenRouter/Kimi fallback ladder and the fail-closed analyzer
# behaviour — is loaded by that one function, unchanged.
#
# Only two things are simulated: delivery transport (the ``test_`` fan branch
# persists locally instead of calling the platform) and waiting.


async def _record_conversation_threads(
    situation: dict,
    *,
    creator_id: str,
    fan_id: str,
    latest_message: str,
    turn_id: str,
) -> None:
    """Persist the unfinished business this turn created or settled.

    Never raises and never blocks a reply. Continuity is memory: losing a
    record costs the next turn some context, and raising here would cost the
    customer their answer. services/reply_provenance.py takes the same position
    for the same reason.

    The extraction is semantic (ai/situation_analyzer.py) and the validation is
    not (services/continuity_extraction.py) — in particular a proposal that
    makes a claim about money is refused in code, because ppv_deliveries is the
    only authority on that and the analyzer reads customer-supplied text.
    """
    from services.continuity_extraction import extract_threads

    try:
        extracted = extract_threads(
            situation,
            creator_id=str(creator_id),
            fan_id=str(fan_id),
            source_turn_id=str(turn_id or ""),
            source_message_fingerprint=fingerprint(latest_message),
        )
    except Exception as exc:  # pragma: no cover - extraction never blocks a turn
        print(f"[CONTINUITY] extraction failed fan={fan_id}: {type(exc).__name__}")
        return

    recorded = 0
    for thread in extracted.threads:
        try:
            if await record_open_thread(thread):
                recorded += 1
        except Exception as exc:
            print(
                f"[CONTINUITY] could not record {thread.kind.value} "
                f"fan={fan_id}: {type(exc).__name__}"
            )

    if recorded or extracted.rejected:
        # The rejection counts are the interesting half. A rate that climbs is
        # how somebody notices the analyzer has started proposing things it
        # should not, which no amount of prompt wording would tell them.
        print(
            f"[CONTINUITY] fan={fan_id} recorded={recorded} "
            f"proposed={len(extracted.threads)} "
            f"rejected={extracted.rejected or 'none'}"
        )


async def _recent_creator_message_rows(fan_id: str) -> list[dict]:
    """Creator messages for this fan, newest-window first, in production order.

    ``get_conversation_history`` returns ``Message`` objects, which carry no row
    id, so the simulator diffs the table directly. Identifying the turn's output
    by id rather than by timestamp keeps multipart replies — which share a
    second — ordered and complete.
    """

    def _load() -> list[dict]:
        result = (
            get_supabase().table("messages")
            .select("id, role, content, sent_at, media_context")
            .eq("fan_id", fan_id)
            .eq("role", "creator")
            .order("sent_at", desc=True)
            .limit(40)
            .execute()
        )
        rows = list(reversed(result.data or []))
        return [
            {
                "id": str(row.get("id")),
                "role": "creator",
                "content": row.get("content"),
                "sent_at": row.get("sent_at"),
                "media_context": row.get("media_context"),
            }
            for row in rows
            if row.get("id")
        ]

    return await asyncio.to_thread(_load)


async def run_simulated_inbound(
    *,
    fan_id: str,
    creator_id: str,
    message: str,
    fast: bool = True,
    include_mirrored_catalog: bool = False,
) -> dict:
    """Persist one fan message and run the real Full Auto turn it triggers.

    Returns the fan message id and every creator message the turn produced, in
    production order. Raises nothing that the caller needs to translate: an
    analyzer that fails closed, or a turn that decides to send nothing, is a
    real Full Auto outcome and is reported as such.

    The whole turn runs inside ``simulation_scope()``. That is not belt and
    braces for the ``test_`` fan branches above — it is the hard invariant. The
    transport refuses every API Fansly request made by this task or by anything
    it spawns, so "zero remote calls" holds even for a code path nobody audited.

    ``include_mirrored_catalog`` decides only what this turn may PLAN against.
    An agency's turn leaves it False and plans against the creator's own
    approved vault and sets; an owner's turn may set it and additionally see
    mirrored cross-tenant test rows. Neither value permits a remote call.
    """
    # Marked as an owner simulation event so the production Supabase database
    # webhook on messages INSERT — which POSTs /generate-suggestions — ignores
    # this row. Without the marker one simulated turn ran the ordinary inbound
    # pipeline as well as the Full Auto turn below, doubling situation analysis,
    # commercial state changes, price learning and the conversation director.
    # The simulator is the sole processor of its own event.
    fan_message_id = await save_message(
        fan_id,
        creator_id,
        "fan",
        message,
        media_context=simulation_message_marker(),
    )
    # Belt and braces for the single-process deployment Railway actually runs:
    # the webhook can recognise this row without depending on how the database
    # webhook serialises a jsonb column. Registered before the turn starts, so
    # a webhook delivery that arrives while we are still analysing is covered.
    mark_simulation_owned_message(fan_message_id)

    history_before = await get_conversation_history(fan_id)
    creator_ids_before = {row["id"] for row in await _recent_creator_message_rows(fan_id)}

    with simulation_scope(include_mirrored_catalog=include_mirrored_catalog):
        # Fan intelligence learning is part of the real inbound pipeline, so the
        # simulated turn runs it too. Awaited rather than spawned, so the
        # simulation scope is still active while it runs and the caller's
        # response reflects a settled turn.
        try:
            # The extractor runs under the same profile the rest of the turn
            # will use, including a test fan's own simulation override.
            extraction_stack = await resolve_ai_stack(
                creator_id=creator_id, fan_id=fan_id
            )
            await learn_from_fan_message(
                creator_id=creator_id,
                fan_id=fan_id,
                fan_message=message,
                source_message_id=fan_message_id,
                conversation_history=history_before,
                profile_id=extraction_stack.profile_id,
            )
        except Exception as exc:
            # Extraction is an enrichment, never a reason to lose the turn.
            print(f"[SIMULATION] fan_intelligence failed fan={fan_id}: {exc}")

        # Identical to deliver_scheduled_auto_reply's invocation: register the
        # task in the pending slot so the pipeline's own staleness checks see a
        # current owner, then await it.
        #
        # The real Auto path reports into this mapping. It is created here and
        # handed down rather than returned, because a task's context is copied
        # on creation: a value the task sets would not travel back to us.
        auto_outcome: dict[str, str] = {}
        task = asyncio.create_task(
            _debounced_auto_reply(
                fan_id,
                creator_id,
                skip_debounce=True,
                skip_availability=True,
                skip_human_delays=bool(fast),
                outcome_sink=auto_outcome,
            )
        )
        _pending_auto_replies[fan_id] = task
        analyzer_degraded = False
        try:
            await task
        except AnalyzerDegradedError as exc:
            # Fail-closed analyzer behaviour is exactly what the simulator is
            # for observing. Report it instead of raising a 500.
            analyzer_degraded = True
            print(f"[SIMULATION] analyzer degraded fan={fan_id}: {exc}")
        except asyncio.CancelledError:
            raise

    creator_messages = [
        row
        for row in await _recent_creator_message_rows(fan_id)
        if row["id"] not in creator_ids_before
    ]

    # Reported, not inferred. A turn that sent nothing because the writer stack
    # failed is a broken deployment; a turn that sent nothing because Full Auto
    # decided to is the product working. The simulator must never present the
    # first as the second.
    if creator_messages:
        outcome = AUTO_OUTCOME_REPLIED
    elif analyzer_degraded:
        outcome = AUTO_OUTCOME_ANALYZER_DEGRADED
    else:
        outcome = auto_outcome.get("outcome", AUTO_OUTCOME_NO_SEND)

    print(f"[SIMULATION] turn complete fan={fan_id} outcome={outcome}")
    return {
        "status": "ok",
        "simulation": True,
        "fast": bool(fast),
        "fan_message_id": fan_message_id,
        "creator_messages": creator_messages,
        "analysis_degraded": analyzer_degraded,
        "outcome": outcome,
    }
