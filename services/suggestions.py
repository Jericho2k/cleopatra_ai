"""Suggestion orchestration service.

Coordinates DB, stage classification, RAG, prompt building, and generation
"""

import asyncio
from core.tasks import spawn
import json
import os
import random
import re
import uuid

import httpx

from ai.generator import generate_replies
from ai.writer_router import select_writer_route
from openai import AsyncOpenAI
from core.config import get_settings
from core.supabase import get_supabase
from ai.prompt_builder import build_prompt
from services.commercial_orchestrator import orchestrate
from models.commercial import ActionType, FanStatus
from db.fan_intelligence_queries import get_fan_intelligence_context
from db.commercial_queries import (
    cancel_actions_for_fan,
    get_creator_policy,
    get_fan_state,
    save_fan_state,
    schedule_action,
)
from services.session_planner import plan_session_for_fan
from services.human_delivery import build_delivery_schedule
from services.ppv_delivery import create_ppv_approval_request
from services.followup_lifecycle import (
    complete_session_state,
    next_reconcile_at,
)
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
from services.session_lifecycle import (
    decrement_cooldown,
    mark_step_declined,
    mark_step_purchased,
    mark_step_sent,
)
from ai.situation_analyzer import analyze_situation
from ai.rag import find_similar_exchanges
from ai.stage_classifier import classify_stage
from db.queries import (
    get_conversation_history,
    get_creator_persona,
    get_fan,
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

_pending_auto_replies: dict[str, asyncio.Task] = {}


async def _sleep_while_current(fan_id: str, seconds: float, *, phase: str) -> bool:
    """Sleep in short slices so a newer fan message can cancel simulated typing."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, float(seconds))
    while True:
        current_task = _pending_auto_replies.get(fan_id)
        if current_task and current_task is not asyncio.current_task():
            print(f"[AUTO TIMING] fan={fan_id} cancelled phase={phase} newer_task=true")
            return False
        remaining = deadline - loop.time()
        if remaining <= 0:
            return True
        await asyncio.sleep(min(0.5, remaining))


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


async def get_suggestions(
    fan_id: str,
    creator_id: str,
    fan_message: str,
    creator_name: str = "a creator",
    save_fan_message: bool = True,
) -> SuggestionResponse:
    conversation_history = await get_conversation_history(fan_id)

    fan_profile = await get_fan(creator_id, fan_id)
    if fan_profile is None:
        fan_profile = Fan(id=fan_id, display_name=fan_id)

    fan_intelligence = await get_fan_intelligence_context(fan_id)
    buyer_lifecycle = await get_fan_lifecycle_context(fan_id)
    affordability = await get_affordability_context(fan_id)
    price_learning = await get_price_learning_context(fan_id)
    creator_persona = await get_creator_persona(creator_id)
    if creator_persona is None:
        creator_persona = Persona()
    creator_legend = await get_creator_legend(creator_id)
    ppv_offers = await get_ppv_offers(creator_id)
    sent_ppv = await get_sent_ppv(fan_id)
    active_session = await get_fan_session(fan_id)

    conversation_stage = classify_stage(conversation_history, fan_profile)

    similar_exchanges = await find_similar_exchanges(
        fan_message, creator_id, enabled=False
    )

    ctx_without_situation = ConversationContext(
        fan_message=fan_message,
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
    )

    situation = await analyze_situation(
        ctx_without_situation,
        telemetry_context={"creator_id": creator_id, "fan_id": fan_id},
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
                async with httpx.AsyncClient() as _hc:
                    plan_resp = await _hc.post(
                        f"http://localhost:8080/plan-session/{creator_id}/{fan_id}",
                        timeout=30,
                    )
                plan_data = plan_resp.json()
                if plan_data.get("status") == "ok":
                    active_session = plan_data.get("session") or await get_fan_session(fan_id)
                    print(f"[SESSION] Auto-planned session for fan={fan_id} items={len((active_session or {}).get('plan', []))}")
                else:
                    print(f"[SESSION] plan-session returned status={plan_data.get('status')} fan={fan_id}")
            except Exception as e:
                print(f"[SESSION PLAN ERROR] {e}")

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
        creator_legend=creator_legend,
        fan_intelligence=fan_intelligence,
        buyer_lifecycle=buyer_lifecycle,
        affordability=affordability,
        price_learning=price_learning,
        session_strategy=session_strategy,
        conversation_director=conversation_director,
    )

    route = select_writer_route(ctx)
    print(
        f"[WRITER ROUTE] fan={fan_id} mode=assisted route={route.route.value} "
        f"reason={route.reason} primary={route.primary_target.model} "
        f"fallback={(route.fallback_target.model if route.fallback_target else 'none')}"
    )
    prompt = build_prompt(ctx)
    replies = await generate_replies(
        prompt,
        creator_persona,
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
            ),
            name=f"fan_intelligence:{fan_id}",
        )

    if _should_update_memory(conversation_history):
        spawn(_update_fan_memory(fan_id, creator_id, conversation_history, fan_profile.total_spent), name="update_fan_memory")
        spawn(_update_fan_ai_summary(fan_id, conversation_history), name="update_fan_ai_summary")

    return SuggestionResponse(suggestions=replies, stage=conversation_stage)


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
) -> None:
    try:
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

        response = await together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
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


async def _debounced_auto_reply(fan_id: str, creator_id: str) -> None:
    """Wait for fan to finish typing, then generate and send one reply."""
    try:
        # Short jittered debounce catches rapid multi-message bursts. A separate
        # length-aware delay is applied after generation to simulate reading/typing.
        delay = random.uniform(7.0, 12.0)
        await asyncio.sleep(delay)

        situation: dict | None = None  # initialized early — assigned properly later

        # Fetch history now — after the wait — to get the most complete picture
        # including any messages the fan sent while we were waiting
        conversation_history = await get_conversation_history(fan_id)

        # Stale generation check: if another task is already pending for this fan
        # (meaning a newer message came in during our wait and reset the timer),
        # abort silently — the newer task will generate the reply
        current_task = _pending_auto_replies.get(fan_id)
        if current_task and current_task is not asyncio.current_task():
            print(f"[AUTO REPLY] Newer task exists — aborting stale generation for fan={fan_id}")
            return
        fan_profile = await get_fan_by_id(fan_id)
        if not fan_profile:
            return
        fan_intelligence = await get_fan_intelligence_context(fan_id)
        buyer_lifecycle = await get_fan_lifecycle_context(fan_id)
        affordability = await get_affordability_context(fan_id)
        price_learning = await get_price_learning_context(fan_id)
        # Frozen for human review (e.g. prior crisis under 'freeze' policy): auto-mode
        # stays out until a human clears the flag in the dashboard.
        if getattr(fan_profile, "needs_human_review", False):
            print(f"[AUTO REPLY] fan={fan_id} is frozen for human review — skipping auto-reply")
            return
        fan_tier = getattr(fan_profile, "spend_tier", "cold") or "cold"

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

        creator_persona = await get_creator_persona(creator_id)
        if creator_persona is None:
            creator_persona = Persona()

        ppv_offers = await get_ppv_offers(creator_id)
        sent_ppv = await get_sent_ppv(fan_id)
        active_session = await get_fan_session(fan_id)
        similar_exchanges = await find_similar_exchanges(latest_message, creator_id, enabled=False)
        conversation_stage = classify_stage(conversation_history, fan_profile)

        # A purchased PPV starts a short text-only bridge before the next step.
        # Sending a PPV does NOT advance the plan; only a confirmed purchase does.
        if active_session and active_session.get("post_ppv_cooldown"):
            active_session = decrement_cooldown(active_session)
            await save_fan_session(fan_id, active_session)
            remaining = active_session.get("cooldown_messages_remaining", 0)
            print(f"[SESSION] cooldown fan={fan_id} remaining={remaining}")

        ctx_without_situation = ConversationContext(
            fan_message=latest_message,
            conversation_history=conversation_history,
            fan_profile=fan_profile,
            creator_persona=creator_persona,
            similar_exchanges=similar_exchanges,
            conversation_stage=conversation_stage,
            creator_name="a creator",
            ppv_offers=ppv_offers,
            sent_ppv=sent_ppv,
            active_session=active_session,
            fan_intelligence=fan_intelligence,
            buyer_lifecycle=buyer_lifecycle,
            affordability=affordability,
            price_learning=price_learning,
        )

        situation = await analyze_situation(
            ctx_without_situation,
            telemetry_context={"creator_id": creator_id, "fan_id": fan_id},
        )
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

        # Sticky-situation policy: if a crisis is flagged and the creator opted to
        # freeze rather than continue, stop here — flag for a human, send nothing.
        if await _crisis_freezes_chat(creator_id, fan_id, situation):
            return

        # Commercial layer: deterministic policy decides what happens next
        # (sell / pause / tease / schedule). Flag-gated so it can be turned off
        # instantly in prod without a deploy.
        commercial_enabled = os.environ.get("COMMERCIAL_LAYER_ENABLED", "").lower() in ("1", "true", "yes")
        decision = None
        if commercial_enabled:
            try:
                cap_ok, _ = await _within_daily_caps(creator_id, sent_ppv, fan_profile)
                decision = await orchestrate(
                    creator_id=creator_id,
                    fan_id=fan_id,
                    situation=situation,
                    fan_has_bought_before=bool(getattr(fan_profile, "total_spent", 0)),
                    within_daily_caps=cap_ok,
                    frozen_for_review=bool(getattr(fan_profile, "needs_human_review", False)),
                    active_session=active_session,
                )
            except Exception as e:
                # Full Auto must fail closed. Silently reverting to the legacy
                # planner can send content after a pause or at the wrong price.
                print(f"[COMMERCIAL ERROR] fan={fan_id}: {e} — auto reply aborted")
                return

        # With commercial v2 enabled, only CREATE_PAID_SESSION may start a plan.
        # PRESENT_SESSION_OPTIONS, PAUSE_* and ordinary chat must never invoke the
        # legacy content planner. When the flag is off, preserve old behavior.
        should_plan = (
            decision is not None and decision.action == ActionType.CREATE_PAID_SESSION
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
                        selected_set_ids=(decision.selected_package_set_ids if decision else None),
                        selected_price_cents=(decision.session_budget_cents if decision else None),
                    )
                    if plan_data.get("status") == "ok":
                        active_session = plan_data.get("session") or await get_fan_session(fan_id)
                        print(f"[SESSION] Planned for fan={fan_id} items={len((active_session or {}).get('plan', []))}")
                    else:
                        print(f"[SESSION] plan-session status={plan_data.get('status')} fan={fan_id}")
                        if commercial_enabled:
                            return
                except Exception as e:
                    print(f"[SESSION PLAN ERROR] {e}")
                    if commercial_enabled:
                        return

        # Inject tip context into situation so prompt builder can use it
        if pending_tip:
            situation["pending_tip"] = pending_tip

        # Resend handler — situation analyzer detected fan can't see sent content
        if situation.get("resend_requested") == "true":
            db = get_supabase()
            fan_data = await asyncio.to_thread(
                lambda: db.table("fans")
                .select("pending_ppv_check, fansly_group_id, platform_fan_id")
                .eq("id", fan_id)
                .single()
                .execute()
            )
            pending = (fan_data.data or {}).get("pending_ppv_check")
            group_id_resend = (fan_data.data or {}).get("fansly_group_id")
            creator_data = await asyncio.to_thread(
                lambda: db.table("creators")
                .select("apifansly_account_id")
                .eq("id", creator_id)
                .single()
                .execute()
            )
            apifansly_id_resend = (creator_data.data or {}).get("apifansly_account_id")

            if pending and group_id_resend and apifansly_id_resend:
                media_id_resend = pending.get("media_id")
                price_resend = pending.get("price")
                if media_id_resend and price_resend:
                    print(f"[PPV RESEND] Resending media={media_id_resend} price={price_resend} for fan={fan_id}")
                    try:
                        async with httpx.AsyncClient() as hc:
                            ppv_resp = await hc.post(
                                f"https://v1.apifansly.com/api/fansly/{apifansly_id_resend}/chats/{group_id_resend}/messages",
                                headers={
                                    "x-api-key": os.environ.get("APIFANSLY_API_KEY"),
                                    "Content-Type": "application/json",
                                },
                                json={
                                    "content": "sorry about that, here it is again 😏",
                                    "mediaId": media_id_resend,
                                    "access_type": "ppv",
                                    "price": price_resend,
                                },
                                timeout=10,
                            )
                        print(f"[PPV RESEND] status={ppv_resp.status_code} body={ppv_resp.text[:200]}")
                        await save_message(fan_id, creator_id, "creator", "sorry about that, here it is again 😏", was_ai_suggested=True)
                        return  # Skip normal generation
                    except Exception as e:
                        print(f"[PPV RESEND ERROR] {e}")
                        # Fall through to normal generation if resend fails

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
                spawn(_verify_ppv_purchase(fan_id, creator_id, pending), name="verify_ppv_purchase")

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
            elif decision.action in {ActionType.CREATE_PAID_SESSION, ActionType.RESUME_PREVIOUS_OFFER}:
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
                    state.selected_package_id = None
                    state.selected_package_set_id = None
                    state.selected_package_set_ids = []
                    state.selected_package_price_cents = None
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

        ctx = ConversationContext(
            fan_message=latest_message,
            conversation_history=conversation_history,
            fan_profile=fan_profile,
            creator_persona=creator_persona,
            similar_exchanges=similar_exchanges,
            conversation_stage=conversation_stage,
            creator_name="a creator",
            situation=situation,
            ppv_offers=ppv_offers,
            sent_ppv=sent_ppv,
            active_session=active_session,
            commercial_decision=decision.model_dump(mode="json") if decision else None,
            fan_intelligence=fan_intelligence,
            buyer_lifecycle=buyer_lifecycle,
            affordability=affordability,
            price_learning=price_learning,
            session_strategy=session_strategy,
            conversation_director=conversation_director,
        )

        route = select_writer_route(ctx)
        print(
            f"[WRITER ROUTE] fan={fan_id} mode=auto route={route.route.value} "
            f"reason={route.reason} primary={route.primary_target.model} "
            f"fallback={(route.fallback_target.model if route.fallback_target else 'none')}"
        )
        prompt = build_prompt(ctx)
        replies = await generate_replies(
            prompt,
            creator_persona,
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

        if not replies:
            return

        reply = replies[0]

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

        # If no group_id yet, try to find it from chats list
        if not group_id and apifansly_account_id and platform_fan_id:
            from main import get_or_fetch_group_id

            group_id = await get_or_fetch_group_id(apifansly_account_id, str(platform_fan_id), fan_id)

        parts = [p.strip() for p in reply.split("|") if p.strip()]
        timing = build_delivery_schedule(latest_message, parts)
        print(
            f"[AUTO TIMING] fan={fan_id} parts={len(parts)} "
            f"initial={timing.initial_delay_seconds:.2f}s "
            f"between={list(timing.inter_part_delays_seconds)}"
        )

        if group_id and apifansly_account_id:
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"https://v1.apifansly.com/api/fansly/{apifansly_account_id}/chats/{str(group_id)}/typing",
                        headers={"x-api-key": os.environ.get("APIFANSLY_API_KEY")},
                        timeout=5,
                    )
            except Exception:
                pass

        if not await _sleep_while_current(
            fan_id, timing.initial_delay_seconds, phase="before_part_1"
        ):
            return

        for i, part in enumerate(parts):
            if i > 0:
                inter_delay = timing.inter_part_delays_seconds[i - 1]
                if not await _sleep_while_current(
                    fan_id, inter_delay, phase=f"before_part_{i + 1}"
                ):
                    return

            ppv_match = re.search(r"\[PPV:([^:]+):(\d+(?:\.\d+)?)\]", part)
            if ppv_match and commercial_enabled and (
                decision is None
                or decision.action not in {ActionType.CREATE_PAID_SESSION, ActionType.SEND_NEXT_PPV_STEP}
            ):
                print(f"[COMMERCIAL GUARD] stripped unauthorized PPV fan={fan_id} action={getattr(decision, 'action', None)}")
                ppv_match = None
                part = re.sub(r"\[PPV:[^\]]+\]", "", part).strip()
            if ppv_match:
                delivery_reference = uuid.uuid4().hex
                text_out = part[: ppv_match.start()].strip()
                media_id = ppv_match.group(1)
                price = float(ppv_match.group(2))

                # If a session is active, send the whole bundle, not just the tagged id.
                media_ids = [media_id]
                if active_session:
                    plan = active_session.get("plan", [])
                    idx = int(active_session.get("current_index", 0) or 0)
                    if idx < len(plan) and plan[idx].get("media_ids"):
                        # The stored plan is authoritative. The writer may emit a
                        # delivery command, but it cannot alter media or price.
                        media_ids = plan[idx]["media_ids"]
                        media_id = media_ids[0]
                        price = float(plan[idx].get("price") or price)

                current_step = None
                if active_session:
                    plan = active_session.get("plan", [])
                    idx = int(active_session.get("current_index", 0) or 0)
                    if 0 <= idx < len(plan):
                        current_step = plan[idx]
                ppv_media_context = {
                    "ppv": {
                        "media_ids": media_ids,
                        "media_id": media_id,
                        "price": price,
                        "access_type": "ppv",
                        "set_id": (current_step or {}).get("set_id"),
                        "step_index": (active_session or {}).get("current_index"),
                        "payment_reference": delivery_reference,
                    }
                }

                approval_policy = await get_creator_policy(creator_id)
                if approval_policy.require_operator_ppv_approval:
                    state_for_approval = await get_fan_state(fan_id)
                    approval = await create_ppv_approval_request(
                        creator_id=creator_id,
                        fan_id=fan_id,
                        message_content=text_out,
                        media_ids=media_ids,
                        price_cents=int(round(price * 100)),
                        set_id=(current_step or {}).get("set_id"),
                        step_index=(active_session or {}).get("current_index"),
                        approved_experience=(
                            state_for_approval.desired_experience
                            or state_for_approval.selected_package_label
                        ),
                    )
                    print(
                        f"[PPV APPROVAL] fan={fan_id} request={approval.get('id')} "
                        f"media={media_ids} price_cents={int(round(price * 100))}"
                    )
                    return
            else:
                text_out = re.sub(r"\[PPV:[^\]]+\]", "", part).strip()
                ppv_media_context = None

            # Platform acceptance is authoritative. Never create a local sent
            # message or PAYMENT_PENDING state for a delivery that failed.
            if not group_id or not apifansly_account_id:
                print(f"[AUTO DELIVERY ERROR] fan={fan_id}: no live delivery route")
                if ppv_match:
                    await freeze_fan_for_review(fan_id, "ppv_delivery_route_missing")
                return

            platform_message_id = None
            try:
                if ppv_match:
                    ppv_content = text_out if text_out else random.choice([
                        "here it is 😏",
                        "just for you...",
                        "this is what I've been saving 😈",
                        "don't say I never spoil you 💋",
                    ])
                    async with httpx.AsyncClient() as hc:
                        ppv_resp = await hc.post(
                            f"https://v1.apifansly.com/api/fansly/{apifansly_account_id}/chats/{str(group_id)}/messages",
                            headers={
                                "x-api-key": os.environ.get("APIFANSLY_API_KEY"),
                                "Content-Type": "application/json",
                            },
                            json={
                                "content": ppv_content,
                                "mediaIds": media_ids,
                                "mediaId": media_id,
                                "access_type": "ppv",
                                "price": price,
                            },
                            timeout=10,
                        )
                    ppv_resp.raise_for_status()
                    try:
                        response_body = ppv_resp.json()
                        platform_message_id = str(
                            response_body.get("id")
                            or response_body.get("data", {}).get("id")
                            or ""
                        ) or None
                    except Exception:
                        platform_message_id = None
                    print(
                        f"[PPV SEND] status={ppv_resp.status_code} media={media_id} "
                        f"price={price} reference={delivery_reference}"
                    )
                else:
                    from main import send_fansly_message

                    delivered = await send_fansly_message(
                        apifansly_account_id, str(group_id), text_out
                    )
                    if not delivered:
                        raise RuntimeError("platform rejected text delivery")
            except Exception as exc:
                print(f"[AUTO DELIVERY ERROR] fan={fan_id}: {exc}")
                if ppv_match:
                    await freeze_fan_for_review(fan_id, "ppv_send_failed")
                return

            try:
                await save_message(
                    fan_id=fan_id,
                    creator_id=creator_id,
                    role="creator",
                    content=text_out,
                    was_ai_suggested=True,
                    media_context=ppv_media_context,
                )
            except Exception as exc:
                # Delivery already happened. Freeze rather than retrying and
                # risking a duplicate message on the live account.
                await freeze_fan_for_review(fan_id, "delivery_sent_but_not_persisted")
                print(f"[AUTO PERSIST ERROR] fan={fan_id}: {exc}")
                return

            if ppv_match:
                try:
                    from datetime import datetime, timedelta, timezone

                    policy = await get_creator_policy(creator_id)
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
                        "sent_at": sent_at.isoformat(),
                        "expires_at": expires_at.isoformat(),
                        "verification_attempts": 0,
                        "platform_message_id": platform_message_id,
                    }
                    db = get_supabase()
                    await asyncio.to_thread(
                        lambda: db.table("fans").update({
                            "pending_ppv_check": pending_check
                        }).eq("id", fan_id).execute()
                    )

                    session = await get_fan_session(fan_id)
                    if session:
                        session = mark_step_sent(
                            session,
                            message_id=platform_message_id,
                        )
                        await save_fan_session(fan_id, session)
                        active_session = session

                    state = await get_fan_state(fan_id)
                    state.status = FanStatus.PAYMENT_PENDING
                    await save_fan_state(fan_id, creator_id, state)

                    reconcile_at = next_reconcile_at(
                        sent_at,
                        expires_at=expires_at,
                        recheck_minutes=policy.ppv_recheck_minutes,
                    )
                    await schedule_action(
                        creator_id=creator_id,
                        fan_id=fan_id,
                        action_type="PPV_RECONCILE",
                        execute_at=reconcile_at,
                        payload={"payment_reference": delivery_reference},
                        dedupe_key=f"ppv-reconcile:{fan_id}:{delivery_reference}",
                    )
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

    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"[DEBOUNCED AUTO REPLY ERROR] fan={fan_id} error={e}")
        import traceback
        traceback.print_exc()
    finally:
        _pending_auto_replies.pop(fan_id, None)


_REACTION_FISHING_LINES = [
    "let me know what you think 🙈",
    "tell me how you feel about it...",
    "dying to know your reaction 😏",
    "don't leave me hanging",
    "what do you think? 👀",
    "hope it was worth the wait",
    "your reaction is everything to me rn",
]


async def _send_post_purchase_reaction(fan_id: str, creator_id: str) -> None:
    """Send a reaction prompt only after the platform confirmed the unlock."""
    try:
        await asyncio.sleep(random.uniform(20.0, 55.0))
        db = get_supabase()
        fan_row = await asyncio.to_thread(
            lambda: db.table("fans")
            .select("fansly_group_id")
            .eq("id", fan_id)
            .single()
            .execute()
        )
        creator_row = await asyncio.to_thread(
            lambda: db.table("creators")
            .select("apifansly_account_id")
            .eq("id", creator_id)
            .single()
            .execute()
        )
        group_id = (fan_row.data or {}).get("fansly_group_id")
        account_id = (creator_row.data or {}).get("apifansly_account_id")
        line = random.choice(_REACTION_FISHING_LINES)
        if not group_id or not account_id:
            print(f"[PPV REACTION] fan={fan_id} skipped=no_live_delivery_route")
            return
        from main import send_fansly_message

        delivered = await send_fansly_message(account_id, str(group_id), line)
        if not delivered:
            print(f"[PPV REACTION] fan={fan_id} skipped=platform_rejected")
            return
        try:
            await save_message(fan_id, creator_id, "creator", line, was_ai_suggested=True)
        except Exception as exc:
            await freeze_fan_for_review(fan_id, "reaction_sent_but_not_persisted")
            print(f"[PPV REACTION PERSIST ERROR] fan={fan_id}: {exc}")
            return
        print(f"[PPV REACTION] fan={fan_id} purchase_confirmed=true line={line}")
    except Exception as e:
        print(f"[PPV REACTION ERROR] {e}")


async def record_ppv_purchase(fan_id: str, media_id: str, amount: float | None = None) -> None:
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
    pending = row.get("pending_ppv_check") or {}
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

    already_recorded = any(str(entry.get("media_id")) == str(media_id) for entry in sales_log)
    old_spent = int(row.get("total_spent", 0) or 0)
    new_spent = old_spent
    if not already_recorded:
        from datetime import datetime
        sales_log.append({
            "date": datetime.utcnow().strftime("%d.%m.%Y"),
            "item": f"PPV media {media_id}",
            "media_id": str(media_id),
            "media_ids": pending.get("media_ids") or [str(media_id)],
            "amount": amount_dollars,
            "chatter": "Operator" if pending.get("source") == "operator" else "AI",
        })
        not_sold = [entry for entry in not_sold if str(media_id) not in str(entry.get("item", ""))]
        new_spent = old_spent + amount_dollars

    await asyncio.to_thread(
        lambda: db.table("fans").update({
            "total_spent": new_spent,
            "spend_tier": _tier(new_spent),
            "sales_log": sales_log,
            "not_sold_log": not_sold,
            "pending_ppv_check": None,
        }).eq("id", fan_id).execute()
    )
    await mark_ppv_purchased(fan_id, str(media_id))
    if creator_id:
        await cancel_actions_for_fan(fan_id, "PPV_RECONCILE")
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

    if session and creator_id:
        try:
            policy = await get_creator_policy(creator_id)
            updated, completed = mark_step_purchased(
                session,
                media_id=str(media_id),
                set_id=(pending or {}).get("set_id"),
                amount_cents=amount_dollars * 100,
                cooldown_messages=policy.post_purchase_cooldown_messages,
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
    elif creator_id:
        # A one-off operator PPV has no executable paid-session plan. Once the
        # purchase is confirmed, clear the payment hold without inventing a
        # session lifecycle.
        state = await get_fan_state(fan_id)
        state.status = FanStatus.IDLE
        state.selected_package_id = None
        state.selected_package_set_id = None
        state.selected_package_set_ids = []
        state.selected_package_label = None
        state.selected_package_price_cents = None
        await save_fan_state(fan_id, creator_id, state)

    if creator_id and not already_recorded:
        spawn(
            _send_post_purchase_reaction(fan_id, creator_id),
            name="send_post_purchase_reaction",
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

        rows = await asyncio.to_thread(
            lambda: db.table("fans")
            .select("id, creator_id, pending_ppv_check")
            .not_.is_("pending_ppv_check", "null")
            .execute()
        )

        if not rows.data:
            return

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=5)
        stale = []

        for row in rows.data:
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


def schedule_auto_reply(fan_id: str, creator_id: str) -> None:
    """Cancel any pending reply for this fan and schedule a new one."""
    existing = _pending_auto_replies.get(fan_id)
    if existing and not existing.done():
        existing.cancel()
        print(f"[AUTO REPLY] Reset timer for fan={fan_id}")

    task = asyncio.create_task(_debounced_auto_reply_with_sleep_check(fan_id, creator_id))

    # Log any unhandled exceptions so they don't disappear silently
    def _on_task_done(t: asyncio.Task) -> None:
        if not t.cancelled() and t.exception():
            import traceback
            print(f"[AUTO REPLY TASK ERROR] fan={fan_id}")
            traceback.print_exception(type(t.exception()), t.exception(), t.exception().__traceback__)

    task.add_done_callback(_on_task_done)
    _pending_auto_replies[fan_id] = task


async def _debounced_auto_reply_with_sleep_check(fan_id: str, creator_id: str) -> None:
    """Check sleep hours before firing auto reply."""
    try:
        from datetime import datetime, timezone
        db = get_supabase()
        creator_row = await asyncio.to_thread(
            lambda: db.table("creators")
            .select("sleep_hours_start, sleep_hours_end")
            .eq("id", creator_id)
            .single()
            .execute()
        )
        data = creator_row.data or {}
        sleep_start = data.get("sleep_hours_start", 0)
        sleep_end = data.get("sleep_hours_end", 7)

        current_hour = datetime.now(timezone.utc).hour
        if sleep_start <= sleep_end:
            in_sleep = sleep_start <= current_hour < sleep_end
        else:
            in_sleep = current_hour >= sleep_start or current_hour < sleep_end

        if in_sleep:
            # Calculate seconds until sleep ends
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            wake_hour = sleep_end
            if now.hour < wake_hour:
                seconds_until_wake = (wake_hour - now.hour) * 3600 - now.minute * 60 - now.second
            else:
                seconds_until_wake = (24 - now.hour + wake_hour) * 3600 - now.minute * 60 - now.second
            # Add small random offset so replies don't all fire at exactly wake time
            seconds_until_wake += random.randint(60, 600)
            print(f"[AUTO REPLY] Sleep hours active — queuing reply in {seconds_until_wake//60}min for fan={fan_id}")
            await asyncio.sleep(seconds_until_wake)
            # Re-check sleep hours after waking (in case settings changed)
            creator_row2 = await asyncio.to_thread(
                lambda: db.table("creators")
                .select("sleep_hours_start, sleep_hours_end")
                .eq("id", creator_id)
                .single()
                .execute()
            )
            data2 = creator_row2.data or {}
            new_start = data2.get("sleep_hours_start", 0)
            new_end = data2.get("sleep_hours_end", 7)
            new_hour = datetime.now(timezone.utc).hour
            if new_start <= new_end:
                still_sleeping = new_start <= new_hour < new_end
            else:
                still_sleeping = new_hour >= new_start or new_hour < new_end
            if still_sleeping:
                print(f"[AUTO REPLY] Still in sleep hours after wake — skipping fan={fan_id}")
                return
            print(f"[AUTO REPLY] Sleep ended — sending queued reply for fan={fan_id}")

        await _debounced_auto_reply(fan_id, creator_id)
    except Exception as e:
        print(f"[SLEEP CHECK ERROR] {e}")
        await _debounced_auto_reply(fan_id, creator_id)


def _should_update_memory(conversation_history: list[Message]) -> bool:
    count = len([m for m in conversation_history if m.role == "fan"])
    return count > 0 and count % 10 == 0
