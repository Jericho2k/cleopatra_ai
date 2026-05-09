"""Suggestion orchestration service.

Coordinates DB, stage classification, RAG, prompt building, and generation
"""

import asyncio
import json
import os
import random
import re

import httpx

from ai.generator import generate_replies
from openai import AsyncOpenAI
from core.config import get_settings
from core.supabase import get_supabase
from ai.prompt_builder import build_prompt
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
    save_fan_session,
    save_message,
    update_fan_memory,
    update_fan_ai_summary,
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

    creator_persona = await get_creator_persona(creator_id)
    if creator_persona is None:
        creator_persona = Persona()
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
    )

    situation = await analyze_situation(ctx_without_situation)

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
    )

    prompt = build_prompt(ctx)
    replies = await generate_replies(prompt, creator_persona)

    if save_fan_message:
        await save_message(fan_id, creator_id, "fan", fan_message)

    if _should_update_memory(conversation_history):
        asyncio.create_task(_update_fan_memory(fan_id, creator_id, conversation_history, fan_profile.total_spent))
        asyncio.create_task(_update_fan_ai_summary(fan_id, conversation_history))

    return SuggestionResponse(suggestions=replies, stage=conversation_stage)


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
            '  "preferences": ["list of content preferences, kinks, or fetishes mentioned or implied"],\n'
            '  "member_note": "Fill in the Member template below with what you know. Leave fields blank if unknown.\\n'
            'Age: \nLocation: \nInterests/hobbies: \nKinks: \nAdditional info: ",\n'
            '  "model_note": "Fill in the Model template below — what has the creator revealed about herself to THIS fan specifically. Leave fields blank if unknown.\\n'
            'Name used: \nLocation told: \nBackground story told: \nKinks shared: \nOther personal details told: "\n'
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
        model_note = data.get("model_note", "")

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
            '  "location": "city/country if mentioned, otherwise null",\n'
            '  "occupation": "job or income signals if mentioned, otherwise null",\n'
            '  "relationship_status": "single/relationship/married/unknown",\n'
            '  "payday": "when they get paid if mentioned, otherwise null",\n'
            '  "kinks": ["list of explicit kinks, fetishes, or sexual preferences mentioned or clearly implied"],\n'
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
        delay = 8  # TEST MODE — slightly longer to catch fast multi-message fans
        await asyncio.sleep(delay)

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

        # Budget qualification gate (NOW safe — latest_message exists)
        if active_session and not active_session.get("budget_qualified"):
            budget_signals = [
                "i have", "i got", "budget", "i can spend", "how much",
                "what's the cheapest", "i'll pay", "send me", "i want to buy",
                "let's do", "let's play", "yes", "sure", "okay", "yeah",
            ]
            if any(s in latest_message.lower() for s in budget_signals):
                active_session["budget_qualified"] = True
                await save_fan_session(fan_id, active_session)
                print(f"[SESSION] Fan budget-qualified for fan={fan_id}")

        # Decrement post-PPV cooldown counter on each fan message
        if active_session and active_session.get("post_ppv_cooldown"):
            remaining = active_session.get("cooldown_messages_remaining", 0) - 1
            if remaining <= 0:
                active_session["post_ppv_cooldown"] = False
                active_session["cooldown_messages_remaining"] = 0
                print(f"[SESSION] Cooldown lifted for fan={fan_id}")
            else:
                active_session["cooldown_messages_remaining"] = remaining
                print(f"[SESSION] Cooldown active, {remaining} messages remaining for fan={fan_id}")
            await save_fan_session(fan_id, active_session)

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
        )

        situation = await analyze_situation(ctx_without_situation)

        # Inject tip context into situation so prompt builder can use it
        if pending_tip:
            situation["pending_tip"] = pending_tip

        # Check if fan is reacting to a pending PPV
        purchase_signal = situation.get("purchase_signal", "none")
        if purchase_signal in ("bought", "declined"):
            db = get_supabase()
            fan_data = await asyncio.to_thread(
                lambda: db.table("fans")
                .select("pending_ppv_check")
                .eq("id", fan_id)
                .single()
                .execute()
            )
            pending = (fan_data.data or {}).get("pending_ppv_check")
            if pending:
                print(f"[PPV SIGNAL] fan={fan_id} signal={purchase_signal} pending={pending}")
                asyncio.create_task(_verify_ppv_purchase(fan_id, creator_id, pending))

            if purchase_signal == "declined" and active_session:
                # Fan declined — find cheapest unsent item below declined price
                # and inject it into session as a retry at a lower price point
                try:
                    declined_price = (pending or {}).get("price", 999)
                    session = await get_fan_session(fan_id)
                    if session:
                        plan = session.get("plan", [])
                        idx = session.get("current_index", 0)
                        remaining_items = [
                            p for p in plan[idx:]
                            if not p.get("sent") and float(p.get("price", 999)) < float(declined_price) * 0.75
                        ]
                        if remaining_items:
                            # Surface cheapest alternative
                            cheaper = min(remaining_items, key=lambda x: x.get("price", 999))
                            # Remove from original position first, then insert at current index
                            original_idx = plan.index(cheaper)
                            plan.pop(original_idx)
                            insert_at = idx if original_idx >= idx else idx - 1
                            plan.insert(insert_at, cheaper)
                            await save_fan_session(fan_id, session)
                            print(f"[SESSION] Declined ${declined_price} — surfaced cheaper item ${cheaper.get('price')} for fan={fan_id}")
                        else:
                            # No cheaper item available — log ceiling and clear session
                            fan_profile_row = await asyncio.to_thread(
                                lambda: get_supabase().table("fans")
                                .select("ai_summary")
                                .eq("id", fan_id)
                                .single()
                                .execute()
                            )
                            summary = (fan_profile_row.data or {}).get("ai_summary") or {}
                            # Store ceiling as just below the declined price
                            summary["price_ceiling"] = round(float(declined_price) * 0.8)
                            await asyncio.to_thread(
                                lambda: get_supabase().table("fans")
                                .update({"ai_summary": summary})
                                .eq("id", fan_id)
                                .execute()
                            )
                            print(f"[SESSION] No cheaper items — stored price_ceiling=${summary['price_ceiling']} for fan={fan_id}")
                except Exception as e:
                    print(f"[SESSION DECLINE HANDLER ERROR] {e}")

        # Auto-trigger session planning if situation calls for it and no session active
        if not active_session and situation:
            move = situation.get("strategic_move", "")
            fan_intent = situation.get("fan_intent", "").lower()
            session_triggers = ["push_for_ppv", "hint_at_content", "build_tension"]
            intent_triggers = ["want", "show", "play", "buy", "see", "content", "hot", "sexy"]
            # Require at least 8 fan messages before planning a session
            # to avoid triggering on casual warmup messages
            fan_msg_count = len([m for m in conversation_history if m.role == "fan"])
            should_plan = fan_msg_count >= 8 and (
                move in session_triggers
                or any(t in fan_intent for t in intent_triggers)
                or any(
                    t in latest_message.lower()
                    for t in ["let's play", "show me", "what do you have", "i want to see", "send me"]
                )
            )
            if should_plan:
                try:
                    import httpx as _httpx

                    async with _httpx.AsyncClient() as _hc:
                        plan_resp = await _hc.post(
                            f"http://localhost:8080/plan-session/{creator_id}/{fan_id}",
                            timeout=30,
                        )
                        plan_data = plan_resp.json()
                        if plan_data.get("status") == "ok":
                            active_session = plan_data.get("session")
                            if not active_session:
                                active_session = await get_fan_session(fan_id)
                            print(
                                f"[SESSION] Auto-planned session for fan={fan_id} items={len((active_session or {}).get('plan', []))}"
                            )
                except Exception as e:
                    print(f"[SESSION PLAN ERROR] {e}")

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
        )

        prompt = build_prompt(ctx)
        replies = await generate_replies(prompt, creator_persona)

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

        await asyncio.sleep(random.randint(3, 8))

        parts = [p.strip() for p in reply.split("|") if p.strip()]

        for i, part in enumerate(parts):
            if i > 0:
                await asyncio.sleep(random.randint(5, 15))

            ppv_match = re.search(r"\[PPV:([^:]+):(\d+(?:\.\d+)?)\]", part)
            if ppv_match:
                text_out = part[: ppv_match.start()].strip()
                media_id = ppv_match.group(1)
                price = float(ppv_match.group(2))
                ppv_media_context = {
                    "ppv": {
                        "media_id": media_id,
                        "price": price,
                        "access_type": "ppv",
                    }
                }
            else:
                text_out = re.sub(r"\[PPV:[^\]]+\]", "", part).strip()
                ppv_media_context = None

            await save_message(
                fan_id=fan_id,
                creator_id=creator_id,
                role="creator",
                content=text_out,
                was_ai_suggested=True,
                media_context=ppv_media_context,
            )

            # After PPV sent: store pending check + queue reaction fishing follow-up
            if ppv_match:
                try:
                    db = get_supabase()
                    await asyncio.to_thread(
                        lambda mid=media_id, pr=price: db.table("fans").update({
                            "pending_ppv_check": {
                                "media_id": mid,
                                "price": pr,
                                "sent_at": __import__("datetime").datetime.utcnow().isoformat(),
                            }
                        }).eq("id", fan_id).execute()
                    )
                    # Send reaction fishing follow-up after short delay
                    asyncio.create_task(_send_reaction_fishing(fan_id, creator_id, group_id, apifansly_account_id))
                except Exception as e:
                    print(f"[PPV PENDING ERROR] {e}")

            # Advance session plan if PPV was sent
            if ppv_match and active_session:
                try:
                    session = await get_fan_session(fan_id)
                    if session:
                        plan = session.get("plan", [])
                        idx = session.get("current_index", 0)
                        if idx < len(plan):
                            plan[idx]["sent"] = True
                            session["current_index"] = idx + 1
                            # Start cooldown — require 2 fan messages before next item
                            session["post_ppv_cooldown"] = True
                            session["cooldown_messages_remaining"] = 2
                            await save_fan_session(fan_id, session)
                            active_session = session
                            print(f"[SESSION] Advanced to item {idx + 1}/{len(plan)} for fan={fan_id}, cooldown started")
                except Exception as e:
                    print(f"[SESSION ADVANCE ERROR] {e}")

            if group_id and apifansly_account_id:
                if ppv_match:
                    async with httpx.AsyncClient() as hc:
                        await hc.post(
                            f"https://v1.apifansly.com/api/fansly/{apifansly_account_id}/chats/{str(group_id)}/messages",
                            headers={
                                "x-api-key": os.environ.get("APIFANSLY_API_KEY"),
                                "Content-Type": "application/json",
                            },
                            json={
                                "content": text_out,
                                "mediaId": media_id,
                                "access_type": "ppv",
                                "price": price,
                            },
                            timeout=10,
                        )
                else:
                    from main import send_fansly_message

                    await send_fansly_message(apifansly_account_id, str(group_id), text_out)

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


async def _send_reaction_fishing(
    fan_id: str,
    creator_id: str,
    group_id: str | None,
    apifansly_account_id: str | None,
) -> None:
    """Send a natural follow-up line after PPV to fish for purchase reaction."""
    try:
        await asyncio.sleep(random.randint(30, 90))
        line = random.choice(_REACTION_FISHING_LINES)
        await save_message(fan_id, creator_id, "creator", line, was_ai_suggested=True)
        if group_id and apifansly_account_id:
            from main import send_fansly_message
            await send_fansly_message(apifansly_account_id, group_id, line)
        print(f"[PPV REACTION] Sent fishing line to fan={fan_id}: {line}")
    except Exception as e:
        print(f"[PPV REACTION ERROR] {e}")


async def _verify_ppv_purchase(
    fan_id: str,
    creator_id: str,
    pending: dict,
) -> None:
    """Call earnings API to verify if fan purchased the PPV."""
    import httpx
    try:
        db = get_supabase()
        creator_row = await asyncio.to_thread(
            lambda: db.table("creators")
            .select("apifansly_account_id, fansly_account_id")
            .eq("id", creator_id)
            .single()
            .execute()
        )
        apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
        fan_row = await asyncio.to_thread(
            lambda: db.table("fans")
            .select("platform_fan_id, total_spent, not_sold_log")
            .eq("id", fan_id)
            .single()
            .execute()
        )
        platform_fan_id = (fan_row.data or {}).get("platform_fan_id")
        current_spent = (fan_row.data or {}).get("total_spent", 0)
        api_key = os.environ.get("APIFANSLY_API_KEY")
        expected_price = float(pending.get("price", 0))
        sent_at_str = pending.get("sent_at", "")

        # Convert sent_at to unix ms for the 'after' param
        try:
            from datetime import datetime, timezone
            sent_dt = datetime.fromisoformat(sent_at_str.replace("Z", "+00:00"))
            after_ms = int(sent_dt.timestamp() * 1000)
        except Exception:
            after_ms = 0

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://v1.apifansly.com/api/fansly/{apifansly_id}/earnings/fans/{platform_fan_id}/stats",
                headers={"x-api-key": api_key},
                params={"after": after_ms},
                timeout=15,
            )
            data = resp.json()
            transactions = data.get("data", {}).get("data", {}).get("response", [])

        # Look for media purchase (type 2110) matching expected price
        purchased = False
        actual_amount = 0
        for tx in transactions:
            if tx.get("type") == 2110:
                gross = tx.get("totalGross", 0)
                # Match within 10% tolerance
                if abs(gross - expected_price) / max(expected_price, 1) < 0.1:
                    purchased = True
                    actual_amount = gross
                    break

        media_id = pending.get("media_id", "")

        if purchased:
            print(f"[PPV VERIFY] fan={fan_id} PURCHASED media={media_id} amount=${actual_amount}")
            # Update price floor — we know they'll pay at least this much
            new_spent = current_spent + int(actual_amount)
            fan_summary_row = await asyncio.to_thread(
                lambda: db.table("fans")
                .select("ai_summary")
                .eq("id", fan_id)
                .single()
                .execute()
            )
            summary = (fan_summary_row.data or {}).get("ai_summary") or {}
            existing_floor = summary.get("price_floor", 0)
            if actual_amount > existing_floor:
                summary["price_floor"] = int(actual_amount)
            await asyncio.to_thread(
                lambda: db.table("fans").update({
                    "total_spent": new_spent,
                    "pending_ppv_check": None,
                    "ai_summary": summary,
                }).eq("id", fan_id).execute()
            )
            # Update session plan item as purchased
            session = await get_fan_session(fan_id)
            if session:
                for item in session.get("plan", []):
                    if item.get("media_id") == media_id:
                        item["purchased"] = True
                await save_fan_session(fan_id, session)
        else:
            print(f"[PPV VERIFY] fan={fan_id} did NOT purchase media={media_id}")
            # Log to not_sold
            not_sold = (fan_row.data or {}).get("not_sold_log") or []
            from datetime import datetime
            not_sold.append({
                "date": datetime.utcnow().strftime("%d.%m.%Y"),
                "item": f"PPV media {media_id}",
                "amount": expected_price,
                "reason": "fan indicated no purchase",
                "chatter": "AI",
            })
            await asyncio.to_thread(
                lambda: db.table("fans").update({
                    "not_sold_log": not_sold,
                    "pending_ppv_check": None,
                }).eq("id", fan_id).execute()
            )

    except Exception as e:
        print(f"[PPV VERIFY ERROR] {e}")


async def sweep_stale_ppv_checks() -> None:
    """
    Background sweep: find fans with a pending_ppv_check older than 20 minutes
    that were never resolved by the reaction-triggered path, and verify them now.
    Called every 15 minutes from the lifespan scheduler.
    """
    from datetime import datetime, timezone, timedelta

    try:
        db = get_supabase()

        # Pull all fans that still have a pending check, along with their creator_id
        rows = await asyncio.to_thread(
            lambda: db.table("fans")
            .select("id, creator_id, pending_ppv_check")
            .not_.is_("pending_ppv_check", "null")
            .execute()
        )

        if not rows.data:
            return

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=20)
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

        print(f"[PPV SWEEP] Found {len(stale)} stale pending check(s) to verify")

        for fan_id, creator_id, pending in stale:
            try:
                await _verify_ppv_purchase(fan_id, creator_id, pending)
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

