"""Turn an old conversation into compact, evidence-backed durable memory.

An existing fan may have five thousand messages behind them. None of them
belong in a writer prompt, and asking the conversational writer to read them
would cost more than the relationship is worth. So history is COMPACTED: read
once, chronologically, in bounded chunks, by a cheap model, into the fan-fact
record the product already has.

What this module is
-------------------
* A separate, cheap extraction stage (``STAGE_HISTORY_EXTRACTION``, Together
  ``zai-org/GLM-5.3-Flash`` at $0.15/M in and $0.50/M out). It exists because
  the input is an archive, and archive-sized input is the one thing the
  conversational writer must never be handed.
* A reuser, not a replacement. Facts go into public.fan_facts through the same
  validation and the same merge rules as live extraction. There is one fan
  memory. Building a second one would mean the contradiction rules, the
  explicit-only rules and the money rules all had to be written twice, and the
  second copy would be the one that was wrong.

What this module is NOT
-----------------------
* It is NOT the conversational writer. Kimi still writes every reply.
* It is NOT the live fan-intelligence extractor. That stage is deliberately
  left exactly as it was.
* It has NO authority over live facts. A historical proposal can create a fact,
  reinforce one, and replace a guess — it can never overturn something the live
  path established explicitly. See ``plan_fact_merge(historical=True)``.
* It never calls the platform. Compaction reads rows already in the database.

Evidence discipline
-------------------
A historical chunk shows the model messages from BOTH speakers, which creates
two failure modes live extraction does not have: a quote that appears nowhere,
and a creator statement attributed to the fan. Both are rejected here,
deterministically, before the model's output can influence anything:

  1. the proposal must name a source message id that is IN this chunk;
  2. that message must be the FAN's;
  3. the evidence quote must actually occur in that message's text;
  4. then every existing live rule applies unchanged — allowed keys, the
     explicit-only set, money parsed from the quote, confidence floors.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ai.model_providers import complete, get_runtime_target
from ai.stack_profiles import STAGE_HISTORY_EXTRACTION, get_profile
from db.fan_history_queries import get_backfill_state, update_backfill_state
from models.fan_intelligence import (
    HistoricalExtractionEnvelope,
    ProposedHistoricalObservation,
    SOURCE_TYPE_HISTORICAL_MESSAGE,
    ValidatedObservation,
)
from models.model_runtime import ModelTelemetryContext
from services.fan_history import all_local_history
from services.fan_intelligence import (
    _merge_one,
    _first_json_object,
    fan_intelligence_enabled,
    validate_observation,
)
from services.model_telemetry import record_model_result


# Messages per extraction chunk. Bounded so one call's input stays small and
# cheap, and so a failed chunk loses one chunk rather than an archive.
HISTORY_CHUNK_MESSAGES = 40

# Chunks per compaction run. Bounded for the same reason paging is bounded:
# background work must be interruptible and must not monopolise anything.
HISTORY_CHUNKS_PER_RUN = 5

# Characters of one message shown to the extractor. A fan pasting an essay
# should not be able to blow out the chunk's token budget on its own.
HISTORY_MESSAGE_CHARS = 600

# How much continuity state is kept. Compact by construction: this is durable
# state a writer reads, not a transcript.
MAX_ONGOING_TOPICS = 8
MAX_COMMERCIAL_CONTEXT = 6
MAX_RELATIONSHIP_SUMMARY_CHARS = 600


_SYSTEM_PROMPT = """You compact OLD conversation history on a paid adult creator platform into durable CRM facts about the FAN.

Return only valid JSON in this exact shape:
{"observations": [...], "ongoing_topics": [...], "commercial_context": [...], "relationship_summary": "..."}

You are given a numbered, chronological excerpt of an ARCHIVED conversation. Each line is labelled with a message id and a speaker.

Allowed fact_key values:
preferred_name, age, location, timezone, occupation, relationship_status,
usual_availability, weekday_availability, weekend_availability, payday,
content_interest, disliked_content, kink_interest, preferred_tone,
preferred_dynamic, preferred_format, hard_limit, stated_budget_cents,
accepted_price_cents, rejected_price_cents, counteroffer_cents,
price_sensitivity, purchase_intent, objection_pattern.

Rules:
- Every observation MUST carry source_message_id naming the exact FAN message it came from, and evidence MUST be a short exact quote from that message. A quote that is not in that message is discarded.
- Only FAN messages are sources. Never extract a fact from a CREATOR line, and never record something the creator said as a fact about the fan.
- One atomic fact per observation. Do not return lists inside value.
- certainty is only "explicit" or "strong_inference". Use strong_inference rarely.
- age, payday, hard limits and all money facts require explicit wording.
- Money values must be integer cents: $40 -> 4000.
- Do not claim a purchase happened. Purchases come from payment events.
- Do not invent a relationship, a closeness or a history that the messages do not show.
- Empty observations is correct when nothing durable was said.

ongoing_topics: up to 8 short phrases for threads of this conversation that were still live — things referred back to, plans made, subjects returned to. Omit one-off small talk.
commercial_context: up to 6 short phrases for what actually happened commercially and is SUPPORTED by the excerpt — prices discussed, objections raised, what he bought or declined. Leave empty if the excerpt shows none.
relationship_summary: at most three sentences describing where this conversation stood. Only what the excerpt supports. No speculation about feelings, wealth or intentions.

Observation shape:
{"category":"commercial","fact_key":"payday","value":"Friday","certainty":"explicit","confidence":0.98,"evidence":"I get paid Friday","source_message_id":"m1042"}
"""


def history_extraction_enabled() -> bool:
    """Whether historical compaction may run.

    Gated on the live fan-intelligence switch as well as its own: compaction
    writes into the same fan_facts table, so a deployment that has deliberately
    not enabled fan intelligence must not acquire it through the back door of a
    history import.
    """
    if not fan_intelligence_enabled():
        return False
    return os.getenv("HISTORY_EXTRACTION_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _message_key(row: dict[str, Any]) -> str:
    """The identifier a proposal must name: platform id, else local row id.

    Platform identity is preferred because that is what the fan's message
    actually is, and it is stable across a re-import. A locally originated row
    (an operator send before platform confirmation) has only its row id, and a
    proposal naming one of those is rejected anyway for being a creator line.
    """
    return str(row.get("fansly_message_id") or row.get("id") or "")


def chunk_messages(
    rows: list[dict[str, Any]],
    *,
    size: int = HISTORY_CHUNK_MESSAGES,
) -> list[list[dict[str, Any]]]:
    """Split chronological history into bounded chunks, preserving order."""
    size = max(1, int(size))
    return [rows[start : start + size] for start in range(0, len(rows), size)]


def render_chunk(rows: list[dict[str, Any]]) -> str:
    """Render one chunk for the extractor: id, speaker, truncated text."""
    lines: list[str] = []
    for row in rows:
        key = _message_key(row)
        if not key:
            continue
        speaker = "FAN" if str(row.get("role") or "") == "fan" else "CREATOR"
        content = " ".join(str(row.get("content") or "").split())
        if not content:
            attachments = (row.get("media_context") or {}).get("attachments") or []
            if not attachments:
                continue
            content = f"[sent {len(attachments)} attachment(s), no text]"
        lines.append(f"[{key}] {speaker}: {content[:HISTORY_MESSAGE_CHARS]}")
    return "\n".join(lines)


def parse_historical_payload(raw_text: str) -> HistoricalExtractionEnvelope:
    """Read one compaction response, repairing the shapes models emit.

    Same three repairs the live extractor performs — strip code fences, take
    the first balanced object out of surrounding prose, accept a bare list —
    reusing the same balanced-brace scanner rather than a second copy of it.
    """
    cleaned = "\n".join(
        line
        for line in (raw_text or "").splitlines()
        if not line.lstrip().startswith("```")
    ).strip()
    if not cleaned:
        raise ValueError("empty historical extraction response")
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        candidate = _first_json_object(cleaned)
        if candidate is None:
            raise
        payload = json.loads(candidate)
    if isinstance(payload, list):
        payload = {"observations": payload}
    return HistoricalExtractionEnvelope.model_validate(payload)


def validate_historical_observation(
    proposed: ProposedHistoricalObservation,
    *,
    chunk_by_id: dict[str, dict[str, Any]],
) -> ValidatedObservation | None:
    """Accept a historical proposal only if its evidence really exists.

    Three historical-specific rejections, then every live rule unchanged:

      * the named source message is not in this chunk -> reject. The model
        invented or mis-copied an id, and an unverifiable claim is not evidence.
      * the named message is the CREATOR's -> reject. This is the failure mode
        that turns "I'm a California girl" into a fact about the fan.
      * the quote does not occur in that message -> reject, via the same exact
        substring check live extraction uses.
    """
    source = chunk_by_id.get(proposed.source_message_id)
    if source is None:
        return None
    if str(source.get("role") or "") != "fan":
        return None

    validated = validate_observation(
        proposed,
        fan_message=str(source.get("content") or ""),
    )
    if validated is None:
        return None
    return validated.model_copy(
        update={"source_type": SOURCE_TYPE_HISTORICAL_MESSAGE}
    )


def merge_continuity(
    existing: dict[str, Any] | None,
    envelope: HistoricalExtractionEnvelope,
    *,
    chunk_last_sent_at: str | None = None,
) -> dict[str, Any]:
    """Fold one chunk's continuity into the durable compact state.

    Bounded and additive. Topics accumulate newest-first and are capped, the
    relationship summary is replaced by the most recent chunk's (history is
    read chronologically, so the last chunk processed is the most recent
    picture), and nothing here is ever allowed to contradict fan_facts — this
    is prose context for a writer, not structured knowledge.
    """
    current = dict(existing or {})

    def _fold(key: str, values: list[str], cap: int) -> None:
        seen: list[str] = []
        for value in list(values) + list(current.get(key) or []):
            text = " ".join(str(value or "").split())
            if not text or len(text) > 160:
                continue
            if any(text.casefold() == kept.casefold() for kept in seen):
                continue
            seen.append(text)
            if len(seen) >= cap:
                break
        if seen:
            current[key] = seen
        elif key in current:
            current.pop(key)

    _fold("ongoing_topics", envelope.ongoing_topics, MAX_ONGOING_TOPICS)
    _fold("commercial_context", envelope.commercial_context, MAX_COMMERCIAL_CONTEXT)

    summary = " ".join(str(envelope.relationship_summary or "").split())
    if summary:
        current["relationship_summary"] = summary[:MAX_RELATIONSHIP_SUMMARY_CHARS]
    if chunk_last_sent_at:
        current["through"] = chunk_last_sent_at
    return current


async def _extract_chunk(
    *,
    creator_id: str,
    fan_id: str,
    rows: list[dict[str, Any]],
    profile_id: str | None,
) -> tuple[HistoricalExtractionEnvelope | None, int, int]:
    """Run one chunk. Returns (envelope, proposed, accepted)."""
    rendered = render_chunk(rows)
    if not rendered:
        return HistoricalExtractionEnvelope(), 0, 0

    profile = get_profile(profile_id)
    spec = profile.stages.get(STAGE_HISTORY_EXTRACTION)
    target = spec.primary_target() if spec else get_runtime_target("HISTORY_EXTRACTOR")
    telemetry = ModelTelemetryContext(
        feature="history_extraction",
        creator_id=creator_id,
        fan_id=fan_id,
        metadata={
            "ai_stack_profile": profile.profile_id,
            "chunk_messages": len(rows),
        },
    )

    result = await complete(
        target,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "ARCHIVED CONVERSATION EXCERPT (chronological):\n" + rendered
                ),
            }
        ],
        max_tokens=spec.resolved_max_tokens() if spec else 1400,
        temperature=spec.temperature if spec else 0.0,
    )
    try:
        envelope = parse_historical_payload(result.text)
    except Exception as exc:
        # One malformed chunk is one chunk. There is no repair round here, on
        # purpose: unlike a live turn, the next run will simply re-read this
        # chunk, and a second call per bad chunk on a 500-page archive is real
        # money for no additional certainty.
        await record_model_result(
            result,
            telemetry,
            success=False,
            parse_valid=False,
            error=f"invalid historical extraction JSON: {exc}",
        )
        print(f"[HISTORY EXTRACTION] invalid chunk fan={fan_id}: {exc}")
        return None, 0, 0

    await record_model_result(result, telemetry, success=True, parse_valid=True)

    chunk_by_id = {
        _message_key(row): row for row in rows if _message_key(row)
    }
    accepted = 0
    for proposed in envelope.observations:
        validated = validate_historical_observation(
            proposed, chunk_by_id=chunk_by_id
        )
        if validated is None:
            continue
        source = chunk_by_id.get(proposed.source_message_id) or {}
        try:
            changed = await _merge_one(
                creator_id=creator_id,
                fan_id=fan_id,
                source_message_id=_message_key(source) or None,
                observation=validated,
                extraction_provider=target.provider,
                extraction_model=target.model,
                historical=True,
            )
        except Exception as exc:
            print(f"[HISTORY EXTRACTION] merge failed fan={fan_id}: {exc}")
            continue
        if changed:
            accepted += 1
    return envelope, len(envelope.observations), accepted


def _rows_after_cursor(
    rows: list[dict[str, Any]],
    *,
    cursor_sent_at: str | None,
    cursor_message_id: str | None,
) -> list[dict[str, Any]]:
    """Everything strictly after the compaction cursor, chronologically.

    Cursor is (sent_at, message id) rather than sent_at alone because platform
    timestamps collide: several messages in one second is normal in chat, and a
    sent_at-only cursor would either re-read or skip the whole group.
    """
    if not cursor_sent_at:
        return list(rows)
    resumed: list[dict[str, Any]] = []
    passed_cursor = False
    for row in rows:
        sent_at = str(row.get("sent_at") or "")
        if passed_cursor:
            resumed.append(row)
            continue
        if sent_at > cursor_sent_at:
            passed_cursor = True
            resumed.append(row)
            continue
        if sent_at == cursor_sent_at and cursor_message_id:
            if _message_key(row) == cursor_message_id:
                passed_cursor = True
    return resumed


async def compact_fan_history(
    *,
    creator_id: str,
    fan_id: str,
    max_chunks: int | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    """Compact the next bounded run of this fan's history into durable memory.

    Resumable: progress is a durable (sent_at, message id) cursor, so a fan
    whose archive stops being processed after chunk twelve continues at chunk
    thirteen rather than paying to re-read twelve chunks of model input.
    """
    if not history_extraction_enabled():
        return {"status": "disabled", "chunks": 0, "accepted": 0}

    state = await get_backfill_state(fan_id)
    if state is None:
        return {"status": "unavailable", "chunks": 0, "accepted": 0}

    rows = await all_local_history(fan_id)
    pending = _rows_after_cursor(
        rows,
        cursor_sent_at=state.get("extraction_cursor_sent_at"),
        cursor_message_id=state.get("extraction_cursor_message_id"),
    )
    if not pending:
        return {
            "status": "up_to_date",
            "chunks": 0,
            "accepted": 0,
            "messages_pending": 0,
        }

    limit = max(1, int(max_chunks or HISTORY_CHUNKS_PER_RUN))
    chunks = chunk_messages(pending)[:limit]
    continuity = state.get("continuity") if isinstance(state.get("continuity"), dict) else {}
    processed = 0
    proposed_total = 0
    accepted_total = 0
    messages_done = 0
    cursor_row: dict[str, Any] | None = None

    for chunk in chunks:
        envelope, proposed, accepted = await _extract_chunk(
            creator_id=creator_id,
            fan_id=fan_id,
            rows=chunk,
            profile_id=profile_id,
        )
        if envelope is None:
            # Stop at the first failed chunk rather than skipping it: the
            # cursor has not moved past it, so the next run retries exactly
            # here instead of leaving a silent hole in the fan's history.
            break
        continuity = merge_continuity(
            continuity,
            envelope,
            chunk_last_sent_at=str(chunk[-1].get("sent_at") or "") or None,
        )
        processed += 1
        proposed_total += proposed
        accepted_total += accepted
        messages_done += len(chunk)
        cursor_row = chunk[-1]

    if cursor_row is not None:
        await update_backfill_state(
            fan_id,
            {
                "extraction_cursor_sent_at": cursor_row.get("sent_at"),
                "extraction_cursor_message_id": _message_key(cursor_row),
                "chunks_extracted": int(state.get("chunks_extracted") or 0) + processed,
                "messages_extracted": int(state.get("messages_extracted") or 0)
                + messages_done,
                "facts_proposed": int(state.get("facts_proposed") or 0) + proposed_total,
                "facts_accepted": int(state.get("facts_accepted") or 0) + accepted_total,
                "continuity": continuity or None,
            },
        )

    remaining = max(0, len(pending) - messages_done)
    print(
        f"[HISTORY EXTRACTION] fan={fan_id} chunks={processed} "
        f"messages={messages_done} proposed={proposed_total} "
        f"accepted={accepted_total} remaining={remaining}"
    )
    return {
        "status": "ok" if processed else "no_progress",
        "chunks": processed,
        "messages_extracted": messages_done,
        "proposed": proposed_total,
        "accepted": accepted_total,
        "messages_pending": remaining,
        "continuity": continuity,
    }
