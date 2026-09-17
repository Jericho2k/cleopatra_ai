"""Why this exact reply exists — recorded on the message itself.

``docs/autonomy_architecture_review.md`` §6 step 1 is the first thing the review
asks for and the precondition for everything after it:

    Link each visible reply to its triggering event, input context, state
    version, decision, actual successful model attempt, transformations, and
    delivery receipt. Confirm deployed SHA and active flags.

Before this module, each of those facts lived somewhere that could not be joined
to a message after the fact. The triggering fan message was a local variable.
The context window sizes were constants in two different prompt builders. The
decision was a log line. The model was recorded as the one *requested*
(finding H). The transformations — inventory repair, delivery-language repair,
tag stripping, bubble merging — left no trace at all, so a reply that reached
the fan in different words than the model produced looked like the model's work.
The delivery receipt existed, on a different row.

§2 is the reason it matters. The supplied failure excerpts cannot identify the
generating model, the deployed commit, the enabled flags or the true payment
state, so no conclusion drawn from them is attributable. Attribution is not a
nicety here; it is what separates a finding from a guess, and the review refuses
to accept a tone instruction or a model swap until it exists.

Design constraints this follows:

*It writes nothing new to the database schema.* The record lands in
``messages.media_context``, the existing jsonb column that already carries the
``ai_stack`` marker, next to it and under its own key. No migration, nothing
customer-visible, and it is readable from the row alone months later — the same
reasoning as ``services.suggestions.message_ai_stack_metadata``, which this
complements rather than replaces.

*It is a record, never a control surface.* Nothing reads a provenance record to
decide anything. A recorder that failed must not be able to stop a reply, so
every method here is total: it takes whatever it is given, keeps what it
understands, and never raises.

*It is bounded.* This is written on every creator message the pipeline sends, so
it stores fingerprints and counts rather than copies. The triggering message is
identified by a hash of its text plus its timestamp, not by the text; the full
flag snapshot is a digest, with the mapping on the build endpoint. A provenance
record must never become a second copy of the conversation.

One turn can send several bubbles. They share a ``turn_id`` and differ only in
``part``, so the parts of one reply can be reassembled and a reply can be told
apart from two replies sent close together.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.build_info import build_snapshot

#: The key this record occupies inside ``messages.media_context``.
PROVENANCE_KEY = "reply_provenance"

#: Which pipeline produced the reply. Recorded because Assisted and Full Auto
#: assemble context differently (finding A was exactly that divergence), and a
#: quality comparison that mixes them is comparing two systems.
#:
#: Named PIPELINE_* rather than MODE_*: ai/writer_style.py already owns MODE_AUTO
#: and MODE_ASSISTED for the writer's reply contract, which is a different
#: distinction, and one module importing both should not have to alias either.
PIPELINE_AUTO = "auto"
PIPELINE_ASSISTED = "assisted"
PIPELINE_PROACTIVE = "proactive"
PIPELINE_SCHEDULED = "scheduled"

#: Transformations applied to the writer's text after generation. Named here so
#: the strings cannot drift between the code that applies them and the record
#: that reports them.
TRANSFORM_INVENTORY_REPAIR = "inventory_repair"
TRANSFORM_DELIVERY_LANGUAGE = "delivery_language_repair"
TRANSFORM_PPV_TAG_STRIPPED = "ppv_tag_stripped"
TRANSFORM_SHAPE_APPLIED = "message_shape_applied"
TRANSFORM_PPV_MERGED = "ppv_single_message_merge"
#: Assisted only: the operator changed the candidate before sending it.
#: The most important transformation of all to record — a human-edited
#: reply attributed to the model is the single easiest way to make a model
#: comparison say the wrong thing.
TRANSFORM_OPERATOR_EDIT = "operator_edit"

#: Delivery kinds, matching the two send paths in the Full Auto pipeline.
DELIVERY_TEXT = "text"
DELIVERY_PPV = "ppv"
DELIVERY_LOCAL_TEST = "local_test"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(text: object) -> str:
    """A short stable identifier for a message's text, without storing it.

    Twelve hex characters of SHA-256. Enough to say "the reply to *this* fan
    message" and to match a transcript line against a provenance record, without
    a second copy of customer text living in a metadata column.
    """
    raw = "" if text is None else str(text)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


@dataclass
class ReplyProvenance:
    """Accumulates one turn's ground truth, then emits it per sent bubble.

    Created at the top of a turn and filled in as the turn makes its decisions,
    in the order the pipeline makes them. ``as_metadata`` is called once per
    delivered part, after the send, when the platform receipt is known.
    """

    creator_id: str
    fan_id: str
    mode: str
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(default_factory=_now)

    # The event that caused this turn to run.
    trigger: dict[str, Any] = field(default_factory=dict)
    # What evidence the turn was allowed to see.
    context: dict[str, Any] = field(default_factory=dict)
    # What the turn decided to do, and which controller decided it.
    decision: dict[str, Any] = field(default_factory=dict)
    # Which model attempt actually answered (ai/generation_trace.py).
    writer: dict[str, Any] = field(default_factory=dict)
    # What was done to the writer's text before anyone saw it.
    transforms: list[str] = field(default_factory=list)

    def as_state(self) -> dict[str, Any]:
        """The recorder's fields, for storing between two HTTP requests.

        Every field is already a fingerprint, a count or an identifier —
        ``ReplyProvenance`` never holds message text — so persisting this adds
        no content anywhere that content was not already going.
        """
        return {
            "creator_id": self.creator_id,
            "fan_id": self.fan_id,
            "mode": self.mode,
            "turn_id": self.turn_id,
            "started_at": self.started_at,
            "trigger": dict(self.trigger),
            "context": dict(self.context),
            "decision": dict(self.decision),
            "writer": dict(self.writer),
            "transforms": list(self.transforms),
        }

    @classmethod
    def from_state(cls, state: Any) -> "ReplyProvenance | None":
        """Rebuild a recorder stored by ``as_state``, or None.

        Total: a row written by an older build, or a partial one, returns None
        rather than raising. A reply must never fail to send because its
        evidence trail could not be rebuilt.
        """
        if not isinstance(state, dict):
            return None
        creator_id = str(state.get("creator_id") or "")
        fan_id = str(state.get("fan_id") or "")
        if not creator_id or not fan_id:
            return None
        restored = cls(
            creator_id=creator_id,
            fan_id=fan_id,
            mode=str(state.get("mode") or PIPELINE_ASSISTED),
        )
        if state.get("turn_id"):
            restored.turn_id = str(state["turn_id"])
        if state.get("started_at"):
            restored.started_at = str(state["started_at"])
        for key in ("trigger", "context", "decision", "writer"):
            value = state.get(key)
            if isinstance(value, dict):
                setattr(restored, key, dict(value))
        transforms = state.get("transforms")
        if isinstance(transforms, list):
            restored.transforms = [str(item) for item in transforms if str(item)]
        return restored

    def record_trigger(
        self,
        *,
        kind: str,
        text: object = None,
        sent_at: object = None,
        history_position: int | None = None,
    ) -> None:
        """Identify the event this reply answers.

        ``messages`` rows are read back without their ids (``db/queries.py``
        selects role, content, sent_at and media_context), so the triggering
        message is identified by a fingerprint of its text plus its timestamp
        and its position in the loaded history. That is a real identifier — it
        matches one row — and it needs no schema change to be true.
        """
        self.trigger = {
            "kind": _clean(kind),
            "text_fingerprint": fingerprint(text),
            "text_chars": len(_clean(text)),
        }
        if sent_at is not None:
            self.trigger["sent_at"] = _clean(getattr(sent_at, "isoformat", lambda: sent_at)())
        if history_position is not None:
            self.trigger["history_position"] = int(history_position)

    def record_context(
        self,
        *,
        history_messages: int,
        analyzer_window: int | None = None,
        writer_window: int | None = None,
        stack_profile: str = "",
        writer_prompt_version: str = "",
        live_state: dict[str, Any] | None = None,
        packet: dict[str, Any] | None = None,
    ) -> None:
        """Record what evidence this turn had, and how much of it was rendered.

        Finding D: understanding and writing see different, shorter windows than
        the history that was loaded, and neither renders a full event history.
        Storing all three numbers is what turns that from a claim about the code
        into a measurable property of each reply — including after the budgeted
        context builder replaces the fixed slices, when the question becomes
        whether the budget was enough rather than what the constant was.

        ``live_state`` is a mapping of block name to whether that block was
        present, never the block's contents: which controllers spoke is
        provenance, what they said is the prompt.
        """
        record: dict[str, Any] = {"history_messages": int(history_messages)}
        if analyzer_window is not None:
            record["analyzer_window"] = int(analyzer_window)
        if writer_window is not None:
            record["writer_window"] = int(writer_window)
        if stack_profile:
            record["stack_profile"] = _clean(stack_profile)
        if writer_prompt_version:
            record["writer_prompt_version"] = _clean(writer_prompt_version)
        if live_state:
            present = sorted(name for name, on in live_state.items() if on)
            record["live_state_blocks"] = present
        if packet:
            # What the budgeted builder actually assembled, including what it
            # had to drop (services/context_packet.py). This is what makes
            # "the model never mentioned the thing he asked about" separable
            # from "the model was never told about it" after the fact.
            record["packet"] = dict(packet)
        self.context = record

    def record_decision(
        self,
        *,
        source: str,
        action: object = None,
        reason: object = None,
        purchase_signal: object = None,
        crisis_signal: object = None,
        resend_requested: object = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record what the turn decided and which authority decided it.

        Finding F: several systems can prescribe what one conversation does
        next. ``source`` names the one whose verdict this reply carries, which
        is the measurement that has to exist before any of them can be removed
        under replay comparison.
        """
        record: dict[str, Any] = {"source": _clean(source)}
        for key, value in (
            ("action", action),
            ("reason", reason),
            ("purchase_signal", purchase_signal),
            ("crisis_signal", crisis_signal),
            ("resend_requested", resend_requested),
        ):
            cleaned = _clean(getattr(value, "value", value))
            if cleaned:
                record[key] = cleaned
        if extra:
            for key, value in extra.items():
                cleaned = _clean(getattr(value, "value", value))
                if cleaned:
                    record[_clean(key)] = cleaned
        self.decision = record

    def record_writer(self, trace: Any) -> None:
        """Record which model attempt actually produced the text (finding H).

        Takes the ``GenerationTrace`` itself rather than a dict so the caller
        cannot accidentally record the requested model here — the whole point of
        the field is that it is the served one.
        """
        if trace is None:
            return
        as_metadata = getattr(trace, "as_metadata", None)
        if callable(as_metadata):
            self.writer = as_metadata()

    def record_transform(self, name: str, applied: bool = True) -> None:
        """Note one repair or reshaping applied to the writer's text.

        Order is preserved and duplicates are ignored, so the record reads as
        the sequence of things that happened to the copy between generation and
        delivery. A reply whose words were changed after generation must not be
        attributed to the model as though they were the model's.
        """
        if not applied:
            return
        cleaned = _clean(name)
        if cleaned and cleaned not in self.transforms:
            self.transforms.append(cleaned)

    def as_metadata(
        self,
        *,
        part: int = 0,
        parts: int = 1,
        delivery_kind: str = DELIVERY_TEXT,
        platform_message_id: object = None,
        delivery_reference: object = None,
        price_cents: int | None = None,
    ) -> dict[str, Any]:
        """The record for one delivered bubble, ready to merge into metadata.

        Called after the send, because the delivery receipt is half the point:
        a reply with no ``platform_message_id`` was never accepted by the
        platform, and the review is explicit that a delivery claim must be tied
        to the operation result rather than to what the copy says happened.
        """
        record: dict[str, Any] = {
            "turn_id": self.turn_id,
            "mode": _clean(self.mode),
            "started_at": self.started_at,
            "recorded_at": _now(),
            "creator_id": _clean(self.creator_id),
            "fan_id": _clean(self.fan_id),
            "part": int(part),
            "parts": int(parts),
            "build": build_snapshot(include_flags=False),
        }
        if self.trigger:
            record["trigger"] = dict(self.trigger)
        if self.context:
            record["context"] = dict(self.context)
        if self.decision:
            record["decision"] = dict(self.decision)
        if self.writer:
            record["writer"] = dict(self.writer)
        if self.transforms:
            record["transforms"] = list(self.transforms)

        delivery: dict[str, Any] = {"kind": _clean(delivery_kind)}
        receipt = _clean(platform_message_id)
        # Absence is recorded explicitly. "No receipt" and "receipt not looked
        # for" must not read the same on a row that is meant to prove delivery.
        delivery["platform_message_id"] = receipt or None
        delivery["accepted_by_platform"] = bool(receipt)
        reference = _clean(delivery_reference)
        if reference:
            delivery["reference"] = reference
        if price_cents is not None:
            delivery["price_cents"] = int(price_cents)
        record["delivery"] = delivery
        return {PROVENANCE_KEY: record}

    def describe(self) -> str:
        """One log line summarising the turn, for Railway logs."""
        writer = self.writer or {}
        actual = writer.get("actual") or {}
        served = actual.get("model") or "none"
        requested = (writer.get("requested") or {}).get("model") or "none"
        return (
            f"[PROVENANCE] turn={self.turn_id} mode={self.mode} "
            f"fan={self.fan_id} "
            f"trigger={self.trigger.get('text_fingerprint', 'none')} "
            f"decision={self.decision.get('action') or self.decision.get('source') or 'none'} "
            f"requested={requested} served={served} "
            f"transforms={','.join(self.transforms) if self.transforms else 'none'}"
        )


def provenance_of(media_context: dict | None) -> dict[str, Any]:
    """Read a provenance record back off a message row, or ``{}``.

    The tolerant direction of the same contract: rows written before this
    existed, and rows whose metadata is some other shape, simply have none.
    """
    if not isinstance(media_context, dict):
        return {}
    record = media_context.get(PROVENANCE_KEY)
    return record if isinstance(record, dict) else {}


def merge_provenance(media_context: dict | None, record: dict[str, Any]) -> dict:
    """Merge a provenance record into a message's existing metadata.

    Mirrors ``services.suggestions._with_ai_stack``: the caller's metadata wins
    on every other key, and this only ever adds its own.
    """
    merged = dict(media_context or {})
    merged.update(record)
    return merged


# ---------------------------------------------------------------------------
# Carrying an Assisted reply's provenance across the operator's decision
# ---------------------------------------------------------------------------
#
# Full Auto generates and delivers inside one function, so its recorder is a
# local variable. Assisted does not: ``get_suggestions`` produces candidates,
# a human reads them, and some time later ``POST /reply`` sends one. Two HTTP
# requests, with a person in between.
#
# Without something spanning that gap, an operator-sent reply is a message with
# no attributable origin — which is the same blind spot the review names in §2,
# only for the mode an agency uses most. The suggestion token closes it: the
# generation request keeps its record here and hands back an opaque key, and the
# send request redeems that key and finishes the record with the receipt and the
# candidate the operator actually chose.
#
# This is deliberately in-process and lossy. A restart, an eviction or a second
# backend replica means the token misses and the reply is persisted with no
# provenance, exactly as it was before this existed. Provenance is a record, so
# losing one must never cost a message; anything stronger would mean a database
# write on the path of every suggestion, for evidence that is only wanted while
# an operator is looking at the screen.

#: How long a suggestion's record is worth keeping. An operator picks a reply in
#: seconds to minutes; beyond that the candidates on their screen are stale
#: anyway, and the record describes a turn that no longer matches the
#: conversation.
SUGGESTION_TOKEN_TTL_SECONDS = 30 * 60

#: How many turns' records may be held at once. Bounded FIFO for the reason
#: core/bounded_state.py gives: a structure that clears itself wholesale loses
#: every operator's in-flight turn at the moment one of them overflows it.
SUGGESTION_TOKEN_MAXSIZE = 512


class SuggestionProvenanceStore:
    """Short-lived, bounded storage for records awaiting an operator's send.

    Eviction is by age first and by insertion order second, so a busy shift
    forgets the turns nobody acted on rather than the ones still on screen.
    """

    __slots__ = ("_entries", "_maxsize", "_ttl")

    def __init__(
        self,
        *,
        maxsize: int = SUGGESTION_TOKEN_MAXSIZE,
        ttl_seconds: float = SUGGESTION_TOKEN_TTL_SECONDS,
    ) -> None:
        self._entries: "OrderedDict[str, tuple[float, ReplyProvenance]]" = OrderedDict()
        self._maxsize = max(1, int(maxsize))
        self._ttl = float(ttl_seconds)

    def _prune(self, now: float) -> None:
        expired = [
            token
            for token, (stored_at, _) in self._entries.items()
            if now - stored_at > self._ttl
        ]
        for token in expired:
            self._entries.pop(token, None)
        while len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)

    def put(self, provenance: "ReplyProvenance", *, now: float | None = None) -> str:
        """Store one turn's record and return the token that redeems it."""
        moment = time.monotonic() if now is None else float(now)
        token = uuid.uuid4().hex
        self._entries[token] = (moment, provenance)
        self._prune(moment)
        return token

    def take(
        self,
        token: object,
        *,
        creator_id: str = "",
        fan_id: str = "",
        now: float | None = None,
    ) -> "ReplyProvenance | None":
        """Redeem a token exactly once, or return ``None``.

        The record is removed on redemption: one generated turn becomes at most
        one sent message, and a token replayed against a second send would
        attribute that message to a turn it did not come from.

        ``creator_id``/``fan_id`` are checked when supplied. A token is an
        opaque handle rather than an authorization, but a record belonging to a
        different conversation is simply the wrong record, and silently
        attaching it would be worse than attaching nothing.
        """
        key = "" if token is None else str(token).strip()
        if not key:
            return None
        moment = time.monotonic() if now is None else float(now)
        self._prune(moment)
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, provenance = entry
        if moment - stored_at > self._ttl:
            self._entries.pop(key, None)
            return None
        if creator_id and provenance.creator_id != str(creator_id):
            return None
        if fan_id and provenance.fan_id != str(fan_id):
            return None
        self._entries.pop(key, None)
        return provenance

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


#: The process-wide store. One per backend replica, which is why a miss is an
#: ordinary outcome rather than an error.
SUGGESTION_PROVENANCE = SuggestionProvenanceStore()
