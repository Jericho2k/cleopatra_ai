"""A reviewer-facing comparison of two conversations that does not say whose.

``eval/README.md`` already states what blind means here and why the existing
model-lab review works that way: a reviewer who knows which system is the new
one is not reading the conversation, they are checking a prediction. This module
is the same idea applied to whole conversations instead of single replies.

WHAT THE REVIEWER SEES
----------------------
Two transcripts per scenario, labelled Conversation A and Conversation B, in an
order drawn from the run seed. The fan's messages are identical between them by
construction, so they are printed once, down the middle, with each side's reply
underneath.

WHAT THE REVIEWER MUST NOT SEE
------------------------------
Any word that identifies a runtime. That means the core ids, and it also means
"baseline" and "candidate" — those two are worse than the ids, because they say
which one is expected to win. :func:`leaked_terms` checks the finished document
for every one of them and :func:`build_review` refuses to return a document that
carries any, so a leak is a raised exception rather than a review that quietly
told the reviewer the answer.

Nothing model-related is shown either: no model names, no latency, no cost, no
fan ids. Those belong in ``metrics.json``, and in the review they would be a
fingerprint linking one side's scenarios together.

THE RUBRIC
----------
Separate dimensions, each rated on its own, because the failures the work is
aimed at are not the same failure. A conversation can be specific and
contribute nothing; it can contribute and lose the scene; it can hold the scene
and interview the fan through it. One number would let any of those cancel any
other.

Initiative is deliberately a "fit" judgement rather than a quantity. Fan-led is
not worse than creator-led, and a rubric that rewarded creator initiative would
be asking for a creator who talks over the fan.

The scales are anchored at 1, 3 and 5 so two reviewers mean roughly the same
thing by a 4, and every dimension has a "not applicable" so a non-commercial
scenario is not rated for commercial naturalness.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

#: Reviewer-facing labels. Deliberately neutral and deliberately not "A is the
#: first one we built".
LABELS: tuple[str, ...] = ("A", "B")

#: Words that would tell the reviewer which runtime wrote which transcript, or
#: which one is expected to win. Checked against the finished document.
ALWAYS_FORBIDDEN: tuple[str, ...] = ("baseline", "candidate", "control", "experimental")

_REDACTION = "[redacted]"


@dataclass(frozen=True)
class Dimension:
    """One thing a reviewer judges, with anchors so a 4 means one thing."""

    key: str
    title: str
    question: str
    anchor_low: str
    anchor_mid: str
    anchor_high: str
    #: True when a scenario can legitimately not exercise this at all.
    optional: bool = False


#: The rubric. Ordered the way a reviewer reads a conversation: what the reply
#: did with the message, then what it did to the conversation, then what it did
#: over time.
DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        key="specificity",
        title="Specific reaction",
        question=(
            "Does each reply respond to the actual thing this fan said, or "
            "could it have followed almost any message?"
        ),
        anchor_low="Generic validation; the replies would fit another conversation unchanged.",
        anchor_mid="Picks up the topic but rarely the particular detail.",
        anchor_high="Reacts to the specific detail, repeatedly and without strain.",
    ),
    Dimension(
        key="contribution",
        title="Contribution vs mirroring",
        question=(
            "Does the creator add something — a thought, a reaction, a feeling, "
            "a callback, an implication, a development of the scene — or does it "
            "paraphrase, validate and ask?"
        ),
        anchor_low="Paraphrase → validate → question, turn after turn.",
        anchor_mid="Adds something occasionally; often mirrors.",
        anchor_high="Brings its own material most turns, and it fits what was said.",
    ),
    Dimension(
        key="initiative_fit",
        title="Initiative fit",
        question=(
            "Is the balance of who moves the conversation right for these "
            "moments? Fan-led, creator-led and shared are all legitimate; the "
            "question is whether the fan is left carrying all the momentum, or "
            "is talked over."
        ),
        anchor_low="Wrong for the moment throughout — the fan carries everything, or is steamrolled.",
        anchor_mid="Mostly fits; a few moments where the balance is off.",
        anchor_high="The balance matches the moment, including when it changes.",
    ),
    Dimension(
        key="question_discipline",
        title="Question discipline",
        question=(
            "Are questions used where a question is natural? Watch for forced "
            "generic questions, repeated question endings, interview behaviour, "
            "and asking where a reaction or contribution would have been more "
            "natural."
        ),
        anchor_low="Interview: nearly every turn ends on a question, several of them generic.",
        anchor_mid="Some forced or repetitive questions among reasonable ones.",
        anchor_high="Questions arrive when they belong and are absent when they do not.",
    ),
    Dimension(
        key="shared_scene_continuity",
        title="Shared-scene continuity",
        question=(
            "When an imagined or shared scene exists, are its established "
            "elements, current focus, unresolved threads and direction "
            "preserved?"
        ),
        anchor_low="The scene is dropped, contradicted or restarted.",
        anchor_mid="Broadly held, with elements lost or quietly changed.",
        anchor_high="The scene persists and develops; earlier elements stay true.",
        optional=True,
    ),
    Dimension(
        key="reference_handling",
        title="Reference handling",
        question=(
            "Are indirect references and short or ambiguous replies read using "
            "the context they arrive in, rather than triggering a restart or a "
            "generic question?"
        ),
        anchor_low="Indirect references are missed; short replies reset the conversation.",
        anchor_mid="Usually resolved; sometimes answered generically.",
        anchor_high="Indirect references land, and a short reply is read in context.",
    ),
    Dimension(
        key="direction_change_adaptation",
        title="Direction-change adaptation",
        question=(
            "When the fan changes direction, does the conversation follow "
            "cheaply — without dragging the old trajectory along, repeatedly "
            "steering back, or resetting everything?"
        ),
        anchor_low="Keeps pushing the old direction, or throws away everything established.",
        anchor_mid="Adapts, but stiffly or after a delay.",
        anchor_high="Turns with the fan and keeps what still applies.",
        optional=True,
    ),
    Dimension(
        key="pacing_naturalness",
        title="Pacing",
        question=(
            "Can the conversation build, hold, continue, cool, redirect, pause "
            "and resume? Constant escalation is not good pacing."
        ),
        anchor_low="One gear — escalating, or flat — regardless of what is happening.",
        anchor_mid="Some variation, some moments pushed or dropped.",
        anchor_high="Moves through registers as the conversation calls for it.",
    ),
    Dimension(
        key="repetition",
        title="Repetition and habit",
        question=(
            "Do phrasings, openings, structures or moves come back often enough "
            "to notice as a reader?"
        ),
        anchor_low="Recognisable formula; the same opening or move again and again.",
        anchor_mid="Occasional tics.",
        anchor_high="Reads as varied writing throughout.",
    ),
    Dimension(
        key="truthful_presence",
        title="Truthful presence",
        question=(
            "Does the creator avoid unsupported claims about its CURRENT "
            "location, activity, clothing, surroundings, schedule or what it "
            "can physically see? A vivid SHARED IMAGINED scene is not an error "
            "here, however vivid — only a present-tense claim about the real "
            "world that nothing supports."
        ),
        anchor_low="States real-world present facts about itself that nothing supports.",
        anchor_mid="Mostly careful; one or two present-world claims that overreach.",
        anchor_high="Imagines freely and never asserts an unsupported present fact.",
    ),
    Dimension(
        key="commercial_naturalness",
        title="Commercial naturalness",
        question=(
            "If anything commercial happens: does it arise from the "
            "conversation, stay in a conversational voice, avoid catalogue copy "
            "and internal package or media detail, and avoid an abrupt pivot? "
            "If the fan declines, is the decline accepted without a pressure "
            "loop or an immediate re-offer?"
        ),
        anchor_low="Catalogue voice, abrupt pivot, internal detail, or pressure after a no.",
        anchor_mid="Reasonable, with moments that read as copy or as a pivot.",
        anchor_high="Indistinguishable from a person mentioning something they have.",
        optional=True,
    ),
    Dimension(
        key="post_event_continuation",
        title="Post-event continuation",
        question=(
            "After a significant event — a refusal, a purchase, a delivery — "
            "does the conversation stay in the moment rather than resetting or "
            "immediately selling again?"
        ),
        anchor_low="Resets or re-sells immediately.",
        anchor_mid="Stays for a beat, then moves on too fast.",
        anchor_high="Stays with what just happened as a person would.",
        optional=True,
    ),
    Dimension(
        key="overall_coherence",
        title="Overall conversational coherence",
        question=(
            "Read end to end, does this hold together as one conversation with "
            "one person?"
        ),
        anchor_low="Disjointed; turns do not belong to the same conversation.",
        anchor_mid="Holds together with visible seams.",
        anchor_high="Reads as one continuous conversation.",
    ),
)


def forbidden_terms(extra: Sequence[str] = ()) -> tuple[str, ...]:
    """Every term the reviewer-facing document must not contain.

    The run's own core ids plus the role words. Passed in rather than imported
    so this stays a pure function of the run it is given — a future runtime id
    needs no edit here.
    """
    terms = {term.strip().lower() for term in ALWAYS_FORBIDDEN}
    for term in extra:
        cleaned = str(term or "").strip().lower()
        if cleaned:
            terms.add(cleaned)
    return tuple(sorted(terms))


def leaked_terms(document: str, terms: Sequence[str]) -> list[str]:
    """Which forbidden terms appear in this document, case-insensitively.

    Whole words only, so a scenario about a fan named Candice is not a leak and
    ``semantic_v2`` inside ``semantic_v2_notes`` still is.
    """
    lowered = str(document or "").lower()
    found = []
    for term in terms:
        if re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9])", lowered):
            found.append(term)
    return sorted(set(found))


def redact(text: str, terms: Sequence[str]) -> str:
    """Remove forbidden terms from text that came from the conversation.

    Applied to transcript content only. A fan or a creator saying a word that
    happens to be on the list must not be able to unblind the review, and must
    not be able to make the build fail either.
    """
    result = str(text or "")
    for term in terms:
        result = re.sub(
            rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9])",
            _REDACTION,
            result,
            flags=re.IGNORECASE,
        )
    return result


def order_for(scenario_id: str, *, seed: int, roles: Sequence[str]) -> list[str]:
    """Which role gets label A for this scenario, decided by the seed.

    Per scenario rather than per run, so a reviewer who works out one scenario
    learns nothing about the next; reproducible from the seed, so the same run
    id and seed rebuild the same document and the same mapping.
    """
    order = list(roles)
    random.Random(f"{seed}:{scenario_id}").shuffle(order)
    return order


def _transcript_rows(turns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": int(turn.get("turn_index", index)),
            "fan": str(turn.get("fan_input") or ""),
            "creator": [str(part) for part in (turn.get("creator_output") or [])],
        }
        for index, turn in enumerate(turns or [])
    ]


def _render_side(label: str, rows: Sequence[dict[str, Any]], terms: Sequence[str]) -> list[str]:
    lines = [f"#### Conversation {label}", ""]
    for row in rows:
        fan = redact(row["fan"], terms).strip()
        lines.append(f"**Turn {row['index']}** — fan: {fan or '*(no message; time passed)*'}")
        if row["creator"]:
            for part in row["creator"]:
                lines.append(f"> {redact(part, terms).strip()}")
        else:
            lines.append("> *(the creator sent nothing this turn)*")
        lines.append("")
    return lines


def _render_rubric(label: str) -> list[str]:
    lines = [f"**Ratings — Conversation {label}**", ""]
    for dimension in DIMENSIONS:
        suffix = " · `n/a` if this scenario does not exercise it" if dimension.optional else ""
        lines.append(f"- **{dimension.title}** (1–5){suffix}: ")
    lines.extend(["", f"**Notes on Conversation {label}:** ", ""])
    return lines


def _render_rubric_guide() -> list[str]:
    lines = [
        "## How to rate",
        "",
        "Rate each dimension on its own. Do not average them into a verdict — "
        "the dimensions are separate because the problems they describe are "
        "separate, and one of them being good is not a reason to forgive "
        "another.",
        "",
        "1 and 5 are the anchors below; 3 is the middle described there. Use "
        "`n/a` where a scenario does not exercise a dimension at all.",
        "",
    ]
    for dimension in DIMENSIONS:
        lines.extend(
            [
                f"### {dimension.title}",
                "",
                dimension.question,
                "",
                f"- **1** — {dimension.anchor_low}",
                f"- **3** — {dimension.anchor_mid}",
                f"- **5** — {dimension.anchor_high}",
                "",
            ]
        )
    return lines


def build_review(
    paired: dict[str, Any],
    *,
    seed: int,
    run_id: str = "",
    extra_forbidden: Sequence[str] = (),
) -> tuple[str, dict[str, Any]]:
    """Build the reviewer document and the mapping that unblinds it.

    Returns ``(markdown, mapping)``. The mapping is a separate object precisely
    so it can be written to a separate file: a review whose key is in the same
    document is not blind, it is a document with the answer further down.

    Raises if the finished document still contains a forbidden term. That is a
    bug in this module rather than something to warn about, and a blind review
    that silently is not blind is worse than no review.
    """
    pairs = list(paired.get("pairs") or [])
    cores = {
        str(side.get("conversation_core"))
        for pair in pairs
        for side in (pair.get("arms") or {}).values()
        if side.get("conversation_core")
    }
    terms = forbidden_terms([*cores, *extra_forbidden])

    lines: list[str] = [
        "# Conversation review",
        "",
        "Two complete conversations per scenario, from two builds of the same "
        "product. The fan's messages are identical in both; only the creator's "
        "replies differ.",
        "",
        "Which build produced which conversation is not recorded in this "
        "document, and the order is different for every scenario. Nothing here "
        "indicates which one anybody expects to be better, because the review "
        "is the evidence for that question rather than a check on an answer.",
        "",
        "Read each conversation end to end before rating either.",
        "",
    ]
    lines.extend(_render_rubric_guide())
    lines.extend(["---", ""])

    mapping: dict[str, Any] = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "scenarios": {},
        "_about": (
            "The key to the blind review. Labels are assigned per scenario from "
            "the seed, so this file is the only way to know which build wrote "
            "which conversation."
        ),
    }

    for pair in pairs:
        scenario_id = str(pair.get("scenario_id") or "")
        arms = pair.get("arms") or {}
        roles = sorted(arms.keys())
        if len(roles) < 2:
            # One arm only — a baseline-only run. There is nothing to compare
            # and nothing to hide, so the scenario is listed and skipped rather
            # than rendered as half a comparison.
            mapping["scenarios"][scenario_id] = {"skipped": "only one arm ran"}
            continue
        ordered = order_for(scenario_id, seed=seed, roles=roles)
        lines.extend([f"## Scenario `{scenario_id}`", ""])
        if pair.get("fan_inputs_identical") is False:
            lines.extend(
                [
                    "> The two conversations did not receive identical fan "
                    "messages. Read them, but do not treat differences between "
                    "them as evidence about the builds.",
                    "",
                ]
            )
        scenario_key: dict[str, Any] = {}
        for label, role in zip(LABELS, ordered):
            side = arms.get(role) or {}
            scenario_key[label] = {
                "role": role,
                "conversation_core": side.get("conversation_core"),
                "fan_id": side.get("fan_id"),
            }
            lines.extend(_render_side(label, _transcript_rows(side.get("turns") or []), terms))
            lines.extend(_render_rubric(label))
        lines.extend(
            [
                "**Which conversation held together better as a conversation, "
                "and on which dimensions?** (A / B / neither — and why)",
                "",
                "**Anything either conversation did that a person would not:** ",
                "",
                "---",
                "",
            ]
        )
        mapping["scenarios"][scenario_id] = scenario_key

    document = "\n".join(lines)
    leaks = leaked_terms(document, terms)
    if leaks:  # pragma: no cover - guarded by tests, never expected in practice
        raise RuntimeError(
            "the blind review document would have identified the runtimes: "
            + ", ".join(leaks)
        )
    return document, mapping


def unblind(mapping: dict[str, Any], scenario_id: str, label: str) -> dict[str, Any]:
    """Which runtime wrote Conversation ``label`` in ``scenario_id``."""
    scenarios = mapping.get("scenarios") or {}
    entry = (scenarios.get(str(scenario_id)) or {}).get(str(label).upper())
    if not entry:
        raise KeyError(f"no mapping for scenario {scenario_id!r} label {label!r}")
    return dict(entry)


def unblind_all(mapping: dict[str, Any]) -> dict[str, dict[str, str]]:
    """The whole key, as ``{scenario_id: {label: core_id}}``."""
    result: dict[str, dict[str, str]] = {}
    for scenario_id, entry in (mapping.get("scenarios") or {}).items():
        if not isinstance(entry, dict) or "skipped" in entry:
            continue
        result[str(scenario_id)] = {
            str(label): str(side.get("conversation_core") or "")
            for label, side in entry.items()
            if isinstance(side, dict)
        }
    return result


def write_review(
    paired: dict[str, Any],
    *,
    seed: int,
    run_id: str,
    review_path,
    mapping_path,
    extra_forbidden: Sequence[str] = (),
) -> tuple[str, dict[str, Any]]:
    """Build and write both halves, to two different files."""
    document, mapping = build_review(
        paired, seed=seed, run_id=run_id, extra_forbidden=extra_forbidden
    )
    review_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(document, encoding="utf-8")
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return document, mapping


__all__ = [
    "ALWAYS_FORBIDDEN",
    "DIMENSIONS",
    "Dimension",
    "LABELS",
    "build_review",
    "forbidden_terms",
    "leaked_terms",
    "order_for",
    "redact",
    "unblind",
    "unblind_all",
    "write_review",
]
