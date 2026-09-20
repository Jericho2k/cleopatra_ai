"""An OPTIONAL model reader of two blinded conversations. Never a scoreboard.

``eval/README.md`` and ``services/trajectory_eval.py`` both record the standing
refusal this module has to live inside: *a scripted cooperative customer and a
model grading its own text are insufficient substitutes for expert human
review*. Nothing here changes that. This exists because a suite of twelve
scenarios run repeatedly is more reading than a person will do every time, and a
consistent first pass that flags where to look is worth having — as long as it
cannot be mistaken for the review.

So four rules are built into the shape of it rather than written in a comment:

**It is off unless asked for.** Nothing calls it. ``scripts/run_conversation_judge.py``
is a separate command over an existing run directory, and a run without one is
complete.

**It never learns which runtime is which.** It receives Conversation A and
Conversation B from the same blinding the human reviewer gets, with the same
leak check applied to the prompt before it is sent. Not because a model would
be offended, but because a prompt that says "the new architecture" is an
instruction, and its output would be a very expensive restatement of the
instruction.

**It produces per-dimension results with evidence, not a number.** Each
dimension gets a rating and a short observable reason that quotes or points at
the conversation. There is no total, no winner field and no average: the
dimensions are separate because the failures are separate.

**It is not asked to think privately.** The prompt asks for the rating and the
observable reason. It does not ask for hidden reasoning, scratchpads, or an
internal monologue to be revealed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from services.blind_conversation_review import (
    DIMENSIONS,
    LABELS,
    forbidden_terms,
    leaked_terms,
    order_for,
    redact,
)

#: What the judge is told it is. Neutral by construction: two builds, no order
#: of merit, no history, no hypothesis.
JUDGE_SYSTEM = """You are reading two complete conversations between a fan and a \
creator on a subscription platform. Both were produced by software; the fan's \
messages are identical in both, so only the creator's replies differ.

Rate each conversation on each dimension separately, using the anchors given. \
Do not produce an overall score, a total, or a winner: the dimensions are \
separate because the qualities are separate.

For every rating give one short reason that points at something observable in \
the conversation — a turn number, a phrase that was used, a thing that was or \
was not picked up. Do not describe your reasoning process; give the finding.

Where a scenario does not exercise a dimension at all, use "n/a" rather than a \
number.

Reply with JSON only, in exactly this shape:

{"A": {"<dimension_key>": {"rating": 1-5 or "n/a", "reason": "..."}, ...},
 "B": {"<dimension_key>": {"rating": 1-5 or "n/a", "reason": "..."}, ...}}"""


def rubric_text() -> str:
    """The same dimensions and anchors the human reviewer is given."""
    lines = []
    for dimension in DIMENSIONS:
        lines.append(f"{dimension.key} — {dimension.title}")
        lines.append(f"  {dimension.question}")
        lines.append(f"  1 = {dimension.anchor_low}")
        lines.append(f"  3 = {dimension.anchor_mid}")
        lines.append(f"  5 = {dimension.anchor_high}")
        if dimension.optional:
            lines.append('  "n/a" is correct when the scenario does not exercise this.')
        lines.append("")
    return "\n".join(lines)


def _transcript(turns: Sequence[dict[str, Any]], terms: Sequence[str]) -> str:
    lines = []
    for index, turn in enumerate(turns or []):
        fan = redact(str(turn.get("fan_input") or ""), terms).strip()
        lines.append(f"[{turn.get('turn_index', index)}] fan: {fan or '(no message; time passed)'}")
        replies = [str(part) for part in (turn.get("creator_output") or [])]
        if replies:
            for part in replies:
                lines.append(f"     creator: {redact(part, terms).strip()}")
        else:
            lines.append("     creator: (sent nothing)")
    return "\n".join(lines)


def build_prompt(
    pair: dict[str, Any], *, seed: int, extra_forbidden: Sequence[str] = ()
) -> tuple[str, list[dict[str, str]], dict[str, str]]:
    """The judge's prompt for one scenario, and the mapping that unblinds it.

    Returns ``(system, messages, mapping)`` where ``mapping`` is
    ``{label: role}``. Raises if the assembled prompt would identify a runtime,
    for the same reason the human document does: a blind review that is not
    blind is worse than none.
    """
    arms = pair.get("arms") or {}
    roles = sorted(arms.keys())
    if len(roles) < 2:
        raise ValueError("a judgement needs two arms")
    cores = [
        str(side.get("conversation_core"))
        for side in arms.values()
        if side.get("conversation_core")
    ]
    terms = forbidden_terms([*cores, *extra_forbidden])
    ordered = order_for(str(pair.get("scenario_id") or ""), seed=seed, roles=roles)

    sections = [
        "Dimensions and anchors:",
        "",
        rubric_text(),
        "",
    ]
    mapping: dict[str, str] = {}
    for label, role in zip(LABELS, ordered):
        mapping[label] = role
        sections.extend(
            [
                f"Conversation {label}:",
                "",
                _transcript((arms.get(role) or {}).get("turns") or [], terms),
                "",
            ]
        )
    user = "\n".join(sections)
    leaks = leaked_terms(user, terms) + leaked_terms(JUDGE_SYSTEM, terms)
    if leaks:  # pragma: no cover - guarded by tests
        raise RuntimeError(
            "the judge prompt would have identified the runtimes: " + ", ".join(leaks)
        )
    return JUDGE_SYSTEM, [{"role": "user", "content": user}], mapping


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_judgement(text: str) -> dict[str, Any]:
    """Read the judge's JSON, keeping only dimensions the rubric defines.

    Tolerant of a model that wrapped the JSON in prose or a code fence, and
    strict about the contents: an invented dimension is dropped rather than
    carried into an artifact where it would look like part of the rubric.
    """
    raw = str(text or "")
    match = _JSON_BLOCK.search(raw)
    if not match:
        return {"error": "no JSON object in the response", "raw": raw[:2000]}
    try:
        payload = json.loads(match.group(0))
    except ValueError as exc:
        return {"error": f"unparseable JSON: {exc}", "raw": raw[:2000]}
    if not isinstance(payload, dict):
        return {"error": "JSON was not an object", "raw": raw[:2000]}

    known = {dimension.key for dimension in DIMENSIONS}
    result: dict[str, Any] = {}
    for label in LABELS:
        side = payload.get(label) or payload.get(label.lower())
        if not isinstance(side, dict):
            continue
        cleaned: dict[str, Any] = {}
        for key, value in side.items():
            if str(key) not in known:
                continue
            if isinstance(value, dict):
                cleaned[str(key)] = {
                    "rating": value.get("rating"),
                    "reason": str(value.get("reason") or "")[:600],
                }
            else:
                cleaned[str(key)] = {"rating": value, "reason": ""}
        result[label] = cleaned
    if not result:
        return {"error": "no labelled ratings in the response", "raw": raw[:2000]}
    return result


def unblind_judgement(
    judgement: dict[str, Any], mapping: dict[str, str]
) -> dict[str, Any]:
    """Re-key a parsed judgement from labels to arm roles.

    Kept separate from parsing so the blinded form is what gets stored next to
    the review, and the unblinded form is produced on demand.
    """
    if "error" in judgement:
        return dict(judgement)
    return {
        mapping.get(label, label): value
        for label, value in judgement.items()
        if label in LABELS
    }


__all__ = [
    "JUDGE_SYSTEM",
    "build_prompt",
    "parse_judgement",
    "rubric_text",
    "unblind_judgement",
]
