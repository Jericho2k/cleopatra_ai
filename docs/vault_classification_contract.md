# Vault classification contract (classifier v3)

Vault media is classified by two sources with different jobs. Neither is
trusted alone.

- **Adult classifier** (`NSFW_CLASSIFIER_PROVIDER`, Sightengine `nudity-2.1`)
  owns explicitness. It is purpose-built for this catalogue and does not
  refuse.
- **Vision model** (`VISION_PROVIDER` / `VISION_MODEL`) owns semantics:
  category, description, outfit, location, lighting, mood, tags.

`services/vault_classification.py` reconciles them. It is pure; all network
access lives in `services/adult_classifier.py`, `services/video_frames.py`,
and `ai/model_providers.py`.

## Provider eligibility

Vault imagery is always an adult workload. `ai.model_providers` refuses to
send it to a provider listed in `ADULT_INELIGIBLE_PROVIDERS` or marked
`adult_policy: ineligible` in `config/model_candidates.json`. Anthropic is
ineligible, so `VISION_PROVIDER=anthropic` fails closed rather than sending
the request.

## Reconciliation rules

- The classifier may **escalate** a category, never **demote** one. An
  under-read is a silent underpricing bug, because `price_min`/`price_max`
  derive from the category. A demotion on classifier noise would move paid
  media toward the free teaser tiers, so a low classifier reading keeps the
  category and only flags the row.
- Ties between classifier classes resolve toward the more explicit class.
- Categories with no declared band (`dictate_video`, `task`, `other`) are
  defined by intent and are never escalated.
- Videos are scored on their **peak** frame, not their average.
- Every disagreement is recorded rather than silently resolved.

## Stored fields

| Column | Meaning |
|--------|---------|
| `classification_version` | `CLASSIFIER_VERSION`; rows below it are stale |
| `classification_source` | `hybrid`, `vision_only`, `classifier_only`, `failed`, `manual` |
| `classification_model` | The vision target plus the classifier provider |
| `classification_confidence` | Numeric 0..1; the dashboard renders it as a percentage |
| `classification_evidence` | `high`, `low`, `vision_only`, `classifier_only`, `unavailable` |
| `classification_needs_review` | Operator should check category and price |
| `classification_disagreement` | Why it was flagged |
| `classifier_explicitness` / `vision_explicitness` | Both raw readings |
| `classifier_scores` | Raw per-class scores, so a mapping change can be re-evaluated without re-billing |
| `analyzed_frame_count` | 0 failed fetch, 1 photo, N sampled video |

## Degradation

No single failure loses an item:

- classifier disabled or erroring → vision-only, labelled;
- vision model unreachable → the classifier's explicitness still picks a
  **priced** tier (never `other`, which is $0), flagged for review;
- neither available → a storable row held for review;
- no `ffmpeg` → videos fall back to filename and album inference.

## Upgrade runs

`POST /categorize-vault/{creator_id}?mode=upgrade` is the only mode that
rewrites existing metadata, so it requires `confirm_upgrade=true` and accepts
`upgrade_scope=all|approved`. It selects rows below `CLASSIFIER_VERSION` that
already have a category; `initial` and `new` only ever select uncategorized
rows.

Bump `CLASSIFIER_VERSION` whenever a change makes existing metadata worth
redoing. The overview endpoint reports `classifier_version`,
`stale_classifications`, and `stale_approved_classifications` from the
`vault_classification_staleness` function in `db/vault_classification_v3.sql`.

## Operational requirements

- Apply `db/vault_classification_v3.sql`.
- `ffmpeg` and `ffprobe` on `PATH` for video keyframes.
- A vision endpoint that accepts multiple images per request; for vLLM that
  means `--limit-mm-per-prompt image=8` at or above `VIDEO_FRAME_COUNT`.
