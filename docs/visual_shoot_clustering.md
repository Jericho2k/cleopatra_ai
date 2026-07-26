# Visual photoshoot clustering

## Why this exists

The vision classifier answers factual inventory questions such as explicitness,
visible anatomy and scene detail. It cannot safely decide that two images came
from the same photoshoot. A generic album plus `bedroom + nude` can contain many
unrelated shoots.

Version 7 therefore separates the contracts:

1. One vision pass supplies the adult taxonomy *and* the rich inventory detail
   (setting, wardrobe, pose, camera angle, background, composition). These come
   from a single reading of the frame rather than being merged after the fact.
2. An image-embedding endpoint supplies visual photoshoot similarity.
3. Deterministic local features preserve palette, lighting, orientation and
   near-duplicate evidence.
4. Complete-link clustering only merges groups when every cross-group image
   pair clears the configured similarity threshold.

Generic albums and the legacy scene slug are not accepted as photoshoot
evidence. If visual embeddings are unavailable, media remains ungrouped unless
the creator deliberately organized it into a named album.

## Providers

Both endpoints are chosen by the operator; nothing here is vendor-specific.

- **Classification** — `VISION_PROVIDER` / `VISION_MODEL`, called through
  `ai/model_providers.py`. The provider's terms must permit adult imagery.
  `assert_adult_eligible` rejects an Anthropic target because it refuses this
  catalogue, and a silent refusal would store a confident-looking "clothed" row
  for explicit media. Serve locally with, for example:
  `vllm serve Qwen/Qwen2.5-VL-72B-Instruct`.
- **Embeddings** — `VAULT_SHOOT_EMBEDDING_BASE_URL` /
  `VAULT_SHOOT_EMBEDDING_MODEL`, any OpenAI-compatible embeddings endpoint
  (vLLM, Infinity, TEI) serving an image embedding model such as SigLIP.

Required runtime values are documented in `.env.example`.

If the embedding endpoint is missing, content classification still succeeds and
records local visual evidence. Automatic set construction deliberately does not
guess photoshoot identity for those items — palette and dHash alone cannot
separate two shoots in the same room, and a wrong merge would put unrelated
media into one sellable set.

## Calibration before a whole-vault run

1. Re-analyze a labelled sample containing multiple known shoots from the same
   generic album.
2. Read `GET /debug-shoot-clusters/{creator_id}` using normal dashboard
   authentication.
3. Compare each returned `media_ids` group with the known shoot boundaries.
4. Raise `VAULT_SHOOT_MIN_SIMILARITY` if unrelated shoots merge. Lower it only
   when a known shoot is consistently split. Prefer splits over false merges.
5. Re-analyze the remaining stale Version 6 items only after the sample is
   correct.
6. Generate draft sets and review them before approval. Approved/manual sets
   are never deleted by generation.

The default threshold is intentionally conservative (`0.86`). It is a starting
point, not a universal truth; real creator vaults must supply the final
calibration evidence.
