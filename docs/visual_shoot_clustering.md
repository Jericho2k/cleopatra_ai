# Visual photoshoot clustering

## Why this exists

Rekognition answers factual inventory questions such as explicitness, visible
anatomy and coarse scene labels. It cannot safely decide that two images came
from the same photoshoot. A generic album plus `bedroom + nude` can contain many
unrelated shoots.

Version 5 therefore separates the contracts:

1. Rekognition supplies adult-content taxonomy.
2. Amazon Nova Multimodal Embeddings supplies visual photoshoot similarity.
3. Deterministic local features preserve palette, lighting, orientation and
   near-duplicate evidence.
4. Complete-link clustering only merges groups when every cross-group image
   pair clears the configured similarity threshold.

Generic albums and the legacy scene slug are not accepted as photoshoot
evidence. If visual embeddings are unavailable, media remains ungrouped unless
the creator deliberately organized it into a named album.

## AWS access

The Railway AWS principal that already calls Rekognition also needs this
least-privilege statement:

```json
{
  "Effect": "Allow",
  "Action": "bedrock:InvokeModel",
  "Resource": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-multimodal-embeddings-v1:0"
}
```

Required runtime values are documented in `.env.example`. Nova Multimodal
Embeddings currently runs in `us-east-1`.

If permission/model access is missing, content classification still succeeds
and records local visual evidence. Automatic set construction deliberately
does not guess photoshoot identity for those items.

## Calibration before a whole-vault run

1. Re-analyze a labelled sample containing multiple known shoots from the same
   generic album.
2. Read `GET /debug-shoot-clusters/{creator_id}` using normal dashboard
   authentication.
3. Compare each returned `media_ids` group with the known shoot boundaries.
4. Raise `VAULT_SHOOT_MIN_SIMILARITY` if unrelated shoots merge. Lower it only
   when a known shoot is consistently split. Prefer splits over false merges.
5. Re-analyze the remaining stale Version 4 items only after the sample is
   correct.
6. Generate draft sets and review them before approval. Approved/manual sets
   are never deleted by generation.

The default threshold is intentionally conservative (`0.86`). It is a starting
point, not a universal truth; real creator vaults must supply the final
calibration evidence.
