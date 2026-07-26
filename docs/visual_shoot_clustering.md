# Visual photoshoot clustering

Vault categorization always uses a local NudeNet ONNX detector for exposed
anatomy and nudity evidence. An optional self-hosted Qwen3-VL endpoint enriches
that result with activity, pose, wardrobe, and scene detail. Endpoint failure
falls back to the local result instead of leaving an item uncategorized.

Local Pillow features preserve palette, lighting, orientation, color
histograms, and near-duplicate hashes.

Complete-link clustering merges a group only when every image pair clears the
configured similarity threshold. Generic album names and legacy scene slugs are
not accepted as shoot evidence. Creator-named albums remain a conservative
fallback.

No Anthropic call, AWS account, IAM policy, Rekognition permission, or Bedrock
model access is used by vault categorization.

## Calibration

1. Re-analyze a labelled sample containing multiple known shoots.
2. Read `GET /debug-shoot-clusters/{creator_id}` using dashboard authentication.
3. Compare the returned groups with the known shoot boundaries.
4. Raise `VAULT_SHOOT_MIN_SIMILARITY` if unrelated shoots merge. Lower it only
   when a known shoot is consistently split.
5. Generate draft sets and review them before approval.

The default threshold is intentionally conservative (`0.94`). Prefer splits to
false merges because a false merge can combine unrelated media in a sellable
set.
