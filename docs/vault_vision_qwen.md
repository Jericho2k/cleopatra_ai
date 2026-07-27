# Hybrid NudeNet, SigLIP2, and Qwen vault analysis

Every image first gets local NudeNet evidence. A small SigLIP2 endpoint then
adds controlled scene, wardrobe, pose, activity, framing, and lighting labels
plus a compact image embedding. Qwen is called only when those signals report
sexual activity, disagree with NudeNet about full nudity, or leave all core
semantics ambiguous.

Qwen3-VL-4B-Instruct is self-hosted so a third-party assistant API cannot block
adult vault images. The model is Apache-2.0 licensed.

## Modal deployment

1. Create a Modal account.
2. Install and authenticate the CLI:

   ```bash
   pip install modal
   modal setup
   ```

3. Populate both persistent model-weight volumes. Run these once for each
   pinned revision, before directing production traffic at a new deployment:

   ```bash
   modal run deploy/modal_qwen_vl.py::download_weights
   modal run deploy/modal_qwen_vl.py::download_semantic_weights
   ```

4. Deploy:

   ```bash
   modal deploy deploy/modal_qwen_vl.py
   ```

5. Create a Modal proxy-auth token and copy its key and secret.
6. Add the endpoint URL and token to Railway:

   ```text
   VAULT_SEMANTIC_BASE_URL=https://YOUR-SEMANTIC-ENDPOINT.modal.run
   VAULT_VISION_BASE_URL=https://YOUR-ENDPOINT.modal.run
   VAULT_VISION_MODAL_KEY=wk-...
   VAULT_VISION_MODAL_SECRET=ws-...
   VAULT_VISION_MODEL=Qwen/Qwen3-VL-4B-Instruct
   VAULT_VISION_TIMEOUT_SECONDS=620
   VAULT_SEMANTIC_TIMEOUT_SECONDS=180
   VAULT_CATEGORIZATION_CONCURRENCY=12
   ```

The Qwen endpoint scales to zero after 60 seconds. The first request can take
longer while Modal loads the model into GPU memory. Weights are pinned to an
immutable Hugging Face revision and pre-populated in a persistent Volume, so
production requests do not depend on a mutable model branch or a full Hub
download. GPU containers use local-only model loading; only the explicit
CPU-based preload step contacts the Hub. The backend follows Modal's
long-running result redirects and allows up to 620 seconds for a cold request.

Qwen uses up to two scale-to-zero L4 containers because it is the exception
path. SigLIP2 uses smaller T4 containers, scales up to eight, and scales down
after 30 seconds. Together they stay within Modal Starter's ten-GPU concurrency
limit. A vault run defaults to 12 concurrent requests. This keeps bulk imports
parallel while ordinary reanalysis still scales to zero.

Runs over 100 items do not block on Qwen merely because all core semantic axes
are weak; they store the controlled best labels and embedding and report
`qwen_status=deferred_bulk`. Strong activity or NudeNet/full-nudity conflicts
still use Qwen. Manual and smaller runs keep the complete fallback path.

Successful enrichment logs:

```text
[CATEGORIZE RAW] ... provider=local_nudenet+siglip2 ... vision=ready
[SHOOT FINGERPRINT] ... status=local_plus_vision ...
[VAULT SEMANTICS] ... status=ready confidence=... ambiguous=...
[VAULT VISION REQUEST] ... status=ready queue_ms=... round_trip_ms=... inference_ms=...
```

If the endpoint is unavailable, malformed, or refuses a frame, the backend
saves the local NudeNet result and logs a safe reason such as `timeout`,
`http_401`, or `http_500`:

```text
[VAULT VISION FALLBACK] reason=http_401
[CATEGORIZE RAW] ... provider=local_nudenet ... vision=fallback vision_error=http_401
```

SigLIP2 produces deterministic controlled labels and an embedding for every
successfully analyzed image. The backend converts those labels into concise
searchable text without paying a text-model token cost. Grouping remains
complete-link and only bridges a pose/crop change when local pixels, controlled
labels, and embeddings agree. Qwen's richer prose is saved only for fallback
items.

Production logs expose `qwen_status`, fallback reasons, semantic confidence,
ambiguous axes, queue time, and inference time. Validate those rates on a
representative vault before changing thresholds; model confidence is not a
substitute for real same-shoot precision and recall.

The web endpoint rejects oversized images, malformed base64, extreme
dimensions, empty/oversized prompts, and unauthenticated requests before GPU
inference. Logs contain only request hashes, sizes, dimensions, timings, model
revision, and status; they do not contain media bytes, prompts, or proxy
secrets.
