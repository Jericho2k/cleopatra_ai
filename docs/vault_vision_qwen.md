# Optional Qwen3-VL vault enrichment

The backend works without this endpoint: NudeNet still classifies nudity and
exposed anatomy locally. Qwen adds richer activity, pose, wardrobe, setting,
and searchable descriptions.

Qwen3-VL-4B-Instruct is self-hosted so a third-party assistant API cannot block
adult vault images. The model is Apache-2.0 licensed.

## Modal deployment

1. Create a Modal account.
2. Install and authenticate the CLI:

   ```bash
   pip install modal
   modal setup
   ```

3. Populate the persistent model-weight volume. Run this once for each pinned
   model revision, before directing production traffic at a new deployment:

   ```bash
   modal run deploy/modal_qwen_vl.py::download_weights
   ```

4. Deploy:

   ```bash
   modal deploy deploy/modal_qwen_vl.py
   ```

5. Create a Modal proxy-auth token and copy its key and secret.
6. Add the endpoint URL and token to Railway:

   ```text
   VAULT_VISION_BASE_URL=https://YOUR-ENDPOINT.modal.run
   VAULT_VISION_MODAL_KEY=wk-...
   VAULT_VISION_MODAL_SECRET=ws-...
   VAULT_VISION_MODEL=Qwen/Qwen3-VL-4B-Instruct
   VAULT_VISION_TIMEOUT_SECONDS=620
   ```

The endpoint scales to zero after 60 seconds. The first request can take
longer while Modal loads the model into GPU memory. Weights are pinned to an
immutable Hugging Face revision and pre-populated in a persistent Volume, so
production requests do not depend on a mutable model branch or a full Hub
download. The backend follows Modal's long-running result redirects and allows
up to 620 seconds for a cold request.

The deployment intentionally uses one scale-to-zero L4 container. Requests
queue instead of starting duplicate GPUs, which keeps low-volume and manual
reanalyzes predictable. Change `max_containers` only after production metrics
show sustained queueing that justifies parallel GPU spend.

Successful enrichment logs:

```text
[CATEGORIZE RAW] ... provider=local_nudenet+qwen_vl ... vision=ready
[SHOOT FINGERPRINT] ... status=local_plus_vision ...
[VAULT VISION REQUEST] ... status=ready queue_ms=... round_trip_ms=... inference_ms=...
```

If the endpoint is unavailable, malformed, or refuses a frame, the backend
saves the local NudeNet result and logs a safe reason such as `timeout`,
`http_401`, or `http_500`:

```text
[VAULT VISION FALLBACK] reason=http_401
[CATEGORIZE RAW] ... provider=local_nudenet ... vision=fallback vision_error=http_401
```

Qwen returns a concise description plus structured setting, background,
wardrobe, materials, styling, lighting, composition, color associations, and
same-shoot continuity markers. Those fields are stored with the item and
included in generated set descriptions. Strong structured continuity can also
support the local color/hash evidence when grouping different poses or crops
from the same shoot.

The web endpoint rejects oversized images, malformed base64, extreme
dimensions, empty/oversized prompts, and unauthenticated requests before GPU
inference. Logs contain only request hashes, sizes, dimensions, timings, model
revision, and status; they do not contain media bytes, prompts, or proxy
secrets.
