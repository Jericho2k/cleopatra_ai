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

3. Deploy:

   ```bash
   modal deploy deploy/modal_qwen_vl.py
   ```

4. Create a Modal proxy-auth token and copy its key and secret.
5. Add the endpoint URL and token to Railway:

   ```text
   VAULT_VISION_BASE_URL=https://YOUR-ENDPOINT.modal.run
   VAULT_VISION_MODAL_KEY=wk-...
   VAULT_VISION_MODAL_SECRET=ws-...
   VAULT_VISION_MODEL=Qwen/Qwen3-VL-4B-Instruct
   VAULT_VISION_TIMEOUT_SECONDS=620
   ```

The endpoint scales to zero after 60 seconds. The first request can take
several minutes while Modal downloads and loads the model. The backend follows
Modal's long-running result redirects and allows up to 620 seconds for that
first request. Warm requests are normally much faster.

Successful enrichment logs:

```text
[CATEGORIZE RAW] ... provider=local_nudenet+qwen_vl ... vision=ready
[SHOOT FINGERPRINT] ... status=local_plus_vision ...
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
