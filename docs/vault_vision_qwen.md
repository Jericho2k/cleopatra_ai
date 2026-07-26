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
   ```

The endpoint scales to zero after 60 seconds. If it is unavailable, malformed,
or refuses a frame, the backend saves the local NudeNet result instead.
