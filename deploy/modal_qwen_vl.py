"""Scale-to-zero Qwen3-VL endpoint for Cleopatra vault enrichment.

Deploy with:
    pip install modal
    modal setup
    modal run deploy/modal_qwen_vl.py::download_weights
    modal deploy deploy/modal_qwen_vl.py
"""
import base64
import binascii
import hashlib
import io
import re
import time

import modal


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MODEL_DIR = "/models"
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2048
MAX_PROMPT_CHARS = 16_000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate==1.14.0",
        "fastapi[standard]==0.140.0",
        "huggingface-hub==1.24.0",
        "pillow==12.3.0",
        "torch==2.13.0",
        "torchvision==0.28.0",
        "transformers==5.14.1",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)
weights = modal.Volume.from_name(
    "cleopatra-qwen3-vl-weights",
    create_if_missing=True,
)
app = modal.App("cleopatra-vault-vision", image=image)


@app.function(
    cpu=2,
    memory=4096,
    timeout=1800,
    volumes={MODEL_DIR: weights},
)
def download_weights() -> dict:
    """Populate the persistent Volume before a GPU handles live traffic."""
    from huggingface_hub import snapshot_download

    started = time.monotonic()
    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=MODEL_DIR,
    )
    weights.commit()
    elapsed = round(time.monotonic() - started, 2)
    print(
        f"[VAULT VISION] weights ready model={MODEL_ID} "
        f"revision={MODEL_REVISION[:12]} seconds={elapsed}"
    )
    return {
        "status": "ready",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "path": path,
        "seconds": elapsed,
    }


@app.cls(
    gpu="L4",
    timeout=600,
    startup_timeout=600,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=60,
    volumes={MODEL_DIR: weights},
)
class VaultVision:
    @modal.enter()
    def load(self):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        started = time.monotonic()
        print(
            f"[VAULT VISION] loading model={MODEL_ID} "
            f"revision={MODEL_REVISION[:12]}"
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=MODEL_DIR,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=MODEL_DIR,
            device_map="auto",
            dtype="auto",
        )
        self.model.eval()
        weights.commit()
        print(
            f"[VAULT VISION] ready model={MODEL_ID} "
            f"revision={MODEL_REVISION[:12]} "
            f"seconds={time.monotonic() - started:.2f}"
        )

    @modal.fastapi_endpoint(
        method="POST",
        docs=False,
        requires_proxy_auth=True,
    )
    def classify(self, payload: dict) -> dict:
        from fastapi import HTTPException
        from PIL import Image
        import torch

        started = time.monotonic()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="JSON object required")
        encoded = payload.get("image_base64")
        prompt = payload.get("prompt")
        if not isinstance(encoded, str) or not encoded:
            raise HTTPException(status_code=422, detail="image_base64 required")
        if len(encoded) > ((MAX_IMAGE_BYTES * 4 // 3) + 16):
            raise HTTPException(status_code=413, detail="image payload too large")
        if not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(status_code=422, detail="prompt required")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise HTTPException(status_code=413, detail="prompt too large")
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail="invalid base64 image",
            ) from exc
        if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="image payload too large")
        try:
            picture = Image.open(io.BytesIO(image_bytes))
            if (
                min(picture.size) < 32
                or max(picture.size) > MAX_IMAGE_DIMENSION
            ):
                raise HTTPException(
                    status_code=422,
                    detail="unsupported image dimensions",
                )
            picture.load()
            picture = picture.convert("RGB")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail="invalid image") from exc
        request_id = re.sub(
            r"[^a-zA-Z0-9_-]",
            "",
            str(payload.get("request_id") or ""),
        )[:64] or hashlib.sha256(image_bytes).hexdigest()[:12]
        print(
            f"[VAULT VISION] classification started request={request_id} "
            f"bytes={len(image_bytes)} dimensions={picture.width}x{picture.height}"
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": picture},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=1400,
                do_sample=False,
            )
        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = self.processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        elapsed_ms = round((time.monotonic() - started) * 1000)
        print(
            f"[VAULT VISION] classification completed request={request_id} "
            f"latency_ms={elapsed_ms} output_chars={len(text)}"
        )
        return {
            "text": text,
            "request_id": request_id,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "latency_ms": elapsed_ms,
        }
