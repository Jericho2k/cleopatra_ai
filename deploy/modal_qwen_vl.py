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
SEMANTIC_MODEL_ID = "google/siglip2-base-patch16-224"
SEMANTIC_MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SEMANTIC_MODEL_DIR = "/semantic-models"
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2048
MAX_PROMPT_CHARS = 16_000

SEMANTIC_AXES = {
    "scene_location": [
        "bedroom", "bathroom", "kitchen", "living room", "studio",
        "shower", "bathtub", "outdoors", "vehicle", "hallway",
        "other indoor room",
    ],
    "wardrobe_state": [
        "full nudity", "partial nudity", "lingerie", "underwear",
        "casual clothing", "dress", "swimwear", "costume", "sleepwear",
    ],
    "pose": [
        "standing", "sitting", "lying down", "kneeling", "crouching",
        "close-up body detail",
    ],
    "activity": [
        "posing", "selfie", "mirror selfie", "showering", "dancing",
        "undressing", "sexual activity", "using an adult toy",
    ],
    "framing": [
        "close-up", "medium shot", "three-quarter shot", "full-body shot",
        "wide shot",
    ],
    "lighting": [
        "warm indoor light", "cool blue light",
        "pink or purple colored light", "natural daylight",
        "bright studio light", "dim low light",
    ],
}
SEMANTIC_TAGS = {
    "background_details": [
        "bed and bedding", "kitchen counter and cabinets", "mirror",
        "shower or bathtub", "sofa", "curtains", "tiled wall", "plain wall",
        "studio backdrop",
    ],
    "wardrobe_items": [
        "bra", "panties", "lingerie set", "bodysuit", "stockings",
        "fishnet clothing", "sheer clothing", "lace clothing", "dress",
        "crop top", "shorts", "skirt", "high heels", "jewelry",
    ],
}

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
semantic_weights = modal.Volume.from_name(
    "cleopatra-siglip2-weights",
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


@app.function(
    cpu=2,
    memory=4096,
    timeout=1800,
    volumes={SEMANTIC_MODEL_DIR: semantic_weights},
)
def download_semantic_weights() -> dict:
    """Populate the pinned SigLIP2 volume before it receives live traffic."""
    from huggingface_hub import snapshot_download

    started = time.monotonic()
    path = snapshot_download(
        repo_id=SEMANTIC_MODEL_ID,
        revision=SEMANTIC_MODEL_REVISION,
        cache_dir=SEMANTIC_MODEL_DIR,
    )
    semantic_weights.commit()
    elapsed = round(time.monotonic() - started, 2)
    print(
        f"[VAULT SEMANTICS] weights ready model={SEMANTIC_MODEL_ID} "
        f"revision={SEMANTIC_MODEL_REVISION[:12]} seconds={elapsed}"
    )
    return {
        "status": "ready",
        "model": SEMANTIC_MODEL_ID,
        "revision": SEMANTIC_MODEL_REVISION,
        "path": path,
        "seconds": elapsed,
    }


def _validated_picture(payload: dict):
    from fastapi import HTTPException
    from PIL import Image

    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="JSON object required")
    encoded = payload.get("image_base64")
    if not isinstance(encoded, str) or not encoded:
        raise HTTPException(status_code=422, detail="image_base64 required")
    if len(encoded) > ((MAX_IMAGE_BYTES * 4 // 3) + 16):
        raise HTTPException(status_code=413, detail="image payload too large")
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
        if min(picture.size) < 32 or max(picture.size) > MAX_IMAGE_DIMENSION:
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
    return picture, image_bytes, request_id


def _feature_tensor(output):
    """Handle tensor and Transformers 5 pooled-output return shapes."""
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    return output


def _axis_confident(top_score: float, margin: float) -> bool:
    """Calibrate SigLIP's low absolute sigmoid scores with rank separation."""
    if top_score < 0.06:
        return False
    relative_margin = margin / max(top_score, 0.001)
    return margin >= 0.012 or relative_margin >= 0.18


@app.cls(
    gpu="T4",
    timeout=180,
    startup_timeout=300,
    min_containers=0,
    max_containers=8,
    buffer_containers=0,
    scaledown_window=30,
    volumes={SEMANTIC_MODEL_DIR: semantic_weights},
)
class VaultSemantics:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModel, AutoProcessor

        started = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(
            SEMANTIC_MODEL_ID,
            revision=SEMANTIC_MODEL_REVISION,
            cache_dir=SEMANTIC_MODEL_DIR,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            SEMANTIC_MODEL_ID,
            revision=SEMANTIC_MODEL_REVISION,
            cache_dir=SEMANTIC_MODEL_DIR,
            local_files_only=True,
            dtype="auto",
        ).to("cuda")
        self.model.eval()
        self.text_features = {}
        for group, labels in {**SEMANTIC_AXES, **SEMANTIC_TAGS}.items():
            prompts = [
                f"This adult creator image clearly shows {label}."
                for label in labels
            ]
            inputs = self.processor(
                text=prompts,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                features = _feature_tensor(
                    self.model.get_text_features(**inputs)
                )
                features = features / features.norm(dim=-1, keepdim=True)
            self.text_features[group] = features
        print(
            f"[VAULT SEMANTICS] ready model={SEMANTIC_MODEL_ID} "
            f"revision={SEMANTIC_MODEL_REVISION[:12]} "
            f"seconds={time.monotonic() - started:.2f}"
        )

    def _scores(self, image_features, group):
        import torch

        logits = image_features @ self.text_features[group].T
        scale = self.model.logit_scale.exp()
        bias = self.model.logit_bias
        return torch.sigmoid((logits * scale) + bias)[0].float().cpu().tolist()

    @modal.fastapi_endpoint(
        method="POST",
        docs=False,
        requires_proxy_auth=True,
    )
    def classify(self, payload: dict) -> dict:
        import numpy as np
        import torch

        started = time.monotonic()
        picture, image_bytes, request_id = _validated_picture(payload)
        inputs = self.processor(
            images=[picture],
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            image_features = _feature_tensor(
                self.model.get_image_features(**inputs)
            )
            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True,
            )

        axes = {}
        ambiguous_axes = []
        confidence_rows = []
        for group, labels in SEMANTIC_AXES.items():
            scores = self._scores(image_features, group)
            ranked = sorted(
                zip(labels, scores),
                key=lambda row: row[1],
                reverse=True,
            )[:3]
            margin = ranked[0][1] - ranked[1][1]
            confident = _axis_confident(ranked[0][1], margin)
            if not confident:
                ambiguous_axes.append(group)
            relative_margin = margin / max(ranked[0][1], 0.001)
            confidence_rows.append(
                min(1.0, (ranked[0][1] / 0.35)) * 0.65
                + min(1.0, relative_margin / 0.3) * 0.35
            )
            axes[group] = {
                "ranked": [
                    {"label": label, "score": round(score, 4)}
                    for label, score in ranked
                ],
                "margin": round(margin, 4),
                "confident": confident,
            }

        tags = {}
        for group, labels in SEMANTIC_TAGS.items():
            ranked = sorted(
                zip(labels, self._scores(image_features, group)),
                key=lambda row: row[1],
                reverse=True,
            )
            tags[group] = [
                {"label": label, "score": round(score, 4)}
                for label, score in ranked[:4]
                if score >= 0.16
            ]

        vector = (
            image_features[0]
            .float()
            .cpu()
            .numpy()
            .astype(np.dtype("<f2"))
        )
        elapsed_ms = round((time.monotonic() - started) * 1000)
        print(
            f"[VAULT SEMANTICS] completed request={request_id} "
            f"latency_ms={elapsed_ms} ambiguous={','.join(ambiguous_axes) or 'none'}"
        )
        return {
            "model": SEMANTIC_MODEL_ID,
            "revision": SEMANTIC_MODEL_REVISION,
            "request_id": request_id,
            "latency_ms": elapsed_ms,
            "confidence": round(
                sum(confidence_rows) / max(len(confidence_rows), 1),
                4,
            ),
            "ambiguous_axes": ambiguous_axes,
            "axes": axes,
            "tags": tags,
            "embedding": {
                "encoding": "float16_base64",
                "dimensions": int(vector.shape[0]),
                "data": base64.b64encode(vector.tobytes()).decode("ascii"),
            },
        }


@app.cls(
    gpu="L4",
    timeout=600,
    startup_timeout=600,
    min_containers=0,
    max_containers=2,
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
            local_files_only=True,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=MODEL_DIR,
            local_files_only=True,
            device_map="auto",
            dtype="auto",
        )
        self.model.eval()
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
