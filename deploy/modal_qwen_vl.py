"""Scale-to-zero Qwen3-VL endpoint for Cleopatra vault enrichment.

Deploy with:
    pip install modal
    modal setup
    modal deploy deploy/modal_qwen_vl.py
"""
import base64
import io

import modal


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_DIR = "/models"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate>=1.0",
        "fastapi[standard]>=0.115",
        "pillow>=11",
        "torch>=2.6",
        "transformers>=4.57.0",
    )
)
weights = modal.Volume.from_name(
    "cleopatra-qwen3-vl-weights",
    create_if_missing=True,
)
app = modal.App("cleopatra-vault-vision", image=image)


@app.cls(
    gpu="L4",
    timeout=600,
    scaledown_window=60,
    volumes={MODEL_DIR: weights},
)
class VaultVision:
    @modal.enter()
    def load(self):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_DIR,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_DIR,
            device_map="auto",
            dtype="auto",
        )
        weights.commit()

    @modal.fastapi_endpoint(
        method="POST",
        docs=False,
        requires_proxy_auth=True,
    )
    def classify(self, payload: dict) -> dict:
        from PIL import Image

        image_bytes = base64.b64decode(payload["image_base64"], validate=True)
        picture = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": picture},
                {"type": "text", "text": str(payload["prompt"])},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=900,
            do_sample=False,
        )
        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = self.processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return {"text": text}
