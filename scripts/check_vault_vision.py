"""Smoke-test the vault vision classifier against one real image.

Everything in the test suite stubs the network, so this is the check that the
configured endpoint actually exists, accepts images, and returns the contract.
Run it before deploying and before any whole-vault upgrade run.

    python scripts/check_vault_vision.py path/to/image.jpg
    python scripts/check_vault_vision.py https://example.com/photo.jpg --video

Exit codes: 0 usable, 1 misconfigured, 2 refused, 3 bad response.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.vision_classifier import (  # noqa: E402
    VaultClassifierConfigurationError,
    VaultClassifierError,
    VaultClassifierRefusalError,
    classify_with_vision,
    vision_target,
)


def load_image(source: str) -> bytes:
    if source.startswith(("http://", "https://")):
        import httpx

        response = httpx.get(source, timeout=30, follow_redirects=True)
        response.raise_for_status()
        raw = response.content
    else:
        raw = Path(source).read_bytes()

    # Match what the classifier is fed in production: a compact RGB JPEG.
    from PIL import Image

    image = Image.open(io.BytesIO(raw))
    image.seek(0)
    image = image.convert("RGB")
    image.thumbnail((896, 896), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=84, optimize=True)
    return buffer.getvalue()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="local path or URL")
    parser.add_argument("--video", action="store_true", help="treat as a video thumbnail")
    parser.add_argument("--album", default="", help="album title for the scene slug")
    parser.add_argument("--json", action="store_true", help="print the raw contract")
    args = parser.parse_args()

    print(f"VAULT_CLASSIFIER_PROVIDER={os.environ.get('VAULT_CLASSIFIER_PROVIDER') or 'vision (default)'}")
    try:
        target = vision_target()
    except VaultClassifierConfigurationError as exc:
        print(f"FAIL  configuration: {exc}")
        return 1
    print(f"target: {target.provider}:{target.model}")
    print(f"base_url: {target.base_url or '(provider default)'}")

    try:
        image_bytes = load_image(args.image)
    except Exception as exc:
        print(f"FAIL  could not read image: {type(exc).__name__}: {exc}")
        return 1
    print(f"image: {len(image_bytes)} bytes after normalization\n")

    try:
        result = await classify_with_vision(
            image_bytes,
            is_video=args.video,
            album_title=args.album,
            filename=Path(args.image).name,
        )
    except VaultClassifierRefusalError as exc:
        print(f"FAIL  the model refused this image: {exc}")
        print(
            "\nThis is the failure mode the swap exists to surface. The model is "
            "too restrictive for this catalogue — try an uncensored finetune."
        )
        return 2
    except VaultClassifierConfigurationError as exc:
        print(f"FAIL  configuration: {exc}")
        return 1
    except VaultClassifierError as exc:
        print(f"FAIL  {exc}")
        return 3

    metadata = result.get("_provider_metadata") or {}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"category      {result['category']}")
        print(f"explicitness  {result['explicitness']}  "
              f"(model said {metadata.get('reported_explicitness')}"
              f"{', ESCALATED from evidence' if metadata.get('explicitness_escalated') else ''})")
        print(f"nudity        {result['nudity']}")
        print(f"anatomy       {result['visible_anatomy'] or '-'}")
        print(f"activity      {result['sexual_activity'] or '-'}")
        print(f"outfit        {result['scene_outfit']}")
        print(f"location      {result['scene_location']}")
        print(f"confidence    {result['confidence']}")
        print(f"\ndescription   {result['description']}")

    if metadata.get("age_review_required"):
        print("\nHELD  the model flagged a possible minor; this item is withheld "
              "for human review and is never sold.")

    print("\nOK  the endpoint is reachable and returned a valid contract.")
    print("Check the values above against what you can see in the image before "
          "running a whole-vault upgrade — a wrong category sets a wrong price.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
