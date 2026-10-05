"""
ComfyUI image provider (self-hosted).

Sends an API-format workflow JSON to a ComfyUI server, waits for it, and saves
the first output image. Workflows live in comfyui_workflows/*.json and use
placeholders that are substituted at run time:

    {{PROMPT}}  {{NEGATIVE}}  {{SEED}}  {{WIDTH}}  {{HEIGHT}}
    {{REF_1}} {{REF_2}} ...   (filenames of reference images uploaded to ComfyUI)

A string that is exactly one placeholder is replaced with the typed value
(so {{SEED}} becomes an int); placeholders embedded in longer strings are
replaced as text.

Two workflows are used: COMFYUI_WORKFLOW when reference images are supplied,
COMFYUI_WORKFLOW_NOREF otherwise.
"""

import json
import logging
import random
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

from config import (
    COMFYUI_URL,
    COMFYUI_WORKFLOW,
    COMFYUI_WORKFLOW_NOREF,
    COMFYUI_TIMEOUT,
    COMFYUI_GEN_WIDTH,
    COMFYUI_GEN_HEIGHT,
    COMFYUI_NEGATIVE_PROMPT,
    THUMBNAIL_WIDTH,
    THUMBNAIL_HEIGHT,
)

logger = logging.getLogger(__name__)

WORKFLOWS_DIR = Path(__file__).parent / "comfyui_workflows"
_PLACEHOLDER = re.compile(r"\{\{([A-Z_0-9]+)\}\}")


def _load_workflow(name: str) -> dict:
    path = WORKFLOWS_DIR / (name if name.endswith(".json") else f"{name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _substitute(node, values: dict):
    """Recursively replace {{PLACEHOLDERS}} in a workflow structure."""
    if isinstance(node, dict):
        return {k: _substitute(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, values) for v in node]
    if isinstance(node, str):
        whole = _PLACEHOLDER.fullmatch(node)
        if whole and whole.group(1) in values:
            return values[whole.group(1)]
        return _PLACEHOLDER.sub(
            lambda m: str(values.get(m.group(1), m.group(0))), node
        )
    return node


def _upload_reference(path: Path) -> str:
    """Upload an image to ComfyUI's input folder; returns the stored filename."""
    with open(path, "rb") as f:
        r = requests.post(
            f"{COMFYUI_URL}/upload/image",
            files={"image": (f"yt_{uuid.uuid4().hex[:8]}_{path.name}", f, "image/png")},
            data={"overwrite": "true"},
            timeout=60,
        )
    r.raise_for_status()
    return r.json()["name"]


def _run_workflow(workflow: dict) -> Optional[bytes]:
    r = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": workflow, "client_id": uuid.uuid4().hex},
        timeout=30,
    )
    if r.status_code != 200:
        logger.error(f"ComfyUI rejected workflow: {r.status_code} {r.text[:500]}")
        return None
    prompt_id = r.json()["prompt_id"]

    deadline = time.time() + COMFYUI_TIMEOUT
    while time.time() < deadline:
        h = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=30).json()
        entry = h.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                logger.error(f"ComfyUI execution error: {status.get('messages', [])[-1:]}")
                return None
            for out in entry.get("outputs", {}).values():
                for img in out.get("images", []):
                    if img.get("type") == "temp":
                        continue
                    v = requests.get(
                        f"{COMFYUI_URL}/view",
                        params={
                            "filename": img["filename"],
                            "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output"),
                        },
                        timeout=60,
                    )
                    v.raise_for_status()
                    return v.content
            if status.get("completed"):
                logger.error("ComfyUI finished but produced no image output")
                return None
        time.sleep(1.5)

    logger.error(f"ComfyUI timed out after {COMFYUI_TIMEOUT}s")
    try:
        requests.post(f"{COMFYUI_URL}/interrupt", timeout=10)
    except requests.RequestException:
        pass
    return None


def generate_with_comfyui(
    prompt: str,
    reference_images: list[Path] = None,
    output_path: Path = None,
    model: str = None,
    concept=None,
    variation=None,
    image=None,
) -> Optional[Path]:
    """
    Generate an image via ComfyUI. Signature matches the other providers.

    `model` selects a workflow by name (a file in comfyui_workflows/); if it is
    None or "comfyui" the configured defaults are used.
    """
    try:
        refs = [p for p in (reference_images or []) if p and Path(p).exists()]
        if model and model != "comfyui":
            workflow_name = model
        else:
            workflow_name = COMFYUI_WORKFLOW if refs else COMFYUI_WORKFLOW_NOREF

        logger.info(f"Generating image with ComfyUI ({workflow_name}, {len(refs)} refs)...")
        workflow = _load_workflow(workflow_name)

        values = {
            "PROMPT": prompt,
            "NEGATIVE": COMFYUI_NEGATIVE_PROMPT,
            "SEED": random.randint(0, 2**32 - 1),
            "WIDTH": COMFYUI_GEN_WIDTH,
            "HEIGHT": COMFYUI_GEN_HEIGHT,
        }
        for i, ref in enumerate(refs, start=1):
            values[f"REF_{i}"] = _upload_reference(Path(ref))

        # Workflows that expect refs we don't have should use the NOREF variant;
        # drop nothing silently - unresolved placeholders will fail loudly in ComfyUI.
        data = _run_workflow(_substitute(workflow, values))
        if not data:
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(".raw.png")
        tmp.write_bytes(data)
        with Image.open(tmp) as img:
            img = img.convert("RGB")
            # Center-crop to 16:9 then resize to the thumbnail size
            target = THUMBNAIL_WIDTH / THUMBNAIL_HEIGHT
            w, h = img.size
            if w / h > target:
                nw = int(h * target)
                img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
            elif w / h < target:
                nh = int(w / target)
                img = img.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
            img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.LANCZOS).save(output_path)
        tmp.unlink(missing_ok=True)
        return output_path

    except requests.ConnectionError:
        logger.error(f"Cannot reach ComfyUI at {COMFYUI_URL}")
        return None
    except FileNotFoundError as e:
        logger.error(f"ComfyUI workflow not found: {e}")
        return None
    except Exception as e:
        logger.error(f"ComfyUI generation failed: {e}")
        return None
