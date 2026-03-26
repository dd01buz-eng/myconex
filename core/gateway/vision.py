"""
MYCONEX Vision Layer
---------------------
Describes image attachments using a local Ollama vision model (llava).

The description is injected into the user message context so the main LLM
gets a textual representation of any image the user sends — screenshots,
photos, diagrams, UI mockups, etc.

Env vars:
    VISION_MODEL       — Ollama vision model (default: llava)
    VISION_TIMEOUT_S   — Per-image timeout in seconds (default: 30)
    VISION_ENABLED     — "false" to disable entirely (default: true)
    OLLAMA_URL         — Ollama base URL (default: http://localhost:11434)

Usage:
    from core.gateway.vision import describe_attachments
    descriptions = await describe_attachments(attachment_urls)
    # returns list of (url, description) for each image URL
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

VISION_MODEL   = os.getenv("VISION_MODEL",     "llava")
VISION_TIMEOUT = float(os.getenv("VISION_TIMEOUT_S", "30"))
VISION_ENABLED = os.getenv("VISION_ENABLED",   "true").lower() != "false"
OLLAMA_URL     = os.getenv("OLLAMA_URL",        "http://localhost:11434")

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"})

_DESCRIBE_PROMPT = (
    "Describe this image in detail. Include:\n"
    "- Any visible text, code, terminal output, or error messages (transcribe exactly)\n"
    "- UI elements, layouts, or interface components\n"
    "- Charts, diagrams, or data visualizations\n"
    "- Photos: subjects, context, notable details\n"
    "Be concise but complete. Start directly with the description."
)


def _is_image_url(url: str) -> bool:
    """Return True if the URL looks like an image."""
    path = url.split("?")[0].lower()
    return any(path.endswith(ext) for ext in _IMAGE_EXTS)


async def _fetch_image_b64(url: str, client: httpx.AsyncClient) -> str | None:
    """Download image and return base64-encoded bytes."""
    try:
        resp = await client.get(url, follow_redirects=True, timeout=15.0)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode()
    except Exception as exc:
        logger.warning("[vision] failed to fetch %s: %s", url, exc)
        return None


async def _describe_one(url: str, client: httpx.AsyncClient) -> tuple[str, str | None]:
    """
    Describe a single image URL.
    Returns (url, description_or_None).
    """
    b64 = await _fetch_image_b64(url, client)
    if b64 is None:
        return url, None

    try:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":   VISION_MODEL,
                "prompt":  _DESCRIBE_PROMPT,
                "images":  [b64],
                "stream":  False,
                "options": {"num_predict": 300, "temperature": 0.1},
            },
            timeout=VISION_TIMEOUT,
        )
        resp.raise_for_status()
        description = resp.json().get("response", "").strip()
        if description:
            logger.info("[vision] described %s (%d chars)", url.split("/")[-1][:40], len(description))
            return url, description
        return url, None
    except Exception as exc:
        logger.warning("[vision] llava call failed for %s: %s", url, exc)
        return url, None


async def describe_attachments(
    urls: list[str],
    max_images: int = 3,
) -> list[tuple[str, str]]:
    """
    Describe all image attachments in parallel.

    Args:
        urls:       List of attachment URLs (may include non-images — they're filtered).
        max_images: Max images to describe per message (default 3).

    Returns:
        List of (url, description) tuples for each successfully described image.
        Non-images and failed descriptions are omitted.
    """
    if not VISION_ENABLED or not urls:
        return []

    image_urls = [u for u in urls if _is_image_url(u)][:max_images]
    if not image_urls:
        return []

    async with httpx.AsyncClient() as client:
        tasks = [_describe_one(url, client) for url in image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    described = []
    for item in results:
        if isinstance(item, Exception):
            logger.debug("[vision] task error: %s", item)
            continue
        url, desc = item
        if desc:
            described.append((url, desc))

    return described


def format_vision_context(descriptions: list[tuple[str, str]]) -> str:
    """
    Format vision descriptions for injection into the user message.
    Returns empty string if no descriptions.
    """
    if not descriptions:
        return ""
    parts = ["[Image analysis — the following image(s) were attached:]"]
    for i, (url, desc) in enumerate(descriptions, 1):
        filename = url.split("/")[-1].split("?")[0][:50]
        parts.append(f"\n**Image {i}** ({filename}):\n{desc}")
    return "\n".join(parts)
