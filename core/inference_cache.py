"""
MYCONEX Semantic Inference Cache
----------------------------------
Two-level cache for LLM responses:

  L1 — Exact cache:  SHA-256(model + normalised prompt) → response
       ~0ms lookup, persisted to ~/.myconex/inference_cache.json

  L2 — Semantic cache: cosine similarity between query embedding and stored
       embeddings.  Threshold default 0.97 — only near-identical phrasings hit.
       Embeddings computed on first put, kept in memory + JSON.

Cache is only populated/queried for "safe" requests:
  • No active conversation history (first turn only)
  • No attachment URLs
  • Prompt ≤ MAX_CACHE_PROMPT_LEN chars
  • Response contained no tool calls

TTL = CACHE_TTL_HOURS hours.  Max entries = CACHE_MAX_ENTRIES (LRU eviction).

Env vars:
  INFERENCE_CACHE_TTL_HOURS   — default 24
  INFERENCE_CACHE_MAX_ENTRIES — default 500
  INFERENCE_CACHE_SIM_THRESH  — default 0.97 (0-1)
  INFERENCE_CACHE_ENABLED     — "false" to disable entirely
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASE            = Path.home() / ".myconex"
_CACHE_FILE      = _BASE / "inference_cache.json"

CACHE_TTL_HOURS   = float(os.getenv("INFERENCE_CACHE_TTL_HOURS",   "24"))
CACHE_MAX_ENTRIES = int(os.getenv("INFERENCE_CACHE_MAX_ENTRIES",   "500"))
CACHE_SIM_THRESH  = float(os.getenv("INFERENCE_CACHE_SIM_THRESH",  "0.97"))
CACHE_ENABLED     = os.getenv("INFERENCE_CACHE_ENABLED", "true").lower() != "false"
MAX_PROMPT_LEN    = 400  # chars; longer prompts are too complex/specific to cache

# ── In-memory store ───────────────────────────────────────────────────────────
# Each entry: {key, prompt, model, response, embedding, ts}
_cache: dict[str, dict[str, Any]] = {}   # key → entry
_loaded = False
_lock   = asyncio.Lock()


# ── Public API ────────────────────────────────────────────────────────────────

def is_cacheable(prompt: str, history: list, attachment_urls: list) -> bool:
    """Return True if this request is eligible for caching."""
    if not CACHE_ENABLED:
        return False
    if history:
        return False  # mid-conversation — context makes every turn unique
    if attachment_urls:
        return False
    if len(prompt) > MAX_PROMPT_LEN:
        return False
    if re.search(r"https?://", prompt):
        return False  # URL-containing queries often retrieve live content
    return True


def has_tool_calls(result_messages: list[dict]) -> bool:
    """Return True if any message in the result had a tool_call role."""
    for msg in result_messages:
        role = msg.get("role", "")
        if role == "tool" or (role == "assistant" and msg.get("tool_calls")):
            return True
    return False


async def get(prompt: str, model: str) -> str | None:
    """
    Check L1 (exact) then L2 (semantic) cache.
    Returns cached response string, or None on miss.
    """
    if not CACHE_ENABLED:
        return None
    await _ensure_loaded()

    now = time.time()
    ttl_secs = CACHE_TTL_HOURS * 3600

    # L1 — exact
    key = _make_key(prompt, model)
    entry = _cache.get(key)
    if entry and (now - entry["ts"]) < ttl_secs:
        logger.debug("[cache] L1 hit — key=%s", key[:12])
        entry["ts"] = now  # LRU refresh
        return entry["response"]

    # L2 — semantic
    query_emb = await _embed(prompt)
    if query_emb is None:
        return None

    best_score = 0.0
    best_entry = None
    for e in _cache.values():
        if e.get("model") != model:
            continue
        if (now - e["ts"]) >= ttl_secs:
            continue
        emb = e.get("embedding")
        if not emb:
            continue
        score = _cosine(query_emb, emb)
        if score > best_score:
            best_score = score
            best_entry = e

    if best_entry and best_score >= CACHE_SIM_THRESH:
        logger.info(
            "[cache] L2 hit — score=%.4f model=%s prompt=%r",
            best_score, model, prompt[:60],
        )
        best_entry["ts"] = now
        return best_entry["response"]

    return None


async def put(prompt: str, model: str, response: str) -> None:
    """Store a response in both L1 and L2 cache."""
    if not CACHE_ENABLED:
        return
    await _ensure_loaded()

    embedding = await _embed(prompt)
    key = _make_key(prompt, model)
    _cache[key] = {
        "key":       key,
        "prompt":    prompt,
        "model":     model,
        "response":  response,
        "embedding": embedding,
        "ts":        time.time(),
    }

    _evict()
    await _persist()
    logger.debug("[cache] stored — key=%s model=%s", key[:12], model)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_key(prompt: str, model: str) -> str:
    normalised = re.sub(r"\s+", " ", prompt.lower().strip())
    raw = f"{model}||{normalised}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _evict() -> None:
    """Remove oldest entries when over capacity."""
    if len(_cache) <= CACHE_MAX_ENTRIES:
        return
    # Sort by last-access timestamp, drop oldest
    sorted_keys = sorted(_cache, key=lambda k: _cache[k]["ts"])
    for k in sorted_keys[: len(_cache) - CACHE_MAX_ENTRIES]:
        del _cache[k]


async def _embed(text: str) -> list[float] | None:
    """Embed text using the knowledge store's EmbeddingClient."""
    try:
        from integrations.knowledge_store import _init, _embedder
        if not await _init():
            return None
        return await _embedder.generate_embedding(text[:512])
    except Exception as exc:
        logger.debug("[cache] embed failed: %s", exc)
        return None


async def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    async with _lock:
        if _loaded:
            return
        _load_from_disk()
        _loaded = True


def _load_from_disk() -> None:
    try:
        if _CACHE_FILE.exists():
            data = json.loads(_CACHE_FILE.read_text())
            now = time.time()
            ttl_secs = CACHE_TTL_HOURS * 3600
            valid = {k: v for k, v in data.items() if (now - v.get("ts", 0)) < ttl_secs}
            _cache.update(valid)
            logger.info(
                "[cache] loaded %d/%d valid entries from disk",
                len(valid), len(data),
            )
    except Exception as exc:
        logger.debug("[cache] could not load from disk: %s", exc)


async def _persist() -> None:
    try:
        _BASE.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            _CACHE_FILE.write_text,
            json.dumps(_cache, ensure_ascii=False),
        )
    except Exception as exc:
        logger.debug("[cache] persist failed: %s", exc)
