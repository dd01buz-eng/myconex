"""
MYCONEX Knowledge Store
------------------------
Thin async wrapper around VectorStore + EmbeddingClient that provides a
single entry point for embedding and querying the unified knowledge base.

All email wisdom, YouTube wisdom, and RSS articles flow through here so
Buzlock can do semantic search across everything it has ever processed.

Usage:
    from integrations.knowledge_store import embed_and_store, search

    await embed_and_store(
        text="...",
        source="email",
        metadata={"subject": "...", "from": "..."},
    )

    results = await search("distributed systems project ideas", limit=5)
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_QDRANT_URL  = os.getenv("QDRANT_URL",  "http://localhost:6333")
_OLLAMA_URL  = os.getenv("OLLAMA_URL",  "http://localhost:11434")
_EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
_COLLECTION  = os.getenv("QDRANT_COLLECTION", "myconex-knowledge")

# Module-level singletons — initialised lazily on first use
_store   = None
_embedder = None
_lock    = asyncio.Lock()
_ready   = False


async def _init() -> bool:
    """Initialise VectorStore and EmbeddingClient if not already done."""
    global _store, _embedder, _ready
    if _ready:
        return True
    async with _lock:
        if _ready:
            return True
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from core.memory.vector_store import VectorStore, EmbeddingClient

            _store   = VectorStore(url=_QDRANT_URL, collection_name=_COLLECTION)
            _embedder = EmbeddingClient(ollama_url=_OLLAMA_URL, model=_EMBED_MODEL)
            await _store.connect()
            _ready = True
            logger.info("[knowledge_store] connected — Qdrant=%s model=%s", _QDRANT_URL, _EMBED_MODEL)
            return True
        except Exception as exc:
            logger.warning("[knowledge_store] init failed (Qdrant/Ollama may not be running): %s", exc)
            return False


async def embed_and_store(
    text: str,
    source: str = "general",
    metadata: dict[str, Any] | None = None,
    memory_type: str = "knowledge",
) -> str | None:
    """
    Embed text and store in Qdrant.

    Args:
        text:        The content to embed and store.
        source:      Origin label — "email", "youtube", "rss", "podcast", "manual".
        metadata:    Extra fields stored alongside the vector.
        memory_type: Qdrant memory_type tag.

    Returns:
        The stored entry ID, or None if unavailable.
    """
    if not text or not text.strip():
        return None
    if not await _init():
        return None
    try:
        from core.memory.vector_store import MemoryEntry
        embedding = await _embedder.generate_embedding(text[:2000])  # cap to avoid OOM
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=text[:4000],
            embedding=embedding,
            metadata={**(metadata or {}), "source": source},
            agent_name="buzlock",
            memory_type=memory_type,
        )
        return await _store.store_memory(entry)
    except Exception as exc:
        logger.warning("[knowledge_store] embed_and_store failed: %s", exc)
        return None


async def embed_and_store_batch(
    items: list[dict[str, Any]],
) -> list[str | None]:
    """
    Embed and store multiple items in a single Ollama batch call.

    Each item dict must have:
        text        (str)  — content to embed
        source      (str)  — "email", "rss", "youtube", "podcast", "manual"
        metadata    (dict) — extra fields (optional)
        memory_type (str)  — Qdrant tag (optional, default "knowledge")

    Returns list of stored IDs (or None for each failed item).
    """
    if not items:
        return []
    if not await _init():
        return [None] * len(items)

    from core.memory.vector_store import MemoryEntry

    # Filter out empty texts but track original positions
    valid: list[tuple[int, dict[str, Any]]] = [
        (i, it) for i, it in enumerate(items) if it.get("text", "").strip()
    ]
    if not valid:
        return [None] * len(items)

    texts = [it["text"][:2000] for _, it in valid]
    try:
        embeddings = await _embedder.generate_embeddings(texts)
    except Exception as exc:
        logger.warning("[knowledge_store] batch embed failed: %s", exc)
        return [None] * len(items)

    results: list[str | None] = [None] * len(items)
    store_tasks = []
    for (orig_idx, it), embedding in zip(valid, embeddings):
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=it["text"][:4000],
            embedding=embedding,
            metadata={**(it.get("metadata") or {}), "source": it["source"]},
            agent_name="buzlock",
            memory_type=it.get("memory_type", "knowledge"),
        )
        store_tasks.append((orig_idx, _store.store_memory(entry)))

    for orig_idx, coro in store_tasks:
        try:
            results[orig_idx] = await coro
        except Exception as exc:
            logger.warning("[knowledge_store] store failed for item %d: %s", orig_idx, exc)

    logger.info(
        "[knowledge_store] batch stored %d/%d items",
        sum(1 for r in results if r is not None), len(items),
    )
    return results


async def search(
    query: str,
    limit: int = 8,
    source_filter: str | None = None,
    score_threshold: float = 0.35,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search over the knowledge base.

    Args:
        query:           Natural-language query (embedded unless query_embedding given).
        limit:           Max results to return.
        source_filter:   Filter by source ("email", "youtube", "rss", etc.).
        score_threshold: Minimum similarity (0-1).
        query_embedding: Pre-computed embedding vector; skips internal embed step.
                         Used by HyDE to pass a hypothetical-document embedding.

    Returns:
        List of dicts with keys: content, score, source, metadata.
    """
    if not await _init():
        return []
    try:
        embedding = query_embedding if query_embedding is not None else await _embedder.generate_embedding(query)
        filters = {"source": source_filter} if source_filter else None
        results = await _store.search_similar(
            query_embedding=embedding,
            limit=limit,
            score_threshold=score_threshold,
            filters=filters,
        )
        return [
            {
                "content":  r.entry.content,
                "score":    round(r.score, 3),
                "source":   r.entry.metadata.get("source", ""),
                "metadata": r.entry.metadata,
            }
            for r in results
        ]
    except Exception as exc:
        logger.warning("[knowledge_store] search failed: %s", exc)
        return []


async def get_stats() -> dict[str, Any]:
    """Return basic stats about the knowledge base."""
    if not await _init():
        return {"available": False}
    try:
        stats = await _store.get_stats()
        stats["available"] = True
        return stats
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def format_results(results: list[dict[str, Any]], query: str = "") -> str:
    """Format search results as readable text for an LLM."""
    if not results:
        return f"No knowledge base results found{f' for: {query!r}' if query else ''}."
    lines = [f"Knowledge base results ({len(results)}) for: {query!r}\n"]
    for i, r in enumerate(results, 1):
        src  = r.get("source", "")
        meta = r.get("metadata", {})
        score = r.get("score", 0)
        label = meta.get("subject") or meta.get("title") or meta.get("url") or src
        lines.append(f"[{i}] {label}  (source={src}, relevance={score})")
        lines.append(r.get("content", "")[:600])
        lines.append("")
    return "\n".join(lines)
