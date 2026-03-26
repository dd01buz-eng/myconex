"""
MYCONEX Memory Consolidator
-----------------------------
Periodic background task that keeps the Qdrant knowledge base lean:

  1. TTL pruning  — deletes entries older than configured age per memory_type:
       context    → 30 days  (short-lived conversation context)
       general    → 60 days  (intermediate working memory)
       knowledge  → 90 days  (long-term; episodic kept indefinitely)

  2. Near-dup consolidation — removes semantically redundant entries in the
     knowledge collection using vector cosine similarity ≥ 0.92.
     Run weekly (cheaper than TTL pass, but meaningful for big inboxes).

Env vars:
  CONSOLIDATOR_INTERVAL_HOURS  — how often to run (default 24)
  CONSOLIDATOR_ENABLED         — "false" to disable (default true)
  CONSOLIDATION_THRESHOLD      — cosine threshold for dup detection (default 0.92)

Usage (wired in buzlock_bot.py):
    from core.memory.consolidator import MemoryConsolidator
    consolidator = MemoryConsolidator()
    asyncio.create_task(consolidator.run_forever())
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CONSOLIDATOR_INTERVAL = int(os.getenv("CONSOLIDATOR_INTERVAL_HOURS", "24")) * 3600
CONSOLIDATOR_ENABLED  = os.getenv("CONSOLIDATOR_ENABLED", "true").lower() != "false"
CONSOL_THRESHOLD      = float(os.getenv("CONSOLIDATION_THRESHOLD", "0.92"))

# TTL per memory_type in days
_TTL: dict[str, int] = {
    "context":   int(os.getenv("TTL_CONTEXT_DAYS",   "30")),
    "general":   int(os.getenv("TTL_GENERAL_DAYS",   "60")),
    "knowledge": int(os.getenv("TTL_KNOWLEDGE_DAYS", "90")),
}

_STAMP_FILE = Path.home() / ".myconex" / "last_consolidation.txt"


class MemoryConsolidator:
    """
    Periodic maintenance pass over the Qdrant knowledge base.

    Lifecycle:
        c = MemoryConsolidator()
        asyncio.create_task(c.run_forever())
        c.stop()
    """

    def __init__(self) -> None:
        self._running = False

    async def run_forever(self) -> None:
        if not CONSOLIDATOR_ENABLED:
            logger.info("[consolidator] disabled via CONSOLIDATOR_ENABLED=false")
            return

        self._running = True
        logger.info(
            "[consolidator] started — interval=%dh TTLs=%s",
            CONSOLIDATOR_INTERVAL // 3600, _TTL,
        )

        # Stagger first run: 2 h after startup so ingesters finish their first pass
        await asyncio.sleep(7200)

        while self._running:
            try:
                await self._run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[consolidator] pass failed: %s", exc)
            await asyncio.sleep(CONSOLIDATOR_INTERVAL)

    def stop(self) -> None:
        self._running = False

    # ── Main pass ─────────────────────────────────────────────────────────────

    async def run_once(self) -> dict:
        """Public one-shot entry point (also used by /consolidate slash command)."""
        return await self._run_once()

    async def _run_once(self) -> dict:
        start = time.time()
        stats: dict = {"ttl_removed": 0, "dupes_removed": 0, "elapsed_s": 0.0}

        store = await self._get_store()
        if store is None:
            logger.warning("[consolidator] Qdrant not available — skipping")
            return stats

        # 1. TTL pruning per memory_type
        for mtype, days in _TTL.items():
            try:
                n = await store.cleanup_old_memories(
                    max_age_days=days,
                    memory_types=[mtype],
                )
                stats["ttl_removed"] += n
                logger.info("[consolidator] TTL %s (%dd): removed %d entries", mtype, days, n)
            except Exception as exc:
                logger.warning("[consolidator] TTL pass for %s failed: %s", mtype, exc)

        # 2. Near-dup consolidation (run weekly — check stamp)
        if self._consolidation_due():
            try:
                n = await store.consolidate_memories(
                    agent_name="buzlock",
                    similarity_threshold=CONSOL_THRESHOLD,
                )
                stats["dupes_removed"] = n
                logger.info("[consolidator] dedup: removed %d near-duplicate entries", n)
                _STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
                _STAMP_FILE.write_text(str(time.time()))
            except Exception as exc:
                logger.warning("[consolidator] dedup pass failed: %s", exc)

        stats["elapsed_s"] = round(time.time() - start, 2)
        logger.info(
            "[consolidator] pass complete — ttl=%d dupes=%d elapsed=%.1fs",
            stats["ttl_removed"], stats["dupes_removed"], stats["elapsed_s"],
        )
        return stats

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    async def _get_store():
        """Return the shared VectorStore, or None if Qdrant is unavailable."""
        try:
            from integrations.knowledge_store import _init, _store
            if await _init():
                return _store
        except Exception as exc:
            logger.debug("[consolidator] could not get store: %s", exc)
        return None

    @staticmethod
    def _consolidation_due() -> bool:
        """Return True if at least 7 days have passed since last dedup run."""
        try:
            if _STAMP_FILE.exists():
                last = float(_STAMP_FILE.read_text().strip())
                return (time.time() - last) >= 7 * 86400
        except Exception:
            pass
        return True  # no stamp → never run → due now
