"""
MYCONEX Memory Distiller
--------------------------
Builds higher-level abstractions from raw ingested memories.

While the consolidator prunes near-duplicates, the distiller *reads* recent
entries and writes summary notes — compressing N granular observations into
1 structured insight that persists across the TTL horizon.

Weekly run (different stamp from consolidator):
  1. Fetch recent knowledge entries (last 30 days) grouped by source type
  2. For each group, ask the local LLM to extract:
       - Dominant themes (with supporting evidence)
       - Trajectory: is interest growing / stable / fading?
       - Any implicit project ideas not yet in the profile
  3. Store summaries in ~/.myconex/memory_distillations.json
  4. Append high-confidence theme shifts to interest_profile.json

The distilled summaries are injected into the morning briefing and into
the system prompt (via _load_memory_for_prompt extension).

Env vars:
  DISTILLER_ENABLED          — "false" to disable (default: true)
  DISTILLER_INTERVAL_DAYS    — days between full distillation runs (default: 7)
  DISTILLER_OLLAMA_MODEL     — model for distillation LLM calls (default: llama3)
  DISTILLER_MAX_ENTRIES      — max entries per source to feed to LLM (default: 40)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE              = Path.home() / ".myconex"
_PROFILE_FILE      = _BASE / "interest_profile.json"
_DISTILL_FILE      = _BASE / "memory_distillations.json"
_STAMP_FILE        = _BASE / "last_distillation.txt"
_EMAIL_FILE        = _BASE / "email_insights.json"
_YT_FILE           = _BASE / "youtube_insights.json"
_RSS_FILE          = _BASE / "rss_insights.json"
_WISDOM_FILE       = _BASE / "wisdom_store.json"

DISTILLER_ENABLED       = os.getenv("DISTILLER_ENABLED",       "true").lower() != "false"
DISTILLER_INTERVAL_DAYS = int(os.getenv("DISTILLER_INTERVAL_DAYS",  "7"))
DISTILLER_MODEL         = os.getenv("DISTILLER_OLLAMA_MODEL",   "llama3")
DISTILLER_MAX_ENTRIES   = int(os.getenv("DISTILLER_MAX_ENTRIES",    "40"))
OLLAMA_URL              = os.getenv("OLLAMA_URL", "http://localhost:11434")

_DISTILL_PROMPT = """You are analyzing a personal knowledge base.
Below are {count} items ingested from {source} over the past {days} days.

Items:
{items}

Produce a JSON object with these keys:
- "dominant_themes": list of 3-5 strings (most recurring topics/ideas)
- "trajectory": one of "growing", "stable", "fading" (is interest in these themes increasing?)
- "trajectory_reason": 1-sentence explanation
- "implicit_ideas": list of 0-3 project/action ideas implied but not stated explicitly
- "summary": 2-3 sentence narrative of what this person has been focused on

Respond with ONLY valid JSON. No markdown fences."""


def _load(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def _save(path: Path, data: Any) -> None:
    _BASE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _recent_entries(path: Path, days: int, max_n: int) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    entries = _load(path, [])
    recent = [
        e for e in entries
        if (e.get("processed_at") or e.get("stored_at") or "") >= cutoff
    ]
    return recent[-max_n:]


async def _call_llm(prompt: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model":   DISTILLER_MODEL,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": {"num_predict": 600, "temperature": 0.2},
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
    except Exception as exc:
        logger.warning("[distiller] LLM call failed: %s", exc)
        return ""


def _extract_json(text: str) -> dict | None:
    import re
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        # Try to find JSON object in the text
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


async def distill_once() -> dict[str, Any]:
    """Run one distillation pass. Returns summary of what was produced."""
    sources = {
        "email":   (_EMAIL_FILE,  30),
        "rss":     (_RSS_FILE,    30),
        "youtube": (_YT_FILE,     30),
    }

    distillations: dict[str, Any] = _load(_DISTILL_FILE, {})
    produced = 0

    for source, (path, days) in sources.items():
        entries = _recent_entries(path, days, DISTILLER_MAX_ENTRIES)
        if len(entries) < 5:
            logger.debug("[distiller] too few entries for %s (%d) — skipping", source, len(entries))
            continue

        # Build item list: combine subject/title + topics + summary excerpt
        item_lines = []
        for e in entries:
            title = e.get("subject") or e.get("title") or e.get("url", "")
            topics = ", ".join(e.get("topics", [])[:4])
            summary = (
                e.get("fabric_summary") or
                e.get("summary") or
                (e.get("raw", {}) or {}).get("summarize", "")
            )[:120]
            line = f"- {title}"
            if topics:   line += f" [{topics}]"
            if summary:  line += f": {summary}"
            item_lines.append(line)

        prompt = _DISTILL_PROMPT.format(
            count=len(entries),
            source=source,
            days=days,
            items="\n".join(item_lines),
        )

        raw = await _call_llm(prompt)
        result = _extract_json(raw)
        if not result:
            logger.warning("[distiller] could not parse LLM response for %s", source)
            continue

        result["source"]     = source
        result["entry_count"] = len(entries)
        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        distillations[source] = result
        produced += 1

        # Propagate implicit_ideas back to interest profile
        ideas = result.get("implicit_ideas", [])
        if ideas:
            profile = _load(_PROFILE_FILE, {})
            existing = set(profile.get("project_ideas", []))
            new_ideas = [i for i in ideas if i not in existing]
            if new_ideas:
                profile.setdefault("project_ideas", []).extend(new_ideas)
                _save(_PROFILE_FILE, profile)
                logger.info("[distiller] added %d implicit ideas from %s", len(new_ideas), source)

        logger.info("[distiller] distilled %s: themes=%s trajectory=%s",
                    source, result.get("dominant_themes", [])[:3], result.get("trajectory"))

    if produced:
        _save(_DISTILL_FILE, distillations)

    _BASE.mkdir(parents=True, exist_ok=True)
    _STAMP_FILE.write_text(str(time.time()))

    return {"sources_distilled": produced, "distillations": distillations}


def _distillation_due() -> bool:
    try:
        if _STAMP_FILE.exists():
            last = float(_STAMP_FILE.read_text().strip())
            return (time.time() - last) >= DISTILLER_INTERVAL_DAYS * 86400
    except Exception:
        pass
    return True


def get_distillation_context() -> str:
    """
    Return a formatted context block from the latest distillations.
    Injected into the system prompt alongside memory.
    """
    distillations = _load(_DISTILL_FILE, {})
    if not distillations:
        return ""

    lines = ["[Memory distillation — higher-level patterns from your knowledge base:]"]
    for source, d in distillations.items():
        if not isinstance(d, dict):
            continue
        themes = ", ".join(d.get("dominant_themes", [])[:4])
        traj   = d.get("trajectory", "")
        summ   = d.get("summary", "")[:200]
        if themes or summ:
            lines.append(f"\n**{source.title()}** ({traj}): {summ}")
            if themes:
                lines.append(f"  Themes: {themes}")

    return "\n".join(lines) if len(lines) > 1 else ""


class MemoryDistiller:
    """Background weekly distillation task."""

    def __init__(self) -> None:
        self._running = False

    async def run_forever(self) -> None:
        if not DISTILLER_ENABLED:
            logger.info("[distiller] disabled")
            return
        self._running = True
        logger.info("[distiller] started — interval=%dd model=%s", DISTILLER_INTERVAL_DAYS, DISTILLER_MODEL)
        await asyncio.sleep(3600 * 3)  # 3h startup delay
        while self._running:
            if _distillation_due():
                try:
                    result = await distill_once()
                    logger.info("[distiller] pass complete: %s", result)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("[distiller] pass failed: %s", exc)
            await asyncio.sleep(3600 * 6)  # check every 6h

    def stop(self) -> None:
        self._running = False
