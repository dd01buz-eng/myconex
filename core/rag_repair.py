"""
MYCONEX RAG Repair
-------------------
When a response gets a 👎 reaction, this module:
1. Logs the query + response to a repair queue
2. Periodically analyzes downvoted entries to identify knowledge gaps
3. Marks those queries for re-ingestion priority or manual review

Gaps are written to ~/.myconex/rag_gaps.json — a list of prompts that
produced poor results, with metadata about why (no RAG hit, low score, etc.)

This feeds two things:
  a) Priority re-ingestion: the RSS/YouTube ingesters check this list and
     prefer sources that match gap topics
  b) Manual review digest: weekly gap report included in the Friday digest
     so the user knows what the knowledge base is missing

Env vars:
  RAG_REPAIR_ENABLED     — "false" to disable (default: true)
  RAG_REPAIR_MIN_GAPS    — minimum gap entries before triggering analysis (default: 3)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASE          = Path.home() / ".myconex"
_GAPS_FILE     = _BASE / "rag_gaps.json"
_FEEDBACK_FILE = _BASE / "feedback_log.jsonl"

RAG_REPAIR_ENABLED  = os.getenv("RAG_REPAIR_ENABLED", "true").lower() != "false"
RAG_REPAIR_MIN_GAPS = int(os.getenv("RAG_REPAIR_MIN_GAPS", "3"))


def _load(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def _load_jsonl(path: Path) -> list[dict]:
    lines = []
    try:
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return lines


def record_rag_miss(
    query: str,
    response: str,
    rag_hit_count: int,
    max_rag_score: float,
) -> None:
    """
    Called when a response that had poor RAG context gets a 👎.
    Records the gap for analysis.
    """
    if not RAG_REPAIR_ENABLED:
        return
    gaps = _load(_GAPS_FILE, [])
    # Dedup: don't record the same query twice
    existing_queries = {g.get("query", "") for g in gaps}
    if query in existing_queries:
        return
    gaps.append({
        "query":         query,
        "response_preview": response[:200],
        "rag_hit_count": rag_hit_count,
        "max_rag_score": round(max_rag_score, 3),
        "ts":            time.time(),
        "recorded_at":   datetime.now(timezone.utc).isoformat(),
        "status":        "open",
    })
    _BASE.mkdir(parents=True, exist_ok=True)
    _GAPS_FILE.write_text(json.dumps(gaps[-200:], indent=2, ensure_ascii=False))
    logger.info("[rag-repair] gap recorded: %r (rag_hits=%d score=%.2f)",
                query[:60], rag_hit_count, max_rag_score)


def get_open_gaps(max_n: int = 20) -> list[dict]:
    """Return open (unresolved) knowledge gaps."""
    gaps = _load(_GAPS_FILE, [])
    return [g for g in gaps if g.get("status") == "open"][-max_n:]


def mark_gap_resolved(query: str) -> None:
    """Mark a gap as resolved (e.g., after the user manually adds content)."""
    gaps = _load(_GAPS_FILE, [])
    for g in gaps:
        if g.get("query") == query:
            g["status"] = "resolved"
            g["resolved_at"] = datetime.now(timezone.utc).isoformat()
    _GAPS_FILE.write_text(json.dumps(gaps, indent=2, ensure_ascii=False))


def get_gap_topics() -> list[str]:
    """
    Return a list of topic keywords from open gaps.
    Used by ingesters to bias content selection toward gap areas.
    """
    gaps = get_open_gaps()
    topics: list[str] = []
    for g in gaps:
        q = g.get("query", "")
        # Extract meaningful words (>4 chars, not common words)
        _STOP = {"what", "when", "where", "which", "about", "there", "their", "would",
                 "could", "should", "please", "tell", "show", "give", "make", "help"}
        words = [w.lower() for w in q.split() if len(w) > 4 and w.lower() not in _STOP]
        topics.extend(words[:3])
    return list(dict.fromkeys(topics))[:15]  # dedup, keep order


def get_gaps_summary() -> str:
    """
    Return a formatted summary of open gaps for the weekly digest / briefing.
    """
    gaps = get_open_gaps(max_n=8)
    if not gaps:
        return ""
    lines = [f"⚠️ **{len(gaps)} knowledge gap(s)** — queries that lacked good RAG context:"]
    for g in gaps[:5]:
        score_str = f"(best match: {g['max_rag_score']})" if g.get("max_rag_score") else ""
        lines.append(f"• {g['query'][:80]} {score_str}")
    if len(gaps) > 5:
        lines.append(f"  _...and {len(gaps)-5} more. Use /gaps to review._")
    return "\n".join(lines)
