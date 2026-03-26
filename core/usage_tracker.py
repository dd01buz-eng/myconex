"""
MYCONEX Usage Tracker
----------------------
Lightweight singleton that records every LLM inference call —
model, prompt tokens, completion tokens, latency, cost estimate —
and persists to ~/.myconex/usage_log.jsonl.

Wired into the gateway's agent run loop so every response is tracked.

Env vars:
  USAGE_TRACK_ENABLED  — "false" to disable (default: true)
  USAGE_COST_PER_1K    — override cost per 1K tokens in USD (default: 0.0 for local)

Dashboard endpoint: GET /api/usage  (added to dashboard/app.py)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASE        = Path.home() / ".myconex"
_USAGE_FILE  = _BASE / "usage_log.jsonl"
_ENABLED     = os.getenv("USAGE_TRACK_ENABLED", "true").lower() != "false"
_COST_PER_1K = float(os.getenv("USAGE_COST_PER_1K", "0.0"))

_lock = threading.Lock()


# ── Token estimation ──────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


# ── Record a call ─────────────────────────────────────────────────────────────

def record(
    model:          str,
    prompt:         str,
    completion:     str,
    latency_s:      float,
    source:         str = "discord",
    cached:         bool = False,
    prompt_tokens:  int | None = None,
    completion_tokens: int | None = None,
) -> None:
    """
    Record a single inference call. Call this after every LLM response.
    Non-blocking — writes to disk synchronously but fast (JSONL append).
    """
    if not _ENABLED:
        return
    pt = prompt_tokens    if prompt_tokens    is not None else _estimate_tokens(prompt)
    ct = completion_tokens if completion_tokens is not None else _estimate_tokens(completion)
    cost = round((pt + ct) / 1000 * _COST_PER_1K, 6)

    entry = {
        "ts":               datetime.now(timezone.utc).isoformat(),
        "model":            model,
        "source":           source,
        "prompt_tokens":    pt,
        "completion_tokens": ct,
        "total_tokens":     pt + ct,
        "latency_s":        round(latency_s, 3),
        "cost_usd":         cost,
        "cached":           cached,
    }
    try:
        _BASE.mkdir(parents=True, exist_ok=True)
        with _lock:
            with open(_USAGE_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.debug("[usage] write error: %s", exc)


# ── Aggregation ───────────────────────────────────────────────────────────────

def get_stats(days: int = 7) -> dict[str, Any]:
    """Return aggregated usage stats for the last *days* days."""
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    total_tokens    = 0
    total_cost      = 0.0
    total_calls     = 0
    cache_hits      = 0
    total_latency   = 0.0
    by_model: dict[str, dict] = {}

    try:
        if not _USAGE_FILE.exists():
            return _empty_stats()
        with _lock:
            lines = _USAGE_FILE.read_text().splitlines()

        for line in lines:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("ts", "") < cutoff:
                continue

            total_calls   += 1
            total_tokens  += e.get("total_tokens", 0)
            total_cost    += e.get("cost_usd", 0.0)
            total_latency += e.get("latency_s", 0.0)
            if e.get("cached"):
                cache_hits += 1

            m = e.get("model", "unknown")
            if m not in by_model:
                by_model[m] = {"calls": 0, "tokens": 0, "cost": 0.0, "latency": 0.0}
            by_model[m]["calls"]   += 1
            by_model[m]["tokens"]  += e.get("total_tokens", 0)
            by_model[m]["cost"]    += e.get("cost_usd", 0.0)
            by_model[m]["latency"] += e.get("latency_s", 0.0)

    except Exception as exc:
        logger.warning("[usage] stats error: %s", exc)
        return _empty_stats()

    avg_lat = round(total_latency / total_calls, 2) if total_calls else 0.0
    cache_rate = round(cache_hits / total_calls * 100) if total_calls else 0

    return {
        "days":          days,
        "total_calls":   total_calls,
        "total_tokens":  total_tokens,
        "total_cost_usd": round(total_cost, 4),
        "avg_latency_s": avg_lat,
        "cache_hit_rate": cache_rate,
        "cache_hits":    cache_hits,
        "by_model":      by_model,
    }


def _empty_stats() -> dict:
    return {
        "days": 7, "total_calls": 0, "total_tokens": 0,
        "total_cost_usd": 0.0, "avg_latency_s": 0.0,
        "cache_hit_rate": 0, "cache_hits": 0, "by_model": {},
    }


# ── Context manager for timing ─────────────────────────────────────────────────

class Track:
    """
    Usage:
        async with Track(model="llama3.1:8b", prompt=user_msg) as t:
            response = await agent.run(...)
        t.record(response)
    """
    def __init__(self, model: str, prompt: str, source: str = "discord", cached: bool = False):
        self.model   = model
        self.prompt  = prompt
        self.source  = source
        self.cached  = cached
        self._start: float = 0.0
        self.latency: float = 0.0

    async def __aenter__(self):
        self._start = time.monotonic()
        return self

    async def __aexit__(self, *_):
        self.latency = time.monotonic() - self._start

    def record(self, completion: str) -> None:
        record(
            model      = self.model,
            prompt     = self.prompt,
            completion = completion,
            latency_s  = self.latency,
            source     = self.source,
            cached     = self.cached,
        )
