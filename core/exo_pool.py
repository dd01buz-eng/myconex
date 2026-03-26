"""
MYCONEX Exo Pool Manager
--------------------------
On-demand distributed inference via exo (https://github.com/exo-explore/exo).

Exo pools multiple machines' GPUs to run models too large for a single node.
This module manages the lifecycle of that pool within MYCONEX:

  • Complexity gate: only tasks scoring >= EXO_COMPLEXITY_THRESHOLD (default 0.85)
    are even considered for exo routing.

  • Consensus gate: before using the exo cluster, a quorum of mesh nodes must
    vote that they can contribute.  Votes are collected over NATS
    (mesh.exo.consensus_request) and cached for EXO_CONSENSUS_TTL seconds so
    the message hot-path sees an instant is_ready() bool — no blocking.

  • Health gate: the local exo process must be reachable at EXO_BASE_URL.

The consensus loop runs as a background task in buzlock_bot.py.

Env vars:
  EXO_BASE_URL               — exo OpenAI-compatible endpoint (default: http://localhost:52415/v1)
  EXO_MODEL                  — model served by the exo cluster (default: llama3.1:70b)
  EXO_COMPLEXITY_THRESHOLD   — min complexity to consider exo (default: 0.85)
  EXO_MIN_CONSENSUS_NODES    — quorum size (default: 2)
  EXO_CONSENSUS_TTL_S        — seconds to cache a passing consensus (default: 120)
  EXO_CONSENSUS_INTERVAL_S   — background poll interval (default: 60)
  EXO_ENABLED                — "false" to disable entirely

Usage (wired in buzlock_bot.py):
    from core.exo_pool import ExoPoolManager
    exo = ExoPoolManager()
    asyncio.create_task(exo.run_consensus_loop())

Usage (in MoE routing):
    from core.exo_pool import get_pool
    pool = get_pool()
    if pool.is_ready(complexity):
        base_url, model = pool.endpoint()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

EXO_BASE_URL          = os.getenv("EXO_BASE_URL",           "http://localhost:52415/v1")
EXO_MODEL             = os.getenv("EXO_MODEL",              "llama3.1:70b")
EXO_COMPLEXITY_THRESH = float(os.getenv("EXO_COMPLEXITY_THRESHOLD", "0.85"))
EXO_MIN_NODES         = int(os.getenv("EXO_MIN_CONSENSUS_NODES",    "2"))
EXO_CONSENSUS_TTL     = float(os.getenv("EXO_CONSENSUS_TTL_S",      "120"))
EXO_POLL_INTERVAL     = float(os.getenv("EXO_CONSENSUS_INTERVAL_S", "60"))
EXO_ENABLED           = os.getenv("EXO_ENABLED", "true").lower() != "false"

# NATS subjects
_SUBJECT_REQUEST  = "mesh.exo.consensus_request"
_SUBJECT_RESPONSE = "mesh.exo.consensus_response"


class ExoPoolManager:
    """
    Background consensus + health manager for the exo cluster.

    State machine:
      UNKNOWN  → first poll pending
      READY    → health check passed + consensus quorum met
      NOT_READY → health failed OR quorum not met

    is_ready(complexity) is O(1) — reads cached state only.
    """

    def __init__(self) -> None:
        self._ready = False
        self._last_consensus_ts: float = 0.0
        self._contributing_nodes: list[str] = []
        self._running = False

    # ── Public API ────────────────────────────────────────────────────────────

    def is_ready(self, complexity: float = 1.0) -> bool:
        """
        Return True iff exo is enabled, complexity exceeds threshold,
        last consensus passed, and the TTL hasn't expired.
        """
        if not EXO_ENABLED:
            return False
        if complexity < EXO_COMPLEXITY_THRESH:
            return False
        if not self._ready:
            return False
        if (time.time() - self._last_consensus_ts) > EXO_CONSENSUS_TTL:
            # TTL expired — mark unready until next poll
            self._ready = False
            return False
        return True

    def endpoint(self) -> tuple[str, str]:
        """Return (base_url, model) for the exo cluster."""
        return EXO_BASE_URL, EXO_MODEL

    def status(self) -> dict:
        """Return a status dict for logging / the /status slash command."""
        age = round(time.time() - self._last_consensus_ts, 1) if self._last_consensus_ts else None
        return {
            "enabled":          EXO_ENABLED,
            "ready":            self._ready,
            "base_url":         EXO_BASE_URL,
            "model":            EXO_MODEL,
            "complexity_gate":  EXO_COMPLEXITY_THRESH,
            "quorum_required":  EXO_MIN_NODES,
            "contributing":     self._contributing_nodes,
            "last_consensus_s": age,
        }

    # ── Background consensus loop ─────────────────────────────────────────────

    async def run_consensus_loop(self) -> None:
        """
        Periodic background task.
        1. Health-check local exo endpoint.
        2. Broadcast consensus request over NATS and collect votes.
        3. Update self._ready.

        Designed to run supervised via buzlock_bot._supervise().
        """
        if not EXO_ENABLED:
            logger.info("[exo] disabled via EXO_ENABLED=false")
            return

        self._running = True
        logger.info(
            "[exo] consensus loop started — endpoint=%s model=%s threshold=%.2f quorum=%d",
            EXO_BASE_URL, EXO_MODEL, EXO_COMPLEXITY_THRESH, EXO_MIN_NODES,
        )

        # Initial delay: let NATS handler start up first
        await asyncio.sleep(30)

        while self._running:
            await self._poll()
            await asyncio.sleep(EXO_POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False

    # ── Internal poll ─────────────────────────────────────────────────────────

    async def _poll(self) -> None:
        """Run one health + consensus check cycle."""
        # 1. Health check
        if not await self._health_check():
            if self._ready:
                logger.info("[exo] cluster unreachable — marking not ready")
                self._ready = False
            return

        # 2. Consensus via NATS (optional — works without NATS, just skips vote)
        votes = await self._gather_votes()
        node_names = [v["node"] for v in votes if v.get("can_contribute")]
        self._contributing_nodes = node_names

        if len(node_names) >= EXO_MIN_NODES:
            if not self._ready:
                logger.info(
                    "[exo] consensus REACHED — %d node(s) contributing: %s",
                    len(node_names), node_names,
                )
            self._ready = True
            self._last_consensus_ts = time.time()
        else:
            if self._ready:
                logger.info(
                    "[exo] consensus LOST — only %d/%d node(s) available: %s",
                    len(node_names), EXO_MIN_NODES, node_names,
                )
            self._ready = False

    async def _health_check(self) -> bool:
        """Return True if the exo /models endpoint responds."""
        base = EXO_BASE_URL.rstrip("/v1").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{base}/v1/models")
                return resp.status_code == 200
        except Exception as exc:
            logger.debug("[exo] health check failed: %s", exc)
            return False

    async def _gather_votes(self) -> list[dict]:
        """
        Publish a consensus request on NATS and collect responses.
        Returns empty list if NATS is not available.
        """
        try:
            from core.messaging.nats_client import MeshNATSClient
            nats_url = os.getenv("NATS_URL", "")
            if not nats_url:
                # No NATS — single-node mode: self-vote only if exo is healthy
                return [{"node": os.getenv("NATS_NODE_NAME", "buzlock"), "can_contribute": True}]

            node_name = os.getenv("NATS_NODE_NAME", "buzlock")
            client = MeshNATSClient(node_name=f"{node_name}-exo-probe", nats_url=nats_url)
            await client.connect()

            votes: list[dict] = []
            vote_lock = asyncio.Lock()

            async def _collect(msg):
                p = msg.payload or {}
                if isinstance(p, dict):
                    async with vote_lock:
                        votes.append(p)

            await client.subscribe(_SUBJECT_RESPONSE, _collect)
            await client.publish(
                _SUBJECT_REQUEST,
                {"type": "exo_consensus", "model": EXO_MODEL, "requester": node_name},
            )

            # Wait up to 5s for votes to arrive
            await asyncio.sleep(5)
            await client.disconnect()
            return votes

        except Exception as exc:
            logger.debug("[exo] NATS consensus gather failed: %s", exc)
            return []


# ── Module-level singleton ────────────────────────────────────────────────────

_pool: Optional[ExoPoolManager] = None


def get_pool() -> ExoPoolManager:
    """Return the module-level singleton ExoPoolManager."""
    global _pool
    if _pool is None:
        _pool = ExoPoolManager()
    return _pool
