"""
MYCONEX NATS Remote Handler
----------------------------
Listens for mesh tasks on NATS and routes them through the local Ollama LLM.
Any peer node on the mesh can submit a task to `mesh.task.submit`; this node
picks it up (queue-group "workers" so only one node handles each task), runs
it through Ollama, and publishes the result to `mesh.task.result.<task_id>`.

Also subscribes to `mesh.node.<node_name>` for direct-addressed messages and
`mesh.broadcast` for network-wide announcements.

Env vars:
    NATS_URL         — nats://host:port  (default: nats://localhost:4222)
    NATS_NODE_NAME   — local identity    (default: buzlock)
    OLLAMA_URL       — http://host:port  (default: http://localhost:11434)
    NATS_LLM_MODEL   — model to use      (default: llama3)

Usage (wired in buzlock_bot.py):
    from core.messaging.remote_handler import NATSRemoteHandler
    handler = NATSRemoteHandler()
    asyncio.create_task(handler.run_forever())
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from core.messaging.nats_client import (
    MeshMessage,
    MeshNATSClient,
    SUBJECT_BROADCAST,
    subject_task_result,
)

logger = logging.getLogger(__name__)

NATS_URL       = os.getenv("NATS_URL",       "nats://localhost:4222")
NATS_NODE_NAME = os.getenv("NATS_NODE_NAME", "buzlock")
OLLAMA_URL     = os.getenv("OLLAMA_URL",     "http://localhost:11434")
NATS_LLM_MODEL = os.getenv("NATS_LLM_MODEL", "llama3")

_SYSTEM_PROMPT = (
    "You are Buzlock, an AI assistant node in the MYCONEX distributed mesh. "
    "Answer the task concisely and accurately."
)


class NATSRemoteHandler:
    """
    Subscribes to NATS task subjects and handles remote tasks via Ollama.

    Lifecycle:
        handler = NATSRemoteHandler()
        asyncio.create_task(handler.run_forever())   # supervised in buzlock_bot
        handler.stop()                               # on shutdown
    """

    def __init__(
        self,
        node_name: str = NATS_NODE_NAME,
        nats_url: str = NATS_URL,
        ollama_url: str = OLLAMA_URL,
        model: str = NATS_LLM_MODEL,
    ) -> None:
        self.node_name = node_name
        self.nats_url  = nats_url
        self.ollama_url = ollama_url
        self.model     = model
        self._client   = MeshNATSClient(node_name=node_name, nats_url=nats_url)
        self._http     = httpx.AsyncClient(timeout=120.0)
        self._running  = False
        self._tasks_handled = 0

    # ── Public API ────────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        """Connect, subscribe, and handle tasks until stopped."""
        self._running = True
        logger.info("[nats-handler] starting — node=%s url=%s", self.node_name, self.nats_url)

        await self._client.connect()

        # Queue-group subscription: only one worker node handles each task
        await self._client.subscribe_tasks(self._on_task, queue="workers")

        # Direct-addressed messages (node-to-node)
        await self._client.subscribe_self(self._on_direct)

        # Broadcast (info / roster updates)
        await self._client.subscribe_broadcast(self._on_broadcast)

        # Exo consensus requests from peer nodes
        await self._client.subscribe("mesh.exo.consensus_request", self._on_exo_consensus)

        logger.info("[nats-handler] subscribed — listening for mesh tasks")

        # Keep alive until stop() is called
        try:
            while self._running:
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()

    def stop(self) -> None:
        self._running = False

    # ── Message handlers ──────────────────────────────────────────────────────

    async def _on_task(self, msg: MeshMessage) -> None:
        """Handle a task from mesh.task.submit."""
        payload = msg.payload or {}
        if not isinstance(payload, dict):
            payload = {"prompt": str(payload)}

        task_id = payload.get("task_id") or payload.get("id") or msg.msg_id
        prompt  = payload.get("prompt") or payload.get("content") or payload.get("text", "")

        if not prompt:
            logger.warning("[nats-handler] task %s has no prompt — skipping", task_id)
            return

        logger.info(
            "[nats-handler] task %s from %s: %r",
            task_id, msg.sender or "unknown", prompt[:80],
        )

        result = await self._run_llm(prompt)
        self._tasks_handled += 1

        response_payload: dict[str, Any] = {
            "task_id": task_id,
            "result":  result,
            "node":    self.node_name,
            "model":   self.model,
        }

        # Reply on the NATS reply-to inbox if provided, otherwise publish to result subject
        if msg.reply_to:
            try:
                resp = MeshMessage(
                    subject=msg.reply_to,
                    payload=response_payload,
                    sender=self.node_name,
                )
                await self._client._nc.publish(msg.reply_to, resp.encode())
                logger.debug("[nats-handler] replied on reply_to=%s", msg.reply_to)
            except Exception as exc:
                logger.warning("[nats-handler] reply failed: %s", exc)
        else:
            try:
                await self._client.publish(subject_task_result(task_id), response_payload)
                logger.debug("[nats-handler] result published → %s", subject_task_result(task_id))
            except Exception as exc:
                logger.warning("[nats-handler] result publish failed: %s", exc)

    async def _on_direct(self, msg: MeshMessage) -> None:
        """Handle a direct-addressed message (mesh.node.<name>)."""
        payload = msg.payload or {}
        kind = payload.get("type", "") if isinstance(payload, dict) else ""

        if kind == "ping":
            # Respond to mesh ping probes
            if msg.reply_to:
                pong = MeshMessage(
                    subject=msg.reply_to,
                    payload={"type": "pong", "node": self.node_name, "tasks_handled": self._tasks_handled},
                    sender=self.node_name,
                )
                await self._client._nc.publish(msg.reply_to, pong.encode())
            logger.debug("[nats-handler] ping from %s — pong sent", msg.sender)

        elif kind == "task":
            # Direct task (same handling as queue task)
            await self._on_task(msg)

        else:
            logger.debug(
                "[nats-handler] direct msg from %s: type=%r payload=%r",
                msg.sender, kind, str(payload)[:120],
            )

    async def _on_broadcast(self, msg: MeshMessage) -> None:
        """Handle broadcast messages — log roster updates, ignore others."""
        payload = msg.payload or {}
        if isinstance(payload, dict) and payload.get("type") == "roster":
            logger.info("[nats-handler] roster update from %s: %s", msg.sender, payload)

    async def _on_exo_consensus(self, msg: MeshMessage) -> None:
        """
        Respond to exo consensus requests from peer nodes.

        Votes True if this node has a GPU with meaningful free VRAM and is
        not currently saturated.  The vote is published to
        mesh.exo.consensus_response so the requester's _gather_votes() sees it.
        """
        import shutil

        can_contribute = False
        vram_free_mb = 0

        # Check if exo binary is accessible on this node
        if shutil.which("exo") or shutil.which("python3"):
            try:
                import subprocess, json as _json
                # Quick nvidia-smi check (works on Linux/Windows)
                proc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3,
                )
                if proc.returncode == 0:
                    vram_free_mb = max(
                        (int(x.strip()) for x in proc.stdout.strip().splitlines() if x.strip()),
                        default=0,
                    )
                    # Require at least 4 GB free to contribute
                    can_contribute = vram_free_mb >= 4096
            except Exception:
                # If nvidia-smi unavailable assume Apple Silicon (always capable)
                import platform
                can_contribute = platform.system() == "Darwin"

        vote = {
            "node":          self.node_name,
            "can_contribute": can_contribute,
            "vram_free_mb":  vram_free_mb,
        }
        try:
            await self._client.publish("mesh.exo.consensus_response", vote)
            logger.debug(
                "[nats-handler] exo vote: can_contribute=%s vram_free=%dMB",
                can_contribute, vram_free_mb,
            )
        except Exception as exc:
            logger.debug("[nats-handler] exo vote publish failed: %s", exc)

    # ── LLM call ──────────────────────────────────────────────────────────────

    async def _run_llm(self, prompt: str) -> str:
        """Run prompt through Ollama and return the response text."""
        try:
            response = await self._http.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model":  self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system",  "content": _SYSTEM_PROMPT},
                        {"role": "user",    "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "").strip() or "[empty response]"
        except Exception as exc:
            logger.error("[nats-handler] Ollama call failed: %s", exc)
            return f"[error: {exc}]"

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def _shutdown(self) -> None:
        try:
            await self._client.disconnect()
        except Exception:
            pass
        try:
            await self._http.aclose()
        except Exception:
            pass
        logger.info(
            "[nats-handler] shut down — %d task(s) handled this session",
            self._tasks_handled,
        )
