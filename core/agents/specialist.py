"""
MYCONEX Specialist Agents
--------------------------
Thin wrappers that configure a NATSRemoteHandler with a specific persona,
system prompt, and model for a given role.

Specialist nodes subscribe to their own tier queue as well as the general
"workers" queue. Tasks can be sent directly to a specialist via:
    mesh.task.submit  → queue group "workers"     (any available node)
    mesh.tier.T1      → queue group "t1-workers"  (researcher/analyst nodes)
    mesh.tier.T2      → queue group "t2-workers"  (coder/executor nodes)
    mesh.node.<name>  → direct-addressed (always goes to the named node)

Roles:
  researcher  — deep analysis, synthesis, summarisation of complex topics
  coder       — code generation, debugging, explanation; uses a code-tuned model
  summarizer  — document summarisation; lighter model, high throughput

Env vars (role-specific):
  SPECIALIST_ROLE        — "researcher", "coder", "summarizer", or "" (default general)
  RESEARCHER_MODEL       — default: llama3
  CODER_MODEL            — default: codellama or deepseek-coder
  SUMMARIZER_MODEL       — default: llama3

Usage:
    from core.agents.specialist import build_specialist_handler
    handler = build_specialist_handler()     # reads SPECIALIST_ROLE from env
    asyncio.create_task(handler.run_forever())
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_ROLE_CONFIGS: dict[str, dict] = {
    "researcher": {
        "model_env":    "RESEARCHER_MODEL",
        "model_default": "llama3",
        "tier":          "T1",
        "queue_suffix":  "t1-workers",
        "system_prompt": (
            "You are a specialist research node in the MYCONEX distributed mesh. "
            "Your role is deep analysis, synthesis, and summarisation of complex topics. "
            "When given a question or topic, provide thorough, well-structured analysis "
            "with citations to evidence where possible. Be comprehensive and precise."
        ),
    },
    "coder": {
        "model_env":    "CODER_MODEL",
        "model_default": os.getenv("CODER_MODEL", "codellama") or "codellama",
        "tier":          "T2",
        "queue_suffix":  "t2-workers",
        "system_prompt": (
            "You are a specialist code node in the MYCONEX distributed mesh. "
            "Your role is code generation, debugging, refactoring, and explanation. "
            "Always produce clean, well-commented, working code. "
            "When debugging, explain the root cause before providing the fix. "
            "Prefer idiomatic solutions over clever ones."
        ),
    },
    "summarizer": {
        "model_env":    "SUMMARIZER_MODEL",
        "model_default": "llama3",
        "tier":          "T3",
        "queue_suffix":  "t3-workers",
        "system_prompt": (
            "You are a specialist summarisation node in the MYCONEX distributed mesh. "
            "Your role is to produce concise, accurate summaries of documents and content. "
            "Extract key points, preserve important nuance, and omit filler. "
            "Format output as bullet points unless prose is explicitly requested."
        ),
    },
}


def build_specialist_handler(role: str | None = None):
    """
    Build and return a configured NATSRemoteHandler for the given role.
    Falls back to a general-purpose handler if role is None or unknown.
    """
    from core.messaging.remote_handler import NATSRemoteHandler

    role = (role or os.getenv("SPECIALIST_ROLE", "")).lower().strip()

    if role not in _ROLE_CONFIGS:
        logger.info("[specialist] no role set — using general-purpose handler")
        return NATSRemoteHandler()

    cfg = _ROLE_CONFIGS[role]
    model = os.getenv(cfg["model_env"], cfg["model_default"])
    node_name = os.getenv("NATS_NODE_NAME", "buzlock") + f"-{role}"

    logger.info("[specialist] building %s handler — model=%s tier=%s", role, model, cfg["tier"])

    handler = NATSRemoteHandler(
        node_name=node_name,
        model=model,
    )
    # Override system prompt used in _run_llm
    handler._specialist_system_prompt = cfg["system_prompt"]
    handler._tier_queue = cfg["queue_suffix"]

    return handler


class SpecialistRegistry:
    """
    Lightweight registry of which specialist roles this node can serve.
    Published to mesh.roster on startup so peers know available capabilities.
    """

    def __init__(self) -> None:
        self._roles: list[str] = []
        role = os.getenv("SPECIALIST_ROLE", "").lower().strip()
        if role in _ROLE_CONFIGS:
            self._roles.append(role)

    @property
    def roles(self) -> list[str]:
        return list(self._roles)

    def to_roster_entry(self) -> dict:
        return {
            "node":    os.getenv("NATS_NODE_NAME", "buzlock"),
            "roles":   self._roles,
            "tier":    _ROLE_CONFIGS[self._roles[0]]["tier"] if self._roles else "T2",
        }
