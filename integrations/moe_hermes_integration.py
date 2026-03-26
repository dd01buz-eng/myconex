"""
MYCONEX MoE-Hermes Integration Layer

Combines two vendored repos to replace Ollama/llama3.1:8b as the primary LLM:

  flash-moe (github.com/danveloper/flash-moe)
    Mixture-of-Experts C/Metal inference engine for Qwen3.5-397B-A17B on
    macOS Apple Silicon. Used as a subprocess backend when the compiled binary
    and model weights are present.

  hermes-agent (github.com/NousResearch/hermes-agent)
    Multi-provider AI agent framework with smart model routing. Provides:
      - hermes_constants: Nous Research & OpenRouter API endpoints
      - agent.smart_model_routing: complexity-based cheap/strong routing

Architecture — MoE expert chain (first success wins per request):
  1. flash-moe local binary  (macOS only; Qwen3.5-397B-A17B-4bit)
  2. Nous Research API        Hermes-3-Llama-3.1-8B  (complexity < 0.45)
  3. Nous Research API        Hermes-3-Llama-3.1-70B  (complexity >= 0.45)
  4. OpenRouter               nousresearch/hermes-3-llama-3.1-70b  (fallback)
  5. Ollama                   original local backend (always-available fallback)

Environment variables (all optional — missing keys skip that expert):
  NOUS_API_KEY       — Nous Research inference API key
  OPENROUTER_API_KEY — OpenRouter API key
  OLLAMA_URL         — override Ollama endpoint (default: http://localhost:11434)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)

# ── hermes-agent path injection ───────────────────────────────────────────────
# Add integrations/hermes-agent to sys.path so we can import its lightweight
# utilities without a full pip install. Heavy modules (run_agent, model_tools)
# are intentionally NOT imported here.
_HERMES_DIR = Path(__file__).parent / "hermes-agent"
if _HERMES_DIR.is_dir() and str(_HERMES_DIR) not in sys.path:
    sys.path.insert(0, str(_HERMES_DIR))

try:
    from hermes_constants import NOUS_API_BASE_URL, OPENROUTER_BASE_URL
    from agent.smart_model_routing import _COMPLEX_KEYWORDS as _HERMES_COMPLEX_KW
    _HERMES_AVAILABLE = True
    logger.debug("hermes-agent utilities loaded from %s", _HERMES_DIR)
except ImportError as _hermes_err:
    # Graceful fallback: define constants inline so the rest of the module works
    NOUS_API_BASE_URL = "https://inference-api.nousresearch.com/v1"
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    _HERMES_COMPLEX_KW: set[str] = set()
    _HERMES_AVAILABLE = False
    logger.debug("hermes-agent import skipped (%s); using inline constants", _hermes_err)

# Import MYCONEX base classes (project-local, always available)
from orchestration.agents.base_agent import (
    AgentConfig,
    AgentContext,
    AgentResult,
    InferenceAgent,
)

# ─── Flash-MoE binary location ────────────────────────────────────────────────
_FLASH_MOE_BINARY = Path(__file__).parent / "flash-moe" / "metal_infer" / "infer"

# Merge hermes-agent keyword set with additional domain-specific terms
_COMPLEX_KEYWORDS: set[str] = _HERMES_COMPLEX_KW | {
    "implement", "implementation", "refactor", "analyze", "analysis",
    "architecture", "design", "optimize", "review", "traceback", "exception",
    "benchmark", "compare", "algorithm", "function", "class", "module",
    "database", "migration", "deployment", "infrastructure", "security",
}


# ─── Expert Configuration ─────────────────────────────────────────────────────

@dataclass
class ExpertConfig:
    """Configuration for a single MoE expert (LLM endpoint)."""

    name: str
    model: str
    base_url: str
    api_key_env: str                     # name of env var holding the API key
    task_types: list[str] = field(default_factory=list)
    min_complexity: float = 0.0          # inclusive lower bound for complexity routing
    max_complexity: float = 1.0          # inclusive upper bound for complexity routing
    is_local: bool = False               # True for subprocess backends (flash-moe)


def default_expert_pool() -> list[ExpertConfig]:
    """
    Default MoE expert pool.

    Priority order within each request:
      1. flash-moe local (macOS only — skipped silently elsewhere)
      2. Nous hermes-fast  (8B, complexity < 0.45)
      3. Nous hermes-strong (70B, complexity >= 0.45)
      4. OpenRouter hermes (70B, universal fallback)
    """
    return [
        ExpertConfig(
            name="flash-moe-local",
            model="qwen3.5-397b-a17b-4bit",
            base_url="",
            api_key_env="",
            task_types=["inference", "chat", "ask", "generate", "completion", "code",
                        "summarize", "translate", "search", "classify"],
            min_complexity=0.0,
            max_complexity=1.0,
            is_local=True,
        ),
        ExpertConfig(
            name="hermes-fast",
            model="NousResearch/Hermes-3-Llama-3.1-8B",
            base_url=NOUS_API_BASE_URL,
            api_key_env="NOUS_API_KEY",
            task_types=["chat", "ask", "summarize", "translate"],
            min_complexity=0.0,
            max_complexity=0.45,
        ),
        ExpertConfig(
            name="hermes-strong",
            model="NousResearch/Hermes-3-Llama-3.1-70B",
            base_url=NOUS_API_BASE_URL,
            api_key_env="NOUS_API_KEY",
            task_types=["inference", "generate", "code", "completion", "chat",
                        "ask", "summarize", "search", "classify"],
            min_complexity=0.45,
            max_complexity=1.0,
        ),
        ExpertConfig(
            name="openrouter-hermes",
            model="nousresearch/hermes-3-llama-3.1-70b",
            base_url=OPENROUTER_BASE_URL,
            api_key_env="OPENROUTER_API_KEY",
            task_types=["inference", "chat", "ask", "generate", "completion", "code",
                        "summarize", "translate", "search", "classify"],
            min_complexity=0.0,
            max_complexity=1.0,
        ),
    ]


# ─── Complexity Scorer ────────────────────────────────────────────────────────

def _score_complexity(messages: list[dict]) -> float:
    """
    Score message complexity for MoE expert selection (0.0 = simple, 1.0 = complex).

    Mirrors the heuristic in hermes-agent's smart_model_routing.py and extends it
    with additional signals relevant to MYCONEX's mesh task types.
    """
    user_text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    )
    if not user_text:
        return 0.5

    score = 0.0

    # Length factor (0 → 0.30)
    score += min(0.30, len(user_text.split()) / 200.0)

    # Code / technical markup (0 or 0.20)
    if "```" in user_text or re.search(r"`[^`\n]+`", user_text):
        score += 0.20

    # Technical keyword density (0 → 0.25)
    word_set = {w.strip(".,;:!?()[]{}\"'`") for w in user_text.lower().split()}
    score += min(0.25, len(word_set & _COMPLEX_KEYWORDS) * 0.08)

    # Multi-turn depth (0 → 0.15)
    user_turns = sum(1 for m in messages if m.get("role") == "user")
    score += min(0.15, user_turns * 0.04)

    # URL presence → research / analysis task (0 or 0.10)
    if re.search(r"https?://", user_text):
        score += 0.10

    return min(1.0, score)


# ─── Flash-MoE Subprocess Backend ────────────────────────────────────────────

class FlashMoEBackend:
    """
    Subprocess wrapper for the flash-moe C/Metal inference binary.

    flash-moe (github.com/danveloper/flash-moe) runs Qwen3.5-397B-A17B
    locally via Apple Metal on macOS Apple Silicon.  Build the binary by
    running `make` inside integrations/flash-moe/metal_infer/ on a macOS
    host with Xcode installed.

    On Linux or when the binary is missing this class returns available=False
    and the MoERouter skips it without error.
    """

    def __init__(
        self,
        binary_path: Path = _FLASH_MOE_BINARY,
        tokens: int = 512,
        timeout: float = 120.0,
    ) -> None:
        self.binary_path = binary_path
        self.tokens = tokens
        self.timeout = timeout

    @property
    def available(self) -> bool:
        """True only when the binary exists and we are running on macOS."""
        return platform.system() == "Darwin" and self.binary_path.is_file()

    async def generate(self, prompt: str) -> str:
        """
        Run flash-moe inference.  Raises RuntimeError when unavailable or when
        the subprocess exits non-zero.
        """
        if not self.available:
            raise RuntimeError(
                f"flash-moe binary not available at {self.binary_path} "
                f"(platform={platform.system()})"
            )

        proc = await asyncio.create_subprocess_exec(
            str(self.binary_path),
            "--prompt", prompt,
            "--tokens", str(self.tokens),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"flash-moe timed out after {self.timeout}s")

        if proc.returncode != 0:
            raise RuntimeError(
                f"flash-moe exited {proc.returncode}: {stderr.decode()[:300]}"
            )

        return stdout.decode().strip()

    @staticmethod
    def messages_to_prompt(messages: list[dict]) -> str:
        """
        Convert OpenAI-style message list to the ChatML prompt format expected
        by the flash-moe Qwen3.5 model.
        """
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant")
        return "\n".join(parts)


# ─── MoE Router ──────────────────────────────────────────────────────────────

class MoERouter:
    """
    Mixture-of-Experts style router.

    Scores input complexity and selects the most appropriate expert model from
    the pool.  Multiple experts are tried in pool-order; the first to succeed
    wins.  Ollama is always the final fallback.

    This mirrors flash-moe's architectural principle of routing tokens through
    the most specialised (and cost-efficient) expert, but operates at the
    request level rather than the token level.
    """

    def __init__(
        self,
        experts: Optional[list[ExpertConfig]] = None,
        flash_moe: Optional[FlashMoEBackend] = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.1:8b",
    ) -> None:
        self.experts = experts if experts is not None else default_expert_pool()
        self.flash_moe = flash_moe or FlashMoEBackend()
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def select_experts(self, task_type: str, complexity: float) -> list[ExpertConfig]:
        """
        Return experts eligible for this (task_type, complexity) pair.

        Ordered by: (1) primary eligibility (task type + complexity band),
        then (2) any expert that handles the task type regardless of complexity.
        """
        primary: list[ExpertConfig] = []
        secondary: list[ExpertConfig] = []
        for expert in self.experts:
            handles_task = task_type in expert.task_types
            in_band = expert.min_complexity <= complexity <= expert.max_complexity
            if handles_task and in_band:
                primary.append(expert)
            elif handles_task:
                secondary.append(expert)
        return primary + secondary

    def has_api_key(self, expert: ExpertConfig) -> bool:
        """Return True when the required API key env var is set and non-empty."""
        return bool(expert.api_key_env and os.getenv(expert.api_key_env))


# ─── HermesMoE Inference Agent ───────────────────────────────────────────────

class HermesMoEAgent(InferenceAgent):
    """
    Primary inference agent for MYCONEX, powered by the combined
    flash-moe / hermes-agent stack.

    Replaces the default Ollama-backed InferenceAgent.

    LLM call chain per request (first success wins):
      1. flash-moe binary      (macOS/Apple Silicon only)
      2. Nous Research API     Hermes-3 8B or 70B based on complexity score
      3. OpenRouter            nousresearch/hermes-3-llama-3.1-70b
      4. Ollama                original local backend (always present as final fallback)

    The complexity score drives Mixture-of-Experts routing at the request
    level, mirroring flash-moe's per-token expert selection.
    """

    def __init__(
        self,
        config: AgentConfig,
        moe_router: Optional[MoERouter] = None,
    ) -> None:
        super().__init__(config)
        self._moe = moe_router or MoERouter(
            ollama_url=config.ollama_url,
            ollama_model=config.model,
        )

    # ── Override: handle_task ─────────────────────────────────────────────────

    async def handle_task(
        self,
        task_id: str,
        task_type: str,
        payload: dict,
        context: Optional[AgentContext] = None,
    ) -> AgentResult:
        """Handle a task, threading task_type into the MoE routing decision."""
        prompt = payload.get("prompt") or payload.get("message") or payload.get("query", "")
        if not prompt:
            return AgentResult(
                task_id=task_id,
                agent_name=self.name,
                success=False,
                error="No prompt in payload",
            )

        if context is None:
            context = AgentContext()

        if not any(t.role == "system" for t in context.history):
            context.add("system", self.config.system_prompt)
        context.add("user", prompt)
        context.trim()

        response = await self.chat(context.to_messages(), task_type=task_type)
        context.add("assistant", response)

        return AgentResult(
            task_id=task_id,
            agent_name=self.name,
            success=True,
            output={"response": response, "session_id": context.session_id},
            model_used=self.config.model,
        )

    # ── Override: chat ────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task_type: str = "chat",
    ) -> str:
        """
        Route the request through the MoE expert chain.

        Scores complexity, selects the best expert, tries experts in order,
        and falls back to Ollama when all remote options are exhausted.
        """
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        complexity = _score_complexity(messages)

        logger.debug(
            "[moe] task=%s complexity=%.2f hermes=%s flash_moe=%s",
            task_type, complexity, _HERMES_AVAILABLE, self._moe.flash_moe.available,
        )

        # ── 0. Exo distributed cluster (consensus-gated, high-complexity only) ──
        try:
            from core.exo_pool import get_pool as _get_exo_pool
            _exo = _get_exo_pool()
            if _exo.is_ready(complexity):
                base_url, exo_model = _exo.endpoint()
                exo_expert = ExpertConfig(
                    name="exo-cluster",
                    model=exo_model,
                    base_url=base_url,
                    api_key_env="",          # exo needs no key
                    task_types=["inference", "generate", "code", "chat",
                                "ask", "summarize", "search", "classify"],
                    min_complexity=0.0,
                    max_complexity=1.0,
                )
                try:
                    result = await self._call_expert_api(
                        messages, exo_expert, temperature, max_tokens
                    )
                    if result:
                        logger.info(
                            "[moe] served by exo-cluster (%s) complexity=%.2f nodes=%s",
                            exo_model, complexity, _exo._contributing_nodes,
                        )
                        return result
                except Exception as exc:
                    logger.warning("[moe] exo-cluster failed (%s); continuing chain", exc)
        except ImportError:
            pass

        # ── 1. flash-moe local binary (macOS only) ────────────────────────────
        if self._moe.flash_moe.available:
            try:
                prompt = FlashMoEBackend.messages_to_prompt(messages)
                result = await self._moe.flash_moe.generate(prompt)
                if result:
                    logger.info("[moe] served by flash-moe local (qwen3.5-397b)")
                    return result
            except Exception as exc:
                logger.warning("[moe] flash-moe failed (%s); trying API experts", exc)

        # ── 2. API experts in MoE-selected order ─────────────────────────────
        for expert in self._moe.select_experts(task_type, complexity):
            if expert.is_local:
                continue  # already attempted above
            if not self._moe.has_api_key(expert):
                logger.debug("[moe] skip %s (no key for %s)", expert.name, expert.api_key_env)
                continue
            try:
                result = await self._call_expert_api(messages, expert, temperature, max_tokens)
                if result:
                    logger.info("[moe] served by %s (%s)", expert.name, expert.model)
                    return result
            except Exception as exc:
                logger.warning("[moe] expert %s failed (%s)", expert.name, exc)

        # ── 3. Ollama fallback ────────────────────────────────────────────────
        logger.info("[moe] falling back to Ollama (%s)", self._moe.ollama_model)
        return await self._chat_ollama(messages, self._moe.ollama_model, temperature, max_tokens)

    async def _call_expert_api(
        self,
        messages: list[dict],
        expert: ExpertConfig,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """POST to an OpenAI-compatible /chat/completions endpoint."""
        api_key = os.getenv(expert.api_key_env, "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": expert.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        url = f"{expert.base_url.rstrip('/')}/chat/completions"
        resp = await self._http.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ── Override: stream_chat ─────────────────────────────────────────────────

    async def stream_chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        task_type: str = "chat",
    ) -> AsyncIterator[str]:
        """
        Streaming token iterator.

        Tries API experts first (SSE streaming), then falls back to Ollama
        streaming.  flash-moe does not support token streaming via subprocess.
        """
        complexity = _score_complexity(messages)
        temperature = self.config.temperature
        max_tokens = self.config.max_tokens

        for expert in self._moe.select_experts(task_type, complexity):
            if expert.is_local:
                continue
            if not self._moe.has_api_key(expert):
                continue
            try:
                async for token in self._stream_expert_api(
                    messages, expert, temperature, max_tokens
                ):
                    yield token
                return
            except Exception as exc:
                logger.warning("[moe] stream expert %s failed (%s)", expert.name, exc)

        # Ollama streaming fallback
        async for token in super().stream_chat(messages, model=self._moe.ollama_model):
            yield token

    async def _stream_expert_api(
        self,
        messages: list[dict],
        expert: ExpertConfig,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Stream SSE tokens from an OpenAI-compatible endpoint."""
        api_key = os.getenv(expert.api_key_env, "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": expert.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        url = f"{expert.base_url.rstrip('/')}/chat/completions"
        async with self._http.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    token = chunk["choices"][0].get("delta", {}).get("content", "")
                    if token:
                        yield token
                except Exception:
                    continue

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        base = super().status()
        base["moe"] = {
            "flash_moe_available": self._moe.flash_moe.available,
            "hermes_utils_loaded": _HERMES_AVAILABLE,
            "experts": [
                {
                    "name": e.name,
                    "model": e.model,
                    "has_key": self._moe.has_api_key(e),
                    "is_local": e.is_local,
                }
                for e in self._moe.experts
            ],
        }
        return base


# ─── Factory ─────────────────────────────────────────────────────────────────

def create_moe_hermes_agent(
    name: str = "inference-primary",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "llama3.1:8b",
    system_prompt: str = "You are a helpful AI assistant in the MYCONEX mesh.",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: float = 60.0,
    max_concurrent: int = 4,
    experts: Optional[list[ExpertConfig]] = None,
    flash_moe_binary: Optional[str] = None,
    flash_moe_tokens: int = 512,
) -> HermesMoEAgent:
    """
    Create a HermesMoEAgent ready to drop into a TaskRouter.

    The agent name is "hermes-moe" in the status label; the actual model
    used per request is chosen dynamically by the MoE router.

    Args:
        name:              Agent identifier registered in AgentRegistry.
        ollama_url:        Ollama endpoint used as final fallback.
        ollama_model:      Ollama model tag for the fallback.
        system_prompt:     Default system prompt injected on first turn.
        temperature:       Default sampling temperature.
        max_tokens:        Default max completion tokens.
        timeout:           HTTP request timeout in seconds.
        max_concurrent:    Max simultaneous tasks (asyncio semaphore).
        experts:           Custom expert pool; uses default_expert_pool() if None.
        flash_moe_binary:  Explicit path to the flash-moe binary; auto-detected if None.
        flash_moe_tokens:  Max tokens to generate via flash-moe subprocess.
    """
    config = AgentConfig(
        name=name,
        agent_type="inference",
        model="hermes-moe",          # label; actual model selected per-request
        ollama_url=ollama_url,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_concurrent=max_concurrent,
    )

    binary_path = Path(flash_moe_binary) if flash_moe_binary else _FLASH_MOE_BINARY
    flash_backend = FlashMoEBackend(binary_path=binary_path, tokens=flash_moe_tokens)

    router = MoERouter(
        experts=experts,
        flash_moe=flash_backend,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
    )

    return HermesMoEAgent(config=config, moe_router=router)
