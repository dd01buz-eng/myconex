"""
MYCONEX Discord Gateway — Hermes-Agent Edition

Full-featured Discord bot powered by hermes-agent's AIAgent for:
  ✦ Complete tool-calling loop — web search, code execution, file ops, memory, MCP, and 40+ more
  ✦ Live streaming responses — status message edits as tokens arrive
  ✦ Tool progress indicators — shows which tool the agent is using in real time
  ✦ Per-channel conversation memory — each channel/thread maintains its own history
  ✦ File & image attachments — URLs passed directly to the agent context
  ✦ Slash commands — /ask /reset /new /status /tools /model
  ✦ Thread support — per-thread context isolation, auto-thread creation
  ✦ Typing indicator while the agent is working
  ✦ Reaction status — 👀 processing → ✅ done / ❌ error

Required env vars (set in .env):
    DISCORD_BOT_TOKEN       — bot token from Discord Developer Portal

Optional API keys (set to activate cloud LLM providers):
    NOUS_API_KEY            — Nous Research API → Hermes-3-Llama-3.1-70B  (highest priority)
    OPENROUTER_API_KEY      — OpenRouter → nousresearch/hermes-3-llama-3.1-70b

Without API keys the bot falls back to Ollama's OpenAI-compatible endpoint (llama3.1:8b).

hermes-agent full tool access requires:
    pip install -e integrations/hermes-agent
Without it the bot silently falls back to single-shot TaskRouter completions.

Provider resolution uses hermes-agent's own system (~/.hermes/config.yaml):
    • `hermes login`                 → free Nous Research Hermes models via OAuth2
    • custom_providers in config.yaml → any local endpoint (Ollama, vLLM, llama.cpp)
    • HERMES_INFERENCE_PROVIDER env  → override provider at runtime
    Falls back to NOUS_API_KEY / OPENROUTER_API_KEY env vars, then Ollama /v1.

Optional env overrides:
    DISCORD_REQUIRE_MENTION         — "true" to only respond when @mentioned
    DISCORD_FREE_RESPONSE_CHANNELS  — comma-separated channel IDs (no mention needed)
    DISCORD_AUTO_THREAD             — "true" to auto-create a thread per @mention
    DISCORD_ALLOW_BOTS              — "none" | "mentions" | "all"
    DISCORD_ALLOWED_USERS           — comma-separated Discord user IDs (empty = all)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Discord ───────────────────────────────────────────────────────────────────
try:
    import discord
    from discord.ext import commands as discord_commands
    _DISCORD_AVAILABLE = True
except ImportError:
    _DISCORD_AVAILABLE = False
    logger.error("discord.py not installed. Run: pip install 'discord.py>=2.0'")

# ── hermes-agent AIAgent ──────────────────────────────────────────────────────
# Inject integrations/hermes-agent into sys.path so its modules are importable
# without a formal pip install of the package.
_HERMES_DIR = Path(__file__).parent.parent.parent / "integrations" / "hermes-agent"
if _HERMES_DIR.is_dir() and str(_HERMES_DIR) not in sys.path:
    sys.path.insert(0, str(_HERMES_DIR))

try:
    from run_agent import AIAgent  # type: ignore[import]
    _AIAGENT_AVAILABLE = True
    logger.info("[discord] hermes-agent AIAgent loaded — full tool access enabled")
except Exception as _hermes_err:
    AIAgent = None  # type: ignore[assignment,misc]
    _AIAGENT_AVAILABLE = False
    logger.warning(
        "[discord] hermes-agent AIAgent unavailable (%s) — falling back to TaskRouter",
        _hermes_err,
    )

# ── Agentic tools (memory / research / task_execution) ────────────────────────
# Register BEFORE AIAgent is constructed so they appear in the tool list.
try:
    from core.gateway.agentic_tools import register_agentic_tools, AGENTIC_TOOLSET
    _AGENTIC_TOOLS_OK = register_agentic_tools()
except Exception as _at_err:
    _AGENTIC_TOOLS_OK = False
    AGENTIC_TOOLSET = "agentic"
    logger.warning("[discord] agentic_tools registration failed: %s", _at_err)

# ── Self-improvement + DLAM task generator ────────────────────────────────────
from core.gateway.self_improvement import HermesSelfImprover
from core.gateway.dlam_tasks import DLAMTaskGenerator

# ── MYCONEX internals ─────────────────────────────────────────────────────────
from orchestration.agents.base_agent import AgentContext
from orchestration.workflows.task_router import TaskRouter

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_MSG_LEN = 2000          # Discord hard character limit
MAX_HISTORY_TURNS = 50      # Conversation turns kept per channel (user+assistant pairs)
STREAM_EDIT_INTERVAL = 1.2  # Min seconds between message edits while streaming
MAX_TRACKED_THREADS = 500   # Thread IDs persisted across restarts
THREAD_STATE_FILE   = Path.home() / ".myconex" / "discord_threads.json"
HISTORY_DIR         = Path.home() / ".myconex" / "chat_histories"
RAG_MAX_CHARS       = 1400  # Max chars of knowledge-base context injected per turn
RAG_RESULTS         = 3     # Number of Qdrant results to inject
FEEDBACK_FILE       = Path.home() / ".myconex" / "feedback_log.jsonl"
_GATEWAY_MEMORY_FILE = Path.home() / ".myconex" / "memory.json"

# Added on top of SOUL.md (loaded from ~/.hermes/SOUL.md).
# Does NOT re-declare identity — SOUL.md owns that.
# Adds MYCONEX mesh context, tool catalogue, and behavioral rules.
_SYSTEM_PROMPT = (
    "You are **Buzlock**, a personal AI running on this machine and powered by MYCONEX. "
    "Node: T2 (RTX 4060 Laptop, Linux, 8 GB VRAM).\n\n"

    "## Memory — use proactively\n\n"

    "**remember(action, key, value)** — persistent key-value store\n"
    "  remember(action='store', key='user_name', value='Alice')  ← save a fact\n"
    "  remember(action='retrieve', key='user_name')  ← exact lookup by key\n"
    "  remember(action='list')  ← see all stored keys\n\n"

    "**search_memory(query, limit, source)** — semantic search over the entire knowledge base\n"
    "  Searches emails, YouTube videos, RSS articles, podcasts, AND all stored remember facts.\n"
    "  Use when the user asks about topics they've read or watched, or to find related knowledge.\n"
    "  search_memory(query='machine learning projects')  ← finds relevant content\n"
    "  search_memory(query='travel', source='email')  ← filter by source\n\n"

    "## Live actions\n\n"

    "**dlam(action, ...)** — PRIMARY web and computer-use tool. Always prefer this over research or web_read.\n"
    "  dlam(action='search', query='latest Rust news')  ← web search via real browser\n"
    "  dlam(action='browse', url='https://...', task='summarise the main points')  ← browse a page\n"
    "  dlam(action='task', task='open Firefox and go to github.com')  ← any keyboard/mouse task\n"
    "  dlam(action='status')  ← check if DLAM is available\n"
    "  Use dlam for ALL web research and browsing tasks. Only fall back to research() or web_read() "
    "if dlam reports it is unavailable.\n\n"

    "**research(query)** — fallback web search (DuckDuckGo, no browser). Use only if dlam is unavailable.\n"
    "  research(query='latest Rust news')\n\n"

    "**web_read(url)** — fallback plain HTTP fetch. Use only if dlam is unavailable.\n"
    "  web_read(url='https://...')\n\n"

    "**task_execution(command)** — run any shell command on this Linux machine\n"
    "  Launch apps, run scripts, move files, check system state — anything a terminal can do.\n"
    "  task_execution(command='steam &')  ← launch Steam\n"
    "  task_execution(command='ls ~/Downloads')\n\n"

    "**python_repl(code, session_id)** — execute Python in a persistent session\n"
    "  Use for calculations, data analysis, file parsing, or any code execution.\n"
    "  python_repl(code='import math; print(math.pi * 5**2)')\n\n"

    "**check_email(action, query, limit)** — search and read Gmail messages\n"
    "  check_email(action='search', query='invoice', limit=5)\n\n"

    "**read_file / write_file / edit_file / list_dir / glob_files / grep_files** — filesystem\n"
    "  Read, write, search, and navigate files on this machine.\n\n"

    "## Rules\n"
    "• **When the user mentions a personal fact** (name, preference, project, etc.): "
    "store it immediately with remember(action='store', ...).\n"
    "• **Before answering personal questions** (\"what do I like?\", \"what did I say about X?\"): "
    "check remember(action='list') or search_memory(query=...) first.\n"
    "• **When action is needed**: call the tool immediately, then report the result. "
    "Never explain what you are about to do.\n"
    "• **For pure conversation**: just respond naturally — no tools needed.\n\n"

    "Discord renders standard Markdown. Keep responses focused and concise."
)

# Injected at every API call turn (including mid-loop post-tool turns).
_EPHEMERAL_PROMPT = (
    "Call tools immediately. Never describe what you would do. "
    "After a tool result, continue to the next step or summarise."
)

# "skills" and "session_search" toolsets inject 3000-char guidance blocks that
# conflict with our custom tool instructions and confuse smaller local models.
# AGENTIC_TOOLSET exposes all 21 MYCONEX tools: remember, search_memory,
# research, web_read, task_execution, python_repl, check_email, filesystem, etc.
_DISCORD_TOOLSETS = [
    AGENTIC_TOOLSET,
]


# ─── History persistence helpers ─────────────────────────────────────────────

def _history_key_to_filename(key: str) -> str:
    """Convert a channel key (may contain ':') to a safe filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in key) + ".json"


def _history_save(key: str, history: list) -> None:
    """Persist channel history to disk."""
    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        path = HISTORY_DIR / _history_key_to_filename(key)
        path.write_text(json.dumps(history, ensure_ascii=False))
    except Exception as exc:
        logger.debug("[discord] history save failed for %s: %s", key, exc)


def _history_load(key: str) -> list:
    """Load persisted channel history from disk (empty list if not found)."""
    try:
        path = HISTORY_DIR / _history_key_to_filename(key)
        if path.exists():
            return json.loads(path.read_text())
    except Exception as exc:
        logger.debug("[discord] history load failed for %s: %s", key, exc)
    return []


def _trim_history_to_turns(messages: list, max_turns: int) -> list:
    """
    Trim a message list to keep at most max_turns user-initiated turns.

    Counts user messages to find the cut point so that all associated
    assistant and tool messages for a kept turn are never orphaned.
    This is correct for agentic conversations where a single user turn
    may produce many tool-call messages before the final assistant reply.
    """
    if not messages:
        return messages
    user_indices = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    if len(user_indices) <= max_turns:
        return messages
    cut_at = user_indices[len(user_indices) - max_turns]
    return messages[cut_at:]


# ─── Memory pre-load helper ───────────────────────────────────────────────────

def _load_memory_for_prompt() -> str:
    """
    Read ~/.myconex/memory.json and return a formatted section for the system
    prompt so the agent knows stored facts without needing a tool call.
    Returns empty string if memory is empty or unreadable.
    """
    try:
        memory_file = _GATEWAY_MEMORY_FILE
        if not memory_file.exists():
            return ""
        data = json.loads(memory_file.read_text())
        if not isinstance(data, dict) or not data:
            return ""
        lines = ["[Stored facts about the user:]"]
        for key, val in list(data.items())[:30]:          # cap at 30 entries
            val_str = str(val)[:200]                       # cap each value at 200 chars
            lines.append(f"  {key}: {val_str}")
        return "\n".join(lines)
    except Exception:
        return ""


# ─── Auto-RAG helper ──────────────────────────────────────────────────────────

async def _rag_context(query: str) -> str:
    """
    Query the Qdrant knowledge base with the user's message and return a
    formatted context block (empty string if unavailable or no results).
    """
    if not query or not query.strip():
        return ""
    try:
        from integrations.knowledge_store import search
        results = await search(query, limit=RAG_RESULTS, score_threshold=0.40)
        if not results:
            return ""
        parts = ["[Relevant knowledge from your personal knowledge base:]\n"]
        total = 0
        for r in results:
            src   = r.get("source", "")
            meta  = r.get("metadata", {})
            label = meta.get("title") or meta.get("subject") or meta.get("feed_title") or src
            score = r.get("score", 0)
            text  = r.get("content", "")
            chunk = f"• [{src}] {label} (relevance {score})\n  {text[:400]}\n"
            if total + len(chunk) > RAG_MAX_CHARS:
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n".join(parts) if len(parts) > 1 else ""
    except Exception as exc:
        logger.debug("[discord] RAG query failed: %s", exc)
        return ""


# ─── Self-improvement: lessons loader ────────────────────────────────────────

_LESSONS_FILE = Path(__file__).parents[2] / "lessons.md"


def _load_lessons() -> str:
    """
    Read lessons.md from the repo root and return a compact behavioral-rules
    block for injection into the system prompt.  Returns empty string if the
    file is missing or empty.
    """
    try:
        if not _LESSONS_FILE.exists():
            return ""
        text = _LESSONS_FILE.read_text(errors="replace").strip()
        if not text:
            return ""
        # Strip the file header (everything before the first ## rule)
        idx = text.find("\n## ")
        if idx != -1:
            text = text[idx:].strip()
        if not text:
            return ""
        return "[Learned behavioral rules — follow strictly]\n" + text
    except Exception:
        return ""


def _auto_generate_lessons() -> None:
    """
    Scan feedback_log.jsonl for patterns in downvoted responses and
    auto-append new lessons to lessons.md when a pattern recurs ≥ 3 times.

    Currently detects:
      - Response style patterns (e.g. heavy bullet lists, very long replies)
      - Repeated common words in downvoted responses that aren't in upvoted ones

    Runs once at startup; safe to call multiple times (deduplicates by rule text).
    """
    try:
        if not FEEDBACK_FILE.exists():
            return
        entries = [
            json.loads(l) for l in FEEDBACK_FILE.read_text().splitlines() if l.strip()
        ]
        if len(entries) < 5:
            return

        neg = [e for e in entries if not e.get("positive")]
        pos = [e for e in entries if e.get("positive")]
        if len(neg) < 3:
            return

        stopwords = {
            "the", "a", "an", "is", "it", "to", "of", "and", "in", "that",
            "was", "for", "on", "are", "be", "as", "at", "by", "we", "you",
            "i", "this", "with", "have", "but", "not", "they", "so", "or",
            "from", "which", "your", "can", "will", "just", "do", "our",
        }

        # Word frequency in negative vs positive responses
        def _word_freq(items: list) -> dict[str, int]:
            freq: dict[str, int] = {}
            for e in items:
                for w in e.get("bot_response_preview", "").lower().split():
                    w = w.strip(".,!?\"':-")
                    if len(w) > 4 and w not in stopwords:
                        freq[w] = freq.get(w, 0) + 1
            return freq

        neg_freq = _word_freq(neg)
        pos_freq = _word_freq(pos)

        # Words that appear ≥3× more in negative responses than positive
        neg_signals = [
            w for w, c in neg_freq.items()
            if c >= 3 and neg_freq.get(w, 0) > pos_freq.get(w, 0) * 2
        ]

        if not neg_signals:
            return

        # Load existing lessons to avoid duplicates
        existing = _LESSONS_FILE.read_text(errors="replace") if _LESSONS_FILE.exists() else ""
        rule_key = f"avoid-pattern-{'_'.join(sorted(neg_signals[:3]))}"
        if rule_key in existing:
            return

        lesson = (
            f"\n\n## [Feedback] — Avoid responses heavy on certain patterns\n\n"
            f"Auto-generated from {len(neg)}/{len(entries)} downvoted responses.\n"
            f"Words/phrases appearing disproportionately in downvoted replies: "
            f"{', '.join(neg_signals[:5])}.\n"
            f"Consider shorter, more direct responses and vary phrasing.\n\n"
            f"<!-- rule-key: {rule_key} -->"
        )
        with _LESSONS_FILE.open("a") as f:
            f.write(lesson)
        logger.info("[self-improve] auto-appended lesson to lessons.md (pattern: %s)",
                    neg_signals[:3])
    except Exception as exc:
        logger.debug("[self-improve] _auto_generate_lessons failed: %s", exc)


# ─── Feedback stats helper ────────────────────────────────────────────────────

def _load_feedback_summary() -> str:
    """
    Read feedback_log.jsonl and return a one-line summary suitable for
    injection into the system prompt.  Returns empty string if no data.
    """
    try:
        if not FEEDBACK_FILE.exists():
            return ""
        lines = [
            json.loads(l) for l in FEEDBACK_FILE.read_text().splitlines()
            if l.strip()
        ]
        if not lines:
            return ""
        total = len(lines)
        pos   = sum(1 for l in lines if l.get("positive"))
        neg   = total - pos
        rate  = round(pos / total * 100)
        # Find common words in downvoted responses (top 3 tokens)
        neg_texts = " ".join(
            l.get("bot_response_preview", "") for l in lines if not l.get("positive")
        ).lower().split()
        stopwords = {"the","a","an","is","it","to","of","and","in","that","was","for","on","are"}
        freq: dict[str, int] = {}
        for w in neg_texts:
            w = w.strip(".,!?\"'")
            if len(w) > 4 and w not in stopwords:
                freq[w] = freq.get(w, 0) + 1
        common_neg = sorted(freq, key=lambda x: -freq[x])[:3]
        neg_hint = ""
        if neg >= 3 and common_neg:
            neg_hint = f" Avoid responses heavy on: {', '.join(common_neg)}."
        return (
            f"[Feedback so far: {pos}/{total} positive ({rate}%).{neg_hint}]"
        )
    except Exception:
        return ""


# ─── Channel State ────────────────────────────────────────────────────────────

class _ChannelState:
    """
    Per-channel conversation state.

    Holds the OpenAI-format message history and a mutable tool_cb slot so we
    can redirect tool-progress updates to the right Discord message on each
    request without recreating the AIAgent.
    """

    __slots__ = ("history", "tool_cb", "clarify", "legacy_ctx", "system_prompt")

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []
        self.tool_cb: Optional[Callable[..., None]] = None
        self.clarify: Optional[_DiscordClarify] = None  # set in _get_or_create_agent
        self.legacy_ctx: Optional[AgentContext] = None  # TaskRouter fallback only
        self.system_prompt: Optional[str] = None        # per-channel override


# ─── Streaming Updater ────────────────────────────────────────────────────────

class _StreamingUpdater:
    """
    Buffers LLM streaming token deltas and rate-limits Discord message edits.

    AIAgent.run_conversation() runs in a ThreadPoolExecutor thread (via
    asyncio.to_thread).  stream_callback is called synchronously from that
    thread, so edits are pushed back to the main event loop via
    asyncio.run_coroutine_threadsafe, respecting Discord's ~5 edits/second
    rate limit via the configurable interval.
    """

    def __init__(
        self,
        message: "discord.Message",
        loop: asyncio.AbstractEventLoop,
        interval: float = STREAM_EDIT_INTERVAL,
    ) -> None:
        self._message = message
        self._loop = loop
        self._interval = interval
        self._parts: List[str] = []
        self._last_edit: float = 0.0

    def on_delta(self, delta: str) -> None:
        """Append a token delta; push an edit when the rate-limit interval elapses."""
        self._parts.append(delta)
        now = time.monotonic()
        if now - self._last_edit >= self._interval:
            self._push()
            self._last_edit = now

    def _push(self) -> None:
        text = "".join(self._parts).strip()
        if not text:
            return
        # Add a cursor glyph while streaming so the user knows it's still going
        display = text if len(text) <= MAX_MSG_LEN - 3 else text[: MAX_MSG_LEN - 6] + " ✍️"
        asyncio.run_coroutine_threadsafe(
            _safe_edit(self._message, display), self._loop
        )


# ─── Clarify Callback ─────────────────────────────────────────────────────────

class _DiscordClarify:
    """
    Bridges the hermes-agent clarify tool to Discord.

    Called synchronously from the AIAgent worker thread.  Posts a question
    (with optional numbered choices) to the Discord channel and blocks until
    the user replies, then returns the reply text.

    Timeout defaults to 5 minutes — after which the agent receives a
    "[no reply — proceed with best guess]" sentinel so it doesn't hang forever.
    """

    TIMEOUT = 300  # seconds

    def __init__(
        self,
        channel: "discord.abc.Messageable",
        author_id: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._channel = channel
        self._author_id = author_id
        self._loop = loop
        # Set while waiting for a reply; cleared by on_message hook
        self._pending: Optional[threading.Event] = None
        self._answer: Optional[str] = None

    def __call__(self, question: str, choices: Optional[List[str]] = None) -> str:
        """Signature required by hermes-agent: (question, choices) -> str."""
        lines = [f"❓ **{question}**"]
        if choices:
            for i, c in enumerate(choices, 1):
                lines.append(f"  **{i}.** {c}")
            lines.append("\n_Reply with a number or type your answer._")

        evt = threading.Event()
        self._pending = evt
        self._answer = None

        asyncio.run_coroutine_threadsafe(
            self._channel.send("\n".join(lines)), self._loop
        )

        evt.wait(timeout=self.TIMEOUT)
        self._pending = None

        if self._answer is None:
            return "[no reply — proceed with best guess]"

        # If the user typed a number and choices were offered, map it back
        if choices:
            stripped = self._answer.strip()
            if stripped.isdigit():
                idx = int(stripped) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
        return self._answer

    def receive(self, text: str) -> None:
        """Call from on_message when a reply arrives while waiting."""
        if self._pending and not self._pending.is_set():
            self._answer = text
            self._pending.set()


# ─── UI Components ────────────────────────────────────────────────────────────
# Defined only when discord.py is available (guarded so import errors don't
# break the module when discord.py is absent).

if _DISCORD_AVAILABLE:
    class ResponseView(discord.ui.View):
        """
        Interactive buttons attached to every bot response.

          🔄 Regenerate  — drop the last assistant turn and re-run the same prompt
          ➕ Continue    — ask the agent to continue its last output
          🆕 New Chat    — clear history and start a fresh conversation
        """

        def __init__(
            self,
            gateway: "DiscordGateway",
            channel_key: str,
            last_prompt: str,
        ) -> None:
            super().__init__(timeout=300)  # buttons expire after 5 minutes
            self._gw = gateway
            self._key = channel_key
            self._last_prompt = last_prompt

        @discord.ui.button(label="🔄 Regenerate", style=discord.ButtonStyle.secondary)
        async def regenerate(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ) -> None:
            await interaction.response.defer()
            # Drop the last assistant turn so the agent produces a new one
            state = self._gw._get_or_create_state(self._key)
            if state.history and state.history[-1].get("role") == "assistant":
                state.history.pop()
            await self._gw._handle_prompt(
                channel=interaction.channel,
                key=self._key,
                content=self._last_prompt,
                author_id=interaction.user.id,
            )

        @discord.ui.button(label="➕ Continue", style=discord.ButtonStyle.secondary)
        async def continue_response(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ) -> None:
            await interaction.response.defer()
            await self._gw._handle_prompt(
                channel=interaction.channel,
                key=self._key,
                content="Continue from where you left off.",
                author_id=interaction.user.id,
            )

        @discord.ui.button(label="🆕 New Chat", style=discord.ButtonStyle.danger)
        async def new_chat(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ) -> None:
            self._gw._reset_channel(self._key)
            await interaction.response.send_message(
                "✅ Started a new conversation.", ephemeral=True
            )
            self.stop()

    class SystemPromptModal(discord.ui.Modal, title="Set System Prompt"):
        """Pop-up form for setting a custom system prompt for this channel."""

        system_input: discord.ui.TextInput = discord.ui.TextInput(
            label="System Prompt",
            style=discord.TextStyle.paragraph,
            placeholder="Leave blank to restore the default MYCONEX system prompt.",
            required=False,
            max_length=2000,
        )

        def __init__(
            self, gateway: "DiscordGateway", channel_key: str
        ) -> None:
            super().__init__()
            self._gw = gateway
            self._key = channel_key

        async def on_submit(self, interaction: discord.Interaction) -> None:
            text = self.system_input.value.strip()
            state = self._gw._get_or_create_state(self._key)
            state.system_prompt = text or None
            if text:
                await interaction.response.send_message(
                    f"✅ Custom system prompt set ({len(text)} chars).", ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    "✅ System prompt reset to default.", ephemeral=True
                )


# ─── Discord Gateway ──────────────────────────────────────────────────────────

class DiscordGateway:
    """
    Full-featured MYCONEX Discord bot backed by hermes-agent's AIAgent.

    Public interface (unchanged from original):
        gw = DiscordGateway(config, router)
        await gw.start()
        await gw.stop()
    """

    def __init__(self, config: dict, router: Optional[TaskRouter] = None) -> None:
        self._config = config
        self._router = router

        # Detect if the primary agent is an RLMAgent — enables richer routing
        self._rlm_agent = None
        if router is not None:
            primary = router.registry.get("inference-primary")
            try:
                from orchestration.agents.rlm_agent import RLMAgent
                if isinstance(primary, RLMAgent):
                    self._rlm_agent = primary
                    logger.info("[discord] RLMAgent detected as primary — full RLM pipeline active")
            except ImportError:
                pass

        discord_cfg = config.get("discord", {})
        self._token: str = os.getenv("DISCORD_BOT_TOKEN", "")
        self._app_id: str = str(discord_cfg.get("application_id", ""))

        self._require_mention: bool = _coerce_bool(
            os.getenv("DISCORD_REQUIRE_MENTION", discord_cfg.get("require_mention", False))
        )
        self._auto_thread: bool = _coerce_bool(
            os.getenv("DISCORD_AUTO_THREAD", discord_cfg.get("auto_thread", False))
        )
        self._allow_bots: str = os.getenv(
            "DISCORD_ALLOW_BOTS", str(discord_cfg.get("allow_bots", "none"))
        )
        self._free_channels: set[str] = set(
            filter(None, os.getenv("DISCORD_FREE_RESPONSE_CHANNELS", "").split(","))
        )
        raw_users = os.getenv("DISCORD_ALLOWED_USERS", "")
        self._allowed_user_ids: set[str] = (
            set(filter(None, raw_users.split(","))) if raw_users else set()
        )

        # Per-channel state + lazy AIAgent pool
        self._states: Dict[str, _ChannelState] = {}
        self._agents: Dict[str, "AIAgent"] = {}  # type: ignore[type-arg]
        # Keys with an agent run currently in progress — guards against
        # concurrent calls on the same channel corrupting conversation history.
        self._in_flight: set[str] = set()

        # Self-improvement and DLAM companion task subsystems
        self._improver  = HermesSelfImprover()
        self._dlam_gen  = DLAMTaskGenerator()
        self._refl_agent: Optional["AIAgent"] = None  # dedicated reflection agent

        self._bot_participated_threads: set[str] = self._load_thread_state()
        self._client: Optional["discord_commands.Bot"] = None
        self._ready = asyncio.Event()
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not _DISCORD_AVAILABLE:
            raise RuntimeError("discord.py not installed — run: pip install 'discord.py>=2.0'")
        if not self._token:
            raise RuntimeError("DISCORD_BOT_TOKEN not set in .env")
        await self._connect()

    async def stop(self) -> None:
        self._running = False
        if self._client and not self._client.is_closed():
            await self._client.close()
        self._ready.clear()
        logger.info("[discord] gateway stopped")

    async def _connect(self) -> bool:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        app_id = int(self._app_id) if self._app_id.isdigit() else None
        self._client = discord_commands.Bot(
            command_prefix="!", intents=intents, application_id=app_id
        )
        self._register_events()
        self._register_slash_commands()

        async def _runner() -> None:
            try:
                await self._client.start(self._token)
            except Exception as exc:
                logger.error("[discord] login failed: %s", exc)
                self._ready.set()

        asyncio.create_task(_runner())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error("[discord] timed out waiting for Discord ready event")
            return False
        return True

    # ── Event Registration ────────────────────────────────────────────────────

    def _register_events(self) -> None:
        client = self._client

        @client.event
        async def on_ready() -> None:
            logger.info("[discord] online as %s (id=%s)", client.user, client.user.id)

            # Run self-improvement loop: analyse feedback patterns and update lessons.md
            try:
                await asyncio.to_thread(_auto_generate_lessons)
                lesson_count = sum(
                    1 for l in (_LESSONS_FILE.read_text(errors="replace")
                                if _LESSONS_FILE.exists() else "").splitlines()
                    if l.startswith("## ")
                )
                logger.info("[self-improve] lessons.md loaded — %d rules active", lesson_count)
            except Exception as exc:
                logger.warning("[self-improve] startup lesson scan failed: %s", exc)

            try:
                synced = await client.tree.sync()
                logger.info("[discord] synced %d slash commands", len(synced))
            except Exception as exc:
                logger.warning("[discord] slash command sync failed: %s", exc)
            self._running = True
            self._ready.set()

            # ── Bot presence ──────────────────────────────────────────────────
            try:
                _, _, model, _ = self._resolve_runtime()
                short_model = model.split("/")[-1][:32]
                activity = discord.Activity(
                    type=discord.ActivityType.listening,
                    name=f"you · {short_model}",
                )
                await client.change_presence(
                    status=discord.Status.online, activity=activity
                )
            except Exception as exc:
                logger.debug("[discord] could not set presence: %s", exc)

            # ── Home channel startup announcement ─────────────────────────────
            home_id = os.getenv("DISCORD_HOME_CHANNEL", "").strip()
            if home_id.lstrip("-").isdigit():
                try:
                    _, _, model, _ = self._resolve_runtime()
                    ch = client.get_channel(int(home_id))
                    if ch:
                        embed = discord.Embed(
                            title="🌐 MYCONEX Online",
                            color=0x00CFFF,
                            description=(
                                "Distributed AI mesh node is ready.\n"
                                f"**Model:** `{model.split('/')[-1]}`\n"
                                f"**hermes-agent:** "
                                f"{'✅ full tools active' if _AIAGENT_AVAILABLE else '⚠️ single-shot fallback'}"
                            ),
                        )
                        embed.set_footer(
                            text="Type a message or use /help to get started."
                        )
                        await ch.send(embed=embed)
                except Exception as exc:
                    logger.debug("[discord] home channel announce failed: %s", exc)

            # ── Notification drain loop ────────────────────────────────────────
            async def _drain_notifications() -> None:
                """Periodically drain the notification bus and post to home channel."""
                try:
                    from core.notifications import drain as _drain
                except Exception:
                    return
                _home_id = os.getenv("DISCORD_HOME_CHANNEL", "").strip()
                while self._running:
                    await asyncio.sleep(60)
                    if not _home_id.lstrip("-").isdigit():
                        continue
                    try:
                        messages = await _drain()
                        if not messages:
                            continue
                        ch = client.get_channel(int(_home_id))
                        if not ch:
                            continue
                        for msg in messages:
                            try:
                                await ch.send(msg)
                            except Exception as exc:
                                logger.debug("[discord] notification send failed: %s", exc)
                    except Exception as exc:
                        logger.debug("[discord] notification drain error: %s", exc)

            asyncio.create_task(_drain_notifications())

            # ── Weekly digest scheduler ───────────────────────────────────────
            async def _post_digest(embed_data: dict) -> None:
                _home_id = os.getenv("DISCORD_HOME_CHANNEL", "").strip()
                if not _home_id.lstrip("-").isdigit():
                    return
                try:
                    ch = client.get_channel(int(_home_id))
                    if ch:
                        embed = discord.Embed(
                            title=embed_data["title"],
                            color=embed_data.get("color", 0x00CFFF),
                            description=embed_data.get("description", ""),
                        )
                        for f in embed_data.get("fields", []):
                            embed.add_field(
                                name=f["name"],
                                value=f["value"],
                                inline=f.get("inline", False),
                            )
                        if embed_data.get("footer"):
                            embed.set_footer(text=embed_data["footer"].get("text", ""))
                        await ch.send(embed=embed)
                except Exception as exc:
                    logger.debug("[discord] digest post failed: %s", exc)

            try:
                from core.digest import schedule_weekly_digest
                asyncio.create_task(schedule_weekly_digest(_post_digest))
            except Exception as exc:
                logger.debug("[discord] could not start digest scheduler: %s", exc)

        @client.event
        async def on_message(message: discord.Message) -> None:
            if message.author == client.user:
                return
            if message.author.bot:
                if self._allow_bots == "none":
                    return
                if self._allow_bots == "mentions" and not client.user.mentioned_in(message):
                    return

            # Feed the reply to any pending clarify callback first
            key = _msg_key(message)
            state = self._states.get(key)
            if state and state.clarify and state.clarify._pending:
                state.clarify.receive(message.content or "")
                return  # consumed — don't also route as a new conversation turn

            await self._handle_message(message)

        @client.event
        async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
            """Log 👍/👎 reactions on bot messages to ~/.myconex/feedback_log.jsonl."""
            if str(payload.emoji) not in ("👍", "👎"):
                return
            if payload.user_id == client.user.id:
                return
            try:
                ch = client.get_channel(payload.channel_id)
                if ch is None:
                    return
                msg = await ch.fetch_message(payload.message_id)
                if msg.author.id != client.user.id:
                    return

                import json as _json
                from datetime import datetime, timezone
                from pathlib import Path as _Path

                _fb_file = _Path.home() / ".myconex" / "feedback_log.jsonl"
                _fb_file.parent.mkdir(parents=True, exist_ok=True)

                # Extract the user query from conversation context if available
                _context_key = f"{payload.channel_id}"
                _state = self._states.get(_context_key)
                _last_query = ""
                if _state and _state.history:
                    for turn in reversed(_state.history):
                        if turn.get("role") == "user":
                            _last_query = turn.get("content", "")[:200]
                            break

                entry = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "reaction": str(payload.emoji),
                    "positive": str(payload.emoji) == "👍",
                    "message_id": str(payload.message_id),
                    "channel_id": str(payload.channel_id),
                    "guild_id": str(payload.guild_id) if payload.guild_id else None,
                    "user_id": str(payload.user_id),
                    "bot_response_preview": msg.content[:300] if msg.content else "",
                    "user_query": _last_query,
                }
                with open(_fb_file, "a", encoding="utf-8") as _f:
                    _f.write(_json.dumps(entry) + "\n")
                logger.info(
                    "[discord] feedback %s on msg %s from user %s",
                    str(payload.emoji), payload.message_id, payload.user_id,
                )
            except Exception as exc:
                logger.debug("[discord] reaction handler error: %s", exc)

    # ── Slash Commands ────────────────────────────────────────────────────────

    def _register_slash_commands(self) -> None:
        tree = self._client.tree

        @tree.command(name="ask", description="Ask a one-shot question (no conversation history)")
        async def slash_ask(interaction: discord.Interaction, prompt: str) -> None:
            await interaction.response.defer(thinking=True)
            key = _interaction_key(interaction)
            try:
                result = await asyncio.to_thread(
                    self._run_agent_sync, key, prompt, history_override=[]
                )
                response = result.get("final_response") or result.get("error") or "No response."
            except Exception as exc:
                response = f"⚠️ {exc}"
            for chunk in _chunk(response):
                await interaction.followup.send(chunk)

        @tree.command(name="reset", description="Clear conversation history for this channel")
        async def slash_reset(interaction: discord.Interaction) -> None:
            self._reset_channel(_interaction_key(interaction))
            await interaction.response.send_message("✅ History cleared.", ephemeral=True)

        @tree.command(name="new", description="Start a fresh conversation (alias for /reset)")
        async def slash_new(interaction: discord.Interaction) -> None:
            self._reset_channel(_interaction_key(interaction))
            await interaction.response.send_message("✅ New conversation started.", ephemeral=True)

        @tree.command(name="status", description="Show MYCONEX node and gateway status")
        async def slash_status(interaction: discord.Interaction) -> None:
            base_url, _, model, api_mode = self._resolve_runtime()
            provider_source = self._resolve_runtime_source()
            flash = _HERMES_DIR.parent / "flash-moe" / "metal_infer" / "infer"
            embed = discord.Embed(title="🌐 MYCONEX Status", color=0x00CFFF)
            embed.add_field(
                name="hermes-agent",
                value="✅ full tools active" if _AIAGENT_AVAILABLE else "⚠️ single-shot fallback",
                inline=True,
            )
            embed.add_field(name="flash-moe", value="✅ compiled" if flash.exists() else "⚫ macOS only", inline=True)
            embed.add_field(name="Active Sessions", value=str(len(self._states)), inline=True)
            embed.add_field(name="Model", value=f"`{model}`", inline=True)
            embed.add_field(name="API Mode", value=f"`{api_mode}`", inline=True)
            embed.add_field(name="Provider", value=f"`{provider_source}`", inline=True)
            embed.add_field(name="Endpoint", value=f"`{base_url}`", inline=False)
            if self._router:
                rs = self._router.status()
                embed.add_field(name="Mesh Tier", value=f"**{rs.get('tier', '?')}**", inline=True)
                agents_lines = []
                for ag in rs.get("agents", []):
                    icon = "🟢" if ag["state"] == "idle" else "🟡"
                    agents_lines.append(f"{icon} `{ag['name']}` ({ag['type']}) — `{ag['model']}`")
                if agents_lines:
                    embed.add_field(name="Local Agents", value="\n".join(agents_lines), inline=False)
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @tree.command(name="tools", description="List available agent tools and their status")
        async def slash_tools(interaction: discord.Interaction) -> None:
            if not _AIAGENT_AVAILABLE:
                await interaction.response.send_message(
                    "⚠️ hermes-agent not loaded — tools unavailable.\n"
                    "Install: `pip install -e integrations/hermes-agent`",
                    ephemeral=True,
                )
                return
            await interaction.response.defer(ephemeral=True, thinking=True)
            try:
                from model_tools import get_available_toolsets, check_toolset_requirements  # type: ignore[import]
                toolsets = get_available_toolsets()
                missing_map = check_toolset_requirements()
                lines = ["**Available Toolsets**\n"]
                for ts_name, info in sorted(toolsets.items()):
                    tools: List[str] = info.get("tools", [])
                    miss = missing_map.get(ts_name, [])
                    ok = "✅" if not miss else f"⚠️ missing: `{'`, `'.join(miss[:3])}`"
                    lines.append(f"**{ts_name}** {ok} — {len(tools)} tool(s)")
                    lines.append("  " + "  ".join(f"`{t}`" for t in tools[:6]))
                    if len(tools) > 6:
                        lines.append(f"  _…{len(tools) - 6} more_")
                    lines.append("")
                for chunk in _chunk("\n".join(lines)):
                    await interaction.followup.send(chunk)
            except Exception as exc:
                await interaction.followup.send(f"⚠️ Error: {exc}")

        @tree.command(name="model", description="Show the active LLM model and provider config")
        async def slash_model(interaction: discord.Interaction) -> None:
            base_url, _, model, api_mode = self._resolve_runtime()
            provider_source = self._resolve_runtime_source()
            ollama_url = self._config.get("ollama", {}).get("url", "http://localhost:11434")
            fallback_model = (
                self._config.get("hermes_moe", {})
                .get("ollama_fallback", {})
                .get("model", "llama3.1:8b")
            )
            embed = discord.Embed(title="🤖 Model Configuration", color=0x5865F2)
            embed.add_field(name="Active Model", value=f"`{model}`", inline=True)
            embed.add_field(name="API Mode", value=f"`{api_mode}`", inline=True)
            embed.add_field(name="Resolved Via", value=f"`{provider_source}`", inline=True)
            embed.add_field(name="Endpoint", value=f"`{base_url}`", inline=False)
            embed.add_field(
                name="Provider Resolution Order",
                value=(
                    "1. `~/.hermes/config.yaml` (hermes login / custom_providers)\n"
                    f"2. Nous Research API — {'✅ set' if os.getenv('NOUS_API_KEY') else '❌ `NOUS_API_KEY` not set'}\n"
                    f"3. OpenRouter API — {'✅ set' if os.getenv('OPENROUTER_API_KEY') else '❌ `OPENROUTER_API_KEY` not set'}\n"
                    f"4. Ollama `{fallback_model}` at `{ollama_url}/v1` — always available"
                ),
                inline=False,
            )
            embed.set_footer(
                text="Run `hermes login` for free Nous Research access · "
                "Add local endpoints via custom_providers in ~/.hermes/config.yaml"
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @tree.command(name="help", description="Show all MYCONEX commands and features")
        async def slash_help(interaction: discord.Interaction) -> None:
            embed = discord.Embed(
                title="🌐 MYCONEX — Command Reference",
                color=0x00CFFF,
                description=(
                    "An intelligent AI assistant running on a distributed mesh. "
                    "Just send a message to chat, or use the commands below."
                ),
            )
            embed.add_field(
                name="💬 Chat",
                value=(
                    "**Message the bot** — type normally in any enabled channel\n"
                    "**Attachments** — share images or files; they're passed to the agent\n"
                    "**DMs** — send a DM for a private conversation\n"
                    "**Threads** — each thread keeps its own history"
                ),
                inline=False,
            )
            embed.add_field(
                name="⚡ Conversation",
                value=(
                    "`/ask <prompt>` — one-shot question (no history)\n"
                    "`/reset` or `/new` — clear conversation history\n"
                    "`/context` — view conversation summary and recent turns\n"
                    "`/system` — set a custom system prompt for this channel\n"
                    "`/export` — export full conversation to a file (sent via DM)"
                ),
                inline=False,
            )
            embed.add_field(
                name="🔧 Info & Config",
                value=(
                    "`/status` — node, model, and mesh status\n"
                    "`/model` — active LLM model and provider resolution\n"
                    "`/tools` — list available agent tools and their status"
                ),
                inline=False,
            )
            embed.add_field(
                name="🖱️ Response Buttons",
                value=(
                    "Every response has interactive buttons:\n"
                    "**🔄 Regenerate** — re-run with a fresh completion\n"
                    "**➕ Continue** — ask the agent to keep going\n"
                    "**🆕 New Chat** — clear history and start fresh"
                ),
                inline=False,
            )
            embed.set_footer(
                text="MYCONEX · Distributed AI Mesh · Inspired by fungal networks"
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @tree.command(
            name="context",
            description="Show the current conversation summary for this channel",
        )
        async def slash_context(interaction: discord.Interaction) -> None:
            key = _interaction_key(interaction)
            state = self._states.get(key)
            if not state or not state.history:
                await interaction.response.send_message(
                    "No conversation history yet.", ephemeral=True
                )
                return
            turns = state.history
            user_turns = [t for t in turns if t.get("role") == "user"]
            asst_turns = [t for t in turns if t.get("role") == "assistant"]
            embed = discord.Embed(
                title="💬 Conversation Context",
                color=0x5865F2,
                description=(
                    f"**{len(user_turns)}** user turns · "
                    f"**{len(asst_turns)}** assistant turns"
                ),
            )
            if state.system_prompt:
                preview = state.system_prompt[:300]
                if len(state.system_prompt) > 300:
                    preview += "…"
                embed.add_field(
                    name="🔧 Custom System Prompt",
                    value=f"```{preview}```",
                    inline=False,
                )
            for t in turns[-6:]:
                role = t.get("role", "?")
                raw = t.get("content") or ""
                if isinstance(raw, list):
                    raw = " ".join(
                        p.get("text", "") for p in raw
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                snippet = str(raw)[:280]
                if len(str(raw)) > 280:
                    snippet += "…"
                icon = "👤" if role == "user" else "🤖"
                embed.add_field(
                    name=f"{icon} {role.capitalize()}",
                    value=snippet or "_(empty)_",
                    inline=False,
                )
            embed.set_footer(
                text=f"Max history: {MAX_HISTORY_TURNS} turns · /reset to clear"
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @tree.command(
            name="system",
            description="Set a custom system prompt for this channel (opens a text dialog)",
        )
        async def slash_system(interaction: discord.Interaction) -> None:
            key = _interaction_key(interaction)
            modal = SystemPromptModal(self, key)
            await interaction.response.send_modal(modal)

        @tree.command(
            name="export",
            description="Export the current conversation as a text file (sent via DM)",
        )
        async def slash_export(interaction: discord.Interaction) -> None:
            key = _interaction_key(interaction)
            state = self._states.get(key)
            if not state or not state.history:
                await interaction.response.send_message(
                    "No conversation history to export.", ephemeral=True
                )
                return
            lines: List[str] = []
            for t in state.history:
                role = t.get("role", "?").upper()
                content = t.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                lines.append(f"[{role}]\n{content}")
            text = "\n\n---\n\n".join(lines)
            buf = io.BytesIO(text.encode("utf-8"))
            try:
                await interaction.user.send(
                    "Here's your MYCONEX conversation export:",
                    file=discord.File(buf, filename="conversation.txt"),
                )
                await interaction.response.send_message(
                    "✅ Conversation exported — check your DMs.", ephemeral=True
                )
            except discord.Forbidden:
                await interaction.response.send_message(
                    "⚠️ Couldn't DM you. Enable DMs from server members and try again.",
                    ephemeral=True,
                )

        @tree.command(name="reflect", description="Trigger Hermes self-reflection and improvement cycle")
        async def slash_reflect(interaction: discord.Interaction) -> None:
            if not _AIAGENT_AVAILABLE:
                await interaction.response.send_message(
                    "⚠️ hermes-agent not loaded — reflection unavailable.", ephemeral=True
                )
                return
            await interaction.response.defer(thinking=True)
            agent = self._get_reflection_agent()
            if agent is None:
                await interaction.followup.send("⚠️ Could not create reflection agent.", ephemeral=True)
                return
            try:
                reflection = await asyncio.to_thread(self._improver.run_reflection, agent)
                if reflection is None:
                    await interaction.followup.send(
                        "Nothing to reflect on yet — have some conversations first.", ephemeral=True
                    )
                    return
                soul_updated = await asyncio.to_thread(self._improver.patch_soul, agent)
                lines = ["**🧠 Hermes Self-Reflection Complete**\n"]
                if summary := reflection.get("skill_summary"):
                    lines.append(f"_{summary}_\n")
                for item in reflection.get("key_learnings", [])[:5]:
                    lines.append(f"💡 {item}")
                for item in reflection.get("behaviors_to_avoid", [])[:3]:
                    lines.append(f"✗ {item}")
                for item in reflection.get("behaviors_to_reinforce", [])[:3]:
                    lines.append(f"✓ {item}")
                if soul_updated:
                    lines.append("\n_✅ SOUL.md updated with new insights._")
                lines.append(f"\n{self._improver.get_tool_stats_summary()}")
                await interaction.followup.send("\n".join(lines), ephemeral=True)
            except Exception as exc:
                await interaction.followup.send(f"⚠️ Reflection error: {exc}", ephemeral=True)

        @tree.command(name="dlam", description="Generate tasks for the Rabbit r1 DLAM desktop agent")
        async def slash_dlam(interaction: discord.Interaction) -> None:
            await interaction.response.defer(thinking=True)
            try:
                from core.gateway.self_improvement import SKILLS_FILE
                project_path = str(Path(__file__).parent.parent.parent)
                tasks = self._dlam_gen.generate_improvement_companion_tasks(
                    tool_stats=self._improver.get_tool_stats(),
                    skills_file_exists=SKILLS_FILE.exists(),
                    project_path=project_path,
                )
                queue_path = self._dlam_gen.save_queue(tasks)
                posted = self._dlam_gen.post_to_endpoint(tasks)
                msg = self._dlam_gen.format_for_discord(tasks)
                if not posted and not tasks:
                    msg = "No tasks generated."
                for chunk in _chunk(msg):
                    await interaction.followup.send(chunk, ephemeral=True)
            except Exception as exc:
                await interaction.followup.send(f"⚠️ DLAM task generation error: {exc}", ephemeral=True)

        @tree.command(name="toolstats", description="Show per-tool call success rates and latency")
        async def slash_toolstats(interaction: discord.Interaction) -> None:
            await interaction.response.send_message(
                self._improver.get_tool_stats_summary(), ephemeral=True
            )

        @tree.command(name="digest", description="Generate a knowledge digest for the past 7 days")
        async def slash_digest(interaction: discord.Interaction) -> None:
            await interaction.response.defer(thinking=True)
            try:
                from core.digest import build_digest_embed, _mark_digest_sent
                embed_data = build_digest_embed(days=7)
                embed = discord.Embed(
                    title=embed_data["title"],
                    color=embed_data.get("color", 0x00CFFF),
                    description=embed_data.get("description", ""),
                )
                for f in embed_data.get("fields", []):
                    embed.add_field(
                        name=f["name"], value=f["value"], inline=f.get("inline", False)
                    )
                if embed_data.get("footer"):
                    embed.set_footer(text=embed_data["footer"].get("text", ""))
                await interaction.followup.send(embed=embed)
            except Exception as exc:
                await interaction.followup.send(f"⚠️ Digest error: {exc}", ephemeral=True)

        @tree.command(name="feedback", description="Show 👍/👎 feedback stats for Buzlock responses")
        async def slash_feedback(interaction: discord.Interaction) -> None:
            try:
                fb_lines: List[Dict[str, Any]] = []
                if FEEDBACK_FILE.exists():
                    fb_lines = [
                        json.loads(l) for l in FEEDBACK_FILE.read_text().splitlines() if l.strip()
                    ]
                total = len(fb_lines)
                pos   = sum(1 for f in fb_lines if f.get("positive"))
                neg   = total - pos
                rate  = round(pos / total * 100) if total else 0
                embed = discord.Embed(title="🗳️ Buzlock Feedback Stats", color=0x5865F2)
                embed.add_field(name="👍 Positive", value=str(pos), inline=True)
                embed.add_field(name="👎 Negative", value=str(neg), inline=True)
                embed.add_field(name="Rate", value=f"{rate}%", inline=True)
                # Recent 5
                if fb_lines:
                    recent = fb_lines[-5:][::-1]
                    recent_text = "\n".join(
                        f"{'👍' if f.get('positive') else '👎'} _{f.get('user_query','')[:60]}_"
                        for f in recent
                    )
                    embed.add_field(name="Recent", value=recent_text or "—", inline=False)
                await interaction.response.send_message(embed=embed, ephemeral=True)
            except Exception as exc:
                await interaction.response.send_message(f"⚠️ {exc}", ephemeral=True)

    # ── Core Message Handler ──────────────────────────────────────────────────

    async def _handle_message(self, message: discord.Message) -> None:
        client = self._client
        content = message.content or ""
        channel = message.channel

        # ── Channel / thread classification ───────────────────────────────────
        is_dm = isinstance(channel, discord.DMChannel)
        is_thread = isinstance(channel, discord.Thread)
        thread_id = str(channel.id) if is_thread else None
        parent_id = (
            str(channel.parent_id)
            if is_thread and channel.parent_id
            else str(channel.id)
        )

        # ── Mention / response gating ─────────────────────────────────────────
        mentioned = client.user.mentioned_in(message)
        in_free = parent_id in self._free_channels
        in_participated = thread_id is not None and thread_id in self._bot_participated_threads

        if (
            self._require_mention
            and not is_dm
            and not mentioned
            and not in_free
            and not in_participated
        ):
            return

        # Strip @mention token from content
        if mentioned:
            content = (
                content
                .replace(f"<@{client.user.id}>", "")
                .replace(f"<@!{client.user.id}>", "")
                .strip()
            )

        if not content and not message.attachments:
            return

        # ── Access control ────────────────────────────────────────────────────
        if self._allowed_user_ids and str(message.author.id) not in self._allowed_user_ids:
            return

        # ── Auto-thread ───────────────────────────────────────────────────────
        if self._auto_thread and mentioned and not is_thread and not is_dm:
            try:
                name = (content[:77] + "…") if len(content) > 80 else content or "conversation"
                created = await channel.create_thread(name=name, message=message)
                channel = created
                thread_id = str(created.id)
                is_thread = True
            except Exception:
                pass

        # Key must reflect the *actual* channel after a possible auto-thread
        # creation above.  _msg_key() reads message.channel (the parent), so
        # when we redirected `channel` to a new thread we must derive the key
        # from the thread ID instead — otherwise all threads from the same
        # parent share one history.
        if is_thread and thread_id and str(message.channel.id) != thread_id:
            guild_id = getattr(message.guild, "id", "noguild")
            key = f"{guild_id}:{thread_id}"
        else:
            key = _msg_key(message)

        attachment_urls = [att.url for att in message.attachments]

        # ── Status indicators ─────────────────────────────────────────────────
        try:
            await message.add_reaction("👀")
        except Exception:
            pass

        # ── Track thread ──────────────────────────────────────────────────────
        if thread_id:
            self._track_thread(thread_id)

        await self._handle_prompt(
            channel=channel,
            key=key,
            content=content,
            author_id=message.author.id,
            attachment_urls=attachment_urls,
            source_message=message,
        )

    # ── Core Prompt Handler ───────────────────────────────────────────────────

    async def _handle_prompt(
        self,
        channel: "discord.abc.Messageable",
        key: str,
        content: str,
        author_id: int,
        attachment_urls: List[str] = (),
        source_message: Optional["discord.Message"] = None,
    ) -> None:
        """
        Run the agent and post the response.

        Called from on_message (via _handle_message) and from ResponseView button
        callbacks.  source_message is the original Discord Message; when present,
        reactions are updated on it.  When absent (button callback), reactions are
        skipped.
        """
        loop = asyncio.get_running_loop()
        status_msg: Optional["discord.Message"] = None
        error: Optional[str] = None
        response: Optional[str] = None

        # ── Concurrency guard ─────────────────────────────────────────────────
        if key in self._in_flight:
            try:
                await channel.send("⏳ Still working on your previous message…")
            except Exception:
                pass
            return
        self._in_flight.add(key)

        try:
            async with channel.typing():
                try:
                    status_msg = await channel.send("⏳ thinking…")
                except Exception:
                    pass

                if _AIAGENT_AVAILABLE:
                    try:
                        result = await self._run_with_hermes(
                            key=key,
                            content=content,
                            attachment_urls=list(attachment_urls),
                            status_msg=status_msg,
                            loop=loop,
                            channel=channel,
                            author_id=author_id,
                        )
                        if result.get("failed") or result.get("error"):
                            error = result.get("error") or "Agent returned an error."
                        else:
                            response = result.get("final_response") or ""
                    except Exception as exc:
                        logger.exception("[discord] hermes agent error on channel %s", key)
                        error = _classify_agent_error(exc)
                else:
                    # ── RLMAgent path (preferred) / TaskRouter fallback ───────
                    if self._rlm_agent is not None:
                        # Full RLM pipeline: context management, delegation,
                        # memory injection, and Discord-formatted response.
                        rlm_author_id  = str(source_message.author.id)  if source_message and hasattr(source_message, "author")  else str(author_id)
                        rlm_channel_id = str(source_message.channel.id) if source_message and hasattr(source_message, "channel") else key
                        rlm_guild_id   = str(source_message.guild.id)   if (source_message and hasattr(source_message, "guild") and source_message.guild) else None
                        try:
                            response = await self._rlm_agent.on_discord_message(
                                message_content=content,
                                author_id=rlm_author_id,
                                channel_id=rlm_channel_id,
                                guild_id=rlm_guild_id,
                            )
                        except Exception as exc:
                            logger.exception("[discord] RLMAgent.on_discord_message error")
                            error = str(exc)
                    elif self._router:
                        # Plain TaskRouter fallback (non-RLM path)
                        state = self._get_or_create_state(key)
                        if state.legacy_ctx is None:
                            state.legacy_ctx = AgentContext()
                        ctx = state.legacy_ctx
                        if len(ctx.history) > 30:
                            ctx.trim(max_turns=30)
                        try:
                            res = await self._router.route("chat", {"prompt": content}, context=ctx)
                            if res.success:
                                response = (res.output or {}).get("response", "")
                            else:
                                error = res.error
                        except Exception as exc:
                            error = str(exc)
                    else:
                        error = "No agent backend configured. Set NOUS_API_KEY or OPENROUTER_API_KEY in .env."

            # ── Update source message reactions ───────────────────────────────
            if source_message and self._client:
                try:
                    await source_message.remove_reaction("👀", self._client.user)
                    await source_message.add_reaction("❌" if error else "✅")
                except Exception:
                    pass

            # ── Error path ────────────────────────────────────────────────────
            if error:
                err_text = f"⚠️ {_truncate(error)}"
                if status_msg:
                    await _safe_edit(status_msg, err_text)
                else:
                    try:
                        await channel.send(err_text)
                    except Exception:
                        pass
                return

            if not response:
                if status_msg:
                    await _safe_edit(status_msg, "_(no response)_")
                return

            # ── Build interactive view ─────────────────────────────────────────
            view: Optional["discord.ui.View"] = None
            if _DISCORD_AVAILABLE:
                view = ResponseView(self, key, content)

            # ── Very long response → upload as file with inline preview ──────
            if len(response) > 3800:
                preview = response[:800] + "\n\n_… (full response attached as file above)_"
                if status_msg:
                    await _safe_edit(status_msg, preview)
                else:
                    try:
                        await channel.send(preview)
                    except Exception:
                        pass
                try:
                    await channel.send(
                        file=discord.File(
                            io.BytesIO(response.encode("utf-8")),
                            filename="response.md",
                        ),
                        view=view,
                    )
                except Exception:
                    pass
                return

            # ── Normal response: edit status msg, attach interactive view ─────
            chunks = _chunk(response)
            if status_msg:
                try:
                    await status_msg.edit(content=chunks[0], view=view)
                except Exception:
                    await _safe_edit(status_msg, chunks[0])
            else:
                try:
                    await channel.send(chunks[0], view=view)
                except Exception:
                    pass
            for extra in chunks[1:]:
                try:
                    await channel.send(extra)
                except Exception:
                    pass

        finally:
            self._in_flight.discard(key)

    # ── Hermes Agent Runner ───────────────────────────────────────────────────

    async def _run_with_hermes(
        self,
        key: str,
        content: str,
        attachment_urls: List[str],
        status_msg: Optional["discord.Message"],
        loop: asyncio.AbstractEventLoop,
        channel: "discord.abc.Messageable",
        author_id: int,
    ) -> Dict[str, Any]:
        """
        Execute AIAgent.run_conversation() inside asyncio.to_thread().

        Wires up:
          • Streaming updater   — edits status_msg as tokens arrive
          • Tool progress cb    — updates status_msg with active tool name
          • Clarify callback    — asks the user a question via Discord and waits
          • Attachment context  — appends URLs to the user message
        """
        state = self._get_or_create_state(key)
        agent = self._get_or_create_agent(key, state, channel, author_id, loop)

        # Streaming live-edit
        streamer = _StreamingUpdater(status_msg, loop) if status_msg else None

        # Tool progress (called synchronously from the worker thread)
        if status_msg:
            def _tool_cb(tool_name: str, args_preview: str) -> None:
                text = f"🔧 **{tool_name}**  `{(args_preview or '')[:90]}`"
                asyncio.run_coroutine_threadsafe(_safe_edit(status_msg, text), loop)
            state.tool_cb = _tool_cb
        else:
            state.tool_cb = None

        # Append attachment URLs so the agent can act on them
        user_message = content
        if attachment_urls:
            urls = "\n".join(f"  • {u}" for u in attachment_urls)
            user_message = f"{content}\n\n[Attached files/images]\n{urls}".strip()

        # Inject accumulated skills from past reflections into the system prompt
        base_prompt = state.system_prompt or _SYSTEM_PROMPT
        system_prompt = base_prompt + self._improver.get_skills_injection()

        # Inject lessons learned from corrections (lessons.md) — these are
        # hard behavioral rules that must be followed.
        lessons = _load_lessons()
        if lessons:
            system_prompt = system_prompt + "\n\n" + lessons

        # Pre-load stored memories so the agent knows facts without needing a tool call.
        # Injected before RAG so personal facts appear first.
        mem_ctx = _load_memory_for_prompt()
        if mem_ctx:
            system_prompt = system_prompt + "\n\n" + mem_ctx

        # Auto-RAG: prepend relevant knowledge base context
        rag_block = await _rag_context(content)
        if rag_block:
            system_prompt = system_prompt + "\n\n" + rag_block

        # Inject feedback summary (updated every call — cheap file read)
        fb_summary = _load_feedback_summary()
        if fb_summary:
            system_prompt = system_prompt + "\n" + fb_summary

        result: Dict[str, Any] = await asyncio.to_thread(
            agent.run_conversation,
            user_message=user_message,
            system_message=system_prompt,
            conversation_history=list(state.history),
            stream_callback=streamer.on_delta if streamer else None,
        )

        # Flush any remaining buffered tokens that hadn't reached the edit
        # interval yet — ensures the final partial chunk is always displayed.
        if streamer:
            streamer._push()

        # Persist the updated history for the next turn.
        # Trim by user-turn count (not raw message count) so agentic tool-call
        # messages don't silently eat into the effective conversation window.
        messages = result.get("messages") or []
        if messages:
            state.history = _trim_history_to_turns(messages, MAX_HISTORY_TURNS)
            _history_save(key, state.history)

        # Record conversation for self-improvement; trigger reflection if due
        self._improver.record_conversation(messages)
        if self._improver.should_reflect() and _AIAGENT_AVAILABLE:
            asyncio.create_task(self._run_background_reflection())

        return result

    def _run_agent_sync(
        self,
        key: str,
        content: str,
        history_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous path used by slash commands via asyncio.to_thread().
        Pass history_override=[] for one-shot /ask (no history).
        """
        state = self._get_or_create_state(key)
        agent = self._agents.get(key)
        if agent is None:
            # No agent yet; can't create one without channel — fall back gracefully
            return {"final_response": "No active session. Send a message first, then use /ask."}
        history = history_override if history_override is not None else list(state.history)
        sync_prompt = _SYSTEM_PROMPT
        mem_ctx = _load_memory_for_prompt()
        if mem_ctx:
            sync_prompt = sync_prompt + "\n\n" + mem_ctx
        fb_summary = _load_feedback_summary()
        if fb_summary:
            sync_prompt = sync_prompt + "\n" + fb_summary
        result = agent.run_conversation(
            user_message=content,
            system_message=sync_prompt,
            conversation_history=history,
        )
        if history_override is None:
            messages = result.get("messages") or []
            if messages:
                state.history = _trim_history_to_turns(messages, MAX_HISTORY_TURNS)
                _history_save(key, state.history)
        return result

    # ── Self-improvement ──────────────────────────────────────────────────────

    async def _run_background_reflection(self) -> None:
        """
        Run a full reflection + SOUL.md patch cycle in a background thread
        using a dedicated agent that doesn't share state with any conversation.
        Errors are logged but never surface to users.
        """
        try:
            agent = self._get_reflection_agent()
            if agent is None:
                return
            logger.info("[self_improve] starting background reflection")
            reflection = await asyncio.to_thread(self._improver.run_reflection, agent)
            if reflection:
                await asyncio.to_thread(self._improver.patch_soul, agent)
                logger.info("[self_improve] background reflection + soul patch complete")
        except Exception as exc:
            logger.warning("[self_improve] background reflection error: %s", exc)

    def _get_reflection_agent(self) -> Optional["AIAgent"]:  # type: ignore[return]
        """
        Lazily create a dedicated AIAgent for self-reflection.
        Uses a smaller max_iterations and no tool/clarify callbacks so it
        doesn't interfere with live conversations.
        """
        if self._refl_agent is not None:
            return self._refl_agent
        if not _AIAGENT_AVAILABLE:
            return None
        try:
            base_url, api_key, model, api_mode = self._resolve_runtime()
            self._refl_agent = AIAgent(  # type: ignore[call-arg]
                base_url=base_url,
                api_key=api_key,
                model=model,
                api_mode=api_mode,
                platform="reflection",
                quiet_mode=True,
                skip_context_files=True,   # reflection prompt is self-contained
                max_iterations=8,
                enabled_toolsets=[],       # no tools needed for reflection
            )
        except Exception as exc:
            logger.warning("[self_improve] could not create reflection agent: %s", exc)
        return self._refl_agent

    # ── State & Agent Management ──────────────────────────────────────────────

    def _get_or_create_state(self, key: str) -> _ChannelState:
        if key not in self._states:
            state = _ChannelState()
            saved = _history_load(key)
            if saved:
                state.history = _trim_history_to_turns(saved, MAX_HISTORY_TURNS)
                logger.debug("[discord] restored %d messages (%d user turns) for %s",
                             len(state.history),
                             sum(1 for m in state.history if m.get("role") == "user"),
                             key)
            self._states[key] = state
        return self._states[key]

    def _get_or_create_agent(
        self,
        key: str,
        state: _ChannelState,
        channel: "discord.abc.Messageable",
        author_id: int,
        loop: asyncio.AbstractEventLoop,
    ) -> "AIAgent":  # type: ignore[return]
        """
        Lazily create one AIAgent per channel key.

        tool_progress_callback and clarify_callback both close over mutable
        slots on state so each message can redirect callbacks without
        recreating the agent.
        """
        # Always refresh the clarify instance (cheap) so it has the right channel
        state.clarify = _DiscordClarify(channel, author_id, loop)

        if key not in self._agents:
            base_url, api_key, model, api_mode = self._resolve_runtime()
            self._agents[key] = AIAgent(  # type: ignore[call-arg]
                base_url=base_url,
                api_key=api_key,
                model=model,
                api_mode=api_mode,
                platform="discord",
                quiet_mode=True,
                skip_context_files=False,  # loads ~/.hermes/SOUL.md + accumulated skills
                max_iterations=30,
                # Focused toolset: removes skills+session_search guidance text
                # that confuses small local models into describing vs executing.
                enabled_toolsets=_DISCORD_TOOLSETS,
                # Injected at every API call (including mid-loop post-tool turns)
                # so the model stays in execution mode rather than describing.
                ephemeral_system_prompt=_EPHEMERAL_PROMPT,
                tool_progress_callback=(
                    lambda tn, ap, *_: state.tool_cb(tn, ap) if state.tool_cb else None
                ),
                clarify_callback=(
                    lambda q, c=None: state.clarify(q, c) if state.clarify else q
                ),
            )
            logger.debug("[discord] new AIAgent for %s (model=%s api_mode=%s)", key, model, api_mode)
        else:
            # Agent exists but clarify_callback must point at fresh instance
            self._agents[key].clarify_callback = state.clarify
        return self._agents[key]

    def _reset_channel(self, key: str) -> None:
        """Destroy all state and the AIAgent for a channel."""
        self._states.pop(key, None)
        self._agents.pop(key, None)
        logger.debug("[discord] channel %s reset", key)

    # ── Provider Resolution ───────────────────────────────────────────────────

    def _resolve_runtime(self) -> tuple[str, str, str, str]:
        """
        Return (base_url, api_key, model, api_mode) for AIAgent construction.

        Resolution order:
          1. hermes-agent's own provider system (~/.hermes/config.yaml):
               - `hermes login`         → Nous Research OAuth2 (free Hermes access)
               - custom_providers       → any local endpoint (Ollama, vLLM, llama.cpp)
               - HERMES_INFERENCE_PROVIDER env → provider override
          2. NOUS_API_KEY env var        → Nous Research inference API
          3. OPENROUTER_API_KEY env var  → OpenRouter
          4. Ollama /v1                  → always available
        """
        # 1. Try hermes-agent's config-driven resolution
        if _AIAGENT_AVAILABLE:
            try:
                from hermes_cli.config import load_config as _load_hermes_config  # type: ignore[import]
                from hermes_cli.runtime_provider import resolve_runtime_provider  # type: ignore[import]

                runtime = resolve_runtime_provider()
                base_url: str = runtime.get("base_url", "").rstrip("/")
                api_key: str = runtime.get("api_key", "")
                api_mode: str = runtime.get("api_mode", "chat_completions") or "chat_completions"

                # Resolve model: prefer config default, else provider-specific default
                hermes_cfg = _load_hermes_config()
                model_cfg = hermes_cfg.get("model", {})
                if isinstance(model_cfg, dict):
                    model: str = (model_cfg.get("default", "") or "").strip()
                elif isinstance(model_cfg, str):
                    model = model_cfg.strip()
                else:
                    model = ""

                if not model:
                    provider = runtime.get("provider", "")
                    if provider == "nous":
                        model = "NousResearch/Hermes-3-Llama-3.1-70B"
                    elif provider in ("openrouter", ""):
                        model = "nousresearch/hermes-3-llama-3.1-70b"
                    else:
                        model = "NousResearch/Hermes-3-Llama-3.1-70B"

                if base_url and api_key:
                    logger.debug(
                        "[discord] resolved provider via hermes config: %s model=%s api_mode=%s",
                        runtime.get("source", "?"), model, api_mode,
                    )
                    return (base_url, api_key, model, api_mode)
            except Exception as exc:
                logger.debug("[discord] hermes provider resolution failed: %s", exc)

        # 2. NOUS_API_KEY env var
        nous_key = os.getenv("NOUS_API_KEY", "")
        if nous_key:
            return (
                "https://inference-api.nousresearch.com/v1",
                nous_key,
                "NousResearch/Hermes-3-Llama-3.1-70B",
                "chat_completions",
            )

        # 3. OPENROUTER_API_KEY env var
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if or_key:
            return (
                "https://openrouter.ai/api/v1",
                or_key,
                "nousresearch/hermes-3-llama-3.1-70b",
                "chat_completions",
            )

        # 4. Ollama /v1 fallback
        ollama_base = self._config.get("ollama", {}).get("url", "http://localhost:11434")
        fallback_model = (
            self._config.get("hermes_moe", {})
            .get("ollama_fallback", {})
            .get("model", "llama3.1:8b")
        )
        return (f"{ollama_base}/v1", "ollama", fallback_model, "chat_completions")

    def _resolve_runtime_source(self) -> str:
        """Return a human-readable label for how the current provider was resolved."""
        if _AIAGENT_AVAILABLE:
            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider  # type: ignore[import]
                runtime = resolve_runtime_provider()
                source = runtime.get("source", "")
                provider = runtime.get("provider", "")
                if runtime.get("base_url") and runtime.get("api_key"):
                    return f"{provider} ({source})" if source else provider
            except Exception:
                pass
        if os.getenv("NOUS_API_KEY"):
            return "NOUS_API_KEY env"
        if os.getenv("OPENROUTER_API_KEY"):
            return "OPENROUTER_API_KEY env"
        return "ollama fallback"

    # ── Thread Tracking ───────────────────────────────────────────────────────

    def _track_thread(self, thread_id: str) -> None:
        self._bot_participated_threads.add(thread_id)
        if len(self._bot_participated_threads) <= MAX_TRACKED_THREADS:
            self._save_thread_state()

    def _load_thread_state(self) -> set[str]:
        try:
            if THREAD_STATE_FILE.exists():
                return set(json.loads(THREAD_STATE_FILE.read_text()).get("threads", []))
        except Exception:
            pass
        return set()

    def _save_thread_state(self) -> None:
        try:
            THREAD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            threads = list(self._bot_participated_threads)[-MAX_TRACKED_THREADS:]
            THREAD_STATE_FILE.write_text(json.dumps({"threads": threads}))
        except Exception:
            pass


# ─── Module-Level Helpers ─────────────────────────────────────────────────────

def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def _msg_key(message: "discord.Message") -> str:
    """Unique key per Discord channel or DM conversation."""
    ch = message.channel
    if isinstance(ch, discord.DMChannel):
        return f"dm:{message.author.id}"
    guild_id = getattr(message.guild, "id", "noguild")
    return f"{guild_id}:{ch.id}"


def _interaction_key(interaction: "discord.Interaction") -> str:
    ch = interaction.channel
    if isinstance(ch, discord.DMChannel):
        return f"dm:{interaction.user.id}"
    guild_id = getattr(interaction.guild, "id", "noguild")
    return f"{guild_id}:{ch.id}"


def _chunk(text: str, limit: int = MAX_MSG_LEN) -> List[str]:
    """Split text into ≤ limit-char chunks, preferring newline boundaries."""
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


def _truncate(text: str, limit: int = MAX_MSG_LEN) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def _safe_edit(message: "discord.Message", content: str) -> None:
    """Edit a Discord message, silently ignoring rate-limit and other errors."""
    try:
        await message.edit(content=_truncate(content))
    except Exception:
        pass


def _classify_agent_error(exc: Exception) -> str:
    """
    Translate raw agent/Ollama exceptions into user-friendly Discord messages.

    Catches the most common failure modes so users see actionable text instead
    of Python tracebacks.
    """
    msg = str(exc)
    low = msg.lower()

    # Ollama / local endpoint not reachable
    if any(k in low for k in ("connection refused", "connect error", "connecterror",
                               "cannot connect", "connection reset", "network unreachable")):
        return (
            "Cannot reach the Ollama endpoint. "
            "Is Ollama running? (`ollama serve` / `systemctl start ollama`)"
        )

    # Model not pulled
    if "model" in low and any(k in low for k in ("not found", "does not exist", "pull")):
        return (
            "The requested model isn't available locally. "
            "Run `ollama pull llama3.1:8b` and try again."
        )

    # Context / token limit exceeded
    if any(k in low for k in ("context length", "token limit", "maximum context",
                               "prompt is too long")):
        return "The conversation is too long for this model. Use `/reset` to start a fresh session."

    # Timeout
    if any(k in low for k in ("timed out", "timeout", "deadline exceeded")):
        return "The model took too long to respond. Try a shorter prompt or use `/reset`."

    # Generic fallback — include the raw text so it's still debuggable
    return f"Agent error: {msg}"
