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

Optional env overrides:
    DISCORD_REQUIRE_MENTION         — "true" to only respond when @mentioned
    DISCORD_FREE_RESPONSE_CHANNELS  — comma-separated channel IDs (no mention needed)
    DISCORD_AUTO_THREAD             — "true" to auto-create a thread per @mention
    DISCORD_ALLOW_BOTS              — "none" | "mentions" | "all"
    DISCORD_ALLOWED_USERS           — comma-separated Discord user IDs (empty = all)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
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

# ── MYCONEX internals ─────────────────────────────────────────────────────────
from orchestration.agents.base_agent import AgentContext
from orchestration.workflows.task_router import TaskRouter

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_MSG_LEN = 2000          # Discord hard character limit
MAX_HISTORY_TURNS = 50      # Conversation turns kept per channel (user+assistant pairs)
STREAM_EDIT_INTERVAL = 1.2  # Min seconds between message edits while streaming
MAX_TRACKED_THREADS = 500   # Thread IDs persisted across restarts
THREAD_STATE_FILE = Path.home() / ".myconex" / "discord_threads.json"

_SYSTEM_PROMPT = (
    "You are MYCONEX, an intelligent assistant running on a distributed AI mesh system "
    "inspired by fungal mycelium networks. You are accessed through Discord.\n\n"
    "You have access to powerful tools: web search, code execution, file operations, "
    "memory, image generation, and more. Use them proactively when they would genuinely "
    "help the user rather than guessing.\n\n"
    "Discord renders standard Markdown: **bold**, *italic*, `code`, ```code blocks```, "
    "> quotes, and — lists. Prefer concise responses but be thorough when the task demands it.\n\n"
    "The mesh node you are running on is part of the MYCONEX distributed system. "
    "Tasks can be routed to specialised nodes (T1 apex → T4 edge) for inference, "
    "embedding, search, and training."
)


# ─── Channel State ────────────────────────────────────────────────────────────

class _ChannelState:
    """
    Per-channel conversation state.

    Holds the OpenAI-format message history and a mutable tool_cb slot so we
    can redirect tool-progress updates to the right Discord message on each
    request without recreating the AIAgent.
    """

    __slots__ = ("history", "tool_cb", "legacy_ctx")

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []
        self.tool_cb: Optional[Callable[..., None]] = None
        self.legacy_ctx: Optional[AgentContext] = None  # TaskRouter fallback only


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
            try:
                synced = await client.tree.sync()
                logger.info("[discord] synced %d slash commands", len(synced))
            except Exception as exc:
                logger.warning("[discord] slash command sync failed: %s", exc)
            self._running = True
            self._ready.set()

        @client.event
        async def on_message(message: discord.Message) -> None:
            if message.author == client.user:
                return
            if message.author.bot:
                if self._allow_bots == "none":
                    return
                if self._allow_bots == "mentions" and not client.user.mentioned_in(message):
                    return
            await self._handle_message(message)

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
            base_url, _, model = self._resolve_runtime()
            flash = _HERMES_DIR.parent / "flash-moe" / "metal_infer" / "infer"
            lines = [
                "**MYCONEX Status**",
                f"• hermes-agent: {'✅ loaded — full tools active' if _AIAGENT_AVAILABLE else '⚠️ not loaded — single-shot fallback'}",
                f"• model: `{model}`",
                f"• endpoint: `{base_url}`",
                f"• Nous API key: {'✅' if os.getenv('NOUS_API_KEY') else '❌ not set'}",
                f"• OpenRouter key: {'✅' if os.getenv('OPENROUTER_API_KEY') else '❌ not set'}",
                f"• flash-moe binary: {'✅ compiled' if flash.exists() else '⚫ not compiled (macOS only)'}",
                f"• active sessions: {len(self._states)}",
            ]
            if self._router:
                rs = self._router.status()
                lines.append(f"• mesh tier: **{rs.get('tier', '?')}**")
                for ag in rs.get("agents", []):
                    icon = "🟢" if ag["state"] == "idle" else "🟡"
                    lines.append(f"  {icon} `{ag['name']}` ({ag['type']}) — `{ag['model']}`")
            await interaction.response.send_message("\n".join(lines), ephemeral=True)

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
            base_url, _, model = self._resolve_runtime()
            ollama_url = self._config.get("ollama", {}).get("url", "http://localhost:11434")
            fallback_model = (
                self._config.get("hermes_moe", {})
                .get("ollama_fallback", {})
                .get("model", "llama3.1:8b")
            )
            lines = [
                "**Model Configuration**",
                f"• active: `{model}` → `{base_url}`",
                "",
                "**Provider Priority**",
                f"1. Nous Research `Hermes-3-Llama-3.1-70B` — {'✅' if os.getenv('NOUS_API_KEY') else '❌ NOUS_API_KEY not set'}",
                f"2. OpenRouter `nousresearch/hermes-3-llama-3.1-70b` — {'✅' if os.getenv('OPENROUTER_API_KEY') else '❌ OPENROUTER_API_KEY not set'}",
                f"3. Ollama `{fallback_model}` at `{ollama_url}/v1` — always available",
            ]
            await interaction.response.send_message("\n".join(lines), ephemeral=True)

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

        key = _msg_key(message)
        attachment_urls = [att.url for att in message.attachments]

        # ── Status indicators ─────────────────────────────────────────────────
        try:
            await message.add_reaction("👀")
        except Exception:
            pass

        loop = asyncio.get_event_loop()
        status_msg: Optional[discord.Message] = None
        error: Optional[str] = None
        response: Optional[str] = None

        async with channel.typing():
            # Post a placeholder message we'll update live as the agent works
            try:
                status_msg = await channel.send("⏳ thinking…")
            except Exception:
                pass

            if _AIAGENT_AVAILABLE:
                try:
                    result = await self._run_with_hermes(
                        key=key,
                        content=content,
                        attachment_urls=attachment_urls,
                        status_msg=status_msg,
                        loop=loop,
                    )
                    if result.get("failed") or result.get("error"):
                        error = result.get("error") or "Agent returned an error."
                    else:
                        response = result.get("final_response") or ""
                except Exception as exc:
                    logger.exception("[discord] hermes agent error on channel %s", key)
                    error = str(exc)
            else:
                # ── TaskRouter fallback ───────────────────────────────────────
                state = self._get_or_create_state(key)
                if state.legacy_ctx is None:
                    state.legacy_ctx = AgentContext()
                ctx = state.legacy_ctx
                if len(ctx.history) > 30:
                    ctx.trim(max_turns=30)
                if self._router:
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

        # ── Track thread ──────────────────────────────────────────────────────
        if thread_id:
            self._track_thread(thread_id)

        # ── Update status message → final response (no delete/resend needed) ──
        if error:
            try:
                await message.remove_reaction("👀", client.user)
                await message.add_reaction("❌")
            except Exception:
                pass
            err_text = f"⚠️ {_truncate(error)}"
            if status_msg:
                await _safe_edit(status_msg, err_text)
            else:
                await channel.send(err_text)
            return

        try:
            await message.remove_reaction("👀", client.user)
            await message.add_reaction("✅")
        except Exception:
            pass

        if not response:
            if status_msg:
                await _safe_edit(status_msg, "_(no response)_")
            return

        chunks = _chunk(response)
        # Edit the status message with the first chunk — seamless transition from "thinking…"
        if status_msg:
            await _safe_edit(status_msg, chunks[0])
        else:
            await channel.send(chunks[0])
        # Additional chunks become separate follow-up messages
        for extra in chunks[1:]:
            await channel.send(extra)

    # ── Hermes Agent Runner ───────────────────────────────────────────────────

    async def _run_with_hermes(
        self,
        key: str,
        content: str,
        attachment_urls: List[str],
        status_msg: Optional["discord.Message"],
        loop: asyncio.AbstractEventLoop,
    ) -> Dict[str, Any]:
        """
        Execute AIAgent.run_conversation() inside asyncio.to_thread().

        Wires up:
          • Streaming updater — edits status_msg as tokens arrive
          • Tool progress cb  — updates status_msg with active tool name
          • Attachment context — appends URLs to the user message
        """
        state = self._get_or_create_state(key)
        agent = self._get_or_create_agent(key, state)

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

        result: Dict[str, Any] = await asyncio.to_thread(
            agent.run_conversation,
            user_message=user_message,
            system_message=_SYSTEM_PROMPT,
            conversation_history=list(state.history),
            stream_callback=streamer.on_delta if streamer else None,
        )

        # Persist the updated history for the next turn
        messages = result.get("messages") or []
        if messages:
            state.history = messages[-(MAX_HISTORY_TURNS * 2):]

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
        agent = self._get_or_create_agent(key, state)
        history = history_override if history_override is not None else list(state.history)
        result = agent.run_conversation(
            user_message=content,
            system_message=_SYSTEM_PROMPT,
            conversation_history=history,
        )
        if history_override is None:
            messages = result.get("messages") or []
            if messages:
                state.history = messages[-(MAX_HISTORY_TURNS * 2):]
        return result

    # ── State & Agent Management ──────────────────────────────────────────────

    def _get_or_create_state(self, key: str) -> _ChannelState:
        if key not in self._states:
            self._states[key] = _ChannelState()
        return self._states[key]

    def _get_or_create_agent(self, key: str, state: _ChannelState) -> "AIAgent":  # type: ignore[return]
        """
        Lazily create one AIAgent per channel key.

        tool_progress_callback closes over state.tool_cb (a mutable slot) so
        each message can redirect tool output to its own status_msg without
        recreating the agent.
        """
        if key not in self._agents:
            base_url, api_key, model = self._resolve_runtime()
            self._agents[key] = AIAgent(  # type: ignore[call-arg]
                base_url=base_url,
                api_key=api_key,
                model=model,
                platform="discord",
                quiet_mode=True,
                skip_context_files=True,
                max_iterations=30,
                tool_progress_callback=(
                    lambda tn, ap: state.tool_cb(tn, ap) if state.tool_cb else None
                ),
            )
            logger.debug("[discord] new AIAgent for %s (model=%s)", key, model)
        return self._agents[key]

    def _reset_channel(self, key: str) -> None:
        """Destroy all state and the AIAgent for a channel."""
        self._states.pop(key, None)
        self._agents.pop(key, None)
        logger.debug("[discord] channel %s reset", key)

    # ── Provider Resolution ───────────────────────────────────────────────────

    def _resolve_runtime(self) -> tuple[str, str, str]:
        """
        Return (base_url, api_key, model) for AIAgent.

        Priority: Nous Research API → OpenRouter → Ollama /v1.
        Ollama ≥ 0.1.14 exposes an OpenAI-compatible endpoint at /v1.
        """
        nous_key = os.getenv("NOUS_API_KEY", "")
        if nous_key:
            return (
                "https://inference-api.nousresearch.com/v1",
                nous_key,
                "NousResearch/Hermes-3-Llama-3.1-70B",
            )
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if or_key:
            return (
                "https://openrouter.ai/api/v1",
                or_key,
                "nousresearch/hermes-3-llama-3.1-70b",
            )
        ollama_base = self._config.get("ollama", {}).get("url", "http://localhost:11434")
        fallback_model = (
            self._config.get("hermes_moe", {})
            .get("ollama_fallback", {})
            .get("model", "llama3.1:8b")
        )
        return (f"{ollama_base}/v1", "ollama", fallback_model)

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
