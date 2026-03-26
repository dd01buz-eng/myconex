"""
MYCONEX — Unified Entry Point
==============================
python -m myconex [--mode MODE] [--config PATH] [OPTIONS]

Modes:
  cli         Interactive REPL backed by a live RLMAgent
  discord     Mesh node + Discord gateway (same as main.py --mode discord)
  api         Mesh node + REST API gateway
  autonomous  Self-improving autonomous loop — MYCONEX runs itself
  worker      Background mesh worker, no interactive interface (default)
  full        All of the above simultaneously

In autonomous mode MYCONEX:
  1. Boots RLMAgent + agent roster
  2. Loads a task queue from ~/.myconex/autonomous_tasks.json
  3. Executes tasks, stores results in persistent memory
  4. Runs the self-optimization loop after every N completions
  5. Self-generates new exploration tasks when the queue is empty
  6. Writes a session digest to ~/.myconex/session_YYYYMMDD.md on exit

This module is the single authoritative entry point; main.py is preserved for
backward compatibility with systemd / existing scripts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ── Optional rich console (degrade gracefully) ────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text
    _RICH = True
except ImportError:
    _RICH = False
    class Console:  # type: ignore[no-redef]
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("─" * 60)

console = Console()

logging.basicConfig(
    level=logging.WARNING,   # suppress library noise in interactive modes
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("myconex")

VERSION = "0.2.0"
_MYCONEX_DIR = Path.home() / ".myconex"
_TASK_QUEUE_FILE = _MYCONEX_DIR / "autonomous_tasks.json"


# ─── Utilities ────────────────────────────────────────────────────────────────

def _info(msg: str) -> None:
    if _RICH:
        console.print(f"[cyan]{msg}[/cyan]")
    else:
        print(msg)

def _success(msg: str) -> None:
    if _RICH:
        console.print(f"[green]{msg}[/green]")
    else:
        print(msg)

def _warn(msg: str) -> None:
    if _RICH:
        console.print(f"[yellow]{msg}[/yellow]")
    else:
        print(f"WARNING: {msg}")

def _error(msg: str) -> None:
    if _RICH:
        console.print(f"[red]{msg}[/red]")
    else:
        print(f"ERROR: {msg}", file=sys.stderr)

def _banner() -> None:
    if _RICH:
        from rich.text import Text
        console.print(Text("""
 ███╗   ███╗██╗   ██╗ ██████╗ ██████╗ ███╗   ██╗███████╗██╗  ██╗
 ████╗ ████║╚██╗ ██╔╝██╔════╝██╔═══██╗████╗  ██║██╔════╝╚██╗██╔╝
 ██╔████╔██║ ╚████╔╝ ██║     ██║   ██║██╔██╗ ██║█████╗   ╚███╔╝
 ██║╚██╔╝██║  ╚██╔╝  ██║     ██║   ██║██║╚██╗██║██╔══╝   ██╔██╗
 ██║ ╚═╝ ██║   ██║   ╚██████╗╚██████╔╝██║ ╚████║███████╗██╔╝ ██╗
 ╚═╝     ╚═╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝
""", style="bold cyan"))
        console.print(f"[dim] Distributed AI Mesh System v{VERSION} — Recursive Language Model Engine[/dim]\n")
    else:
        print(f"\nMYCONEX v{VERSION} — Distributed AI Mesh\n")


def _load_config(config_path: Optional[str]) -> dict:
    """Delegate to main.py config loader."""
    sys.path.insert(0, str(Path(__file__).parent))
    from main import load_config  # type: ignore[import]
    return load_config(config_path)


async def _discover_services(verbose: bool = False) -> tuple[Optional[Any], Optional[Any]]:
    """
    Run mDNS service discovery and update the global config in-place.

    Loads a typed MyconexConfig, runs mDNS advertisement and resolution, and
    applies discovered URLs back to the global config singleton so all service
    clients pick them up.

    Returns (watcher, advertiser) — keep both references for shutdown cleanup.
    Returns (None, None) if discovery is unavailable or not needed.
    """
    try:
        from config import load_config, apply_discovered_urls
        from core.discovery.mesh_discovery import (
            ServiceAdvertiser,
            resolve_service_urls,
            ServiceDiscoveryError,
        )
        from zeroconf.asyncio import AsyncZeroconf
    except ImportError as e:
        logger.debug(f"Service discovery unavailable: {e}")
        return None, None

    try:
        cfg = load_config()
        zc = AsyncZeroconf()

        # Advertise local services (hub nodes will register; worker nodes probe nothing)
        advertiser = ServiceAdvertiser(zc=zc, node_name=cfg.mesh.node_name or "node", cfg=cfg)
        await advertiser.start()

        # Discover hub services
        urls, watcher = await resolve_service_urls(zc=zc, cfg=cfg, timeout=10.0)
        apply_discovered_urls(cfg, urls)

        if verbose:
            logger.info(f"[discovery] NATS:   {cfg.mesh.nats_url}")
            logger.info(f"[discovery] Redis:  {cfg.mesh.redis_url}")
            logger.info(f"[discovery] Qdrant: {cfg.mesh.qdrant_url}")

        return watcher, advertiser

    except Exception as e:
        # Check if it's a ServiceDiscoveryError (imported locally so check by name)
        if type(e).__name__ == "ServiceDiscoveryError":
            logger.error(str(e))
            sys.exit(1)
        logger.debug(f"Service discovery skipped: {e}")
        return None, None


async def _input_async(prompt: str = "") -> str:
    """Non-blocking stdin read — runs input() in a thread so the event loop stays live."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, input, prompt)


# ─── RLMAgent Bootstrap ───────────────────────────────────────────────────────

async def boot_rlm_agent(config: dict, verbose: bool = False):
    """
    Bootstrap the RLMAgent with all components wired.

    Returns:
        (agent, router) tuple — both started and ready for dispatch.
    """
    from orchestration.workflows.task_router import TaskRouter
    from orchestration.agents.rlm_agent import create_rlm_agent

    # Load .env secrets
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
    except ImportError:
        pass

    if verbose:
        logging.getLogger("myconex").setLevel(logging.INFO)

    # Discover mesh services via mDNS (falls back to .env, fails loudly if neither)
    _service_watcher, _service_advertiser = await _discover_services(verbose=verbose)

    ollama_url   = config.get("ollama", {}).get("url", "http://localhost:11434")
    litellm_url  = config.get("litellm", {}).get("url", "http://localhost:4000")
    moe_cfg      = config.get("hermes_moe", {})
    rlm_cfg      = config.get("rlm", {})
    ollama_model = moe_cfg.get("ollama_fallback", {}).get("model", "llama3.1:8b")

    # Detect hardware tier for router provisioning
    try:
        from core.classifier.hardware import HardwareDetector
        hw = HardwareDetector().detect()
        tier = hw.tier
    except Exception:
        tier = "T2"

    router = TaskRouter(
        node_tier=tier,
        ollama_url=ollama_url,
        litellm_url=litellm_url,
        use_rlm=True,
        rlm_config={
            "ollama_model": ollama_model,
            "system_prompt": rlm_cfg.get(
                "system_prompt",
                "You are MYCONEX — a recursive, self-improving AI mesh agent. "
                "You can decompose complex tasks, delegate to specialists, run Python, "
                "search the web, and learn from every interaction.",
            ),
            "temperature":             rlm_cfg.get("temperature", 0.7),
            "max_tokens":              rlm_cfg.get("max_tokens", 4096),
            "context_budget":          rlm_cfg.get("context_budget", 16384),
            "memory_namespace":        rlm_cfg.get("memory_namespace", "global"),
            "enable_self_optimization": rlm_cfg.get("enable_self_optimization", True),
            "flash_moe_binary":        moe_cfg.get("flash_moe", {}).get("binary_path"),
        },
    )
    await router.start()

    # Grab the primary agent (already an RLMAgent from TaskRouter provisioning)
    agent = router.registry.get("inference-primary")
    if agent is None:
        raise RuntimeError("Router did not provision inference-primary — check TaskRouter setup.")

    # Register Phase 2 tools in the hermes registry
    try:
        from core.gateway.agentic_tools import register_agentic_tools
        register_agentic_tools()
    except Exception as exc:
        logger.debug("agentic_tools registration skipped: %s", exc)

    # ── Wire hermes-agent integration ──────────────────────────────────────────
    hermes_cfg_raw = config.get("hermes", {})
    if hermes_cfg_raw.get("enabled", True):
        try:
            from integrations.hermes_bridge import (
                setup_hermes_path,
                HermesToolBridge,
                HermesGatewayBridge,
                register_hermes_tools_in_myconex,
            )
            from orchestration.agents.hermes_agent import (
                create_hermes_agent,
                HermesAgentConfig,
            )

            if setup_hermes_path():
                # 1. Load hermes tools into shared registry
                if hermes_cfg_raw.get("load_tools", True):
                    n = register_hermes_tools_in_myconex()
                    logger.info("hermes tools loaded: %d tools available", n)

                # 2. Provision HermesAgent and register with router
                h_config = HermesAgentConfig(
                    model=hermes_cfg_raw.get("model") or None,
                    base_url=hermes_cfg_raw.get("base_url") or None,
                    api_key=hermes_cfg_raw.get("api_key") or None,
                    enabled_toolsets=hermes_cfg_raw.get("enabled_toolsets") or None,
                    disabled_toolsets=hermes_cfg_raw.get("disabled_toolsets") or None,
                    max_iterations=int(hermes_cfg_raw.get("max_iterations", 90)),
                    save_trajectories=hermes_cfg_raw.get("save_trajectories", False),
                    skip_memory=hermes_cfg_raw.get("skip_memory", False),
                )
                hermes_agent = create_hermes_agent(
                    hermes_config=h_config,
                    idle_ttl_s=float(hermes_cfg_raw.get("idle_ttl_s", 1800)),
                )
                router.register_agent(hermes_agent)
                logger.info(
                    "HermesAgent registered [model=%s]", h_config.model
                )
        except Exception as exc:
            logger.debug("hermes integration skipped: %s", exc)

    return agent, router, _service_watcher, _service_advertiser


# ─── Mode: CLI REPL ───────────────────────────────────────────────────────────

async def run_cli(config: dict, verbose: bool = False) -> None:
    """
    Interactive REPL backed by the live RLMAgent.

    Commands:
      /status     — show agent and context status
      /memory     — search or dump persistent memory
      /tools      — list registered tools
      /repl CODE  — run Python in the agent's persistent REPL
      /web URL    — fetch URL as text
      /search Q   — web search
      /reset      — clear conversation history
      /help       — show this help
      /quit       — exit
    """
    _banner()
    _info("Booting RLMAgent…")

    agent, router, _watcher, _advertiser = await boot_rlm_agent(config, verbose=verbose)

    _success(f"RLMAgent ready  [model={agent.config.model}  backend={agent.config.backend}]")
    if _RICH:
        console.rule()
    console.print("Type a message or a /command. Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit.\n")

    from orchestration.agents.base_agent import AgentContext
    context = AgentContext(session_id="cli-session")
    stop = False

    while not stop:
        try:
            raw = await _input_async("\n[you] ")
        except (EOFError, KeyboardInterrupt):
            break

        raw = raw.strip()
        if not raw:
            continue

        # ── Built-in commands ──────────────────────────────────────────────
        if raw.startswith("/"):
            parts = raw.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                stop = True

            elif cmd == "/help":
                console.print(
                    "\n[bold]Commands:[/bold]\n"
                    "  /status          — agent & context status\n"
                    "  /memory [query]  — search persistent memory\n"
                    "  /tools           — list registered tools\n"
                    "  /repl <code>     — run Python in persistent REPL\n"
                    "  /web <url>       — fetch a URL as text\n"
                    "  /search <query>  — DuckDuckGo search\n"
                    "  /reset           — clear conversation history\n"
                    "  /verbose         — toggle verbose logging\n"
                    "  /quit            — exit\n"
                )

            elif cmd == "/status":
                st = agent.status()
                console.print(json.dumps(st, indent=2, default=str))

            elif cmd == "/memory":
                if hasattr(agent, "persistent_memory") and agent.persistent_memory:
                    entries = agent.persistent_memory.search(arg or "", top_k=10)
                    if entries:
                        for e in entries:
                            console.print(f"  [{e.category}] {e.key}: {e.content[:200]}")
                    else:
                        console.print("  (no memory entries)")
                else:
                    console.print("  (persistent memory not enabled)")

            elif cmd == "/tools":
                if hasattr(agent, "_tools"):
                    for name, tool in agent._tools.items():
                        console.print(f"  [bold]{name}[/bold] — {tool.description[:80]}")
                else:
                    console.print("  (tool registry not available)")

            elif cmd == "/repl":
                if not arg:
                    console.print("  Usage: /repl <python code>")
                else:
                    result = await agent.run_python(arg)
                    console.print(f"  {result}")

            elif cmd == "/web":
                if not arg:
                    console.print("  Usage: /web <url>")
                else:
                    _info("  Fetching…")
                    text = await agent.read_webpage(arg)
                    console.print(text[:2000])

            elif cmd == "/search":
                if not arg:
                    console.print("  Usage: /search <query>")
                else:
                    from core.gateway.agentic_tools import handle_research
                    result = handle_research(query=arg, max_results=5)
                    console.print(result)

            elif cmd == "/reset":
                context = AgentContext(session_id="cli-session")
                _success("  Conversation history cleared.")

            elif cmd == "/verbose":
                level = logging.getLogger().level
                new_level = logging.DEBUG if level > logging.DEBUG else logging.WARNING
                logging.getLogger().setLevel(new_level)
                _info(f"  Logging: {'DEBUG' if new_level == logging.DEBUG else 'WARNING'}")

            else:
                _warn(f"  Unknown command: {cmd}. Type /help for options.")

            continue

        # ── LLM inference ─────────────────────────────────────────────────
        task_id = str(uuid.uuid4())[:8]
        if _RICH:
            with console.status("[cyan]thinking…[/cyan]", spinner="dots"):
                result = await agent.dispatch(
                    task_id=task_id,
                    task_type="chat",
                    payload={"prompt": raw},
                    context=context,
                )
        else:
            print("[thinking…]")
            result = await agent.dispatch(
                task_id=task_id,
                task_type="chat",
                payload={"prompt": raw},
                context=context,
            )

        if result.success and isinstance(result.output, dict):
            response = result.output.get("response", "")
            meta = result.metadata
            if _RICH:
                console.print(f"\n[bold green][myconex][/bold green] {response}")
                if meta.get("decomposed"):
                    console.print(
                        f"[dim]  ↳ {meta.get('plan_strategy','?')} plan, "
                        f"{result.output.get('sub_task_count','?')} sub-tasks[/dim]"
                    )
                elif meta.get("complexity") is not None:
                    console.print(f"[dim]  ↳ complexity={meta['complexity']:.2f}  "
                                  f"tokens={meta.get('tokens_used','?')}[/dim]")
            else:
                print(f"\n[myconex] {response}\n")
        else:
            _error(f"Error: {result.error}")

    _info("\nShutting down…")
    if _watcher: await _watcher.stop()
    if _advertiser: await _advertiser.stop()
    await router.stop()
    _success("Goodbye.")


# ─── Mode: Autonomous Loop ────────────────────────────────────────────────────

class AutonomousLoop:
    """
    MYCONEX autonomous self-improvement loop.

    The agent runs continuously, processing tasks from a queue, self-generating
    new ones when the queue empties, and writing a session digest on exit.

    Task queue file: ~/.myconex/autonomous_tasks.json
    Format: [{"task_type": "chat", "prompt": "..."}, ...]

    Self-generated tasks are chosen from a rotation of introspection prompts:
      - Analyse own performance patterns
      - Summarise recent persistent memory
      - Search for relevant news/papers on agent AI
      - Run a codebase health check on MYCONEX
      - Verify tool availability and report findings
    """

    SELF_TASKS = [
        ("chat",    "Analyse MYCONEX's recent performance metrics and suggest one concrete improvement."),
        ("search",  "Find the latest research on recursive language models and self-improving agents."),
        ("code",    "Review the MYCONEX orchestration/agents/ directory and identify any code quality issues."),
        ("chat",    "Summarise what you have learned from the persistent memory store this session."),
        ("chat",    "Describe the current state of the MYCONEX mesh: components, capabilities, and gaps."),
        ("search",  "Find news about distributed AI systems and edge inference published in the last week."),
        ("code",    "Check all tool handlers in core/gateway/agentic_tools.py and verify they are correct."),
        ("chat",    "Generate a list of 5 new features that would make MYCONEX more capable."),
    ]

    def __init__(self, agent, router, config: dict, interval_s: float = 5.0) -> None:
        self.agent = agent
        self.router = router
        self.config = config
        self.interval_s = interval_s
        self._task_index = 0
        self._completed = 0
        self._errors = 0
        self._session_log: list[dict] = []
        self._stop = asyncio.Event()
        self._novelty_scanner = None   # injected by run_autonomous
        self._metrics = None           # injected by run_autonomous

    async def run(self) -> None:
        """Main autonomous loop — runs until stop() is called or SIGINT."""
        _success("Autonomous mode active. Press Ctrl+C to stop and write session digest.")
        _info(f"Task queue: {_TASK_QUEUE_FILE}")
        _info(f"Task interval: {self.interval_s}s\n")

        while not self._stop.is_set():
            task = self._next_task()
            if task is None:
                await asyncio.sleep(self.interval_s)
                continue

            task_type = task.get("task_type", "chat")
            prompt    = task.get("prompt", "")
            task_id   = str(uuid.uuid4())[:8]

            _info(f"[{task_id}] {task_type}: {prompt[:80]}…")

            start = time.time()
            try:
                result = await self.agent.dispatch(
                    task_id=task_id,
                    task_type=task_type,
                    payload={"prompt": prompt},
                )
                duration = (time.time() - start) * 1000

                if result.success:
                    self._completed += 1
                    response = ""
                    if isinstance(result.output, dict):
                        response = result.output.get("response", "")
                    _success(f"  ✓ {task_id} ({duration:.0f}ms): {response[:120]}")
                    self._log(task_id, task_type, prompt, response, True, duration)
                else:
                    self._errors += 1
                    _error(f"  ✗ {task_id}: {result.error}")
                    self._log(task_id, task_type, prompt, result.error or "", False, duration)

            except Exception as exc:
                self._errors += 1
                _error(f"  ✗ {task_id} exception: {exc}")
                self._log(task_id, task_type, prompt, str(exc), False, 0)

            await asyncio.sleep(self.interval_s)

        await self._write_digest()

    def set_novelty_scanner(self, scanner) -> None:
        """Inject a NoveltyScanner so novel proposals are dequeued as tasks."""
        self._novelty_scanner = scanner

    def set_metrics(self, metrics) -> None:
        """Inject a MetricsCollector for recording task outcomes."""
        self._metrics = metrics

    def stop(self) -> None:
        self._stop.set()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _next_task(self) -> Optional[dict]:
        """Pop a task from the file queue, or return a self-generated task."""
        # Try file queue first
        if _TASK_QUEUE_FILE.exists():
            try:
                queue = json.loads(_TASK_QUEUE_FILE.read_text())
                if queue:
                    task = queue.pop(0)
                    _TASK_QUEUE_FILE.write_text(json.dumps(queue, indent=2))
                    return task
            except Exception as exc:
                logger.debug("autonomous task queue read error: %s", exc)

        # Try novelty scanner proposals
        if self._novelty_scanner is not None:
            try:
                proposal = self._novelty_scanner.dequeue_proposal()
                if proposal is not None:
                    title = proposal.get("title", "Novelty proposal")
                    desc = proposal.get("description", "")
                    prompt = f"{title}: {desc}" if desc else title
                    return {"task_type": "chat", "prompt": prompt, "_source": "novelty"}
            except Exception as exc:
                logger.debug("novelty scanner dequeue error: %s", exc)

        # Self-generate from rotation
        idx = self._task_index % len(self.SELF_TASKS)
        self._task_index += 1
        task_type, prompt = self.SELF_TASKS[idx]
        return {"task_type": task_type, "prompt": prompt}

    def _log(
        self, task_id: str, task_type: str, prompt: str,
        response: str, success: bool, duration_ms: float,
    ) -> None:
        self._session_log.append({
            "task_id": task_id,
            "task_type": task_type,
            "prompt": prompt[:200],
            "response": response[:500],
            "success": success,
            "duration_ms": round(duration_ms, 1),
            "ts": datetime.utcnow().isoformat(),
        })
        if self._metrics is not None:
            try:
                self._metrics.record_task(success=success, duration_ms=duration_ms)
            except Exception:
                pass

    async def _write_digest(self) -> None:
        """Write a markdown session digest to ~/.myconex/session_YYYYMMDD.md"""
        _MYCONEX_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.utcnow().strftime("%Y%m%d_%H%M")
        digest_path = _MYCONEX_DIR / f"session_{date_str}.md"

        lines = [
            f"# MYCONEX Autonomous Session — {date_str} UTC",
            f"\n**Tasks completed:** {self._completed}  "
            f"**Errors:** {self._errors}  "
            f"**Total:** {len(self._session_log)}\n",
            "## Task Log\n",
        ]
        for entry in self._session_log:
            status = "✅" if entry["success"] else "❌"
            lines.append(
                f"### {status} [{entry['task_id']}] {entry['task_type']} "
                f"({entry['duration_ms']:.0f}ms)\n"
                f"**Prompt:** {entry['prompt']}\n\n"
                f"**Response:** {entry['response']}\n"
            )

        # Append memory snapshot
        if hasattr(self.agent, "persistent_memory") and self.agent.persistent_memory:
            entries = list(self.agent.persistent_memory._entries.values())
            lines.append("\n## Persistent Memory Snapshot\n")
            for e in sorted(entries, key=lambda x: x.importance, reverse=True)[:20]:
                lines.append(f"- **[{e.category}]** {e.key}: {e.content[:200]}")

        digest_path.write_text("\n".join(lines))
        _success(f"\nSession digest written to {digest_path}")


async def run_autonomous(config: dict, interval_s: float = 5.0, verbose: bool = False) -> None:
    """Boot and run the autonomous self-improvement loop."""
    _banner()
    _info("Booting RLMAgent for autonomous operation…")

    agent, router, _watcher, _advertiser = await boot_rlm_agent(config, verbose=verbose)

    # Ensure task queue file exists
    _MYCONEX_DIR.mkdir(parents=True, exist_ok=True)
    if not _TASK_QUEUE_FILE.exists():
        _TASK_QUEUE_FILE.write_text("[]")
        _info(f"Created empty task queue at {_TASK_QUEUE_FILE}")
        _info("Add tasks via: echo '[{\"task_type\": \"chat\", \"prompt\": \"...\"}]' > ~/.myconex/autonomous_tasks.json\n")

    loop_runner = AutonomousLoop(agent, router, config, interval_s=interval_s)
    bg_tasks: list[asyncio.Task] = []

    # ── Wire: Metrics ─────────────────────────────────────────────────────────
    try:
        from core.metrics import get_metrics
        metrics = get_metrics()
        loop_runner.set_metrics(metrics)
        _success("Metrics collector attached.")

        async def _metrics_bg():
            while True:
                await asyncio.sleep(300)   # write snapshot every 5 min
                try:
                    metrics.write_periodic_report()
                except Exception as exc:
                    logger.debug("metrics snapshot error: %s", exc)

        bg_tasks.append(asyncio.create_task(_metrics_bg(), name="metrics-bg"))
    except Exception as exc:
        _warn(f"Metrics not available: {exc}")

    # ── Wire: Plugin Loader ────────────────────────────────────────────────────
    try:
        from core.plugin_loader import PluginLoader
        plugin_loader = PluginLoader()
        loaded_meta = await plugin_loader.load_all()
        if loaded_meta:
            _success(f"Plugins loaded: {', '.join(loaded_meta.keys())}")
            plugin_loader.registry.wire_into_agentic_tools()
        else:
            _info("No plugins found in plugins/ directory.")

        async def _plugin_watcher_bg():
            """Poll for plugin file changes every 30 s."""
            while True:
                await asyncio.sleep(30)
                try:
                    if plugin_loader._watcher:
                        plugin_loader._watcher.check()
                except Exception as exc:
                    logger.debug("plugin watcher error: %s", exc)

        bg_tasks.append(asyncio.create_task(_plugin_watcher_bg(), name="plugin-watcher"))
    except Exception as exc:
        _warn(f"Plugin loader not available: {exc}")

    # ── Wire: Self-Healer ─────────────────────────────────────────────────────
    try:
        from core.self_healer import create_self_healer
        healer = create_self_healer()

        async def _healer_bg():
            """Run health check every 10 min."""
            while True:
                await asyncio.sleep(600)
                try:
                    report = healer.run_health_check()
                    if not report.get("healthy", True):
                        _warn(f"Health issues detected: {report.get('issues', [])}")
                        healer.run_recovery(report)
                except Exception as exc:
                    logger.debug("self-healer error: %s", exc)

        bg_tasks.append(asyncio.create_task(_healer_bg(), name="self-healer"))
        _success("Self-healer background monitor started (10 min interval).")
    except Exception as exc:
        _warn(f"Self-healer not available: {exc}")

    # ── Wire: MCP Servers ────────────────────────────────────────────────────
    try:
        from core.mcp_client import setup_mcp_from_config
        mcp_client = await setup_mcp_from_config(config)
        if mcp_client:
            n_mcp = await mcp_client.register_with_myconex()
            _success(f"MCP: connected {len(mcp_client._servers)} servers, {n_mcp} tools registered.")
        else:
            _info("MCP: no servers configured (add mcp.servers to mesh_config.yaml).")
    except Exception as exc:
        _warn(f"MCP client not available: {exc}")

    # ── Wire: Hermes Gateway (multi-platform messaging) ───────────────────────
    try:
        from integrations.hermes_bridge import HermesGatewayBridge
        platforms = HermesGatewayBridge.list_enabled_platforms(
            config.get("hermes", {}).get("gateway_config_path") or None
        )
        if platforms:
            ok = await HermesGatewayBridge.start(
                config.get("hermes", {}).get("gateway_config_path") or None
            )
            if ok:
                _success(f"Hermes gateway started: {', '.join(platforms)}")
            else:
                _warn("Hermes gateway config found but failed to start.")
        else:
            _info("Hermes gateway: no platforms configured (~/.hermes/gateway.yaml).")

        async def _gateway_shutdown():
            await HermesGatewayBridge.stop()

    except Exception as exc:
        _warn(f"Hermes gateway not available: {exc}")

    # ── Wire: Novelty Scanner ─────────────────────────────────────────────────
    try:
        from core.novelty_scanner import create_novelty_scanner
        scanner = create_novelty_scanner(agent=agent)
        loop_runner.set_novelty_scanner(scanner)

        novelty_interval_hours = config.get("novelty", {}).get("scan_interval_hours", 6)

        async def _novelty_bg():
            while True:
                await asyncio.sleep(novelty_interval_hours * 3600)
                try:
                    await scanner.run_once()
                    _info(f"Novelty scan complete — queue depth: {scanner.queue.depth()}")
                except Exception as exc:
                    logger.debug("novelty scanner error: %s", exc)

        # Run one initial scan shortly after startup
        async def _novelty_initial():
            await asyncio.sleep(30)
            try:
                await scanner.run_once()
                _info(f"Initial novelty scan complete — queue depth: {scanner.queue.depth()}")
            except Exception as exc:
                logger.debug("initial novelty scan error: %s", exc)

        bg_tasks.append(asyncio.create_task(_novelty_bg(), name="novelty-scanner"))
        bg_tasks.append(asyncio.create_task(_novelty_initial(), name="novelty-initial"))
        _success(f"Novelty scanner wired (scan interval: {novelty_interval_hours}h).")
    except Exception as exc:
        _warn(f"Novelty scanner not available: {exc}")

    # ── Signal handling ───────────────────────────────────────────────────────
    stop_event = asyncio.Event()

    def _sig_handler(sig):
        _warn(f"\nReceived {sig.name}, stopping autonomous loop…")
        loop_runner.stop()
        stop_event.set()
        for t in bg_tasks:
            t.cancel()

    ev_loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        ev_loop.add_signal_handler(sig, lambda s=sig: _sig_handler(s))

    await loop_runner.run()

    # Cancel background tasks on clean exit
    for t in bg_tasks:
        t.cancel()
    if bg_tasks:
        await asyncio.gather(*bg_tasks, return_exceptions=True)

    if _watcher: await _watcher.stop()
    if _advertiser: await _advertiser.stop()
    await router.stop()


# ─── Mode: Discord + Mesh ─────────────────────────────────────────────────────

async def run_discord_with_rlm(config: dict, verbose: bool = False) -> None:
    """
    Full stack: mesh node + Discord gateway with RLMAgent as the primary backend.

    The Discord gateway already uses TaskRouter internally; booting via
    boot_rlm_agent() ensures the primary agent is an RLMAgent.
    """
    _banner()
    _info("Booting RLMAgent + Discord gateway…")

    agent, router, _watcher, _advertiser = await boot_rlm_agent(config, verbose=verbose)

    # Hand off to the existing mesh runner with mode="discord"
    sys.path.insert(0, str(Path(__file__).parent))
    from main import run_node  # type: ignore[import]

    # Inject already-started router into the node runner via config override
    config["_prestarted_router"] = router   # main.py checks this key
    try:
        await run_node(config, mode="discord")
    except Exception:
        # main.py doesn't support _prestarted_router yet — fall back to
        # letting it boot its own router alongside the Discord gateway.
        await run_node(config, mode="discord")
    finally:
        if _watcher: await _watcher.stop()
        if _advertiser: await _advertiser.stop()
        await router.stop()


# ─── Mode: worker / api / full ───────────────────────────────────────────────

async def run_standard(config: dict, mode: str) -> None:
    """Delegate to the existing main.py node runner for worker/api/full modes."""
    sys.path.insert(0, str(Path(__file__).parent))
    from main import run_node  # type: ignore[import]
    await run_node(config, mode=mode)


# ─── CLI Argument Parsing ─────────────────────────────────────────────────────

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m myconex",
        description="MYCONEX — Distributed AI Mesh System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  cli         Interactive REPL (default when run interactively)
  autonomous  Continuous self-improving task loop
  discord     Mesh node + Discord gateway
  api         Mesh node + REST API
  worker      Background mesh worker only
  full        Mesh + API + Discord simultaneously

Examples:
  python -m myconex                          # CLI REPL
  python -m myconex --mode autonomous        # Autonomous loop
  python -m myconex --mode discord           # Discord bot
  python -m myconex --mode api --port 8765   # REST API
  python -m myconex --mode autonomous --interval 10
        """,
    )
    parser.add_argument(
        "--mode", "-m",
        default="cli",
        choices=["cli", "autonomous", "discord", "api", "worker", "full"],
        help="Execution mode (default: cli)",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        metavar="PATH",
        help="Path to node config YAML (default: config/mesh_config.yaml)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Task interval for autonomous mode (default: 5.0s)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (INFO) logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"MYCONEX {VERSION}",
    )
    # Pass-through for main.py subcommands
    parser.add_argument(
        "--host",
        default=None,
        help="API host override",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="API port override",
    )
    return parser.parse_args()


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("myconex").setLevel(logging.DEBUG)

    config = _load_config(args.config)

    if args.host:
        config.setdefault("api", {})["host"] = args.host
    if args.port:
        config.setdefault("api", {})["port"] = args.port

    try:
        if args.mode == "cli":
            asyncio.run(run_cli(config, verbose=args.verbose))

        elif args.mode == "autonomous":
            asyncio.run(run_autonomous(config, interval_s=args.interval, verbose=args.verbose))

        elif args.mode == "discord":
            asyncio.run(run_discord_with_rlm(config, verbose=args.verbose))

        elif args.mode in ("worker", "api", "full"):
            asyncio.run(run_standard(config, mode=args.mode))

        else:
            _error(f"Unknown mode: {args.mode}")
            sys.exit(1)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
