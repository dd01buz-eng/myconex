"""
MYCONEX Unified Configuration
================================
Single source of truth for all MYCONEX settings.

Priority (highest wins):
  1. Environment variables (MYCONEX_* prefix)
  2. .env file (project root)
  3. config/mesh_config.yaml
  4. Dataclass defaults

Usage:
    from config import load_config, MyconexConfig
    cfg = load_config()
    print(cfg.ollama.url)
    print(cfg.discord.token)
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent
_ENV_FILE = _ROOT / ".env"
_DEFAULT_YAML = _ROOT / "config" / "mesh_config.yaml"


# ─── Sub-configs ─────────────────────────────────────────────────────────────

@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434"
    default_model: str = "llama3.1:8b"
    timeout_s: float = 60.0


@dataclass
class LlamaCppConfig:
    url: str = "http://localhost:8080"
    enabled: bool = False


@dataclass
class LMStudioConfig:
    url: str = "http://localhost:1234"
    enabled: bool = False


@dataclass
class LiteLLMConfig:
    url: str = "http://localhost:4000"
    enabled: bool = False


@dataclass
class BackendConfig:
    """Which inference backend to use as default."""
    default: str = "ollama"   # "ollama" | "llamacpp" | "lmstudio" | "litellm"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    llamacpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    lmstudio: LMStudioConfig = field(default_factory=LMStudioConfig)
    litellm: LiteLLMConfig = field(default_factory=LiteLLMConfig)


@dataclass
class DiscordConfig:
    token: str = ""
    enabled: bool = False
    require_mention: bool = False
    auto_thread: bool = False
    allow_bots: str = "none"   # "none" | "mentions" | "all"
    free_response_channels: list = field(default_factory=list)
    allowed_users: list = field(default_factory=list)


@dataclass
class MemoryConfig:
    dir: str = str(Path.home() / ".myconex" / "memory")
    namespace: str = "global"
    max_entries: int = 500
    summarize_threshold: int = 400
    session_ttl_s: float = 3600.0


@dataclass
class TokenBudgetConfig:
    context_budget: int = 16384      # total tokens for RLM context tree
    max_tokens_per_call: int = 4096  # max tokens per LLM call
    memory_fraction: float = 0.15   # fraction reserved for memory context


@dataclass
class AgentRosterConfig:
    enabled: bool = True
    default_tier: str = "T2"
    divisions: list = field(default_factory=lambda: [
        "engineering", "research", "security", "data", "devops", "qa"
    ])


@dataclass
class RLMConfig:
    enabled: bool = True
    decompose_threshold: float = 0.60   # complexity score above which tasks are decomposed
    max_delegation_depth: int = 4
    optimize_every_n: int = 20
    enable_self_optimization: bool = True
    system_prompt: str = (
        "You are MYCONEX — a recursive, self-improving AI mesh agent. "
        "You can decompose complex tasks, delegate to specialists, run Python, "
        "search the web, and learn from every interaction."
    )


@dataclass
class AutonomousLoopConfig:
    enabled: bool = False
    cycle_interval_s: float = 30.0
    max_cycles: Optional[int] = None
    dry_run: bool = False
    reflect_every_n: int = 10


@dataclass
class MeshConfig:
    node_name: str = ""
    tier: str = ""           # auto-detected if empty
    nats_url: str = "nats://localhost:4222"
    redis_url: str = "redis://localhost:6379"
    qdrant_url: str = "http://localhost:6333"
    mdns_enabled: bool = True


@dataclass
class APIConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    enabled: bool = False


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"


# ─── Hermes Config ───────────────────────────────────────────────────────────

@dataclass
class HermesConfig:
    """Configuration for the hermes-agent integration."""
    enabled: bool = True
    model: str = ""                          # defaults to HERMES_MODEL env or claude-sonnet-4-6
    base_url: str = ""                       # defaults to OPENROUTER or Anthropic
    api_key: str = ""                        # defaults to OPENROUTER_API_KEY / ANTHROPIC_API_KEY
    enabled_toolsets: list = field(default_factory=list)   # empty = all toolsets
    disabled_toolsets: list = field(default_factory=list)
    max_iterations: int = 90
    save_trajectories: bool = False
    skip_memory: bool = False
    idle_ttl_s: float = 1800.0
    gateway_config_path: str = ""           # path to hermes gateway.yaml
    load_tools: bool = True                  # register hermes tools in MYCONEX


# ─── MCP Config ───────────────────────────────────────────────────────────────

@dataclass
class MCPServerEntry:
    """A single MCP server definition."""
    name: str = ""
    command: str = ""
    args: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    timeout_s: float = 30.0
    enabled: bool = True


@dataclass
class MCPConfig:
    """Configuration for MCP server connectivity."""
    enabled: bool = False
    servers: list = field(default_factory=list)   # list of MCPServerEntry / dicts


# ─── Root Config ─────────────────────────────────────────────────────────────

@dataclass
class MyconexConfig:
    """Root configuration object for the entire MYCONEX system."""
    backend: BackendConfig = field(default_factory=BackendConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tokens: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    roster: AgentRosterConfig = field(default_factory=AgentRosterConfig)
    rlm: RLMConfig = field(default_factory=RLMConfig)
    autonomous: AutonomousLoopConfig = field(default_factory=AutonomousLoopConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hermes: HermesConfig = field(default_factory=HermesConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_legacy_dict(self) -> dict:
        """Convert to the dict format expected by the existing main.py."""
        return {
            "ollama": {"url": self.backend.ollama.url},
            "litellm": {"url": self.backend.litellm.url},
            "nats": {"url": self.mesh.nats_url},
            "redis": {"url": self.mesh.redis_url},
            "qdrant": {"url": self.mesh.qdrant_url},
            "node": {"name": self.mesh.node_name, "tier": self.mesh.tier},
            "api": {"host": self.api.host, "port": self.api.port},
            "discord": {
                "enabled": self.discord.enabled,
                "require_mention": self.discord.require_mention,
                "auto_thread": self.discord.auto_thread,
                "allow_bots": self.discord.allow_bots,
            },
            "hermes_moe": {
                "enabled": True,
                "temperature": 0.7,
                "max_tokens": self.tokens.max_tokens_per_call,
                "ollama_fallback": {"model": self.backend.ollama.default_model},
            },
            "rlm": {
                "system_prompt": self.rlm.system_prompt,
                "max_tokens": self.tokens.max_tokens_per_call,
                "context_budget": self.tokens.context_budget,
                "memory_namespace": self.memory.namespace,
                "enable_self_optimization": self.rlm.enable_self_optimization,
                "backend": self.backend.default,
            },
        }


# ─── Loaders ─────────────────────────────────────────────────────────────────

def _load_dotenv(path: Path = _ENV_FILE) -> None:
    """Parse .env file and set environment variables (no external deps)."""
    if not path.is_file():
        return
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                # Strip inline comments (e.g. VALUE=foo  # comment)
                comment_pos = val.find(" #")
                if comment_pos >= 0:
                    val = val[:comment_pos]
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as exc:
        logger.debug("dotenv load failed: %s", exc)


def _load_yaml(path: Path) -> dict:
    """Load a YAML file (requires PyYAML; falls back to empty dict)."""
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore[import]
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.debug("PyYAML not installed — YAML config skipped")
        return {}
    except Exception as exc:
        logger.warning("YAML config load error (%s): %s", path, exc)
        return {}


def _apply_env(cfg: MyconexConfig) -> None:
    """Apply MYCONEX_* environment variable overrides."""
    ev = os.environ

    # Backend
    if v := ev.get("OLLAMA_URL"):
        cfg.backend.ollama.url = v
    if v := ev.get("LLAMACPP_URL"):
        cfg.backend.llamacpp.url = v
        cfg.backend.llamacpp.enabled = True
    if v := ev.get("LMSTUDIO_URL"):
        cfg.backend.lmstudio.url = v
        cfg.backend.lmstudio.enabled = True
    if v := ev.get("LITELLM_URL"):
        cfg.backend.litellm.url = v
        cfg.backend.litellm.enabled = True
    if v := ev.get("MYCONEX_BACKEND"):
        cfg.backend.default = v

    # Discord
    if v := ev.get("DISCORD_BOT_TOKEN"):
        cfg.discord.token = v
        cfg.discord.enabled = True
    if v := ev.get("DISCORD_REQUIRE_MENTION"):
        cfg.discord.require_mention = v.lower() in ("1", "true", "yes")
    if v := ev.get("DISCORD_AUTO_THREAD"):
        cfg.discord.auto_thread = v.lower() in ("1", "true", "yes")
    if v := ev.get("DISCORD_ALLOW_BOTS"):
        cfg.discord.allow_bots = v
    if v := ev.get("DISCORD_FREE_RESPONSE_CHANNELS"):
        cfg.discord.free_response_channels = [c.strip() for c in v.split(",") if c.strip()]
    if v := ev.get("DISCORD_ALLOWED_USERS"):
        cfg.discord.allowed_users = [u.strip() for u in v.split(",") if u.strip()]

    # Mesh
    if v := ev.get("NATS_URL"):
        cfg.mesh.nats_url = v
    if v := ev.get("REDIS_URL"):
        cfg.mesh.redis_url = v
    if v := ev.get("QDRANT_URL"):
        cfg.mesh.qdrant_url = v
    if v := ev.get("MYCONEX_NODE"):
        cfg.mesh.node_name = v
    if v := ev.get("MYCONEX_TIER"):
        cfg.mesh.tier = v

    # RLM
    if v := ev.get("MYCONEX_RLM_DECOMPOSE_THRESHOLD"):
        cfg.rlm.decompose_threshold = float(v)
    if v := ev.get("MYCONEX_CONTEXT_BUDGET"):
        cfg.tokens.context_budget = int(v)

    # Memory
    if v := ev.get("MYCONEX_MEMORY_DIR"):
        cfg.memory.dir = v
    if v := ev.get("MYCONEX_MEMORY_NAMESPACE"):
        cfg.memory.namespace = v

    # API
    if v := ev.get("MYCONEX_API_HOST"):
        cfg.api.host = v
    if v := ev.get("MYCONEX_API_PORT"):
        cfg.api.port = int(v)

    # Logging
    if v := ev.get("MYCONEX_LOG_LEVEL"):
        cfg.logging.level = v.upper()


def _apply_yaml(cfg: MyconexConfig, data: dict) -> None:
    """Apply values from the legacy YAML config dict."""
    if not data:
        return

    if "ollama" in data:
        cfg.backend.ollama.url = data["ollama"].get("url", cfg.backend.ollama.url)
    if "litellm" in data:
        cfg.backend.litellm.url = data["litellm"].get("url", cfg.backend.litellm.url)
    if "nats" in data:
        cfg.mesh.nats_url = data["nats"].get("url", cfg.mesh.nats_url)
    if "redis" in data:
        cfg.mesh.redis_url = data["redis"].get("url", cfg.mesh.redis_url)
    if "qdrant" in data:
        cfg.mesh.qdrant_url = data["qdrant"].get("url", cfg.mesh.qdrant_url)
    if "node" in data:
        cfg.mesh.node_name = data["node"].get("name", cfg.mesh.node_name)
        cfg.mesh.tier = data["node"].get("tier", cfg.mesh.tier)
    if "api" in data:
        cfg.api.host = data["api"].get("host", cfg.api.host)
        cfg.api.port = int(data["api"].get("port", cfg.api.port))
    if "discord" in data:
        d = data["discord"]
        cfg.discord.enabled = bool(d.get("enabled", cfg.discord.enabled))
        cfg.discord.require_mention = bool(d.get("require_mention", cfg.discord.require_mention))
        cfg.discord.auto_thread = bool(d.get("auto_thread", cfg.discord.auto_thread))
    if "hermes_moe" in data:
        m = data["hermes_moe"]
        cfg.tokens.max_tokens_per_call = int(m.get("max_tokens", cfg.tokens.max_tokens_per_call))
    if "rlm" in data:
        r = data["rlm"]
        cfg.rlm.system_prompt = r.get("system_prompt", cfg.rlm.system_prompt)
        cfg.rlm.enable_self_optimization = bool(r.get("enable_self_optimization", True))
        cfg.tokens.context_budget = int(r.get("context_budget", cfg.tokens.context_budget))
        cfg.memory.namespace = r.get("memory_namespace", cfg.memory.namespace)


def apply_discovered_urls(cfg: "MyconexConfig", urls: "ServiceURLs") -> None:
    """
    Apply mDNS-discovered service URLs to cfg.mesh in-place.

    Only applies a URL when:
      - The ServiceURLs field is not None (service was found)
      - No explicit user env var is set for that service (user config wins)
    """
    if urls.nats_url and not os.environ.get("NATS_URL"):
        cfg.mesh.nats_url = urls.nats_url
    if urls.redis_url and not os.environ.get("REDIS_URL"):
        cfg.mesh.redis_url = urls.redis_url
    if urls.qdrant_url and not os.environ.get("QDRANT_URL"):
        cfg.mesh.qdrant_url = urls.qdrant_url


# ─── Public API ───────────────────────────────────────────────────────────────

def load_config(
    yaml_path: Optional[str] = None,
    env_file: Optional[str] = None,
    apply_env: bool = True,
) -> MyconexConfig:
    """
    Load the unified MYCONEX configuration.

    Args:
        yaml_path:  Path to YAML config (default: config/mesh_config.yaml).
        env_file:   Path to .env file (default: project root .env).
        apply_env:  Whether to apply MYCONEX_* environment variable overrides.

    Returns:
        Fully populated MyconexConfig.
    """
    # 1. Load .env into os.environ
    _load_dotenv(Path(env_file) if env_file else _ENV_FILE)

    # 2. Start with defaults
    cfg = MyconexConfig()

    # 3. Apply YAML
    yaml_data = _load_yaml(Path(yaml_path) if yaml_path else _DEFAULT_YAML)
    _apply_yaml(cfg, yaml_data)

    # 4. Apply environment overrides
    if apply_env:
        _apply_env(cfg)

    logger.debug(
        "config loaded: backend=%s ollama=%s discord=%s",
        cfg.backend.default, cfg.backend.ollama.url, cfg.discord.enabled,
    )
    return cfg


# ─── Singleton ────────────────────────────────────────────────────────────────

_GLOBAL_CONFIG: Optional[MyconexConfig] = None


def get_config() -> MyconexConfig:
    """Return the global config singleton (loading it on first call)."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = load_config()
    return _GLOBAL_CONFIG


def reset_config() -> None:
    """Clear the global config singleton (useful in tests)."""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = None
