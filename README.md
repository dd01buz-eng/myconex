# MYCONEX

**Distributed AI Mesh System — Inspired by Fungal Networks**

MYCONEX is a self-organizing, peer-to-peer AI mesh that lets heterogeneous machines (from Raspberry Pis to GPU workstations) collaborate as a single intelligent system. Like a mycelium network, every node contributes what it can and routes work to where it runs best.

Built on Recursive Language Model (RLM) principles: the primary agent manages its own context, decomposes complex tasks, and delegates to specialists — with every interaction feeding back into a persistent self-improvement loop.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          MYCONEX Mesh                               │
│                                                                     │
│  ┌──────────────┐   NATS pub/sub   ┌──────────────┐               │
│  │  T1 Node     │◄────────────────►│  T2 Node     │               │
│  │  (70B GPU)   │                  │  (8B GPU)    │               │
│  │              │                  │              │               │
│  │  RLMAgent    │                  │  RLMAgent    │               │
│  │  ┌─────────┐ │                  │  ┌─────────┐ │               │
│  │  │MoE Chain│ │                  │  │MoE Chain│ │               │
│  │  │flash-moe│ │                  │  │Nous 8B  │ │               │
│  │  │Nous 70B │ │                  │  │Ollama   │ │               │
│  │  └─────────┘ │                  │  └─────────┘ │               │
│  └──────────────┘                  └──────────────┘               │
│           ▲                                ▲                        │
│           │    Redis (state) · Qdrant (RAG)│                        │
│           └───────────────────────────────┘                        │
│                                                                     │
│  ┌─────────────────────────────────────────┐                       │
│  │              Agent Roster               │                       │
│  │  Engineering │ Research │ Security      │                       │
│  │  Data        │ DevOps   │ QA            │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │
│  │ Discord  │  │ REST API │  │Dashboard │  │  Autonomous Loop   │ │
│  │ Gateway  │  │ :8765    │  │Watch Mode│  │  (self-optimize)   │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### Recursive Language Models (RLM)
`RLMAgent` manages its own context via a persistent Python REPL and spawns sub-instances to delegate token-heavy sub-tasks. Complexity scoring (`_score_complexity()`) drives the decomposition threshold — tasks scoring above **0.60** are automatically decomposed into parallel sub-tasks.

### Agent-as-Employee Pattern
Agents have roles (`MANAGER`, `WORKER`, `SPECIALIST`). Managers decompose and delegate; workers execute and report back. The `delegate()` / `delegate_parallel()` API on `BaseAgent` routes tasks through `TaskRouter`.

### Mixture-of-Experts Expert Chain
```
flash-moe (C/Metal) → Nous 8B → Nous 70B → OpenRouter → Ollama fallback
```
Each expert is tried in order; the first successful result is returned.

### Tier-Aware Routing
Hardware is auto-detected at startup. Tiers:
| Tier | Hardware | Primary Model |
|------|----------|---------------|
| T1   | 70B+ GPU | Nous-Hermes 70B |
| T2   | 8B GPU   | Nous-Hermes 8B  |
| T3   | CPU only | Ollama fallback |
| T4   | Edge     | BitNet 1-bit    |

---

## Project Layout

```
myconex/
├── __main__.py                    # Entry point (all modes)
├── main.py                        # Legacy worker/mesh node launcher
├── config.py                      # Unified configuration system
├── buzlock_bot.py                 # Buzlock bot integration
├── CLAUDE.md                      # Claude Code behavioral rules
├── lessons.md                     # Self-improvement lesson log
│
├── orchestration/
│   ├── agents/
│   │   ├── base_agent.py          # BaseAgent + AgentRole + delegate()
│   │   ├── rlm_agent.py           # RLMAgent — top-level orchestrator
│   │   ├── hermes_agent.py        # Nous-Hermes GGUF agent wrapper
│   │   └── context_manager.py     # ContextFrame, SessionMemory, PersistentMemory
│   ├── workflows/
│   │   └── task_router.py         # TaskRouter — agent lifecycle + routing
│   └── agent_roster.py            # Division-based 6-division roster
│
├── core/
│   ├── gateway/
│   │   ├── agentic_tools.py       # Tool registry + handlers
│   │   ├── python_repl.py         # PersistentPythonREPL, REPLPool, CodebaseIndex
│   │   ├── discord_gateway.py     # Discord bot → RLMAgent wiring
│   │   ├── api_gateway.py         # REST API gateway (:8765)
│   │   ├── mesh_gateway.py        # Inter-node mesh communication
│   │   ├── chat_history_retriever.py
│   │   └── self_improvement.py    # Self-improvement pipeline
│   ├── coordinator/
│   │   └── orchestrator.py        # Mesh task lifecycle, node roles, topology
│   ├── memory/
│   │   └── vector_store.py        # Qdrant vector store / RAG
│   ├── classifier/
│   │   └── hardware.py            # Hardware detection → tier assignment
│   ├── autonomous_loop.py         # 4-phase self-optimization loop
│   ├── mcp_client.py              # MCP protocol client
│   ├── self_healer.py             # Automatic fault recovery
│   ├── novelty_scanner.py         # Novel signal detection
│   ├── plugin_loader.py           # Dynamic plugin loading
│   ├── metrics.py                 # Metrics collection
│   ├── digest.py                  # Session digest generation
│   ├── notifications.py           # Notification routing
│   ├── discovery/                 # mDNS mesh discovery
│   └── messaging/                 # NATS pub/sub client
│
├── dashboard/
│   └── app.py                     # Web dashboard with Watch Mode
│
├── tools/
│   ├── sandbox_executor.py        # Resource-limited subprocess execution
│   ├── document_processor.py      # PDF/HTML/text ingestion pipeline
│   └── intel_aggregator.py        # Multi-source intelligence gathering
│
├── integrations/
│   ├── moe_hermes_integration.py  # HermesMoEAgent (MoE primary)
│   ├── hermes_bridge.py           # Bridge to hermes-agent process
│   ├── dlam_client.py             # DLAM integration
│   ├── fabric_client.py           # Fabric AI integration
│   ├── knowledge_store.py         # External knowledge store
│   ├── rss_monitor.py             # RSS feed monitoring
│   ├── signal_detector.py         # Signal/anomaly detection
│   ├── gmail_reader.py            # Gmail ingestion
│   ├── email_ingester.py          # Email ingestion pipeline
│   ├── youtube_ingester.py        # YouTube transcript ingestion
│   ├── podcast_ingester.py        # Podcast audio ingestion
│   ├── flash-moe/                 # C/Metal flash-moe inference
│   └── hermes-agent/              # Nous-Hermes GGUF agent
│
├── services/                      # Docker Compose: NATS, Redis, Qdrant, LiteLLM
├── mobile/                        # Capacitor + Vite + Tailwind mobile scaffold
├── api/                           # API server module
└── tests/
    └── test_myconex.py            # Comprehensive test suite
```

---

## Quick Start

### Prerequisites

```bash
python3 --version  # 3.10+

# Optional — install for full feature set
pip install PyYAML httpx pdfminer.six feedparser discord.py nats-py redis
```

### Install

```bash
git clone <repo>
cd myconex
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Run (CLI mode)

```bash
python3 -m myconex --mode cli
```

Available commands in the CLI REPL:

| Command | Description |
|---------|-------------|
| `/status` | Show agent and router status |
| `/memory <query>` | Query persistent memory |
| `/tools` | List available tools |
| `/repl <code>` | Execute Python in the persistent REPL |
| `/web <url>` | Fetch and summarize a webpage |
| `/search <query>` | Search the MYCONEX codebase |
| `/reset` | Reset the current session |
| `/verbose` | Toggle verbose logging |
| `/quit` | Exit |

### Run (other modes)

```bash
# Autonomous self-optimization loop
python3 -m myconex --mode autonomous --interval 30

# Discord bot with RLMAgent
python3 -m myconex --mode discord

# REST API server
python3 -m myconex --mode api

# Background worker node (mesh)
python3 -m myconex --mode worker

# Full stack (worker + api + discord)
python3 -m myconex --mode full
```

### Docker (services)

```bash
cd services
docker compose up -d   # starts NATS, Redis, Qdrant, LiteLLM
```

---

## Configuration

Configuration is resolved in this priority order (highest wins):

1. Environment variables (`MYCONEX_*` prefix)
2. `.env` file (project root)
3. `config/mesh_config.yaml`
4. Dataclass defaults

### Common Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama inference endpoint |
| `LLAMACPP_URL` | — | llama.cpp server URL (enables llamacpp backend) |
| `LMSTUDIO_URL` | — | LM Studio URL (enables lmstudio backend) |
| `LITELLM_URL` | `http://localhost:4000` | LiteLLM proxy URL |
| `MYCONEX_BACKEND` | `ollama` | Default backend: `ollama`, `llamacpp`, `lmstudio`, `litellm` |
| `DISCORD_BOT_TOKEN` | — | Discord bot token (enables Discord gateway) |
| `DISCORD_REQUIRE_MENTION` | `false` | Only respond when mentioned |
| `DISCORD_AUTO_THREAD` | `false` | Auto-create threads for responses |
| `NATS_URL` | `nats://localhost:4222` | NATS messaging URL |
| `REDIS_URL` | `redis://localhost:6379` | Redis state store URL |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector store URL |
| `MYCONEX_NODE` | auto | Node name in the mesh |
| `MYCONEX_TIER` | auto | Force tier: `T1`–`T4` |
| `MYCONEX_CONTEXT_BUDGET` | `16384` | Total token budget for RLM context |
| `MYCONEX_MEMORY_DIR` | `~/.myconex/memory` | Persistent memory directory |
| `MYCONEX_MEMORY_NAMESPACE` | `global` | Memory namespace |
| `MYCONEX_API_HOST` | `127.0.0.1` | API server host |
| `MYCONEX_API_PORT` | `8765` | API server port |
| `MYCONEX_LOG_LEVEL` | `INFO` | Logging level |

### Programmatic Config

```python
from config import load_config

cfg = load_config()
print(cfg.backend.ollama.url)
print(cfg.discord.enabled)
print(cfg.tokens.context_budget)
```

---

## Agent Tools

Tools are registered in `core/gateway/agentic_tools.py` and available to all agents:

| Tool | Description |
|------|-------------|
| `python_repl` | Execute Python in a persistent, session-keyed namespace |
| `web_read` | Fetch a URL and return structured title/headings/text |
| `codebase_search` | Search the MYCONEX codebase by keyword |
| `gguf_infer` | Run inference on a local GGUF model via llama-cpp-python |
| `memory_store` | Store a fact in persistent cross-session memory |
| `memory_retrieve` | Retrieve relevant memories by query |
| `delegate` | Delegate a sub-task to a specialized agent |

---

## Agent Divisions

The `AgentRoster` organizes agents into 6 divisions:

| Division | Specialties |
|----------|-------------|
| **Engineering** | Python, Go, APIs, architecture, refactoring |
| **Research** | Literature review, summarization, analysis |
| **Security** | Threat modeling, auditing, pentesting |
| **Data** | ML, statistics, pandas, SQL, pipelines |
| **DevOps** | CI/CD, containers, infrastructure, monitoring |
| **QA** | Testing, coverage, regression, fuzzing |

---

## Autonomous Optimization Loop

When run with `--mode autonomous`, MYCONEX performs a continuous 4-phase self-improvement cycle:

```
┌─────────────────────────────────────────────────────────┐
│                  Optimization Cycle                     │
│                                                         │
│  1. ANALYSE   LLM identifies improvement opportunity   │
│       ↓       in the MYCONEX codebase                  │
│  2. SANDBOX   Generates patch script, runs it in       │
│       ↓       resource-limited subprocess              │
│  3. VERIFY    LLM evaluates sandbox output for         │
│       ↓       correctness and safety                   │
│  4. RECORD    Appends lesson to lessons.md,            │
│               writes JSONL audit log                   │
└─────────────────────────────────────────────────────────┘
```

- Audit log: `~/.myconex/audit.jsonl`
- Metrics: `~/.myconex/metrics.json`
- Lessons: `lessons.md` (project root, injected into every system prompt)
- Session digests: `~/.myconex/session_YYYYMMDD.md`
- Reflection digest: every N cycles (configurable via `reflect_every_n`)

---

## Sandboxed Execution

`SandboxExecutor` provides resource-limited subprocess execution for agent-generated code:

```python
from tools.sandbox_executor import SandboxConfig, SandboxExecutor

executor = SandboxExecutor(
    default_config=SandboxConfig(
        timeout_s=30.0,
        max_memory_mb=512,
        max_processes=32,
    )
)

result = await executor.run_python("import math; print(math.pi)")
print(result.output)   # 3.141592653589793

# Parallel execution
results = await executor.run_parallel_python([
    "print('task 1')",
    "print('task 2')",
    "print('task 3')",
])
```

> **Note:** The sandbox applies `resource.setrlimit` for memory and process limits on Linux/macOS. It is not a container boundary — for stronger isolation, wrap in Docker or nsjail.

---

## Document Ingestion

```python
from tools.document_processor import DocumentProcessor

proc = DocumentProcessor()
result = await proc.process("/path/to/paper.pdf")
print(result.to_agent_payload(max_chars=4000))
```

Supported formats: `.pdf` (pdfminer.six or pdftotext CLI fallback), `.html`, `.htm`, `.txt`, `.md`

Scientific papers get automatic section extraction: Abstract, Introduction, Methods, Results, References.

---

## Intelligence Aggregation

```python
from tools.intel_aggregator import IntelAggregator, IntelSource

agg = IntelAggregator()
report = await agg.gather([
    IntelSource(type="web",   uri="https://example.com/article", label="article"),
    IntelSource(type="file",  uri="/data/report.txt",            label="report"),
    IntelSource(type="shell", uri="cat /var/log/system.log",     label="logs"),
    IntelSource(type="rss",   uri="https://feeds.example.com/",  label="feed"),
])

for r in report.results:
    print(f"[{r.label}] {r.content[:200]}")
```

Source types: `web`, `api`, `file`, `rss`, `search`, `db`, `shell`

---

## Running Tests

```bash
cd myconex
python3 -m pytest tests/test_myconex.py -v

# Or directly
python3 tests/test_myconex.py
```

Test groups:
- **A** — BaseAgent: delegation, routing, complexity scoring
- **B** — PersistentPythonREPL: state persistence, safety
- **C** — ContextFrame: token budgeting, pruning, hierarchy
- **D** — SessionMemory: pattern extraction, scoring
- **E** — DocumentProcessor: HTML/text parsing
- **F** — IntelAggregator: multi-source gathering
- **G** — AgentRoster: division management, routing
- **H** — SandboxExecutor: isolation, timeout, resource limits
- **I** — Config: load/override priority
- **J** — RLMAgent: task handling, decomposition (mocked LLM)

---

## Self-Improvement

MYCONEX tracks its own behavioral lessons in `lessons.md`. Every lesson is injected into the system prompt at session start, so corrections carry forward automatically without fine-tuning.

The self-improvement loop:
1. Detects a correction or confirmed improvement
2. Identifies the *class* of issue (not just the instance)
3. Writes a concise prevention rule to `lessons.md`
4. Applies the rule immediately in the current session

---

## Documentation

| Doc | Description |
|-----|-------------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System layers, component map, data flow |
| [`docs/FEATURES.md`](docs/FEATURES.md) | All features with lifecycle tags |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Architectural decisions and rationale |
| [`docs/MEMORY.md`](docs/MEMORY.md) | Runtime quirks and contributor gotchas |
| [`docs/TODO.md`](docs/TODO.md) | Prioritized task backlog |
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | Human-impact change history |
| [`docs/DISCORD_SETUP.md`](docs/DISCORD_SETUP.md) | Discord bot portal setup guide |

---

## License

See `LICENSE` file.
