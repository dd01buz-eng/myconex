# MYCONEX — Architecture

## Overview

MYCONEX is a distributed AI mesh system inspired by fungal networks. Heterogeneous machines collaborate as a single intelligent system, routing work to where it runs best. The primary metaphor: every node contributes what it can; computation flows like nutrients through mycelium.

---

## System Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Input Gateways                               │
│  Discord Gateway │ REST API (:8765) │ CLI REPL │ Mesh Gateway       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                     Orchestration Layer                              │
│  RLMAgent (top-level orchestrator)                                  │
│  ├── TaskRouter (agent lifecycle + routing)                         │
│  ├── AgentRoster (6-division specialist pool)                       │
│  └── Complexity Scorer (threshold: 0.60 → decompose)               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      Inference Layer (MoE Chain)                    │
│  flash-moe (C/Metal) → Nous 8B → Nous 70B → OpenRouter → Ollama   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      Mesh / State Layer                             │
│  NATS pub/sub │ Redis (state) │ Qdrant (vectors) │ mDNS discovery  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### Entry Points

| File | Purpose |
|------|---------|
| `__main__.py` | Unified entry (`python -m myconex --mode X`) |
| `main.py` | Legacy launcher (backward-compat with systemd) |
| `config.py` | Unified config with env var / yaml / default priority |

### Orchestration (`orchestration/`)

| File | Purpose |
|------|---------|
| `agents/base_agent.py` | `BaseAgent` — abstract base, `delegate()`, `delegate_parallel()`, `AgentRole` |
| `agents/rlm_agent.py` | `RLMAgent` — primary orchestrator, context management, decomposition |
| `agents/context_manager.py` | `ContextFrame`, `SessionMemory`, `PersistentMemory` |
| `agents/hermes_agent.py` | Nous-Hermes GGUF agent wrapper |
| `workflows/task_router.py` | `TaskRouter` — registers agents, routes tasks, manages lifecycle |
| `agent_roster.py` | 6-division specialist roster |

### Core (`core/`)

| Path | Purpose |
|------|---------|
| `gateway/discord_gateway.py` | Discord bot → RLMAgent wiring, slash commands, clarify callbacks |
| `gateway/api_gateway.py` | REST API gateway (:8765) |
| `gateway/agentic_tools.py` | Tool registry and handlers |
| `gateway/python_repl.py` | `PersistentPythonREPL`, `REPLPool`, `CodebaseIndex` |
| `gateway/mesh_gateway.py` | Inter-node mesh communication |
| `gateway/chat_history_retriever.py` | Chat history access |
| `gateway/self_improvement.py` | Self-improvement pipeline |
| `coordinator/orchestrator.py` | Mesh task lifecycle, node roles, topology state |
| `discovery/mesh_discovery.py` | mDNS peer discovery |
| `messaging/nats_client.py` | NATS pub/sub client, heartbeat |
| `memory/vector_store.py` | Qdrant vector store / RAG |
| `classifier/hardware.py` | Hardware detection → tier assignment |
| `autonomous_loop.py` | 4-phase self-optimization loop |
| `mcp_client.py` | MCP protocol client |
| `self_healer.py` | Automatic fault recovery |
| `novelty_scanner.py` | Detects novel/interesting signals |
| `plugin_loader.py` | Dynamic plugin loading |
| `metrics.py` | Metrics collection |
| `digest.py` | Session digest generation |
| `notifications.py` | Notification routing |

### Tools (`tools/`)

| File | Purpose |
|------|---------|
| `sandbox_executor.py` | `SandboxExecutor` — resource-limited subprocess execution |
| `document_processor.py` | PDF/HTML/text ingestion |
| `intel_aggregator.py` | Multi-source intelligence gathering (web, rss, file, shell, db) |

### Integrations (`integrations/`)

| File | Purpose |
|------|---------|
| `moe_hermes_integration.py` | `HermesMoEAgent` — MoE primary backend |
| `hermes-agent/` | Nous-Hermes GGUF runner |
| `flash-moe/` | C/Metal flash-moe inference |
| `dlam_client.py` | DLAM integration |
| `fabric_client.py` | Fabric AI integration |
| `knowledge_store.py` | External knowledge store |
| `hermes_bridge.py` | Bridge to hermes-agent process |
| `rss_monitor.py` | RSS feed monitoring |
| `signal_detector.py` | Signal/anomaly detection |
| `gmail_reader.py` | Gmail ingestion |
| `email_ingester.py` | Email ingestion pipeline |
| `youtube_ingester.py` | YouTube transcript ingestion |
| `podcast_ingester.py` | Podcast audio ingestion |

---

## Tier System

Hardware is auto-detected at startup and assigned a tier that governs which models run locally:

| Tier | Hardware | Primary Model |
|------|----------|---------------|
| T1 | 70B+ GPU | Nous-Hermes 70B (flash-moe) |
| T2 | 8B GPU | Nous-Hermes 8B |
| T3 | CPU only | Ollama fallback |
| T4 | Edge/IoT | BitNet 1-bit |

The `MeshOrchestrator` in `core/coordinator/orchestrator.py` routes `MeshTask` objects to the best available peer based on tier and required role.

---

## Data Flow: Discord Message → Response

```
1. Discord message received
   └── discord_gateway.py: DiscordGateway.on_message()

2. Check if RAG needed (skip on trivial messages → ~350ms saved)
   └── chat_history_retriever.py: fetch recent context

3. Build task payload, route to RLMAgent
   └── task_router.py: TaskRouter.route(type="chat", payload={...})

4. RLMAgent scores complexity
   ├── score < 0.60: handle directly with MoE chain
   └── score ≥ 0.60: decompose → delegate_parallel() to specialist agents

5. MoE inference chain (first successful result wins)
   flash-moe → Nous 8B → Nous 70B → OpenRouter → Ollama

6. Response returned via clarify_callback or direct reply
   └── discord_gateway.py: send to channel/thread
```

---

## Autonomous Loop

When run with `--mode autonomous`, MYCONEX cycles through 4 phases:

```
ANALYSE  → LLM identifies improvement opportunity in codebase
SANDBOX  → Generates patch script, runs in SandboxExecutor
VERIFY   → LLM evaluates sandbox output for correctness/safety
RECORD   → Appends lesson to lessons.md, writes JSONL audit log
```

Lessons from `lessons.md` are injected into every subsequent system prompt, closing the self-improvement loop.

---

## Configuration Priority

1. Environment variables (`MYCONEX_*` prefix) — highest
2. `.env` file (project root)
3. `config/mesh_config.yaml`
4. Dataclass defaults — lowest

---

## External Services

| Service | Default | Purpose |
|---------|---------|---------|
| NATS | `nats://localhost:4222` | Mesh pub/sub messaging |
| Redis | `redis://localhost:6379` | Shared mesh state |
| Qdrant | `http://localhost:6333` | Vector store / RAG |
| Ollama | `http://localhost:11434` | Local LLM inference fallback |
| LiteLLM | `http://localhost:4000` | LLM proxy |
| Discord | token via env | Bot gateway |

All services are optional — MYCONEX degrades gracefully when they're unavailable.

---

## Agent Divisions

| Division | Specialties |
|----------|-------------|
| Engineering | Python, Go, APIs, architecture, refactoring |
| Research | Literature review, summarization, analysis |
| Security | Threat modeling, auditing, pentesting |
| Data | ML, statistics, pandas, SQL, pipelines |
| DevOps | CI/CD, containers, infrastructure, monitoring |
| QA | Testing, coverage, regression, fuzzing |

---

## Key Invariants

- Agents must subclass `BaseAgent` and implement `can_handle()` + `handle_task()`
- New agents get `set_router()` called automatically on `TaskRouter.register_agent()`
- Never import `moe_hermes_integration` from `base_agent.py` — circular dependency
- Async all the way: no `asyncio.run()` inside agents; use `await`
- MoE decomposition threshold: `0.60` (see `rlm_agent.py:DECOMPOSE_THRESHOLD`)
