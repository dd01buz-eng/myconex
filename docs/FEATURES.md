# MYCONEX — Features

## Core Mesh System

- [stable] Distributed AI mesh with tier-aware routing (T1–T4 by hardware)
- [stable] NATS pub/sub messaging between nodes
- [stable] mDNS peer discovery (`core/discovery/mesh_discovery.py`)
- [stable] Redis state store for shared mesh state
- [stable] Hardware auto-detection at startup (`core/classifier/hardware.py`)

## Agent System

- [stable] `BaseAgent` with `delegate()` / `delegate_parallel()` API
- [stable] `RLMAgent` — top-level recursive orchestrator with context management
- [stable] `TaskRouter` — agent lifecycle management and routing
- [stable] Agent roster with 6 divisions (Engineering, Research, Security, Data, DevOps, QA)
- [stable] Complexity scoring — tasks above 0.60 auto-decompose
- [stable] Agent roles: MANAGER, WORKER, SPECIALIST
- [stable] `HermesMoEAgent` — Mixture-of-Experts primary backend
- [stable] `hermes_agent.py` — Nous-Hermes GGUF agent wrapper

## LLM Backends / Inference

- [stable] MoE expert chain: flash-moe (C/Metal) → Nous 8B → Nous 70B → OpenRouter → Ollama fallback
- [stable] Ollama backend support
- [stable] llama.cpp backend support
- [stable] LM Studio backend support
- [stable] LiteLLM proxy support
- [stable] GGUF local inference via llama-cpp-python (`agentic_tools.py:gguf_infer`)
- [stable] Hermes-agent GGUF runner (`integrations/hermes-agent/`)
- [stable] flash-moe C/Metal runner (`integrations/flash-moe/`)

## Gateways / Interfaces

- [stable] Discord gateway with slash commands (`/ask`, `/reset`, `/status`, `/tier`)
- [stable] Discord clarify callback — agent can ask follow-up questions via Discord
- [stable] REST API gateway on `:8765` (`core/gateway/api_gateway.py`)
- [stable] CLI interactive REPL (`--mode cli`)
- [stable] Mesh gateway for inter-node communication (`core/gateway/mesh_gateway.py`)
- [stable] MCP client integration (`core/mcp_client.py`)

## Dashboard

- [stable] Web dashboard with Watch Mode (`dashboard/app.py`)

## Autonomous / Self-Improvement

- [stable] 4-phase autonomous optimization loop (`core/autonomous_loop.py`): Analyse → Sandbox → Verify → Record
- [stable] Self-improvement via `lessons.md` injection into every system prompt
- [stable] Session digests written to `~/.myconex/session_YYYYMMDD.md`
- [stable] JSONL audit log at `~/.myconex/audit.jsonl`
- [stable] Metrics tracking at `~/.myconex/metrics.json`
- [stable] Self-healer (`core/self_healer.py`)
- [stable] Novelty scanner (`core/novelty_scanner.py`)

## Memory / Context

- [stable] `ContextFrame` with token budgeting and pruning
- [stable] `SessionMemory` — pattern extraction and scoring
- [stable] `PersistentMemory` — cross-session storage in `~/.myconex/memory/`
- [stable] Vector store / RAG via Qdrant (`core/memory/vector_store.py`)
- [stable] RAG query skip on trivial messages (~350ms savings)
- [stable] Chat history retrieval (`core/gateway/chat_history_retriever.py`)

## Tools

- [stable] `python_repl` — persistent session-keyed Python REPL (`core/gateway/python_repl.py`)
- [stable] `web_read` — fetch and extract structured content from URLs
- [stable] `codebase_search` — search MYCONEX codebase by keyword
- [stable] `gguf_infer` — local GGUF model inference
- [stable] `memory_store` / `memory_retrieve` — persistent cross-session memory
- [stable] `delegate` — sub-task delegation to specialized agents
- [stable] `SandboxExecutor` — resource-limited subprocess execution (`tools/sandbox_executor.py`)
- [stable] `DocumentProcessor` — PDF/HTML/text ingestion (`tools/document_processor.py`)
- [stable] `IntelAggregator` — multi-source intelligence gathering (`tools/intel_aggregator.py`)

## Integrations

- [stable] DLAM client (`integrations/dlam_client.py`)
- [stable] Fabric client (`integrations/fabric_client.py`)
- [stable] Knowledge store (`integrations/knowledge_store.py`)
- [stable] RSS monitor (`integrations/rss_monitor.py`)
- [stable] Signal detector (`integrations/signal_detector.py`)
- [stable] Buzlock bot (`buzlock_bot.py`)
- [beta] Gmail reader (`integrations/gmail_reader.py`)
- [beta] Email ingester (`integrations/email_ingester.py`)
- [beta] YouTube ingester (`integrations/youtube_ingester.py`)
- [beta] Podcast ingester (`integrations/podcast_ingester.py`)

## Infrastructure / Deployment

- [stable] Docker Compose stack: NATS, Redis, Qdrant, LiteLLM (`services/`)
- [stable] Systemd unit files (`services/systemd/`)
- [stable] Mobile app scaffold (`mobile/`) — Capacitor + Vite + Tailwind
- [stable] Plugin loader (`core/plugin_loader.py`)
- [stable] Notifications system (`core/notifications.py`)
- [stable] Metrics collection (`core/metrics.py`)
- [stable] Digest generation (`core/digest.py`)

## Run Modes

- [stable] `cli` — interactive REPL
- [stable] `discord` — Discord gateway + mesh node
- [stable] `api` — REST API + mesh node
- [stable] `autonomous` — self-improving autonomous loop
- [stable] `worker` — background mesh worker
- [stable] `full` — all modes simultaneously
