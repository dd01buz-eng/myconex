# MYCONEX — Architectural Decisions

---

## Recursive Language Model (RLM) as Primary Orchestrator

**Decision:** `RLMAgent` manages its own context, scores task complexity, and decomposes above threshold 0.60 rather than using a fixed pipeline.

**Why:** Traditional fixed pipelines can't adapt to task complexity. RLM allows the system to decide at runtime whether to handle a task directly or decompose it — reducing unnecessary token usage on simple tasks and enabling parallelism on complex ones.

**Alternatives:** Fixed multi-agent chains (LangChain-style); rejected because they add latency and don't adapt to task difficulty.

**Consequences:** Complexity scoring quality directly affects performance. If scoring is miscalibrated, simple tasks get over-decomposed (slow) or complex tasks aren't decomposed (poor quality).

---

## Mixture-of-Experts Chain with Ordered Fallback

**Decision:** Inference chain: flash-moe (C/Metal) → Nous 8B → Nous 70B → OpenRouter → Ollama. First successful result wins.

**Why:** No single model is available on all hardware. The chain lets the system use the fastest/cheapest option that works, gracefully degrading when specialized runners aren't available.

**Alternatives:** Round-robin or load-balanced routing; rejected because ordering by speed/capability is more predictable and simpler to debug.

**Consequences:** The first model in the chain that can serve a request always wins, even if a later model would produce better output. Trade-off: latency vs. quality.

---

## Tier-Aware Hardware Routing (T1–T4)

**Decision:** Auto-detect hardware at startup, assign a tier (T1=70B GPU, T2=8B GPU, T3=CPU, T4=edge), and route mesh tasks to appropriate nodes.

**Why:** MYCONEX runs on heterogeneous hardware. A Raspberry Pi shouldn't receive 70B inference tasks. Tier assignment makes routing deterministic without requiring manual configuration.

**Alternatives:** Capability negotiation at request time; rejected as too complex for initial implementation.

**Consequences:** Tiers are coarse — a T2 node with a fast GPU might be better than a slow T1. Can be overridden via `MYCONEX_TIER` env var.

---

## NATS for Mesh Messaging (Not HTTP)

**Decision:** Use NATS pub/sub for inter-node communication rather than REST HTTP calls between nodes.

**Why:** NATS is designed for distributed systems: it handles fan-out, routing, and at-most-once delivery natively. HTTP would require each node to know all other nodes' addresses and manage retries manually.

**Alternatives:** gRPC, ZeroMQ, MQTT; NATS chosen for simplicity and good Python support.

**Consequences:** NATS is a required external service for multi-node mesh mode. Single-node mode works without it.

---

## Self-Improvement via lessons.md Injection

**Decision:** Inject `lessons.md` into every system prompt so the model learns from past corrections without fine-tuning.

**Why:** Fine-tuning is expensive and slow. Prompt injection is immediate, zero-cost, and auditable — any human can read the lesson log and understand what behavior was corrected.

**Alternatives:** Few-shot examples in config; rejected because they'd need manual curation. Vector retrieval of relevant lessons; considered but adds latency and complexity for modest benefit.

**Consequences:** System prompt grows with every lesson added. Must monitor prompt size vs. context budget (`MYCONEX_CONTEXT_BUDGET`). Stale/conflicting lessons should be marked `[RESOLVED]` rather than deleted.

---

## RAG Skip on Trivial Messages

**Decision:** Skip Qdrant vector query when a message is classified as trivial (short, conversational), saving ~350ms per request.

**Why:** RAG adds latency that isn't justified for simple exchanges like "thanks" or "ok". Classifier is a fast heuristic, not LLM-based.

**Alternatives:** Always query RAG; rejected because latency matters for Discord UX.

**Consequences:** Trivial classification must not be overly aggressive — topic-relevant short messages could miss context. Threshold should be tunable.

---

## Discord as Primary Human Interface

**Decision:** Discord bot is the primary interactive interface; CLI and REST API are secondary.

**Why:** The target users are developers/researchers who live in Discord. Discord provides threading, slash commands, and persistent chat history for free.

**Alternatives:** Dedicated web UI; mobile app scaffold exists but is lower priority.

**Consequences:** Discord bot token is required for interactive use in team settings. CLI mode is available for local/headless use.

---

## No __init__.py Files Unless Package Boundary Needed

**Decision:** Avoid adding `__init__.py` to directories unless a true package boundary is needed.

**Why:** Unnecessary `__init__.py` files create implicit package coupling and complicate imports. Python 3.3+ supports namespace packages without them.

**Alternatives:** Always add `__init__.py` (conventional Python packaging style); rejected to keep the repo lightweight.

**Consequences:** Some directories that look like packages (`tools/`, `integrations/`) are actually namespace packages. Import paths work but are sensitive to `PYTHONPATH` setup.

---

## BaseAgent Cannot Import moe_hermes_integration

**Decision:** `base_agent.py` must never import from `moe_hermes_integration.py`.

**Why:** `moe_hermes_integration` imports from `base_agent`, creating a circular dependency. Discovered in practice (see `lessons.md`).

**Alternatives:** Restructure to break the cycle; considered but adds complexity. Current approach: complexity scoring lives in `base_agent._estimate_complexity()` independently.

**Consequences:** Any shared logic between `base_agent` and `moe_hermes_integration` must be extracted to a third module or passed via dependency injection.
