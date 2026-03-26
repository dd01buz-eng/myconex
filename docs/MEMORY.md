# MYCONEX — Project Memory

Runtime quirks, gotchas, and non-obvious behaviors for contributors.

---

## Circular Import: base_agent ↔ moe_hermes_integration

`moe_hermes_integration.py` imports from `base_agent.py`. Importing in the other direction causes a circular import at module load time. The fix is to use `TYPE_CHECKING` guards or pass dependencies via injection. Complexity scoring in `base_agent` uses a local `_estimate_complexity()` function for this reason.

---

## asyncio.get_event_loop() is Deprecated in Python 3.10+

Use `asyncio.get_running_loop()` inside async functions. Use `asyncio.new_event_loop()` / `asyncio.run()` in sync entry points. Calling `asyncio.get_event_loop().run_until_complete()` inside an already-running event loop raises `RuntimeError: This event loop is already running`.

---

## Agents Without a Router Can't Delegate

Agents created outside a `TaskRouter` (e.g., in tests or scripts) have `_router = None`. Calling `delegate()` on them returns an error result. Always create agents via `TaskRouter` or call `agent.set_router(router)` manually before delegating.

---

## RAG Skip Heuristic

Qdrant vector queries are skipped for "trivial" messages to save ~350ms. The heuristic is a fast classifier, not an LLM call. If a short but topically relevant message is getting poor context, check `chat_history_retriever.py` and the trivial-message classifier.

---

## lessons.md is Injected into Every System Prompt

All lessons in `lessons.md` are loaded at session start and included in the RLMAgent system prompt. As the file grows, monitor against `MYCONEX_CONTEXT_BUDGET` (default: 16384 tokens). Mark superseded lessons `[RESOLVED]` rather than deleting them.

---

## Discord Bot Application ID

Application ID: `1469408384070586432`. Bot token must be in `.env` as `DISCORD_BOT_TOKEN`. See `docs/DISCORD_SETUP.md` for full portal setup.

---

## Slash Commands Auto-Sync on Ready

Slash commands (`/ask`, `/reset`, `/status`, `/tier`) are synced to Discord on `on_ready`. Global sync can take up to 1 hour. For instant sync during development, switch to guild-level sync.

---

## Config Resolution Order

Priority (highest to lowest): env vars (`MYCONEX_*`) → `.env` file → `config/mesh_config.yaml` → dataclass defaults. When debugging unexpected config values, check all four sources.

---

## SandboxExecutor Isolation Limits

`SandboxExecutor` applies `resource.setrlimit` for memory and process limits on Linux/macOS only. It is NOT a container boundary. For stronger isolation in production, wrap in Docker or nsjail.

---

## MoE Chain Stops at First Success

The inference chain (flash-moe → Nous 8B → Nous 70B → OpenRouter → Ollama) returns the first successful result. If flash-moe is available but produces poor output, subsequent models in the chain won't be tried. There is no quality-based fallback — only availability-based.

---

## Complexity Threshold is 0.60

Tasks scoring above `0.60` from `_score_complexity()` are automatically decomposed into parallel sub-tasks. Adjust `DECOMPOSE_THRESHOLD` in `rlm_agent.py` if tasks are being over- or under-decomposed.

---

## No __init__.py in tools/ and integrations/

These directories use Python namespace packages. Imports work but are sensitive to `PYTHONPATH`. Don't add `__init__.py` unless you're explicitly creating a package boundary.

---

## main.py vs __main__.py

`__main__.py` is the canonical entry point (`python -m myconex`). `main.py` is the legacy launcher kept for backward compatibility with systemd scripts. New features should go in `__main__.py`.

---

## Persistent Memory Location

Cross-session memory is stored in `~/.myconex/memory/`. Audit logs are at `~/.myconex/audit.jsonl`. Session digests at `~/.myconex/session_YYYYMMDD.md`. Metrics at `~/.myconex/metrics.json`.
