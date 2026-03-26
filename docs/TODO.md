# MYCONEX — TODO

---

## WIP

*(no active multi-session tasks)*

---

## High

- Wire `SandboxExecutor` resource limits for Windows (currently Linux/macOS only via `resource.setrlimit`)
- Add quality-based fallback in MoE chain (not just availability-based)
- Tune trivial-message RAG skip classifier to avoid dropping relevant short messages

---

## Medium

- Mobile app (`mobile/`) — Capacitor + Vite scaffold exists, not connected to backend
- Flesh out `gmail_reader.py` / `email_ingester.py` (currently beta)
- Add guild-level slash command sync option for faster Discord development iteration
- Add `lessons.md` size monitor — warn when approaching context budget
- Document `core/mcp_client.py` usage and supported MCP servers

---

## Low

- Migrate `main.py` (legacy) users to `__main__.py` once systemd scripts are updated
- Add `MYCONEX_DECOMPOSE_THRESHOLD` env var so complexity threshold is configurable without code changes
- `podcast_ingester.py` and `youtube_ingester.py` — complete implementation (currently beta stubs)

---

## Long-term

- Container-grade sandboxing for `SandboxExecutor` (Docker/nsjail wrapping)
- Quality-scoring feedback loop: record which MoE chain member produced the best result per task type
- Multi-node mesh testing harness (current tests are single-node only)
- Plugin marketplace / discovery for `core/plugin_loader.py`
- Autonomous task generation quality — ensure self-generated tasks are useful, not circular
