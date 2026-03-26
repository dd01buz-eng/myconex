# MYCONEX — Changelog

Most recent changes first. Human-impact focus, not git messages.

---

## [2026-03] RAG skip on trivial messages

Queries to Qdrant are now skipped when a message is classified as trivial (short, conversational). Saves ~350ms per lightweight reply. Classifier is a fast heuristic.

*Commit: ae06772*

---

## [2026-03] Self-improvement loop fully closed

`lessons.md` is now injected into every system prompt, so behavioral corrections from one session carry forward automatically. No more repeated mistakes across sessions.

*Commit: 051a77b*

---

## [2026-03] DLAM, Buzlock bot, dashboard Watch Mode, full tool suite

Major feature addition:
- DLAM client integration (`integrations/dlam_client.py`)
- Buzlock bot (`buzlock_bot.py`)
- Dashboard with Watch Mode (`dashboard/app.py`)
- Full agentic tool suite wired in

*Commit: 90452fe*

---

## [2026-03] Agent clarify callbacks via Discord

Agent can now ask follow-up clarifying questions via Discord mid-conversation rather than proceeding with ambiguous instructions. Implemented via `clarify_callback` in `discord_gateway.py`.

*Commit: fa92ee0*

---

## [2026-03] Discord gateway rewritten with hermes-agent provider resolution

Discord gateway now uses the hermes-agent provider to resolve LLM calls, enabling the full MoE chain from Discord. Full tool loop added.

*Commits: a1aa70d, 37e656d*

---

## [2026-03] flash-moe + hermes-agent as primary MoE backend

Replaced prior LLM backend with the flash-moe (C/Metal) → Nous 8B → Nous 70B → OpenRouter → Ollama chain. `HermesMoEAgent` is now the primary inference layer.

*Commit: 9628c24*

---

## [2026-03] X/Twitter bookmarks collected for Hermes agent reference

78 bookmarks with URLs, authors, timestamps, and content added to `docs/bookmarks.json` for RAG/reference use.

*Commit: 42a1e5c*

---

## [2026-03] Initial commit

Core MYCONEX project: AI Discord bot with orchestration, mesh networking, and DLAM integration. Included: `BaseAgent`, `RLMAgent`, `TaskRouter`, `AgentRoster`, Discord gateway, REST API, CLI REPL, autonomous loop, NATS/Redis/Qdrant services.

*Commit: 3fa36f2*
