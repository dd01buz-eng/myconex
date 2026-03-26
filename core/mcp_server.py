"""
MYCONEX MCP Server
-------------------
Exposes MYCONEX as a Model Context Protocol (MCP) server so Claude Desktop,
Cursor, Cline, and other MCP clients can query the knowledge base directly.

Protocol: MCP over stdio (JSON-RPC 2.0) — the standard for local MCP servers.

Tools exposed:
  search_knowledge  — semantic search over the vector knowledge base
  get_signals       — recent cross-source signals
  get_profile       — interest profile (top topics, project ideas)
  get_gaps          — open RAG knowledge gaps
  submit_text       — ingest a text note into the knowledge base
  web_search        — web search via the configured provider

Usage (add to Claude Desktop's mcpServers config):
  {
    "myconex": {
      "command": "python3",
      "args": ["/home/user/myconex/core/mcp_server.py"],
      "env": {}
    }
  }

Or run directly:
    cd ~/myconex && source venv/bin/activate && python core/mcp_server.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("mcp")

_BASE = Path.home() / ".myconex"


# ── JSON-RPC helpers ──────────────────────────────────────────────────────────

def _ok(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}

def _err(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# ── Tool definitions ──────────────────────────────────────────────────────────

_TOOLS = [
    {
        "name": "search_knowledge",
        "description": "Semantic search over MYCONEX knowledge base (emails, articles, videos, podcasts, documents)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_signals",
        "description": "Get recent cross-source signals — topics appearing across multiple content sources",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max signals (default 10)", "default": 10},
            },
        },
    },
    {
        "name": "get_profile",
        "description": "Get the user's interest profile: top topics, recent project ideas",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_gaps",
        "description": "Get open RAG knowledge gaps — queries that previously lacked good context",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max gaps (default 10)", "default": 10},
            },
        },
    },
    {
        "name": "submit_text",
        "description": "Ingest a text note into the MYCONEX knowledge base",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text":  {"type": "string", "description": "Text content to ingest"},
                "title": {"type": "string", "description": "Optional title"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web using the configured provider (SearXNG / Brave / DuckDuckGo)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":   {"type": "string", "description": "Search query"},
                "limit":   {"type": "integer", "description": "Max results (default 6)", "default": 6},
            },
            "required": ["query"],
        },
    },
]


# ── Tool handlers ─────────────────────────────────────────────────────────────

async def _handle_tool(name: str, args: dict) -> str:
    if name == "search_knowledge":
        return await _tool_search(args.get("query", ""), int(args.get("limit", 5)))
    elif name == "get_signals":
        return _tool_signals(int(args.get("limit", 10)))
    elif name == "get_profile":
        return _tool_profile()
    elif name == "get_gaps":
        return _tool_gaps(int(args.get("limit", 10)))
    elif name == "submit_text":
        return await _tool_submit_text(args.get("text", ""), args.get("title", ""))
    elif name == "web_search":
        return await _tool_web_search(args.get("query", ""), int(args.get("limit", 6)))
    else:
        raise ValueError(f"Unknown tool: {name}")


async def _tool_search(query: str, limit: int) -> str:
    try:
        from integrations.knowledge_store import search
        results = await search(query, limit=limit)
        if not results:
            return "No results found."
        lines = []
        for r in results:
            src   = r.get("source", "")
            title = r.get("title", "")[:80]
            text  = r.get("text", "")[:300]
            score = r.get("score", 0)
            lines.append(f"[{src}] {title} (score: {score:.2f})\n{text}")
        return "\n\n---\n\n".join(lines)
    except Exception as exc:
        return f"Search failed: {exc}"


def _tool_signals(limit: int) -> str:
    try:
        sigs = json.loads((_BASE / "signals_log.json").read_text()) if (_BASE / "signals_log.json").exists() else []
        sigs = sigs[-limit:][::-1]
        if not sigs:
            return "No signals detected yet."
        lines = []
        for s in sigs:
            topic   = s.get("topic", "")
            sources = ", ".join(s.get("sources", []))
            snippet = s.get("top_hit", "")[:200]
            lines.append(f"**{topic}** [{sources}]\n{snippet}")
        return "\n\n".join(lines)
    except Exception as exc:
        return f"Failed to load signals: {exc}"


def _tool_profile() -> str:
    try:
        p = json.loads((_BASE / "interest_profile.json").read_text()) if (_BASE / "interest_profile.json").exists() else {}
        topics = sorted(p.get("topics", {}).items(), key=lambda x: -x[1])[:15]
        ideas  = p.get("project_ideas", [])[:5]
        out = []
        if topics:
            out.append("Top topics: " + ", ".join(f"{t}({c})" for t, c in topics))
        if ideas:
            out.append("Recent project ideas:\n" + "\n".join(f"• {i[:100]}" for i in ideas))
        return "\n\n".join(out) or "No profile data yet."
    except Exception as exc:
        return f"Failed to load profile: {exc}"


def _tool_gaps(limit: int) -> str:
    try:
        from core.rag_repair import get_open_gaps
        gaps = get_open_gaps(max_n=limit)
        if not gaps:
            return "No open knowledge gaps."
        lines = [f"• {g['query'][:100]} (best match: {g.get('max_rag_score', '?')})" for g in gaps]
        return f"{len(gaps)} open gap(s):\n" + "\n".join(lines)
    except Exception as exc:
        return f"Failed to load gaps: {exc}"


async def _tool_submit_text(text: str, title: str) -> str:
    try:
        from integrations.knowledge_store import store_item
        item_id = await store_item(text=text, source="mcp", title=title or "MCP Note")
        return f"Stored as {item_id}" if item_id else "Stored (no ID returned)."
    except Exception as exc:
        return f"Failed to store: {exc}"


async def _tool_web_search(query: str, limit: int) -> str:
    try:
        from integrations.search_provider import web_search, format_results
        results = await web_search(query, max_results=limit)
        return format_results(results)
    except Exception as exc:
        return f"Search failed: {exc}"


# ── MCP request dispatcher ────────────────────────────────────────────────────

async def _dispatch(req: dict) -> dict | None:
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params") or {}

    if method == "initialize":
        return _ok(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "myconex", "version": "1.0.0"},
        })

    elif method == "tools/list":
        return _ok(req_id, {"tools": _TOOLS})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments") or {}
        try:
            result = await _handle_tool(tool_name, tool_args)
            return _ok(req_id, {
                "content": [{"type": "text", "text": result}],
                "isError": False,
            })
        except Exception as exc:
            return _ok(req_id, {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            })

    elif method == "notifications/initialized":
        return None  # no response for notifications

    else:
        if req_id is not None:
            return _err(req_id, -32601, f"Method not found: {method}")
        return None


# ── Main stdio loop ───────────────────────────────────────────────────────────

async def _run() -> None:
    logger.info("MYCONEX MCP server starting on stdio")
    loop = asyncio.get_running_loop()

    reader = asyncio.StreamReader()
    proto  = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: proto, sys.stdin)

    while True:
        try:
            line = await reader.readline()
            if not line:
                break
            req = json.loads(line.decode())
            response = await _dispatch(req)
            if response is not None:
                _send(response)
        except asyncio.CancelledError:
            break
        except json.JSONDecodeError:
            _send(_err(None, -32700, "Parse error"))
        except Exception as exc:
            logger.error("dispatch error: %s", exc)
            _send(_err(None, -32603, str(exc)))


if __name__ == "__main__":
    asyncio.run(_run())
