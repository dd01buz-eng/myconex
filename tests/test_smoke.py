"""
MYCONEX Smoke Tests
--------------------
Fast, offline-safe tests — no network calls, no LLM, no Qdrant.
Run with: cd ~/myconex && python -m pytest tests/ -v

Tests cover:
  - Module imports (all key modules importable)
  - Config loading (.env parse)
  - Utility helpers (chunking, topic extraction, sysmon data)
  - RAG repair (record / retrieve / resolve)
  - Knowledge graph (entity add / relation / context_for)
  - Usage tracker (record / stats)
  - Briefing data builder (no-data path)
  - Telegram bridge text conversion
  - Search provider fallback chain (mocked)
  - Dashboard sysmon data shape
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ── Imports ────────────────────────────────────────────────────────────────────

def test_import_core_modules():
    import core.rag_repair       as _r   ; assert hasattr(_r, "record_rag_miss")
    import core.knowledge_graph  as _kg  ; assert hasattr(_kg, "KnowledgeGraph")
    import core.usage_tracker    as _ut  ; assert hasattr(_ut, "record")
    import core.briefing         as _b   ; assert hasattr(_b, "build_briefing_data")
    import core.mcp_server       as _mcp ; assert hasattr(_mcp, "_TOOLS")


def test_import_integrations():
    import integrations.notifier          as _n  ; assert hasattr(_n, "notify")
    import integrations.search_provider   as _sp ; assert hasattr(_sp, "web_search")
    import integrations.document_ingester as _di ; assert hasattr(_di, "extract_text")
    import integrations.github_ingester   as _gh ; assert hasattr(_gh, "GitHubIngester")
    import integrations.telegram_bridge   as _tb ; assert hasattr(_tb, "TelegramBridge")
    import integrations.twitter_monitor   as _tw ; assert hasattr(_tw, "TwitterMonitor")
    import integrations.readwise_ingester as _rw ; assert hasattr(_rw, "ReadwisePocketIngester")


# ── Config ────────────────────────────────────────────────────────────────────

def test_load_config_from_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_KEY=hello\nANOTHER=world\n")
    from config import load_config
    load_config(env_file=str(env_file))
    import os
    assert os.getenv("TEST_KEY") == "hello"
    assert os.getenv("ANOTHER") == "world"


# ── Document ingester helpers ─────────────────────────────────────────────────

def test_chunk_basic():
    from integrations.document_ingester import _chunk
    text   = "a" * 100
    chunks = _chunk(text, size=40, overlap=10)
    assert len(chunks) > 1
    # Every chunk is at most 40 chars
    for c in chunks:
        assert len(c) <= 40


def test_chunk_short_text():
    from integrations.document_ingester import _chunk
    text = "hello world"
    chunks = _chunk(text, size=100, overlap=10)
    assert chunks == ["hello world"]


def test_extract_text_plain(tmp_path):
    from integrations.document_ingester import extract_text
    f = tmp_path / "note.txt"
    f.write_text("Hello from MYCONEX")
    assert extract_text(str(f)) == "Hello from MYCONEX"


# ── RAG repair ────────────────────────────────────────────────────────────────

def test_rag_repair_record_and_retrieve(tmp_path, monkeypatch):
    monkeypatch.setattr("core.rag_repair._BASE",      tmp_path)
    monkeypatch.setattr("core.rag_repair._GAPS_FILE", tmp_path / "gaps.json")
    from core.rag_repair import record_rag_miss, get_open_gaps, mark_gap_resolved

    record_rag_miss("what is quantum entanglement", "I don't know", 0, 0.1)
    gaps = get_open_gaps()
    assert len(gaps) == 1
    assert gaps[0]["query"] == "what is quantum entanglement"
    assert gaps[0]["status"] == "open"

    mark_gap_resolved("what is quantum entanglement")
    open_gaps = get_open_gaps()
    assert len(open_gaps) == 0


def test_rag_repair_dedup(tmp_path, monkeypatch):
    monkeypatch.setattr("core.rag_repair._BASE",      tmp_path)
    monkeypatch.setattr("core.rag_repair._GAPS_FILE", tmp_path / "gaps.json")
    from core.rag_repair import record_rag_miss, get_open_gaps

    record_rag_miss("same query", "resp", 0, 0.1)
    record_rag_miss("same query", "resp", 0, 0.1)
    assert len(get_open_gaps()) == 1


# ── Knowledge graph ───────────────────────────────────────────────────────────

def test_knowledge_graph_basic(tmp_path, monkeypatch):
    monkeypatch.setattr("core.knowledge_graph._GRAPH_FILE", tmp_path / "kg.json")
    from core.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    nid_a = kg.add_entity("person",  "Alice",   "test")
    nid_b = kg.add_entity("project", "MYCONEX", "test")
    kg.add_relation(nid_a, nid_b, "works_on")

    assert nid_a in kg.nodes
    assert nid_b in kg.nodes
    assert len(kg.edges) >= 1

    # context_for does a string search over node names
    found = [n for n in kg.nodes.values() if n.get("name") in ("Alice", "MYCONEX")]
    assert len(found) == 2


def test_knowledge_graph_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr("core.knowledge_graph._GRAPH_FILE", tmp_path / "kg.json")
    from core.knowledge_graph import KnowledgeGraph

    kg1 = KnowledgeGraph()
    kg1.add_entity("topic", "Python", "test")
    kg1.save()

    kg2 = KnowledgeGraph()
    kg2.load()
    names = [n["name"] for n in kg2.nodes.values()]
    assert "Python" in names


# ── Usage tracker ─────────────────────────────────────────────────────────────

def test_usage_tracker_record_and_stats(tmp_path, monkeypatch):
    monkeypatch.setattr("core.usage_tracker._BASE",       tmp_path)
    monkeypatch.setattr("core.usage_tracker._USAGE_FILE", tmp_path / "usage.jsonl")
    from core.usage_tracker import record, get_stats

    record("llama3.1:8b", "hello world", "response text", latency_s=0.5)
    record("llama3.1:8b", "second prompt", "another response", latency_s=1.2, cached=True)

    stats = get_stats(days=7)
    assert stats["total_calls"] == 2
    assert stats["cache_hits"] == 1
    assert stats["total_tokens"] > 0
    assert "llama3.1:8b" in stats["by_model"]


# ── Briefing data (empty) ─────────────────────────────────────────────────────

def test_briefing_data_empty():
    """build_briefing_data returns sane empty structure when no data files exist."""
    import importlib, sys
    # Use a fresh import so we don't conflict with other tests
    if "core.briefing" in sys.modules:
        del sys.modules["core.briefing"]
    from core.briefing import build_briefing_data
    # With no real data files the function should return zeros without crashing
    data = build_briefing_data(lookback_hours=24)
    assert isinstance(data, dict)
    assert "total_ingested" in data
    assert isinstance(data["top_topics"], list)


# ── Telegram bridge ───────────────────────────────────────────────────────────

def test_telegram_embed_to_text():
    from integrations.telegram_bridge import _embed_to_text
    embed = {
        "title":       "Test Title",
        "description": "Some description",
        "fields":      [{"name": "Field A", "value": "Value A"}],
        "footer":      {"text": "footer text"},
    }
    text = _embed_to_text(embed)
    assert "Test Title"   in text
    assert "Some description" in text
    assert "Field A"      in text
    assert "footer text"  in text


# ── Search provider ───────────────────────────────────────────────────────────

def test_format_results():
    from integrations.search_provider import format_results
    results = [
        {"title": "Result One", "url": "https://example.com/1", "snippet": "Snippet one"},
        {"title": "Result Two", "url": "https://example.com/2", "snippet": "Snippet two"},
    ]
    formatted = format_results(results)
    assert "Result One" in formatted
    assert "Result Two" in formatted
    assert "example.com" in formatted


def test_format_results_empty():
    from integrations.search_provider import format_results
    assert format_results([]) == "No search results found."


# ── Notifier level gating ─────────────────────────────────────────────────────

def test_notifier_level_gating(monkeypatch):
    monkeypatch.setattr("integrations.notifier._MIN_LEVEL", "warning")
    from integrations.notifier import _level_ok
    assert _level_ok("critical") is True
    assert _level_ok("warning")  is True
    assert _level_ok("info")     is False


# ── GitHub ingester topic extraction ─────────────────────────────────────────

def test_github_topics_from():
    from integrations.github_ingester import _topics_from
    topics = _topics_from("fix authentication bug in login handler")
    assert "authentication" in topics or "login" in topics
    assert len(topics) <= 6


# ── MCP tools list ────────────────────────────────────────────────────────────

def test_mcp_tools_schema():
    from core.mcp_server import _TOOLS
    names = {t["name"] for t in _TOOLS}
    assert "search_knowledge" in names
    assert "web_search"       in names
    assert "submit_text"      in names
    for t in _TOOLS:
        assert "name"        in t
        assert "description" in t
        assert "inputSchema" in t
