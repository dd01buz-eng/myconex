"""
MYCONEX Knowledge Graph
------------------------
Lightweight entity-relationship graph stored as JSON on disk.
No external database required — uses NetworkX for in-memory graph operations
persisted to ~/.myconex/knowledge_graph.json.

Tracks:
  - People (names, email addresses) → mentioned in which sources
  - Projects / topics → co-occurring concepts
  - Sources → which entities they discuss

Nodes have: type, name, count (mentions), last_seen
Edges have: relation, weight (co-occurrence count), last_seen

Populated by the ingestion pipeline (email, rss, youtube).
Queried by the Discord agent for context ("what do my sources say about Person X?")

Env vars:
  KG_ENABLED         — "false" to disable (default: true)
  KG_MAX_NODES       — max nodes before oldest are pruned (default: 2000)
  KG_EXTRACT_MODEL   — Ollama model for NER extraction (default: llama3)
  KG_NER_ENABLED     — "false" to disable LLM-based NER, use regex only (default: true)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASE       = Path.home() / ".myconex"
_GRAPH_FILE = _BASE / "knowledge_graph.json"

KG_ENABLED       = os.getenv("KG_ENABLED",       "true").lower() != "false"
KG_MAX_NODES     = int(os.getenv("KG_MAX_NODES",       "2000"))
KG_MODEL         = os.getenv("KG_EXTRACT_MODEL",  "llama3")
KG_NER_ENABLED   = os.getenv("KG_NER_ENABLED",    "true").lower() != "false"
OLLAMA_URL       = os.getenv("OLLAMA_URL", "http://localhost:11434")

_NER_PROMPT = """Extract named entities from the following text.
Return ONLY a JSON object with these keys:
- "people": list of full names mentioned
- "projects": list of project names, product names, or initiative names
- "topics": list of key technical or domain topics (max 5)
- "organizations": list of company/org names

Text: {text}

Respond with ONLY valid JSON."""


# ── Graph data structure (pure dict, no NetworkX dep required) ────────────────

class KnowledgeGraph:
    """
    In-memory knowledge graph with JSON persistence.

    nodes: {node_id: {type, name, count, last_seen, sources: set}}
    edges: {(src_id, dst_id): {relation, weight, last_seen}}
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}  # key = "src_id||dst_id"
        self._dirty = False

    # ── Node operations ───────────────────────────────────────────────────────

    def _node_id(self, entity_type: str, name: str) -> str:
        return f"{entity_type}:{name.lower().strip()}"

    def add_entity(self, entity_type: str, name: str, source: str) -> str:
        nid = self._node_id(entity_type, name)
        if nid in self.nodes:
            self.nodes[nid]["count"] += 1
            self.nodes[nid]["last_seen"] = time.time()
            self.nodes[nid].setdefault("sources", [])
            if source and source not in self.nodes[nid]["sources"]:
                self.nodes[nid]["sources"].append(source)
        else:
            self.nodes[nid] = {
                "id":        nid,
                "type":      entity_type,
                "name":      name,
                "count":     1,
                "last_seen": time.time(),
                "sources":   [source] if source else [],
            }
        self._dirty = True
        return nid

    def add_relation(self, nid_a: str, nid_b: str, relation: str = "co-occurs") -> None:
        if nid_a == nid_b:
            return
        key = f"{min(nid_a, nid_b)}||{max(nid_a, nid_b)}"
        if key in self.edges:
            self.edges[key]["weight"] += 1
            self.edges[key]["last_seen"] = time.time()
        else:
            self.edges[key] = {
                "src":       nid_a,
                "dst":       nid_b,
                "relation":  relation,
                "weight":    1,
                "last_seen": time.time(),
            }
        self._dirty = True

    # ── Query ─────────────────────────────────────────────────────────────────

    def find(self, query: str, entity_type: str | None = None, top_k: int = 10) -> list[dict]:
        """Find nodes matching a name query (case-insensitive substring)."""
        q = query.lower()
        results = []
        for node in self.nodes.values():
            if entity_type and node["type"] != entity_type:
                continue
            if q in node["name"].lower():
                results.append(node)
        return sorted(results, key=lambda n: -n["count"])[:top_k]

    def neighbours(self, nid: str, top_k: int = 8) -> list[dict]:
        """Return most co-occurring entities for a node."""
        related = []
        for key, edge in self.edges.items():
            if edge["src"] == nid or edge["dst"] == nid:
                other_id = edge["dst"] if edge["src"] == nid else edge["src"]
                other = self.nodes.get(other_id)
                if other:
                    related.append({**other, "edge_weight": edge["weight"]})
        return sorted(related, key=lambda n: -n["edge_weight"])[:top_k]

    def context_for(self, query: str) -> str:
        """
        Return a formatted graph context block for the given query.
        Used for injection into the system prompt.
        """
        matches = self.find(query, top_k=3)
        if not matches:
            return ""
        parts = ["[Knowledge graph context:]"]
        for node in matches:
            sources = ", ".join(node.get("sources", [])[:3])
            parts.append(f"• **{node['name']}** ({node['type']}, {node['count']}× mentions) — {sources}")
            nbrs = self.neighbours(node["id"], top_k=4)
            if nbrs:
                nbr_str = ", ".join(n["name"] for n in nbrs)
                parts.append(f"  Related: {nbr_str}")
        return "\n".join(parts)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        if not self._dirty:
            return
        _BASE.mkdir(parents=True, exist_ok=True)
        data = {"nodes": self.nodes, "edges": self.edges}
        _GRAPH_FILE.write_text(json.dumps(data, ensure_ascii=False))
        self._dirty = False

    def load(self) -> None:
        try:
            if _GRAPH_FILE.exists():
                data = json.loads(_GRAPH_FILE.read_text())
                self.nodes = data.get("nodes", {})
                self.edges = data.get("edges", {})
                logger.info("[kg] loaded %d nodes, %d edges", len(self.nodes), len(self.edges))
        except Exception as exc:
            logger.warning("[kg] load failed: %s", exc)

    def prune(self) -> None:
        """Remove oldest/least-mentioned nodes when over capacity."""
        if len(self.nodes) <= KG_MAX_NODES:
            return
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: (n["count"], n["last_seen"]))
        to_remove = {n["id"] for n in sorted_nodes[:len(self.nodes) - KG_MAX_NODES]}
        for nid in to_remove:
            del self.nodes[nid]
        # Remove edges referencing deleted nodes
        self.edges = {k: v for k, v in self.edges.items()
                      if v["src"] not in to_remove and v["dst"] not in to_remove}
        self._dirty = True
        logger.info("[kg] pruned to %d nodes", len(self.nodes))


# ── Module singleton ──────────────────────────────────────────────────────────

_graph: KnowledgeGraph | None = None
_loaded = False


def get_graph() -> KnowledgeGraph:
    global _graph, _loaded
    if _graph is None:
        _graph = KnowledgeGraph()
    if not _loaded:
        _graph.load()
        _loaded = True
    return _graph


# ── Extraction helpers ────────────────────────────────────────────────────────

_EMAIL_RE    = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.I)
_CAPS_NAME   = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\b")


def _extract_regex(text: str) -> dict[str, list[str]]:
    """Fast regex-based entity extraction (no LLM)."""
    emails = _EMAIL_RE.findall(text)
    names  = _CAPS_NAME.findall(text)
    # Filter common false positives
    _stop = {"The", "This", "That", "Here", "There", "When", "Where", "Which",
              "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
              "January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"}
    names = [n for n in names if n not in _stop]
    return {"people": list(set(emails + names)), "projects": [], "topics": [], "organizations": []}


async def _extract_ner_llm(text: str) -> dict[str, list[str]]:
    """LLM-based NER via Ollama."""
    import httpx as _httpx
    prompt = _NER_PROMPT.format(text=text[:1500])
    try:
        async with _httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": KG_MODEL, "prompt": prompt, "stream": False,
                      "options": {"num_predict": 200, "temperature": 0.0}},
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
    except Exception as exc:
        logger.debug("[kg] NER LLM failed: %s", exc)
        return _extract_regex(text)


async def ingest_text(text: str, source: str, title: str = "") -> None:
    """
    Extract entities from text and add them to the knowledge graph.
    Call from ingesters after processing each item.
    """
    if not KG_ENABLED or not text:
        return
    g = get_graph()

    try:
        if KG_NER_ENABLED:
            entities = await _extract_ner_llm(text)
        else:
            entities = _extract_regex(text)
    except Exception as exc:
        logger.debug("[kg] extraction failed: %s", exc)
        return

    node_ids: list[str] = []

    # Add source node
    src_nid = g.add_entity("source", source, source)

    for entity_type, names in [
        ("person",       entities.get("people",        [])),
        ("project",      entities.get("projects",      [])),
        ("topic",        entities.get("topics",        [])),
        ("organization", entities.get("organizations", [])),
    ]:
        for name in names:
            if not name or len(name) < 2:
                continue
            nid = g.add_entity(entity_type, name, source)
            node_ids.append(nid)
            g.add_relation(src_nid, nid, "mentions")

    # Add co-occurrence edges between entities in the same document
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            g.add_relation(node_ids[i], node_ids[j], "co-occurs")

    g.prune()
    await asyncio.to_thread(g.save)
