"""
MYCONEX Vector Store
Qdrant-based vector storage for agent memory, embeddings, and semantic search.
Handles conversation context, knowledge retrieval, and persistent agent state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    SearchRequest,
    VectorParams,
)

logger = logging.getLogger(__name__)

# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single memory entry with vector embedding."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    agent_name: str = ""
    conversation_id: str = ""
    memory_type: str = "general"  # general, context, knowledge, episodic

    def to_point(self) -> PointStruct:
        """Convert to Qdrant point structure."""
        return PointStruct(
            id=self.id,
            vector=self.embedding,
            payload={
                "content": self.content,
                "metadata": self.metadata,
                "timestamp": self.timestamp,
                "agent_name": self.agent_name,
                "conversation_id": self.conversation_id,
                "memory_type": self.memory_type,
            }
        )

    @classmethod
    def from_payload(cls, point_id: str, payload: Dict[str, Any], vector: List[float]) -> "MemoryEntry":
        """Create from Qdrant payload."""
        return cls(
            id=str(point_id),
            content=payload.get("content", ""),
            embedding=vector,
            metadata=payload.get("metadata", {}),
            timestamp=payload.get("timestamp", time.time()),
            agent_name=payload.get("agent_name", ""),
            conversation_id=payload.get("conversation_id", ""),
            memory_type=payload.get("memory_type", "general"),
        )


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    entry: MemoryEntry
    score: float
    distance: float

    @property
    def relevance(self) -> float:
        """Normalized relevance score (0-1, higher is better)."""
        # Convert distance to similarity (assuming cosine distance)
        return 1.0 - min(self.distance, 1.0)


# ─── Vector Store ────────────────────────────────────────────────────────────

class VectorStore:
    """
    Qdrant-based vector store for agent memory and semantic search.

    Features:
    - Embedding storage and retrieval
    - Conversation context persistence
    - Semantic search with filtering
    - Memory consolidation and cleanup
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "myconex-memory",
        vector_size: int = 768,  # Default for nomic-embed-text
        distance: Distance = Distance.COSINE,
    ):
        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        self.client: Optional[QdrantClient] = None
        self._initialized = False

    async def connect(self) -> None:
        """Connect to Qdrant and ensure collection exists."""
        try:
            self.client = QdrantClient(url=self.url)

            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"[vector_store] creating collection '{self.collection_name}'")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance,
                    ),
                )
            else:
                logger.info(f"[vector_store] using existing collection '{self.collection_name}'")

            self._initialized = True
            logger.info(f"[vector_store] connected to Qdrant at {self.url}")

        except Exception as e:
            logger.error(f"[vector_store] connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Close connection to Qdrant."""
        if self.client:
            # Qdrant client doesn't have explicit disconnect, just cleanup
            self.client = None
            self._initialized = False

    def _ensure_connected(self) -> None:
        """Ensure we're connected to Qdrant."""
        if not self._initialized or not self.client:
            raise RuntimeError("VectorStore not connected. Call connect() first.")

    # ─── Memory Operations ───────────────────────────────────────────────────

    async def store_memory(self, entry: MemoryEntry) -> str:
        """
        Store a memory entry in the vector database.

        Args:
            entry: MemoryEntry to store

        Returns:
            The ID of the stored entry
        """
        self._ensure_connected()

        try:
            point = entry.to_point()
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            logger.debug(f"[vector_store] stored memory: {entry.id}")
            return entry.id

        except Exception as e:
            logger.error(f"[vector_store] failed to store memory: {e}")
            raise

    async def store_conversation_turn(
        self,
        agent_name: str,
        conversation_id: str,
        role: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a conversation turn with embedding."""
        entry = MemoryEntry(
            id=f"conv_{conversation_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            agent_name=agent_name,
            conversation_id=conversation_id,
            memory_type="context",
        )
        return await self.store_memory(entry)

    async def store_knowledge(
        self,
        agent_name: str,
        content: str,
        embedding: List[float],
        source: str = "",
        tags: Optional[List[str]] = None,
    ) -> str:
        """Store knowledge/fact with embedding."""
        entry = MemoryEntry(
            id=f"knowledge_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            content=content,
            embedding=embedding,
            metadata={
                "source": source,
                "tags": tags or [],
            },
            agent_name=agent_name,
            memory_type="knowledge",
        )
        return await self.store_memory(entry)

    # ─── Retrieval Operations ────────────────────────────────────────────────

    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors using embedding.

        Args:
            query_embedding: Query vector
            limit: Maximum results to return
            score_threshold: Minimum similarity score (0-1)
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        self._ensure_connected()

        try:
            # Build search filters
            search_filters = None
            if filters:
                from qdrant_client.http.models import Filter, FieldCondition, MatchValue
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Multiple values - use MatchAny
                        conditions.append(
                            FieldCondition(key=f"metadata.{key}", match={"any": value})
                        )
                    else:
                        # Single value
                        conditions.append(
                            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                        )
                if conditions:
                    search_filters = Filter(must=conditions)

            # Perform search
            # query_points is the current API (client.search removed in qdrant-client 1.7+)
            qr = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
                query_filter=search_filters,
                score_threshold=score_threshold,
            )
            results = qr.points

            # Convert to SearchResult objects
            search_results = []
            for hit in results:
                payload = hit.payload
                entry = MemoryEntry.from_payload(hit.id, payload, hit.vector)
                search_results.append(SearchResult(
                    entry=entry,
                    score=hit.score,
                    distance=1.0 - hit.score if self.distance == Distance.COSINE else hit.score,
                ))

            logger.debug(f"[vector_store] search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"[vector_store] search failed: {e}")
            raise

    async def get_conversation_context(
        self,
        conversation_id: str,
        limit: int = 20,
    ) -> List[MemoryEntry]:
        """Retrieve recent conversation context."""
        self._ensure_connected()

        try:
            # Search with conversation filter
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {"key": "conversation_id", "match": {"value": conversation_id}},
                        {"key": "memory_type", "match": {"value": "context"}},
                    ]
                },
                limit=limit,
                order_by={"key": "timestamp", "direction": "desc"},
            )

            entries = []
            for point in results[0]:  # results[0] contains the points
                payload = point.payload
                entry = MemoryEntry.from_payload(point.id, payload, point.vector)
                entries.append(entry)

            # Sort by timestamp (oldest first for conversation flow)
            entries.sort(key=lambda x: x.timestamp)
            return entries

        except Exception as e:
            logger.error(f"[vector_store] failed to get conversation context: {e}")
            raise

    async def get_agent_memories(
        self,
        agent_name: str,
        memory_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[MemoryEntry]:
        """Get memories for a specific agent."""
        self._ensure_connected()

        try:
            # Build filter
            filter_conditions = [{"key": "agent_name", "match": {"value": agent_name}}]
            if memory_type:
                filter_conditions.append({"key": "memory_type", "match": {"value": memory_type}})

            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={"must": filter_conditions},
                limit=limit,
                order_by={"key": "timestamp", "direction": "desc"},
            )

            entries = []
            for point in results[0]:
                payload = point.payload
                entry = MemoryEntry.from_payload(point.id, payload, point.vector)
                entries.append(entry)

            return entries

        except Exception as e:
            logger.error(f"[vector_store] failed to get agent memories: {e}")
            raise

    # ─── Maintenance Operations ──────────────────────────────────────────────

    async def cleanup_old_memories(
        self,
        max_age_days: int = 30,
        memory_types: Optional[List[str]] = None,
    ) -> int:
        """
        Remove old memories to prevent database bloat.

        Args:
            max_age_days: Remove memories older than this
            memory_types: Types of memories to clean up (default: context only)

        Returns:
            Number of memories deleted
        """
        self._ensure_connected()

        try:
            from qdrant_client.http.models import Filter, FieldCondition, MatchAny, Range

            cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
            types_to_clean = memory_types or ["context"]

            delete_filter = Filter(
                must=[
                    FieldCondition(key="timestamp", range=Range(lt=cutoff_time)),
                    FieldCondition(key="memory_type", match=MatchAny(any=types_to_clean)),
                ]
            )
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=delete_filter,
            )
            # qdrant delete returns an UpdateResult; deleted count is not always exposed
            deleted = getattr(result, "result", None)
            count = deleted if isinstance(deleted, int) else 0
            logger.info("[vector_store] TTL cleanup: removed ~%d entries older than %dd", count, max_age_days)
            return count

        except Exception as e:
            logger.error(f"[vector_store] cleanup failed: {e}")
            raise

    async def consolidate_memories(
        self,
        agent_name: str = "buzlock",
        similarity_threshold: float = 0.92,
        page_size: int = 200,
    ) -> int:
        """
        Consolidate near-duplicate memories using vector similarity.

        Strategy:
          1. Page through entries sorted oldest-first.
          2. For each entry, query for the nearest neighbour above
             similarity_threshold in the same memory_type.
          3. If a newer duplicate exists, mark the older one for deletion.
          4. Delete all marked entries in one batch.

        Args:
            agent_name:           Agent namespace to consolidate.
            similarity_threshold: Cosine similarity above which entries are
                                  considered duplicates (default 0.92).
            page_size:            Entries fetched per scroll page.

        Returns:
            Number of entries removed.
        """
        self._ensure_connected()

        try:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue

            agent_filter = Filter(
                must=[FieldCondition(key="agent_name", match=MatchValue(value=agent_name))]
            )

            # Page through all entries with vectors
            to_delete: List[str] = []
            seen_ids: set = set()
            offset = None

            while True:
                scroll_kwargs: Dict[str, Any] = dict(
                    collection_name=self.collection_name,
                    scroll_filter=agent_filter,
                    limit=page_size,
                    with_vectors=True,
                    with_payload=True,
                )
                if offset is not None:
                    scroll_kwargs["offset"] = offset

                points, next_offset = self.client.scroll(**scroll_kwargs)
                if not points:
                    break

                for point in points:
                    pid = str(point.id)
                    if pid in seen_ids or pid in to_delete:
                        continue
                    if point.vector is None:
                        continue

                    # Find the nearest neighbour of this entry
                    neighbours = self.client.query_points(
                        collection_name=self.collection_name,
                        query=point.vector,
                        query_filter=agent_filter,
                        limit=2,          # top-2: first is self, second is nearest other
                        score_threshold=similarity_threshold,
                        with_payload=True,
                    ).points

                    for nb in neighbours:
                        nb_id = str(nb.id)
                        if nb_id == pid or nb_id in to_delete:
                            continue
                        # Keep the newer entry; delete the older one
                        nb_ts = nb.payload.get("timestamp", 0) if nb.payload else 0
                        pt_ts = point.payload.get("timestamp", 0) if point.payload else 0
                        older_id = pid if pt_ts <= nb_ts else nb_id
                        to_delete.append(older_id)
                        seen_ids.add(older_id)

                    seen_ids.add(pid)

                if next_offset is None:
                    break
                offset = next_offset

            if to_delete:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=to_delete,
                )

            logger.info(
                "[vector_store] consolidation: removed %d near-duplicate entries (agent=%s threshold=%.2f)",
                len(to_delete), agent_name, similarity_threshold,
            )
            return len(to_delete)

        except Exception as e:
            logger.error(f"[vector_store] consolidation failed: {e}")
            raise

    # ─── Statistics ──────────────────────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        self._ensure_connected()

        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection": self.collection_name,
                "vector_count": info.points_count,
                "vector_size": self.vector_size,
                "distance_metric": self.distance.value,
            }
        except Exception as e:
            logger.error(f"[vector_store] failed to get stats: {e}")
            return {}


# ─── Embedding Client ────────────────────────────────────────────────────────

class EmbeddingClient:
    """
    Client for generating embeddings using Ollama or external services.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=30.0)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Ollama."""
        try:
            response = await self.client.post(
                f"{self.ollama_url}/api/embed",
                json={
                    "model": self.model,
                    "input": text,
                }
            )
            response.raise_for_status()
            data = response.json()
            # /api/embed returns {"embeddings": [[...]]} (list of lists)
            embs = data.get("embeddings") or data.get("embedding")
            return embs[0] if isinstance(embs[0], list) else embs

        except Exception as e:
            logger.error(f"[embedding] failed to generate embedding: {e}")
            raise

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in a single batch HTTP call."""
        if not texts:
            return []
        try:
            response = await self.client.post(
                f"{self.ollama_url}/api/embed",
                json={
                    "model": self.model,
                    "input": texts,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            embs = data.get("embeddings") or data.get("embedding")
            # Normalise: each element must be a flat list of floats
            return [e if isinstance(e, list) else e for e in embs]
        except Exception as e:
            logger.error("[embedding] batch generate_embeddings failed: %s", e)
            raise

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# ─── Convenience Functions ───────────────────────────────────────────────────

async def create_memory_store(
    qdrant_url: str = "http://localhost:6333",
    ollama_url: str = "http://localhost:11434",
) -> Tuple[VectorStore, EmbeddingClient]:
    """
    Create and connect both vector store and embedding client.

    Returns:
        Tuple of (VectorStore, EmbeddingClient)
    """
    store = VectorStore(url=qdrant_url)
    embedder = EmbeddingClient(ollama_url=ollama_url)

    await store.connect()

    return store, embedder