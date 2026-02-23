"""
brainiac/servers/research-server/src/research_server/server.py

FastMCP research server exposing:
  - search_web          : DuckDuckGo text search
  - store_memory        : persist text chunks to the default ChromaDB collection
  - query_memory        : semantic recall from the default ChromaDB collection
  - get_library_card    : return all Master Registry entries (collection catalogue)
  - query_master_registry: semantic search against the Master Registry index
  - create_collection   : create a new leaf VDB collection and register it
  - update_collection   : upsert documents into an existing collection and refresh its registry entry
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from research_server.librarian import (
    PARTITION_THRESHOLD,
    SEMANTIC_GAP_THRESHOLD,
    UPDATE_THRESHOLD,
    _decide_lifecycle,
    _flatten_chroma_results,
)

import chromadb
from chromadb.config import Settings as ChromaSettings
from duckduckgo_search import DDGS
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("research-server")

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "brainiac_memory")
USE_REMOTE_CHROMA: bool = (
    os.getenv("USE_REMOTE_CHROMA", "false").lower() == "true"
)
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "8"))

# ---------------------------------------------------------------------------
# ChromaDB client factory
# ---------------------------------------------------------------------------


def _build_chroma_client() -> chromadb.ClientAPI:
    """Return a ChromaDB client (remote HTTP or local ephemeral)."""
    if USE_REMOTE_CHROMA:
        logger.info(
            "Connecting to remote ChromaDB at %s:%s", CHROMA_HOST, CHROMA_PORT
        )
        return chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    logger.info("Using local ephemeral ChromaDB (no persistence)")
    return chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )


_chroma: chromadb.ClientAPI = _build_chroma_client()

# Default leaf collection (legacy / general purpose).
_collection: chromadb.Collection = _chroma.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

# Master Registry — index-of-indexes.  One document per leaf collection:
# document text = topic summary, metadata = {collection_name, last_updated, doc_count}.
MASTER_REGISTRY_NAME: str = "master_registry"
_registry: chromadb.Collection = _chroma.get_or_create_collection(
    name=MASTER_REGISTRY_NAME,
    metadata={"hnsw:space": "cosine"},
)
logger.info("Master Registry initialised (%s)", MASTER_REGISTRY_NAME)

# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------
mcp: FastMCP = FastMCP(
    name="brainiac-research-server",
    instructions=(
        "Provides dynamic web search via DuckDuckGo and "
        "persistent vector memory via ChromaDB."
    ),
)

# ---------------------------------------------------------------------------
# Input/output models
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class MemoryEntry(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search_web(
    query: str = Field(..., description="The search query string."),
    max_results: int = Field(
        MAX_SEARCH_RESULTS,
        ge=1,
        le=20,
        description="Maximum number of results to return.",
    ),
) -> list[SearchResult]:
    """Search the web using DuckDuckGo and return structured results.

    Args:
        query: Natural-language search query.
        max_results: Cap on returned results (default 8, max 20).

    Returns:
        A list of SearchResult objects with title, url, and snippet.
    """
    logger.info("[search_web] query=%r max_results=%d", query, max_results)
    results: list[SearchResult] = []
    try:
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=max_results):
                results.append(
                    SearchResult(
                        title=hit.get("title", ""),
                        url=hit.get("href", ""),
                        snippet=hit.get("body", ""),
                    )
                )
    except Exception as exc:
        logger.error("[search_web] DuckDuckGo error: %s", exc, exc_info=True)
        raise RuntimeError(f"Web search failed: {exc}") from exc

    logger.info("[search_web] returned %d results", len(results))
    return results


@mcp.tool()
def get_current_time() -> str:
    """Return the current date and time in ISO 8601 format with timezone.

    Use this tool when the user asks about "today", "now", "current time",
    or any query that requires knowing what day/time it is (e.g., weather,
    news, events, schedules).

    Returns:
        ISO 8601 timestamp string (e.g., "2026-02-22T15:30:00+00:00").
    """
    now_iso: str = datetime.now(timezone.utc).isoformat()
    logger.info("[get_current_time] returning %s", now_iso)
    return now_iso


@mcp.tool()
def store_memory(
    text: str = Field(..., description="Text to embed and store."),
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata attached to this entry.",
    ),
    doc_id: str | None = Field(
        None,
        description="Optional stable ID; auto-generated if omitted.",
    ),
) -> MemoryEntry:
    """Embed and persist a text chunk into ChromaDB.

    Args:
        text: The text content to store.
        metadata: Optional metadata dict (e.g. source URL, timestamp).
        doc_id: Optional stable document ID; a UUID is generated if absent.

    Returns:
        A MemoryEntry with the assigned id, text, and metadata.
    """
    entry_id = doc_id or str(uuid.uuid4())
    logger.info("[store_memory] id=%s chars=%d", entry_id, len(text))
    try:
        _collection.upsert(
            ids=[entry_id],
            documents=[text],
            metadatas=[metadata],
        )
    except Exception as exc:
        logger.error("[store_memory] ChromaDB error: %s", exc, exc_info=True)
        raise RuntimeError(f"Memory storage failed: {exc}") from exc

    logger.info("[store_memory] stored id=%s", entry_id)
    return MemoryEntry(id=entry_id, text=text, metadata=metadata)


@mcp.tool()
def query_memory(
    query: str = Field(..., description="Semantic search query."),
    top_k: int = Field(5, ge=1, le=50, description="Number of results."),
) -> list[MemoryEntry]:
    """Retrieve semantically similar memory entries from ChromaDB.

    Args:
        query: The text to search against stored embeddings.
        top_k: Number of top matches to return.

    Returns:
        A list of MemoryEntry objects ordered by relevance.
    """
    logger.info("[query_memory] query=%r top_k=%d", query, top_k)
    try:
        results = _collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        logger.error("[query_memory] ChromaDB error: %s", exc, exc_info=True)
        raise RuntimeError(f"Memory query failed: {exc}") from exc

    entries: list[MemoryEntry] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    for doc_id, doc, meta in zip(ids, docs, metas):
        entries.append(MemoryEntry(id=doc_id, text=doc, metadata=meta or {}))

    logger.info("[query_memory] returned %d entries", len(entries))
    return entries


# ---------------------------------------------------------------------------
# Librarian tools — Master Registry management
# ---------------------------------------------------------------------------


@mcp.tool()
def get_library_card() -> str:
    """Return the full Master Registry catalogue as a JSON list.

    Each entry describes one leaf VDB collection currently held in memory.
    The system-server injects this list into the router's system prompt so
    the routing model can reference existing collections by name.

    Returns:
        JSON-encoded list of ``{collection_name, summary, last_updated,
        doc_count}`` dicts, or an error dict on failure.
    """
    import json as _json

    logger.info("[get_library_card] fetching all master_registry entries")
    try:
        result = _registry.get(include=["documents", "metadatas"])
        entries: list[dict[str, Any]] = []
        docs: list[str] = result.get("documents") or []
        metas: list[dict[str, Any]] = result.get("metadatas") or []
        for doc, meta in zip(docs, metas):
            entries.append(
                {
                    "collection_name": (meta or {}).get("collection_name", ""),
                    "summary": doc,
                    "last_updated": (meta or {}).get("last_updated", ""),
                    "doc_count": (meta or {}).get("doc_count", 0),
                }
            )
        logger.info("[get_library_card] returned %d entries", len(entries))
        return _json.dumps(entries, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("[get_library_card] error: %s", exc, exc_info=True)
        return _json.dumps({"error": str(exc)})


@mcp.tool()
def query_master_registry(
    query: str = Field(..., description="Semantic search query against the Master Registry."),
    top_k: int = Field(3, ge=1, le=20, description="Maximum leaf collections to return."),
) -> str:
    """Semantic search against the Master Registry to find relevant leaf collections.

    Embeds ``query`` and finds the closest collection summaries stored in the
    Master Registry.  Returns collection metadata and the maximum cosine
    similarity score so the caller can detect Semantic Gaps.

    Args:
        query: Natural-language description of the information needed.
        top_k: Number of most-similar collections to return (default 3).

    Returns:
        JSON-encoded ``{max_similarity, has_gap, collections: [{collection_name,
        summary, similarity, last_updated, doc_count}]}``.
        ``has_gap`` is ``true`` when ``max_similarity < SEMANTIC_GAP_THRESHOLD``.
        Returns an error dict on failure.
    """
    import json as _json

    logger.info("[query_master_registry] query=%r top_k=%d", query, top_k)
    try:
        count: int = _registry.count()
        if count == 0:
            logger.info("[query_master_registry] registry is empty — gap detected")
            return _json.dumps(
                {
                    "max_similarity": 0.0,
                    "has_gap": True,
                    "collections": [],
                }
            )

        actual_k = min(top_k, count)
        raw = _registry.query(
            query_texts=[query],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )
        flat = _flatten_chroma_results(raw)

        collections: list[dict[str, Any]] = []
        similarities: list[float] = []
        for item in flat:
            sim: float = item["similarity"] if item["similarity"] is not None else 0.0
            similarities.append(sim)
            meta: dict[str, Any] = item["metadata"]
            collections.append(
                {
                    "collection_name": meta.get("collection_name", ""),
                    "summary": item["document"],
                    "similarity": round(sim, 4),
                    "last_updated": meta.get("last_updated", ""),
                    "doc_count": meta.get("doc_count", 0),
                }
            )

        max_similarity: float = max(similarities) if similarities else 0.0
        has_gap: bool = max_similarity < SEMANTIC_GAP_THRESHOLD
        logger.info(
            "[query_master_registry] max_similarity=%.4f has_gap=%s",
            max_similarity,
            has_gap,
        )
        return _json.dumps(
            {
                "max_similarity": round(max_similarity, 4),
                "has_gap": has_gap,
                "semantic_gap_threshold": SEMANTIC_GAP_THRESHOLD,
                "collections": collections,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        logger.error("[query_master_registry] error: %s", exc, exc_info=True)
        return _json.dumps({"error": str(exc)})


@mcp.tool()
def create_collection(
    summary: str = Field(
        ...,
        description="One-sentence topic summary stored in the Master Registry.",
    ),
    name: str = Field(
        "",
        description=(
            "Desired collection name slug (e.g. 'python_basics'). "
            "A short UUID suffix is appended automatically. "
            "Leave blank to auto-generate from the summary."
        ),
    ),
    documents: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Documents to upsert.  Each dict must have a ``'text'`` key and "
            "an optional ``'metadata'`` dict and optional ``'id'`` string."
        ),
    ),
) -> str:
    """Create a new leaf VDB collection and register it in the Master Registry.

    Called by the system-server LibrarianAgent when a Semantic Gap is detected
    and the incoming document batch centroid is below PARTITION_THRESHOLD
    (i.e. no existing collection is a close-enough match for an upsert).

    Args:
        name: Short slug for the collection name; UUID suffix appended.
        summary: Human-readable topic description written to the Master Registry.
        documents: List of ``{text, metadata?, id?}`` dicts to store.

    Returns:
        JSON-encoded ``{collection_name, action, doc_count}`` on success,
        or an error dict on failure.
    """
    import json as _json
    import re as _re

    logger.info("[create_collection] name_hint=%r docs=%d", name, len(documents))
    try:
        # Build a deterministic, filesystem-safe collection name.
        short_id: str = str(uuid.uuid4())[:8]
        slug = _re.sub(r"[^a-z0-9_]", "_", name.lower().strip()) if name else "topic"
        col_name: str = f"{slug}_{short_id}"

        leaf: chromadb.Collection = _chroma.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Upsert provided documents into the new leaf collection.
        now_iso: str = datetime.now(timezone.utc).isoformat()
        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict[str, Any]] = []
        for doc in documents:
            doc_id: str = str(doc.get("id") or uuid.uuid4())
            doc_meta: dict[str, Any] = dict(doc.get("metadata") or {})
            doc_meta.setdefault("collection_name", col_name)
            doc_meta.setdefault("ingested_at", now_iso)
            doc_meta.setdefault("related_collections", [])
            ids.append(doc_id)
            texts.append(str(doc.get("text", "")))
            metas.append(doc_meta)

        if ids:
            leaf.upsert(ids=ids, documents=texts, metadatas=metas)

        # Register the new collection in the Master Registry.
        _registry.upsert(
            ids=[col_name],
            documents=[summary],
            metadatas=[
                {
                    "collection_name": col_name,
                    "last_updated": now_iso,
                    "doc_count": len(ids),
                }
            ],
        )

        logger.info(
            "[create_collection] created %r with %d docs", col_name, len(ids)
        )
        return _json.dumps(
            {"collection_name": col_name, "action": "create", "doc_count": len(ids)},
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.error("[create_collection] error: %s", exc, exc_info=True)
        return _json.dumps({"error": str(exc)})


@mcp.tool()
def update_collection(
    name: str = Field(..., description="Exact name of the existing leaf collection."),
    documents: list[dict[str, Any]] = Field(
        ...,
        description=(
            "New documents to upsert.  Each dict must have a ``'text'`` key and "
            "an optional ``'metadata'`` dict and optional ``'id'`` string."
        ),
    ),
    summary: str = Field(
        "",
        description=(
            "Updated topic summary.  If non-empty, overwrites the existing "
            "Master Registry entry summary."
        ),
    ),
) -> str:
    """Upsert new documents into an existing leaf collection and refresh its Master Registry entry.

    Called by the system-server LibrarianAgent when a Semantic Gap is detected
    but the incoming document batch scores above UPDATE_THRESHOLD against an
    existing collection (i.e. the topic already has a home).

    Args:
        name: Exact collection name as registered in the Master Registry.
        documents: List of ``{text, metadata?, id?}`` dicts to upsert.
        summary: Optional refreshed topic summary for the Master Registry.

    Returns:
        JSON-encoded ``{collection_name, action, new_docs, total_docs}`` on
        success, or an error dict on failure.
    """
    import json as _json

    logger.info("[update_collection] name=%r new_docs=%d", name, len(documents))
    try:
        leaf: chromadb.Collection = _chroma.get_collection(name=name)

        now_iso: str = datetime.now(timezone.utc).isoformat()
        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict[str, Any]] = []
        for doc in documents:
            doc_id: str = str(doc.get("id") or uuid.uuid4())
            doc_meta: dict[str, Any] = dict(doc.get("metadata") or {})
            doc_meta.setdefault("collection_name", name)
            doc_meta.setdefault("ingested_at", now_iso)
            doc_meta.setdefault("related_collections", [])
            ids.append(doc_id)
            texts.append(str(doc.get("text", "")))
            metas.append(doc_meta)

        if ids:
            leaf.upsert(ids=ids, documents=texts, metadatas=metas)

        total: int = leaf.count()

        # Refresh the Master Registry entry.
        existing_meta_result = _registry.get(ids=[name], include=["documents", "metadatas"])
        existing_docs: list[str] = existing_meta_result.get("documents") or []
        existing_summary: str = summary or (existing_docs[0] if existing_docs else name)

        _registry.upsert(
            ids=[name],
            documents=[existing_summary],
            metadatas=[
                {
                    "collection_name": name,
                    "last_updated": now_iso,
                    "doc_count": total,
                }
            ],
        )

        logger.info(
            "[update_collection] updated %r: +%d docs (total %d)",
            name, len(ids), total,
        )
        return _json.dumps(
            {
                "collection_name": name,
                "action": "update",
                "new_docs": len(ids),
                "total_docs": total,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.error("[update_collection] error: %s", exc, exc_info=True)
        return _json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Launch the research server (called by the Poetry script entry point)."""
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8100"))
    logger.info("Starting brainiac-research-server on %s:%d", host, port)
    mcp.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    run()
