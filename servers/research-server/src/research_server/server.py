"""
brainiac/servers/research-server/src/research_server/server.py

FastMCP research server exposing:
  - search_web  : DuckDuckGo text search
  - store_memory: persist text chunks to ChromaDB
  - query_memory: semantic recall from ChromaDB
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

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
_collection: chromadb.Collection = _chroma.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------
mcp: FastMCP = FastMCP(
    name="brainiac-research-server",
    version="0.1.0",
    description=(
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
