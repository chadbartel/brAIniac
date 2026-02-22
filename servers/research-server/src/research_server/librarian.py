"""
brainiac/servers/research-server/src/research_server/librarian.py

Autonomous Learning Architecture — Librarian constants and lifecycle helpers.

This module is the single source of truth for all similarity thresholds and
the collection lifecycle decision function.  It contains NO I/O; all ChromaDB
mutations and registrations happen in server.py via the MCP tools that import
from here.

Exposed interfaces:
  SEMANTIC_GAP_THRESHOLD  — minimum cosine similarity to skip live web search
  PARTITION_THRESHOLD     — lower bound of the Partition band (Create sub-coll)
  UPDATE_THRESHOLD        — lower bound of the Update band (upsert into existing)
  _decide_lifecycle()     — maps a cosine similarity score to a lifecycle action
  _flatten_chroma_results() — normalise raw ChromaDB query output to flat dicts
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("research-server.librarian")

# ---------------------------------------------------------------------------
# Similarity thresholds (§8.2 / §8.3 of copilot-instructions.md)
# All orchestrator / tool code MUST import these — never hardcode the values.
# ---------------------------------------------------------------------------

SEMANTIC_GAP_THRESHOLD: float = 0.85
"""Minimum cosine similarity to consider the VDB context sufficient.

If the best-matching leaf document scores below this against the user query,
a Semantic Gap is declared and the Research → Vectorize → Register workflow
is triggered.
"""

PARTITION_THRESHOLD: float = 0.85
"""Lower bound of the Partition band.

Incoming document batches whose centroid scores in [PARTITION_THRESHOLD,
UPDATE_THRESHOLD) against the nearest existing collection trigger the creation
of a sub-collection (``<parent>_sub_<uuid4_short>``).
"""

UPDATE_THRESHOLD: float = 0.95
"""Lower bound of the Update band.

Incoming document batches whose centroid scores ≥ UPDATE_THRESHOLD against
the nearest existing collection are upserted directly into that collection.
"""


# ---------------------------------------------------------------------------
# Lifecycle decision helper
# ---------------------------------------------------------------------------


def _decide_lifecycle(similarity: float) -> str:
    """Return the lifecycle action for a new document batch.

    Decision table (§8.3):
      ≥ UPDATE_THRESHOLD (0.95)               → ``'update'``
      ≥ PARTITION_THRESHOLD (0.85) and < 0.95 → ``'partition'``
      < PARTITION_THRESHOLD (0.85)             → ``'create'``

    Args:
        similarity: Cosine similarity between the incoming document batch
            centroid and the nearest existing leaf collection centroid.

    Returns:
        One of ``'update'``, ``'partition'``, or ``'create'``.
    """
    if similarity >= UPDATE_THRESHOLD:
        return "update"
    if similarity >= PARTITION_THRESHOLD:
        return "partition"
    return "create"


# ---------------------------------------------------------------------------
# ChromaDB result normaliser
# ---------------------------------------------------------------------------


def _flatten_chroma_results(
    results: dict[str, Any],
    *,
    source_collection: str = "",
) -> list[dict[str, Any]]:
    """Flatten a raw ChromaDB query result dict into a list of document dicts.

    ChromaDB returns lists-of-lists (one inner list per query embedding).
    This helper collapses the first result set into a flat list, converting
    cosine distances (``1 − similarity``) to similarity scores.

    Args:
        results: Raw result dict from ``Collection.query()`` with include
            ``["documents", "metadatas", "distances"]``.
        source_collection: Optional collection name to attach as ``source_collection``
            on every returned dict, aiding cross-collection provenance tracking.

    Returns:
        List of ``{id, document, metadata, distance, similarity, source_collection}``
        dicts, ordered by ChromaDB's relevance ranking.
    """
    flat: list[dict[str, Any]] = []

    ids: list[str] = results.get("ids", [[]])[0]
    docs: list[str] = results.get("documents", [[]])[0]
    metas: list[dict[str, Any]] = results.get("metadatas", [[]])[0]
    raw_distances: list[float] = results.get("distances", [[]])[0]

    # Pad distances with None if absent so zip is always safe.
    distances: list[float | None] = (
        raw_distances if raw_distances else [None] * len(ids)
    )

    for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
        # ChromaDB cosine space: distance ∈ [0, 1], similarity = 1 − distance.
        similarity: float | None = (1.0 - dist) if dist is not None else None
        flat.append(
            {
                "id": doc_id,
                "document": doc,
                "metadata": meta or {},
                "distance": dist,
                "similarity": similarity,
                "source_collection": source_collection,
            }
        )

    return flat
