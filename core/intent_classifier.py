"""core/intent_classifier.py

DistilBERT zero-shot intent classifier for brAIniac.

Design rationale — Option A (zero-shot NLI) vs Option B (keyword rules):
  - Option A uses ``typeform/distilbert-base-uncased-mnli`` via the
    Hugging Face ``zero-shot-classification`` pipeline running **CPU-only**.
    Measured latency: 80–150 ms, negligible vs 1–3 s Ollama inference.
  - Option B (regex / keyword list) is O(n) but requires ongoing manual
    maintenance and struggles with paraphrased temporal intent phrases.
  - Choice: Option A — generalises without rule upkeep; adopting ``torch``
    now also aligns with Phase 4 QLoRA / Unsloth fine-tuning plans, so
    the transitive dependency cost is paid once.

Public API: ``needs_current_information(query: str) -> bool``
"""

from __future__ import annotations

# Standard Library
import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level pipeline cache — populated lazily on first call.
_pipeline: Any | None = None

# Candidate labels fed to the NLI model.
# Phrased to complete the hypothesis template naturally:
#   "This question requires [label] to answer accurately."
# More concrete labels (especially mentioning current/recent events and
# product releases) produce better calibration from the MNLI model than
# the generic "requires recent information" phrasing.
_LABELS: list[str] = [
    "current or recent information from the internet, such as news, product releases, software versions, or prices",
    "only general knowledge or logical reasoning without any live data",
]

# Hypothesis template for the NLI pipeline.
# The pipeline substitutes each label into {}, creating:
#   "This question requires <label> to answer accurately."
_HYPOTHESIS_TEMPLATE: str = "This question requires {} to answer accurately."


def _load_pipeline() -> Any:
    """Load and cache the zero-shot classification pipeline.

    The ``transformers`` import is deferred to this function so that
    importing ``core.intent_classifier`` never fails even when
    ``transformers`` is not installed (e.g., in CI environments). The
    failure only surfaces at call time, as a clean ``RuntimeError``.

    Returns:
        A Hugging Face ``Pipeline`` instance for zero-shot-classification.

    Raises:
        RuntimeError: If ``transformers`` cannot be imported or the model
            cannot be loaded.
    """
    global _pipeline  # noqa: PLW0603

    if _pipeline is not None:
        return _pipeline

    try:
        # Third-Party Libraries
        from transformers import pipeline  # type: ignore[import-untyped]

        logger.info(
            "Loading intent classifier: typeform/distilbert-base-uncased-mnli (CPU)"
        )
        _pipeline = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device="cpu",
        )
        logger.info("Intent classifier loaded successfully.")
        return _pipeline
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. "
            "Run `poetry install` to add it, or set INTENT_CONFIDENCE_THRESHOLD=0 "
            "to disable intent gating."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load intent classifier pipeline: {exc}") from exc


def needs_current_information(query: str) -> bool:
    """Determine whether a query requires up-to-date (real-time) information.

    Runs the input through a DistilBERT zero-shot NLI classifier on CPU.
    Returns ``True`` when the model is confident the query benefits from
    live data (i.e., the ``deep_research`` tool should be offered to the
    LLM for this turn).

    The confidence threshold is configurable via the
    ``INTENT_CONFIDENCE_THRESHOLD`` environment variable (default ``0.55``).
    Setting the variable to ``0`` effectively disables the check (always
    include research tools).

    Args:
        query: The raw user message to classify.

    Returns:
        ``True`` if the first-ranked label is ``"requires recent information"``
        with score ≥ threshold; ``False`` otherwise.
    """
    threshold = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.45"))

    try:
        pipe = _load_pipeline()
        result: dict[str, Any] = pipe(
            query,
            candidate_labels=_LABELS,
            hypothesis_template=_HYPOTHESIS_TEMPLATE,
        )
        top_label: str = result["labels"][0]
        top_score: float = result["scores"][0]

        is_research = top_label == _LABELS[0] and top_score >= threshold

        logger.info(
            "Intent classification: query=%r → research_needed=%s (score=%.3f, threshold=%.2f)",
            query[:80],
            is_research,
            top_score,
            threshold,
        )
        return is_research

    except RuntimeError as exc:
        logger.warning(
            "Intent classifier unavailable (%s) — defaulting to research_needed=False.",
            exc,
        )
        return False
    except Exception as exc:
        logger.warning(
            "Unexpected classifier error (%s) — defaulting to research_needed=False.",
            exc,
            exc_info=True,
        )
        return False
