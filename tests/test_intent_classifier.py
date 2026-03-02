"""tests/test_intent_classifier.py

Unit tests for core/intent_classifier.py.

All tests mock ``transformers.pipeline`` so they run entirely offline
without downloading any models.
"""

from __future__ import annotations

# Standard Library
from unittest.mock import MagicMock, call, patch

# Third-Party Libraries
import pytest

# Local Modules
import core.intent_classifier as cls_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline_mock(label: str, score: float) -> MagicMock:
    """Return a mock pipeline callable that returns a fixed NLI result.

    Args:
        label: The top-ranked label string.
        score: Confidence score for ``label``.
    """
    pipe = MagicMock()
    pipe.return_value = {
        "sequence": "dummy",
        "labels": [label, "other label"],
        "scores": [score, 1.0 - score],
    }
    return pipe


# ---------------------------------------------------------------------------
# _load_pipeline
# ---------------------------------------------------------------------------


class TestLoadPipeline:
    """Tests for the lazy-loading _load_pipeline() function."""

    def setup_method(self) -> None:
        """Reset module-level cache before each test."""
        cls_module._pipeline = None

    def test_lazy_load_on_first_call(self) -> None:
        """Pipeline is instantiated on the first call to _load_pipeline()."""
        mock_pipe = _make_pipeline_mock(cls_module._LABELS[0], 0.9)
        with patch("transformers.pipeline", return_value=mock_pipe) as mock_factory:
            result = cls_module._load_pipeline()
        mock_factory.assert_called_once_with(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device="cpu",
        )
        assert result is mock_pipe

    def test_pipeline_cached_after_first_load(self) -> None:
        """Subsequent calls return the cached instance without re-loading."""
        mock_pipe = _make_pipeline_mock(cls_module._LABELS[0], 0.9)
        with patch("transformers.pipeline", return_value=mock_pipe) as mock_factory:
            first = cls_module._load_pipeline()
            second = cls_module._load_pipeline()
            third = cls_module._load_pipeline()

        assert first is second is third
        # transformers.pipeline should only have been called *once*
        mock_factory.assert_called_once()

    def test_import_error_raises_runtime_error(self) -> None:
        """ImportError from transformers is wrapped as RuntimeError."""
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(RuntimeError, match="transformers is not installed"):
                cls_module._load_pipeline()


# ---------------------------------------------------------------------------
# needs_current_information
# ---------------------------------------------------------------------------


class TestNeedsCurrentInformation:
    """Tests for the public needs_current_information() function."""

    def setup_method(self) -> None:
        """Reset cache before each test."""
        cls_module._pipeline = None

    def _patch_pipeline(self, label: str, score: float) -> MagicMock:
        """Inject a mock pipeline into the module cache and return the mock."""
        mock_pipe = _make_pipeline_mock(label, score)
        cls_module._pipeline = mock_pipe
        return mock_pipe

    # -- temporal queries (should return True) --------------------------------

    def test_temporal_query_above_threshold_returns_true(self) -> None:
        """A temporal query with score above threshold returns True."""
        self._patch_pipeline(cls_module._LABELS[0], 0.90)
        assert (
            cls_module.needs_current_information("What is the latest iPhone model?")
            is True
        )

    def test_temporal_query_exactly_at_threshold_returns_true(self) -> None:
        """Score exactly equal to the default threshold (0.45) returns True."""
        self._patch_pipeline(cls_module._LABELS[0], 0.45)
        assert (
            cls_module.needs_current_information("current stock price of NVDA") is True
        )

    # -- non-temporal queries (should return False) ---------------------------

    def test_non_temporal_query_returns_false(self) -> None:
        """A timeless factual query returns False."""
        self._patch_pipeline(cls_module._LABELS[1], 0.88)
        assert (
            cls_module.needs_current_information("What is the speed of light?") is False
        )

    def test_temporal_query_below_threshold_returns_false(self) -> None:
        """A temporal label that scores below the default threshold (0.45) is treated as False."""
        self._patch_pipeline(cls_module._LABELS[0], 0.35)
        assert cls_module.needs_current_information("maybe something recent?") is False

    # -- threshold configuration ----------------------------------------------

    def test_custom_threshold_via_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """INTENT_CONFIDENCE_THRESHOLD env var overrides the default 0.45."""
        monkeypatch.setenv("INTENT_CONFIDENCE_THRESHOLD", "0.80")
        # Score of 0.75 is above 0.45 but below 0.80 — should be False
        self._patch_pipeline(cls_module._LABELS[0], 0.75)
        assert cls_module.needs_current_information("news today") is False

    def test_zero_threshold_always_returns_true_for_temporal_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting threshold to 0 makes any temporal-label win return True."""
        monkeypatch.setenv("INTENT_CONFIDENCE_THRESHOLD", "0.0")
        self._patch_pipeline(cls_module._LABELS[0], 0.01)
        assert cls_module.needs_current_information("anything") is True

    # -- pipeline called with the correct candidate labels --------------------

    def test_correct_labels_passed_to_pipeline(self) -> None:
        """The pipeline must receive the canonical candidate labels and hypothesis template."""
        mock_pipe = self._patch_pipeline(cls_module._LABELS[0], 0.9)
        cls_module.needs_current_information("latest GPU releases")
        mock_pipe.assert_called_once()
        _, kwargs = mock_pipe.call_args
        assert "candidate_labels" in kwargs
        assert kwargs["candidate_labels"] == cls_module._LABELS
        assert "hypothesis_template" in kwargs
        assert kwargs["hypothesis_template"] == cls_module._HYPOTHESIS_TEMPLATE

    # -- fault tolerance ------------------------------------------------------

    def test_runtime_error_defaults_to_false(self) -> None:
        """If _load_pipeline raises RuntimeError, return False gracefully."""
        with patch.object(
            cls_module, "_load_pipeline", side_effect=RuntimeError("no model")
        ):
            assert cls_module.needs_current_information("some query") is False

    def test_unexpected_exception_defaults_to_false(self) -> None:
        """Any unexpected exception from the pipeline returns False safely."""
        mock_pipe = MagicMock(side_effect=Exception("unknown GPU error"))
        cls_module._pipeline = mock_pipe
        assert cls_module.needs_current_information("GPU news") is False

    # -- lazy load verified across multiple public calls ----------------------

    def test_pipeline_loaded_once_across_multiple_calls(self) -> None:
        """_load_pipeline is effectively called only once for many invocations."""
        mock_pipe = _make_pipeline_mock(cls_module._LABELS[0], 0.9)
        with patch("transformers.pipeline", return_value=mock_pipe) as mock_factory:
            # Reset cache to force the lazy load path
            cls_module._pipeline = None
            for _ in range(5):
                cls_module.needs_current_information("query")
        mock_factory.assert_called_once()
