"""tests/test_research_server.py

Unit tests for servers/research_server/server.py.

All network and LLM calls are fully mocked — tests run 100 % offline.
The test strategy mirrors test_base_tools.py: import the FastMCP-wrapped
tool and unwrap it via ``.fn`` to get the plain callable.
"""

from __future__ import annotations

# Standard Library
import io
import json
import urllib.error
from typing import Any
from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

# Third-Party Libraries
import pytest

# Local Modules
# Local Modules — unwrap FastMCP FunctionTool to get the plain callable
from servers.research_server.server import deep_research as _deep_research_tool
from servers.research_server.server import (
    _deep_research,
    _search_searxng,
    _format_results_block,
)

deep_research = _deep_research_tool.fn


# ---------------------------------------------------------------------------
# Shared fixtures / factories
# ---------------------------------------------------------------------------


def _mock_searxng_response(results: list[dict[str, Any]]) -> MagicMock:
    """Build a mock urllib response that returns a SearXNG JSON payload."""
    payload = json.dumps({"results": results}).encode()
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = payload
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _mock_ollama_chat_response(content: str) -> dict[str, Any]:
    """Build a minimal Ollama chat response dict."""
    return {"message": {"role": "assistant", "content": content}}


_SAMPLE_RESULTS_1 = [
    {
        "url": "https://example.com/a",
        "title": "NVIDIA Blackwell",
        "snippet": "Blackwell launched Q1 2025.",
    },
    {
        "url": "https://example.com/b",
        "title": "GB200 specs",
        "snippet": "GB200 has 192GB HBM3e.",
    },
]

_SAMPLE_RESULTS_2 = [
    {
        "url": "https://example.com/c",
        "title": "RTX 5090 review",
        "snippet": "RTX 5090 ships Feb 2025.",
    },
]


# ---------------------------------------------------------------------------
# _search_searxng
# ---------------------------------------------------------------------------


class TestSearchSearxng:
    """Unit tests for the _search_searxng helper."""

    def test_returns_structured_results(self) -> None:
        """Results are remapped to url/title/snippet."""
        mock_resp = _mock_searxng_response(_SAMPLE_RESULTS_1)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            results = _search_searxng(
                "NVIDIA Blackwell", top_n=5, searxng_host="http://localhost:8080"
            )

        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/a"
        assert results[0]["title"] == "NVIDIA Blackwell"
        assert "Blackwell" in results[0]["snippet"]

    def test_top_n_limits_results(self) -> None:
        """top_n parameter truncates the result list."""
        mock_resp = _mock_searxng_response(_SAMPLE_RESULTS_1)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            results = _search_searxng(
                "query", top_n=1, searxng_host="http://localhost:8080"
            )
        assert len(results) == 1

    def test_non_200_raises_value_error(self) -> None:
        """A non-200 HTTP status raises ValueError."""
        mock_resp = MagicMock()
        mock_resp.status = 503
        mock_resp.read.return_value = b"{}"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(ValueError, match="HTTP 503"):
                _search_searxng("query", top_n=5, searxng_host="http://localhost:8080")

    def test_error_key_in_response_raises_value_error(self) -> None:
        """An 'error' key in the JSON response raises ValueError."""
        payload = json.dumps({"error": "rate limited"}).encode()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(ValueError, match="rate limited"):
                _search_searxng("query", top_n=5, searxng_host="http://localhost:8080")

    def test_url_error_propagates(self) -> None:
        """A URLError (service unreachable) is not silently swallowed."""
        with patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("refused")
        ):
            with pytest.raises(urllib.error.URLError):
                _search_searxng("query", top_n=5, searxng_host="http://localhost:8080")


# ---------------------------------------------------------------------------
# _format_results_block
# ---------------------------------------------------------------------------


class TestFormatResultsBlock:
    """Tests for the context-block formatter."""

    def test_contains_iteration_header(self) -> None:
        block = _format_results_block(_SAMPLE_RESULTS_1, iteration=2)
        assert "iteration 2" in block

    def test_contains_all_urls(self) -> None:
        block = _format_results_block(_SAMPLE_RESULTS_1, iteration=1)
        assert "https://example.com/a" in block
        assert "https://example.com/b" in block


# ---------------------------------------------------------------------------
# _deep_research (IterDRAG loop)
# ---------------------------------------------------------------------------


class TestDeepResearch:
    """Integration-style tests for the full IterDRAG loop (all I/O mocked)."""

    def _make_urlopen_side_effect(self, result_batches: list[list[dict[str, Any]]]):
        """Return a side_effect function that cycles through result batches."""
        responses = [_mock_searxng_response(batch) for batch in result_batches]
        calls: list[int] = [0]

        def side_effect(url: str, timeout: int = 10):
            idx = calls[0]
            calls[0] += 1
            if idx < len(responses):
                return responses[idx]
            return responses[-1]  # repeat last for extra calls

        return side_effect

    def test_returns_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_deep_research must always return a parseable JSON string."""
        monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "1")
        monkeypatch.setenv("SEARXNG_HOST", "http://localhost:8080")

        with (
            patch(
                "urllib.request.urlopen",
                side_effect=self._make_urlopen_side_effect([_SAMPLE_RESULTS_1]),
            ),
            patch("servers.research_server.server.Client") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client.chat.return_value = _mock_ollama_chat_response(
                "NVIDIA launched Blackwell [1]."
            )
            mock_client_cls.return_value = mock_client

            result = _deep_research("What has NVIDIA released recently?")

        data = json.loads(result)
        assert "answer" in data
        assert "citations" in data
        assert "iterations" in data
        assert isinstance(data["citations"], list)

    def test_iteration_count_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """IterDRAG runs exactly RESEARCH_MAX_ITERATIONS SearXNG queries."""
        monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "2")
        monkeypatch.setenv("SEARXNG_HOST", "http://localhost:8080")

        batches = [_SAMPLE_RESULTS_1, _SAMPLE_RESULTS_2]
        urlopen_calls: list[str] = []

        def counting_urlopen(url: str, timeout: int = 10):
            urlopen_calls.append(url)
            idx = len(urlopen_calls) - 1
            return _mock_searxng_response(batches[min(idx, len(batches) - 1)])

        with (
            patch("urllib.request.urlopen", side_effect=counting_urlopen),
            patch("servers.research_server.server.Client") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client.chat.return_value = _mock_ollama_chat_response("Answer.")
            mock_client_cls.return_value = mock_client

            result = _deep_research("NVIDIA hardware 2025")

        # 2 iterations → 2 SearXNG calls + 1 sub-query refinement + 1 synthesis
        assert len(urlopen_calls) == 2

        data = json.loads(result)
        assert data["iterations"] == 2

    def test_citation_deduplication(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Identical URLs across iterations appear only once in citations."""
        monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "2")
        monkeypatch.setenv("SEARXNG_HOST", "http://localhost:8080")

        # Both iterations return the same results (same URLs)
        duplicate_results = [
            {
                "url": "https://example.com/a",
                "title": "A",
                "content": "Some content A.",
            },
        ]
        batches = [duplicate_results, duplicate_results]

        with (
            patch(
                "urllib.request.urlopen",
                side_effect=self._make_urlopen_side_effect(batches),
            ),
            patch("servers.research_server.server.Client") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client.chat.return_value = _mock_ollama_chat_response(
                "Deduped answer."
            )
            mock_client_cls.return_value = mock_client

            result = _deep_research("duplicate test")

        data = json.loads(result)
        # URL https://example.com/a appears in both iterations; must be seen once
        urls = [c["url"] for c in data["citations"]]
        assert urls.count("https://example.com/a") == 1

    def test_searxng_failure_returns_error_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When SearXNG is unreachable for all iterations, answer explains the issue."""
        monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "2")
        monkeypatch.setenv("SEARXNG_HOST", "http://localhost:8080")

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = _deep_research("anything")

        data = json.loads(result)
        assert "SearXNG" in data["answer"]
        assert data["citations"] == []
        assert data["iterations"] == 0


# ---------------------------------------------------------------------------
# FastMCP wrapper (deep_research.fn)
# ---------------------------------------------------------------------------


class TestDeepResearchTool:
    """Smoke test that the FastMCP-wrapped deep_research delegates to _deep_research."""

    def test_fn_unwrap_delegates_correctly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The .fn-unwrapped tool must return the same value as _deep_research."""
        monkeypatch.setenv("RESEARCH_MAX_ITERATIONS", "1")
        monkeypatch.setenv("SEARXNG_HOST", "http://localhost:8080")

        expected = json.dumps(
            {"answer": "mocked", "citations": [], "iterations": 0},
        )
        with patch(
            "servers.research_server.server._deep_research", return_value=expected
        ) as mock_dr:
            result = deep_research("test query")

        mock_dr.assert_called_once_with(query="test query")
        assert result == expected
