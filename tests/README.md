# brAIniac Test Suite

Comprehensive testing for all brAIniac components with both CLI and web-based interfaces.

## Test Structure

```text
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                # Pytest fixtures and configuration
├── test_memory.py             # Unit tests for RollingMemory
├── test_chat.py               # Unit tests for ChatEngine
├── test_base_tools.py         # Unit tests for MCP tools
├── test_integration.py        # Integration tests
├── web_test_interface.py      # Gradio web testing interface
└── README.md                  # This file
```

## Quick Start

### Run All Tests

```bash
# Install test dependencies
poetry install --with dev

# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run with coverage report
poetry run pytest --cov

# Generate HTML coverage report
poetry run pytest --cov --cov-report=html
# Then open htmlcov/index.html in your browser
```

### Run Specific Test Files

```bash
# Test memory module only
poetry run pytest tests/test_memory.py

# Test chat engine only
poetry run pytest tests/test_chat.py

# Test tools only
poetry run pytest tests/test_base_tools.py

# Integration tests only
poetry run pytest tests/test_integration.py
```

### Run Tests by Marker

```bash
# Run only unit tests
poetry run pytest -m unit

# Run only integration tests
poetry run pytest -m integration

# Run only slow tests (require external services)
poetry run pytest -m slow
```

## Web Test Interface

Launch an interactive browser-based test interface:

```bash
poetry run python tests/web_test_interface.py
```

Then open your browser to: <http://127.0.0.1:7860>

### Web Interface Features

1. **💬 Chat Engine Tab**
   - Interactive chat with mocked LLM
   - Test rolling memory behavior
   - View real-time statistics
   - Clear history and restart

2. **🛠️ Tool Testing Tab**
   - Execute `get_current_time()` tool
   - Execute `web_search()` tool with custom parameters
   - View formatted JSON results

3. **💾 Memory Testing Tab**
   - Test rolling window behavior
   - Fill memory to capacity
   - Test FIFO message eviction
   - Verify system message preservation

4. **ℹ️ System Info Tab**
   - Component status overview
   - Test coverage information
   - Quick test scenarios

## Test Coverage

Current test coverage by module:

| Module | Tests | Coverage |
| -------- | ------- | ---------- |
| `core/memory.py` | 15 tests | ~100% |
| `core/chat.py` | 14 tests | ~95% |
| `servers/base_tools/server.py` | 14 tests | ~100% |
| Integration | 8 tests | - |

## Test Categories

### Unit Tests

**test_memory.py** - `RollingMemory` class:

- Initialization with default/custom parameters
- Adding single and multiple messages
- FIFO rolling window behavior
- System message preservation
- Clear operation
- Various window sizes

**test_chat.py** - `ChatEngine` class:

- Initialization and configuration
- Ollama client integration (mocked)
- Tool registration and execution
- Error handling
- History management
- Context accumulation

**test_base_tools.py** - MCP tools:

- `get_current_time()` output validation
- `web_search()` parameter handling
- JSON response structure
- Edge cases (empty queries, etc.)

### Integration Tests

**test_integration.py**:

- Memory + ChatEngine integration
- Tool execution in context
- Full conversation flows
- Error recovery
- Large context handling
- Clear and resume workflows

## Writing New Tests

### Test File Template

```python
"""tests/test_yourmodule.py

Description of what this test file covers.
"""

from __future__ import annotations

import pytest

from your_module import YourClass


class TestYourClass:
    """Test suite for YourClass."""

    def test_something(self) -> None:
        """Test specific behavior."""
        # Arrange
        instance = YourClass()
        
        # Act
        result = instance.do_something()
        
        # Assert
        assert result == expected_value
```

### Using Fixtures

```python
def test_with_fixture(mock_ollama_client: Mock) -> None:
    """Test using a shared fixture from conftest.py."""
    # Fixture is automatically injected
    assert mock_ollama_client is not None
```

### Parametrized Tests

```python
@pytest.mark.parametrize(
    "input_value,expected",
    [
        ("test1", "result1"),
        ("test2", "result2"),
        ("test3", "result3"),
    ],
)
def test_multiple_cases(input_value: str, expected: str) -> None:
    """Test multiple input/output combinations."""
    result = process(input_value)
    assert result == expected
```

### Mocking External Services

```python
from unittest.mock import Mock, patch

@patch("core.chat.Client")
def test_with_mock(mock_client_class: Mock) -> None:
    """Test with mocked Ollama client."""
    mock_client = Mock()
    mock_client.chat.return_value = {"message": {"content": "Test"}}
    mock_client_class.return_value = mock_client
    
    # Your test code here
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# .github/workflows/test.yml example
- name: Run tests
  run: |
    poetry install --with dev
    poetry run pytest --cov --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Test Best Practices

1. **Isolation**: Each test should be independent
2. **Mocking**: Mock external dependencies (Ollama, ChromaDB)
3. **Coverage**: Aim for >90% coverage on core modules
4. **Speed**: Unit tests should run in <1s each
5. **Clarity**: Use descriptive test names and docstrings
6. **Type Hints**: All test functions should have type hints

## Troubleshooting

### Import Errors

If you see import errors, ensure you're running from the project root:

```bash
cd /path/to/brAIniac
poetry run pytest
```

### Tests Hanging

Some tests may appear to hang if they're trying to connect to real services. Ensure all external services are properly mocked.

### Coverage Not Generated

```bash
# Install coverage dependency
poetry install --with dev

# Run with coverage explicitly
poetry run pytest --cov=core --cov=servers
```

## Future Test Additions

Phase 2 and beyond will include:

- [ ] ChromaDB integration tests
- [ ] SearXNG search tests (when real implementation added)
- [ ] Letta/MemGPT memory tests
- [ ] Multi-agent orchestration tests
- [ ] Performance benchmarks
- [ ] Load tests for VRAM usage
- [ ] End-to-end Docker container tests

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Gradio Documentation](https://gradio.app/docs/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
