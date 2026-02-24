#!/bin/bash
# Quick script to run the brAIniac test suite

echo "============================================"
echo "brAIniac Test Suite Runner"
echo "============================================"
echo ""

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Please install Poetry first."
    exit 1
fi

echo "Installing test dependencies..."
poetry install --with dev

echo ""
echo "============================================"
echo "Running Tests"
echo "============================================"
echo ""

# Run pytest with coverage
poetry run pytest -v --cov=core --cov=servers --cov-report=term-missing --cov-report=html

echo ""
echo "============================================"
echo "Test Results"
echo "============================================"
echo ""

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
    echo ""
    echo "Coverage report generated in htmlcov/index.html"
    echo "Open it in your browser to view detailed coverage."
else
    echo "❌ Some tests failed. See output above for details."
    exit 1
fi

echo ""
echo "To run the web test interface:"
echo "  poetry run python tests/web_test_interface.py"
echo ""
